"""Train the small 1D-Transformer under Protocol A and/or LOSO.

Same output layout and per-fold contract as scripts/run_cnn.py — predictions
parquet keyed on (spectrum_id, file_id), per-fold encoder snapshot for any
later memprobe v2 run, ModelResult JSON.

Two carry-overs from the CNN session (per plan/10§cnn-architectural-fixes):
1. Per-bin standardize at input (handled inside SmallTransformer1D via a
   register_buffer that train_cnn_fold fits on outer-train and bakes in).
2. Sanity-check param count vs spec on first instantiation.

Progress UX: tqdm progress bars at three nested levels —
    outer  : protocol-level fold iter
    inner  : per-fold epoch iter (transient, leave=False)
The training loop (atlas/train.py) owns the inner bar; this script owns the
outer one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from atlas.evaluate import (
    aggregate,
    evaluate_fold,
    write_model_result,
    write_predictions_parquet,
)
from atlas.models_cnn import select_device  # device picker reused
from atlas.models_transformer import SmallTransformer1D, count_params
from atlas.splits import load_splits
from atlas.train import TrainConfig, train_cnn_fold


def _short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def _runs_log_append(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_one(
    *,
    protocol: str,
    splits_path: Path,
    spec_df: pd.DataFrame,
    X_full: np.ndarray,
    outputs_dir: Path,
    seed: int,
    cfg: TrainConfig,
    device: torch.device,
    model_name: str = "transformer_small",
    runs_log: Path | None = None,
) -> dict:
    splits = load_splits(splits_path)
    folds = splits["folds"]

    run_id = (
        f"{datetime.now().strftime('%Y-%m-%d')}_{model_name}_{protocol}_"
        f"{_short_hash(model_name + protocol + str(seed))}"
    )
    run_dir = outputs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id,
        "model": model_name,
        "protocol": protocol,
        "splits_path": str(splits_path),
        "seed": seed,
        "device": str(device),
        "n_epochs": cfg.n_epochs,
        "batch_size": cfg.batch_size,
        "warmup_epochs": cfg.warmup_epochs,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "label_smoothing": cfg.label_smoothing,
        "patience": cfg.patience,
        "augmentation": {
            "p_noise": cfg.aug.p_noise, "p_scale": cfg.aug.p_scale,
            "p_shift": cfg.aug.p_shift, "p_baseline": cfg.aug.p_baseline,
            "p_mixup": cfg.aug.p_mixup, "mixup_alpha": cfg.aug.mixup_alpha,
        },
        "sklearn_version": splits["meta"]["sklearn_version"],
        "split_cache_hash": splits["meta"]["cache_hash"],
        "torch_version": torch.__version__,
    }
    with open(run_dir / "config.resolved.json", "w") as f:
        json.dump(config, f, indent=2)

    file_ids_all = spec_df["file_id"].to_numpy()
    y_all = spec_df["primary_class"].to_numpy()
    sub_all = spec_df["subclass"].to_numpy()
    n_bins = X_full.shape[1]

    fold_results = []
    t_run_start = time.perf_counter()

    fold_pbar = tqdm(folds, desc=f"{protocol} folds", total=len(folds), ncols=100)
    for fold in fold_pbar:
        fold_id = fold["fold"]
        train_idx = np.asarray(fold["train_row_indices"], dtype=np.int64)
        test_idx = np.asarray(fold["test_row_indices"], dtype=np.int64)
        fold_seed = int(fold["fold_seed"])

        X_train = X_full[train_idx]
        y_train = y_all[train_idx]
        groups_train = file_ids_all[train_idx]
        X_test = X_full[test_idx]

        n_train_files = len(set(groups_train))
        n_test_files = len(set(file_ids_all[test_idx]))
        tqdm.write(
            f"\n[{run_id}] fold={fold_id}  "
            f"n_train={train_idx.size} ({n_train_files} files)  "
            f"n_test={test_idx.size} ({n_test_files} files)"
        )

        def factory(seed_in: int) -> SmallTransformer1D:
            del seed_in
            return SmallTransformer1D(n_bins=n_bins, n_classes=4)

        # Per-fold tqdm label so the inner epoch bar tells you which fold
        # it's progressing on.
        fold_cfg = TrainConfig(
            n_epochs=cfg.n_epochs,
            batch_size=cfg.batch_size,
            cpu_batch_size=cfg.cpu_batch_size,
            warmup_epochs=cfg.warmup_epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            label_smoothing=cfg.label_smoothing,
            grad_clip=cfg.grad_clip,
            patience=cfg.patience,
            num_workers=cfg.num_workers,
            aug=cfg.aug,
            log_every=cfg.log_every,
            use_tqdm=cfg.use_tqdm,
            tqdm_desc=f"fold={fold_id}",
        )

        proba_test, info, train_dt, trained_model = train_cnn_fold(
            model_factory=factory,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            X_test=X_test,
            fold_seed=fold_seed,
            device=device,
            n_bins=n_bins,
            n_classes=4,
            cfg=fold_cfg,
            log_fn=tqdm.write,
        )

        encoder_path = run_dir / f"encoder_fold_{fold_id}.pt"
        torch.save(
            {k: v.detach().cpu() for k, v in trained_model.state_dict().items()},
            encoder_path,
        )

        pred_path = run_dir / f"predictions_fold_{fold_id}.parquet"
        write_predictions_parquet(
            fold_id=fold_id,
            file_ids=file_ids_all[test_idx],
            subclass=sub_all[test_idx],
            y_true=y_all[test_idx],
            y_proba=proba_test,
            out_path=pred_path,
        )

        fr = evaluate_fold(
            fold_id=fold_id,
            protocol=protocol,
            model_name=model_name,
            y_true=y_all[test_idx],
            y_proba=proba_test,
            file_ids=file_ids_all[test_idx],
            subclass=sub_all[test_idx],
            n_train=int(train_idx.size),
            n_train_files=n_train_files,
            n_test_files=n_test_files,
            training_time_s=train_dt,
            best_hyperparams={"val_macro_f1": info["best_val_macro_f1"],
                              "best_epoch": info["best_epoch"],
                              "n_epochs_run": info["n_epochs_run"],
                              "n_params": info["n_params"]},
            predictions_path=str(pred_path),
            notes=[f"device={info['device']}", f"class_weights={info['class_weights']}"],
        )
        fold_results.append(fr)

        # Concise per-fold summary as a tqdm.write line — survives below the
        # progress bars.
        tqdm.write(
            f"  -> file_F1={fr.file_macro_f1:.3f}  "
            f"spec_F1={fr.spectrum_macro_f1:.3f}  "
            f"val_f1_best={info['best_val_macro_f1']:.3f}@ep{info['best_epoch']}  "
            f"time={train_dt:.1f}s"
        )
        rec = fr.file_per_class_recall
        tqdm.write(
            f"     file-level recall: " + "  ".join(f"{c}={rec[c]:.2f}" for c in rec)
        )

        # Update the outer fold bar postfix with running mean file-F1.
        running = float(np.mean([f.file_macro_f1 for f in fold_results]))
        fold_pbar.set_postfix(running_file_F1=f"{running:.3f}")

    fold_pbar.close()

    mr = aggregate(fold_results, model_name=model_name, protocol=protocol)
    write_model_result(mr, run_dir / "model_result.json")
    t_run = time.perf_counter() - t_run_start

    summary = {
        "run_id": run_id,
        "model": model_name,
        "protocol": protocol,
        "spectrum_macro_f1_mean": mr.spectrum_macro_f1_mean,
        "spectrum_macro_f1_sd": mr.spectrum_macro_f1_sd,
        "file_macro_f1_mean": mr.file_macro_f1_mean,
        "file_macro_f1_sd": mr.file_macro_f1_sd,
        "file_macro_f1_min": mr.file_macro_f1_min,
        "file_macro_f1_max": mr.file_macro_f1_max,
        "file_macro_f1_per_fold": mr.file_macro_f1_per_fold,
        "duration_s": t_run,
        "timestamp": datetime.now().isoformat(),
    }
    if runs_log is not None:
        _runs_log_append(runs_log, summary)

    print(
        f"\n[{run_id}] DONE  "
        f"file_macro_f1 = {mr.file_macro_f1_mean:.3f} +/- {mr.file_macro_f1_sd:.3f}  "
        f"min={mr.file_macro_f1_min:.3f}  max={mr.file_macro_f1_max:.3f}  "
        f"duration={t_run:.1f}s"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--protocols", nargs="+", default=["group_kfold", "loso"],
        choices=["group_kfold", "loso"],
    )
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "data_cache"))
    ap.add_argument("--splits-dir", default=str(REPO_ROOT / "data_cache" / "splits"))
    ap.add_argument("--outputs-dir", default=str(REPO_ROOT / "outputs"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default=None,
                    help="Override device. Default = auto (MPS > CUDA > CPU).")
    ap.add_argument("--no-aug", action="store_true",
                    help="Sanity-check mode: disable all augmentation.")
    ap.add_argument("--folds", nargs="*", default=None,
                    help="Optional subset of fold ids to run (e.g. for sanity check).")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    splits_dir = Path(args.splits_dir)
    outputs_dir = Path(args.outputs_dir)
    runs_log = outputs_dir / "runs.jsonl"

    spec_df = pd.read_parquet(cache_dir / "spectra.parquet")
    X_full = np.load(cache_dir / "spectra_array_preprocessed.npy")

    device = torch.device(args.device) if args.device else select_device()

    from atlas.train import AugConfig
    aug = AugConfig() if not args.no_aug else AugConfig(
        p_noise=0.0, p_scale=0.0, p_shift=0.0, p_baseline=0.0, p_mixup=0.0,
    )

    cfg = TrainConfig(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        aug=aug,
    )

    print(f"Loaded preprocessed array shape={X_full.shape}, dtype={X_full.dtype}")
    print(f"Device: {device}  torch: {torch.__version__}")
    print(f"Train cfg: n_epochs={cfg.n_epochs}  batch_size={cfg.batch_size}  "
          f"lr={cfg.lr}  patience={cfg.patience}  aug={'OFF' if args.no_aug else 'ON'}")
    print(f"Protocols: {args.protocols}\n")

    tmp = SmallTransformer1D(n_bins=X_full.shape[1], n_classes=4)
    print(f"SmallTransformer1D params: {count_params(tmp):,}  patches: {tmp.n_patches}")
    del tmp

    PROTOCOL_TO_FILE = {"group_kfold": "protocol_a.json", "loso": "protocol_b.json"}

    all_summaries = []
    for protocol in args.protocols:
        splits_path = splits_dir / PROTOCOL_TO_FILE[protocol]
        if args.folds:
            # Filter the splits down to the requested fold ids — used by the
            # sanity-check pass. We modify the loaded splits in-place inside
            # run_one via a monkeypatched load_splits return is too cute;
            # the simplest approach is to load, filter, save a temp file.
            splits = load_splits(splits_path)
            requested = set(str(x) for x in args.folds)
            kept = [f for f in splits["folds"] if str(f["fold"]) in requested]
            if not kept:
                print(f"  no folds in {splits_path} matched --folds={args.folds}; "
                      f"available: {[str(f['fold']) for f in splits['folds']]}")
                continue
            splits["folds"] = kept
            tmp_path = outputs_dir / f"_tmp_splits_{protocol}_{int(time.time())}.json"
            with open(tmp_path, "w") as f:
                json.dump(splits, f)
            splits_path = tmp_path

        summary = run_one(
            protocol=protocol,
            splits_path=splits_path,
            spec_df=spec_df,
            X_full=X_full,
            outputs_dir=outputs_dir,
            seed=args.seed,
            cfg=cfg,
            device=device,
            runs_log=runs_log,
        )
        all_summaries.append(summary)
        print()

    print("=" * 78)
    print(f"{'model':>18}  {'protocol':>11}  {'file_F1 mean':>12}  {'SD':>5}  "
          f"{'min':>5}  {'max':>5}  per-fold")
    print("-" * 78)
    for r in all_summaries:
        per_fold = " ".join(f"{x:.2f}" for x in r["file_macro_f1_per_fold"])
        print(
            f"{r['model']:>18}  {r['protocol']:>11}  "
            f"{r['file_macro_f1_mean']:>12.3f}  {r['file_macro_f1_sd']:>5.3f}  "
            f"{r['file_macro_f1_min']:>5.3f}  {r['file_macro_f1_max']:>5.3f}  "
            f"{per_fold}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
