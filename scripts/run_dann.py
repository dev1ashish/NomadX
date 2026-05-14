"""Train the CNN with DANN (domain-adversarial) under Protocol A and/or LOSO.

Mirrors scripts/run_cnn.py:
    - Same output layout (run_dir, predictions parquet per fold, encoder
      snapshot per fold, model_result.json, runs.jsonl summary line).
    - Same per-fold contract (train.train_dann_fold returns proba_test +
      info + trained_model identical to train_cnn_fold).

DANN-specific surface area:
    - --lambda-max (default 0.1)
    - --warmup-epochs-dann (default 10)
    - --domain-hidden (default 64)
    - --no-aug (sanity-check) -- mirrors run_transformer.py
    - --folds (subset, sanity-check) -- mirrors run_transformer.py

Per-fold history rows now carry class_loss / domain_loss / domain_acc /
lambda_grl alongside val_macro_f1. config.resolved.json carries the DANN
hyperparams so a future memprobe v2 run can be traced back to the recipe
that produced the encoder.

tqdm UX matches run_transformer.py: outer fold bar + per-fold inner epoch
bar; outer postfix shows running mean file-F1.
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
from atlas.models_cnn import DANNCNN1D, count_params, select_device
from atlas.splits import load_splits
from atlas.train import AugConfig, DANNConfig, TrainConfig, train_dann_fold


def _short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def _runs_log_append(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _build_domain_labels(spec_df: pd.DataFrame, grouping: str) -> np.ndarray:
    """Return a (N,) array of per-row domain labels (strings) for the DANN GRL target.

    Three groupings supported:
      file_id   - 87-way default. Same as original DANN setup.
      subclass  - 10-way: 9 bacterial subclasses + "H2O" (since H2O files have
                  subclass=None in spec_df). The GRL no longer penalizes
                  within-subclass shared features, which should preserve the
                  easy-commensal recognition signal that lambda=0.3 destroys.
      cal_date  - 13-way: extract YYMMDD from the trailing _NNNNNN segment of
                  file_id. The most biologically meaningful nuisance variable
                  per plan/07§batch-effect.
    """
    import re
    if grouping == "file_id":
        return spec_df["file_id"].to_numpy()
    if grouping == "subclass":
        sub = spec_df["subclass"].to_numpy()
        # Replace pandas None with the string "H2O" so the domain head sees a
        # valid categorical target on water rows.
        return np.array([s if s is not None else "H2O" for s in sub])
    if grouping == "cal_date":
        pat = re.compile(r"_(\d{6})$")
        def _parse(fid: str) -> str:
            m = pat.search(fid)
            return m.group(1) if m else "unknown"
        return np.array([_parse(f) for f in spec_df["file_id"].to_numpy()])
    raise ValueError(f"unknown domain grouping: {grouping}")


def run_one(
    *,
    protocol: str,
    splits_path: Path,
    spec_df: pd.DataFrame,
    X_full: np.ndarray,
    domain_labels_all: np.ndarray,  # (N,) full-dataset per-row domain labels
    outputs_dir: Path,
    seed: int,
    cfg: TrainConfig,
    dann_cfg: DANNConfig,
    domain_grouping: str,
    device: torch.device,
    model_name: str,
    runs_log: Path | None = None,
) -> dict:
    splits = load_splits(splits_path)
    folds = splits["folds"]

    run_id = (
        f"{datetime.now().strftime('%Y-%m-%d')}_{model_name}_{protocol}_"
        f"{_short_hash(model_name + protocol + str(seed) + str(dann_cfg.lambda_max))}"
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
        "dann": {
            "lambda_max": dann_cfg.lambda_max,
            "warmup_epochs_dann": dann_cfg.warmup_epochs_dann,
            "domain_hidden": dann_cfg.domain_hidden,
            "domain_grouping": domain_grouping,
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

    fold_pbar = tqdm(folds, desc=f"{protocol} folds", total=len(folds), ncols=120)
    for fold in fold_pbar:
        fold_id = fold["fold"]
        train_idx = np.asarray(fold["train_row_indices"], dtype=np.int64)
        test_idx = np.asarray(fold["test_row_indices"], dtype=np.int64)
        fold_seed = int(fold["fold_seed"])

        X_train = X_full[train_idx]
        y_train = y_all[train_idx]
        groups_train = file_ids_all[train_idx]
        domain_labels_train = domain_labels_all[train_idx]
        X_test = X_full[test_idx]

        n_train_files = len(set(groups_train))
        n_test_files = len(set(file_ids_all[test_idx]))
        n_domains_this_fold = len(set(domain_labels_train.tolist()))
        tqdm.write(
            f"\n[{run_id}] fold={fold_id}  "
            f"n_train={train_idx.size} ({n_train_files} files)  "
            f"n_test={test_idx.size} ({n_test_files} files)  "
            f"n_domains={n_domains_this_fold} (grouping={domain_grouping})"
        )

        # Factory takes (fold_seed, n_domains). _set_seeds(fold_seed) inside
        # train_dann_fold runs BEFORE this factory is invoked, so the encoder
        # init draws the same parameters as vanilla SmallCNN1D with the same
        # fold_seed (domain head is appended after super().__init__()).
        def factory(seed_in: int, n_domains: int) -> DANNCNN1D:
            del seed_in
            return DANNCNN1D(
                n_bins=n_bins,
                n_classes=4,
                n_domains=n_domains,
                domain_hidden=dann_cfg.domain_hidden,
            )

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

        proba_test, info, train_dt, trained_model = train_dann_fold(
            model_factory=factory,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            domain_labels=domain_labels_train,
            X_test=X_test,
            fold_seed=fold_seed,
            device=device,
            n_bins=n_bins,
            n_classes=4,
            cfg=fold_cfg,
            dann_cfg=dann_cfg,
            log_fn=tqdm.write,
        )

        # Encoder snapshot for memprobe v2. We strip the domain_head before
        # saving so memprobe v2 (which loads via SmallCNN1D + load_state_dict)
        # doesn't choke on unexpected keys. Saving via {strict=False} on the
        # consumer side would also work, but stripping here keeps the v2
        # script identical to the vanilla CNN flow.
        encoder_state = {
            k: v.detach().cpu()
            for k, v in trained_model.state_dict().items()
            if not k.startswith("domain_head.")
        }
        torch.save(encoder_state, run_dir / f"encoder_fold_{fold_id}.pt")
        # Full state dict (encoder + domain head + buffers) saved separately
        # so a future session can reload the discriminator if needed.
        torch.save(
            {k: v.detach().cpu() for k, v in trained_model.state_dict().items()},
            run_dir / f"full_state_fold_{fold_id}.pt",
        )

        # Per-fold history JSON (curves for class_loss / domain_loss / val_f1)
        with open(run_dir / f"history_fold_{fold_id}.json", "w") as f:
            json.dump({
                "fold": fold_id,
                "n_domains": info["n_domains"],
                "lambda_max": info["lambda_max"],
                "warmup_epochs_dann": info["warmup_epochs_dann"],
                "history": info["history"],
                "best_val_macro_f1": info["best_val_macro_f1"],
                "best_epoch": info["best_epoch"],
                "n_epochs_run": info["n_epochs_run"],
            }, f, indent=2)

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
            best_hyperparams={
                "val_macro_f1": info["best_val_macro_f1"],
                "best_epoch": info["best_epoch"],
                "n_epochs_run": info["n_epochs_run"],
                "n_params": info["n_params"],
                "n_domains": info["n_domains"],
                "lambda_max": info["lambda_max"],
            },
            predictions_path=str(pred_path),
            notes=[
                f"device={info['device']}",
                f"class_weights={info['class_weights']}",
                f"lambda_max={info['lambda_max']}",
                f"warmup_epochs_dann={info['warmup_epochs_dann']}",
            ],
        )
        fold_results.append(fr)

        # Final-epoch domain-loss / domain-acc lines tell us if the
        # discriminator actually fought back during training.
        last = info["history"][-1] if info["history"] else {}
        tqdm.write(
            f"  -> file_F1={fr.file_macro_f1:.3f}  "
            f"spec_F1={fr.spectrum_macro_f1:.3f}  "
            f"val_f1_best={info['best_val_macro_f1']:.3f}@ep{info['best_epoch']}  "
            f"final dom_loss={last.get('domain_loss', float('nan')):.2f}  "
            f"final dom_acc={last.get('train_domain_acc', float('nan')):.2f}  "
            f"time={train_dt:.1f}s"
        )
        rec = fr.file_per_class_recall
        tqdm.write("     file-level recall: " + "  ".join(f"{c}={rec[c]:.2f}" for c in rec))

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
        "lambda_max": dann_cfg.lambda_max,
        "warmup_epochs_dann": dann_cfg.warmup_epochs_dann,
        "domain_grouping": domain_grouping,
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
    ap.add_argument("--lambda-max", type=float, default=0.1,
                    help="DANN coefficient ceiling. 0 = vanilla-CNN equivalent on encoder updates.")
    ap.add_argument("--warmup-epochs-dann", type=int, default=10,
                    help="Linear warmup over which lambda goes 0 -> lambda_max.")
    ap.add_argument("--domain-hidden", type=int, default=64,
                    help="Hidden width of the domain MLP.")
    ap.add_argument("--domain-grouping", default="file_id",
                    choices=["file_id", "subclass", "cal_date"],
                    help="Granularity of the GRL target. file_id (87-way, default) is "
                         "the original DANN setup; subclass (10-way) and cal_date (13-way) "
                         "test the grouped-domain hypothesis (less destructive, preserves "
                         "within-strain or within-acquisition-batch shared features).")
    ap.add_argument("--model-name", default=None,
                    help="Override model_name used for the run_id. Defaults "
                         "to cnn_dann_lam<X> e.g. cnn_dann_lam0.10.")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    splits_dir = Path(args.splits_dir)
    outputs_dir = Path(args.outputs_dir)
    runs_log = outputs_dir / "runs.jsonl"

    spec_df = pd.read_parquet(cache_dir / "spectra.parquet")
    X_full = np.load(cache_dir / "spectra_array_preprocessed.npy")
    # Build the per-row domain label array up-front (full dataset). The
    # per-fold slice happens inside run_one. Doing it here once means we
    # crash early if spec_df is missing the column the chosen grouping
    # needs, rather than mid-sweep.
    domain_labels_all = _build_domain_labels(spec_df, args.domain_grouping)
    n_unique_global = len(set(domain_labels_all.tolist()))
    print(
        f"Domain grouping: {args.domain_grouping} -> {n_unique_global} unique "
        f"domains globally; per-fold counts will be smaller (outer-train only)."
    )

    device = torch.device(args.device) if args.device else select_device()

    aug = AugConfig() if not args.no_aug else AugConfig(
        p_noise=0.0, p_scale=0.0, p_shift=0.0, p_baseline=0.0, p_mixup=0.0,
    )

    cfg = TrainConfig(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        aug=aug,
    )
    dann_cfg = DANNConfig(
        lambda_max=args.lambda_max,
        warmup_epochs_dann=args.warmup_epochs_dann,
        domain_hidden=args.domain_hidden,
    )

    # Tag model_name with the domain grouping when it's not the default.
    if args.model_name:
        model_name = args.model_name
    elif args.domain_grouping == "file_id":
        model_name = f"cnn_dann_lam{args.lambda_max:.2f}"
    else:
        model_name = f"cnn_dann_lam{args.lambda_max:.2f}_dom_{args.domain_grouping}"

    print(f"Loaded preprocessed array shape={X_full.shape}, dtype={X_full.dtype}")
    print(f"Device: {device}  torch: {torch.__version__}")
    print(
        f"Train cfg: n_epochs={cfg.n_epochs}  batch_size={cfg.batch_size}  "
        f"lr={cfg.lr}  patience={cfg.patience}  aug={'OFF' if args.no_aug else 'ON'}"
    )
    print(
        f"DANN cfg: lambda_max={dann_cfg.lambda_max}  "
        f"warmup_epochs_dann={dann_cfg.warmup_epochs_dann}  "
        f"domain_hidden={dann_cfg.domain_hidden}"
    )
    print(f"Protocols: {args.protocols}\n")

    # Param-count print: use 87 domains as a representative count (Protocol A
    # outer-train ~70, LOSO ~78; gets re-sized per fold inside the factory).
    tmp = DANNCNN1D(
        n_bins=X_full.shape[1], n_classes=4,
        n_domains=87, domain_hidden=args.domain_hidden,
    )
    print(
        f"DANNCNN1D params (encoder+class+domain @ 87 domains): "
        f"{count_params(tmp):,}"
    )
    del tmp

    PROTOCOL_TO_FILE = {"group_kfold": "protocol_a.json", "loso": "protocol_b.json"}

    all_summaries = []
    for protocol in args.protocols:
        splits_path = splits_dir / PROTOCOL_TO_FILE[protocol]
        if args.folds:
            splits = load_splits(splits_path)
            requested = set(str(x) for x in args.folds)
            kept = [f for f in splits["folds"] if str(f["fold"]) in requested]
            if not kept:
                print(
                    f"  no folds in {splits_path} matched --folds={args.folds}; "
                    f"available: {[str(f['fold']) for f in splits['folds']]}"
                )
                continue
            splits["folds"] = kept
            tmp_path = outputs_dir / f"_tmp_splits_{protocol}_dann_{int(time.time())}.json"
            with open(tmp_path, "w") as f:
                json.dump(splits, f)
            splits_path = tmp_path

        summary = run_one(
            protocol=protocol,
            splits_path=splits_path,
            spec_df=spec_df,
            X_full=X_full,
            domain_labels_all=domain_labels_all,
            outputs_dir=outputs_dir,
            seed=args.seed,
            cfg=cfg,
            dann_cfg=dann_cfg,
            domain_grouping=args.domain_grouping,
            device=device,
            model_name=model_name,
            runs_log=runs_log,
        )
        all_summaries.append(summary)
        print()

    print("=" * 90)
    print(
        f"{'model':>18}  {'protocol':>11}  {'lam':>5}  "
        f"{'file_F1 mean':>12}  {'SD':>5}  {'min':>5}  {'max':>5}  per-fold"
    )
    print("-" * 90)
    for r in all_summaries:
        per_fold = " ".join(f"{x:.2f}" for x in r["file_macro_f1_per_fold"])
        print(
            f"{r['model']:>18}  {r['protocol']:>11}  {r['lambda_max']:>5.2f}  "
            f"{r['file_macro_f1_mean']:>12.3f}  {r['file_macro_f1_sd']:>5.3f}  "
            f"{r['file_macro_f1_min']:>5.3f}  {r['file_macro_f1_max']:>5.3f}  "
            f"{per_fold}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
