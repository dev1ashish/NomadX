"""Run classical models under Protocol A and/or LOSO.

For each (model, protocol):
    - Iterate the fold artifact's folds.
    - Slice the preprocessed array with the fold's train/test row_indices.
    - Random-search HPO on inner fold 0; refit on full outer-train.
    - Compute FoldResult via atlas.evaluate.
    - Save per-fold predictions to parquet, ModelResult to json.

Output layout:
    outputs/2026-05-14_<model>_<protocol>_<hash>/
        config.resolved.json
        model_result.json
        predictions_fold{0..K-1}.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Pin BLAS threads BEFORE numpy import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from atlas.evaluate import (  # noqa: E402
    aggregate,
    evaluate_fold,
    write_model_result,
    write_predictions_parquet,
)
from atlas.models_classical import MODEL_REGISTRY, get_model_spec  # noqa: E402
from atlas.splits import load_splits  # noqa: E402


def _short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def _runs_log_append(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_one(
    *,
    model_name: str,
    protocol: str,
    splits_path: Path,
    spec_df: pd.DataFrame,
    X_full: np.ndarray,
    outputs_dir: Path,
    seed: int,
    runs_log: Path,
) -> dict:
    spec = get_model_spec(model_name)
    splits = load_splits(splits_path)
    folds = splits["folds"]

    run_id = (
        f"{datetime.now().strftime('%Y-%m-%d')}_{model_name}_{protocol}_"
        f"{_short_hash(model_name + protocol + str(seed))}"
    )
    run_dir = outputs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    config = {
        "run_id": run_id,
        "model": model_name,
        "protocol": protocol,
        "splits_path": str(splits_path),
        "seed": seed,
        "n_trials": spec.n_trials,
        "sklearn_version": splits["meta"]["sklearn_version"],
        "split_cache_hash": splits["meta"]["cache_hash"],
    }
    with open(run_dir / "config.resolved.json", "w") as f:
        json.dump(config, f, indent=2)

    file_ids_all = spec_df["file_id"].to_numpy()
    y_all = spec_df["primary_class"].to_numpy()
    sub_all = spec_df["subclass"].to_numpy()

    from atlas.models_classical import train_fold

    fold_results = []
    t_run_start = time.perf_counter()
    fold_iter = tqdm(folds, desc=f"{model_name}/{protocol}", ncols=80, position=0)
    for fold in fold_iter:
        fold_id = fold["fold"]
        fold_iter.set_postfix(fold=str(fold_id)[:14])
        train_idx = np.asarray(fold["train_row_indices"], dtype=np.int64)
        test_idx = np.asarray(fold["test_row_indices"], dtype=np.int64)
        fold_seed = int(fold["fold_seed"])

        X_train = X_full[train_idx]
        y_train = y_all[train_idx]
        groups_train = file_ids_all[train_idx]
        X_test = X_full[test_idx]

        print(
            f"[{run_id}] fold={fold_id} "
            f"n_train={train_idx.size} ({len(set(groups_train))} files) "
            f"n_test={test_idx.size} ({len(set(file_ids_all[test_idx]))} files)"
        )
        proba_test, best_hp, train_dt = train_fold(
            spec=spec,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            X_test=X_test,
            fold_seed=fold_seed,
        )

        # Save predictions
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
            n_train_files=len(set(groups_train)),
            n_test_files=len(set(file_ids_all[test_idx])),
            training_time_s=train_dt,
            best_hyperparams=best_hp,
            predictions_path=str(pred_path),
        )
        fold_results.append(fr)
        print(
            f"  -> spectrum_macro_f1={fr.spectrum_macro_f1:.3f}  "
            f"file_macro_f1={fr.file_macro_f1:.3f}  "
            f"brier_file={fr.file_brier:.3f}"
        )

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
    _runs_log_append(runs_log, summary)

    print(
        f"[{run_id}] DONE  "
        f"file_macro_f1 = {mr.file_macro_f1_mean:.3f} +/- {mr.file_macro_f1_sd:.3f}  "
        f"(per-fold: {[f'{x:.3f}' for x in mr.file_macro_f1_per_fold]})  "
        f"duration={t_run:.1f}s"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        default=[m.name for m in MODEL_REGISTRY],
        help="Subset of models to run (default: all 6).",
    )
    ap.add_argument(
        "--protocols",
        nargs="+",
        default=["group_kfold", "loso"],
        choices=["group_kfold", "loso"],
    )
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "data_cache"))
    ap.add_argument("--splits-dir", default=str(REPO_ROOT / "data_cache" / "splits"))
    ap.add_argument("--outputs-dir", default=str(REPO_ROOT / "outputs"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    splits_dir = Path(args.splits_dir)
    outputs_dir = Path(args.outputs_dir)
    runs_log = outputs_dir / "runs.jsonl"

    spec_df = pd.read_parquet(cache_dir / "spectra.parquet")
    X_full = np.load(cache_dir / "spectra_array_preprocessed.npy")

    print(f"Loaded preprocessed array shape={X_full.shape}, dtype={X_full.dtype}")
    print(f"Models to run: {args.models}")
    print(f"Protocols: {args.protocols}")
    print()

    PROTOCOL_TO_FILE = {"group_kfold": "protocol_a.json", "loso": "protocol_b.json"}

    all_results = []
    for protocol in args.protocols:
        splits_path = splits_dir / PROTOCOL_TO_FILE[protocol]
        for model_name in args.models:
            t_start = time.perf_counter()
            summary = run_one(
                model_name=model_name,
                protocol=protocol,
                splits_path=splits_path,
                spec_df=spec_df,
                X_full=X_full,
                outputs_dir=outputs_dir,
                seed=args.seed,
                runs_log=runs_log,
            )
            all_results.append(summary)
            print(f"  total: {time.perf_counter() - t_start:.1f}s\n")

    # Final table
    print()
    print("=" * 70)
    print(f"{'model':>10}  {'protocol':>11}  {'file_F1 mean':>12}  {'SD':>5}  "
          f"{'min':>5}  {'max':>5}  per-fold")
    print("-" * 70)
    for r in all_results:
        per_fold = " ".join(f"{x:.2f}" for x in r["file_macro_f1_per_fold"])
        print(
            f"{r['model']:>10}  {r['protocol']:>11}  "
            f"{r['file_macro_f1_mean']:>12.3f}  {r['file_macro_f1_sd']:>5.3f}  "
            f"{r['file_macro_f1_min']:>5.3f}  {r['file_macro_f1_max']:>5.3f}  "
            f"{per_fold}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
