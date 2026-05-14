"""Run a soft-vote ensemble over pre-computed model run dirs.

Mirrors scripts/run_classical.py's output layout:
    outputs/<run_id>/
        config.resolved.json
        predictions_fold_*.parquet
        model_result.json

Usage:
    .venv/bin/python scripts/run_ensemble.py \
        --runs outputs/2026-05-14_plsda_loso_9b4a9cb3 outputs/2026-05-14_xgb_loso_2d3b9e19 \
        --protocol loso \
        --name plsda_xgb
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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from atlas.evaluate import (  # noqa: E402
    aggregate,
    evaluate_fold,
    write_model_result,
    write_predictions_parquet,
)
from atlas.ensemble import discover_fold_ids, soft_vote_ensemble  # noqa: E402


def _short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def _runs_log_append(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run dirs to ensemble (must each contain predictions_fold_*.parquet).",
    )
    ap.add_argument(
        "--protocol",
        choices=["group_kfold", "loso"],
        required=True,
    )
    ap.add_argument(
        "--name",
        required=True,
        help="Short ensemble name, e.g. plsda_xgb, plsda_cnn, plsda_xgb_cnn.",
    )
    ap.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=None,
        help="Optional per-run weights (default: uniform).",
    )
    ap.add_argument("--outputs-dir", default=str(REPO_ROOT / "outputs"))
    args = ap.parse_args()

    run_dirs = [Path(p).resolve() for p in args.runs]
    for d in run_dirs:
        if not d.exists():
            print(f"ERROR: run dir not found: {d}", file=sys.stderr)
            return 1

    outputs_dir = Path(args.outputs_dir)
    runs_log = outputs_dir / "runs.jsonl"

    fold_ids = discover_fold_ids(run_dirs)

    model_name = f"ens_{args.name}"
    run_id = (
        f"{datetime.now().strftime('%Y-%m-%d')}_{model_name}_{args.protocol}_"
        f"{_short_hash(model_name + args.protocol + '|'.join(args.runs))}"
    )
    run_dir = outputs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id,
        "model": model_name,
        "protocol": args.protocol,
        "input_runs": [str(d) for d in run_dirs],
        "weights": args.weights or [1.0 / len(run_dirs)] * len(run_dirs),
        "fold_ids": fold_ids,
    }
    with open(run_dir / "config.resolved.json", "w") as f:
        json.dump(config, f, indent=2)

    fold_results = []
    held_out_parents: dict[str, str] = {}
    t_run_start = time.perf_counter()
    for fold_id in fold_ids:
        t0 = time.perf_counter()
        _sids, file_ids, subclass, y_true, y_proba = soft_vote_ensemble(
            run_dirs, fold_id, weights=args.weights
        )
        elapsed = time.perf_counter() - t0
        # For LOSO each fold has a single held-out parent class; record it for
        # the per-strain parent-recall table below.
        uniq = sorted(set(y_true.tolist()))
        held_out_parents[fold_id] = uniq[0] if len(uniq) == 1 else ""

        pred_path = run_dir / f"predictions_fold_{fold_id}.parquet"
        write_predictions_parquet(
            fold_id=fold_id,
            file_ids=file_ids,
            subclass=subclass,
            y_true=y_true,
            y_proba=y_proba,
            out_path=pred_path,
        )

        fr = evaluate_fold(
            fold_id=fold_id,
            protocol=args.protocol,
            model_name=model_name,
            y_true=y_true,
            y_proba=y_proba,
            file_ids=file_ids,
            subclass=subclass,
            n_train=0,
            n_train_files=0,
            n_test_files=len(set(file_ids)),
            training_time_s=elapsed,
            best_hyperparams={"input_runs": [str(d) for d in run_dirs]},
            predictions_path=str(pred_path),
        )
        fold_results.append(fr)
        print(
            f"[{run_id}] fold={fold_id}  "
            f"spectrum_macro_f1={fr.spectrum_macro_f1:.3f}  "
            f"file_macro_f1={fr.file_macro_f1:.3f}  "
            f"per-subclass (file): "
            + ", ".join(f"{k}={v:.2f}" for k, v in fr.per_subclass_recall_file.items()
                        if not (isinstance(v, float) and v != v))  # skip NaN
        )

    mr = aggregate(fold_results, model_name=model_name, protocol=args.protocol)
    write_model_result(mr, run_dir / "model_result.json")
    t_run = time.perf_counter() - t_run_start

    # LOSO parent-class recall: per-strain, look up the held-out parent's recall
    # in file_per_class_recall (which is recall_score over PRIMARY_CLASSES).
    parent_recalls = []
    for fr in fold_results:
        held = held_out_parents.get(str(fr.fold_id), "")
        if held and held in fr.file_per_class_recall:
            parent_recalls.append(fr.file_per_class_recall[held])
        else:
            # Fallback for group_kfold: balanced_acc is reasonable proxy.
            parent_recalls.append(fr.file_balanced_acc)

    summary = {
        "run_id": run_id,
        "model": model_name,
        "protocol": args.protocol,
        "input_runs": [str(d) for d in run_dirs],
        "file_macro_f1_mean": mr.file_macro_f1_mean,
        "file_macro_f1_sd": mr.file_macro_f1_sd,
        "file_macro_f1_per_fold": mr.file_macro_f1_per_fold,
        "parent_recall_per_fold": parent_recalls,
        "parent_recall_mean": float(np.mean(parent_recalls)),
        "duration_s": t_run,
        "timestamp": datetime.now().isoformat(),
    }
    _runs_log_append(runs_log, summary)

    print()
    print(f"[{run_id}] DONE")
    print(
        f"  file_macro_f1 = {mr.file_macro_f1_mean:.3f} +/- {mr.file_macro_f1_sd:.3f}"
    )
    if args.protocol == "loso":
        # Print per-strain parent recall (file-level), the headline LOSO metric.
        print(f"  parent_recall_mean (LOSO) = {summary['parent_recall_mean']:.3f}")
        print("  per-strain parent recall (file-level):")
        for fr, pr in zip(fold_results, parent_recalls):
            held = held_out_parents.get(str(fr.fold_id), "?")
            print(f"    {str(fr.fold_id):>12}  ({held})  parent_recall={pr:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
