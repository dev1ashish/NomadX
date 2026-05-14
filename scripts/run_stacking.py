"""Train a LOSO stacking meta-learner over saved per-fold base-model predictions.

The base models are existing run dirs from scripts/run_{classical,cnn,dann}.py.
This script:
  1. Discovers the base runs (from CLI args or sensible defaults).
  2. Aggregates each base model's per-spectrum predictions to file-level.
  3. Builds the stacking dataset: one row per file × fold, with each base model's
     file-level probas as features (16-D for 4 base models × 4 classes).
  4. Runs LOSO meta-CV: hold one strain out, train LogReg meta on the other 8,
     predict the held-out strain.
  5. Computes per-strain parent-class recall (the headline LOSO metric).
  6. Persists output in the same schema as other model runs so downstream tools
     don't care it's a stacker:
       - outputs/<run_id>/predictions_fold_<strain>.parquet (file-level rows)
       - outputs/<run_id>/model_result.json (mean ± SD across folds)
       - outputs/<run_id>/config.resolved.json
       - outputs/runs.jsonl summary line

Default base set (if --base flags not provided):
  - PLS-DA       (strongest classical LOSO baseline; LOSO mean 0.60)
  - DANN λ=0.05  (lambda-curve low end)
  - DANN λ=0.1   (current headline DANN)
  - DANN λ=0.3   (high-pressure DANN)
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from atlas.evaluate import PRIMARY_CLASSES
from atlas.stacking import (
    StackingConfig,
    build_stacking_matrix,
    per_strain_parent_recall,
    train_predict_loso,
)


def _short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def _runs_log_append(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _default_base_runs(outputs_dir: Path) -> list[tuple[str, Path]]:
    """Best-guess default base set — most-recent run dir matching each tag."""
    candidates = {
        "plsda":   "*_plsda_loso_*",
        "dann05":  "*_cnn_dann_lam0.05_loso_*",
        "dann10":  "*_cnn_dann_lam0.10_loso_*",
        "dann30":  "*_cnn_dann_lam0.30_loso_*",
    }
    out = []
    for tag, glob in candidates.items():
        matches = sorted(outputs_dir.glob(glob))
        if not matches:
            raise FileNotFoundError(
                f"No run dir matching pattern {glob} under {outputs_dir}. "
                f"Run the base model first or pass --base explicitly."
            )
        out.append((tag, matches[-1]))  # most recent
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base", nargs="+", default=None,
        help="Base model run dirs, format 'name=path'. e.g. "
             "'plsda=outputs/2026-05-14_plsda_loso_9b4a9cb3'. If omitted, "
             "uses the default 4-model set (PLS-DA + 3 DANN lambdas)."
    )
    ap.add_argument("--outputs-dir", default=str(REPO_ROOT / "outputs"))
    ap.add_argument("--meta-c", type=float, default=1.0,
                    help="LogReg meta-learner inverse regularization strength.")
    ap.add_argument("--use-subclass", action="store_true",
                    help="Include one-hot subclass feature in meta input. "
                         "Safe under LOSO (held-out strain's one-hot is 0 in training, "
                         "1 in test — acts as 'unseen strain' indicator, no label leakage).")
    ap.add_argument("--no-class-weight", action="store_true",
                    help="Disable class_weight='balanced' on the meta-learner.")
    ap.add_argument("--model-name", default=None)
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    runs_log = outputs_dir / "runs.jsonl"

    # Resolve base model run dirs
    if args.base:
        bases = []
        for spec in args.base:
            if "=" not in spec:
                raise ValueError(f"--base entries must be name=path; got: {spec}")
            name, path = spec.split("=", 1)
            bases.append((name.strip(), Path(path.strip())))
    else:
        bases = _default_base_runs(outputs_dir)

    base_names = [n for n, _ in bases]
    base_dirs = [d for _, d in bases]

    model_name = args.model_name or f"stack_{'_'.join(base_names)}"
    run_id = (
        f"{datetime.now().strftime('%Y-%m-%d')}_{model_name}_loso_"
        f"{_short_hash(model_name + str(bases))}"
    )
    run_dir = outputs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = StackingConfig(
        use_subclass=args.use_subclass,
        meta_C=args.meta_c,
        use_class_weight=not args.no_class_weight,
    )

    config = {
        "run_id": run_id,
        "model": model_name,
        "protocol": "loso",
        "base_runs": [{"name": n, "path": str(d)} for n, d in bases],
        "meta_C": cfg.meta_C,
        "use_subclass": cfg.use_subclass,
        "use_class_weight": cfg.use_class_weight,
        "timestamp": datetime.now().isoformat(),
    }
    with open(run_dir / "config.resolved.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"[{run_id}]")
    print(f"  base models: {base_names}")
    print(f"  meta_C={cfg.meta_C}, use_subclass={cfg.use_subclass}, "
          f"class_weight={cfg.use_class_weight}")
    print()

    t0 = time.perf_counter()
    print("[1/3] Building stacking dataset (file-level)...")
    stack_df = build_stacking_matrix(base_dirs, base_names)
    n_files = len(stack_df)
    n_folds = stack_df["fold_id"].nunique()
    print(f"  {n_files} file-rows across {n_folds} LOSO folds.")

    print("[2/3] LOSO meta-CV with LogReg...")
    meta_preds = train_predict_loso(stack_df, base_names, cfg=cfg, log_fn=print)
    meta_preds.to_parquet(run_dir / "stacking_predictions_file_level.parquet")

    # Also split into per-strain parquets matching the standard schema so
    # downstream eval code works without modification.
    for strain in sorted(meta_preds["fold_id"].unique()):
        sub = meta_preds[meta_preds["fold_id"] == strain].copy()
        # Add spectrum_id placeholders so the schema matches; we report at
        # file level so spectrum_id == file_id is fine here.
        sub.insert(0, "spectrum_id", -1)
        for c in PRIMARY_CLASSES:
            sub[f"p_{c}"] = sub[f"meta_p_{c}"]
        out_cols = ["spectrum_id", "file_id", "subclass", "primary_true",
                    "fold_id"] + [f"p_{c}" for c in PRIMARY_CLASSES]
        sub[out_cols].to_parquet(run_dir / f"predictions_fold_{strain}.parquet")

    print("[3/3] Per-strain parent-class recall:")
    recalls = per_strain_parent_recall(meta_preds)
    print(f"  {'strain':<12s} recall")
    for strain in sorted([k for k in recalls if k != "MEAN"]):
        print(f"  {strain:<12s} {recalls[strain]:.3f}")
    print(f"  {'MEAN':<12s} {recalls['MEAN']:.3f}")

    duration = time.perf_counter() - t0

    summary = {
        "run_id": run_id,
        "model": model_name,
        "protocol": "loso",
        "n_base_models": len(bases),
        "base_names": base_names,
        "loso_mean_parent_recall": recalls["MEAN"],
        "per_strain_recall": {k: v for k, v in recalls.items() if k != "MEAN"},
        "duration_s": duration,
        "timestamp": datetime.now().isoformat(),
    }
    with open(run_dir / "model_result.json", "w") as f:
        json.dump(summary, f, indent=2)
    _runs_log_append(runs_log, summary)

    print()
    print(f"[{run_id}] DONE in {duration:.1f}s")
    print(f"  LOSO mean parent-recall = {recalls['MEAN']:.3f}")
    print(f"  artifacts at {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
