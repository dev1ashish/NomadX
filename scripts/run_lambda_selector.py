"""Per-strain optimal-lambda selector for DANN base models (LOSO only).

For each LOSO test strain, use a non-leaky signal (inner-val macro-F1 on
the outer-train, or test-time file-level confidence) to pick which DANN
lambda to use for the final prediction.

Default base set: {DANN(0.05), DANN(0.1), DANN(0.3)} from the most-recent
LOSO runs in outputs/. Three selectors run in one invocation:

  hard    -- argmax(inner_val_f1) per strain. Leakage-free.
  soft    -- softmax(inner_val_f1)-weighted average. Leakage-free.
  router  -- per-file argmax(mean max-proba). Mild leakage (uses test-set
             confidence but not labels); documented honestly.

Outputs three run dirs (one per selector) with the standard
predictions_fold_<strain>.parquet schema so the rest of the eval pipeline
works without modification.
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
from atlas.lambda_selector import (
    LambdaCandidate,
    hard_predictions_for_fold,
    soft_predictions_for_fold,
    router_predictions_for_fold,
    per_strain_parent_recall_from_predictions,
)


def _short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def _runs_log_append(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _default_candidates(outputs_dir: Path) -> list[LambdaCandidate]:
    candidates = [
        ("lam0.05", "*_cnn_dann_lam0.05_loso_*"),
        ("lam0.10", "*_cnn_dann_lam0.10_loso_*"),
        ("lam0.30", "*_cnn_dann_lam0.30_loso_*"),
    ]
    out = []
    for name, glob in candidates:
        matches = sorted([m for m in outputs_dir.glob(glob) if "dom_" not in m.name])
        if not matches:
            raise FileNotFoundError(f"No run dir matching {glob} in {outputs_dir}")
        out.append(LambdaCandidate(name=name, run_dir=matches[-1]))
    return out


def discover_fold_ids(run_dir: Path) -> list[str]:
    return sorted([p.stem[len("predictions_fold_"):]
                   for p in run_dir.glob("predictions_fold_*.parquet")])


def run_selector(
    selector_name: str,
    candidates: list[LambdaCandidate],
    fold_ids: list[str],
    outputs_dir: Path,
    runs_log: Path,
    temperature: float = 1.0,
) -> dict:
    """Run one selector variant across all LOSO folds."""
    cand_tag = "_".join(c.name for c in candidates)
    model_name = f"lambda_select_{selector_name}_{cand_tag}"
    run_id = (
        f"{datetime.now().strftime('%Y-%m-%d')}_{model_name}_loso_"
        f"{_short_hash(model_name + str([str(c.run_dir) for c in candidates]))}"
    )
    run_dir = outputs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id,
        "model": model_name,
        "selector": selector_name,
        "candidates": [{"name": c.name, "run_dir": str(c.run_dir)} for c in candidates],
        "temperature": temperature if selector_name == "soft" else None,
        "timestamp": datetime.now().isoformat(),
    }
    with open(run_dir / "config.resolved.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[{run_id}] selector={selector_name}")
    if selector_name == "soft":
        print(f"  temperature={temperature}")

    per_strain_recall = {}
    selection_log = {}  # fold -> which candidate (or weights)

    for fold_id in fold_ids:
        if selector_name == "hard":
            pred_df, info = hard_predictions_for_fold(candidates, fold_id)
        elif selector_name == "soft":
            pred_df, info = soft_predictions_for_fold(candidates, fold_id, temperature=temperature)
        elif selector_name == "router":
            pred_df, info = router_predictions_for_fold(candidates, fold_id, signal="max_proba")
        elif selector_name == "router_margin":
            pred_df, info = router_predictions_for_fold(candidates, fold_id, signal="margin")
        else:
            raise ValueError(f"unknown selector: {selector_name}")

        pred_df.to_parquet(run_dir / f"predictions_fold_{fold_id}.parquet")
        recall = per_strain_parent_recall_from_predictions(pred_df)
        per_strain_recall[fold_id] = recall
        selection_log[fold_id] = info

        if selector_name == "hard":
            tag = info.get("selected", "?")
            vals = info.get("inner_val_f1", {})
            val_summary = " ".join(f"{n}={v:.3f}" for n, v in vals.items())
            print(f"  fold={fold_id:12s} recall={recall:.3f}  picked={tag}  ({val_summary})")
        elif selector_name == "soft":
            ws = info.get("weights", {})
            w_summary = " ".join(f"{n}={w:.2f}" for n, w in ws.items())
            print(f"  fold={fold_id:12s} recall={recall:.3f}  weights={w_summary}")
        else:  # router or router_margin
            routing = info.get("routing", {})
            # Count how often each candidate was chosen across this fold's files
            counts = {c.name: 0 for c in candidates}
            for chosen in routing.values():
                counts[chosen] = counts.get(chosen, 0) + 1
            counts_str = " ".join(f"{n}={k}" for n, k in counts.items())
            print(f"  fold={fold_id:12s} recall={recall:.3f}  routing_counts={counts_str}")

    mean_recall = float(np.mean(list(per_strain_recall.values())))
    print(f"  MEAN parent-recall = {mean_recall:.3f}")

    summary = {
        "run_id": run_id,
        "model": model_name,
        "selector": selector_name,
        "loso_mean_parent_recall": mean_recall,
        "per_strain_recall": per_strain_recall,
        "selection_log": selection_log,
    }
    with open(run_dir / "model_result.json", "w") as f:
        json.dump(summary, f, indent=2)
    _runs_log_append(runs_log, {
        "run_id": run_id,
        "model": model_name,
        "protocol": "loso",
        "loso_mean_parent_recall": mean_recall,
        "per_strain_recall": per_strain_recall,
        "timestamp": datetime.now().isoformat(),
    })
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default=str(REPO_ROOT / "outputs"))
    ap.add_argument("--selectors", nargs="+", default=["hard", "soft", "router"],
                    choices=["hard", "soft", "router", "router_margin"])
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Softmax temperature for soft selector. Default 1.0; "
                         "smaller -> more concentrated on the best lambda.")
    ap.add_argument(
        "--base", nargs="+", default=None,
        help="Override default DANN-lambda set with arbitrary base run dirs, "
             "format 'name=path' (one per base). Example: "
             "'plsda=outputs/2026-05-14_plsda_loso_9b4a9cb3 "
             "dann10=outputs/2026-05-14_cnn_dann_lam0.10_loso_c9ff8f33'. "
             "Note: hard/soft selectors require history_fold_*.json with "
             "best_val_macro_f1 in each base run dir (DANN/CNN runs have it; "
             "PLS-DA classical runs do not). If a base is missing the history "
             "file, hard/soft fall back to NaN→worst slot. Router uses only "
             "test-time confidence and works for any base.",
    )
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    runs_log = outputs_dir / "runs.jsonl"

    if args.base:
        candidates = []
        for spec in args.base:
            if "=" not in spec:
                raise ValueError(f"--base entries must be name=path; got: {spec}")
            name, path = spec.split("=", 1)
            candidates.append(LambdaCandidate(name=name.strip(),
                                              run_dir=Path(path.strip())))
    else:
        candidates = _default_candidates(outputs_dir)

    print("Base candidates (most-recent LOSO runs):")
    for c in candidates:
        print(f"  {c.name:12s} -> {c.run_dir.name}")

    # All candidates must have the same fold set
    fold_ids = discover_fold_ids(candidates[0].run_dir)
    for c in candidates[1:]:
        other_folds = discover_fold_ids(c.run_dir)
        if set(other_folds) != set(fold_ids):
            raise ValueError(
                f"fold set mismatch: {candidates[0].name} has {fold_ids}, "
                f"{c.name} has {other_folds}"
            )
    print(f"\nFold ids: {fold_ids}")

    summaries = []
    for sel in args.selectors:
        s = run_selector(sel, candidates, fold_ids, outputs_dir, runs_log,
                         temperature=args.temperature)
        summaries.append(s)

    print("\n" + "=" * 70)
    print(f"{'selector':>10s}  {'mean parent-recall':>20s}")
    print("-" * 70)
    for s in summaries:
        print(f"{s['selector']:>10s}  {s['loso_mean_parent_recall']:>20.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
