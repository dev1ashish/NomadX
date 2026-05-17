"""Temperature-scaled soft-vote over pre-computed LOSO predictions.

Fits a per-base scalar temperature on the 8-other-folds (LOO-on-strains)
calibration set, applies it to the held-out fold's predictions, then
soft-votes uniform across the included bases.

Two variants run by default in one invocation:
  - 4-base temperature-scaled (all 4 included)
  - 3-base temperature-scaled excluding PLS-DA (sanity check)

Outputs use the standard predictions_fold_<strain>.parquet schema so
downstream eval works unchanged.

Usage:
    .venv/bin/python scripts/run_calibrated_ensemble.py \
        --base plsda=outputs/2026-05-14_plsda_loso_9b4a9cb3 \
               dann=outputs/2026-05-14_cnn_dann_lam0.10_loso_c9ff8f33 \
               patch5=outputs/2026-05-14_transformer_p5_loso_74a86747 \
               cnn2ch=outputs/2026-05-14_cnn_small_deriv_loso_702b55c3
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

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from atlas.calibrated_ensemble import (  # noqa: E402
    temperature_calibrated_softvote_loso,
)


def _short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def _runs_log_append(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _run_one(
    tag: str,
    base_dirs: list[Path],
    base_names: list[str],
    exclude: list[str],
    outputs_dir: Path,
    runs_log: Path,
) -> dict:
    model_name = f"tcal_{tag}"
    run_id = (
        f"{datetime.now().strftime('%Y-%m-%d')}_{model_name}_loso_"
        f"{_short_hash(model_name + str([str(d) for d in base_dirs]) + str(exclude))}"
    )
    run_dir = outputs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    res = temperature_calibrated_softvote_loso(
        base_dirs, base_names, exclude_in_softvote=exclude
    )
    duration = time.perf_counter() - t0

    config = {
        "run_id": run_id,
        "model": model_name,
        "protocol": "loso",
        "base_runs": [{"name": n, "path": str(d)}
                      for n, d in zip(base_names, base_dirs)],
        "excluded_from_softvote": exclude,
        "softvote_bases": res["softvote_names"],
        "temperatures_per_fold": res["temperatures"],
        "timestamp": datetime.now().isoformat(),
    }
    with open(run_dir / "config.resolved.json", "w") as f:
        json.dump(config, f, indent=2)

    for fold_id, df in res["predictions"].items():
        df.to_parquet(run_dir / f"predictions_fold_{fold_id}.parquet")

    summary = {
        "run_id": run_id,
        "model": model_name,
        "protocol": "loso",
        "loso_mean_parent_recall": res["mean_parent_recall"],
        "per_strain_recall": res["per_strain_parent_recall"],
        "duration_s": duration,
        "timestamp": datetime.now().isoformat(),
    }
    with open(run_dir / "model_result.json", "w") as f:
        json.dump(summary, f, indent=2)
    _runs_log_append(runs_log, summary)

    print(f"\n[{run_id}] tag={tag}  duration={duration:.2f}s")
    print(f"  softvote_bases: {res['softvote_names']}")
    print(f"  excluded_from_softvote: {exclude}")
    print(f"  per-fold temperatures (mean over folds):")
    mean_T = {}
    for fold_id, T_dict in res["temperatures"].items():
        for n, T in T_dict.items():
            mean_T.setdefault(n, []).append(T)
    for n, Ts in mean_T.items():
        print(f"    {n:>10s} : T_mean = {np.mean(Ts):.3f}  "
              f"(range {np.min(Ts):.3f}–{np.max(Ts):.3f})")
    print(f"  per-strain parent recall:")
    for strain, r in sorted(res["per_strain_parent_recall"].items()):
        print(f"    {strain:>12s}  {r:.3f}")
    print(f"  MEAN parent-recall = {res['mean_parent_recall']:.3f}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base", nargs="+", required=True,
        help="Base run dirs, format 'name=path'. Order doesn't matter.",
    )
    ap.add_argument("--outputs-dir", default=str(REPO_ROOT / "outputs"))
    ap.add_argument(
        "--exclude-plsda-sanity", action="store_true", default=True,
        help="Also run a 3-base variant excluding PLS-DA from the soft-vote "
             "(default True — useful diagnostic).",
    )
    args = ap.parse_args()

    bases = []
    for spec in args.base:
        if "=" not in spec:
            raise ValueError(f"--base entries must be name=path; got: {spec}")
        name, path = spec.split("=", 1)
        bases.append((name.strip(), Path(path.strip())))
    base_names = [n for n, _ in bases]
    base_dirs = [d for _, d in bases]

    outputs_dir = Path(args.outputs_dir)
    runs_log = outputs_dir / "runs.jsonl"

    print("=" * 70)
    print("Temperature-scaled soft-vote — bases:")
    for n, d in bases:
        print(f"  {n:>10s} -> {d.name}")
    print("=" * 70)

    summaries = []
    summaries.append(_run_one(
        tag="4base_all", base_dirs=base_dirs, base_names=base_names,
        exclude=[], outputs_dir=outputs_dir, runs_log=runs_log,
    ))
    if args.exclude_plsda_sanity and "plsda" in base_names:
        summaries.append(_run_one(
            tag="3deep_noplsda", base_dirs=base_dirs, base_names=base_names,
            exclude=["plsda"], outputs_dir=outputs_dir, runs_log=runs_log,
        ))

    print("\n" + "=" * 70)
    print(f"{'variant':>20s}  {'mean parent-recall':>22s}")
    print("-" * 70)
    for s in summaries:
        print(f"{s['model']:>20s}  {s['loso_mean_parent_recall']:>22.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
