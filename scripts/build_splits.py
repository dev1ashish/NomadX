"""Build Protocol A (StratifiedGroupKFold-5) + Protocol B (LOSO-9) splits.

Reads:
    data_cache/spectra.parquet
    data_cache/metadata.parquet
    data_cache/qc_mask.npy

Writes:
    data_cache/splits/protocol_a.json
    data_cache/splits/protocol_b.json

Smoke check: zero pixel/file leakage across folds. Halts on violation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Pin BLAS threads BEFORE numpy import to ensure deterministic FP reductions.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from atlas.splits import (  # noqa: E402
    MASTER_SEED,
    SplitConfig,
    build_all_splits,
    verify_no_leakage,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "data_cache"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data_cache" / "splits"))
    ap.add_argument("--seed", type=int, default=MASTER_SEED)
    ap.add_argument("--n-outer", type=int, default=5)
    ap.add_argument("--n-inner", type=int, default=4)
    args = ap.parse_args()

    cfg = SplitConfig(
        cache_dir=Path(args.cache_dir),
        out_dir=Path(args.out_dir),
        n_outer_folds=args.n_outer,
        n_inner_folds=args.n_inner,
        seed=args.seed,
    )

    print(f"[splits] reading cache from {cfg.cache_dir}")
    print(f"[splits] writing to {cfg.out_dir}")
    print(f"[splits] seed={cfg.seed}  outer={cfg.n_outer_folds}  inner={cfg.n_inner_folds}")

    a, b = build_all_splits(cfg)

    qc_mask = np.load(cfg.cache_dir / "qc_mask.npy")
    n_total = int(qc_mask.size)

    print()
    print("=" * 60)
    print("PROTOCOL A — StratifiedGroupKFold(5) with H2O pre-balance")
    print("=" * 60)
    rep_a = verify_no_leakage(a, n_total, qc_mask)
    for fc in rep_a["fold_checks"]:
        cd = fc.get("test_class_dist", {})
        h2o = cd.get("H2O", 0)
        bact = sum(v for k, v in cd.items() if k != "H2O")
        ovr = fc.get("calibration_date_overlap", [])
        print(
            f"  fold {fc['fold']}: {fc['n_test_files']:>2} test files "
            f"({fc['n_test_rows']:>4} rows)  "
            f"class_dist={cd}  "
            f"cal_date_overlap={len(ovr)} {ovr if ovr else ''}"
        )
    if a["warnings"]:
        print()
        print("WARNINGS:")
        for w in a["warnings"]:
            print(f"  - {w}")

    print()
    print("=" * 60)
    print("PROTOCOL B — Leave-One-Strain-Out (9 folds)")
    print("=" * 60)
    rep_b = verify_no_leakage(b, n_total, qc_mask)
    for fc in rep_b["fold_checks"]:
        sub = fc["fold"]
        ovr = fc.get("calibration_date_overlap", [])
        print(
            f"  fold {sub:>14}: {fc['n_test_files']:>2} test files "
            f"({fc['n_test_rows']:>4} rows)  "
            f"cal_date_overlap={len(ovr)}"
        )
    if b["warnings"]:
        print()
        print("WARNINGS:")
        for w in b["warnings"]:
            print(f"  - {w}")

    print()
    print("LEAKAGE SMOKE CHECK: PASS  (no pixel or file appears in both train and test of any fold)")
    print()
    print(f"  Protocol A meta hash: {a['meta']['cache_hash'][:16]}...")
    print(f"  Protocol B meta hash: {b['meta']['cache_hash'][:16]}...")
    print(f"  sklearn version:      {a['meta']['sklearn_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
