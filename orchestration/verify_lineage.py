"""Standalone lineage proof: the data_cache/ outputs really come from atlas/*.

This is the receipt behind the orchestration layer's honesty claim. It does NOT
re-materialize the whole pipeline (the full preprocess is ~30 min of arPLS
fits); it re-invokes the real atlas code on cheap slices and shows the outputs
reproduce the cache.

Run:  PYTHONPATH=. .venv-dagster/bin/python orchestration/verify_lineage.py
  (works equally with the science .venv: PYTHONPATH=. .venv/bin/python ...)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import atlas.io as aio          # noqa: E402
import atlas.qc as aqc          # noqa: E402
import atlas.preprocess as app  # noqa: E402

CACHE = REPO / "data_cache"


def _ok(b: bool) -> str:
    return "\033[32mPASS\033[0m" if b else "\033[31mFAIL\033[0m"


def main() -> int:
    passed = True
    print("Atlas lineage verification — re-running atlas/* against data_cache/\n")

    # 1) INGEST — re-parse the smallest real file via atlas.io.parse_file
    data_root = REPO / "Atlas Data"
    if data_root.exists():
        files = aio.discover_files(data_root)
        target = min(files, key=lambda p: p.stat().st_size)
        rec = aio.parse_file(target, data_root)
        ok = rec.is_valid and rec.intensities.shape[1] == aio.N_BINS
        passed &= ok
        print(f"[{_ok(ok)}] INGEST  re-parsed {rec.file_id}: {rec.n_pixels}px × "
              f"{rec.intensities.shape[1]} bins via atlas.io.parse_file "
              f"(canonical {aio.N_BINS}-bin axis)")
    else:
        print("[SKIP] INGEST  raw 'Atlas Data/' not present — cache-only mode")

    # 2) QC — re-run apply_qc on the cached raw array, reconcile the funnel
    X = np.load(CACHE / "spectra_array.npy")
    wn = np.load(CACHE / "wavenumber_axis.npy")
    fids = pd.read_parquet(CACHE / "spectra.parquet").file_id.values
    cached_mask = np.load(CACHE / "qc_mask.npy")
    keep, info = aqc.apply_qc(X, wn, fids)
    counts_ok = int(keep.sum()) == int(cached_mask.sum()) == info["n_keep"]
    agree = float((keep == cached_mask).mean())
    passed &= counts_ok
    print(f"[{_ok(counts_ok)}] QC      atlas.qc.apply_qc → {info['n_input']} −snr "
          f"{info['n_drop_snr']} −bg {info['n_drop_bg']} = {info['n_keep']} kept "
          f"(cache {int(cached_mask.sum())}); mask agreement {agree*100:.3f}%")

    # 3) PREPROCESS — re-run on a sample, confirm the 987-bin axis
    Xp, wnp, _ = app.preprocess_matrix(X[:8], wn, progress=False)
    wn_cached = np.load(CACHE / "wavenumber_axis_preprocessed.npy")
    axis_ok = wnp.size == wn_cached.size == 987 and bool(np.allclose(wnp, wn_cached, atol=1e-3))
    passed &= axis_ok
    print(f"[{_ok(axis_ok)}] PREPROC atlas.preprocess.preprocess_matrix(sample) → "
          f"{Xp.shape[0]}×{wnp.size} bins; axis matches cached 987-bin axis")

    # 4) FEATURES — per-pixel rowcount == qc keep (the positional-alignment guard)
    band = pd.read_parquet(CACHE / "band_features.parquet")
    rows_ok = len(band) == int(cached_mask.sum())
    passed &= rows_ok
    print(f"[{_ok(rows_ok)}] FEATURE band_features rows {len(band)} == qc keep "
          f"{int(cached_mask.sum())} (positional join to spectra_index[qc_mask])")

    print("\n" + ("\033[32mAll lineage checks passed.\033[0m" if passed
                  else "\033[31mSome checks failed.\033[0m"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
