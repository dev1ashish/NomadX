"""Apply full preprocessing to the cached spectra and save the result.

Reads:
    data_cache/spectra_array.npy        (raw, interpolated to canonical wn)
    data_cache/wavenumber_axis.npy
    data_cache/spectra.parquet          (for file_ids -> QC)

Writes:
    data_cache/spectra_array_preprocessed.npy      (N, ~987) float32
    data_cache/wavenumber_axis_preprocessed.npy    (~987,)
    data_cache/qc_mask.npy                         (N,) bool   True = keep
    data_cache/qc_info.json                        per-file retention etc.

Re-run any time atlas/preprocess.py changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from atlas.preprocess import preprocess_matrix  # noqa: E402
from atlas.qc import apply_qc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "data_cache"))
    ap.add_argument("--snr-threshold", type=float, default=5.0)
    ap.add_argument("--no-cosmic", action="store_true")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--no-snv", action="store_true")
    ap.add_argument("--skip-qc", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    X = np.load(cache / "spectra_array.npy")
    wn = np.load(cache / "wavenumber_axis.npy")

    import pandas as pd
    spec_df = pd.read_parquet(cache / "spectra.parquet")
    file_ids = spec_df["file_id"].astype(str).to_numpy()
    print(f"loaded X={X.shape}  wn={wn.shape}  file_ids={file_ids.shape}")

    # QC needs pre-crop wn for the noise band (1800-2500), so run BEFORE preprocessing
    if not args.skip_qc:
        # Apply minimal smoothing first so SNR isn't dominated by single-bin noise
        from atlas.preprocess import smooth_savgol
        X_for_qc = np.array([smooth_savgol(X[i]) for i in range(X.shape[0])])
        keep, qc_info = apply_qc(
            X_for_qc, wn, file_ids,
            snr_threshold=args.snr_threshold,
            drop_background=True,
        )
        print(
            f"QC: keep={qc_info['n_keep']}/{qc_info['n_input']} "
            f"(drop_snr={qc_info['n_drop_snr']}, drop_bg={qc_info['n_drop_bg']}, "
            f"median_snr={qc_info['median_snr_overall']:.1f})"
        )
        np.save(cache / "qc_mask.npy", keep)
        with (cache / "qc_info.json").open("w") as f:
            json.dump(qc_info, f, indent=2)
    else:
        keep = np.ones(X.shape[0], dtype=bool)

    # Full preprocessing pipeline
    Xp, wnp, _ = preprocess_matrix(
        X, wn,
        do_cosmic=not args.no_cosmic,
        do_baseline=not args.no_baseline,
        do_smooth=not args.no_smooth,
        do_crop=not args.no_crop,
        do_snv=not args.no_snv,
        progress=True,
    )
    print(f"\npreprocessed: {Xp.shape}  wn range [{wnp[0]:.1f}, {wnp[-1]:.1f}]")
    print(f"sanity: mean={Xp.mean():.2e} std={Xp.std():.3f}  (expect ~0, ~1)")

    np.save(cache / "spectra_array_preprocessed.npy", Xp)
    np.save(cache / "wavenumber_axis_preprocessed.npy", wnp)
    print(f"wrote {cache / 'spectra_array_preprocessed.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
