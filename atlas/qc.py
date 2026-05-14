"""Quality-control filters for preprocessed Raman spectra (PLAN.md §3.3.B).

Two filters:
    1. SNR ratio over a signal band vs noise band -- spectra below threshold
       are mostly noise (out-of-focus, no cells under laser, etc.).
    2. Background pixels -- in some maps, many pixels never touched a cell.
       Identify them as candidates by being in the bottom 10th percentile
       of integrated fingerprint intensity within their file AND being
       spectrally flat.

Both work on **pre-crop** spectra (need the noise region 1800-2500 cm^-1 which
is outside our crop window) -- so call before `crop_two_regions`.
"""

from __future__ import annotations

import numpy as np


def snr_per_spectrum(
    X: np.ndarray,
    wn: np.ndarray,
    *,
    signal_band: tuple[float, float] = (900.0, 1700.0),
    noise_band: tuple[float, float] = (1800.0, 2500.0),
) -> np.ndarray:
    """Compute SNR per spectrum as mean(signal_band) / std(noise_band).

    The signal band covers most of the bacterial Raman fingerprint.
    The noise band is mostly silent in bacteria so its std is a good
    estimate of baseline noise level.

    Works on 1-D or 2-D (N, M) input.
    """
    if X.ndim == 1:
        X = X[None, :]
    sig_mask = (wn >= signal_band[0]) & (wn <= signal_band[1])
    noi_mask = (wn >= noise_band[0]) & (wn <= noise_band[1])
    if sig_mask.sum() == 0 or noi_mask.sum() == 0:
        raise ValueError(
            f"SNR bands empty: signal {sig_mask.sum()} bins, noise {noi_mask.sum()} bins"
        )
    sig = X[:, sig_mask].mean(axis=1)
    noi = X[:, noi_mask].std(axis=1) + 1e-8
    return (sig / noi).astype(np.float32)


def background_mask(
    X: np.ndarray,
    wn: np.ndarray,
    file_ids: np.ndarray,
    *,
    fingerprint_band: tuple[float, float] = (600.0, 1700.0),
    pct: float = 10.0,
    flatness_mad_factor: float = 2.0,
) -> np.ndarray:
    """Mark candidate background pixels (laser hit no cell).

    For each file, find pixels in the bottom `pct` percentile of integrated
    fingerprint intensity AND with low spectral variation (MAD <
    flatness_mad_factor * median MAD across the file). Background pixels
    are featureless flat spectra at low intensity.

    Returns a boolean mask of length N (True = candidate background).
    """
    fp_mask = (wn >= fingerprint_band[0]) & (wn <= fingerprint_band[1])
    integrated = X[:, fp_mask].sum(axis=1)
    spec_mad = np.median(np.abs(X - np.median(X, axis=1, keepdims=True)), axis=1)
    is_bg = np.zeros(X.shape[0], dtype=bool)
    for fid in np.unique(file_ids):
        idx = np.where(file_ids == fid)[0]
        if idx.size < 5:
            continue
        low_int = integrated[idx] <= np.percentile(integrated[idx], pct)
        med_mad = np.median(spec_mad[idx])
        low_mad = spec_mad[idx] < flatness_mad_factor * med_mad
        is_bg[idx[low_int & low_mad]] = True
    return is_bg


def apply_qc(
    X: np.ndarray,
    wn: np.ndarray,
    file_ids: np.ndarray,
    *,
    snr_threshold: float = 5.0,
    drop_background: bool = True,
) -> tuple[np.ndarray, dict]:
    """Compute the full QC mask: True = keep, False = drop.

    Returns (keep_mask, info_dict) where info has per-file retention rates.
    """
    snr = snr_per_spectrum(X, wn)
    snr_ok = snr >= snr_threshold
    bg = background_mask(X, wn, file_ids) if drop_background else np.zeros_like(snr_ok, dtype=bool)
    keep = snr_ok & (~bg)

    per_file = {}
    for fid in np.unique(file_ids):
        m = file_ids == fid
        per_file[str(fid)] = {
            "n": int(m.sum()),
            "kept": int((m & keep).sum()),
            "retention": float((m & keep).mean()),
            "median_snr": float(np.median(snr[m])),
            "n_bg": int((m & bg).sum()),
        }

    info = {
        "n_input": int(X.shape[0]),
        "n_keep": int(keep.sum()),
        "n_drop_snr": int((~snr_ok).sum()),
        "n_drop_bg": int(bg.sum()),
        "median_snr_overall": float(np.median(snr)),
        "per_file": per_file,
    }
    return keep, info
