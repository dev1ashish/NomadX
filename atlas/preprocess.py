"""Preprocessing pipeline for Atlas Raman spectra (per PLAN.md §3.3.B).

Pipeline order (per spectrum):
    1. Cosmic-ray spike removal  — median-filter outlier detection.
    2. Baseline subtraction      — arPLS (pybaselines.whittaker.arpls).
    3. Smoothing                 — Savitzky-Golay (window=9, polyorder=3).
    4. Crop                      — fingerprint 400-1800 + C-H 2800-3050.
    5. SNV normalization         — zero mean, unit variance per spectrum.
    6. (optional) 2nd derivative for classical-model feature augmentation.

Functions are pure NumPy and work on either a 1-D spectrum or a 2-D (N, M)
matrix where M is the number of wavenumber bins.

The default knobs match PLAN.md. Override via kwargs if you want to compare.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import medfilt, savgol_filter

from pybaselines.whittaker import arpls


# ---- atomic steps ----------------------------------------------------------


def remove_cosmic_rays(
    spec: np.ndarray, *, threshold: float = 5.0, window: int = 7
) -> np.ndarray:
    """Replace 1-2 bin upward spikes with the local median.

    Cosmic rays show up as very narrow, intense upward spikes in Raman
    spectra. They have a different shape (~1 bin wide) than real Raman
    peaks (~5-15 bins wide), so a median filter robustly removes them.

    We compute median-filtered version, then mark bins where the residual
    is far above the noise floor (estimated by MAD for robustness against
    the spike itself).
    """
    spec = np.asarray(spec, dtype=np.float32)
    if window % 2 == 0:
        window += 1
    med = medfilt(spec, kernel_size=window)
    residual = spec - med
    mad = np.median(np.abs(residual - np.median(residual)))
    sigma = 1.4826 * mad  # MAD -> ~stdev for normal
    if sigma < 1e-6:
        return spec.copy()
    z = residual / sigma
    cleaned = spec.copy()
    cleaned[z > threshold] = med[z > threshold]
    return cleaned


def remove_baseline_arpls(
    spec: np.ndarray, *, lam: float = 1e5, max_iter: int = 50,
    diff_order: int = 2,
) -> np.ndarray:
    """Subtract arPLS-estimated baseline. Returns baseline-corrected spectrum.

    arPLS (asymmetrically reweighted penalized least squares) fits a smooth
    curve under the spectrum, treating downward-pointing residuals as
    "definitely peaks" (don't fit those) and upward-pointing residuals as
    "could be baseline" (fit those). Effective at removing fluorescence.
    """
    baseline, _ = arpls(
        spec, lam=lam, max_iter=max_iter, diff_order=diff_order
    )
    return (spec - baseline).astype(np.float32)


def smooth_savgol(
    spec: np.ndarray, *, window: int = 9, polyorder: int = 3
) -> np.ndarray:
    """Savitzky-Golay smoothing — fits a local polynomial of `polyorder`
    in a window of `window` bins. Reduces high-frequency noise while
    preserving peak shape (better than a moving average for Raman)."""
    if window % 2 == 0:
        window += 1
    return savgol_filter(spec, window, polyorder).astype(np.float32)


def derivative_2(
    spec: np.ndarray, *, window: int = 9, polyorder: int = 3
) -> np.ndarray:
    """2nd-derivative via Savitzky-Golay. Mathematically removes any linear
    or constant baseline, sharpens peaks. Standard chemometrics trick."""
    if window % 2 == 0:
        window += 1
    return savgol_filter(spec, window, polyorder, deriv=2).astype(np.float32)


def crop_two_regions(
    X: np.ndarray,
    wn: np.ndarray,
    *,
    r1: tuple[float, float] = (400.0, 1800.0),
    r2: tuple[float, float] = (2800.0, 3050.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop to fingerprint + C-H stretch. Returns (cropped, wn_cropped, mask).

    Works on either a 1-D spectrum or a 2-D (N, M) matrix.
    """
    m1 = (wn >= r1[0]) & (wn <= r1[1])
    m2 = (wn >= r2[0]) & (wn <= r2[1])
    keep = m1 | m2
    if X.ndim == 1:
        return X[keep], wn[keep], keep
    return X[:, keep], wn[keep], keep


def snv(X: np.ndarray) -> np.ndarray:
    """Standard normal variate — z-score per spectrum (not per bin).

    Corrects multiplicative scatter from focus / cell-density variation,
    so that all spectra live on the same intensity scale.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        mu = X.mean()
        sd = X.std() + 1e-8
        return ((X - mu) / sd).astype(np.float32)
    mu = X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True) + 1e-8
    return ((X - mu) / sd).astype(np.float32)


# ---- pipeline orchestration ------------------------------------------------


def preprocess_spectrum(
    spec: np.ndarray,
    wn: np.ndarray,
    *,
    do_cosmic: bool = True,
    do_baseline: bool = True,
    do_smooth: bool = True,
    do_crop: bool = True,
    do_snv: bool = True,
    arpls_lam: float = 1e5,
    savgol_window: int = 9,
    savgol_poly: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Full per-spectrum pipeline. Returns (processed_spec, processed_wn)."""
    out = np.asarray(spec, dtype=np.float32).copy()
    if do_cosmic:
        out = remove_cosmic_rays(out)
    if do_baseline:
        out = remove_baseline_arpls(out, lam=arpls_lam)
    if do_smooth:
        out = smooth_savgol(out, window=savgol_window, polyorder=savgol_poly)
    if do_crop:
        out, wn_out, _ = crop_two_regions(out, wn)
    else:
        wn_out = wn
    if do_snv:
        out = snv(out)
    return out, wn_out


def preprocess_matrix(
    X: np.ndarray,
    wn: np.ndarray,
    *,
    do_cosmic: bool = True,
    do_baseline: bool = True,
    do_smooth: bool = True,
    do_crop: bool = True,
    do_snv: bool = True,
    arpls_lam: float = 1e5,
    savgol_window: int = 9,
    savgol_poly: int = 3,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply pipeline to N spectra at once.

    Returns (X_processed, wn_processed, keep_mask).
    Slow steps (cosmic-ray, arPLS) are row-by-row; vectorized steps
    (smoothing, crop, SNV) operate on the matrix.
    """
    Xp = np.asarray(X, dtype=np.float32).copy()
    n = Xp.shape[0]

    iterator = range(n)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(range(n), desc="preprocess", unit="spec")
        except ImportError:
            pass

    if do_cosmic or do_baseline or do_smooth:
        for i in iterator:
            s = Xp[i]
            if do_cosmic:
                s = remove_cosmic_rays(s)
            if do_baseline:
                s = remove_baseline_arpls(s, lam=arpls_lam)
            if do_smooth:
                s = smooth_savgol(s, window=savgol_window, polyorder=savgol_poly)
            Xp[i] = s

    if do_crop:
        Xp, wn_out, keep = crop_two_regions(Xp, wn)
    else:
        wn_out = wn
        keep = np.ones_like(wn, dtype=bool)

    if do_snv:
        Xp = snv(Xp)

    return Xp, wn_out, keep


def derivative_matrix(
    X: np.ndarray, *, window: int = 9, polyorder: int = 3
) -> np.ndarray:
    """2nd-derivative each row. Vectorized."""
    if window % 2 == 0:
        window += 1
    return savgol_filter(X, window, polyorder, deriv=2, axis=1).astype(np.float32)
