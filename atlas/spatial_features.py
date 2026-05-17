"""Per-file moment-statistic spatial / cross-pixel features.

Stage 15E of plan/15. Captures within-file pixel heterogeneity at named
regions and band centers. No spatial coordinates required (variance,
kurtosis, skewness are aggregate moments).

Plan/15 §7 R6 pre-flight check: the Atlas corpus has min=70, median=72,
max=180 pixels per file — no files clear the ≥200-pixel threshold needed
for Moran's I / GLCM on intensity maps. Those features are dropped from
Stage 15E; moment statistics work at any pixel count ≥ 50.

Public surface
--------------
- ``pixel_variance_per_region(X, wn, file_ids, regions=...) -> pd.DataFrame``
- ``pixel_cv_per_region(X, wn, file_ids, regions=...) -> pd.DataFrame``
- ``pixel_moment_at_band(X, wn, file_ids, band_centers, moment=...) -> pd.DataFrame``
- ``feature_frame_spatial(X, wn, spec_df) -> pd.DataFrame``  (one-shot)

Output convention: a DataFrame indexed by ``file_id`` with one row per file
and column names prefixed ``spat_``.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Per-region pixel-intensity statistics
# ---------------------------------------------------------------------------

DEFAULT_REGIONS: dict[str, tuple[float, float]] = {
    "lps_chain":   (800.0,  1200.0),   # Stage 1 anchor region
    "ch_stretch":  (2800.0, 3000.0),
}


def _integrate_region(X: np.ndarray, wn: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Trapezoidal AUC over [lo, hi] per row of X."""
    m = (wn >= lo) & (wn <= hi)
    if m.sum() < 2:
        return np.full(X.shape[0], np.nan, dtype=np.float64)
    return np.trapz(X[:, m], wn[m], axis=1)


def pixel_variance_per_region(
    X: np.ndarray,
    wn: np.ndarray,
    file_ids: np.ndarray | pd.Series,
    regions: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Within-file variance of per-pixel region-AUC values.

    For each region, integrate the per-pixel spectrum over the region to
    get a (N_pixels,) vector of region-AUCs, then compute the variance of
    that vector per file. Returns DataFrame indexed by file_id.
    """
    if regions is None:
        regions = DEFAULT_REGIONS
    file_ids = np.asarray(file_ids)
    cols: dict[str, np.ndarray] = {}
    fids: list[str] = list(pd.unique(file_ids))
    fid_to_rows = {fid: np.where(file_ids == fid)[0] for fid in fids}
    for region_key, (lo, hi) in regions.items():
        per_pixel_auc = _integrate_region(X, wn, lo, hi)
        out = np.empty(len(fids), dtype=np.float64)
        for i, fid in enumerate(fids):
            rows = fid_to_rows[fid]
            out[i] = float(np.var(per_pixel_auc[rows], ddof=1)) if len(rows) > 1 else np.nan
        cols[f"spat_var_{region_key}"] = out
    return pd.DataFrame(cols, index=pd.Index(fids, name="file_id"))


def pixel_cv_per_region(
    X: np.ndarray,
    wn: np.ndarray,
    file_ids: np.ndarray | pd.Series,
    regions: dict[str, tuple[float, float]] | None = None,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """Within-file coefficient of variation (std/|mean|) of per-pixel
    region-AUC values. Scale-invariant version of variance — preferred for
    LOSO transfer per plan/15 §4.1.

    Uses absolute value of mean in denominator since SNV'd AUCs can be
    negative; CV is meant as "spread relative to magnitude," so |mean| is
    the right normalizer.
    """
    if regions is None:
        regions = DEFAULT_REGIONS
    file_ids = np.asarray(file_ids)
    cols: dict[str, np.ndarray] = {}
    fids: list[str] = list(pd.unique(file_ids))
    fid_to_rows = {fid: np.where(file_ids == fid)[0] for fid in fids}
    for region_key, (lo, hi) in regions.items():
        per_pixel_auc = _integrate_region(X, wn, lo, hi)
        out = np.empty(len(fids), dtype=np.float64)
        for i, fid in enumerate(fids):
            rows = fid_to_rows[fid]
            if len(rows) < 2:
                out[i] = np.nan
                continue
            v = per_pixel_auc[rows]
            mu = float(np.mean(v))
            sd = float(np.std(v, ddof=1))
            out[i] = sd / (abs(mu) + eps)
        cols[f"spat_cv_{region_key}"] = out
    return pd.DataFrame(cols, index=pd.Index(fids, name="file_id"))


def pixel_moment_at_band(
    X: np.ndarray,
    wn: np.ndarray,
    file_ids: np.ndarray | pd.Series,
    band_centers: dict[str, float],
    moment: str = "kurt",
    half_width: float = 10.0,
) -> pd.DataFrame:
    """Higher-moment statistic of per-pixel intensity-AUC at named bands.

    For each `(name, center)` in `band_centers`, integrate ±`half_width`
    around `center` per pixel, then compute the requested moment of that
    (N_pixels,) vector per file.

    moment options:
        'kurt'  — Fisher's excess kurtosis (0 for normal), scipy.stats.kurtosis
        'skew'  — Pearson skewness
        'std'   — standard deviation (mostly for sanity vs variance)
    """
    if moment not in ("kurt", "skew", "std"):
        raise ValueError(f"moment must be kurt|skew|std; got {moment!r}")
    file_ids = np.asarray(file_ids)
    cols: dict[str, np.ndarray] = {}
    fids: list[str] = list(pd.unique(file_ids))
    fid_to_rows = {fid: np.where(file_ids == fid)[0] for fid in fids}
    for band_name, center in band_centers.items():
        if center < wn[0] or center > wn[-1]:
            cols[f"spat_{moment}_{band_name}"] = np.full(len(fids), np.nan)
            continue
        mask = (wn >= center - half_width) & (wn <= center + half_width)
        if mask.sum() < 2:
            cols[f"spat_{moment}_{band_name}"] = np.full(len(fids), np.nan)
            continue
        per_pixel = np.trapz(X[:, mask], wn[mask], axis=1)
        out = np.empty(len(fids), dtype=np.float64)
        for i, fid in enumerate(fids):
            rows = fid_to_rows[fid]
            if len(rows) < 4:
                out[i] = np.nan
                continue
            v = per_pixel[rows]
            if moment == "kurt":
                out[i] = float(stats.kurtosis(v, fisher=True, bias=False))
            elif moment == "skew":
                out[i] = float(stats.skew(v, bias=False))
            else:  # std
                out[i] = float(np.std(v, ddof=1))
        cols[f"spat_{moment}_{band_name}"] = out
    return pd.DataFrame(cols, index=pd.Index(fids, name="file_id"))


# Default LPS anchor bands for moment features (plan/15 DD18).
LPS_ANCHOR_BANDS: dict[str, float] = {
    "lps_1050": 1050.0,
    "lps_1117": 1117.0,
    "lps_1194": 1194.0,
}


def feature_frame_spatial(
    X: np.ndarray,
    wn: np.ndarray,
    spec_df: pd.DataFrame,
    *,
    regions: dict[str, tuple[float, float]] | None = None,
    moment_bands: dict[str, float] | None = None,
) -> pd.DataFrame:
    """One-shot per-file spatial feature DataFrame.

    Returns DataFrame indexed by file_id with columns:
        spat_var_lps_chain, spat_var_ch_stretch          (variance per region)
        spat_cv_lps_chain,  spat_cv_ch_stretch           (CV per region)
        spat_kurt_lps_{1050,1117,1194}                   (kurtosis per LPS band)
        spat_skew_lps_{1050,1117,1194}                   (skew per LPS band)
    """
    if regions is None:
        regions = DEFAULT_REGIONS
    if moment_bands is None:
        moment_bands = LPS_ANCHOR_BANDS
    if "file_id" not in spec_df.columns:
        raise ValueError("spec_df must have a 'file_id' column")
    file_ids = spec_df["file_id"].values

    var_df  = pixel_variance_per_region(X, wn, file_ids, regions=regions)
    cv_df   = pixel_cv_per_region(X, wn, file_ids, regions=regions)
    kurt_df = pixel_moment_at_band(X, wn, file_ids, moment_bands, moment="kurt")
    skew_df = pixel_moment_at_band(X, wn, file_ids, moment_bands, moment="skew")
    return pd.concat([var_df, cv_df, kurt_df, skew_df], axis=1)
