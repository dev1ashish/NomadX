"""Scale-invariant spectral features for the Atlas Raman dataset.

Stage 15B of plan/15. Pure functions over `(X: ndarray[N, B], wn: ndarray[B])`,
no dependency on the named-band catalog in `band_features.py`. These features
target LOSO generalization where Stage 5's fixed-AUC features failed.

Public surface:
  - dwt_features(X, wavelet='db4', max_level=6) -> dict[str, ndarray[N]]
  - DEFAULT_ROI_PCA                              (regions × n_components)
  - fit_roi_pca(X_train, wn, regions) -> dict (fitted PCAs)
  - transform_roi_pca(X, wn, fitted) -> dict[str, ndarray[N]]
  - roi_pca_features(X, wn, regions=DEFAULT_ROI_PCA) -> dict[str, ndarray[N]]
    (convenience: fits PCA on the same X it transforms — for caching only)
  - fit_sam_templates(X_train, y_train, sub_train, region_mask=None) -> dict
  - transform_sam(X, templates, region_mask=None) -> dict[str, ndarray[N]]
  - sam_features(X, wn, primary_class, subclass) -> dict[str, ndarray[N]]
    (convenience: fits templates on the same X)
  - feature_frame_spectral(X, wn, spec_df=None) -> pd.DataFrame
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# DWT energies + entropies
# ---------------------------------------------------------------------------

def dwt_features(
    X: np.ndarray,
    wavelet: str = "db4",
    max_level: int = 6,
) -> dict[str, np.ndarray]:
    """Discrete Wavelet Transform energy and entropy per detail level.

    For each spectrum, decomposes into approximation + `max_level` detail
    coefficients via `pywt.wavedec`. Returns 2*max_level features:
        dwt_energy_L<k> = sum(c_L<k>²) for k = 1..max_level
        dwt_entropy_L<k> = -sum(p · log p) where p = c_L<k>² / sum(c_L<k>²)

    Lower-index levels (L1) capture highest-frequency detail; higher-index
    levels (L6) capture coarser features (~30-80 cm⁻¹ width for db4).
    """
    import pywt
    N, B = X.shape
    out: dict[str, np.ndarray] = {}
    # Pre-allocate
    energies = np.zeros((N, max_level), dtype=np.float64)
    entropies = np.zeros((N, max_level), dtype=np.float64)
    eps = 1e-12
    for i in range(N):
        coeffs = pywt.wavedec(X[i], wavelet=wavelet, level=max_level, mode="symmetric")
        # coeffs = [cA_max, cD_max, cD_max-1, ..., cD_1]; details are coeffs[1:]
        # Index L1 (highest freq) = coeffs[-1]; L<max_level> (lowest freq) = coeffs[1]
        for L in range(1, max_level + 1):
            c = coeffs[-L] if L != 0 else coeffs[0]  # L=1 → coeffs[-1]
            # Energy
            energy = float(np.sum(c ** 2))
            energies[i, L - 1] = energy
            # Entropy
            if energy < eps:
                entropies[i, L - 1] = 0.0
                continue
            p = (c ** 2) / energy
            mask = p > eps
            entropies[i, L - 1] = float(-(p[mask] * np.log(p[mask])).sum())

    for L in range(1, max_level + 1):
        out[f"dwt_energy_L{L}"] = energies[:, L - 1]
        out[f"dwt_entropy_L{L}"] = entropies[:, L - 1]
    return out


# ---------------------------------------------------------------------------
# ROI-PCA scores
# ---------------------------------------------------------------------------

# Default regions + components per region.
DEFAULT_ROI_PCA: dict[str, tuple[tuple[float, float], int]] = {
    "lps":       ((800.0, 1200.0), 5),
    "amide":     ((1500.0, 1700.0), 3),
    "chstretch": ((2800.0, 3050.0), 3),
}


def fit_roi_pca(
    X_train: np.ndarray,
    wn: np.ndarray,
    regions: dict[str, tuple[tuple[float, float], int]] | None = None,
    random_state: int = 42,
) -> dict[str, dict]:
    """Fit one PCA per region. Returns dict region_key → {pca, mask, n_components}.

    The PCA is a sklearn PCA; mask is a boolean array indicating which wn bins
    belong to the region.
    """
    from sklearn.decomposition import PCA
    if regions is None:
        regions = DEFAULT_ROI_PCA
    fitted: dict[str, dict] = {}
    for key, ((lo, hi), n_comp) in regions.items():
        mask = (wn >= lo) & (wn <= hi)
        if mask.sum() < n_comp + 1:
            fitted[key] = {"pca": None, "mask": mask, "n_components": n_comp}
            continue
        pca = PCA(n_components=n_comp, random_state=random_state)
        pca.fit(X_train[:, mask])
        fitted[key] = {"pca": pca, "mask": mask, "n_components": n_comp}
    return fitted


def transform_roi_pca(
    X: np.ndarray,
    wn: np.ndarray,
    fitted: dict[str, dict],
) -> dict[str, np.ndarray]:
    """Apply previously-fit ROI PCAs. Returns dict feature_name → (N,) score."""
    out: dict[str, np.ndarray] = {}
    for key, info in fitted.items():
        if info["pca"] is None:
            for k in range(info["n_components"]):
                out[f"pca_{key}_PC{k+1}"] = np.full(X.shape[0], np.nan)
            continue
        scores = info["pca"].transform(X[:, info["mask"]])
        for k in range(scores.shape[1]):
            out[f"pca_{key}_PC{k+1}"] = scores[:, k]
    return out


def roi_pca_features(
    X: np.ndarray,
    wn: np.ndarray,
    regions: dict[str, tuple[tuple[float, float], int]] | None = None,
) -> dict[str, np.ndarray]:
    """Convenience: fit and transform PCA on the same X. **Use only for caching
    or fully within-fold; downstream LOSO classifier must use `fit_roi_pca` +
    `transform_roi_pca` separately to avoid leakage.**"""
    fitted = fit_roi_pca(X, wn, regions=regions)
    return transform_roi_pca(X, wn, fitted)


# ---------------------------------------------------------------------------
# SAM (Spectral Angle Mapper) features
# ---------------------------------------------------------------------------

LPS_REGION_FOR_SAM: tuple[float, float] = (800.0, 1200.0)


def _safe_sam(X: np.ndarray, templates: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Cosine-angle (in radians) between each row of X and each row of templates.

    Returns (N_X, N_templates) array of angles in [0, π].
    """
    # Normalize rows of X and templates
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    T_norm = np.linalg.norm(templates, axis=1, keepdims=True)
    Xn = X / (X_norm + eps)
    Tn = templates / (T_norm + eps)
    cos_sim = np.clip(Xn @ Tn.T, -1.0, 1.0)
    return np.arccos(cos_sim)


def fit_sam_templates(
    X_train: np.ndarray,
    y_primary: np.ndarray,
    y_subclass: np.ndarray,
    wn: np.ndarray | None = None,
    region: tuple[float, float] | None = None,
) -> dict:
    """Fit class-mean and subclass-mean templates from training data.

    If `region` is provided, also fits region-restricted templates.

    Returns dict with keys:
        class_means_full: (n_classes, B) mean per primary class
        class_labels: list of class names in order
        sub_means_full:  (n_subs, B) mean per subclass
        sub_labels: list of subclass names in order
        class_means_region / sub_means_region (if region provided)
        region_mask (if region provided)
    """
    out: dict = {}
    primary_classes = sorted(set(y_primary))
    sub_classes = sorted(set(y_subclass))

    cm_full = np.array(
        [X_train[y_primary == c].mean(0) for c in primary_classes],
        dtype=np.float64,
    )
    sm_full = np.array(
        [X_train[y_subclass == s].mean(0) for s in sub_classes],
        dtype=np.float64,
    )
    out["class_means_full"] = cm_full
    out["class_labels"] = primary_classes
    out["sub_means_full"] = sm_full
    out["sub_labels"] = sub_classes

    if region is not None and wn is not None:
        mask = (wn >= region[0]) & (wn <= region[1])
        out["region_mask"] = mask
        out["class_means_region"] = cm_full[:, mask]
        out["sub_means_region"] = sm_full[:, mask]
    return out


def transform_sam(
    X: np.ndarray,
    templates: dict,
    wn: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Apply fitted SAM templates to X. Returns dict feature_name → (N,) angle.

    Naming convention:
        sam_class_<label>       full-spectrum angle to primary-class mean
        sam_lps_class_<label>   LPS-region angle to primary-class mean
        sam_sub_<label>         full-spectrum angle to subclass mean
        sam_lps_sub_<label>     LPS-region angle to subclass mean
    """
    out: dict[str, np.ndarray] = {}
    # Full-spectrum
    angles_class_full = _safe_sam(X, templates["class_means_full"])
    for i, c in enumerate(templates["class_labels"]):
        out[f"sam_class_{c}"] = angles_class_full[:, i]
    angles_sub_full = _safe_sam(X, templates["sub_means_full"])
    for i, s in enumerate(templates["sub_labels"]):
        out[f"sam_sub_{s}"] = angles_sub_full[:, i]
    # Region-restricted
    if "region_mask" in templates and wn is not None:
        m = templates["region_mask"]
        X_region = X[:, m]
        angles_class_lps = _safe_sam(X_region, templates["class_means_region"])
        for i, c in enumerate(templates["class_labels"]):
            out[f"sam_lps_class_{c}"] = angles_class_lps[:, i]
        angles_sub_lps = _safe_sam(X_region, templates["sub_means_region"])
        for i, s in enumerate(templates["sub_labels"]):
            out[f"sam_lps_sub_{s}"] = angles_sub_lps[:, i]
    return out


def sam_features(
    X: np.ndarray,
    wn: np.ndarray,
    primary_class: np.ndarray,
    subclass: np.ndarray,
    region: tuple[float, float] | None = LPS_REGION_FOR_SAM,
) -> dict[str, np.ndarray]:
    """Convenience: fit SAM templates on X then transform. **Leaks labels into
    the templates by definition** — downstream LOSO classifier must use
    `fit_sam_templates(X_train, ...)` + `transform_sam(X_test, ...)` instead.
    """
    templates = fit_sam_templates(X, primary_class, subclass, wn=wn, region=region)
    return transform_sam(X, templates, wn=wn)


# ---------------------------------------------------------------------------
# One-shot feature frame
# ---------------------------------------------------------------------------

def feature_frame_spectral(
    X: np.ndarray,
    wn: np.ndarray,
    *,
    spec_df: pd.DataFrame | None = None,
    dwt: bool = True,
    pca: bool = True,
    sam: bool = True,
    wavelet: str = "db4",
    max_level: int = 6,
    sam_region: tuple[float, float] | None = LPS_REGION_FOR_SAM,
) -> pd.DataFrame:
    """One-shot DataFrame builder for Stage 15B features.

    If `spec_df` is None, the SAM features are skipped (no labels to build
    templates).
    """
    cols: dict[str, np.ndarray] = {}
    if dwt:
        for k, v in dwt_features(X, wavelet=wavelet, max_level=max_level).items():
            cols[k] = v
    if pca:
        for k, v in roi_pca_features(X, wn).items():
            cols[k] = v
    if sam and spec_df is not None:
        primary = spec_df["primary_class"].values
        sub = spec_df["subclass"].fillna("H2O").values
        for k, v in sam_features(X, wn, primary, sub, region=sam_region).items():
            cols[k] = v
    return pd.DataFrame(cols)
