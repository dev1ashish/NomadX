"""Chemistry-aware band features for Atlas preprocessed Raman spectra.

Functions are pure over a `(X: ndarray[N, B], wn: ndarray[B])` interface so
they compose with both raw-cache and preprocessed-cache loads.

The band catalog is the canonical reference from plan/14 §2.1, with the
**empirical anchor revised post-Stage-1** (plan/07 2026-05-17, plan/14 §2.4):
the literature primary triple 1338/1454/1658 did not replicate at file-level
on this dataset, while the 800–1200 cm⁻¹ LPS chain region carried the actual
within-bacterial discriminative signal. The catalog still includes the
literature bands as supporting features for the negative-finding writeup.

Public surface (per plan/14 §5.1):
  - BANDS                   : catalog of named bands
  - MACROMOLECULE_GROUPS    : 5 groups mapping band names → group key
  - PRIMARY_TRIPLE          : literature STEC↔non-STEC bands (now: supporting)
  - EMPIRICAL_ANCHOR_BANDS  : Atlas-derived top discriminators (now: headline)
  - integrate_band(X, wn, center, half_width=10) -> ndarray[N]
  - integrate_region(X, wn, lo, hi) -> ndarray[N]
  - macromolecule_vector(X, wn) -> dict[str, ndarray[N]]
  - band_ratios(X, wn, pairs=None) -> dict[str, ndarray[N]]
  - fit_peak(X_row, wn, center, window=30, model="lorentz") -> PeakFit
  - fit_peaks_batch(X, wn, centers) -> ndarray[N, P, 4]
  - feature_frame(X, wn, ratios=True, fits=True) -> pd.DataFrame
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Canonical catalog
# ---------------------------------------------------------------------------

# Each entry: (key, center_cm-1, macromolecule_group, source, primary_triple_flag)
# Group keys defined in MACROMOLECULE_GROUPS below.
BANDS: dict[str, dict] = {
    # ---- LPS / O-antigen region (empirical anchors, Stage 1 winners) ----
    "lps_1050":       {"center": 1050.0, "group": "lipid_carb",        "source": "atlas_eda",  "primary": False, "empirical": True,  "note": "top E. coli vs Salmonella discriminator (3-class ANOVA)"},
    "lps_1117":       {"center": 1117.0, "group": "lipid_carb",        "source": "atlas_eda",  "primary": False, "empirical": True,  "note": "top E. coli STEC↔Non-STEC discriminator"},
    "lps_1194":       {"center": 1194.0, "group": "lipid_carb",        "source": "atlas_eda",  "primary": False, "empirical": True,  "note": "top E. coli STEC↔Non-STEC discriminator"},
    # ---- Aromatic amino acids (protein side chains) ----
    "aa_762":         {"center": 762.0,  "group": "aromatic_aa",       "source": "briefing",   "primary": False, "empirical": False, "note": "Trp ring"},
    "aa_831":         {"center": 831.0,  "group": "aromatic_aa",       "source": "briefing",   "primary": False, "empirical": False, "note": "Tyr Fermi"},
    "aa_855":         {"center": 855.0,  "group": "aromatic_aa",       "source": "briefing",   "primary": False, "empirical": False, "note": "Tyr"},
    "aa_1004":        {"center": 1004.0, "group": "aromatic_aa",       "source": "briefing",   "primary": False, "empirical": False, "note": "Phe ring breathing — total-protein anchor"},
    "aa_1014":        {"center": 1014.0, "group": "aromatic_aa",       "source": "briefing",   "primary": False, "empirical": False, "note": "Phe / Trp"},
    "aa_1176":        {"center": 1176.0, "group": "aromatic_aa",       "source": "briefing",   "primary": False, "empirical": False, "note": "Tyr / Phe"},
    "aa_1212":        {"center": 1212.0, "group": "aromatic_aa",       "source": "briefing",   "primary": False, "empirical": False, "note": "Tyr / Phe C-C"},
    "aa_1617":        {"center": 1617.0, "group": "aromatic_aa",       "source": "briefing",   "primary": False, "empirical": False, "note": "Trp"},
    # ---- Nucleic acids ----
    "na_720":         {"center": 720.0,  "group": "nucleic_acid",      "source": "briefing",   "primary": False, "empirical": False, "note": "A/G/T ring breathing"},
    "na_786":         {"center": 786.0,  "group": "nucleic_acid",      "source": "briefing",   "primary": False, "empirical": False, "note": "DNA/RNA U,C,T ring (dual — also AA)"},
    "na_1335":        {"center": 1335.0, "group": "nucleic_acid",      "source": "briefing",   "primary": False, "empirical": False, "note": "CH₂/CH₃ wag"},
    "na_1338":        {"center": 1338.0, "group": "nucleic_acid",      "source": "cisek_2013", "primary": True,  "empirical": False, "note": "PRIMARY (literature) — falsified Stage 1"},
    "na_1362":        {"center": 1362.0, "group": "nucleic_acid",      "source": "briefing",   "primary": False, "empirical": False, "note": "G ring"},
    "na_1485":        {"center": 1485.0, "group": "nucleic_acid",      "source": "briefing",   "primary": False, "empirical": False, "note": "G ring"},
    "na_1530":        {"center": 1530.0, "group": "nucleic_acid",      "source": "briefing",   "primary": False, "empirical": False, "note": "NA bases"},
    "na_1575":        {"center": 1575.0, "group": "nucleic_acid",      "source": "briefing",   "primary": False, "empirical": False, "note": "purine bases"},
    # ---- Protein amide ----
    "amide_iii_1242": {"center": 1242.0, "group": "protein_amide",     "source": "briefing",   "primary": False, "empirical": False, "note": "amide III β-sheet"},
    "amide_i_1658":   {"center": 1658.0, "group": "protein_amide",     "source": "cisek_2013", "primary": True,  "empirical": False, "note": "PRIMARY (literature) amide I β/random — falsified Stage 1"},
    "amide_i_1662":   {"center": 1662.0, "group": "protein_amide",     "source": "briefing",   "primary": False, "empirical": False, "note": "amide I α-helix"},
    # ---- Lipid / carbohydrate ----
    "lipid_1080":     {"center": 1080.0, "group": "lipid_carb",        "source": "briefing",   "primary": False, "empirical": False, "note": "phospholipid backbone"},
    "lipid_1451":     {"center": 1451.0, "group": "lipid_carb",        "source": "briefing",   "primary": False, "empirical": False, "note": "CH₂ deformation"},
    "lipid_1454":     {"center": 1454.0, "group": "lipid_carb",        "source": "cisek_2013", "primary": True,  "empirical": False, "note": "PRIMARY (literature) lipid/carb — falsified Stage 1 (d=-0.47 sign reversed)"},
    "lipid_1585":     {"center": 1585.0, "group": "lipid_carb",        "source": "briefing",   "primary": False, "empirical": False, "note": "lipid/carb/NA overlap"},
    "lipid_2850":     {"center": 2850.0, "group": "lipid_carb",        "source": "atlas_eda",  "primary": False, "empirical": False, "note": "sym CH₂ stretch (lipid)"},
    "lipid_2930":     {"center": 2930.0, "group": "lipid_carb",        "source": "atlas_eda",  "primary": False, "empirical": False, "note": "asym CH₂ stretch"},
    # ---- Salmonella-specific (Yuan 2024) ----
    "salm_616":       {"center": 616.0,  "group": "metabolite",        "source": "yuan_2024",  "primary": False, "empirical": False, "note": "COO⁻ wag"},
    "salm_925":       {"center": 925.0,  "group": "metabolite",        "source": "yuan_2024",  "primary": False, "empirical": False, "note": "C-C skeletal"},
    "salm_1486":      {"center": 1486.0, "group": "nucleic_acid",      "source": "yuan_2024",  "primary": False, "empirical": False, "note": "G ring"},
    "salm_1542":      {"center": 1542.0, "group": "metabolite",        "source": "yuan_2024",  "primary": False, "empirical": False, "note": "C=C"},
}

# 5 macromolecule groups, plus continuous LPS region. Plan/14 §2.2.
MACROMOLECULE_GROUPS: dict[str, list[str]] = {
    "aromatic_aa": [k for k, v in BANDS.items()
                    if v["group"] == "aromatic_aa"],
    "protein_amide": [k for k, v in BANDS.items()
                      if v["group"] == "protein_amide"],
    "nucleic_acid": [k for k, v in BANDS.items()
                     if v["group"] == "nucleic_acid"],
    "lipid_carbohydrate": [k for k, v in BANDS.items()
                            if v["group"] == "lipid_carb"],
    "metabolite": [k for k, v in BANDS.items()
                    if v["group"] == "metabolite"],
}

# LPS region uses continuous integration, not individual ±10 windows.
# 400–900 = general LPS detection (briefing).
# 800–1200 = E. coli vs Salmonella discriminator (user emphasis + Stage 1).
LPS_REGIONS: dict[str, tuple[float, float]] = {
    "lps_o_antigen_full":  (400.0,  900.0),
    "lps_chain_discrim":   (800.0, 1200.0),
}

# Literature primary triple (now demoted to supporting features).
PRIMARY_TRIPLE: tuple[str, str, str] = ("na_1338", "lipid_1454", "amide_i_1658")

# Empirical anchor bands (Stage 1 winners, the actual headline features).
EMPIRICAL_ANCHOR_BANDS: tuple[str, str, str] = ("lps_1050", "lps_1117", "lps_1194")


# ---------------------------------------------------------------------------
# AUC / integration
# ---------------------------------------------------------------------------

def integrate_band(
    X: np.ndarray,
    wn: np.ndarray,
    center: float,
    half_width: float = 10.0,
) -> np.ndarray:
    """Trapezoidal AUC over [center - half_width, center + half_width]."""
    m = (wn >= center - half_width) & (wn <= center + half_width)
    if m.sum() < 2:
        return np.full(X.shape[0], np.nan, dtype=np.float64)
    return np.trapz(X[:, m], wn[m], axis=1)


def integrate_region(
    X: np.ndarray,
    wn: np.ndarray,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Trapezoidal AUC over [lo, hi]."""
    m = (wn >= lo) & (wn <= hi)
    if m.sum() < 2:
        return np.full(X.shape[0], np.nan, dtype=np.float64)
    return np.trapz(X[:, m], wn[m], axis=1)


def macromolecule_vector(
    X: np.ndarray,
    wn: np.ndarray,
    half_width: float = 10.0,
) -> dict[str, np.ndarray]:
    """Per-spectrum AUC sum per macromolecule group + the two LPS regions.

    Returns a dict keyed by group name. Each value is shape (N,).
    Bands outside the wn range contribute 0 (skipped silently).
    """
    out: dict[str, np.ndarray] = {}
    for group, band_keys in MACROMOLECULE_GROUPS.items():
        total = np.zeros(X.shape[0], dtype=np.float64)
        n_used = 0
        for k in band_keys:
            center = BANDS[k]["center"]
            if center < wn[0] or center > wn[-1]:
                continue
            total += integrate_band(X, wn, center, half_width=half_width)
            n_used += 1
        if n_used == 0:
            total[:] = np.nan
        out[group] = total
    # LPS regions: continuous integration.
    for key, (lo, hi) in LPS_REGIONS.items():
        if lo < wn[0]:
            lo = float(wn[0])
        if hi > wn[-1]:
            hi = float(wn[-1])
        out[key] = integrate_region(X, wn, lo, hi)
    return out


# ---------------------------------------------------------------------------
# Band ratios
# ---------------------------------------------------------------------------

# Default ratio set — chosen to cancel multiplicative file-level offsets.
# Format: dict[ratio_name] = (band_or_group_key_numerator, denominator).
# A key starting with "group:" or "lps:" refers to macromolecule_vector output.
DEFAULT_RATIOS: dict[str, tuple[str, str]] = {
    # Within-protein composition
    "amide_over_na":       ("group:protein_amide", "group:nucleic_acid"),
    "aa_over_amide":       ("group:aromatic_aa",   "group:protein_amide"),
    # Energy partitioning
    "lipid_over_protein":  ("group:lipid_carbohydrate", "group:protein_amide"),
    "na_over_lipid":       ("group:nucleic_acid", "group:lipid_carbohydrate"),
    # E. coli vs Salmonella anchor (LPS chain emphasis)
    "lps_chain_over_protein": ("lps:lps_chain_discrim", "group:protein_amide"),
    # Within-LPS-region empirical anchors (Stage 1 winners)
    "lps_1117_over_1050":  ("band:lps_1117", "band:lps_1050"),
    "lps_1194_over_1050":  ("band:lps_1194", "band:lps_1050"),
    # Literature ratios (kept for supporting comparisons)
    "amide_1658_over_na_1338": ("band:amide_i_1658", "band:na_1338"),  # Cisek-style
    "phe_1004_over_amide_1658": ("band:aa_1004",   "band:amide_i_1658"),
    "na_786_over_phe_1004":     ("band:na_786",    "band:aa_1004"),
}


def band_ratios(
    X: np.ndarray,
    wn: np.ndarray,
    pairs: dict[str, tuple[str, str]] | None = None,
    eps: float = 1e-9,
    half_width: float = 10.0,
) -> dict[str, np.ndarray]:
    """Compute band-ratio features. See DEFAULT_RATIOS for naming convention.

    Keys with prefix "band:" → integrate_band on the named band center.
    Keys with prefix "group:" → macromolecule_vector group output.
    Keys with prefix "lps:" → macromolecule_vector LPS-region output.
    """
    pairs = pairs or DEFAULT_RATIOS
    macro = macromolecule_vector(X, wn, half_width=half_width)

    def resolve(spec: str) -> np.ndarray:
        kind, _, name = spec.partition(":")
        if kind == "band":
            return integrate_band(X, wn, BANDS[name]["center"], half_width=half_width)
        elif kind == "group":
            return macro[name]
        elif kind == "lps":
            return macro[name]
        else:
            raise ValueError(f"unknown ratio spec prefix in {spec!r}")

    out: dict[str, np.ndarray] = {}
    for ratio_name, (num_spec, den_spec) in pairs.items():
        num = resolve(num_spec)
        den = resolve(den_spec)
        # Use sign-preserving safe division: shift denominator off zero by eps
        # in its own sign direction.
        safe_den = np.where(np.abs(den) < eps, np.sign(den) * eps + eps, den)
        out[ratio_name] = num / safe_den
    return out


# ---------------------------------------------------------------------------
# Lorentzian peak fitting
# ---------------------------------------------------------------------------

@dataclass
class PeakFit:
    """Result of a Lorentzian peak fit.

    NaN means fit failed (caller should fall back to integrate_band).
    """
    center: float
    height: float
    fwhm: float
    area: float
    baseline: float
    rmse: float
    success: bool


def _lorentzian(x: np.ndarray, a: float, x0: float, gamma: float, c: float) -> np.ndarray:
    """Lorentzian + flat baseline: a * gamma² / ((x - x0)² + gamma²) + c.
    height = a, center = x0, FWHM = 2*gamma, area = pi * a * gamma.
    """
    return a * (gamma ** 2) / ((x - x0) ** 2 + gamma ** 2) + c


def _pseudovoigt_linbase(
    x: np.ndarray,
    a: float, x0: float, sigma: float, gamma: float, eta: float,
    b: float, c: float,
) -> np.ndarray:
    """Pseudo-Voigt with linear baseline (Stage 15A — plan/15 §5).

    pv(x) = η · L(x; x0, γ) + (1-η) · G(x; x0, σ)
    y     = a · pv(x) + b · (x - x0) + c

    Where L and G are unit-height Lorentzian and Gaussian profiles centered at x0.
    """
    lor = (gamma ** 2) / ((x - x0) ** 2 + gamma ** 2)
    gau = np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2))
    return a * (eta * lor + (1.0 - eta) * gau) + b * (x - x0) + c


def fit_peak(
    spec: np.ndarray,
    wn: np.ndarray,
    center: float,
    window: float = 30.0,
    initial_fwhm: float = 12.0,
) -> PeakFit:
    """Fit a Lorentzian within ±window cm⁻¹ around `center`."""
    nan_fit = PeakFit(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, success=False)
    m = (wn >= center - window) & (wn <= center + window)
    if m.sum() < 5:
        return nan_fit
    x = wn[m].astype(np.float64)
    y = spec[m].astype(np.float64)
    if not np.all(np.isfinite(y)):
        return nan_fit

    # Initial guesses
    baseline_init = float(np.percentile(y, 5))
    height_init = float(y.max() - baseline_init)
    if height_init <= 0:
        # Inverted peak (post-SNV some peaks point down) — flip and fit.
        baseline_init = float(np.percentile(y, 95))
        height_init = float(y.min() - baseline_init)
    x0_init = float(center)
    gamma_init = float(initial_fwhm / 2.0)

    try:
        popt, _pcov = curve_fit(
            _lorentzian, x, y,
            p0=[height_init, x0_init, gamma_init, baseline_init],
            bounds=([-np.inf, center - window, 0.5, -np.inf],
                    [ np.inf, center + window, window, np.inf]),
            maxfev=2000,
        )
    except (RuntimeError, ValueError):
        return nan_fit

    a, x0, gamma, c = popt
    fwhm = 2.0 * gamma
    area = float(np.pi * a * gamma)
    residual = y - _lorentzian(x, *popt)
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    return PeakFit(
        center=float(x0),
        height=float(a),
        fwhm=float(fwhm),
        area=area,
        baseline=float(c),
        rmse=rmse,
        success=True,
    )


def fit_peaks_batch(
    X: np.ndarray,
    wn: np.ndarray,
    band_keys: Iterable[str],
    window: float = 30.0,
) -> dict[str, np.ndarray]:
    """Fit each named band on every spectrum (legacy Lorentzian).

    Returns a dict where each key maps to (N, 4) array of
    (center, height, fwhm, area). NaN where the fit failed.

    For Stage 15A and later, prefer `fit_peaks_batch_pseudovoigt` which uses
    pseudo-Voigt + linear baseline and yields much higher success rates.
    """
    band_keys = list(band_keys)
    out = {k: np.full((X.shape[0], 4), np.nan, dtype=np.float64) for k in band_keys}
    for k in band_keys:
        center = BANDS[k]["center"]
        if center < wn[0] or center > wn[-1]:
            continue
        for i in range(X.shape[0]):
            fit = fit_peak(X[i], wn, center, window=window)
            if fit.success:
                out[k][i] = [fit.center, fit.height, fit.fwhm, fit.area]
    return out


# ---------------------------------------------------------------------------
# Pseudo-Voigt peak fitting (Stage 15A — plan/15)
# ---------------------------------------------------------------------------

@dataclass
class PseudoVoigtFit:
    """Pseudo-Voigt + linear baseline fit result. NaN fields = fit failed."""
    center: float       # x0
    height: float       # a
    fwhm: float         # convex-combination approximation: η·2γ + (1-η)·2σ·√(2 ln 2)
    sigma: float        # Gaussian half-width
    gamma: float        # Lorentzian half-width
    eta: float          # mixing fraction (0=Gaussian, 1=Lorentzian)
    baseline_slope: float       # b — linear coefficient on (x - x0)
    baseline_intercept: float   # c
    area: float         # ∫ a·pv(x) dx ≈ a·π·γ·η + a·σ·√(2π)·(1-η)
    rmse: float
    success: bool


_FWHM_G_CONST = 2.0 * np.sqrt(2.0 * np.log(2.0))   # ≈ 2.3548


def fit_peak_pseudovoigt(
    spec: np.ndarray,
    wn: np.ndarray,
    center: float,
    window: float = 30.0,
    initial_fwhm: float = 12.0,
) -> PseudoVoigtFit:
    """Fit pseudo-Voigt + linear baseline within ±window cm⁻¹ around `center`.

    Returns a PseudoVoigtFit; if any step fails, all numeric fields are NaN and
    `success` is False.
    """
    nan_fit = PseudoVoigtFit(
        center=np.nan, height=np.nan, fwhm=np.nan,
        sigma=np.nan, gamma=np.nan, eta=np.nan,
        baseline_slope=np.nan, baseline_intercept=np.nan,
        area=np.nan, rmse=np.nan, success=False,
    )
    m = (wn >= center - window) & (wn <= center + window)
    if m.sum() < 7:
        return nan_fit
    x = wn[m].astype(np.float64)
    y = spec[m].astype(np.float64)
    if not np.all(np.isfinite(y)):
        return nan_fit

    # Robust amplitude/baseline guesses against either-sign peaks.
    base_lo = float(np.percentile(y, 10))
    base_hi = float(np.percentile(y, 90))
    if (y.max() - base_lo) >= (base_hi - y.min()):
        a_init = float(y.max() - base_lo)
        c_init = base_lo
    else:
        a_init = float(y.min() - base_hi)
        c_init = base_hi
    gamma_init = max(2.0, float(initial_fwhm / 2.0))
    sigma_init = max(2.0, float(initial_fwhm / _FWHM_G_CONST))
    eta_init = 0.5
    b_init = 0.0

    try:
        popt, _pcov = curve_fit(
            _pseudovoigt_linbase, x, y,
            p0=[a_init, float(center), sigma_init, gamma_init,
                eta_init, b_init, c_init],
            bounds=(
                [-np.inf, center - window, 1.5, 1.5, 0.0, -np.inf, -np.inf],
                [ np.inf, center + window,  30.0,  30.0, 1.0,  np.inf,  np.inf],
            ),
            maxfev=4000,
        )
    except (RuntimeError, ValueError):
        return nan_fit

    a, x0, sigma, gamma, eta, b, c = popt
    fwhm_l = 2.0 * gamma
    fwhm_g = _FWHM_G_CONST * sigma
    fwhm = eta * fwhm_l + (1.0 - eta) * fwhm_g  # convex-combination approximation
    area = float(np.pi * gamma * a * eta + sigma * np.sqrt(2.0 * np.pi) * a * (1.0 - eta))
    residual = y - _pseudovoigt_linbase(x, *popt)
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    return PseudoVoigtFit(
        center=float(x0),
        height=float(a),
        fwhm=float(fwhm),
        sigma=float(sigma),
        gamma=float(gamma),
        eta=float(eta),
        baseline_slope=float(b),
        baseline_intercept=float(c),
        area=area,
        rmse=rmse,
        success=True,
    )


def fit_peaks_batch_pseudovoigt(
    X: np.ndarray,
    wn: np.ndarray,
    band_keys: Iterable[str],
    window: float = 30.0,
) -> dict[str, np.ndarray]:
    """Pseudo-Voigt batch fit. Returns dict band_key -> (N, 6) array of
    (center, height, fwhm, eta, area, rmse). NaN where fit failed.

    eta is included so downstream features can use "how Lorentzian vs Gaussian"
    as a discriminator. Sigma/gamma individually omitted to keep the column
    count manageable; recoverable from FWHM and η if ever needed.
    """
    band_keys = list(band_keys)
    out = {k: np.full((X.shape[0], 6), np.nan, dtype=np.float64) for k in band_keys}
    for k in band_keys:
        center = BANDS[k]["center"]
        if center < wn[0] or center > wn[-1]:
            continue
        for i in range(X.shape[0]):
            fit = fit_peak_pseudovoigt(X[i], wn, center, window=window)
            if fit.success:
                out[k][i] = [fit.center, fit.height, fit.fwhm,
                             fit.eta, fit.area, fit.rmse]
    return out


# ---------------------------------------------------------------------------
# ROI moments (Stage 15A — SP10-SP16)
# ---------------------------------------------------------------------------

# 6 named regions covering the preprocessed wavenumber axis.
DEFAULT_ROIS: dict[str, tuple[float, float]] = {
    "fingerprint_low":         (400.0,  800.0),
    "lps_chain":               (800.0, 1200.0),    # Stage 1/2 empirical anchor
    "protein_amide_extended": (1200.0, 1500.0),
    "amide_aromatic":         (1500.0, 1700.0),
    "silent":                 (1700.0, 2800.0),    # quality-check region
    "ch_stretch":             (2800.0, 3050.0),
}


def roi_moments(
    X: np.ndarray,
    wn: np.ndarray,
    regions: dict[str, tuple[float, float]] | None = None,
) -> dict[str, np.ndarray]:
    """6-statistic moments per ROI per spectrum.

    For each region, returns columns:
        roi_<name>_mean       — mean intensity
        roi_<name>_std        — std intensity
        roi_<name>_skew       — Fisher-Pearson skew (NaN if std=0)
        roi_<name>_kurt       — excess kurtosis (NaN if std=0)
        roi_<name>_centroid   — Σ(wn · y) / Σ(y); spectral center of mass
        roi_<name>_entropy    — Shannon entropy of normalized |y|

    `centroid` uses Σ(y) directly (signed) — meaningful when most y > 0;
    for SNV-centered data the centroid is interpreted relative to the region.
    """
    if regions is None:
        regions = DEFAULT_ROIS
    from scipy.stats import skew, kurtosis, entropy
    N = X.shape[0]
    out: dict[str, np.ndarray] = {}
    for name, (lo, hi) in regions.items():
        m = (wn >= lo) & (wn <= hi)
        if m.sum() < 3:
            for stat in ("mean", "std", "skew", "kurt", "centroid", "entropy"):
                out[f"roi_{name}_{stat}"] = np.full(N, np.nan, dtype=np.float64)
            continue
        sub = X[:, m]
        sub_wn = wn[m]
        out[f"roi_{name}_mean"]  = sub.mean(axis=1)
        out[f"roi_{name}_std"]   = sub.std(axis=1)
        # skew/kurt are NaN where std=0
        sk = skew(sub, axis=1, bias=False, nan_policy="omit")
        kt = kurtosis(sub, axis=1, fisher=True, bias=False, nan_policy="omit")
        out[f"roi_{name}_skew"] = np.asarray(sk, dtype=np.float64)
        out[f"roi_{name}_kurt"] = np.asarray(kt, dtype=np.float64)
        # Centroid via signed weights — fragile when Σ y ≈ 0; use eps
        sum_y = sub.sum(axis=1)
        eps = 1e-9
        safe_sum = np.where(np.abs(sum_y) < eps,
                            np.sign(sum_y) * eps + eps, sum_y)
        out[f"roi_{name}_centroid"] = (sub * sub_wn[None, :]).sum(axis=1) / safe_sum
        # Shannon entropy of normalized |y| (always non-negative input).
        abs_sub = np.abs(sub)
        norm = abs_sub / (abs_sub.sum(axis=1, keepdims=True) + eps)
        out[f"roi_{name}_entropy"] = -np.sum(
            norm * np.log(norm + eps), axis=1
        )
    return out


# ---------------------------------------------------------------------------
# EMSC scatter correction (Stage 15A — SP25)
# ---------------------------------------------------------------------------

def emsc_correct(
    X: np.ndarray,
    wn: np.ndarray,
    reference: np.ndarray | None = None,
    poly_degree: int = 2,
    return_corrected: bool = True,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Extended Multiplicative Scatter Correction.

    Models each spectrum as:
        y = a + b · ref + sum_{k=1..poly_degree} c_k · wn^k + chem_residual

    Returns:
        corrected — (N, B) — chem_residual (if return_corrected=True), else None
        coefs     — (N, 2 + poly_degree) — columns [a, b, c1, ..., c_poly]

    The 4 coefficients (with poly_degree=2) are scatter-correction features that
    capture per-spectrum baseline offset, reference scaling, linear and
    quadratic wavenumber-dependent baselines.
    """
    N, B = X.shape
    if reference is None:
        reference = X.mean(axis=0)
    # Build the design matrix: columns [1, ref, wn, wn^2, ...]
    cols = [np.ones(B), reference]
    wn_norm = (wn - wn.mean()) / (wn.std() + 1e-9)  # numerical stability
    for k in range(1, poly_degree + 1):
        cols.append(wn_norm ** k)
    A = np.stack(cols, axis=1)            # (B, 2 + poly_degree)
    # Least-squares solve per spectrum: y = A @ θ + residual
    # Solve all spectra at once via pinv.
    AtA_inv_At = np.linalg.pinv(A)        # (2+poly_degree, B)
    coefs = (AtA_inv_At @ X.T).T          # (N, 2+poly_degree)
    if return_corrected:
        # chem_residual = y - A @ θ
        residual = X - coefs @ A.T
        return residual, coefs
    return None, coefs


# ---------------------------------------------------------------------------
# Derivative features (Stage 15A — SP1, SP2)
# ---------------------------------------------------------------------------

def derivative_band_auc(
    X: np.ndarray,
    wn: np.ndarray,
    deriv: int,
    band_keys: Iterable[str],
    half_width: float = 10.0,
    sg_window: int = 11,
    sg_poly: int = 3,
) -> dict[str, np.ndarray]:
    """Trapezoidal AUC of |y'| (deriv=1) or |y''| (deriv=2) over each named
    band's ±half_width window. Uses Savitzky-Golay smoothed derivatives.

    Returns dict band_key -> (N,) AUC values.
    """
    from scipy.signal import savgol_filter
    assert deriv in (1, 2), "deriv must be 1 or 2"
    if sg_window % 2 == 0:
        sg_window += 1
    # wn spacing in cm-1 per bin (uniform-grid assumption — true for our cache)
    dx = float(np.median(np.diff(wn)))
    Y_deriv = savgol_filter(
        X, window_length=sg_window, polyorder=sg_poly,
        deriv=deriv, delta=dx, axis=1, mode="interp",
    )
    abs_deriv = np.abs(Y_deriv)
    out: dict[str, np.ndarray] = {}
    for k in band_keys:
        center = BANDS[k]["center"]
        if center < wn[0] or center > wn[-1]:
            continue
        m = (wn >= center - half_width) & (wn <= center + half_width)
        if m.sum() < 2:
            out[k] = np.full(X.shape[0], np.nan, dtype=np.float64)
            continue
        out[k] = np.trapz(abs_deriv[:, m], wn[m], axis=1)
    return out


# ---------------------------------------------------------------------------
# Stage 15D — biology-specific features
# ---------------------------------------------------------------------------
#
# Five families (~15 features) covering the BIO1-BIO29 catalog in plan/15 §3.2:
#   - cytochrome_features          (BIO1-BIO4)
#   - protein_secondary_structure  (BIO20-BIO22)
#   - phb_features                 (BIO24-BIO25 — K-12 falsifier)
#   - aromatic_aa_features         (BIO26-BIO29)
#   - nucleic_conformation_features (BIO18-BIO19)
#
# All AUCs use integrate_band(X, wn, center, half_width=10).
# Ratios are computed in-function so missing-band cases return NaN cleanly.

def _safe_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """num / den with NaN where |den| < eps. Inputs broadcast row-wise."""
    out = np.full_like(num, np.nan, dtype=np.float64)
    ok = np.abs(den) > eps
    out[ok] = num[ok] / den[ok]
    return out


def cytochrome_features(
    X: np.ndarray,
    wn: np.ndarray,
    half_width: float = 10.0,
) -> dict[str, np.ndarray]:
    """BIO1-BIO4: cytochrome bands at 752 / 1127 / 1356 / 1372 / 1585.

    Note: 785 nm excitation is off-resonance for heme cytochromes (R5 in
    plan/15 §7) — these features may carry weak signal. Included for
    completeness and to test the cytochrome hypothesis directly.
    """
    auc_752  = integrate_band(X, wn, 752.0,  half_width)
    auc_1004 = integrate_band(X, wn, 1004.0, half_width)
    auc_1127 = integrate_band(X, wn, 1127.0, half_width)
    auc_1356 = integrate_band(X, wn, 1356.0, half_width)
    auc_1372 = integrate_band(X, wn, 1372.0, half_width)
    auc_1585 = integrate_band(X, wn, 1585.0, half_width)
    return {
        "bio_cyt_pyrrole_ratio": _safe_ratio(auc_752, auc_1004),   # BIO1
        "bio_cyt_ox_state":      _safe_ratio(auc_1356, auc_1372),  # BIO2
        # BIO3 cyt_center_1585 — see protein_secondary_structure: handled
        # via the existing fit_amide_i_1662 / pseudo-Voigt path if 1585 is
        # added to fit_bands. Skipped here to keep this function fit-free.
        "bio_cyt_total":         auc_752 + auc_1127 + auc_1585,    # BIO4
    }


def protein_secondary_structure(
    X: np.ndarray,
    wn: np.ndarray,
    half_width: float = 10.0,
) -> dict[str, np.ndarray]:
    """BIO20-BIO21: α-helix and β-sheet ratios. BIO22 (amide-I FWHM) is
    already in the cache via fit_amide_i_1662_fwhm (Stage 15A pseudo-Voigt)
    — duplicated here at 1655 for completeness when fits are off.
    """
    auc_1652 = integrate_band(X, wn, 1652.0, half_width)
    auc_1670 = integrate_band(X, wn, 1670.0, half_width)
    auc_1232 = integrate_band(X, wn, 1232.0, half_width)
    auc_1270 = integrate_band(X, wn, 1270.0, half_width)
    return {
        "bio_alpha_helix_score":  _safe_ratio(auc_1652, auc_1670),   # BIO20
        "bio_beta_sheet_amide3":  _safe_ratio(auc_1232, auc_1270),   # BIO21
    }


def phb_features(
    X: np.ndarray,
    wn: np.ndarray,
    half_width: float = 10.0,
) -> dict[str, np.ndarray]:
    """BIO24-BIO25: polyhydroxybutyrate (PHB) accumulation markers.

    K-12 falsifier — laboratory-domesticated K-12 is hypothesized to
    accumulate PHB anomalously vs. clinical STEC strains. If
    `bio_phb_carbonyl` shows file-level |d| ≥ 0.5 K-12-vs-other-STEC,
    this is the FIRST K-12-specific feature in the project.
    """
    auc_1730 = integrate_band(X, wn, 1730.0, half_width)   # BIO24
    auc_1058 = integrate_band(X, wn, 1058.0, half_width)
    auc_1450 = integrate_band(X, wn, 1450.0, half_width)
    # BIO25: joint occurrence × normalization
    phb_score = _safe_ratio(auc_1730 * auc_1058, auc_1450 ** 2)
    return {
        "bio_phb_carbonyl": auc_1730,
        "bio_phb_score":    phb_score,
    }


def aromatic_aa_features(
    X: np.ndarray,
    wn: np.ndarray,
    half_width: float = 10.0,
) -> dict[str, np.ndarray]:
    """BIO26-BIO29: Tyr doublet, Trp content, Trp env, Trp/Phe (virulence_aa_sig)."""
    auc_850  = integrate_band(X, wn, 850.0,  half_width)
    auc_830  = integrate_band(X, wn, 830.0,  half_width)
    auc_759  = integrate_band(X, wn, 759.0,  half_width)
    auc_1552 = integrate_band(X, wn, 1552.0, half_width)
    auc_1340 = integrate_band(X, wn, 1340.0, half_width)
    auc_1360 = integrate_band(X, wn, 1360.0, half_width)
    auc_1004 = integrate_band(X, wn, 1004.0, half_width)
    trp_total = auc_759 + auc_1552
    return {
        "bio_tyr_doublet_ratio": _safe_ratio(auc_850, auc_830),       # BIO26
        "bio_trp_content":       trp_total,                            # BIO27
        "bio_trp_indole_env":    _safe_ratio(auc_1340, auc_1360),     # BIO28
        "bio_virulence_aa_sig":  _safe_ratio(trp_total, auc_1004),    # BIO29
    }


def nucleic_conformation_features(
    X: np.ndarray,
    wn: np.ndarray,
    half_width: float = 10.0,
) -> dict[str, np.ndarray]:
    """BIO18-BIO19: A-form RNA fraction + RNA/DNA ratio."""
    auc_815 = integrate_band(X, wn, 815.0, half_width)
    auc_835 = integrate_band(X, wn, 835.0, half_width)
    auc_813 = integrate_band(X, wn, 813.0, half_width)
    auc_788 = integrate_band(X, wn, 788.0, half_width)
    return {
        "bio_na_a_form_fraction": _safe_ratio(auc_815, auc_815 + auc_835),  # BIO18
        "bio_rna_dna_ratio":      _safe_ratio(auc_813, auc_788),            # BIO19
    }


def biology_features(
    X: np.ndarray,
    wn: np.ndarray,
    half_width: float = 10.0,
) -> dict[str, np.ndarray]:
    """One-shot dispatch: all 5 biology feature families."""
    out: dict[str, np.ndarray] = {}
    out.update(cytochrome_features(X, wn, half_width))
    out.update(protein_secondary_structure(X, wn, half_width))
    out.update(phb_features(X, wn, half_width))
    out.update(aromatic_aa_features(X, wn, half_width))
    out.update(nucleic_conformation_features(X, wn, half_width))
    return out


# ---------------------------------------------------------------------------
# Feature frame
# ---------------------------------------------------------------------------

# Default set of bands to Lorentzian-fit (avoids fitting all 30 — slow).
# Empirical anchors + literature triple + a few biology anchors.
DEFAULT_FIT_BANDS: tuple[str, ...] = (
    "lps_1050", "lps_1117", "lps_1194",          # empirical anchors
    "na_1338", "lipid_1454", "amide_i_1658",     # literature triple
    "aa_1004",                                    # Phe protein anchor
    "amide_iii_1242",                             # protein backbone
)


def feature_frame(
    X: np.ndarray,
    wn: np.ndarray,
    *,
    ratios: bool = True,
    fits: bool = True,
    fit_model: str = "pseudovoigt",       # "pseudovoigt" | "lorentzian"
    fit_bands: Iterable[str] = DEFAULT_FIT_BANDS,
    half_width: float = 10.0,
    fit_window: float = 30.0,
    rois: bool = True,                     # Stage 15A — SP10-SP16
    emsc: bool = True,                     # Stage 15A — SP25
    derivatives: bool = True,              # Stage 15A — SP1, SP2
    deriv_bands: Iterable[str] | None = None,   # defaults to DEFAULT_FIT_BANDS
    biology: bool = True,                  # Stage 15D — BIO1-29 (5 families)
    emsc_reference: np.ndarray | None = None,
    sg_window: int = 11,
    sg_poly: int = 3,
) -> pd.DataFrame:
    """One-shot DataFrame of all band-aware features for each spectrum.

    Columns (in stable order):
      auc_<group>            — 5 macromolecule groups
      auc_lps_o_antigen_full — 400–900 LPS continuous AUC
      auc_lps_chain_discrim  — 800–1200 LPS continuous AUC (Stage 1 anchor)
      auc_<band_key>         — per-band ±10 AUC for every catalog band
      ratio_<name>           — band-ratio features (if ratios=True)
      fit_<band>_{center,height,fwhm,eta,area,rmse}  — pseudo-Voigt fit params
        (if fits=True; with fit_model="lorentzian", returns only
         center/height/fwhm/area — eta/rmse columns absent for backwards-compat)
      roi_<region>_{mean,std,skew,kurt,centroid,entropy} — Stage 15A (if rois=True)
      emsc_{a,b,c1,c2}       — Stage 15A scatter coefs (if emsc=True)
      d1_auc_<band>          — Stage 15A 1st-derivative AUC (if derivatives=True)
      d2_auc_<band>          — Stage 15A 2nd-derivative AUC (if derivatives=True)
    """
    cols: dict[str, np.ndarray] = {}

    # Macromolecule group AUCs + LPS regions
    macro = macromolecule_vector(X, wn, half_width=half_width)
    for g in MACROMOLECULE_GROUPS:
        cols[f"auc_{g}"] = macro[g]
    for lps_key in LPS_REGIONS:
        cols[f"auc_{lps_key}"] = macro[lps_key]

    # Per-band AUCs
    for band_key, spec in BANDS.items():
        center = spec["center"]
        if center < wn[0] or center > wn[-1]:
            continue
        cols[f"auc_{band_key}"] = integrate_band(X, wn, center, half_width=half_width)

    # Ratios
    if ratios:
        for ratio_name, vals in band_ratios(X, wn, half_width=half_width).items():
            cols[f"ratio_{ratio_name}"] = vals

    # Peak fits (Stage 15A: pseudo-Voigt default; lorentzian as legacy option)
    if fits:
        if fit_model == "pseudovoigt":
            fits_dict = fit_peaks_batch_pseudovoigt(X, wn, fit_bands, window=fit_window)
            for band_key, mat in fits_dict.items():
                cols[f"fit_{band_key}_center"] = mat[:, 0]
                cols[f"fit_{band_key}_height"] = mat[:, 1]
                cols[f"fit_{band_key}_fwhm"]   = mat[:, 2]
                cols[f"fit_{band_key}_eta"]    = mat[:, 3]
                cols[f"fit_{band_key}_area"]   = mat[:, 4]
                cols[f"fit_{band_key}_rmse"]   = mat[:, 5]
        elif fit_model == "lorentzian":
            fits_dict = fit_peaks_batch(X, wn, fit_bands, window=fit_window)
            for band_key, mat in fits_dict.items():
                cols[f"fit_{band_key}_center"] = mat[:, 0]
                cols[f"fit_{band_key}_height"] = mat[:, 1]
                cols[f"fit_{band_key}_fwhm"]   = mat[:, 2]
                cols[f"fit_{band_key}_area"]   = mat[:, 3]
        else:
            raise ValueError(f"fit_model must be 'pseudovoigt' or 'lorentzian', got {fit_model!r}")

    # ROI moments (Stage 15A)
    if rois:
        for col_name, vals in roi_moments(X, wn).items():
            cols[col_name] = vals

    # EMSC scatter coefficients (Stage 15A)
    if emsc:
        _residual, emsc_coefs = emsc_correct(
            X, wn, reference=emsc_reference, poly_degree=2, return_corrected=False
        )
        # columns: a, b, c1, c2
        cols["emsc_a"]  = emsc_coefs[:, 0]
        cols["emsc_b"]  = emsc_coefs[:, 1]
        cols["emsc_c1"] = emsc_coefs[:, 2]
        cols["emsc_c2"] = emsc_coefs[:, 3]

    # Derivative AUCs (Stage 15A)
    if derivatives:
        d_bands = list(deriv_bands or DEFAULT_FIT_BANDS)
        d1 = derivative_band_auc(X, wn, deriv=1, band_keys=d_bands,
                                 half_width=half_width,
                                 sg_window=sg_window, sg_poly=sg_poly)
        d2 = derivative_band_auc(X, wn, deriv=2, band_keys=d_bands,
                                 half_width=half_width,
                                 sg_window=sg_window, sg_poly=sg_poly)
        for k, vals in d1.items():
            cols[f"d1_auc_{k}"] = vals
        for k, vals in d2.items():
            cols[f"d2_auc_{k}"] = vals

    # Stage 15D — biology-specific features (5 families)
    if biology:
        for k, vals in biology_features(X, wn, half_width=half_width).items():
            cols[k] = vals

    df = pd.DataFrame(cols)
    return df
