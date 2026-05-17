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
    """Fit each named band on every spectrum.

    Returns a dict where each key maps to (N, 4) array of
    (center, height, fwhm, area). NaN where the fit failed.
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
    fit_bands: Iterable[str] = DEFAULT_FIT_BANDS,
    half_width: float = 10.0,
    fit_window: float = 30.0,
) -> pd.DataFrame:
    """One-shot DataFrame of all band-aware features for each spectrum.

    Columns (in stable order):
      auc_<group>            — 5 macromolecule groups
      auc_lps_o_antigen_full — 400–900 LPS continuous AUC
      auc_lps_chain_discrim  — 800–1200 LPS continuous AUC (Stage 1 anchor)
      auc_<band_key>         — per-band ±10 AUC for every catalog band
      ratio_<name>           — band-ratio features (if ratios=True)
      fit_<band>_{center,height,fwhm,area} — Lorentzian fit params (if fits=True)
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

    # Lorentzian fits
    if fits:
        fits_dict = fit_peaks_batch(X, wn, fit_bands, window=fit_window)
        for band_key, mat in fits_dict.items():
            cols[f"fit_{band_key}_center"] = mat[:, 0]
            cols[f"fit_{band_key}_height"] = mat[:, 1]
            cols[f"fit_{band_key}_fwhm"]   = mat[:, 2]
            cols[f"fit_{band_key}_area"]   = mat[:, 3]

    df = pd.DataFrame(cols)
    return df
