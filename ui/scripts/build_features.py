"""Atlas Raman UI — Feature catalog sidecar builder (W5).

Produces `ui/public/data/feature_catalog.json` covering all 259 engineered
features across the 4 cache families (band 166 + spectral 51 + unmix/MCR 32 +
spatial 10). Per-pixel caches (band, spectral) are mean-pooled to file level
so every feature is computed on the same 87-file basis used by Stage 15F.

For each feature we compute:
- Cohen's d (file-level, pooled-SD) for STEC vs. Non-STEC.
- Cohen's d for E. coli (STEC+Non-STEC) vs. Salmonella.
- The Stage 15F MI rank (1-indexed) if the feature appears in
  `artifacts/stage15f_feature_columns.json`.
- Per-class distribution stats (mean, std, n) — used by the front-end to
  approximate violin/box plots without shipping all 7,122 rows.

Plan reference: `plan/ui/ULTRAPLAN.md` §4 W5 + §6.
Sidecar contract mirrored in `ui/lib/types.ts` (Feature).

Usage (idempotent):
    cd ui && python scripts/build_features.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

UI_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = UI_DIR.parent
DATA_CACHE = REPO_ROOT / "data_cache"
ARTIFACTS = REPO_ROOT / "artifacts"
OUT_PATH = UI_DIR / "public" / "data" / "feature_catalog.json"

CLASSES = ("STEC", "Non-STEC", "Salmonella", "H2O")

# Heuristic markers for the "bio" family inside the band-feature cache.
# Stage 15D introduced `bio_*` features that live in band_features.parquet
# but represent biochemical aggregates rather than per-band fits.
BIO_PREFIXES = ("bio_",)

# Region inference from feature name. Keep this conservative — front-end will
# only badge a region when we have one.
REGION_KEYWORDS = {
    "lps_chain": ("lps_chain", "lps_1050", "lps_1117", "lps_1194",
                  "lps_o_antigen", "lps_discrim"),
    "ch_stretch": ("ch_stretch", "chstretch", "lipid_2850", "lipid_2930"),
    "amide_i": ("amide_i_1658", "amide_i_1662"),
    "amide_iii": ("amide_iii_1242",),
    "aromatic_aa": ("aa_1004", "aromatic_aa"),
    "nucleic_acid": ("na_1338", "nucleic_acid"),
    "silent": ("roi_silent",),
    "lipid": ("lipid_1080", "lipid_1451", "lipid_1454", "lipid_1585"),
    "metabolite": ("metabolite",),
}


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def cohen_d(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pooled-SD Cohen's d for two 1D arrays. Returns None if either group
    has <2 finite samples or pooled SD is zero/NaN."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return None
    va = float(a.var(ddof=1))
    vb = float(b.var(ddof=1))
    # Pooled standard deviation (Hedges-style without small-sample correction).
    pooled = math.sqrt(((a.size - 1) * va + (b.size - 1) * vb)
                       / (a.size + b.size - 2))
    if not math.isfinite(pooled) or pooled == 0.0:
        return None
    d = (float(a.mean()) - float(b.mean())) / pooled
    return round(d, 4) if math.isfinite(d) else None


def per_class_stats(series: pd.Series, classes: pd.Series) -> dict:
    """Mean / std / n grouped by primary_class. NaN-safe; rounded for JSON
    payload weight."""
    out: dict[str, dict[str, float | int]] = {}
    for cls in CLASSES:
        vals = series[classes == cls].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            out[cls] = {"mean": 0.0, "std": 0.0, "n": 0}
            continue
        out[cls] = {
            "mean": round(float(finite.mean()), 6),
            "std": round(float(finite.std(ddof=1)) if finite.size > 1 else 0.0, 6),
            "n": int(finite.size),
        }
    return out


def infer_region(feature_name: str) -> str | None:
    name = feature_name.lower()
    for region, kws in REGION_KEYWORDS.items():
        if any(kw in name for kw in kws):
            return region
    return None


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def load_file_level_features() -> tuple[pd.DataFrame, pd.Series, dict[str, str]]:
    """Return a DataFrame indexed by file_id with one column per feature,
    plus the per-file primary_class series and a {feature: family} map."""

    # --- metadata + QC mapping ---------------------------------------------
    metadata = pd.read_parquet(DATA_CACHE / "metadata.parquet")
    metadata = metadata.set_index("file_id")
    primary_class = metadata["primary_class"].astype(str)

    qc_mask = np.load(DATA_CACHE / "qc_mask.npy").astype(bool)
    spectra = pd.read_parquet(DATA_CACHE / "spectra.parquet")
    if len(spectra) != qc_mask.size:
        raise RuntimeError(
            f"qc_mask ({qc_mask.size}) and spectra ({len(spectra)}) mismatch")
    spectra_qc = spectra.loc[qc_mask].reset_index(drop=True)

    # --- per-pixel caches → mean-pool to file level ------------------------
    band = pd.read_parquet(DATA_CACHE / "band_features.parquet")
    spectral = pd.read_parquet(DATA_CACHE / "spectral_features.parquet")
    if len(band) != len(spectra_qc) or len(spectral) != len(spectra_qc):
        raise RuntimeError(
            "per-pixel cache row counts don't match QC-passed spectra; "
            f"band={len(band)} spectral={len(spectral)} qc={len(spectra_qc)}")

    band = band.copy()
    band["__file_id__"] = spectra_qc["file_id"].to_numpy()
    spectral = spectral.copy()
    spectral["__file_id__"] = spectra_qc["file_id"].to_numpy()

    # Numeric mean-pool. Some columns may be non-numeric (e.g. sam_lps_sub_*
    # one-hots); coerce + skip NaN-only groups.
    band_num = band.drop(columns="__file_id__").apply(pd.to_numeric,
                                                     errors="coerce")
    spectral_num = spectral.drop(columns="__file_id__").apply(pd.to_numeric,
                                                              errors="coerce")
    band_file = band_num.groupby(band["__file_id__"]).mean()
    spectral_file = spectral_num.groupby(spectral["__file_id__"]).mean()
    band_file.index.name = "file_id"
    spectral_file.index.name = "file_id"

    # --- per-file caches load directly ------------------------------------
    unmix = pd.read_parquet(DATA_CACHE / "unmix_features.parquet")
    # `mcr_residual_norm_mean` is the unmixing residual, not a per-component
    # feature — drop it so MCR family count = 32 (8 components × 4 stats),
    # matching the 259-feature catalog called for in plan §4 W5.
    if "mcr_residual_norm_mean" in unmix.columns:
        unmix = unmix.drop(columns=["mcr_residual_norm_mean"])
    spatial = pd.read_parquet(DATA_CACHE / "spatial_features.parquet")

    # Reindex everything to the metadata file order so groups align.
    file_order = primary_class.index
    band_file = band_file.reindex(file_order)
    spectral_file = spectral_file.reindex(file_order)
    unmix = unmix.reindex(file_order)
    spatial = spatial.reindex(file_order)

    # --- family map -------------------------------------------------------
    family_map: dict[str, str] = {}
    for col in band_file.columns:
        family_map[col] = "bio" if col.startswith(BIO_PREFIXES) else "band"
    for col in spectral_file.columns:
        family_map[col] = "spectral"
    for col in unmix.columns:
        family_map[col] = "mcr"
    for col in spatial.columns:
        family_map[col] = "spatial"

    # --- combined wide table ----------------------------------------------
    combined = pd.concat(
        [band_file, spectral_file, unmix, spatial],
        axis=1,
    )
    # Guard: drop accidental duplicates (none expected; defensive).
    combined = combined.loc[:, ~combined.columns.duplicated()]

    return combined, primary_class, family_map


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    combined, primary_class, family_map = load_file_level_features()

    # Stage 15F MI rank lookup.
    with (ARTIFACTS / "stage15f_feature_columns.json").open() as fh:
        stage15f_35: list[str] = json.load(fh)
    mi_rank = {name: idx + 1 for idx, name in enumerate(stage15f_35)}

    classes = primary_class
    stec_mask = (classes == "STEC")
    nonstec_mask = (classes == "Non-STEC")
    salm_mask = (classes == "Salmonella")
    ecoli_mask = stec_mask | nonstec_mask

    features_out: list[dict] = []
    per_class_out: dict[str, dict] = {}

    for col in combined.columns:
        series = combined[col]
        family = family_map.get(col, "band")
        region = infer_region(col)

        vals = series.to_numpy(dtype=float)
        d_sn = cohen_d(vals[stec_mask.to_numpy()],
                       vals[nonstec_mask.to_numpy()])
        d_es = cohen_d(vals[ecoli_mask.to_numpy()],
                       vals[salm_mask.to_numpy()])

        entry: dict = {
            "name": col,
            "family": family,
            "d_stec_nonstec": d_sn,
            "d_ecoli_salm": d_es,
            "mi_rank_stage15f": mi_rank.get(col),
        }
        if region is not None:
            entry["region"] = region
        features_out.append(entry)

        per_class_out[col] = per_class_stats(series, classes)

    # Top-15 by |d_stec_nonstec|.
    ranked = sorted(
        (f for f in features_out if f["d_stec_nonstec"] is not None),
        key=lambda f: abs(f["d_stec_nonstec"]),
        reverse=True,
    )
    top_15 = [f["name"] for f in ranked[:15]]

    payload = {
        "features": features_out,
        "per_class_stats": per_class_out,
        "top_15_stec_nonstec": top_15,
        "stage15f_35": stage15f_35,
    }

    with OUT_PATH.open("w") as fh:
        json.dump(payload, fh, indent=2)

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)} "
          f"({len(features_out)} features, {size_kb:.1f} KB)")
    if len(features_out) != 259:
        raise SystemExit(
            f"FAIL: expected 259 features, got {len(features_out)}")


if __name__ == "__main__":
    main()
