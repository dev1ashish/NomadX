"""Stage 2 of plan/14 — build band-features cache + smoke checks.

Pre-registered in plan/08 (2026-05-17 Stage 2 entry).

Pipeline:
  1. Smoke checks (synthetic Lorentzian recovery, AUC sanity on flat signal)
  2. Compute feature_frame on the full QC-passed dataset
  3. Save to data_cache/band_features.parquet (row-aligned with spectra.parquet
     after qc_mask is applied)
  4. Print pre-registered sanity numbers:
     - per-class macromolecule vector means + Cohen's d at LPS group
     - per-class 800–1200 LPS chain region AUC (Stage 1 empirical anchor)
     - Lorentzian fit success rate per band
     - fitted peak-center mean shift from catalog center per band
"""
from __future__ import annotations
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

REPO = Path("/Users/devashishthapliyal/Documents/NomadX")
sys.path.insert(0, str(REPO))

from atlas.io import load_cache
from atlas import band_features as bf

CACHE = REPO / "data_cache"
OUT = REPO / "outputs" / "band_chemistry"
OUT.mkdir(parents=True, exist_ok=True)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


# =============================================================================
# Step 1 — Smoke checks
# =============================================================================
print("=" * 70)
print("Stage 2 smoke checks")
print("=" * 70)

# 1a) Synthetic pseudo-Voigt recovery (Stage 15A check)
print("\n[1a] Synthetic pseudo-Voigt recovery (center=1454, height=2.0, FWHM≈15, η=0.5)")
wn_test = np.linspace(1400, 1500, 200)
# Build a true pseudo-Voigt with σ ≈ γ ≈ 7.5 so FWHM_L=15, FWHM_G≈17.6, mix=0.5
true = bf._pseudovoigt_linbase(
    wn_test, a=2.0, x0=1454.0, sigma=7.5, gamma=7.5, eta=0.5, b=0.0, c=0.0,
)
noisy = true + np.random.default_rng(0).normal(0, 0.02, size=true.shape)
fit_pv = bf.fit_peak_pseudovoigt(noisy, wn_test, center=1454.0, window=40.0)
ok_center = abs(fit_pv.center - 1454.0) < 0.5
ok_height = abs(fit_pv.height - 2.0) < 0.1
ok_eta    = 0.3 < fit_pv.eta < 0.7
print(f"  pv.center={fit_pv.center:.3f} (truth 1454.0, |err|<0.5? {ok_center})")
print(f"  pv.height={fit_pv.height:.4f} (truth 2.000, |err|<0.1? {ok_height})")
print(f"  pv.fwhm  ={fit_pv.fwhm:.3f} (truth ≈ 16.3)")
print(f"  pv.eta   ={fit_pv.eta:.3f} (truth 0.500, 0.3<η<0.7? {ok_eta})")
assert ok_center and ok_height and ok_eta, "pseudo-Voigt recovery failed"

# 1b) integrate_band on flat unit signal
print("\n[1b] integrate_band on flat unit signal")
X_flat = np.ones((4, 100))
wn_flat = np.linspace(1000, 1200, 100)
auc = bf.integrate_band(X_flat, wn_flat, center=1100.0, half_width=10.0)
m = (wn_flat >= 1090.0) & (wn_flat <= 1110.0)
expected = float(wn_flat[m][-1] - wn_flat[m][0])  # trapz of unit signal = width
print(f"  AUC for flat-unit signal: {auc[0]:.4f}  (expected ≈ {expected:.4f})")
assert np.allclose(auc, expected, atol=0.05), f"Expected ~{expected}, got {auc}"

# 1c) Band-ratio sanity: identical numerator and denominator → ~1
print("\n[1c] Band-ratio sanity (band/band → ~1.0 by construction)")
X_synth = np.random.default_rng(0).uniform(0.5, 1.5, size=(5, 200))
wn_synth = np.linspace(900, 1300, 200)
r = bf.band_ratios(X_synth, wn_synth,
                   pairs={"self": ("band:lps_1050", "band:lps_1050")})
print(f"  self-ratio mean: {r['self'].mean():.4f}  (expected 1.0)")
assert np.allclose(r["self"], 1.0, atol=1e-6)

print("\nAll smoke checks PASSED ✅")


# =============================================================================
# Step 2 — Load preprocessed cache
# =============================================================================
print("\n" + "=" * 70)
print("Step 2 — loading preprocessed data")
print("=" * 70)
spec_df, _Xraw, _wnraw, meta = load_cache(CACHE)
X = np.load(CACHE / "spectra_array_preprocessed.npy")
wn = np.load(CACHE / "wavenumber_axis_preprocessed.npy")
qc = np.load(CACHE / "qc_mask.npy")
spec_df = spec_df.reset_index(drop=True)
spec_df_qc = spec_df[qc].reset_index(drop=True)
Xq = X[qc]
print(f"  spectra: {Xq.shape}  wn: {wn[0]:.1f}–{wn[-1]:.1f} ({len(wn)} bins)")
print(f"  classes: {spec_df_qc['primary_class'].value_counts().to_dict()}")


# =============================================================================
# Step 3 — Compute feature frame on full dataset
# =============================================================================
print("\n" + "=" * 70)
print("Step 3 — computing feature_frame")
print("=" * 70)
t0 = time.time()
df = bf.feature_frame(Xq, wn, ratios=True, fits=True)
dt = time.time() - t0
print(f"  shape: {df.shape}  ({len(df.columns)} columns)")
print(f"  elapsed: {dt:.1f}s  ({1000*dt/len(df):.1f} ms/spectrum)")
print(f"  columns by prefix:")
print(f"    auc_   : {sum(c.startswith('auc_') for c in df.columns)}")
print(f"    ratio_ : {sum(c.startswith('ratio_') for c in df.columns)}")
print(f"    fit_   : {sum(c.startswith('fit_') for c in df.columns)}")

# Row-align with spec_df_qc and save
df.index = spec_df_qc.index
out_path = CACHE / "band_features.parquet"
df.to_parquet(out_path, compression="snappy")
print(f"  wrote {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")


# =============================================================================
# Step 4 — Pre-registered sanity numbers
# =============================================================================
print("\n" + "=" * 70)
print("Step 4 — pre-registered sanity checks (plan/08 Stage 2 predictions)")
print("=" * 70)

# Join feature frame with class labels
joined = df.copy()
joined["primary_class"] = spec_df_qc["primary_class"].values
joined["subclass"]      = spec_df_qc["subclass"].fillna("H2O").values
joined["file_id"]       = spec_df_qc["file_id"].values
file_level = joined.groupby("file_id").agg("mean", numeric_only=True)
file_level["primary_class"] = joined.groupby("file_id")["primary_class"].first()

# ---- Sanity (i): macromolecule LPS AUC differs E. coli vs Salmonella ----
print("\n[i] LPS group AUC: E. coli vs Salmonella (file-level)")
ecoli_files = file_level["primary_class"].isin(["STEC", "Non-STEC"])
salmo_files = file_level["primary_class"] == "Salmonella"
for region in ("auc_lps_o_antigen_full", "auc_lps_chain_discrim"):
    a = file_level.loc[ecoli_files, region].values
    b = file_level.loc[salmo_files, region].values
    d = cohens_d(a, b)
    direction = "E.coli > Salmonella" if a.mean() > b.mean() else "E.coli < Salmonella"
    print(f"  {region:30s} d={d:+.3f}  ({direction})  "
          f"E.coli n={len(a)} Salmonella n={len(b)}")

# ---- Sanity (ii): macromolecule NA / amide / AA: STEC vs Non-STEC ----
print("\n[ii] Macromolecule group AUC: STEC vs Non-STEC (file-level, plan/08 predicted |d|<0.3)")
stec_files = file_level["primary_class"] == "STEC"
nstec_files = file_level["primary_class"] == "Non-STEC"
for g in ("auc_aromatic_aa", "auc_protein_amide", "auc_nucleic_acid",
          "auc_lipid_carbohydrate", "auc_metabolite"):
    a = file_level.loc[stec_files, g].values
    b = file_level.loc[nstec_files, g].values
    d = cohens_d(a, b)
    print(f"  {g:30s} d={d:+.3f}  STEC n={len(a)} Non-STEC n={len(b)}")

# ---- Sanity (iii): empirical anchor 800–1200 LPS chain: STEC vs Non-STEC ----
print("\n[iii] 800–1200 LPS chain: STEC vs Non-STEC (plan/08 predicted |d|≥0.4)")
a = file_level.loc[stec_files, "auc_lps_chain_discrim"].values
b = file_level.loc[nstec_files, "auc_lps_chain_discrim"].values
d_anchor = cohens_d(a, b)
print(f"  auc_lps_chain_discrim  d={d_anchor:+.3f}  STEC n={len(a)} Non-STEC n={len(b)}")
print(f"  Anchor band AUCs (file-level d STEC vs Non-STEC):")
for band in ("lps_1050", "lps_1117", "lps_1194"):
    col = f"auc_{band}"
    a = file_level.loc[stec_files, col].values
    b = file_level.loc[nstec_files, col].values
    d = cohens_d(a, b)
    print(f"    {col:25s} d={d:+.3f}")

# ---- Sanity (iv): pseudo-Voigt fit success rate (Stage 15A target: ≥80% on empirical anchors) ----
print("\n[iv] Pseudo-Voigt fit success rate (success = finite center within ±20 cm⁻¹, FWHM ∈ [5, 60])")
for band_key in bf.DEFAULT_FIT_BANDS:
    center_col = f"fit_{band_key}_center"
    fwhm_col   = f"fit_{band_key}_fwhm"
    if center_col not in df.columns:
        continue
    catalog_center = bf.BANDS[band_key]["center"]
    fitted = df[center_col].values
    fwhms  = df[fwhm_col].values
    success = (
        np.isfinite(fitted)
        & (np.abs(fitted - catalog_center) <= 20.0)
        & (fwhms >= 5.0) & (fwhms <= 60.0)
    )
    rate = success.mean()
    pass_mark = "✅" if rate >= 0.80 else ("⚠️" if rate >= 0.60 else "❌")
    print(f"  {band_key:18s} catalog={catalog_center:7.1f}  rate={rate*100:5.1f}% {pass_mark}  "
          f"(n_ok={success.sum()}/{len(success)})")

# ---- Sanity (v): fitted peak-center mean drift ----
print("\n[v] Mean fitted peak-center drift from catalog center")
print("    (per primary class, on successful fits only)")
drift_rows = []
for band_key in bf.DEFAULT_FIT_BANDS:
    center_col = f"fit_{band_key}_center"
    fwhm_col   = f"fit_{band_key}_fwhm"
    if center_col not in df.columns:
        continue
    catalog = bf.BANDS[band_key]["center"]
    fitted_all = df[center_col].values
    fwhms_all  = df[fwhm_col].values
    mask = (
        np.isfinite(fitted_all)
        & (np.abs(fitted_all - catalog) <= 20.0)
        & (fwhms_all >= 5.0) & (fwhms_all <= 40.0)
    )
    row = {"band": band_key, "catalog": catalog}
    for cls in ["STEC", "Non-STEC", "Salmonella", "H2O"]:
        cls_mask = mask & (spec_df_qc["primary_class"].values == cls)
        if cls_mask.sum() > 5:
            mean_drift = float(fitted_all[cls_mask].mean() - catalog)
            row[cls] = mean_drift
        else:
            row[cls] = np.nan
    drift_rows.append(row)
drift_df = pd.DataFrame(drift_rows)
print(drift_df.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
drift_df.to_csv(OUT / "04_stage2_peak_drift.csv", index=False)

# ---- Sanity (vi): feature cache size ----
print("\n[vi] Feature cache size")
print(f"  rows: {df.shape[0]} (expected 7,122)")
print(f"  cols: {df.shape[1]} (predicted 130–145 for Stage 15A)")
print(f"  prefix counts:")
for prefix in ("auc_", "ratio_", "fit_", "roi_", "emsc_", "d1_", "d2_"):
    n = sum(c.startswith(prefix) for c in df.columns)
    print(f"    {prefix:10s} {n}")

# ---- Sanity (vii): H2O macromolecule AUCs vs bacterial ----
print("\n[vii] H₂O class: should have LOWER macromolecule AUCs than bacterial classes")
h2o_files = file_level["primary_class"] == "H2O"
bact_files = file_level["primary_class"].isin(["STEC", "Non-STEC", "Salmonella"])
for g in ("auc_aromatic_aa", "auc_protein_amide", "auc_nucleic_acid",
          "auc_lipid_carbohydrate"):
    h = file_level.loc[h2o_files, g].values
    b = file_level.loc[bact_files, g].values
    d = cohens_d(h, b)
    direction = "H2O < bacteria" if h.mean() < b.mean() else "H2O > bacteria"
    print(f"  {g:30s} d={d:+.3f}  ({direction})")

# ---- Sanity (viii): Stage 15A targets — EMSC, D2-AUC, ROI centroid ----
print("\n[viii] Stage 15A target checks")
print("  EMSC b-coefficient (multiplicative scatter), E. coli vs Salmonella file-level d:")
emsc_b = file_level["emsc_b"].values
ecoli = file_level["primary_class"].isin(["STEC", "Non-STEC"]).values
salm  = (file_level["primary_class"] == "Salmonella").values
d_emsc = cohens_d(emsc_b[ecoli], emsc_b[salm])
print(f"    auc_emsc_b              d={d_emsc:+.3f}  E.coli n={ecoli.sum()} Salmonella n={salm.sum()}")
print("  D2-AUC lps_1194, STEC vs Non-STEC file-level d:")
if "d2_auc_lps_1194" in file_level.columns:
    stec = file_level["primary_class"].values == "STEC"
    nstec = file_level["primary_class"].values == "Non-STEC"
    d_d2 = cohens_d(file_level.loc[stec, "d2_auc_lps_1194"].values,
                    file_level.loc[nstec, "d2_auc_lps_1194"].values)
    print(f"    d2_auc_lps_1194         d={d_d2:+.3f}")
print("  ROI centroid in lps_chain, E. coli vs Salmonella file-level d:")
if "roi_lps_chain_centroid" in file_level.columns:
    d_cen = cohens_d(file_level.loc[ecoli, "roi_lps_chain_centroid"].values,
                     file_level.loc[salm, "roi_lps_chain_centroid"].values)
    print(f"    roi_lps_chain_centroid  d={d_cen:+.3f}")

# ---- Per-class macromolecule vector summary ----
print("\n[summary] Per-class macromolecule vector means (file-level)")
mac_cols = [c for c in df.columns if c.startswith("auc_") and not c.startswith("auc_lps")]
summary = file_level.groupby("primary_class")[mac_cols].mean()
print(summary.round(3).to_string())
summary.round(4).to_csv(OUT / "04_stage2_per_class_macromolecule.csv")

print("\n" + "=" * 70)
print(f"Stage 2 DONE — feature cache at {out_path}")
print("=" * 70)
