"""Build data_cache/spectral_features.parquet (Stage 15B).

Pre-registered in plan/experiments/2026-05-18_stage15b_dwt_pca_sam.md.

Computes DWT + ROI-PCA + SAM features on the QC-passed preprocessed array,
row-aligned with the existing data_cache/band_features.parquet (7,122 rows).

Sanity checks at the end:
  - DWT energy L4 file-level d, E. coli vs Salmonella
  - ROI-PCA LPS top-5 cumulative variance
  - ROI-PCA LPS PC1 file-level d, STEC vs Non-STEC
  - SAM_lps_sub_* file-level AUROC for STEC vs Non-STEC
  - SAM_class_H2O file-level AUROC for H2O vs bacteria
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
from atlas import spectral_features as sf
from sklearn.metrics import roc_auc_score

CACHE = REPO / "data_cache"
OUT_PARQUET = CACHE / "spectral_features.parquet"


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def best_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return max(AUC, 1-AUC). labels: 0/1."""
    if scores.size < 4 or len(set(labels)) < 2:
        return float("nan")
    if not np.all(np.isfinite(scores)):
        return float("nan")
    a = roc_auc_score(labels, scores)
    return max(a, 1 - a)


# ---------- load ----------
print("Loading cache …")
spec_df, _X, _wn, meta = load_cache(CACHE)
X = np.load(CACHE / "spectra_array_preprocessed.npy")
wn = np.load(CACHE / "wavenumber_axis_preprocessed.npy")
qc = np.load(CACHE / "qc_mask.npy")
spec_df = spec_df.reset_index(drop=True)
spec_df_qc = spec_df[qc].reset_index(drop=True)
Xq = X[qc]
print(f"  spectra: {Xq.shape}  wn: {wn[0]:.1f}-{wn[-1]:.1f}  ({len(wn)} bins)")
print(f"  classes: {spec_df_qc['primary_class'].value_counts().to_dict()}")


# ---------- build feature frame ----------
print("\nBuilding spectral feature frame (DWT + ROI-PCA + SAM) …")
t0 = time.perf_counter()
df = sf.feature_frame_spectral(Xq, wn, spec_df=spec_df_qc)
elapsed = time.perf_counter() - t0
print(f"  shape: {df.shape}  ({len(df.columns)} columns)")
print(f"  elapsed: {elapsed:.1f}s")
print(f"  prefix counts:")
for prefix in ("dwt_", "pca_", "sam_"):
    n = sum(c.startswith(prefix) for c in df.columns)
    print(f"    {prefix:6s} {n}")

# Row-align and save
df.index = spec_df_qc.index
df.to_parquet(OUT_PARQUET, compression="snappy")
print(f"  wrote {OUT_PARQUET}  ({OUT_PARQUET.stat().st_size / 1024:.1f} KB)")


# ---------- Stage 15B sanity checks ----------
print("\n" + "=" * 70)
print("Stage 15B sanity checks (plan/experiments/...stage15b... predictions)")
print("=" * 70)

joined = df.copy()
joined["primary_class"] = spec_df_qc["primary_class"].values
joined["subclass"] = spec_df_qc["subclass"].fillna("H2O").values
joined["file_id"] = spec_df_qc["file_id"].values

file_level = joined.groupby("file_id").agg("mean", numeric_only=True)
file_level["primary_class"] = joined.groupby("file_id")["primary_class"].first()

stec = (file_level["primary_class"] == "STEC").values
nstec = (file_level["primary_class"] == "Non-STEC").values
ecoli = file_level["primary_class"].isin(["STEC", "Non-STEC"]).values
salm = (file_level["primary_class"] == "Salmonella").values
h2o = (file_level["primary_class"] == "H2O").values
bact = file_level["primary_class"].isin(["STEC", "Non-STEC", "Salmonella"]).values

# (i) DWT energy / entropy per class
print("\n[i] DWT energy file-level d, E. coli vs Salmonella")
for L in range(1, 7):
    col = f"dwt_energy_L{L}"
    if col not in file_level.columns:
        continue
    d = cohens_d(file_level.loc[ecoli, col].values,
                 file_level.loc[salm, col].values)
    print(f"  {col:18s}  d={d:+.3f}  (E.coli vs Salmonella)")

print("\n[i'] DWT entropy file-level d, E. coli vs Salmonella")
for L in range(1, 7):
    col = f"dwt_entropy_L{L}"
    if col not in file_level.columns:
        continue
    d = cohens_d(file_level.loc[ecoli, col].values,
                 file_level.loc[salm, col].values)
    print(f"  {col:18s}  d={d:+.3f}")

# (ii) ROI-PCA cumulative variance
print("\n[ii] ROI-PCA cumulative variance explained")
# We need to refit briefly to get the variance ratios
fitted = sf.fit_roi_pca(Xq, wn)
for key, info in fitted.items():
    if info["pca"] is None:
        print(f"  {key:10s}: PCA not fit")
        continue
    vr = info["pca"].explained_variance_ratio_
    print(f"  {key:10s}: per-PC = {np.round(vr, 3).tolist()}  cumulative = {vr.cumsum()[-1]:.3f}")

# (iii) ROI-PCA file-level d, STEC vs Non-STEC
print("\n[iii] ROI-PCA file-level d, STEC vs Non-STEC")
for col in sorted([c for c in file_level.columns if c.startswith("pca_")]):
    d = cohens_d(file_level.loc[stec, col].values,
                 file_level.loc[nstec, col].values)
    print(f"  {col:20s}  d={d:+.3f}")

# (iv) SAM file-level AUROC for class pairs
print("\n[iv] SAM file-level AUROC, STEC vs Non-STEC (lower angle = closer)")
sam_cols = [c for c in file_level.columns if c.startswith("sam_")]
results = []
for col in sam_cols:
    scores = file_level[col].values
    # STEC vs Non-STEC
    labels_sn = file_level["primary_class"].isin(["STEC", "Non-STEC"]).values
    labels_stec = (file_level["primary_class"] == "STEC").values
    if labels_sn.sum() < 2:
        continue
    auc_stec = best_auroc(scores[labels_sn], labels_stec[labels_sn])
    results.append((col, auc_stec))

# Top 10 SAM features for STEC vs Non-STEC
results.sort(key=lambda kv: -kv[1])
print("  top 10 SAM features by STEC↔Non-STEC AUROC:")
for col, auc in results[:10]:
    flag = "🟢" if auc >= 0.80 else ("🟡" if auc >= 0.65 else "🔴")
    print(f"    {auc:.3f}  {flag}  {col}")

# (v) SAM E. coli vs Salmonella AUROC
print("\n[v] SAM file-level AUROC, E. coli vs Salmonella")
results_es = []
for col in sam_cols:
    scores = file_level[col].values
    labels = file_level["primary_class"].isin(["STEC", "Non-STEC", "Salmonella"]).values
    is_ecoli = file_level["primary_class"].isin(["STEC", "Non-STEC"]).values
    if labels.sum() < 2:
        continue
    auc = best_auroc(scores[labels], is_ecoli[labels])
    results_es.append((col, auc))
results_es.sort(key=lambda kv: -kv[1])
print("  top 10 SAM features by E.coli↔Salmonella AUROC:")
for col, auc in results_es[:10]:
    flag = "🟢" if auc >= 0.80 else ("🟡" if auc >= 0.65 else "🔴")
    print(f"    {auc:.3f}  {flag}  {col}")

# (vi) SAM H2O vs bacteria
print("\n[vi] SAM file-level AUROC, H2O vs bacteria (sanity)")
for col in sam_cols:
    if "H2O" not in col:
        continue
    scores = file_level[col].values
    auc = best_auroc(scores, h2o.astype(int))
    print(f"    {col:30s}  AUROC={auc:.3f}")

# Save sanity report as JSON
import json
report = {
    "n_features": int(df.shape[1]),
    "n_spectra": int(df.shape[0]),
    "dwt_energy_d_ecoli_vs_salm": {
        f"L{L}": cohens_d(file_level.loc[ecoli, f"dwt_energy_L{L}"].values,
                          file_level.loc[salm, f"dwt_energy_L{L}"].values)
        for L in range(1, 7)
    },
    "pca_cumulative_variance": {
        k: float(info["pca"].explained_variance_ratio_.sum())
        for k, info in fitted.items() if info["pca"] is not None
    },
    "pca_lps_pc1_d_stec_vs_nstec": cohens_d(
        file_level.loc[stec, "pca_lps_PC1"].values,
        file_level.loc[nstec, "pca_lps_PC1"].values,
    ),
    "best_sam_stec_vs_nstec": {col: auc for col, auc in results[:10]},
    "best_sam_ecoli_vs_salm": {col: auc for col, auc in results_es[:10]},
}
out_json = REPO / "outputs" / "band_chemistry" / "stage15b"
out_json.mkdir(parents=True, exist_ok=True)
with (out_json / "01_stage15b_summary.json").open("w") as f:
    json.dump(report, f, indent=2, default=float)
print(f"\nwrote {out_json / '01_stage15b_summary.json'}")

print("\n=== Stage 15B DONE ===")
