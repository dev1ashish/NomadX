"""Build data_cache/spatial_features.parquet (Stage 15E).

Pre-registered in plan/experiments/2026-05-18_stage15e_spatial_features.md.

Computes per-file pixel-moment statistics:
  - within-file variance + CV of LPS-chain (800-1200) and CH-stretch (2800-3000) region AUCs
  - kurtosis + skew of per-pixel AUC at lps_1050 / lps_1117 / lps_1194 anchor bands

Sanity checks at the end:
  - H2O class should have LOWER variance/CV than bacterial classes (uniform water)
  - File-level Cohen's d for STEC↔Non-STEC and E.coli↔Salm
  - K-12 vs other-STEC for any unexpected K-12 axis
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

REPO = Path("/Users/devashishthapliyal/Documents/NomadX")
sys.path.insert(0, str(REPO))

from atlas.io import load_cache
from atlas import spatial_features as sp
from sklearn.metrics import roc_auc_score

CACHE = REPO / "data_cache"
OUT_PARQUET = CACHE / "spatial_features.parquet"
OUT_DIR = REPO / "outputs" / "band_chemistry" / "stage15e"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def cohens_d(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return np.nan
    pooled = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return float((a.mean()-b.mean())/pooled) if pooled > 0 else 0.0


def auroc_either(scores, labels):
    s = np.asarray(scores, dtype=float); lab = np.asarray(labels)
    mask = ~np.isnan(s) & ~np.isnan(lab.astype(float))
    if mask.sum() < 4 or len(set(lab[mask])) < 2: return np.nan
    a = roc_auc_score(lab[mask], s[mask])
    return max(a, 1 - a)


# ---------- load ----------
print("Loading cache …")
spec_df, _, _, _ = load_cache(CACHE)
X = np.load(CACHE / "spectra_array_preprocessed.npy")
wn = np.load(CACHE / "wavenumber_axis_preprocessed.npy")
qc = np.load(CACHE / "qc_mask.npy")
spec_df = spec_df.reset_index(drop=True)
spec_df_qc = spec_df[qc].reset_index(drop=True)
Xq = X[qc]
print(f"  spectra: {Xq.shape}  wn: {wn[0]:.1f}-{wn[-1]:.1f}  ({len(wn)} bins)")
print(f"  files: {spec_df_qc['file_id'].nunique()}")
print(f"  pixels-per-file: min={spec_df_qc.groupby('file_id').size().min()}, "
      f"median={spec_df_qc.groupby('file_id').size().median():.0f}, "
      f"max={spec_df_qc.groupby('file_id').size().max()}")


# ---------- compute features ----------
print("\nBuilding feature frame …")
t0 = time.perf_counter()
df_feat = sp.feature_frame_spatial(Xq, wn, spec_df_qc)
elapsed = time.perf_counter() - t0
print(f"  shape: {df_feat.shape}  elapsed: {elapsed:.2f}s")
print(f"  columns: {list(df_feat.columns)}")
print(f"  any NaN? {df_feat.isna().any().any()}")

df_feat.to_parquet(OUT_PARQUET, compression="snappy")
print(f"  wrote {OUT_PARQUET}  ({OUT_PARQUET.stat().st_size / 1024:.1f} KB)")


# ---------- per-class means + d analysis ----------
class_per_file = spec_df_qc.groupby("file_id")["primary_class"].first()
sub_per_file   = spec_df_qc.groupby("file_id")["subclass"].first().fillna("H2O")
fl = df_feat.copy()
fl["primary_class"] = fl.index.map(class_per_file)
fl["subclass"]      = fl.index.map(sub_per_file)

stec    = (fl["primary_class"] == "STEC").values
nstec   = (fl["primary_class"] == "Non-STEC").values
ecoli   = fl["primary_class"].isin(["STEC", "Non-STEC"]).values
salm    = (fl["primary_class"] == "Salmonella").values
h2o     = (fl["primary_class"] == "H2O").values
k12     = (fl["subclass"] == "K-12").values
stec_no_k12 = stec & ~k12

print("\n=== Per-class means ===")
feat_cols = [c for c in fl.columns if c.startswith("spat_")]
class_means = fl.groupby("primary_class")[feat_cols].mean()
print(class_means.round(3).T.to_string())


print("\n=== Spatial feature signal table ===")
rows = []
for c in feat_cols:
    v = fl[c].astype(float).values
    rows.append({
        "feature": c,
        "d_STEC_vs_NSTEC":    cohens_d(v[stec], v[nstec]),
        "d_Ecoli_vs_Salm":    cohens_d(v[ecoli], v[salm]),
        "d_H2O_vs_bact":      cohens_d(v[h2o], v[~h2o]),
        "d_K12_vs_otherSTEC": cohens_d(v[k12], v[stec_no_k12]),
        "auroc_STEC_vs_NSTEC": auroc_either(v[ecoli], stec[ecoli].astype(int)),
    })
sig_df = pd.DataFrame(rows).set_index("feature")
print(sig_df.round(3).to_string())


# ---------- H2O sanity check ----------
print("\n=== H2O sanity (should be LOWER variance/CV than bacteria) ===")
for c in feat_cols:
    if "var" in c or "cv" in c:
        h2o_mean = fl.loc[h2o, c].mean()
        bact_mean = fl.loc[~h2o, c].mean()
        passes = h2o_mean < bact_mean
        print(f"  {c:30s}  H2O={h2o_mean:+.3f}  bact={bact_mean:+.3f}  {'✅' if passes else '❌'}")


# ---------- summary JSON ----------
report = {
    "n_features": len(feat_cols),
    "n_files": int(len(fl)),
    "elapsed_seconds": float(elapsed),
    "per_class_mean": class_means.round(4).to_dict(orient="index"),
    "features": sig_df.round(4).to_dict(orient="index"),
    "k12_subclass_n": int(k12.sum()),
    "other_stec_n": int(stec_no_k12.sum()),
}
(OUT_DIR / "01_stage15e_summary.json").write_text(json.dumps(report, indent=2, default=float))
print(f"\nwrote {OUT_DIR / '01_stage15e_summary.json'}")
print("\n=== Stage 15E build DONE ===")
