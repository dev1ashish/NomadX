"""Stage 15D inspection — file-level Cohen's d for biology features.

One-off: computes per-feature d for STEC vs Non-STEC, E. coli vs Salmonella,
H2O vs bacteria, and K-12 vs other-STEC (PHB falsifier).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/devashishthapliyal/Documents/NomadX")
sys.path.insert(0, str(REPO))

from atlas.io import load_cache
from sklearn.metrics import roc_auc_score

CACHE = REPO / "data_cache"


def cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def auroc_either(scores, labels):
    s = np.asarray(scores, dtype=float)
    lab = np.asarray(labels)
    mask = ~np.isnan(s) & ~np.isnan(lab.astype(float))
    if mask.sum() < 4 or len(set(lab[mask])) < 2:
        return np.nan
    a = roc_auc_score(lab[mask], s[mask])
    return max(a, 1 - a)


spec_df, _, _, _ = load_cache(CACHE)
qc = np.load(CACHE / "qc_mask.npy")
spec_df = spec_df.reset_index(drop=True)
spec_df_qc = spec_df[qc].reset_index(drop=True)
df = pd.read_parquet(CACHE / "band_features.parquet").reset_index(drop=True)
df = df.iloc[: len(spec_df_qc)].copy()
df["file_id"] = spec_df_qc["file_id"].values
df["primary_class"] = spec_df_qc["primary_class"].values
df["subclass"] = spec_df_qc["subclass"].fillna("H2O").values
bio_cols = [c for c in df.columns if c.startswith("bio_")]
print(f"bio cols: {len(bio_cols)}")

fl_num = df.groupby("file_id")[bio_cols].mean()
fl_cls = df.groupby("file_id")["primary_class"].first()
fl_sub = df.groupby("file_id")["subclass"].first()
fl = fl_num.join(fl_cls).join(fl_sub)
print(f"files: {len(fl)}")
print(f"subclass counts: {fl['subclass'].value_counts().to_dict()}")

stec = (fl["primary_class"] == "STEC").values
nstec = (fl["primary_class"] == "Non-STEC").values
ecoli = fl["primary_class"].isin(["STEC", "Non-STEC"]).values
salm = (fl["primary_class"] == "Salmonella").values
h2o = (fl["primary_class"] == "H2O").values
k12 = (fl["subclass"] == "K-12").values
stec_no_k12 = stec & ~k12
print(
    f"STEC files: {stec.sum()},  K-12 STEC: {k12.sum()},  other-STEC: {stec_no_k12.sum()}"
)

rows = []
for c in bio_cols:
    v = fl[c].astype(float).values
    rows.append(
        {
            "feature": c,
            "d_STEC_vs_NSTEC": cohens_d(v[stec], v[nstec]),
            "d_Ecoli_vs_Salm": cohens_d(v[ecoli], v[salm]),
            "d_H2O_vs_bact": cohens_d(
                v[h2o], v[~h2o & (fl["primary_class"] != "H2O")]
            ),
            "d_K12_vs_otherSTEC": cohens_d(v[k12], v[stec_no_k12]),
            "auroc_STEC_vs_NSTEC": auroc_either(v[ecoli], stec[ecoli].astype(int)),
        }
    )
bio_df = pd.DataFrame(rows).set_index("feature")
pd.set_option("display.precision", 3)
print("\n=== Biology feature signal table ===")
print(bio_df.round(3).to_string())

print("\n--- K-12 falsifier (PHB) ---")
k12_phb = fl.loc[k12, "bio_phb_carbonyl"].values
oth_phb = fl.loc[stec_no_k12, "bio_phb_carbonyl"].values
print(
    f"  K-12 phb_carbonyl mean:  {k12_phb.mean():+.3f}  std: {k12_phb.std():.3f}  n={len(k12_phb)}"
)
print(
    f"  other-STEC phb_carbonyl: {oth_phb.mean():+.3f}  std: {oth_phb.std():.3f}  n={len(oth_phb)}"
)
print(f"  d(K-12 vs other-STEC):   {cohens_d(k12_phb, oth_phb):+.3f}")

print("\n--- Virulence AA sig (Trp/Phe) ---")
print(
    f"  STEC      mean: {fl.loc[stec, 'bio_virulence_aa_sig'].mean():+.3f}"
)
print(
    f"  Non-STEC  mean: {fl.loc[nstec, 'bio_virulence_aa_sig'].mean():+.3f}"
)
print(
    f"  d(STEC vs Non-STEC):     {cohens_d(fl.loc[stec,'bio_virulence_aa_sig'].values, fl.loc[nstec,'bio_virulence_aa_sig'].values):+.3f}"
)

print("\n--- Per-class mean of every biology feature ---")
class_means = fl.groupby("primary_class")[bio_cols].mean()
print(class_means.round(3).T.to_string())

out = REPO / "outputs" / "band_chemistry" / "stage15d"
out.mkdir(parents=True, exist_ok=True)
report = {
    "n_features": len(bio_cols),
    "features": bio_df.round(4).to_dict(orient="index"),
    "k12_phb_falsifier": {
        "k12_n": int(k12.sum()),
        "other_stec_n": int(stec_no_k12.sum()),
        "k12_phb_mean": float(k12_phb.mean()),
        "other_stec_phb_mean": float(oth_phb.mean()),
        "d_k12_vs_other_stec": float(cohens_d(k12_phb, oth_phb)),
    },
    "per_class_mean": class_means.round(4).to_dict(orient="index"),
}
(out / "01_stage15d_summary.json").write_text(
    json.dumps(report, indent=2, default=float)
)
print(f"\nwrote {out / '01_stage15d_summary.json'}")
