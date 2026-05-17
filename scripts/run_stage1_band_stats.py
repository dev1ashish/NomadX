"""Stage 1 of plan/14 — bacteria-only ANOVA + primary-triple effect sizes.

Pre-registered in plan/08_expectations.md (2026-05-17 entry).

Inputs (all from data_cache/):
  spectra_array_preprocessed.npy   — (N_qc, 987) float32, SNV + baseline-corrected
  wavenumber_axis_preprocessed.npy — (987,) float32
  qc_mask.npy                      — (N_total,) bool
  spectra.parquet                  — row-aligned metadata pre-QC
  metadata.parquet                 — per-file metadata

Outputs:
  outputs/band_chemistry/01_bacteria_only_anova_top30.csv
  outputs/band_chemistry/01_bacteria_only_anova.png
  outputs/band_chemistry/02_primary_triple_stats.csv
  outputs/band_chemistry/02_primary_triple_violin.png
  outputs/band_chemistry/02_primary_triple_per_strain.png
  outputs/band_chemistry/03_stage1_summary.md
"""
from __future__ import annotations
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu
from sklearn.metrics import roc_auc_score

REPO = Path("/Users/devashishthapliyal/Documents/NomadX")
sys.path.insert(0, str(REPO))

from atlas.io import load_cache

CACHE = REPO / "data_cache"
OUT = REPO / "outputs" / "band_chemistry"
OUT.mkdir(parents=True, exist_ok=True)

PRIMARY_TRIPLE = [
    (1338, "1338  (NA, stx)",    "nucleic_acid"),
    (1454, "1454  (lipid+carb)", "lipid_carb"),
    (1658, "1658  (amide-I)",    "protein_amide"),
]
TRIPLE_HALF_WIDTH = 10.0   # ±10 cm⁻¹

# Style
CLASS_COLORS = {"STEC": "#d62728", "Non-STEC": "#1f77b4",
                "Salmonella": "#2ca02c", "H2O": "#7f7f7f"}
SUBCLASS_COLORS = {
    "O103H2": "#d62728", "O121H19": "#e7665a", "O157H7": "#a02124",
    "83972": "#1f77b4", "ATCC25922": "#5fa6d8", "K-12": "#114573",
    "Dublin": "#2ca02c", "Heidelburg": "#73c073", "Typhimurium": "#175f17",
}
STEC_SUBS = ["O157H7", "O121H19", "O103H2"]
NONSTEC_SUBS = ["ATCC25922", "K-12", "83972"]
SALMO_SUBS = ["Typhimurium", "Heidelburg", "Dublin"]


# ---------- load ----------
print("Loading cache …")
spec_df, _X_raw, _wn_raw, meta = load_cache(CACHE)
X = np.load(CACHE / "spectra_array_preprocessed.npy")
wn = np.load(CACHE / "wavenumber_axis_preprocessed.npy")
qc = np.load(CACHE / "qc_mask.npy")

spec_df = spec_df.reset_index(drop=True)
spec_df = spec_df[qc].reset_index(drop=True)
X = X[qc]
print(f"  QC-passed spectra: {len(X)}")
print(f"  wn range: {wn[0]:.1f} – {wn[-1]:.1f}  ({len(wn)} bins)")
print(f"  classes: {spec_df['primary_class'].value_counts().to_dict()}")


# ---------- RQ4: bacteria-only ANOVA ----------
print("\n=== RQ4: bacteria-only ANOVA (H₂O excluded) ===")
bact_mask = spec_df["primary_class"].values != "H2O"
Xb = X[bact_mask]
yb = spec_df.loc[bact_mask, "primary_class"].values
print(f"  bacterial spectra: {len(Xb)}; classes: {pd.Series(yb).value_counts().to_dict()}")

# Per-bin one-way ANOVA across 3 bacterial classes
groups = [Xb[yb == c] for c in ["STEC", "Non-STEC", "Salmonella"]]
F = np.empty(Xb.shape[1])
pvals = np.empty(Xb.shape[1])
for j in range(Xb.shape[1]):
    F[j], pvals[j] = f_oneway(groups[0][:, j], groups[1][:, j], groups[2][:, j])

top_idx = np.argsort(F)[::-1][:30]
top_df = pd.DataFrame({
    "rank": np.arange(1, 31),
    "wn_cm-1": np.round(wn[top_idx], 1),
    "F": np.round(F[top_idx], 2),
    "log10F": np.round(np.log10(np.maximum(F[top_idx], 1e-3)), 3),
    "p": pvals[top_idx],
})
print("Top 10 bacterial-discriminative bins:")
print(top_df.head(10).to_string(index=False))
top_df.to_csv(OUT / "01_bacteria_only_anova_top30.csv", index=False)

# Verdict booleans
top_wn = wn[top_idx]
fingerprint_count = int(((top_wn >= 1100) & (top_wn <= 1750)).sum())
ch_count = int(((top_wn >= 2800) & (top_wn <= 3000)).sum())
lps_count = int(((top_wn >= 800) & (top_wn <= 1200)).sum())

def _band_nearby(top_wn, center, tol=15):
    return any(abs(top_wn - center) <= tol)

triple_hits = {
    1338: _band_nearby(top_wn, 1338),
    1454: _band_nearby(top_wn, 1454),
    1658: _band_nearby(top_wn, 1658),
}
n_triple = sum(triple_hits.values())
print(f"\nTop-30 region distribution:")
print(f"  fingerprint (1100–1750): {fingerprint_count}/30")
print(f"  C-H stretch (2800–3000): {ch_count}/30")
print(f"  LPS (800–1200):          {lps_count}/30")
print(f"  primary triple {{1338,1454,1658}} hits (±15 cm⁻¹): {triple_hits}  → {n_triple}/3")


# ANOVA visualization
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
axes[0].plot(wn, np.log10(np.maximum(F, 1e-3)), color="#1f77b4", lw=0.9)
axes[0].set_ylabel("log10(F), bacteria-only ANOVA")
axes[0].set_title("Bacteria-only per-bin discriminative power (STEC vs Non-STEC vs Salmonella, H₂O excluded)")
axes[0].grid(alpha=0.3)

# Mark primary triple
for center, label, _ in PRIMARY_TRIPLE:
    axes[0].axvline(center, color="#d55e00", lw=1, alpha=0.7, ls="--")
    axes[0].text(center, axes[0].get_ylim()[1] * 0.95, f"{center}*",
                 rotation=90, ha="right", va="top", fontsize=9,
                 color="#d55e00", fontweight="bold")

# Histogram of top-30 wavenumbers
axes[1].hist(wn[top_idx], bins=np.arange(400, 3100, 100), color="#888", edgecolor="white")
axes[1].axvspan(800, 1200, color="#1f77b4", alpha=0.08, label="LPS region")
axes[1].axvspan(1100, 1750, color="#2ca02c", alpha=0.08, label="fingerprint")
axes[1].axvspan(2800, 3000, color="#d62728", alpha=0.08, label="C-H stretch")
for center, _, _ in PRIMARY_TRIPLE:
    axes[1].axvline(center, color="#d55e00", lw=1, alpha=0.7, ls="--")
axes[1].set_xlabel("Wavenumber (cm⁻¹)")
axes[1].set_ylabel("# of top-30 bins")
axes[1].set_title("Distribution of top-30 bacteria-only ANOVA bins")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "01_bacteria_only_anova.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  wrote {OUT / '01_bacteria_only_anova.png'}")


# ---------- RQ4 follow-up: E. coli-only ANOVA (STEC vs Non-STEC only) ----------
# The 3-class test above is dominated by E. coli vs Salmonella signal; the
# primary triple is a 2-class within-E. coli discriminator and needs a 2-class test.
print("\n=== RQ4 follow-up: E. coli only (STEC vs Non-STEC) per-bin t-test ===")
ecoli_mask = np.isin(spec_df["primary_class"].values, ["STEC", "Non-STEC"])
Xe = X[ecoli_mask]
ye = (spec_df.loc[ecoli_mask, "primary_class"].values == "STEC").astype(int)
print(f"  E. coli spectra: {len(Xe)}  ({int(ye.sum())} STEC + {int((1-ye).sum())} Non-STEC)")

t_ec = np.empty(Xe.shape[1])
p_ec = np.empty(Xe.shape[1])
for j in range(Xe.shape[1]):
    t_ec[j], p_ec[j] = ttest_ind(Xe[ye == 1, j], Xe[ye == 0, j], equal_var=False)
abs_t = np.abs(t_ec)
top_ec_idx = np.argsort(abs_t)[::-1][:30]
top_ec_df = pd.DataFrame({
    "rank": np.arange(1, 31),
    "wn_cm-1": np.round(wn[top_ec_idx], 1),
    "abs_t": np.round(abs_t[top_ec_idx], 2),
    "sign": np.where(t_ec[top_ec_idx] > 0, "STEC>", "STEC<"),
    "p": p_ec[top_ec_idx],
})
print("Top 10 E. coli-only discriminative bins:")
print(top_ec_df.head(10).to_string(index=False))
top_ec_df.to_csv(OUT / "01b_ecoli_only_ttest_top30.csv", index=False)

top_ec_wn = wn[top_ec_idx]
ec_triple_hits = {
    1338: _band_nearby(top_ec_wn, 1338),
    1454: _band_nearby(top_ec_wn, 1454),
    1658: _band_nearby(top_ec_wn, 1658),
}
ec_n_triple = sum(ec_triple_hits.values())
ec_fingerprint = int(((top_ec_wn >= 1100) & (top_ec_wn <= 1750)).sum())
ec_lps = int(((top_ec_wn >= 800) & (top_ec_wn <= 1200)).sum())
print(f"\nE. coli-only top-30 distribution:")
print(f"  fingerprint (1100–1750): {ec_fingerprint}/30")
print(f"  LPS (800–1200):          {ec_lps}/30")
print(f"  primary triple {{1338,1454,1658}} hits (±15 cm⁻¹): {ec_triple_hits}  → {ec_n_triple}/3")


# ---------- RQ1: STEC vs Non-STEC at primary triple ----------
print("\n=== RQ1: STEC vs Non-STEC at 1338 / 1454 / 1658 ===")

def integrate_band(X, wn, center, half_width=TRIPLE_HALF_WIDTH):
    m = (wn >= center - half_width) & (wn <= center + half_width)
    # Trapezoidal AUC
    return np.trapz(X[:, m], wn[m], axis=1)

def cohens_d(a, b):
    pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else 0.0

stec_mask_spec = spec_df["primary_class"].values == "STEC"
nstec_mask_spec = spec_df["primary_class"].values == "Non-STEC"

# Per-file mean AUC (file-level test) — true independence
file_means_by_band = {}
for center, label, group in PRIMARY_TRIPLE:
    auc = integrate_band(X, wn, center)
    spec_df_tmp = spec_df.assign(_auc=auc)
    per_file = spec_df_tmp.groupby("file_id").agg(
        auc=("_auc", "mean"),
        pc=("primary_class", "first"),
    )
    file_means_by_band[center] = per_file

# Compute stats per band
rows = []
band_aucs_spec = {}   # spectrum-level for plots
band_aucs_file = {}   # file-level for plots
for center, label, group in PRIMARY_TRIPLE:
    # Spectrum-level
    auc = integrate_band(X, wn, center)
    a_spec = auc[stec_mask_spec]
    b_spec = auc[nstec_mask_spec]
    d_spec = cohens_d(a_spec, b_spec)
    t_spec, p_spec_t = ttest_ind(a_spec, b_spec, equal_var=False)
    u_spec, p_spec_u = mannwhitneyu(a_spec, b_spec, alternative="two-sided")

    # File-level (independent units)
    pf = file_means_by_band[center]
    a_file = pf.loc[pf["pc"] == "STEC", "auc"].values
    b_file = pf.loc[pf["pc"] == "Non-STEC", "auc"].values
    d_file = cohens_d(a_file, b_file)
    t_file, p_file_t = ttest_ind(a_file, b_file, equal_var=False)
    u_file, p_file_u = mannwhitneyu(a_file, b_file, alternative="two-sided")

    # AUROC for STEC vs Non-STEC on file-level
    y_file = np.r_[np.ones(len(a_file)), np.zeros(len(b_file))]
    s_file = np.r_[a_file, b_file]
    auroc_file = roc_auc_score(y_file, s_file)
    auroc_file = max(auroc_file, 1 - auroc_file)  # report as ≥0.5

    rows.append({
        "band_cm-1": center,
        "macromolecule": group,
        "n_stec_files": len(a_file), "n_nstec_files": len(b_file),
        "mean_stec_file": a_file.mean(), "mean_nstec_file": b_file.mean(),
        "shift_sign": "STEC>Non-STEC" if a_file.mean() > b_file.mean() else "STEC<Non-STEC",
        "cohen_d_file": d_file,
        "welch_t_file": t_file, "welch_p_file": p_file_t,
        "mannwhitney_u_file": u_file, "mannwhitney_p_file": p_file_u,
        "auroc_file": auroc_file,
        "cohen_d_spec": d_spec,
        "welch_p_spec": p_spec_t,
        "mannwhitney_p_spec": p_spec_u,
        "n_stec_spec": int(stec_mask_spec.sum()),
        "n_nstec_spec": int(nstec_mask_spec.sum()),
    })
    band_aucs_spec[center] = (a_spec, b_spec)
    band_aucs_file[center] = pf

triple_df = pd.DataFrame(rows)
print(triple_df[["band_cm-1", "shift_sign", "cohen_d_file", "welch_p_file",
                 "mannwhitney_p_file", "auroc_file", "cohen_d_spec",
                 "welch_p_spec"]].to_string(index=False))
triple_df.to_csv(OUT / "02_primary_triple_stats.csv", index=False)

# Branching verdict
n_clear = int(((triple_df["welch_p_file"] < 0.05) & (triple_df["cohen_d_file"].abs() >= 0.3)).sum())
if n_clear == 3:
    branch = "A"
    branch_text = "ALL 3 bands cleared: published bands confirmed on this dataset."
elif n_clear in (1, 2):
    branch = "B"
    branch_text = f"{n_clear}/3 bands cleared: partial confirmation."
else:
    branch = "C"
    branch_text = "0/3 bands cleared at file-level: published bands may not be load-bearing here. RE-ANCHOR ON BACTERIA-ONLY ANOVA TOP-30."
print(f"\nBranching verdict (file-level p<0.05 AND |d|≥0.3): branch ({branch}) — {branch_text}")


# ---------- Plots: per-band STEC vs Non-STEC ----------
print("\nMaking primary-triple violin plot …")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for col, (center, label, group) in enumerate(PRIMARY_TRIPLE):
    # Spectrum-level violin (informational)
    ax = axes[0, col]
    a_spec, b_spec = band_aucs_spec[center]
    parts = ax.violinplot([a_spec, b_spec], positions=[0, 1], showmeans=True,
                          widths=0.7)
    for body, color in zip(parts["bodies"], [CLASS_COLORS["STEC"], CLASS_COLORS["Non-STEC"]]):
        body.set_facecolor(color); body.set_alpha(0.55); body.set_edgecolor("black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["STEC", "Non-STEC"])
    ax.set_title(f"{label}\n(spectrum-level, n≈4600)", fontsize=10)
    ax.set_ylabel("Integrated AUC (±10 cm⁻¹)")
    row = triple_df.iloc[col]
    ax.text(0.5, 1.02,
            f"d={row['cohen_d_spec']:.2f}  p={row['welch_p_spec']:.1e}",
            transform=ax.transAxes, ha="center", fontsize=9, color="#444")
    ax.grid(alpha=0.3)

    # File-level strip + box (the real evidence bar)
    ax = axes[1, col]
    pf = band_aucs_file[center]
    a_file = pf.loc[pf["pc"] == "STEC", "auc"].values
    b_file = pf.loc[pf["pc"] == "Non-STEC", "auc"].values
    for i, (vals, color) in enumerate(zip([a_file, b_file],
                                          [CLASS_COLORS["STEC"], CLASS_COLORS["Non-STEC"]])):
        ax.boxplot([vals], positions=[i], widths=0.4, showfliers=False,
                   patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.35, edgecolor="black"),
                   medianprops=dict(color="black", lw=1.5))
        jitter = np.random.default_rng(0).normal(i, 0.05, size=len(vals))
        ax.scatter(jitter, vals, s=22, color=color, edgecolor="black", lw=0.4, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["STEC", "Non-STEC"])
    ax.set_title(f"{label}\n(file-level, n=27 STEC vs 25 Non-STEC)", fontsize=10)
    ax.set_ylabel("Integrated AUC (file mean)")
    row = triple_df.iloc[col]
    cleared = row["welch_p_file"] < 0.05 and abs(row["cohen_d_file"]) >= 0.3
    badge = "✅ CLEARED" if cleared else "❌ not cleared"
    ax.text(0.5, 1.02,
            f"d={row['cohen_d_file']:.2f}  p={row['welch_p_file']:.3f}  AUROC={row['auroc_file']:.2f}  {badge}",
            transform=ax.transAxes, ha="center", fontsize=9,
            color="#2ca02c" if cleared else "#d62728")
    ax.grid(alpha=0.3)

fig.suptitle(f"Primary STEC↔non-STEC triple: 1338 / 1454 / 1658 cm⁻¹   —   branch ({branch})",
             fontsize=12, y=1.01, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "02_primary_triple_violin.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  wrote {OUT / '02_primary_triple_violin.png'}")


# ---------- Per-strain breakdown plot ----------
print("Making per-strain breakdown …")
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=False)
all_subs = STEC_SUBS + NONSTEC_SUBS + SALMO_SUBS
positions = np.arange(len(all_subs))

for ax, (center, label, group) in zip(axes, PRIMARY_TRIPLE):
    pf = band_aucs_file[center]
    # Join subclass via file metadata
    sub_of_file = meta.set_index("file_id")["subclass"].to_dict()
    pf = pf.assign(subclass=[sub_of_file.get(fid) for fid in pf.index])
    for i, sub in enumerate(all_subs):
        vals = pf.loc[pf["subclass"] == sub, "auc"].values
        if len(vals) == 0:
            continue
        color = SUBCLASS_COLORS[sub]
        ax.boxplot([vals], positions=[i], widths=0.55, showfliers=False,
                   patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.40, edgecolor="black"),
                   medianprops=dict(color="black", lw=1.4))
        rng = np.random.default_rng(i + int(center))
        jitter = rng.normal(i, 0.06, size=len(vals))
        ax.scatter(jitter, vals, s=20, color=color, edgecolor="black", lw=0.3, zorder=3)

    # Vertical separators between class blocks
    ax.axvline(2.5, color="#444", lw=0.7, ls="--", alpha=0.4)
    ax.axvline(5.5, color="#444", lw=0.7, ls="--", alpha=0.4)
    # Class labels using axis-fraction coordinates (independent of data y-limits)
    ax.text(1, 1.02, "STEC", ha="center", fontsize=9,
            color=CLASS_COLORS["STEC"], fontweight="bold",
            transform=ax.get_xaxis_transform())
    ax.text(4, 1.02, "Non-STEC", ha="center", fontsize=9,
            color=CLASS_COLORS["Non-STEC"], fontweight="bold",
            transform=ax.get_xaxis_transform())
    ax.text(7, 1.02, "Salmonella", ha="center", fontsize=9,
            color=CLASS_COLORS["Salmonella"], fontweight="bold",
            transform=ax.get_xaxis_transform())

    ax.set_xticks(positions); ax.set_xticklabels(all_subs, rotation=45, ha="right")
    ax.set_title(label, fontsize=11)
    ax.set_ylabel("Integrated AUC (file mean)")
    ax.grid(alpha=0.3)

fig.suptitle("Per-strain breakdown of primary-triple AUC (file-level)",
             fontsize=12, y=1.02, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "02_primary_triple_per_strain.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  wrote {OUT / '02_primary_triple_per_strain.png'}")


# ---------- Markdown summary ----------
summary_path = OUT / "03_stage1_summary.md"
with summary_path.open("w") as f:
    f.write(f"# Stage 1 results — bacteria-only ANOVA + primary triple\n\n")
    f.write(f"_Generated by `scripts/run_stage1_band_stats.py`._\n\n")
    f.write(f"## RQ4 — Bacteria-only ANOVA top-30 region distribution\n\n")
    f.write(f"| Region | Count in top-30 |\n|---|---|\n")
    f.write(f"| Fingerprint (1100–1750 cm⁻¹) | {fingerprint_count}/30 |\n")
    f.write(f"| C-H stretch (2800–3000)      | {ch_count}/30 |\n")
    f.write(f"| LPS (800–1200)               | {lps_count}/30 |\n")
    f.write(f"| Primary triple hits (±15 cm⁻¹) | {n_triple}/3 → {triple_hits} |\n\n")
    f.write(f"Top-10 ANOVA-F bins:\n\n")
    f.write(top_df.head(10).to_string(index=False))
    f.write("\n\n")
    f.write(f"## RQ1 — STEC vs Non-STEC at primary triple\n\n")
    f.write(f"**Branch ({branch})** — {branch_text}\n\n")
    f.write(triple_df[["band_cm-1", "shift_sign", "cohen_d_file", "welch_p_file",
                       "mannwhitney_p_file", "auroc_file", "cohen_d_spec"]].to_string(index=False))
    f.write("\n")
print(f"  wrote {summary_path}")

# Machine-readable dump for findings-append step
json_dump = {
    "branch": branch,
    "triple_hits_in_3class_anova_top30": triple_hits,
    "n_triple_in_3class_top30": n_triple,
    "top30_region_distribution_3class": {
        "fingerprint_1100_1750": fingerprint_count,
        "ch_stretch_2800_3000": ch_count,
        "lps_800_1200": lps_count,
    },
    "top30_bins_3class": top_df.head(30).to_dict(orient="records"),
    "triple_hits_in_ecoli_only_top30": ec_triple_hits,
    "n_triple_in_ecoli_top30": ec_n_triple,
    "top30_region_distribution_ecoli": {
        "fingerprint_1100_1750": ec_fingerprint,
        "lps_800_1200": ec_lps,
    },
    "top30_bins_ecoli": top_ec_df.head(30).to_dict(orient="records"),
    "primary_triple_stats": triple_df.to_dict(orient="records"),
}
with (OUT / "03_stage1_summary.json").open("w") as f:
    json.dump(json_dump, f, indent=2, default=float)

print("\n=== Stage 1 DONE ===")
