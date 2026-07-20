"""
Honest replacement for the NotebookLM 'Biology-Grounded Breakthroughs' left panel.

Pulls the REAL bio_alpha_helix_score from the feature cache, aggregates to the
file level (the unit the project actually classifies, and the unit the headline
Cohen's d = -0.986 was computed on), and renders the true distribution with the
real effect size + single-feature AUROC annotated. Shows the genuine overlap the
NotebookLM cartoon erased.

Run: .venv/bin/python FINAL/notebooks/make_honest_alpha_helix.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ROOT = "/Users/devashishthapliyal/Documents/NomadX"
CACHE = os.path.join(ROOT, "data_cache")
OUT = os.path.join(ROOT, "FINAL", "notebooks", "bio_alpha_helix_honest.png")

FEATURE = "bio_alpha_helix_score"

# ---- Load + align (band_features rows == spectra[qc_mask] rows) ----
bf = pd.read_parquet(os.path.join(CACHE, "band_features.parquet")).reset_index(drop=True)
sp = pd.read_parquet(os.path.join(CACHE, "spectra.parquet"))
qc = np.load(os.path.join(CACHE, "qc_mask.npy")).astype(bool)
sp_qc = sp[qc].reset_index(drop=True)

assert len(bf) == len(sp_qc), f"row mismatch: {len(bf)} vs {len(sp_qc)}"

px = pd.DataFrame({
    FEATURE: bf[FEATURE].values,
    "file_id": sp_qc["file_id"].values,
    "primary_class": sp_qc["primary_class"].values,
})
n_px_total = len(px)
px = px.dropna(subset=[FEATURE])
print(f"Pixel-level rows: {n_px_total} total, {len(px)} non-NaN "
      f"({n_px_total - len(px)} NaN dropped)")

# ---- Mean-pool to file level ----
file_lv = (px.groupby(["file_id", "primary_class"])[FEATURE]
             .mean().reset_index())

stec_f = file_lv.loc[file_lv["primary_class"] == "STEC", FEATURE].values
nonstec_f = file_lv.loc[file_lv["primary_class"] == "Non-STEC", FEATURE].values


def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    sp_ = np.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2)
                  / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / sp_ if sp_ > 1e-12 else 0.0


d_file = cohens_d(stec_f, nonstec_f)

# single-feature separability (higher score -> Non-STEC, so label Non-STEC = 1)
y = np.r_[np.zeros(len(stec_f)), np.ones(len(nonstec_f))]
s = np.r_[stec_f, nonstec_f]
auroc = roc_auc_score(y, s)
auroc = max(auroc, 1 - auroc)  # direction-agnostic separability

# overlap: how many STEC files land above the Non-STEC minimum, etc.
overlap_lo = max(stec_f.min(), nonstec_f.min())
overlap_hi = min(stec_f.max(), nonstec_f.max())
stec_in_overlap = int(((stec_f >= overlap_lo) & (stec_f <= overlap_hi)).sum())
nonstec_in_overlap = int(((nonstec_f >= overlap_lo) & (nonstec_f <= overlap_hi)).sum())

print(f"\nFile-level n: STEC={len(stec_f)}, Non-STEC={len(nonstec_f)}")
print(f"STEC    mean={stec_f.mean():.3f}  sd={stec_f.std(ddof=1):.3f}  "
      f"range=[{stec_f.min():.3f}, {stec_f.max():.3f}]")
print(f"NonSTEC mean={nonstec_f.mean():.3f}  sd={nonstec_f.std(ddof=1):.3f}  "
      f"range=[{nonstec_f.min():.3f}, {nonstec_f.max():.3f}]")
print(f"Cohen's d (file-level, STEC - Non-STEC) = {d_file:.3f}  (headline: -0.986)")
print(f"Single-feature AUROC = {auroc:.3f}")
print(f"Overlap band [{overlap_lo:.3f}, {overlap_hi:.3f}]: "
      f"{stec_in_overlap}/{len(stec_f)} STEC files and "
      f"{nonstec_in_overlap}/{len(nonstec_f)} Non-STEC files fall inside it")

# ---- Plot ----
STEC_C = "#E8A33D"      # orange (matches slide)
NONSTEC_C = "#C44FA8"   # magenta/pink (matches slide)
rng = np.random.default_rng(0)

fig, ax = plt.subplots(figsize=(8, 6))

groups = [("STEC", stec_f, STEC_C), ("Non-STEC", nonstec_f, NONSTEC_C)]
for pos, (name, vals, color) in enumerate(groups):
    vp = ax.violinplot(vals, positions=[pos], widths=0.7, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor(color)
        body.set_alpha(0.25)
        body.set_edgecolor(color)
    # individual files as jittered dots (so the viewer SEES how few there are)
    jit = rng.uniform(-0.10, 0.10, size=len(vals))
    ax.scatter(pos + jit, vals, color=color, s=42, alpha=0.85,
               edgecolors="white", linewidths=0.6, zorder=3)
    # mean bar
    ax.hlines(vals.mean(), pos - 0.22, pos + 0.22, color="black",
              linewidth=2.6, zorder=4)
    ax.text(pos, vals.max() + 0.15, f"n={len(vals)} files",
            ha="center", va="bottom", fontsize=10, color=color, fontweight="bold")

# shade the true overlap band
ax.axhspan(overlap_lo, overlap_hi, color="grey", alpha=0.10, zorder=0)
ax.text(1.55, (overlap_lo + overlap_hi) / 2,
        f"overlap zone\n{stec_in_overlap}/{len(stec_f)} STEC +\n"
        f"{nonstec_in_overlap}/{len(nonstec_f)} Non-STEC",
        ha="center", va="center", fontsize=8.5, color="dimgrey",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.7))

ax.set_xticks([0, 1])
ax.set_xticklabels(["STEC", "Non-STEC"], fontsize=12)
ax.set_ylabel("bio_alpha_helix_score  (per-file mean)", fontsize=11)
ax.set_title(
    "bio_alpha_helix_score — REAL file-level distribution\n"
    f"Cohen's d = {d_file:.2f}  |  single-feature AUROC = {auroc:.2f}  "
    f"|  black bar = mean",
    fontsize=11)
ax.set_xlim(-0.6, 2.1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, axis="y", linestyle=":", alpha=0.4)

cap = ("Non-STEC commensals carry more amide-I alpha-helix signal on average than "
       "pathogenic STEC\n(they lack the beta-sheet-heavy virulence proteins). A real, "
       "large effect (|d|~1) — but the\ngroups OVERLAP substantially, which is why this "
       "single feature cannot cleanly classify a\nheld-out strain. Honest separation, not "
       "the near-perfect split of the illustrative slide.")
fig.text(0.5, -0.02, cap, ha="center", va="top", fontsize=8.5, color="#333333")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUT}")
