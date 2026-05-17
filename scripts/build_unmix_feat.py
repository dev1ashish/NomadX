"""Build data_cache/unmix_features.parquet (Stage 15C — MCR-ALS unmixing).

Pre-registered in plan/experiments/2026-05-18_stage15c_mcr_als_unmixing.md.

Fits MCR-ALS K=8 globally on the 7,122 QC-passed preprocessed spectra (SNV'd,
shifted to ≥0 for non-negativity), aggregates per-pixel concentrations into
file-level features, and dumps inspection artifacts:

  - data_cache/unmix_features.parquet                       (87 × 4·K cols)
  - outputs/band_chemistry/stage15c/pure_spectra.npy        (K × B)
  - outputs/band_chemistry/stage15c/concentrations.npy      (N_qc × K)
  - outputs/band_chemistry/stage15c/pure_spectra.png        (component overlay)
  - outputs/band_chemistry/stage15c/per_class_heatmap.png   (4 × K)
  - outputs/band_chemistry/stage15c/01_stage15c_summary.json

> **R2 caveat:** this is a global fit for feature exploration only. The Stage
> 15F LOSO classifier MUST refit per fold via ``atlas.unmix_features.MCRALSWrapper``.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd

REPO = Path("/Users/devashishthapliyal/Documents/NomadX")
sys.path.insert(0, str(REPO))

from atlas.io import load_cache
from atlas import unmix_features as uf
from atlas.band_features import BANDS

CACHE = REPO / "data_cache"
OUT = REPO / "outputs" / "band_chemistry" / "stage15c"
OUT.mkdir(parents=True, exist_ok=True)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


# ---------- load ----------
print("Loading cache …")
spec_df_full, _X_raw, _wn_raw, _meta = load_cache(CACHE)
X = np.load(CACHE / "spectra_array_preprocessed.npy")
wn = np.load(CACHE / "wavenumber_axis_preprocessed.npy")
qc = np.load(CACHE / "qc_mask.npy")
spec_df = spec_df_full.reset_index(drop=True)
spec_df_qc = spec_df[qc].reset_index(drop=True)
Xq_full = X[qc]
file_ids = spec_df_qc["file_id"].values
print(f"  spectra: {Xq_full.shape}  wn: {wn[0]:.1f}-{wn[-1]:.1f}  ({len(wn)} bins)")
print(f"  files:   {len(set(file_ids))}")
print(f"  classes: {spec_df_qc['primary_class'].value_counts().to_dict()}")

# ---------- restrict to biology-informative range, then offset ----------
# Crop FIRST then offset by region-min. Run #1 (full 400-3050 + global X.min)
# produced 6 of 8 components dominated by 470-550 cm⁻¹ edge artifacts because
# SNV's most-negative cells live in that low-wn substrate region — global
# shifting compressed them into a synthetic peak. Cropping first excludes
# those bins entirely, so the shift no longer fabricates an edge bump.
WN_LO, WN_HI = 600.0, 1800.0
region_mask = (wn >= WN_LO) & (wn <= WN_HI)
wn_used = wn[region_mask]
Xq = Xq_full[:, region_mask]
print(f"  restricted to [{WN_LO:.0f}, {WN_HI:.0f}] cm⁻¹  ({len(wn_used)} bins)")
xmin = float(Xq.min())
xmax = float(Xq.max())
print(f"  region range: [{xmin:.3f}, {xmax:.3f}]")
Xshift = Xq - xmin
print(f"  after shift by -region_min: [{Xshift.min():.3f}, {Xshift.max():.3f}]")
n_clipped = 0  # no clipping in this path
# Use wn_used downstream
wn = wn_used

# ---------- fit MCR-ALS K=8 globally ----------
N_COMPONENTS = 8
print(f"\nFitting MCR-ALS K={N_COMPONENTS} on {Xshift.shape[0]} spectra …")
t0 = time.perf_counter()
wrap = uf.MCRALSWrapper(
    n_components=N_COMPONENTS,
    max_iter=200,
    tol_err_change=1e-8,
    offset_pct=5.0,
    # ConstraintNorm divided by zero on the cropped fingerprint data (one ST
    # row collapsed to all zeros mid-iteration). Disabled — per-file features
    # care about C patterns, not absolute scale.
    normalize_spectra=False,
    random_state=42,
)
wrap.fit(Xshift)
elapsed = time.perf_counter() - t0
print(f"  converged in {wrap.n_iter} iterations  ({elapsed:.1f}s)")
print(f"  pure_var_idx wn: {wn[wrap.pure_var_idx]}")
print(f"  ST shape: {wrap.pure_spectra.shape}  C shape: {wrap.concentrations.shape}")

ST = wrap.pure_spectra                  # (K, B)
C = wrap.concentrations                 # (N, K)
err_history = wrap._result.err_history

np.save(OUT / "pure_spectra.npy", ST)
np.save(OUT / "concentrations.npy", C)
print(f"  wrote {OUT}/pure_spectra.npy  {OUT}/concentrations.npy")


# ---------- per-file feature summary ----------
print("\nBuilding per-file feature summary …")
df_feat = uf.mcr_concentration_summary(C, file_ids)
print(f"  shape: {df_feat.shape}  (87 files × {df_feat.shape[1]} features)")

# Sanity: residual norm per file
D_calc = C @ ST                         # (N, B)
resid = Xshift - D_calc
resid_norm_per_pixel = np.linalg.norm(resid, axis=1)
resid_df = pd.DataFrame({
    "file_id": file_ids,
    "_resid": resid_norm_per_pixel,
}).groupby("file_id")["_resid"].mean()
df_feat["mcr_residual_norm_mean"] = df_feat.index.map(resid_df)

df_feat.to_parquet(CACHE / "unmix_features.parquet", compression="snappy")
print(f"  wrote {CACHE / 'unmix_features.parquet'}  "
      f"({(CACHE / 'unmix_features.parquet').stat().st_size / 1024:.1f} KB)")


# ---------- per-class concentration mean (4 × K heatmap data) ----------
joined = pd.DataFrame(C, columns=[f"C{k+1}" for k in range(N_COMPONENTS)])
joined["primary_class"] = spec_df_qc["primary_class"].values
joined["file_id"] = file_ids

file_means = joined.groupby("file_id").agg("mean", numeric_only=True)
file_means["primary_class"] = joined.groupby("file_id")["primary_class"].first()

class_means_4xK = file_means.groupby("primary_class").agg("mean", numeric_only=True)
class_means_4xK = class_means_4xK[[f"C{k+1}" for k in range(N_COMPONENTS)]]
print("\nMean concentration per class (4 × K):")
print(class_means_4xK.round(3))


# ---------- file-level Cohen's d per component-feature ----------
fl = df_feat.copy()
# Attach primary_class to file_id index
class_per_file = spec_df_qc.groupby("file_id")["primary_class"].first()
fl["primary_class"] = fl.index.map(class_per_file)

stec = (fl["primary_class"] == "STEC").values
nstec = (fl["primary_class"] == "Non-STEC").values
ecoli = fl["primary_class"].isin(["STEC", "Non-STEC"]).values
salm = (fl["primary_class"] == "Salmonella").values

print("\nFile-level Cohen's d per MCR feature:")
feat_cols = [c for c in fl.columns if c.startswith("mcr_C")]
rows_d = []
for col in feat_cols:
    vals = fl[col].astype(float).values
    d_stec = cohens_d(vals[stec], vals[nstec])
    d_es = cohens_d(vals[ecoli], vals[salm])
    rows_d.append((col, d_stec, d_es))
rows_d.sort(key=lambda r: -abs(r[1]))
print(f"  {'feature':<22s} {'d(STEC↔Non-STEC)':>20s} {'d(E.coli↔Salm)':>20s}")
for col, d_sn, d_es in rows_d[:15]:
    print(f"  {col:<22s} {d_sn:>+20.3f} {d_es:>+20.3f}")


# ---------- inspection figures ----------
print("\nWriting inspection figures …")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pure spectra overlay
fig, axes = plt.subplots(N_COMPONENTS, 1, figsize=(11, 1.7 * N_COMPONENTS), sharex=True)
band_centers_to_annotate = sorted({int(b["center"]) for b in BANDS.values()})
for k, ax in enumerate(axes):
    ax.plot(wn, ST[k], lw=1.0, color="C0")
    ax.set_title(f"Component {k+1} (purest var = {wn[wrap.pure_var_idx[k]]:.0f} cm⁻¹)", fontsize=9)
    ax.set_ylabel("S^T")
    ax.grid(alpha=0.3)
    # Annotate a few key band centers
    yl = ax.get_ylim()
    for c in [1004, 1450, 1660, 752, 1100, 1090, 815, 1730]:
        if wn[0] <= c <= wn[-1]:
            ax.axvline(c, color="gray", alpha=0.25, lw=0.6)
            ax.text(c, yl[1] * 0.85, str(c), rotation=90, fontsize=6, color="gray", ha="center")
axes[-1].set_xlabel("Wavenumber (cm⁻¹)")
fig.suptitle(f"MCR-ALS K={N_COMPONENTS} pure spectra (global fit, n_iter={wrap.n_iter})", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig(OUT / "pure_spectra.png", dpi=130)
plt.close(fig)
print(f"  wrote {OUT / 'pure_spectra.png'}")

# Per-class concentration heatmap
fig, ax = plt.subplots(figsize=(7, 3.2))
hmap = class_means_4xK.values
im = ax.imshow(hmap, aspect="auto", cmap="viridis")
ax.set_yticks(range(class_means_4xK.shape[0]))
ax.set_yticklabels(class_means_4xK.index.tolist())
ax.set_xticks(range(N_COMPONENTS))
ax.set_xticklabels([f"C{k+1}" for k in range(N_COMPONENTS)])
ax.set_title("MCR-ALS mean concentration per file, per primary class")
for i in range(hmap.shape[0]):
    for j in range(hmap.shape[1]):
        ax.text(j, i, f"{hmap[i, j]:.2f}", ha="center", va="center",
                color="white" if hmap[i, j] < hmap.mean() else "black",
                fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.85, label="mean(C)")
fig.tight_layout()
fig.savefig(OUT / "per_class_heatmap.png", dpi=130)
plt.close(fig)
print(f"  wrote {OUT / 'per_class_heatmap.png'}")

# Convergence trace
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(err_history, "-o", ms=3, lw=1)
ax.set_yscale("log")
ax.set_xlabel("Half-iteration (C / ST alternation)")
ax.set_ylabel("MSE")
ax.set_title(f"MCR-ALS convergence (n_iter={wrap.n_iter})")
ax.grid(alpha=0.4)
fig.tight_layout()
fig.savefig(OUT / "convergence.png", dpi=130)
plt.close(fig)
print(f"  wrote {OUT / 'convergence.png'}")


# ---------- summary JSON ----------
def _safe(v):
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    return v


report = {
    "n_components": N_COMPONENTS,
    "n_spectra": int(Xshift.shape[0]),
    "n_bins": int(Xshift.shape[1]),
    "n_files": int(len(set(file_ids))),
    "n_iter": int(wrap.n_iter),
    "final_err": float(err_history[-1]) if err_history else None,
    "preprocessing": (
        f"spectra_array_preprocessed.npy (arPLS+SG+SNV) cropped to wn in "
        f"[{WN_LO:.0f}, {WN_HI:.0f}] cm⁻¹, then shifted by -region_min"
    ),
    "wn_lo": float(WN_LO),
    "wn_hi": float(WN_HI),
    "region_min_shift": float(-xmin),
    "pure_var_idx": wrap.pure_var_idx.tolist(),
    "pure_var_wn": [float(wn[i]) for i in wrap.pure_var_idx],
    "elapsed_seconds": float(elapsed),
    "per_class_mean_concentration": class_means_4xK.to_dict(orient="index"),
    "top_features_stec_vs_nstec": [
        {"feature": col, "d_stec_vs_nstec": _safe(d_sn), "d_ecoli_vs_salm": _safe(d_es)}
        for col, d_sn, d_es in rows_d[:15]
    ],
    "cache_columns": int(df_feat.shape[1]),
    "cache_rows": int(df_feat.shape[0]),
}
with (OUT / "01_stage15c_summary.json").open("w") as f:
    json.dump(report, f, indent=2, default=float)
print(f"\nwrote {OUT / '01_stage15c_summary.json'}")
print("\n=== Stage 15C build DONE ===")
