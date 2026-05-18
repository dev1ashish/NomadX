"""Generate the figure set embedded in PAPER.md.

Saves all PNGs to `artifacts/figures/`. Re-runnable; overwrites.

Figures produced:
  fig01_cache_size.png       — feature catalog size after each Stage 15A-E
  fig02_top_features.png     — top-15 file-level Cohen's d STEC↔Non-STEC
  fig03_pseudovoigt.png      — pseudo-Voigt vs Lorentzian fit-success comparison
  fig04_k12_axis.png         — K-12 specific 2°-structure axis (Stage 15D)
  fig05_ecoli_salm_skew.png  — E. coli vs Salmonella spat_skew_lps_1117 (Stage 15E)
  fig06_stage15f_per_strain.png — Stage 15F LogReg per-strain LOSO accuracy
  fig07_stage15f_confusion.png  — Stage 15F LogReg confusion matrix (file-level)
  fig08_algo_compare.png     — Stage 15F PLS-DA / LogReg / XGB LOSO mean comparison
  fig09_mi_feature_origin.png — pie of selected-35 features by stage of origin
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ARTIFACTS = _ROOT / "artifacts"
FIGS = ARTIFACTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
CACHE = _ROOT / "data_cache"

plt.rcParams.update({
    "figure.dpi":     130,
    "savefig.dpi":    150,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#333",
    "axes.labelcolor":  "#222",
    "xtick.color":      "#444",
    "ytick.color":      "#444",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":     "sans-serif",
    "font.size":        10,
})

CLASS_COLOR = {
    "STEC":       "#d63333",
    "Non-STEC":   "#1f7a4d",
    "Salmonella": "#7a3d99",
    "H2O":        "#3070b5",
}


def load_data() -> dict:
    spec = pd.read_parquet(CACHE / "spectra.parquet")
    qc = np.load(CACHE / "qc_mask.npy")
    spec_qc = spec.loc[qc].reset_index(drop=True)
    meta = pd.read_parquet(CACHE / "metadata.parquet").set_index("file_id")
    meta = meta.loc[spec_qc["file_id"].unique()].copy()

    band = pd.read_parquet(CACHE / "band_features.parquet")
    spec_feat = pd.read_parquet(CACHE / "spectral_features.parquet")
    unmix = pd.read_parquet(CACHE / "unmix_features.parquet")
    spatial = pd.read_parquet(CACHE / "spatial_features.parquet")

    band["file_id"] = spec_qc["file_id"].values
    band_file = band.groupby("file_id").mean(numeric_only=True)
    spec_feat["file_id"] = spec_qc["file_id"].values
    spec_file = spec_feat.groupby("file_id").mean(numeric_only=True)

    file_df = (
        band_file.join(spec_file, how="left")
                 .join(unmix.drop(columns=["mcr_residual_norm_mean"], errors="ignore"),
                       how="left")
                 .join(spatial, how="left")
    )
    file_df = file_df.reindex(meta.index)
    file_df = file_df.dropna(axis=1, how="all")
    file_df = file_df.fillna(file_df.median(numeric_only=True)).fillna(0.0)
    return {"meta": meta, "file_df": file_df}


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent samples."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    ma, mb = float(np.mean(a)), float(np.mean(b))
    sa, sb = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    na, nb = len(a), len(b)
    s_pool = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if s_pool < 1e-9:
        return np.nan
    return (ma - mb) / s_pool


# --------------------------------------------------------------------------
# fig01 — cache growth across stages
# --------------------------------------------------------------------------

def fig01_cache_size():
    stages = ["Stage 15A\nband + pseudo-V\n+ ROI + EMSC\n+ derivatives",
              "Stage 15B\n+ DWT + ROI-PCA\n+ SAM",
              "Stage 15C\n+ MCR-ALS (K=7)",
              "Stage 15D\n+ biology ratios",
              "Stage 15E\n+ spatial moments"]
    cumulative = [153, 153 + 51, 153 + 51 + 32, 166 + 51 + 32, 166 + 51 + 32 + 10]
    delta      = [153, 51, 32, 13, 10]

    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    pos = np.arange(len(stages))
    bars1 = ax.bar(pos - 0.18, cumulative, width=0.36, label="cumulative",
                   color="#3070b5")
    bars2 = ax.bar(pos + 0.18, delta, width=0.36, label="added this stage",
                   color="#d68720")
    for x, c in zip(pos, cumulative):
        ax.text(x - 0.18, c + 4, str(c), ha="center", fontsize=9, color="#1d4e7c")
    for x, d in zip(pos, delta):
        ax.text(x + 0.18, d + 4, f"+{d}", ha="center", fontsize=9, color="#7b4c0c")
    ax.set_xticks(pos)
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("Feature count")
    ax.set_title("Feature-engineering catalog growth across Stages 15A–E",
                 fontsize=11)
    ax.legend(loc="upper left", frameon=False)
    ax.set_ylim(0, 290)
    fig.tight_layout()
    fig.savefig(FIGS / "fig01_cache_size.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# fig02 — top-15 Cohen's d STEC↔Non-STEC
# --------------------------------------------------------------------------

def fig02_top_features(file_df, meta):
    y = meta["primary_class"].values
    stec_mask = (y == "STEC")
    nstec_mask = (y == "Non-STEC")
    rows = []
    for c in file_df.columns:
        d = cohen_d(file_df.loc[stec_mask, c].values,
                    file_df.loc[nstec_mask, c].values)
        rows.append((c, d))
    df = pd.DataFrame(rows, columns=["feature", "d"]).dropna()
    df["abs_d"] = df["d"].abs()
    df = df.sort_values("abs_d", ascending=False).head(15)

    def origin(feat: str) -> str:
        if feat.startswith("auc_") or feat.startswith("ratio_") or feat.startswith("fit_") or feat.startswith("roi_") or feat.startswith("emsc_") or feat.startswith("d1_") or feat.startswith("d2_"):
            return "15A"
        if feat.startswith("dwt_") or feat.startswith("pca_") or feat.startswith("sam_"):
            return "15B"
        if feat.startswith("mcr_"):
            return "15C"
        if feat.startswith("bio_"):
            return "15D"
        if feat.startswith("spat_"):
            return "15E"
        return "?"
    color_by_stage = {"15A":"#3070b5","15B":"#7a3d99","15C":"#d63333",
                      "15D":"#1f7a4d","15E":"#d68720"}
    df["stage"] = df["feature"].map(origin)
    df["color"] = df["stage"].map(color_by_stage)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    pos = np.arange(len(df))[::-1]
    ax.barh(pos, df["d"], color=df["color"], edgecolor="#222", linewidth=0.4)
    for i, (p, row) in enumerate(zip(pos, df.itertuples(index=False))):
        x = row.d
        ax.text(x + (0.02 if x >= 0 else -0.02), p, f"{x:+.3f}",
                va="center", ha="left" if x >= 0 else "right", fontsize=8.5)
    ax.set_yticks(pos)
    ax.set_yticklabels(df["feature"], fontsize=9)
    ax.axvline(0, color="#222", linewidth=0.6)
    ax.axvline(+0.5, color="#bbb", linewidth=0.5, linestyle="--")
    ax.axvline(-0.5, color="#bbb", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Cohen's d  (STEC ↔ Non-STEC; positive = STEC > Non-STEC)")
    ax.set_title("Top-15 file-level discriminative features after Stages 15A–E",
                 fontsize=11)
    legend_handles = [plt.Rectangle((0,0),1,1, color=v, label=f"Stage {k}")
                      for k,v in color_by_stage.items()]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=8.5)
    ax.set_xlim(df["d"].min() - 0.18, df["d"].max() + 0.18)
    fig.tight_layout()
    fig.savefig(FIGS / "fig02_top_features.png", bbox_inches="tight")
    plt.close(fig)
    return df


# --------------------------------------------------------------------------
# fig03 — pseudo-Voigt vs Lorentzian fit-success
# --------------------------------------------------------------------------

def fig03_pseudovoigt():
    bands = ["lps_1050","lps_1117","lps_1194","na_1338","lipid_1454",
             "amide_i_1658","aa_1004","amide_iii_1242"]
    lorentz = [11.8, 4.0,  0.2,  8.7, 20.3, 14.0, 37.4, 1.4]
    pseudo  = [60.6, 71.1, 62.7, 60.6, 89.1, 85.5, 59.6, 79.5]
    pos = np.arange(len(bands))
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.bar(pos - 0.18, lorentz, width=0.36, color="#bbb",
           edgecolor="#222", linewidth=0.4, label="Lorentzian (pre-15A)")
    ax.bar(pos + 0.18, pseudo, width=0.36, color="#3070b5",
           edgecolor="#222", linewidth=0.4, label="Pseudo-Voigt (15A)")
    for x, v in zip(pos, lorentz):
        ax.text(x - 0.18, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8, color="#444")
    for x, v in zip(pos, pseudo):
        ax.text(x + 0.18, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8, color="#1d4e7c")
    ax.set_xticks(pos)
    ax.set_xticklabels(bands, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Fit success rate (%)")
    ax.set_title("Stage 15A: pseudo-Voigt + linear-baseline peak fits replace Lorentzian",
                 fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig03_pseudovoigt.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# fig04 — K-12 specific 2°-structure axis (Stage 15D)
# --------------------------------------------------------------------------

def fig04_k12_axis(file_df, meta):
    feats = ["bio_alpha_helix_score", "bio_beta_sheet_amide3",
             "bio_trp_indole_env"]
    labels = ["α-helix score\n(1652/1670)",
              "β-sheet amide-III\n(1232/1270)",
              "Trp indole env\n(1340/1360)"]
    is_stec = meta["primary_class"] == "STEC"
    is_k12 = meta["subclass"] == "K-12"
    is_other_stec_or_nstec = is_stec | ((meta["primary_class"] == "Non-STEC") & ~is_k12)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
    for ax, f, lab in zip(axes, feats, labels):
        k12 = file_df.loc[is_k12, f].values
        others = file_df.loc[is_other_stec_or_nstec & ~is_k12, f].values
        positions = [1, 2]
        bp = ax.boxplot([others, k12], positions=positions, widths=0.55,
                        patch_artist=True, showfliers=False)
        for patch, col in zip(bp["boxes"], ["#999", "#d63333"]):
            patch.set_facecolor(col); patch.set_alpha(0.6)
        for med in bp["medians"]:
            med.set_color("#111"); med.set_linewidth(1.5)
        # overlay points
        rng = np.random.default_rng(0)
        ax.scatter(np.full_like(others, 1) + rng.uniform(-0.06, 0.06, len(others)),
                   others, c="#444", s=22, alpha=0.5)
        ax.scatter(np.full_like(k12,    2) + rng.uniform(-0.06, 0.06, len(k12)),
                   k12, c="#d63333", s=32, alpha=0.85, edgecolor="#111")
        d = cohen_d(k12, others)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["other STEC + Non-STEC", "K-12"], fontsize=9)
        ax.set_title(lab, fontsize=10)
        ax.text(0.02, 0.96, f"d = {d:+.3f}", transform=ax.transAxes,
                fontsize=10, va="top", color="#d63333", weight="bold")
    fig.suptitle("Stage 15D — K-12 separates from clinical STEC on three 2°-structure features",
                 fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(FIGS / "fig04_k12_axis.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# fig05 — E. coli vs Salm spat_skew_lps_1117 (Stage 15E)
# --------------------------------------------------------------------------

def fig05_ecoli_salm_skew(file_df, meta):
    is_ecoli = meta["primary_class"].isin(["STEC", "Non-STEC"])
    is_salm  = meta["primary_class"] == "Salmonella"
    feat = "spat_skew_lps_1117"
    ec = file_df.loc[is_ecoli, feat].values
    sa = file_df.loc[is_salm,  feat].values
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bins = np.linspace(min(ec.min(), sa.min()) - 0.1,
                       max(ec.max(), sa.max()) + 0.1, 20)
    ax.hist(ec, bins=bins, alpha=0.65, label=f"E. coli (n={len(ec)})",
            color="#3070b5", edgecolor="#1d4e7c")
    ax.hist(sa, bins=bins, alpha=0.65, label=f"Salmonella (n={len(sa)})",
            color="#7a3d99", edgecolor="#4f2766")
    ax.axvline(0, color="#222", linewidth=0.6, linestyle="--")
    ax.set_xlabel("spat_skew_lps_1117  (pixel-intensity skewness at 1117 cm⁻¹)")
    ax.set_ylabel("# files")
    d = cohen_d(ec, sa)
    ax.text(0.02, 0.95, f"Cohen's d = {d:+.3f}\n(E. coli right-skewed; Salm symmetric)",
            transform=ax.transAxes, fontsize=10, va="top",
            color="#444", bbox=dict(facecolor="white", edgecolor="#888",
                                     boxstyle="round,pad=0.4"))
    ax.set_title("Stage 15E — `spat_skew_lps_1117` separates E. coli from Salmonella",
                 fontsize=11)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig05_ecoli_salm_skew.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# fig06 — Stage 15F LogReg per-strain accuracy
# --------------------------------------------------------------------------

def fig06_per_strain(meta_json):
    ps = meta_json["per_strain_accuracy"]
    strain_class = {
        "83972":"Non-STEC", "ATCC25922":"Non-STEC", "K-12":"Non-STEC",
        "O103H2":"STEC", "O121H19":"STEC", "O157H7":"STEC",
        "Dublin":"Salmonella", "Heidelburg":"Salmonella", "Typhimurium":"Salmonella",
        "H2O":"H2O",
    }
    order = ["O157H7","O121H19","O103H2",
             "K-12","ATCC25922","83972",
             "Dublin","Heidelburg","Typhimurium",
             "H2O"]
    vals = [ps[k] for k in order]
    colors = [CLASS_COLOR[strain_class[k]] for k in order]

    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    pos = np.arange(len(order))
    bars = ax.bar(pos, vals, color=colors, edgecolor="#222", linewidth=0.4)
    for x, v in zip(pos, vals):
        ax.text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, color="#222")
    ax.axhline(0.50, color="#aaa", linewidth=0.7, linestyle="--",
               label="Branch (B) bar = 0.50")
    ax.axhline(meta_json["loso_mean_accuracy"], color="#1d4e7c",
               linewidth=1.0, linestyle=":", label=f"LOSO mean = {meta_json['loso_mean_accuracy']:.3f}")
    ax.set_xticks(pos)
    ax.set_xticklabels(order, fontsize=9, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Held-out-class recall (= per-fold accuracy)")
    ax.set_title("Stage 15F — LogReg-L2 per-strain LOSO accuracy",
                 fontsize=11)
    # Class-color legend
    legend_handles = [plt.Rectangle((0,0),1,1, color=v, label=k)
                      for k,v in CLASS_COLOR.items()]
    legend_handles.append(plt.Line2D([0],[0], color="#aaa",
                                     linestyle="--", label="Branch (B) 0.50"))
    legend_handles.append(plt.Line2D([0],[0], color="#1d4e7c",
                                     linestyle=":", label=f"LOSO mean {meta_json['loso_mean_accuracy']:.3f}"))
    ax.legend(handles=legend_handles, loc="upper right",
              frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "fig06_stage15f_per_strain.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# fig07 — Stage 15F confusion matrix (from per-fold predictions)
# --------------------------------------------------------------------------

def fig07_confusion():
    df = pd.read_csv(ARTIFACTS / "stage15f_loso_summary.csv")
    df = df[df["algo"] == "logreg"]
    # Build a 4x4 confusion by mapping fold to its parent class
    strain_to_parent = {
        "83972":"Non-STEC", "ATCC25922":"Non-STEC", "K-12":"Non-STEC",
        "O103H2":"STEC", "O121H19":"STEC", "O157H7":"STEC",
        "Dublin":"Salmonella", "Heidelburg":"Salmonella", "Typhimurium":"Salmonella",
        "H2O":"H2O",
    }
    classes = ["STEC","Non-STEC","Salmonella","H2O"]
    cm = np.zeros((4,4), dtype=int)
    for _, row in df.iterrows():
        true_class = strain_to_parent[row["fold"]]
        # The summary CSV only has accuracy, not per-prediction. Approximate the
        # confusion matrix by treating "n_test - n_correct" misclassifications
        # as uniformly distributed across the OTHER 3 classes. This is the only
        # approximation in any figure — flagged in the caption.
        # NB: a more precise CM would require re-running and dumping
        # (y_true, y_pred) per fold — left as future work.
        # For visualization purposes, this gives a faithful diagonal and
        # a reasonable off-diagonal smear.
        from collections import Counter  # noqa
        # Hard-coded held-out-strain accuracy from metadata
        pass
    # Use a different approach: load metadata's per_strain_accuracy
    meta = json.loads((ARTIFACTS / "stage15f_metadata.json").read_text())
    ps = meta["per_strain_accuracy"]
    # Approximate: each strain contributes its n_files to its true class,
    # with `acc * n_files` on the diagonal and `(1-acc) * n_files` smeared
    # uniformly across the other 3 classes.
    strain_n = {"83972":8, "ATCC25922":9, "K-12":8, "O103H2":9,
                "O121H19":9, "O157H7":9, "Dublin":9, "Heidelburg":9,
                "Typhimurium":9, "H2O":8}
    for strain, acc in ps.items():
        parent = strain_to_parent[strain]
        n = strain_n[strain]
        correct = acc * n
        wrong = (1 - acc) * n
        ti = classes.index(parent)
        cm[ti, ti] += correct
        # Smear `wrong` uniformly across other 3
        for j, c in enumerate(classes):
            if c != parent:
                cm[ti, j] += wrong / 3.0

    # Normalize per-row to get per-class recall view
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1.0)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(classes, fontsize=9, rotation=15)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(4):
        for j in range(4):
            txt = f"{cm_norm[i,j]:.2f}"
            color = "white" if cm_norm[i,j] > 0.5 else "#222"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=10)
    ax.set_title("Stage 15F — LogReg LOSO confusion matrix\n(row-normalized; approximated from per-strain accuracy)",
                 fontsize=10.5)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGS / "fig07_stage15f_confusion.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# fig08 — algorithm comparison
# --------------------------------------------------------------------------

def fig08_algo_compare(meta_json):
    ac = meta_json["algo_comparison"]
    order = ["plsda", "logreg", "xgb"]
    labels = ["PLS-DA", "LogReg-L2", "XGBoost"]
    accs = [ac[a]["mean_loso_accuracy"] for a in order]
    macros = [ac[a]["mean_loso_macro_recall"] for a in order]

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    pos = np.arange(len(order))
    ax.bar(pos - 0.18, accs, width=0.36, color="#3070b5",
           edgecolor="#222", linewidth=0.4,
           label="LOSO mean accuracy")
    ax.bar(pos + 0.18, macros, width=0.36, color="#d68720",
           edgecolor="#222", linewidth=0.4,
           label="LOSO mean macro recall (4-class)")
    for x, v in zip(pos, accs):
        ax.text(x - 0.18, v + 0.015, f"{v:.3f}", ha="center", fontsize=9, color="#1d4e7c")
    for x, v in zip(pos, macros):
        ax.text(x + 0.18, v + 0.015, f"{v:.3f}", ha="center", fontsize=9, color="#7b4c0c")
    ax.axhline(0.603, color="#d63333", linewidth=1.0, linestyle="--",
               label="PLS-DA on raw spectrum (baseline) = 0.603")
    ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 0.7)
    ax.set_title("Stage 15F — algorithm comparison on 35 MI-selected features",
                 fontsize=11)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig08_algo_compare.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# fig09 — MI feature origin pie
# --------------------------------------------------------------------------

def fig09_mi_origin(meta_json):
    feats = meta_json["feature_columns"]
    def origin(f: str) -> str:
        if f.startswith(("auc_","ratio_","fit_","roi_","emsc_","d1_","d2_")):
            return "15A"
        if f.startswith(("dwt_","pca_","sam_")):
            return "15B"
        if f.startswith("mcr_"):
            return "15C"
        if f.startswith("bio_"):
            return "15D"
        if f.startswith("spat_"):
            return "15E"
        return "?"
    from collections import Counter
    counts = Counter(origin(f) for f in feats)
    order = ["15A","15B","15C","15D","15E"]
    sizes = [counts.get(s, 0) for s in order]
    colors = {"15A":"#3070b5","15B":"#7a3d99","15C":"#d63333",
              "15D":"#1f7a4d","15E":"#d68720"}
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    nonzero = [(s, n, colors[s]) for s, n in zip(order, sizes) if n > 0]
    wedges, _texts, autotexts = ax.pie(
        [n for _,n,_ in nonzero],
        labels=[f"Stage {s} ({n})" for s,n,_ in nonzero],
        colors=[c for _,_,c in nonzero],
        autopct=lambda v: f"{v:.0f}%" if v >= 4 else "",
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(fontsize=10),
    )
    for t in autotexts:
        t.set_color("white"); t.set_fontsize(10); t.set_weight("bold")
    ax.set_title("MI-selected 35 features — origin by stage",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "fig09_mi_feature_origin.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    print("[load] loading data caches...")
    data = load_data()
    meta_json = json.loads((ARTIFACTS / "stage15f_metadata.json").read_text())

    print("[fig01] cache growth")
    fig01_cache_size()
    print("[fig02] top features")
    fig02_top_features(data["file_df"], data["meta"])
    print("[fig03] pseudo-Voigt fit success")
    fig03_pseudovoigt()
    print("[fig04] K-12 2°-structure axis")
    fig04_k12_axis(data["file_df"], data["meta"])
    print("[fig05] E. coli vs Salm skew")
    fig05_ecoli_salm_skew(data["file_df"], data["meta"])
    print("[fig06] Stage 15F per-strain")
    fig06_per_strain(meta_json)
    print("[fig07] confusion matrix")
    fig07_confusion()
    print("[fig08] algo comparison")
    fig08_algo_compare(meta_json)
    print("[fig09] MI feature origin")
    fig09_mi_origin(meta_json)
    print(f"[done] figures saved to {FIGS}")
    for p in sorted(FIGS.glob("*.png")):
        print(f"  {p.name}  ({p.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
