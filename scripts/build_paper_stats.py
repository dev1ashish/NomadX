"""Post-hoc statistical analysis of Stage 15F LOSO predictions.

Reads `artifacts/stage15f_loso_predictions.parquet` (one row per file × algo,
with y_true / y_pred / proba_<class>) and produces:

  fig07_stage15f_confusion.png   — REAL confusion matrix (replaces approximation)
  fig10_per_class_f1.png         — per-class precision / recall / F1 for LogReg
  fig11_bootstrap_ci.png         — 5000-resample bootstrap 95% CI on LOSO mean per algo
  fig12_logreg_coefs.png         — LogReg standardized coefficients per class (top 15)

And dumps to JSON:

  artifacts/stage15f_paper_stats.json — bootstrap CIs, McNemar p-values, per-class metrics
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ARTIFACTS = _ROOT / "artifacts"
FIGS = ARTIFACTS / "figures"

PRIMARY_CLASSES = ["STEC", "Non-STEC", "Salmonella", "H2O"]
CLASS_COLOR = {
    "STEC":       "#d63333",
    "Non-STEC":   "#1f7a4d",
    "Salmonella": "#7a3d99",
    "H2O":        "#3070b5",
}
ALGO_LABEL = {"plsda": "PLS-DA", "logreg": "LogReg-L2", "xgb": "XGBoost"}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_predictions() -> pd.DataFrame:
    path = ARTIFACTS / "stage15f_loso_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — re-run scripts/run_stage15f_final.py first"
        )
    df = pd.read_parquet(path)
    print(f"[load] {len(df)} predictions × {df.columns.tolist()}")
    return df


# ---------------------------------------------------------------------------
# fig07 — REAL confusion matrix
# ---------------------------------------------------------------------------

def fig07_real_confusion(df: pd.DataFrame) -> dict:
    """Replace the approximated fig07 with a real per-file confusion."""
    out = {}
    for algo in ["logreg"]:  # production algo
        sub = df[df["algo"] == algo]
        cm = confusion_matrix(sub["y_true"], sub["y_pred"], labels=PRIMARY_CLASSES)
        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
        out[algo] = {"cm_counts": cm.tolist(),
                     "cm_normalized": cm_norm.tolist()}
        fig, ax = plt.subplots(figsize=(5.8, 4.6))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1.0)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(PRIMARY_CLASSES, fontsize=9, rotation=15)
        ax.set_yticklabels(PRIMARY_CLASSES, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(4):
            for j in range(4):
                n = int(cm[i, j])
                color = "white" if cm_norm[i, j] > 0.5 else "#222"
                ax.text(j, i, f"{cm_norm[i,j]:.2f}\n(n={n})",
                        ha="center", va="center", color=color, fontsize=9.5)
        ax.set_title(f"Stage 15F — {ALGO_LABEL[algo]} LOSO confusion matrix "
                     f"(real, n={len(sub)} files)\nrow-normalized",
                     fontsize=10.5)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        fig.tight_layout()
        fig.savefig(FIGS / "fig07_stage15f_confusion.png", bbox_inches="tight")
        plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# fig10 — per-class precision / recall / F1
# ---------------------------------------------------------------------------

def fig10_per_class_f1(df: pd.DataFrame) -> dict:
    out = {}
    for algo in ["plsda", "logreg", "xgb"]:
        sub = df[df["algo"] == algo]
        prec = precision_score(sub["y_true"], sub["y_pred"],
                               labels=PRIMARY_CLASSES, average=None,
                               zero_division=0)
        rec = recall_score(sub["y_true"], sub["y_pred"],
                           labels=PRIMARY_CLASSES, average=None,
                           zero_division=0)
        f1 = f1_score(sub["y_true"], sub["y_pred"],
                      labels=PRIMARY_CLASSES, average=None,
                      zero_division=0)
        out[algo] = {
            "precision": dict(zip(PRIMARY_CLASSES, prec.tolist())),
            "recall":    dict(zip(PRIMARY_CLASSES, rec.tolist())),
            "f1":        dict(zip(PRIMARY_CLASSES, f1.tolist())),
            "macro_f1":  float(f1_score(sub["y_true"], sub["y_pred"],
                                        labels=PRIMARY_CLASSES,
                                        average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(sub["y_true"], sub["y_pred"],
                                          labels=PRIMARY_CLASSES,
                                          average="weighted", zero_division=0)),
        }
    # Plot — focus on LogReg
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    pos = np.arange(len(PRIMARY_CLASSES))
    sub = df[df["algo"] == "logreg"]
    prec = precision_score(sub["y_true"], sub["y_pred"], labels=PRIMARY_CLASSES,
                           average=None, zero_division=0)
    rec = recall_score(sub["y_true"], sub["y_pred"], labels=PRIMARY_CLASSES,
                       average=None, zero_division=0)
    f1 = f1_score(sub["y_true"], sub["y_pred"], labels=PRIMARY_CLASSES,
                  average=None, zero_division=0)
    width = 0.27
    bars_p = ax.bar(pos - width, prec, width=width, color="#3070b5",
                    edgecolor="#222", linewidth=0.4, label="Precision")
    bars_r = ax.bar(pos, rec, width=width, color="#1f7a4d",
                    edgecolor="#222", linewidth=0.4, label="Recall")
    bars_f = ax.bar(pos + width, f1, width=width, color="#d68720",
                    edgecolor="#222", linewidth=0.4, label="F1")
    for bars, vals in [(bars_p, prec), (bars_r, rec), (bars_f, f1)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}",
                    ha="center", fontsize=8.5)
    ax.set_xticks(pos)
    ax.set_xticklabels(PRIMARY_CLASSES, fontsize=10)
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title("Stage 15F — LogReg per-class precision / recall / F1 (file-level LOSO)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "fig10_per_class_f1.png", bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# fig11 — bootstrap CIs on LOSO mean per algo
# ---------------------------------------------------------------------------

def fig11_bootstrap_ci(df: pd.DataFrame, n_boot: int = 5000) -> dict:
    """Bootstrap by resampling files with replacement; recompute LOSO mean."""
    out = {}
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    pos = np.arange(3)
    algos = ["plsda", "logreg", "xgb"]
    means = []
    los = []
    his = []
    for algo in algos:
        sub = df[df["algo"] == algo].reset_index(drop=True)
        sub["correct"] = (sub["y_true"] == sub["y_pred"]).astype(int)
        file_ids = sub["file_id"].values
        correct = sub["correct"].values
        # Group by file (each file appears once per algo in single-seed run)
        unique_files = np.unique(file_ids)
        # Aggregate per-file (should already be 1:1 in single-seed runs)
        per_file_acc = sub.groupby("file_id")["correct"].mean().values
        n = len(per_file_acc)
        boot_means = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            boot_means[b] = float(per_file_acc[idx].mean())
        mean = float(per_file_acc.mean())
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        means.append(mean); los.append(lo); his.append(hi)
        out[algo] = {
            "loso_mean_file_accuracy": mean,
            "boot_ci_95_lo": float(lo),
            "boot_ci_95_hi": float(hi),
            "n_files": int(n),
        }
        print(f"  {algo}: mean={mean:.3f}, 95% CI = [{lo:.3f}, {hi:.3f}]")
    yerr_lo = np.array(means) - np.array(los)
    yerr_hi = np.array(his) - np.array(means)
    colors = ["#3070b5", "#1f7a4d", "#d68720"]
    ax.bar(pos, means, color=colors, edgecolor="#222", linewidth=0.4,
           alpha=0.85)
    ax.errorbar(pos, means, yerr=[yerr_lo, yerr_hi], fmt="none",
                ecolor="#222", capsize=8, capthick=1.2, linewidth=1.2)
    for x, m, lo, hi in zip(pos, means, los, his):
        ax.text(x, m + 0.025, f"{m:.3f}\n[{lo:.3f}, {hi:.3f}]",
                ha="center", fontsize=9, color="#222")
    ax.set_xticks(pos)
    ax.set_xticklabels([ALGO_LABEL[a] for a in algos], fontsize=10)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("LOSO mean file-accuracy")
    ax.axhline(0.603, color="#d63333", linewidth=0.9, linestyle="--",
               label="PLS-DA on raw spectrum baseline (9-fold) = 0.603")
    ax.set_title(f"Stage 15F — bootstrap 95% CIs on LOSO mean file-accuracy ({n_boot} resamples)",
                 fontsize=10.5)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig11_bootstrap_ci.png", bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# McNemar paired test
# ---------------------------------------------------------------------------

def mcnemar_pair(df: pd.DataFrame, algo_a: str, algo_b: str) -> dict:
    """McNemar test on per-file correct/incorrect comparing two algos."""
    a = df[df["algo"] == algo_a].set_index("file_id")
    b = df[df["algo"] == algo_b].set_index("file_id")
    common = a.index.intersection(b.index)
    a_correct = (a.loc[common, "y_true"] == a.loc[common, "y_pred"]).values
    b_correct = (b.loc[common, "y_true"] == b.loc[common, "y_pred"]).values
    n10 = int(((a_correct == True) & (b_correct == False)).sum())   # a wins
    n01 = int(((a_correct == False) & (b_correct == True)).sum())   # b wins
    n00 = int(((a_correct == False) & (b_correct == False)).sum())
    n11 = int(((a_correct == True) & (b_correct == True)).sum())
    # McNemar exact (binomial) two-sided p
    from scipy.stats import binomtest
    n_disc = n10 + n01
    if n_disc == 0:
        p = 1.0
    else:
        k = min(n10, n01)
        p = float(binomtest(k, n_disc, p=0.5, alternative="two-sided").pvalue)
    return {
        "algo_a": algo_a, "algo_b": algo_b,
        "n_a_only_correct": n10, "n_b_only_correct": n01,
        "n_both_correct": n11, "n_both_wrong": n00,
        "n_total": int(len(common)),
        "mcnemar_p": p,
    }


# ---------------------------------------------------------------------------
# fig12 — LogReg standardized coefficients per class
# ---------------------------------------------------------------------------

def fig12_logreg_coefs() -> dict:
    """Inspect the deployed LogReg's coefficients per class.

    The classifier sits behind a StandardScaler, so the coefficients are
    already on a comparable scale (each input feature has unit variance).
    """
    clf_pkg = joblib.load(ARTIFACTS / "stage15f_classifier.joblib")
    feature_cols = json.loads(
        (ARTIFACTS / "stage15f_feature_columns.json").read_text()
    )
    # If the saved object is a Pipeline (logreg/plsda case), the classifier
    # is at .named_steps["clf"]. If it's the _LabelEncodedPipeline (xgb),
    # the underlying pipeline is at .pipeline.
    if hasattr(clf_pkg, "named_steps"):
        pipe = clf_pkg
    else:
        pipe = clf_pkg.pipeline
    lr = pipe.named_steps["clf"]
    if not hasattr(lr, "coef_"):
        print("[fig12] classifier has no coef_ — skipping coefficient figure")
        return {}
    # coef_ shape: (n_classes, n_features). classes_ = sorted alphabetically by default.
    coefs = lr.coef_
    classes = list(lr.classes_)
    print(f"[fig12] coef_ shape: {coefs.shape}; classes: {classes}")

    # For each class, pick top-8 positive + top-3 negative coefs by magnitude
    fig, axes = plt.subplots(1, len(classes), figsize=(15, 4.2), sharey=False)
    if len(classes) == 1:
        axes = [axes]
    coef_dump = {}
    for ax, c, w in zip(axes, classes, coefs):
        # Sort by absolute coefficient
        order = np.argsort(np.abs(w))[::-1][:11]
        w_top = w[order]
        f_top = [feature_cols[i] for i in order]
        # Build colored bars: red for positive (→ this class), blue for negative
        colors = ["#d63333" if v > 0 else "#3070b5" for v in w_top]
        positions = np.arange(len(w_top))[::-1]
        ax.barh(positions, w_top, color=colors, edgecolor="#222", linewidth=0.4)
        for p, v in zip(positions, w_top):
            ax.text(v + (0.03 if v >= 0 else -0.03), p,
                    f"{v:+.2f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8)
        ax.set_yticks(positions)
        ax.set_yticklabels(f_top, fontsize=8)
        ax.axvline(0, color="#222", linewidth=0.6)
        ax.set_title(f"class = {c}", fontsize=10)
        if ax is axes[0]:
            ax.set_xlabel("Standardized coefficient")
        coef_dump[c] = dict(zip(f_top, [float(v) for v in w_top]))
    fig.suptitle("Stage 15F — deployed LogReg-L2 top-11 standardized coefficients per class\n"
                 "(red = increases probability of this class; blue = decreases)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "fig12_logreg_coefs.png", bbox_inches="tight")
    plt.close(fig)
    return coef_dump


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[stats] computing paper statistics")
    df = load_predictions()
    out: dict = {}

    print("\n[fig07] real confusion matrix")
    out["confusion"] = fig07_real_confusion(df)

    print("\n[fig10] per-class precision/recall/F1")
    out["per_class_metrics"] = fig10_per_class_f1(df)

    print("\n[fig11] bootstrap CIs (5000 resamples)")
    out["bootstrap_ci"] = fig11_bootstrap_ci(df, n_boot=5000)

    print("\n[mcnemar] paired tests")
    out["mcnemar"] = {
        "logreg_vs_plsda": mcnemar_pair(df, "logreg", "plsda"),
        "logreg_vs_xgb":   mcnemar_pair(df, "logreg", "xgb"),
        "plsda_vs_xgb":    mcnemar_pair(df, "plsda", "xgb"),
    }
    for k, v in out["mcnemar"].items():
        print(f"  {k}: p = {v['mcnemar_p']:.4f}   "
              f"(a-only={v['n_a_only_correct']}, b-only={v['n_b_only_correct']}, "
              f"both-correct={v['n_both_correct']}, n={v['n_total']})")

    print("\n[fig12] LogReg standardized coefficients per class")
    out["logreg_coefs"] = fig12_logreg_coefs()

    # 9-strain-only mean (excluding H2O) — apples-to-apples vs baselines
    sub_no_h2o = df[df["fold"] != "H2O"].copy()
    sub_no_h2o["correct"] = (sub_no_h2o["y_true"] == sub_no_h2o["y_pred"]).astype(int)
    nine_strain_means = {}
    for algo in ["plsda", "logreg", "xgb"]:
        s = sub_no_h2o[sub_no_h2o["algo"] == algo]
        nine_strain_means[algo] = float(s["correct"].mean())
        print(f"  {algo} 9-strain LOSO mean (no H2O): {nine_strain_means[algo]:.3f}")
    out["nine_strain_loso_mean"] = nine_strain_means

    # Save
    (ARTIFACTS / "stage15f_paper_stats.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    print(f"\n[done] stats → {ARTIFACTS / 'stage15f_paper_stats.json'}")


if __name__ == "__main__":
    main()
