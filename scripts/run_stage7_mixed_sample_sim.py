"""Stage 7 — mixed-sample degradation simulation.

Pre-registered in plan/08 (2026-05-18 entry). Tests the briefing's "10-20% drop
vs pure-culture" prediction against our Stage 5 XGB classifier.

Method:
  1. Train Stage 5 XGBoost on all pure-culture data using the 13 anchor features.
  2. For each of 3 pairwise class mixtures:
     - Take per-file mean preprocessed spectra (X_full, shape 7122 x 987)
     - For every (file_a in class A, file_b in class B) pair:
       - Compute file-mean spectrum for each
       - Mix at alpha in [0.0, 0.05, ..., 1.0]: S_mix = alpha * S_a + (1-alpha) * S_b
       - Compute band features on the mixed spectrum
       - Predict with the trained XGB
     - Aggregate accuracy@majority-class per alpha
     - Plot degradation curve

Outputs:
  outputs/band_chemistry/stage7/01_degradation_curves.png
  outputs/band_chemistry/stage7/02_per_pair_curves.csv
  outputs/band_chemistry/stage7/03_briefing_check.json
"""
from __future__ import annotations
from pathlib import Path
import sys
import json
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

REPO = Path("/Users/devashishthapliyal/Documents/NomadX")
sys.path.insert(0, str(REPO))
from atlas.io import load_cache
from atlas import band_features as bf

CACHE = REPO / "data_cache"
OUT = REPO / "outputs" / "band_chemistry" / "stage7"
OUT.mkdir(parents=True, exist_ok=True)

ANCHOR_FEATURES = [
    "auc_lps_1050", "auc_lps_1117", "auc_lps_1194",
    "auc_aa_1004", "auc_aa_1176", "auc_aa_1617", "auc_lipid_1080",
    "ratio_lipid_over_protein", "ratio_lps_1117_over_1050", "ratio_lps_1194_over_1050",
    "auc_na_1338", "auc_lipid_1454", "auc_amide_i_1658",
]

# ---------- load ----------
print("Loading cache …")
spec_df, _X, _wn, meta = load_cache(CACHE)
X = np.load(CACHE / "spectra_array_preprocessed.npy")
wn = np.load(CACHE / "wavenumber_axis_preprocessed.npy")
qc = np.load(CACHE / "qc_mask.npy")
spec_df = spec_df.reset_index(drop=True)
spec_df_qc = spec_df[qc].reset_index(drop=True)
Xq = X[qc]

feat = pd.read_parquet(CACHE / "band_features.parquet").reset_index(drop=True)
feat["primary_class"] = spec_df_qc["primary_class"].values
feat["file_id"] = spec_df_qc["file_id"].values
print(f"  spectra: {Xq.shape}, classes: {spec_df_qc['primary_class'].value_counts().to_dict()}")

# ---------- train XGB on pure cultures ----------
LABEL_INDEX = {c: i for i, c in enumerate(["H2O", "Non-STEC", "STEC", "Salmonella"])}
INDEX_TO_LABEL = ["H2O", "Non-STEC", "STEC", "Salmonella"]

X_feat = feat[ANCHOR_FEATURES].values
y_int = np.array([LABEL_INDEX[c] for c in spec_df_qc["primary_class"].values])

scaler = StandardScaler().fit(X_feat)
X_feat_scaled = scaler.transform(X_feat)

print("\nTraining XGBoost on all pure-culture data …")
t0 = time.perf_counter()
clf = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.08,
    subsample=0.85, colsample_bytree=0.85,
    objective="multi:softprob", random_state=42,
    n_jobs=4, eval_metric="mlogloss",
    tree_method="hist", verbosity=0,
)
clf.fit(X_feat_scaled, y_int)
print(f"  trained in {time.perf_counter()-t0:.1f}s")

# Sanity: training accuracy
train_pred = clf.predict(X_feat_scaled)
print(f"  training accuracy: {(train_pred == y_int).mean():.3f}")

# ---------- build per-file mean spectra (preprocessed) ----------
print("\nBuilding per-file mean preprocessed spectra …")
file_means = {}     # file_id -> (987,) mean spectrum
file_class = {}     # file_id -> primary_class
file_subclass = {}
for fid in spec_df_qc["file_id"].unique():
    m = spec_df_qc["file_id"].values == fid
    file_means[fid] = Xq[m].mean(0)
    file_class[fid] = spec_df_qc.loc[m, "primary_class"].iloc[0]
    file_subclass[fid] = spec_df_qc.loc[m, "subclass"].fillna("H2O").iloc[0]
print(f"  {len(file_means)} file means")

# ---------- mixed-sample simulation ----------
ALPHAS = np.round(np.arange(0.0, 1.0001, 0.05), 3)
PAIRS = [
    ("STEC", "Non-STEC"),
    ("STEC", "Salmonella"),
    ("Non-STEC", "Salmonella"),
]

def predict_class(mixed_spectrum):
    """Compute band features for a single mixed spectrum and predict."""
    X_one = mixed_spectrum.reshape(1, -1)
    fdf = bf.feature_frame(X_one, wn, ratios=True, fits=False)  # skip Lorentzian
    Xf = fdf[ANCHOR_FEATURES].values
    Xf_scaled = scaler.transform(Xf)
    pred_idx = clf.predict(Xf_scaled)[0]
    return INDEX_TO_LABEL[pred_idx]


# For each pairwise mixture, scan alpha
rows = []
for cls_a, cls_b in PAIRS:
    files_a = [f for f, c in file_class.items() if c == cls_a]
    files_b = [f for f, c in file_class.items() if c == cls_b]
    print(f"\nMixture {cls_a} × {cls_b}: {len(files_a)} × {len(files_b)} = "
          f"{len(files_a) * len(files_b)} file pairs × {len(ALPHAS)} alphas")
    t0 = time.perf_counter()
    for fa in files_a:
        sa = file_means[fa]
        for fb in files_b:
            sb = file_means[fb]
            for alpha in ALPHAS:
                # alpha = fraction of class A in the mixture
                mix = alpha * sa + (1 - alpha) * sb
                pred = predict_class(mix)
                # Majority class is A if alpha > 0.5, B if alpha < 0.5, ambiguous at 0.5
                if alpha > 0.5:
                    majority = cls_a
                elif alpha < 0.5:
                    majority = cls_b
                else:
                    majority = "tie"
                rows.append({
                    "pair": f"{cls_a}×{cls_b}",
                    "alpha": alpha,
                    "majority": majority,
                    "predicted": pred,
                    "correct": (pred == majority) if majority != "tie" else None,
                    "file_a": fa, "file_b": fb,
                })
    print(f"  done in {time.perf_counter()-t0:.1f}s")

df = pd.DataFrame(rows)
print(f"\nTotal mixed-sample predictions: {len(df)}")

# ---------- aggregate ----------
agg = df.dropna(subset=["correct"]).groupby(["pair", "alpha"])["correct"].mean().reset_index()
agg.rename(columns={"correct": "majority_accuracy"}, inplace=True)
print("\nAccuracy at α = 0.7, 0.8, 0.9, 1.0 by pair:")
for pair in df["pair"].unique():
    sub = agg[agg["pair"] == pair]
    for a in [0.5, 0.7, 0.8, 0.9, 1.0]:
        v = sub.loc[sub["alpha"] == a, "majority_accuracy"]
        if len(v):
            print(f"  {pair:30s}  α={a:.2f}  acc={v.values[0]:.3f}")

agg.to_csv(OUT / "02_per_pair_curves.csv", index=False)

# Also tabulate "predicted class distribution" at α=0.5 (the 50/50 mix)
print("\nAt α=0.5 (50/50 mixture), what does the classifier predict?")
mid = df[df["alpha"] == 0.5]
for pair in PAIRS:
    pair_str = f"{pair[0]}×{pair[1]}"
    sub = mid[mid["pair"] == pair_str]
    dist = sub["predicted"].value_counts(normalize=True).round(3).to_dict()
    print(f"  {pair_str:30s}  prediction distribution: {dist}")


# ---------- plot ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
pair_colors = {"STEC×Non-STEC": "#a02124",
               "STEC×Salmonella": "#175f17",
               "Non-STEC×Salmonella": "#114573"}

for ax, pair in zip(axes, df["pair"].unique()):
    sub = agg[agg["pair"] == pair]
    # Two halves: alpha > 0.5 (A is majority) and alpha < 0.5 (B is majority)
    upper = sub[sub["alpha"] > 0.5].sort_values("alpha")
    lower = sub[sub["alpha"] < 0.5].sort_values("alpha", ascending=False)
    # X-axis = majority fraction (so both halves can share x ∈ [0.5, 1.0])
    upper_x = upper["alpha"].values
    upper_y = upper["majority_accuracy"].values
    lower_x = 1 - lower["alpha"].values   # flip: alpha=0.0 → majority=B at frac 1.0
    lower_y = lower["majority_accuracy"].values

    ax.plot(upper_x, upper_y, marker="o", lw=2, color=pair_colors[pair],
            label=f"{pair.split('×')[0]} majority")
    ax.plot(lower_x, lower_y, marker="s", lw=2, ls="--", color=pair_colors[pair],
            label=f"{pair.split('×')[1]} majority")
    # Shade the briefing's "10-20% drop" zone
    ax.axhspan(0.65, 0.80, color="orange", alpha=0.10, label="briefing 10-20% drop zone")
    ax.axhline(0.5, color="black", lw=0.5, ls=":", alpha=0.5, label="chance (binary)")
    ax.axvline(0.7, color="black", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlabel("Majority class fraction α")
    ax.set_ylabel("Majority-class prediction accuracy" if pair == "STEC×Non-STEC" else "")
    ax.set_title(pair, fontsize=11)
    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

fig.suptitle("Stage 7 — Mixed-sample degradation (linear mixture of per-file means; "
             "Stage 5 XGB classifier)", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "01_degradation_curves.png", bbox_inches="tight", dpi=130)
plt.close()
print(f"\nwrote {OUT / '01_degradation_curves.png'}")

# ---------- verdict ----------
def lookup(pair, alpha):
    s = agg.loc[(agg["pair"] == pair) & (agg["alpha"] == alpha), "majority_accuracy"]
    return float(s.values[0]) if len(s) else None

result = {
    "pairs": list(df["pair"].unique()),
    "accuracy_at_alpha_1.0": {p: lookup(p, 1.0) for p in df["pair"].unique()},
    "accuracy_at_alpha_0.9": {p: lookup(p, 0.9) for p in df["pair"].unique()},
    "accuracy_at_alpha_0.8": {p: lookup(p, 0.8) for p in df["pair"].unique()},
    "accuracy_at_alpha_0.7": {p: lookup(p, 0.7) for p in df["pair"].unique()},
    "accuracy_at_alpha_0.6": {p: lookup(p, 0.6) for p in df["pair"].unique()},
}
# Briefing check: at α=0.7, is each pair's accuracy in [0.55, 0.75]?
checks = {}
all_in_range = True
two_in_range = 0
for p in df["pair"].unique():
    v = lookup(p, 0.7)
    in_range = (v is not None) and (0.55 <= v <= 0.75)
    checks[p] = {"alpha_0.7": v, "in_briefing_range": in_range}
    if in_range:
        two_in_range += 1
    if not in_range:
        all_in_range = False
result["briefing_check_alpha_0.7"] = checks
if all_in_range:
    result["verdict"] = "A"
    result["verdict_text"] = "Briefing 10-20% drop validated on all 3 pairs at α=0.7."
elif two_in_range >= 2:
    result["verdict"] = "B"
    result["verdict_text"] = f"Briefing partially validated ({two_in_range}/3 pairs in range)."
else:
    # Check direction
    avg = np.mean([v for v in [lookup(p, 0.7) for p in df["pair"].unique()] if v is not None])
    if avg < 0.55:
        result["verdict"] = "C"
        result["verdict_text"] = f"Briefing NOT replicated — classifier drops faster than 10-20% (mean acc@0.7 = {avg:.3f})."
    else:
        result["verdict"] = "D"
        result["verdict_text"] = f"Surprise robustness — classifier resists mixing better than briefing predicts (mean acc@0.7 = {avg:.3f})."

print(f"\n=== Verdict: ({result['verdict']}) ===")
print(result["verdict_text"])

with (OUT / "03_briefing_check.json").open("w") as f:
    json.dump(result, f, indent=2, default=float)
print(f"wrote {OUT / '03_briefing_check.json'}")

print("\n=== Stage 7 DONE ===")
