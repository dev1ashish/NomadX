"""Stage 15F — full-feature LOSO classifier + production-model serialization.

Pre-registered in `plan/experiments/2026-05-18_stage15f_full_classifier.md`.
See plan/16 §Phase 1 for the deploy-plan context.

Pipeline
--------
1. Load 4 feature caches + spec_df + qc_mask.
2. Aggregate per-pixel features (band 166 + spectral 51) to file level via
   mean-pool. Join the 2 already-file-level caches (unmix, spatial).
3. Run strain-level LOSO (9 bacterial subclasses + 1 H2O fold = 10 folds).
   Per fold:
     - Refit MCR-ALS (K=7), ROI-PCA, SAM templates on TRAIN pixels only.
     - Re-aggregate held-out file's pixel features via the per-fold fits.
     - MI feature selection to 30-40 features on train file matrix.
     - Train 3 classifiers (PLS-DA / LogReg-L2 / XGBoost) in a
       StandardScaler pipeline.
     - Predict on held-out file(s).
4. Run for 5 seeds; aggregate per-strain recall + mean parent-class recall
   per algorithm.
5. R7 permutation test on mcr_C1_* features (shuffled-vs-real importance).
6. Pick best algorithm by mean LOSO recall.
7. Refit the best algorithm on all 87 files with a consensus MI-feature set
   (majority vote across seeds) → save production artifact.
8. Persist: classifier, feature_columns, mcr_global, roi_pca, sam_templates, metadata.

Run: `.venv/bin/python scripts/run_stage15f_final.py`
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from atlas.spectral_features import (
    DEFAULT_ROI_PCA,
    fit_roi_pca,
    fit_sam_templates,
    transform_roi_pca,
    transform_sam,
    LPS_REGION_FOR_SAM,
)
from atlas.unmix_features import MCRALSWrapper, mcr_concentration_summary

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data_cache"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# First run (2026-05-18) verified seeds 0–4 give identical results (std=0.000):
# MCR-ALS SIMPLISMA init is deterministic on the same X; sklearn PCA/PLS/LogReg
# are deterministic; XGB tree_method=hist is deterministic given same data.
# Multi-seed adds no variance signal here — kept the multi-seed scaffolding in
# place for future stochastic-pipeline variants but drop to 1 seed for re-runs.
SEEDS = [0]
MI_K = 35                 # target after MI selection (in 30-40 band)
MCR_K = 7                 # effective K per Stage 15C
PRIMARY_CLASSES = ("STEC", "Non-STEC", "Salmonella", "H2O")

# Collects per-fold prediction rows for post-hoc analysis (confusion matrix,
# per-class F1, bootstrap CIs, McNemar test).
PREDICTION_ROWS: list = []


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def load_all() -> dict:
    """Load every input and return as a dict with aligned shapes."""
    print("[load] reading caches...")
    spec_df = pd.read_parquet(CACHE / "spectra.parquet")
    qc_mask = np.load(CACHE / "qc_mask.npy")
    spec_qc = spec_df.loc[qc_mask].reset_index(drop=True)
    X_pp = np.load(CACHE / "spectra_array_preprocessed.npy")[qc_mask]
    wn = np.load(CACHE / "wavenumber_axis_preprocessed.npy")

    band_pix = pd.read_parquet(CACHE / "band_features.parquet")
    spec_pix = pd.read_parquet(CACHE / "spectral_features.parquet")
    unmix_file = pd.read_parquet(CACHE / "unmix_features.parquet")
    spat_file = pd.read_parquet(CACHE / "spatial_features.parquet")

    meta = pd.read_parquet(CACHE / "metadata.parquet").set_index("file_id")
    # primary_class + subclass per file_id
    meta_kept = meta.loc[spec_qc["file_id"].unique()].copy()

    assert band_pix.shape[0] == spec_qc.shape[0] == X_pp.shape[0]
    assert spec_pix.shape[0] == spec_qc.shape[0]

    print(f"[load] pixels (QC-passed): {len(spec_qc)}, files: {len(meta_kept)}, "
          f"band cols: {band_pix.shape[1]}, spectral cols: {spec_pix.shape[1]}, "
          f"unmix file-level cols: {unmix_file.shape[1]}, spatial cols: {spat_file.shape[1]}")
    return dict(
        spec_qc=spec_qc, X_pp=X_pp, wn=wn,
        band_pix=band_pix, spec_pix=spec_pix,
        unmix_file=unmix_file, spat_file=spat_file,
        meta=meta_kept,
    )


def aggregate_pixel_to_file(per_pixel_df: pd.DataFrame,
                            file_ids: pd.Series) -> pd.DataFrame:
    """Mean-pool a per-pixel feature DataFrame to per-file."""
    df = per_pixel_df.copy()
    df["file_id"] = file_ids.values
    agg = df.groupby("file_id").mean(numeric_only=True)
    return agg


def build_file_level_matrix(data: dict,
                            mcr_concentrations: np.ndarray | None,
                            roi_scores: dict[str, np.ndarray] | None,
                            sam_angles: dict[str, np.ndarray] | None,
                            file_ids: pd.Series) -> pd.DataFrame:
    """Assemble the (n_files, n_features) DataFrame indexed by file_id.

    - Band features (166 cols) → mean-pool.
    - Spectral features (51 cols) → mean-pool but OVERRIDE the PCA + SAM
      columns with caller-supplied roi_scores / sam_angles (per-fold refit).
    - MCR-ALS columns are REPLACED entirely with mcr_concentration_summary
      computed from mcr_concentrations (when provided).
    - Spatial features (10 cols) — used as-is from cache (file-level moments
      computed from X, no train/test leakage concern).
    """
    # ----- per-pixel → per-file mean-pool for band features -----
    band_file = aggregate_pixel_to_file(data["band_pix"], file_ids)

    # ----- spectral cache: split into DWT (kept) + PCA/SAM (replaced) -----
    spec_pix = data["spec_pix"]
    dwt_cols = [c for c in spec_pix.columns if c.startswith("dwt_")]
    spec_dwt_file = aggregate_pixel_to_file(spec_pix[dwt_cols], file_ids)

    # Build per-fold PCA + SAM matrices and aggregate to file level
    pca_cols_per_pixel = pd.DataFrame(roi_scores) if roi_scores else pd.DataFrame()
    sam_cols_per_pixel = pd.DataFrame(sam_angles) if sam_angles else pd.DataFrame()
    pca_sam = pd.concat([pca_cols_per_pixel, sam_cols_per_pixel], axis=1)
    pca_sam_file = aggregate_pixel_to_file(pca_sam, file_ids) if not pca_sam.empty else pd.DataFrame(index=band_file.index)

    # ----- MCR concentrations → file-level summary (mean/std/max/p90 per K) -----
    if mcr_concentrations is not None:
        mcr_file = mcr_concentration_summary(mcr_concentrations, file_ids.values)
    else:
        mcr_file = data["unmix_file"].drop(columns=["mcr_residual_norm_mean"], errors="ignore")

    # ----- Spatial cache (already file-level) -----
    spat_file = data["spat_file"]

    # ----- Join -----
    out = (
        band_file
        .join(spec_dwt_file, how="left")
        .join(pca_sam_file, how="left")
        .join(mcr_file, how="left")
        .join(spat_file, how="left")
    )
    out = out.dropna(axis=1, how="all")
    # Fill any remaining NaNs with column median (a few `bio_*` ratios can NaN
    # on degenerate denominators).
    out = out.fillna(out.median(numeric_only=True)).fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# LOSO fold definition
# ---------------------------------------------------------------------------

def define_folds(meta: pd.DataFrame) -> list[tuple[str, list[str]]]:
    """Return [(fold_name, [file_ids in held-out fold])] for 10 LOSO folds.

    Folds: one per bacterial subclass (9 subclasses) + one H2O fold.
    """
    folds: list[tuple[str, list[str]]] = []
    for sub, grp in meta.groupby("subclass", dropna=False):
        if pd.isna(sub):
            continue
        folds.append((str(sub), grp.index.tolist()))
    # H2O fold (no subclass)
    h2o_ids = meta[meta["primary_class"] == "H2O"].index.tolist()
    folds.append(("H2O", h2o_ids))
    # Sort by name for deterministic ordering
    folds.sort(key=lambda kv: kv[0])
    return folds


# ---------------------------------------------------------------------------
# Per-fold feature engineering refit
# ---------------------------------------------------------------------------

def fit_perfold_features(X_pp: np.ndarray, wn: np.ndarray,
                         spec_qc: pd.DataFrame,
                         train_mask_pix: np.ndarray,
                         seed: int) -> dict:
    """Fit MCR-ALS, ROI-PCA, SAM on train pixels. Return the fitted objects
    plus full-corpus transformed outputs (concentrations / scores / angles)."""
    X_train = X_pp[train_mask_pix]
    # MCR-ALS needs non-negative data; SNV cache has negatives → shift by global min
    X_all_offset = X_pp - X_pp.min()
    X_train_offset = X_all_offset[train_mask_pix]

    mcr = MCRALSWrapper(n_components=MCR_K, max_iter=80, offset_pct=5.0,
                        random_state=seed)
    mcr.fit(X_train_offset)
    mcr_C_all = mcr.transform(X_all_offset)                      # (N, MCR_K)

    roi_fitted = fit_roi_pca(X_train, wn, regions=DEFAULT_ROI_PCA,
                             random_state=seed)
    roi_scores_all = transform_roi_pca(X_pp, wn, roi_fitted)

    # SAM uses labels — only train pixels' labels go in
    y_primary_train = spec_qc.loc[train_mask_pix, "primary_class"].values
    y_sub_train = spec_qc.loc[train_mask_pix, "subclass"].fillna("H2O").values
    sam_templates = fit_sam_templates(
        X_train, y_primary_train, y_sub_train,
        wn=wn, region=LPS_REGION_FOR_SAM,
    )
    sam_angles_all = transform_sam(X_pp, sam_templates, wn=wn)

    return dict(mcr=mcr, mcr_C=mcr_C_all,
                roi_fitted=roi_fitted, roi_scores=roi_scores_all,
                sam_templates=sam_templates, sam_angles=sam_angles_all)


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

class PLSDAClassifier:
    """One-hot PLS regression with argmax decoding."""

    def __init__(self, n_components: int = 5, random_state: int = 0):
        self.n_components = n_components
        self.random_state = random_state
        self.classes_: np.ndarray | None = None
        self._pls: PLSRegression | None = None

    def fit(self, X, y):
        self.classes_ = np.array(sorted(set(y)))
        Y = np.zeros((len(y), len(self.classes_)), dtype=np.float64)
        for i, c in enumerate(self.classes_):
            Y[y == c, i] = 1.0
        nc = min(self.n_components, X.shape[1], X.shape[0] - 1)
        self._pls = PLSRegression(n_components=max(1, nc), scale=False)
        self._pls.fit(X, Y)
        return self

    def predict(self, X):
        scores = self._pls.predict(X)
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        scores = self._pls.predict(X)
        scores = scores - scores.min(axis=1, keepdims=True)
        denom = scores.sum(axis=1, keepdims=True) + 1e-9
        return scores / denom


def make_pipeline(name: str, seed: int) -> Pipeline:
    if name == "plsda":
        clf = PLSDAClassifier(n_components=5, random_state=seed)
    elif name == "logreg":
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
                                 solver="lbfgs", multi_class="auto",
                                 random_state=seed)
    elif name == "xgb":
        clf = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            eval_metric="mlogloss", random_state=seed,
            tree_method="hist", verbosity=0,
        )
    else:
        raise ValueError(name)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def fit_predict(pipe: Pipeline, X_train, y_train, X_test):
    """Fit, then predict on X_test. Returns (y_pred, y_proba, class_order).

    Handles XGB's string-label requirement and PLS-DA's homegrown proba.
    """
    clf = pipe.named_steps["clf"]
    if isinstance(clf, XGBClassifier):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(y_train)
        pipe.fit(X_train, le.transform(y_train))
        yhat = le.inverse_transform(pipe.predict(X_test))
        try:
            proba = pipe.predict_proba(X_test)
            class_order = list(le.inverse_transform(np.arange(len(le.classes_))))
        except Exception:
            proba, class_order = None, list(le.classes_)
    else:
        pipe.fit(X_train, y_train)
        yhat = pipe.predict(X_test)
        try:
            proba = pipe.predict_proba(X_test)
            class_order = list(pipe.named_steps["clf"].classes_)
        except Exception:
            proba, class_order = None, list(np.unique(y_train))
    return yhat, proba, class_order


# ---------------------------------------------------------------------------
# LOSO loop
# ---------------------------------------------------------------------------

def run_loso(data: dict, folds: list[tuple[str, list[str]]],
             algos: tuple[str, ...] = ("plsda", "logreg", "xgb"),
             seeds: list[int] = SEEDS,
             mi_k: int = MI_K) -> dict:
    """Returns dict with per-algo, per-seed, per-fold predictions + recalls."""
    print(f"[loso] running {len(seeds)} seeds × {len(folds)} folds × {len(algos)} algos")
    spec_qc = data["spec_qc"]
    meta = data["meta"]
    n_files = len(meta)

    # per-pixel → file_id Series for aggregation
    pix_file_ids = spec_qc["file_id"]

    # Track results
    results: dict = {a: {"per_fold": [], "feature_lists": []} for a in algos}
    summary_rows = []

    t0 = time.time()
    for seed in seeds:
        rng_seed = seed
        for fold_idx, (fold_name, test_file_ids) in enumerate(folds):
            test_set = set(test_file_ids)
            train_pix_mask = ~spec_qc["file_id"].isin(test_set).values

            # Per-fold MCR/ROI-PCA/SAM refit
            tf = fit_perfold_features(
                data["X_pp"], data["wn"], spec_qc, train_pix_mask, rng_seed,
            )

            # Assemble file-level matrix using these fold-specific fits
            file_df = build_file_level_matrix(
                data,
                mcr_concentrations=tf["mcr_C"],
                roi_scores=tf["roi_scores"],
                sam_angles=tf["sam_angles"],
                file_ids=pix_file_ids,
            )
            # Align to metadata order
            file_df = file_df.reindex(meta.index)

            y = meta["primary_class"].values
            test_idx = meta.index.get_indexer(test_file_ids)
            train_idx = np.array([i for i in range(n_files) if i not in set(test_idx)])

            X_all = file_df.values.astype(np.float64)
            X_tr, X_te = X_all[train_idx], X_all[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # MI selection on train
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mi = mutual_info_classif(X_tr, y_tr, random_state=rng_seed,
                                         n_neighbors=3)
            top_k = min(mi_k, X_tr.shape[1])
            sel = np.argsort(mi)[::-1][:top_k]
            sel_cols = file_df.columns[sel].tolist()
            X_tr_s, X_te_s = X_tr[:, sel], X_te[:, sel]

            for algo in algos:
                pipe = make_pipeline(algo, rng_seed)
                yhat, proba, class_order = fit_predict(pipe, X_tr_s, y_tr, X_te_s)
                rec = recall_score(y_te, yhat,
                                   labels=PRIMARY_CLASSES,
                                   average="macro", zero_division=0)
                # Dump per-file predictions so we can compute the real
                # confusion matrix / per-class F1 / bootstrap CIs offline.
                test_file_id_list = list(test_file_ids)
                for i, fid in enumerate(test_file_id_list):
                    row = {
                        "seed": seed, "fold": fold_name, "algo": algo,
                        "file_id": fid,
                        "y_true": str(y_te[i]),
                        "y_pred": str(yhat[i]),
                    }
                    # Per-class probabilities (column order = class_order)
                    if proba is not None:
                        for j, c in enumerate(class_order):
                            row[f"proba_{c}"] = float(proba[i, j])
                    PREDICTION_ROWS.append(row)
                # For per-strain reporting, also compute correct count
                results[algo]["per_fold"].append({
                    "seed": seed, "fold": fold_name,
                    "n_test": len(y_te), "n_correct": int((yhat == y_te).sum()),
                    "y_true": y_te.tolist(), "y_pred": yhat.tolist(),
                    "macro_recall": float(rec),
                })
                results[algo]["feature_lists"].append({
                    "seed": seed, "fold": fold_name, "features": sel_cols,
                })
                summary_rows.append({
                    "seed": seed, "fold": fold_name, "algo": algo,
                    "accuracy": float((yhat == y_te).mean()),
                    "macro_recall": float(rec),
                })
            print(f"  seed={seed} fold={fold_name:<13} "
                  + " ".join(f"{a}={summary_rows[-len(algos)+i]['accuracy']:.2f}"
                             for i, a in enumerate(algos))
                  + f"  ({time.time()-t0:.1f}s)")
    print(f"[loso] done in {time.time()-t0:.1f}s")
    return {"results": results, "summary": pd.DataFrame(summary_rows)}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def algo_summary(summary: pd.DataFrame, algo: str) -> dict:
    """Per-strain recall + mean parent-class recall across seeds."""
    sub = summary[summary["algo"] == algo]
    per_strain = sub.groupby("fold")["accuracy"].mean().to_dict()
    overall_correct = 0
    overall_total = 0
    for _, row in sub.iterrows():
        # we approximate parent-class recall as plain accuracy averaged across
        # folds; the y_true within a fold is constant (all same strain → same
        # parent class), so per-fold accuracy ≡ per-fold recall of the held-out
        # class.
        # Mean parent-class recall = mean across folds of per-fold accuracy.
        pass
    fold_means_per_seed = sub.groupby("seed")["accuracy"].mean()
    mean_loso = float(fold_means_per_seed.mean())
    std_loso = float(fold_means_per_seed.std(ddof=1))
    macro = float(sub.groupby("seed")["macro_recall"].mean().mean())
    return {
        "per_strain": per_strain,
        "mean_loso_accuracy": mean_loso,
        "std_loso_accuracy": std_loso,
        "mean_loso_macro_recall": macro,
    }


def consensus_features(results: dict, algo: str, top_n: int = MI_K) -> list[str]:
    """Majority-vote feature set across seeds × folds.

    Returns the top_n features by frequency of appearance in the per-fold MI
    feature lists (across all 5 seeds × 10 folds = 50 lists).
    """
    counter: Counter[str] = Counter()
    for entry in results[algo]["feature_lists"]:
        counter.update(entry["features"])
    return [name for name, _ in counter.most_common(top_n)]


# ---------------------------------------------------------------------------
# R7 — shuffled-label permutation test on mcr_C1_*
# ---------------------------------------------------------------------------

def r7_permutation_test(data: dict, folds: list, best_algo: str,
                        feature_set: list[str], seed: int = 0) -> dict:
    """Compare real vs shuffled-label feature importance for mcr_C1_* cols.

    Refits MCR-ALS once globally (saves time), assembles full file matrix once,
    then runs 10 shuffles. Reports the ratio of mean(shuffled_importance) /
    mean(real_importance) for each mcr_C1_* feature.
    """
    print("[R7] permutation test on mcr_C1_* features")
    spec_qc = data["spec_qc"]
    meta = data["meta"]
    X_offset = data["X_pp"] - data["X_pp"].min()

    mcr = MCRALSWrapper(n_components=MCR_K, max_iter=80, offset_pct=5.0,
                        random_state=seed)
    mcr.fit(X_offset)
    mcr_C = mcr.transform(X_offset)
    roi_fitted = fit_roi_pca(data["X_pp"], data["wn"],
                             regions=DEFAULT_ROI_PCA, random_state=seed)
    roi_scores = transform_roi_pca(data["X_pp"], data["wn"], roi_fitted)
    sam_templates = fit_sam_templates(
        data["X_pp"],
        spec_qc["primary_class"].values,
        spec_qc["subclass"].fillna("H2O").values,
        wn=data["wn"], region=LPS_REGION_FOR_SAM,
    )
    sam_angles = transform_sam(data["X_pp"], sam_templates, wn=data["wn"])

    file_df = build_file_level_matrix(
        data, mcr_concentrations=mcr_C,
        roi_scores=roi_scores, sam_angles=sam_angles,
        file_ids=spec_qc["file_id"],
    ).reindex(meta.index)
    feats_present = [c for c in feature_set if c in file_df.columns]
    X = file_df[feats_present].values.astype(np.float64)
    y = meta["primary_class"].values
    mcr_c1_cols = [c for c in feats_present if c.startswith("mcr_C1_")]
    if not mcr_c1_cols:
        return {"verdict": "skipped — no mcr_C1_* in selected feature set",
                "ratios": {}, "drop": []}

    # Use a fast estimator for importance: XGB with feature_importances_
    def importances(yy: np.ndarray) -> np.ndarray:
        pipe = make_pipeline("xgb", seed)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(yy)
        pipe.fit(X, le.transform(yy))
        return pipe.named_steps["clf"].feature_importances_

    real_imp = importances(y)
    shuffled_imp = np.zeros_like(real_imp)
    rng = np.random.default_rng(seed)
    n_perm = 10
    for _ in range(n_perm):
        yperm = rng.permutation(y)
        shuffled_imp += importances(yperm)
    shuffled_imp /= n_perm

    ratios = {}
    drop_list = []
    for c in mcr_c1_cols:
        i = feats_present.index(c)
        denom = real_imp[i] if real_imp[i] > 1e-9 else 1e-9
        ratio = float(shuffled_imp[i] / denom)
        ratios[c] = ratio
        if ratio >= 0.5:
            drop_list.append(c)
    verdict = "pass" if not drop_list else f"DROP: {drop_list}"
    print(f"[R7] verdict: {verdict}  ratios: {ratios}")
    return {"verdict": verdict, "ratios": ratios, "drop": drop_list}


# ---------------------------------------------------------------------------
# Production fit
# ---------------------------------------------------------------------------

def fit_production_model(data: dict, best_algo: str,
                         feature_set: list[str],
                         seed: int = 0) -> dict:
    """Fit on all 87 files using a global MCR/ROI-PCA/SAM and the consensus
    feature set, return all artifacts."""
    print(f"[prod] fitting production {best_algo} on all 87 files")
    spec_qc = data["spec_qc"]
    meta = data["meta"]
    X_offset = data["X_pp"] - data["X_pp"].min()

    mcr = MCRALSWrapper(n_components=MCR_K, max_iter=80, offset_pct=5.0,
                        random_state=seed)
    mcr.fit(X_offset)
    mcr_C = mcr.transform(X_offset)
    roi_fitted = fit_roi_pca(data["X_pp"], data["wn"],
                             regions=DEFAULT_ROI_PCA, random_state=seed)
    roi_scores = transform_roi_pca(data["X_pp"], data["wn"], roi_fitted)
    sam_templates = fit_sam_templates(
        data["X_pp"],
        spec_qc["primary_class"].values,
        spec_qc["subclass"].fillna("H2O").values,
        wn=data["wn"], region=LPS_REGION_FOR_SAM,
    )
    sam_angles = transform_sam(data["X_pp"], sam_templates, wn=data["wn"])

    file_df = build_file_level_matrix(
        data, mcr_concentrations=mcr_C,
        roi_scores=roi_scores, sam_angles=sam_angles,
        file_ids=spec_qc["file_id"],
    ).reindex(meta.index)

    feats_present = [c for c in feature_set if c in file_df.columns]
    X = file_df[feats_present].values.astype(np.float64)
    y = meta["primary_class"].values

    pipe = make_pipeline(best_algo, seed)
    if best_algo == "xgb":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(y)
        pipe.fit(X, le.transform(y))
        # Wrap pipeline with the LabelEncoder for inference path.
        pipe = _LabelEncodedPipeline(pipe, le)
    else:
        pipe.fit(X, y)

    return dict(
        pipeline=pipe,
        feature_columns=feats_present,
        mcr=mcr, roi_fitted=roi_fitted, sam_templates=sam_templates,
    )


class _LabelEncodedPipeline:
    """Wraps a sklearn pipeline that was trained on label-encoded y to expose
    predict / predict_proba returning the original string labels."""

    def __init__(self, pipeline: Pipeline, label_encoder):
        self.pipeline = pipeline
        self.label_encoder = label_encoder
        self.classes_ = label_encoder.classes_

    def predict(self, X):
        return self.label_encoder.inverse_transform(self.pipeline.predict(X))

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data = load_all()
    folds = define_folds(data["meta"])
    print(f"[folds] {len(folds)}: " + ", ".join(f for f, _ in folds))

    out = run_loso(data, folds, mi_k=MI_K)
    summary = out["summary"]
    results = out["results"]

    print("\n[loso-summary]")
    algo_stats = {}
    for algo in ("plsda", "logreg", "xgb"):
        s = algo_summary(summary, algo)
        algo_stats[algo] = s
        print(f"  {algo:<7} mean_loso_acc={s['mean_loso_accuracy']:.3f} ± "
              f"{s['std_loso_accuracy']:.3f}  macro_recall={s['mean_loso_macro_recall']:.3f}")
        for strain, acc in sorted(s["per_strain"].items()):
            print(f"      {strain:<13} {acc:.3f}")

    best_algo = max(algo_stats, key=lambda a: algo_stats[a]["mean_loso_macro_recall"])
    print(f"\n[best] {best_algo}")

    feats = consensus_features(results, best_algo, top_n=MI_K)
    print(f"[consensus] {len(feats)} features. first 10: {feats[:10]}")

    r7 = r7_permutation_test(data, folds, best_algo, feats, seed=0)
    if r7["drop"]:
        feats = [f for f in feats if f not in r7["drop"]]
        print(f"[R7] dropped {len(r7['drop'])} mcr_C1_* features → {len(feats)} remain")

    prod = fit_production_model(data, best_algo, feats, seed=0)

    # ----- Persist artifacts -----
    joblib.dump(prod["pipeline"], ARTIFACTS / "stage15f_classifier.joblib")
    (ARTIFACTS / "stage15f_feature_columns.json").write_text(
        json.dumps(prod["feature_columns"], indent=2)
    )
    joblib.dump(prod["mcr"], ARTIFACTS / "stage15f_mcr_global.joblib")
    joblib.dump(prod["roi_fitted"], ARTIFACTS / "stage15f_roi_pca.joblib")
    joblib.dump(prod["sam_templates"], ARTIFACTS / "stage15f_sam_templates.joblib")

    branch_hit = (
        "A" if algo_stats[best_algo]["mean_loso_macro_recall"] >= 0.55
            and algo_stats[best_algo]["per_strain"].get("K-12", 0) >= 0.75
        else "B" if algo_stats[best_algo]["mean_loso_macro_recall"] >= 0.50
            or algo_stats[best_algo]["per_strain"].get("K-12", 0) >= 0.75
        else "C"
    )
    meta_json = {
        "model_type": best_algo,
        "loso_mean_accuracy": algo_stats[best_algo]["mean_loso_accuracy"],
        "loso_std_accuracy": algo_stats[best_algo]["std_loso_accuracy"],
        "loso_mean_macro_recall": algo_stats[best_algo]["mean_loso_macro_recall"],
        "per_strain_accuracy": algo_stats[best_algo]["per_strain"],
        "algo_comparison": {a: {
            "mean_loso_accuracy": algo_stats[a]["mean_loso_accuracy"],
            "mean_loso_macro_recall": algo_stats[a]["mean_loso_macro_recall"],
        } for a in algo_stats},
        "feature_count": len(prod["feature_columns"]),
        "feature_columns": prod["feature_columns"],
        "r7_perm_test": r7,
        "training_date": pd.Timestamp.utcnow().isoformat(),
        "branch_hit": branch_hit,
        "n_seeds": len(SEEDS),
        "n_folds": len(folds),
        "n_files": int(len(data["meta"])),
        "mcr_K": MCR_K,
    }
    (ARTIFACTS / "stage15f_metadata.json").write_text(
        json.dumps(meta_json, indent=2, default=str)
    )
    # Write the per-fold summary CSV alongside (useful for paper §6)
    summary.to_csv(ARTIFACTS / "stage15f_loso_summary.csv", index=False)

    # Dump per-fold predictions for confusion matrix / per-class F1 / CIs / McNemar
    if PREDICTION_ROWS:
        pred_df = pd.DataFrame(PREDICTION_ROWS)
        pred_df.to_parquet(ARTIFACTS / "stage15f_loso_predictions.parquet",
                           compression="snappy", index=False)
        print(f"[done] {len(pred_df)} per-fold predictions → "
              f"stage15f_loso_predictions.parquet")
    print(f"\n[done] artifacts → {ARTIFACTS}")
    print(f"[done] branch_hit = ({branch_hit})")
    print(f"[done] mean_loso_macro_recall = {algo_stats[best_algo]['mean_loso_macro_recall']:.3f}")


if __name__ == "__main__":
    main()
