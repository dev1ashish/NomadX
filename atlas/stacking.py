"""Stacking meta-learner over pre-computed per-fold base-model predictions.

The single most-important difference from `atlas/ensemble.py` (which does
uniform soft-voting): a meta-learner is TRAINED to combine base-model
predictions, so it can USE the disagreement structure rather than averaging
it away. The ensemble code averaged DANN's K-12 win at 0.50 proba with
PLS-DA's confidently-wrong Salmonella vote and produced argmax=Salmonella;
the meta-learner sees "DANN says X with proba P_dann; PLS-DA says Y with
proba P_plsda" and can learn "trust DANN when DANN says X-with-high-P".

Three design choices baked in:

1. Stack at FILE level, not spectrum level. Within-file pixel cosine ≈ 0.997
   means spectrum-level stacking just learns to soft-vote within file. File
   level is the smaller, cleaner data regime (~70 training files per LOSO
   fold) and matches how the final prediction will be consumed.

2. Meta-features = each base model's file-level predict_proba (4 dims),
   concatenated across N base models. N=4 base models → 16 features per file.

3. Leave-one-strain-out cross-validation on the meta-learner too. For each
   held-out strain X, train meta on the 8 other strains' (file, base-probas,
   true-class) rows, predict the held-out strain. This is "stacking with
   out-of-fold predictions" — standard practice; the small remaining leakage
   (each base model's predictions on meta-train strain Y were made by a base
   model that itself held out Y, not X) is bounded and not worth nested CV
   at our sample size.

Inputs: a list of base-model run dirs, each containing predictions_fold_*.parquet
written by scripts/run_{classical,cnn,dann}.py. Outputs: per-fold predictions
parquet + ModelResult JSON in the standard atlas output schema, so downstream
evaluation code treats the stacker like any other model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from atlas.evaluate import PRIMARY_CLASSES


PROBA_COLS = [f"p_{c}" for c in PRIMARY_CLASSES]
N_CLASSES = len(PRIMARY_CLASSES)


@dataclass(frozen=True)
class StackingConfig:
    """Hyperparameters for the meta-learner.

    use_subclass:
        If True, include a one-hot subclass column as an additional feature.
        This is leakage-aware: under LOSO the held-out strain has its own
        subclass identifier that's never seen in training; the one-hot for
        that strain is 1 in test, but 0 in every training row. The meta-learner
        cannot learn anything strain-specific from "this is K-12" because
        every K-12 training row was held out. So enabling this gives the
        meta-learner a "this is an unseen strain" indicator without leaking
        labels. Default False to keep the first run interpretable.

    meta_C:
        LogReg regularization strength. Default 1.0; the meta-learner is in
        a small-data regime (~70 train files) with 16-D features, so default
        regularization is appropriate.

    use_class_weight:
        Balance class weights. Default True (consistent with rest of codebase).
    """

    use_subclass: bool = False
    meta_C: float = 1.0
    use_class_weight: bool = True


def aggregate_file_level_proba(per_spectrum_df: pd.DataFrame) -> pd.DataFrame:
    """Soft-vote a single base model's predictions to file level.

    Input: predictions_fold_*.parquet DataFrame with columns spectrum_id,
    file_id, subclass, primary_true, fold_id, p_*.

    Output: one row per file_id with columns file_id, subclass, primary_true,
    fold_id, p_H2O, p_Non-STEC, p_STEC, p_Salmonella (mean over the file's
    spectra).
    """
    file_proba = (
        per_spectrum_df
        .groupby("file_id", as_index=False)[PROBA_COLS]
        .mean()
    )
    # Carry through the file-level scalars (subclass, primary_true, fold_id).
    # These are constant within file by construction; .first() is unambiguous.
    scalars = (
        per_spectrum_df
        .groupby("file_id", as_index=False)[["subclass", "primary_true", "fold_id"]]
        .first()
    )
    return file_proba.merge(scalars, on="file_id")


def build_stacking_matrix(
    base_run_dirs: list[Path | str],
    base_names: list[str] | None = None,
) -> pd.DataFrame:
    """Build the file-level stacking dataset by concatenating each base model's
    file-aggregated predictions across folds.

    Returns one row per (file_id) per fold; columns:
        file_id, subclass, primary_true, fold_id,
        <base_0>_p_H2O, <base_0>_p_Non-STEC, <base_0>_p_STEC, <base_0>_p_Salmonella,
        <base_1>_p_H2O, ..., <base_N>_p_Salmonella

    Asserts each fold has identical (file_id) set across base models — if any
    base model's per-fold parquet is missing files, raises immediately.
    """
    base_run_dirs = [Path(d) for d in base_run_dirs]
    if base_names is None:
        base_names = [d.name for d in base_run_dirs]
    assert len(base_names) == len(base_run_dirs)

    per_model_frames: list[pd.DataFrame] = []
    for name, d in zip(base_names, base_run_dirs):
        file_frames = []
        for p in sorted(d.glob("predictions_fold_*.parquet")):
            spec_df = pd.read_parquet(p)
            file_df = aggregate_file_level_proba(spec_df)
            file_frames.append(file_df)
        if not file_frames:
            raise FileNotFoundError(f"no predictions_fold_*.parquet under {d}")
        all_folds = pd.concat(file_frames, axis=0, ignore_index=True)
        # Rename proba columns to namespace by base model
        all_folds = all_folds.rename(
            columns={c: f"{name}::{c}" for c in PROBA_COLS}
        )
        per_model_frames.append(all_folds)

    # Inner-merge on (file_id, subclass, primary_true, fold_id) across base
    # models. This is the right join because every file should appear in
    # every base model's predictions for the same fold; a row dropped here
    # means the base models disagree on which fold a file is in.
    base = per_model_frames[0]
    for next_df in per_model_frames[1:]:
        before = len(base)
        base = base.merge(
            next_df,
            on=["file_id", "subclass", "primary_true", "fold_id"],
            how="inner",
        )
        if len(base) != before:
            raise ValueError(
                f"stacking merge dropped rows: {before} -> {len(base)}. "
                f"Base models disagree on (file_id, fold_id) coverage."
            )
    return base


def stacking_feature_columns(base_names: list[str]) -> list[str]:
    """Return the ordered list of feature columns produced by build_stacking_matrix."""
    return [f"{n}::{c}" for n in base_names for c in PROBA_COLS]


def train_predict_loso(
    stack_df: pd.DataFrame,
    base_names: list[str],
    cfg: StackingConfig | None = None,
    log_fn=print,
) -> pd.DataFrame:
    """LOSO meta-CV: for each fold_id (strain) X, train meta on the other 8 strains'
    rows, predict the X rows. Returns the input frame augmented with `meta_p_<class>`
    columns and `meta_pred` (the argmax class string).

    Assumes the input was produced by build_stacking_matrix() — i.e. fold_id is
    the LOSO-held-out strain (string), one row per file per fold.
    """
    cfg = cfg or StackingConfig()
    feat_cols = stacking_feature_columns(base_names)
    if cfg.use_subclass:
        # One-hot the subclass column. Add to feat_cols.
        sub_dummies = pd.get_dummies(stack_df["subclass"].fillna("H2O_class"),
                                     prefix="sub")
        stack_df = pd.concat([stack_df, sub_dummies], axis=1)
        feat_cols = feat_cols + list(sub_dummies.columns)

    fold_ids = sorted(stack_df["fold_id"].unique())
    log_fn(f"[stacking] LOSO meta-CV over {len(fold_ids)} folds: {fold_ids}")
    log_fn(f"[stacking] feature columns ({len(feat_cols)}): {feat_cols[:8]} ...")

    out_rows = []
    for held in fold_ids:
        train_df = stack_df[stack_df["fold_id"] != held]
        test_df = stack_df[stack_df["fold_id"] == held]

        X_tr = train_df[feat_cols].to_numpy(dtype=np.float64)
        y_tr = train_df["primary_true"].to_numpy()
        X_te = test_df[feat_cols].to_numpy(dtype=np.float64)

        meta = LogisticRegression(
            C=cfg.meta_C,
            max_iter=2000,
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced" if cfg.use_class_weight else None,
            n_jobs=1,
            random_state=42,
        )
        meta.fit(X_tr, y_tr)

        proba = meta.predict_proba(X_te)
        # Reorder columns to canonical PRIMARY_CLASSES order
        proba_canonical = np.zeros((proba.shape[0], N_CLASSES), dtype=np.float64)
        for k, cls in enumerate(meta.classes_):
            idx = PRIMARY_CLASSES.index(cls)
            proba_canonical[:, idx] = proba[:, k]
        # Renormalize defensively
        row_sum = proba_canonical.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        proba_canonical = proba_canonical / row_sum

        pred = np.array([PRIMARY_CLASSES[i] for i in proba_canonical.argmax(axis=1)])

        out = test_df[["file_id", "subclass", "primary_true", "fold_id"]].copy()
        for k, cls in enumerate(PRIMARY_CLASSES):
            out[f"meta_p_{cls}"] = proba_canonical[:, k]
        out["meta_pred"] = pred
        out_rows.append(out)

        log_fn(f"  fold={held}: trained on {len(train_df)} files, predicted {len(test_df)} files")

    return pd.concat(out_rows, axis=0, ignore_index=True)


def per_strain_parent_recall(meta_preds_df: pd.DataFrame) -> dict:
    """Compute per-strain (fold_id) parent-class recall from a meta_preds DataFrame.

    Returns dict: {fold_id (strain): recall_float, ..., "MEAN": mean_float}.
    """
    rows = {}
    for strain in sorted(meta_preds_df["fold_id"].unique()):
        sub = meta_preds_df[meta_preds_df["fold_id"] == strain]
        correct = (sub["meta_pred"] == sub["primary_true"]).sum()
        rows[strain] = float(correct / len(sub))
    rows["MEAN"] = float(np.mean(list(rows.values())))
    return rows
