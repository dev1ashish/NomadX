"""Temperature-scaled soft-vote across per-fold predictions.

The mechanism finding from `plan/07§2026-05-15--4-architecture-re-ensemble-fails-to-beat-plsda`:
PLS-DA's mean spectrum-level max-proba is 0.747 vs 0.43-0.49 for the three
deep models. In a uniform soft-vote PLS-DA's vote effectively carries ~1.7×
the weight of any deep model. This module fits a per-base scalar temperature
T_b on the cross-fold predictions, applies it to each base's probas, then
soft-votes.

Calibration fit is LOO-on-strains: for held-out strain X, fit T_b on the
union of predictions from the OTHER 8 LOSO folds' parquets. This carries
the same leakage tolerance as the stacking meta-learner — bounded by the
fact that all 9 LOSO checkpoints share the same training pipeline.

Outputs use the standard predictions_fold_<strain>.parquet schema so
downstream evaluation works unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from atlas.evaluate import PRIMARY_CLASSES


PROBA_COLS = [f"p_{c}" for c in PRIMARY_CLASSES]
META_COLS = ["subclass", "primary_true", "fold_id"]
EPS = 1e-12


def _probas_to_logits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0)
    return np.log(p)


def _temp_softmax(logits: np.ndarray, T: float) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _nll(T: float, logits: np.ndarray, y_idx: np.ndarray) -> float:
    p = _temp_softmax(logits, T)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y_idx)), y_idx], EPS, 1.0))))


def fit_temperature(probas: np.ndarray, y_true_str: np.ndarray) -> float:
    """Fit scalar T ∈ [0.05, 20.0] to minimize NLL on (proba, y) pairs."""
    if len(probas) == 0:
        return 1.0
    logits = _probas_to_logits(probas)
    y_idx = np.array([PRIMARY_CLASSES.index(y) for y in y_true_str])
    res = minimize_scalar(
        _nll, args=(logits, y_idx), bounds=(0.05, 20.0), method="bounded"
    )
    return float(res.x)


def apply_temperature(probas: np.ndarray, T: float) -> np.ndarray:
    return _temp_softmax(_probas_to_logits(probas), T)


def load_fold_predictions(run_dir: Path) -> dict[str, pd.DataFrame]:
    """{fold_id (strain str) -> per-spectrum DataFrame}."""
    out: dict[str, pd.DataFrame] = {}
    for p in sorted(Path(run_dir).glob("predictions_fold_*.parquet")):
        fold_id = p.stem[len("predictions_fold_"):]
        out[fold_id] = pd.read_parquet(p)
    if not out:
        raise FileNotFoundError(f"no predictions_fold_*.parquet under {run_dir}")
    return out


def temperature_calibrated_softvote_loso(
    base_run_dirs: list[Path | str],
    base_names: list[str] | None = None,
    weights: list[float] | None = None,
    exclude_in_softvote: list[str] | None = None,
) -> dict:
    """Run LOO-on-strains temperature-calibrated soft-vote.

    For each held-out fold X:
      For each base b:
        Fit T_b on the union of predictions from the OTHER 8 folds.
        Apply T_b to base b's predictions for fold X.
      Soft-vote the 4 calibrated proba arrays (uniform weights by default).
      Output per-fold predictions parquet in the standard schema.

    Parameters
    ----------
    base_run_dirs : list of paths
    base_names : optional names parallel to base_run_dirs
    weights : optional uniform weights
    exclude_in_softvote : optional list of base_names to fit T for but NOT
        include in the averaged soft-vote. Useful for the "PLS-DA-excluded
        sanity check" — fit calibration for all 4 (for diagnostic), but only
        average the 3 deep models.

    Returns
    -------
    dict with:
      'predictions': dict[fold_id -> per-spectrum DataFrame in standard schema]
      'temperatures': dict[fold_id -> dict[base_name -> T]]
      'per_strain_parent_recall': dict[fold_id -> file-level parent recall]
      'mean_parent_recall': float
    """
    base_run_dirs = [Path(d) for d in base_run_dirs]
    if base_names is None:
        base_names = [d.name for d in base_run_dirs]
    if exclude_in_softvote is None:
        exclude_in_softvote = []

    per_base_preds = {n: load_fold_predictions(d)
                      for n, d in zip(base_names, base_run_dirs)}
    fold_ids = sorted(per_base_preds[base_names[0]].keys())
    for n in base_names[1:]:
        assert set(per_base_preds[n].keys()) == set(fold_ids), \
            f"fold-id mismatch between {base_names[0]} and {n}"

    softvote_names = [n for n in base_names if n not in exclude_in_softvote]
    if weights is None:
        weights = [1.0 / len(softvote_names)] * len(softvote_names)
    elif len(weights) != len(softvote_names):
        raise ValueError("weights length must match the included base count")

    out_predictions: dict[str, pd.DataFrame] = {}
    temperatures: dict[str, dict[str, float]] = {}
    per_strain_recall: dict[str, float] = {}

    for held in fold_ids:
        T_per_base = {}
        calibrated_for_held: dict[str, np.ndarray] = {}

        # Reference DataFrame for the held-out strain (rows / metadata).
        ref_df = per_base_preds[base_names[0]][held].reset_index(drop=True)

        for n in base_names:
            # Fit T_b on the OTHER 8 folds for base b.
            other_frames = [per_base_preds[n][f] for f in fold_ids if f != held]
            cal_df = pd.concat(other_frames, axis=0, ignore_index=True)
            T = fit_temperature(
                cal_df[PROBA_COLS].to_numpy(),
                cal_df["primary_true"].to_numpy(),
            )
            T_per_base[n] = T

            # Apply T_b to the held-out fold's predictions (aligned to ref_df).
            df = per_base_preds[n][held]
            # Reindex to ref_df's (spectrum_id, file_id) order
            merged = ref_df[["spectrum_id", "file_id"]].merge(
                df[["spectrum_id", "file_id"] + PROBA_COLS],
                on=["spectrum_id", "file_id"], how="inner",
            )
            assert len(merged) == len(ref_df), \
                f"row drop on fold={held} base={n}: {len(ref_df)} -> {len(merged)}"
            calibrated_for_held[n] = apply_temperature(
                merged[PROBA_COLS].to_numpy(), T
            )

        temperatures[held] = T_per_base

        # Soft-vote the included bases.
        accum = np.zeros((len(ref_df), len(PROBA_COLS)), dtype=np.float64)
        for w, n in zip(weights, softvote_names):
            accum += w * calibrated_for_held[n]
        # Renormalize defensively.
        row_sum = accum.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        accum = accum / row_sum

        # Build the predictions_fold_<strain>.parquet schema.
        out_df = ref_df[["spectrum_id", "file_id"] + META_COLS].copy()
        for k, c in enumerate(PROBA_COLS):
            out_df[c] = accum[:, k]
        out_predictions[held] = out_df

        # File-level parent recall for this fold.
        file_proba = out_df.groupby("file_id")[PROBA_COLS].mean()
        file_true = out_df.groupby("file_id")["primary_true"].first()
        pred = np.array([PRIMARY_CLASSES[i]
                         for i in file_proba.to_numpy().argmax(axis=1)])
        per_strain_recall[held] = float((pred == file_true.to_numpy()).mean())

    mean_recall = float(np.mean(list(per_strain_recall.values())))

    return {
        "predictions": out_predictions,
        "temperatures": temperatures,
        "per_strain_parent_recall": per_strain_recall,
        "mean_parent_recall": mean_recall,
        "softvote_names": softvote_names,
    }
