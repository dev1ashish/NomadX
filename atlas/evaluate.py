"""Shared evaluation primitives for classical + (later) CNN/Transformer.

Result schema:
    FoldResult — one fold's metrics, confusions, predictions path.
    ModelResult — list of FoldResult + aggregate mean ± SD.

Statistical reporting rules (per plan/03_architecture.md sec G):
    - Effective N for significance = number of files (87), not spectra (7,122).
      Within-file cosine ~0.997 means pixel-level CIs are wrong by ~9x.
    - Across-fold summary = mean +/- SD, labelled SD not CI.
    - LOSO summary = mean + (min, max), not SD.
    - Confusions = SUM of counts across folds (not averaged normalized).
    - Multi-class Brier = proper formulation, NOT averaged one-vs-rest.
    - Reliability diagrams = 10 quantile bins, equal-frequency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


PRIMARY_CLASSES = ["H2O", "Non-STEC", "STEC", "Salmonella"]
N_PRIMARY = len(PRIMARY_CLASSES)
SUBCLASSES_BACT = [
    "83972", "ATCC25922", "K-12", "Dublin", "Heidelburg", "Typhimurium",
    "O103H2", "O121H19", "O157H7",
]
SUBCLASS_ROWS_10x10 = SUBCLASSES_BACT + ["H2O"]


def class_to_idx(y: np.ndarray) -> np.ndarray:
    """Map string class labels to 0..3 in canonical PRIMARY_CLASSES order."""
    lookup = {c: i for i, c in enumerate(PRIMARY_CLASSES)}
    return np.array([lookup[c] for c in y], dtype=np.int32)


def file_aggregate_softvote(
    proba: np.ndarray, file_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-spectrum predict_proba -> per-file soft vote.

    Returns (unique_file_ids, file_proba [F, K], file_pred [F]).
    """
    unique = np.unique(file_ids)
    out = np.zeros((unique.size, proba.shape[1]), dtype=np.float64)
    for i, fid in enumerate(unique):
        out[i] = proba[file_ids == fid].mean(axis=0)
    pred = np.argmax(out, axis=1)
    return unique, out, pred


def multiclass_brier(proba: np.ndarray, y_true_idx: np.ndarray, n_classes: int) -> float:
    """Proper multi-class Brier: mean over samples of sum_k (p_k - y_k)^2.

    proba: (N, K) probability matrix.
    y_true_idx: (N,) integer labels.
    """
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true_idx)), y_true_idx] = 1.0
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


def reliability_quantile_bins(
    proba_top: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> list[dict]:
    """Reliability diagram points using equal-frequency (quantile) bins.

    proba_top: max-confidence per sample.
    correct: 1 if argmax matches truth, 0 otherwise.
    """
    quantiles = np.quantile(proba_top, np.linspace(0, 1, n_bins + 1))
    quantiles[0] -= 1e-9
    quantiles[-1] += 1e-9
    bins = np.digitize(proba_top, quantiles) - 1
    bins = np.clip(bins, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = bins == b
        if m.sum() == 0:
            continue
        out.append({
            "bin": int(b),
            "count": int(m.sum()),
            "mean_predicted": float(proba_top[m].mean()),
            "fraction_correct": float(correct[m].mean()),
        })
    return out


@dataclass
class FoldResult:
    fold_id: int | str
    protocol: str
    model_name: str

    # Spectrum-level metrics
    spectrum_macro_f1: float
    spectrum_balanced_acc: float
    spectrum_per_class_recall: dict[str, float]
    spectrum_brier: float

    # File-level (soft-vote) metrics — headline for our reporting
    file_macro_f1: float
    file_balanced_acc: float
    file_per_class_recall: dict[str, float]
    file_brier: float

    # Confusion matrices
    confusion_4x4_spectrum: list[list[int]]   # rows = true, cols = pred
    confusion_4x4_file: list[list[int]]
    confusion_10x10_spectrum: list[list[int]] # rows = 9 subclasses + H2O, cols = 4 primary preds
    confusion_10x10_file: list[list[int]]

    # Per-subclass recall under primary classification
    per_subclass_recall_spectrum: dict[str, float]
    per_subclass_recall_file: dict[str, float]

    # Calibration
    reliability_spectrum: list[dict]
    reliability_file: list[dict]

    # Bookkeeping
    n_train: int
    n_test: int
    n_train_files: int
    n_test_files: int
    training_time_s: float
    predictions_path: str | None = None
    best_hyperparams: dict | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class ModelResult:
    model_name: str
    protocol: str
    folds: list[FoldResult]
    # Aggregates: mean ± SD over folds at spectrum and file levels
    spectrum_macro_f1_mean: float = 0.0
    spectrum_macro_f1_sd: float = 0.0
    spectrum_macro_f1_per_fold: list[float] = field(default_factory=list)
    file_macro_f1_mean: float = 0.0
    file_macro_f1_sd: float = 0.0
    file_macro_f1_per_fold: list[float] = field(default_factory=list)
    # LOSO summary: also report range
    file_macro_f1_min: float = 0.0
    file_macro_f1_max: float = 0.0
    # Summed confusion matrices across folds
    confusion_4x4_spectrum_sum: list[list[int]] = field(default_factory=list)
    confusion_4x4_file_sum: list[list[int]] = field(default_factory=list)
    confusion_10x10_file_sum: list[list[int]] = field(default_factory=list)


def aggregate(folds: list[FoldResult], model_name: str, protocol: str) -> ModelResult:
    """Aggregate per-fold results into a ModelResult."""
    s_f1 = np.array([f.spectrum_macro_f1 for f in folds])
    f_f1 = np.array([f.file_macro_f1 for f in folds])

    c44s = np.sum([np.array(f.confusion_4x4_spectrum) for f in folds], axis=0)
    c44f = np.sum([np.array(f.confusion_4x4_file) for f in folds], axis=0)
    c10f = np.sum([np.array(f.confusion_10x10_file) for f in folds], axis=0)

    return ModelResult(
        model_name=model_name,
        protocol=protocol,
        folds=folds,
        spectrum_macro_f1_mean=float(s_f1.mean()),
        spectrum_macro_f1_sd=float(s_f1.std(ddof=1)) if len(s_f1) > 1 else 0.0,
        spectrum_macro_f1_per_fold=s_f1.tolist(),
        file_macro_f1_mean=float(f_f1.mean()),
        file_macro_f1_sd=float(f_f1.std(ddof=1)) if len(f_f1) > 1 else 0.0,
        file_macro_f1_per_fold=f_f1.tolist(),
        file_macro_f1_min=float(f_f1.min()),
        file_macro_f1_max=float(f_f1.max()),
        confusion_4x4_spectrum_sum=c44s.tolist(),
        confusion_4x4_file_sum=c44f.tolist(),
        confusion_10x10_file_sum=c10f.tolist(),
    )


def evaluate_fold(
    *,
    fold_id: int | str,
    protocol: str,
    model_name: str,
    y_true: np.ndarray,            # (N,) string labels
    y_proba: np.ndarray,           # (N, K) probability matrix
    file_ids: np.ndarray,          # (N,) file_id per spectrum
    subclass: np.ndarray,          # (N,) subclass per spectrum (None for H2O -> "H2O")
    n_train: int,
    n_train_files: int,
    n_test_files: int,
    training_time_s: float,
    best_hyperparams: dict | None = None,
    predictions_path: str | None = None,
    notes: list[str] | None = None,
) -> FoldResult:
    """Compute all per-fold metrics from (y_true, y_proba, file_ids, subclass)."""
    n = y_true.size
    y_true_idx = class_to_idx(y_true)
    y_pred_idx = np.argmax(y_proba, axis=1)
    y_pred = np.array([PRIMARY_CLASSES[i] for i in y_pred_idx])

    # Spectrum-level
    s_macro = f1_score(y_true, y_pred, labels=PRIMARY_CLASSES, average="macro", zero_division=0)
    s_bacc = balanced_accuracy_score(y_true, y_pred)
    s_per_class = recall_score(y_true, y_pred, labels=PRIMARY_CLASSES, average=None, zero_division=0)
    s_per_class_dict = {c: float(s_per_class[i]) for i, c in enumerate(PRIMARY_CLASSES)}
    s_brier = multiclass_brier(y_proba, y_true_idx, N_PRIMARY)
    s_conf_44 = confusion_matrix(y_true, y_pred, labels=PRIMARY_CLASSES).tolist()

    # 10x10: rows = 9 bacterial subclasses + H2O, cols = 4 primary predicted
    sub_labels = np.array([s if s is not None and s != "None" else "H2O" for s in subclass])
    s_conf_10 = np.zeros((len(SUBCLASS_ROWS_10x10), N_PRIMARY), dtype=int)
    for i, sub_label in enumerate(SUBCLASS_ROWS_10x10):
        m = sub_labels == sub_label
        if m.sum() == 0:
            continue
        for j, pri in enumerate(PRIMARY_CLASSES):
            s_conf_10[i, j] = int(((y_pred == pri) & m).sum())

    s_per_sub = {}
    for sub_label in SUBCLASS_ROWS_10x10:
        m = sub_labels == sub_label
        if m.sum() == 0:
            s_per_sub[sub_label] = float("nan")
            continue
        true_parent_idx = y_true_idx[m]  # they all share the same parent class
        pred_parent_idx = y_pred_idx[m]
        s_per_sub[sub_label] = float((pred_parent_idx == true_parent_idx).mean())

    # Reliability (spectrum-level): top-class confidence vs correctness
    proba_top = y_proba.max(axis=1)
    correct = (y_pred_idx == y_true_idx).astype(int)
    rel_spec = reliability_quantile_bins(proba_top, correct, n_bins=10)

    # File-level via soft vote
    unique_files, file_proba, file_pred_idx = file_aggregate_softvote(y_proba, file_ids)
    # Need ground-truth label per file
    file_to_class = {}
    file_to_sub = {}
    for f in unique_files:
        m = file_ids == f
        # all spectra in a file share the class (by construction)
        file_to_class[f] = y_true[m][0]
        file_to_sub[f] = sub_labels[m][0]
    file_y_true = np.array([file_to_class[f] for f in unique_files])
    file_y_true_idx = class_to_idx(file_y_true)
    file_y_pred = np.array([PRIMARY_CLASSES[i] for i in file_pred_idx])

    f_macro = f1_score(file_y_true, file_y_pred, labels=PRIMARY_CLASSES, average="macro", zero_division=0)
    f_bacc = balanced_accuracy_score(file_y_true, file_y_pred)
    f_per_class = recall_score(file_y_true, file_y_pred, labels=PRIMARY_CLASSES, average=None, zero_division=0)
    f_per_class_dict = {c: float(f_per_class[i]) for i, c in enumerate(PRIMARY_CLASSES)}
    f_brier = multiclass_brier(file_proba, file_y_true_idx, N_PRIMARY)
    f_conf_44 = confusion_matrix(file_y_true, file_y_pred, labels=PRIMARY_CLASSES).tolist()

    file_sub_labels = np.array([file_to_sub[f] for f in unique_files])
    f_conf_10 = np.zeros((len(SUBCLASS_ROWS_10x10), N_PRIMARY), dtype=int)
    for i, sub_label in enumerate(SUBCLASS_ROWS_10x10):
        m = file_sub_labels == sub_label
        if m.sum() == 0:
            continue
        for j, pri in enumerate(PRIMARY_CLASSES):
            f_conf_10[i, j] = int(((file_y_pred == pri) & m).sum())

    f_per_sub = {}
    for sub_label in SUBCLASS_ROWS_10x10:
        m = file_sub_labels == sub_label
        if m.sum() == 0:
            f_per_sub[sub_label] = float("nan")
            continue
        f_per_sub[sub_label] = float((file_pred_idx[m] == file_y_true_idx[m]).mean())

    file_proba_top = file_proba.max(axis=1)
    file_correct = (file_pred_idx == file_y_true_idx).astype(int)
    rel_file = reliability_quantile_bins(file_proba_top, file_correct, n_bins=10)

    return FoldResult(
        fold_id=fold_id,
        protocol=protocol,
        model_name=model_name,
        spectrum_macro_f1=float(s_macro),
        spectrum_balanced_acc=float(s_bacc),
        spectrum_per_class_recall=s_per_class_dict,
        spectrum_brier=float(s_brier),
        file_macro_f1=float(f_macro),
        file_balanced_acc=float(f_bacc),
        file_per_class_recall=f_per_class_dict,
        file_brier=float(f_brier),
        confusion_4x4_spectrum=s_conf_44,
        confusion_4x4_file=f_conf_44,
        confusion_10x10_spectrum=s_conf_10.tolist(),
        confusion_10x10_file=f_conf_10.tolist(),
        per_subclass_recall_spectrum=s_per_sub,
        per_subclass_recall_file=f_per_sub,
        reliability_spectrum=rel_spec,
        reliability_file=rel_file,
        n_train=n_train,
        n_test=n,
        n_train_files=n_train_files,
        n_test_files=n_test_files,
        training_time_s=float(training_time_s),
        predictions_path=predictions_path,
        best_hyperparams=best_hyperparams,
        notes=notes or [],
    )


def write_model_result(mr: ModelResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(mr), f, indent=2)


def write_predictions_parquet(
    *,
    fold_id: int | str,
    file_ids: np.ndarray,
    subclass: np.ndarray,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_path: Path,
) -> None:
    """Save predictions keyed by spectrum_id + file_id (not positional)."""
    df = pd.DataFrame({
        "spectrum_id": np.arange(len(file_ids)),
        "file_id": file_ids,
        "subclass": subclass,
        "primary_true": y_true,
        "fold_id": [str(fold_id)] * len(file_ids),
    })
    for i, c in enumerate(PRIMARY_CLASSES):
        df[f"p_{c}"] = y_proba[:, i]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
