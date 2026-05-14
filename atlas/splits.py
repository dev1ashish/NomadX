"""File-level cross-validation splits for Raman classification.

Two protocols, both file-level (no pixel ever crosses train/test):

Protocol A — StratifiedGroupKFold(5):
    groups=file_id, stratify=primary_class. With only 8 H2O files the auto
    stratifier produces empty H2O test folds (sklearn issue #33085), so we
    pre-balance H2O to 1 or 2 files per test fold round-robin, then run
    StratifiedGroupKFold on the remaining 79 bacterial files and merge.

Protocol B — Leave-One-Strain-Out:
    9 folds, one per bacterial subclass. Hold out ALL files of that
    subclass; train on the other 8 subclasses + H2O. H2O is permanently
    in training (8 files is too few for class-level holdout).

Output: two master JSON files at data_cache/splits/. Each fold stores
file_ids and the row indices into the 7,999-row preprocessed array (QC
mask is already factored in -- only QC-passing row indices appear in
any fold). See plan/03_architecture.md sec C for the schema.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold


MASTER_SEED = 42


@dataclass(frozen=True)
class SplitConfig:
    cache_dir: Path
    out_dir: Path
    n_outer_folds: int = 5
    n_inner_folds: int = 4
    seed: int = MASTER_SEED


def fold_seed(master_seed: int, fold_idx: int) -> int:
    """Derive a deterministic per-fold seed."""
    return (master_seed * 31337 + fold_idx) % (2**31)


def _cache_hash(qc_mask: np.ndarray, primary_class: np.ndarray, subclass: np.ndarray, seed: int) -> str:
    """Hash of inputs that determine fold assignment.

    If any of these change, splits become invalid and must be rebuilt.
    """
    h = hashlib.sha256()
    h.update(qc_mask.tobytes())
    h.update(primary_class.astype(str).tobytes())
    # Replace None with empty string for stable hashing
    sub = pd.Series(subclass).fillna("").astype(str).to_numpy()
    h.update(sub.tobytes())
    h.update(str(seed).encode("utf-8"))
    return h.hexdigest()


def _file_level_frame(spec_df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """One row per file, sorted by file_id for stable splitter input order."""
    file_rows = (
        spec_df.groupby("file_id", as_index=False)
        .agg(primary_class=("primary_class", "first"), subclass=("subclass", "first"))
    )
    # Pull ac_calibration_date for the calibration-date sanity check.
    cal = metadata[["file_id", "ac_calibration_date"]].drop_duplicates("file_id")
    file_rows = file_rows.merge(cal, on="file_id", how="left")
    file_rows = file_rows.sort_values("file_id", kind="mergesort").reset_index(drop=True)
    return file_rows


def _balanced_h2o_assignment(h2o_files: list[str], n_folds: int, rng: np.random.Generator) -> dict[str, int]:
    """Round-robin H2O files into test folds.

    With 8 H2O files and 5 folds => 3 folds get 2 H2O, 2 folds get 1 H2O.
    Shuffle order so the assignment is seed-dependent but deterministic.
    """
    shuffled = list(h2o_files)
    rng.shuffle(shuffled)
    return {f: i % n_folds for i, f in enumerate(shuffled)}


def _resolve_row_indices(
    test_file_ids: list[str],
    train_file_ids: list[str],
    spec_df: pd.DataFrame,
    qc_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve QC-passing row indices into the 7,999-row preprocessed array.

    Returns (train_row_indices, test_row_indices) as int64 numpy arrays sorted ascending.
    """
    file_id_arr = spec_df["file_id"].to_numpy()
    test_set = set(test_file_ids)
    train_set = set(train_file_ids)
    test_rows = np.where(np.isin(file_id_arr, list(test_set)) & qc_mask)[0]
    train_rows = np.where(np.isin(file_id_arr, list(train_set)) & qc_mask)[0]
    return np.sort(train_rows), np.sort(test_rows)


def _class_dist(test_file_ids: list[str], file_frame: pd.DataFrame) -> dict[str, int]:
    sub = file_frame[file_frame["file_id"].isin(test_file_ids)]
    return sub["primary_class"].value_counts().to_dict()


def _calibration_overlap(
    train_file_ids: list[str], test_file_ids: list[str], file_frame: pd.DataFrame
) -> list[str]:
    """Return calibration dates appearing in BOTH train and test (warning only)."""
    train_dates = set(
        file_frame.loc[file_frame["file_id"].isin(train_file_ids), "ac_calibration_date"].dropna()
    )
    test_dates = set(
        file_frame.loc[file_frame["file_id"].isin(test_file_ids), "ac_calibration_date"].dropna()
    )
    return sorted(train_dates & test_dates)


def build_protocol_a(
    spec_df: pd.DataFrame,
    metadata: pd.DataFrame,
    qc_mask: np.ndarray,
    cfg: SplitConfig,
) -> dict:
    """StratifiedGroupKFold(5) with H2O pre-balancing.

    Strategy:
        1. Round-robin H2O files to fold indices.
        2. Run StratifiedGroupKFold on the 79 bacterial files (no H2O).
        3. For each fold, merge bacterial test files with the H2O files
           pre-assigned to that fold index.
    """
    file_frame = _file_level_frame(spec_df, metadata)
    rng = np.random.default_rng(cfg.seed)

    h2o_files = file_frame.loc[file_frame["primary_class"] == "H2O", "file_id"].tolist()
    bact_frame = file_frame[file_frame["primary_class"] != "H2O"].reset_index(drop=True)

    h2o_assign = _balanced_h2o_assignment(h2o_files, cfg.n_outer_folds, rng)

    sgkf = StratifiedGroupKFold(
        n_splits=cfg.n_outer_folds, shuffle=True, random_state=cfg.seed
    )

    folds: list[dict] = []
    warnings: list[str] = []
    for fold_idx, (_, test_idx) in enumerate(
        sgkf.split(
            X=np.zeros(len(bact_frame)),
            y=bact_frame["primary_class"].to_numpy(),
            groups=bact_frame["file_id"].to_numpy(),
        )
    ):
        bact_test_files = bact_frame.iloc[test_idx]["file_id"].tolist()
        h2o_test_files = [f for f, a in h2o_assign.items() if a == fold_idx]
        test_files = sorted(bact_test_files + h2o_test_files)
        train_files = sorted(set(file_frame["file_id"]) - set(test_files))

        train_rows, test_rows = _resolve_row_indices(test_files, train_files, spec_df, qc_mask)
        cal_overlap = _calibration_overlap(train_files, test_files, file_frame)
        if cal_overlap:
            warnings.append(
                f"fold {fold_idx}: calibration dates in BOTH train and test: {cal_overlap}"
            )

        folds.append(
            {
                "fold": fold_idx,
                "test_file_ids": test_files,
                "train_file_ids": train_files,
                "test_row_indices": train_rows_to_list(test_rows),
                "train_row_indices": train_rows_to_list(train_rows),
                "n_test_spectra": int(test_rows.size),
                "n_train_spectra": int(train_rows.size),
                "test_class_dist": _class_dist(test_files, file_frame),
                "n_test_h2o_files": len(h2o_test_files),
                "calibration_date_overlap": cal_overlap,
                "fold_seed": fold_seed(cfg.seed, fold_idx),
            }
        )

    return {
        "meta": _build_meta(spec_df, qc_mask, cfg, protocol="group_kfold"),
        "protocol": "group_kfold",
        "n_folds": cfg.n_outer_folds,
        "warnings": warnings,
        "folds": folds,
    }


def build_protocol_b(
    spec_df: pd.DataFrame,
    metadata: pd.DataFrame,
    qc_mask: np.ndarray,
    cfg: SplitConfig,
) -> dict:
    """LOSO: one fold per bacterial subclass. H2O always in training."""
    file_frame = _file_level_frame(spec_df, metadata)
    bact_subclasses = sorted(
        file_frame.loc[
            file_frame["primary_class"] != "H2O", "subclass"
        ].dropna().unique().tolist()
    )

    folds: list[dict] = []
    warnings: list[str] = []
    for fold_idx, sub in enumerate(bact_subclasses):
        test_files = sorted(file_frame.loc[file_frame["subclass"] == sub, "file_id"].tolist())
        train_files = sorted(set(file_frame["file_id"]) - set(test_files))

        train_rows, test_rows = _resolve_row_indices(test_files, train_files, spec_df, qc_mask)
        held_out_parent = file_frame.loc[file_frame["subclass"] == sub, "primary_class"].iloc[0]
        cal_overlap = _calibration_overlap(train_files, test_files, file_frame)
        if cal_overlap:
            warnings.append(
                f"fold {sub}: calibration dates in BOTH train and test: {cal_overlap}"
            )

        folds.append(
            {
                "fold": sub,
                "held_out_subclass": sub,
                "held_out_parent_class": held_out_parent,
                "test_file_ids": test_files,
                "train_file_ids": train_files,
                "test_row_indices": train_rows_to_list(test_rows),
                "train_row_indices": train_rows_to_list(train_rows),
                "n_test_spectra": int(test_rows.size),
                "n_train_spectra": int(train_rows.size),
                "n_test_files": len(test_files),
                "calibration_date_overlap": cal_overlap,
                "fold_seed": fold_seed(cfg.seed, fold_idx),
            }
        )

    return {
        "meta": _build_meta(spec_df, qc_mask, cfg, protocol="loso"),
        "protocol": "loso",
        "n_folds": len(bact_subclasses),
        "warnings": warnings,
        "folds": folds,
    }


def train_rows_to_list(arr: np.ndarray) -> list[int]:
    """Convert int array to python list for JSON serialization."""
    return arr.astype(np.int64).tolist()


def _build_meta(
    spec_df: pd.DataFrame, qc_mask: np.ndarray, cfg: SplitConfig, *, protocol: str
) -> dict:
    return {
        "protocol": protocol,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": cfg.seed,
        "sklearn_version": sklearn.__version__,
        "n_spectra_total": int(qc_mask.size),
        "n_spectra_qc": int(qc_mask.sum()),
        "n_files": int(spec_df["file_id"].nunique()),
        "cache_hash": _cache_hash(
            qc_mask,
            spec_df["primary_class"].to_numpy(),
            spec_df["subclass"].to_numpy(),
            cfg.seed,
        ),
    }


def verify_no_leakage(split_artifact: dict, n_total: int, qc_mask: np.ndarray) -> dict:
    """Smoke check: assert zero pixel/file leakage and qc-mask compliance.

    Returns a report dict; raises AssertionError on any violation.
    """
    report = {
        "protocol": split_artifact["protocol"],
        "n_folds": len(split_artifact["folds"]),
        "fold_checks": [],
    }
    qc_indices = set(np.where(qc_mask)[0].tolist())

    for fold in split_artifact["folds"]:
        train_idx = set(fold["train_row_indices"])
        test_idx = set(fold["test_row_indices"])
        train_files = set(fold["train_file_ids"])
        test_files = set(fold["test_file_ids"])

        # 1. No row index in both train and test.
        row_overlap = train_idx & test_idx
        assert not row_overlap, f"fold {fold['fold']}: {len(row_overlap)} rows in both train and test"

        # 2. No file_id in both train and test.
        file_overlap = train_files & test_files
        assert not file_overlap, f"fold {fold['fold']}: file overlap {file_overlap}"

        # 3. All row indices are within bounds.
        all_idx = train_idx | test_idx
        assert max(all_idx) < n_total, f"fold {fold['fold']}: row index >= {n_total}"
        assert min(all_idx) >= 0, f"fold {fold['fold']}: negative row index"

        # 4. All row indices are QC-passing.
        non_qc = all_idx - qc_indices
        assert not non_qc, f"fold {fold['fold']}: {len(non_qc)} non-QC rows in split"

        report["fold_checks"].append(
            {
                "fold": fold["fold"],
                "n_train_rows": len(train_idx),
                "n_test_rows": len(test_idx),
                "n_train_files": len(train_files),
                "n_test_files": len(test_files),
                "test_class_dist": fold.get("test_class_dist", {}),
                "calibration_date_overlap": fold.get("calibration_date_overlap", []),
            }
        )

    return report


def write_splits(artifact: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)


def load_splits(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_all_splits(cfg: SplitConfig) -> tuple[dict, dict]:
    """Top-level entry: build both protocols, verify, write to JSON."""
    spec_df = pd.read_parquet(cfg.cache_dir / "spectra.parquet")
    metadata = pd.read_parquet(cfg.cache_dir / "metadata.parquet")
    qc_mask = np.load(cfg.cache_dir / "qc_mask.npy")

    a = build_protocol_a(spec_df, metadata, qc_mask, cfg)
    b = build_protocol_b(spec_df, metadata, qc_mask, cfg)

    n_total = int(qc_mask.size)
    verify_no_leakage(a, n_total, qc_mask)
    verify_no_leakage(b, n_total, qc_mask)

    write_splits(a, cfg.out_dir / "protocol_a.json")
    write_splits(b, cfg.out_dir / "protocol_b.json")
    return a, b
