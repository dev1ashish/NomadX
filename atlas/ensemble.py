"""Soft-vote ensemble across pre-computed per-fold predictions.

Inputs are run dirs produced by scripts/run_classical.py and scripts/run_cnn.py.
Each run dir contains `predictions_fold_*.parquet` keyed on (spectrum_id,
file_id) with columns p_H2O, p_Non-STEC, p_STEC, p_Salmonella in canonical
PRIMARY_CLASSES order. No re-training is done here; we merge per-spectrum
probabilities on (spectrum_id, file_id) and average across input runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from atlas.evaluate import PRIMARY_CLASSES


PROBA_COLS = [f"p_{c}" for c in PRIMARY_CLASSES]
MERGE_KEYS = ["spectrum_id", "file_id"]
META_COLS = ["subclass", "primary_true", "fold_id"]


def _fold_id_from_filename(name: str) -> str:
    # `predictions_fold_0.parquet` -> "0"; `predictions_fold_K-12.parquet` -> "K-12"
    stem = Path(name).stem
    assert stem.startswith("predictions_fold_"), stem
    return stem[len("predictions_fold_"):]


def load_predictions(run_dir: Path | str) -> dict[str, pd.DataFrame]:
    """Read predictions_fold_*.parquet from a run dir.

    Returns dict keyed by fold_id (str). Each DataFrame is exactly the parquet
    on disk — schema described in atlas.evaluate.write_predictions_parquet.
    """
    run_dir = Path(run_dir)
    out: dict[str, pd.DataFrame] = {}
    for f in sorted(run_dir.glob("predictions_fold_*.parquet")):
        fold_id = _fold_id_from_filename(f.name)
        out[fold_id] = pd.read_parquet(f)
    if not out:
        raise FileNotFoundError(f"No predictions_fold_*.parquet under {run_dir}")
    return out


def soft_vote_ensemble(
    run_dirs: list[Path | str],
    fold_id: str,
    weights: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average per-spectrum proba across runs for a single fold.

    Merges on (spectrum_id, file_id) — NOT positional. Asserts same row count
    across all input runs for the fold (i.e. same test set).

    Returns
    -------
    spectrum_ids : (N,) int
    file_ids     : (N,) object
    subclass     : (N,) object
    y_true       : (N,) object  (primary class strings)
    y_proba      : (N, K) float  (averaged, in PRIMARY_CLASSES order)
    """
    if len(run_dirs) < 2:
        raise ValueError("Ensemble requires at least 2 run dirs")

    if weights is None:
        weights = [1.0 / len(run_dirs)] * len(run_dirs)
    else:
        if len(weights) != len(run_dirs):
            raise ValueError("weights length must match run_dirs")
        s = sum(weights)
        weights = [w / s for w in weights]

    frames = []
    for d in run_dirs:
        d = Path(d)
        p = d / f"predictions_fold_{fold_id}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"missing fold parquet: {p}")
        frames.append(pd.read_parquet(p))

    # Sanity: all same length
    n0 = len(frames[0])
    for d, f in zip(run_dirs, frames):
        if len(f) != n0:
            raise ValueError(
                f"row count mismatch on fold {fold_id}: "
                f"{run_dirs[0]} has {n0} rows, {d} has {len(f)}"
            )

    # Align by (spectrum_id, file_id) via successive inner merges
    base = frames[0][MERGE_KEYS + META_COLS + PROBA_COLS].copy()
    base = base.rename(columns={c: f"{c}_r0" for c in PROBA_COLS})

    accum = np.zeros((n0, len(PROBA_COLS)), dtype=np.float64)
    accum += weights[0] * base[[f"{c}_r0" for c in PROBA_COLS]].to_numpy()

    for i, f in enumerate(frames[1:], start=1):
        renamed = f[MERGE_KEYS + PROBA_COLS].rename(
            columns={c: f"{c}_r{i}" for c in PROBA_COLS}
        )
        before = len(base)
        base = base.merge(renamed, on=MERGE_KEYS, how="inner")
        if len(base) != before:
            raise ValueError(
                f"merge dropped rows on fold {fold_id} when adding run {run_dirs[i]}: "
                f"{before} -> {len(base)}"
            )
        accum = accum * 0  # rebuild from scratch to keep order matching `base`
        for j in range(i + 1):
            cols = [f"{c}_r{j}" for c in PROBA_COLS]
            accum = accum + weights[j] * base[cols].to_numpy()

    # Renormalize defensively (weights already sum to 1; tiny float drift)
    row_sum = accum.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    accum = accum / row_sum

    spectrum_ids = base["spectrum_id"].to_numpy()
    file_ids = base["file_id"].to_numpy()
    subclass = base["subclass"].to_numpy()
    y_true = base["primary_true"].to_numpy()
    return spectrum_ids, file_ids, subclass, y_true, accum.astype(np.float64)


def discover_fold_ids(run_dirs: list[Path | str]) -> list[str]:
    """Return fold_ids present in ALL run dirs. Errors if the intersection is empty."""
    sets = []
    for d in run_dirs:
        sets.append(set(load_predictions(d).keys()))
    common = sorted(set.intersection(*sets)) if sets else []
    if not common:
        raise ValueError(f"No common fold_ids across {run_dirs}")
    # Stable ordering: numeric folds in numeric order, then alphabetical
    def keyfn(s: str):
        try:
            return (0, int(s), "")
        except ValueError:
            return (1, 0, s)
    return sorted(common, key=keyfn)
