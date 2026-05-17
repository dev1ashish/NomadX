"""Per-strain optimal-lambda selector for DANN base models.

The per-strain table in plan/07§dann-lambda-curve-completed shows the
DANN lambda has non-monotonic strain optima:
    ATCC25922 wins at lambda=0.1 (0.89)
    K-12      wins at lambda=0.3 (0.88)
    O103H2    wins at lambda=0.3 (0.89)
    O157H7    ties at lambda=0.05/0.3 (0.67)
    83972     wins at lambda=0 / 0.05 (0.88)
    ...

No single lambda is universally best. This module picks the lambda
PER LOSO TEST STRAIN, using the only signal we have that does NOT
look at the held-out strain's labels: the best inner-validation
macro-F1 each lambda achieved on the OUTER-TRAIN (which already
held out the test strain).

Three selector variants:

  hard:    For each strain X, pick the single lambda with the
           highest inner val_macro_f1 on the outer-train fold.
           Predictions = that lambda's saved predictions for X.

  soft:    For each strain X, weight the 3 lambdas by softmax of
           their inner val_macro_f1 (temperature = 1.0 by default).
           Predictions = weighted average of the 3 lambdas'
           per-spectrum probas.

  router:  For each test FILE, pick the base model whose mean
           max-proba on that file's spectra is highest. This uses
           test-set confidence (not labels), which is mild leakage
           but defensible -- analogous to "let the model abstain
           via low confidence." Documented honestly in the writeup.

All three selectors save predictions in the standard
predictions_fold_<strain>.parquet schema so downstream evaluation
code (atlas/evaluate.py) treats the selector like any other model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.evaluate import PRIMARY_CLASSES


PROBA_COLS = [f"p_{c}" for c in PRIMARY_CLASSES]


@dataclass(frozen=True)
class LambdaCandidate:
    """One base DANN run + its tag (e.g. lambda value as a string)."""
    name: str
    run_dir: Path


def read_inner_val_f1(run_dir: Path, fold_id: str) -> float:
    """Read best_val_macro_f1 from history_fold_<fold_id>.json.

    Returns NaN if missing -- caller's responsibility to handle (typically
    by treating the candidate as "no signal", giving it the worst slot).
    """
    p = run_dir / f"history_fold_{fold_id}.json"
    if not p.exists():
        return float("nan")
    with open(p) as f:
        h = json.load(f)
    return float(h.get("best_val_macro_f1", float("nan")))


def hard_select(
    candidates: list[LambdaCandidate],
    fold_id: str,
) -> tuple[LambdaCandidate, dict]:
    """Pick the single candidate with highest inner val_macro_f1 on this fold.

    Returns (chosen, info) where info has per-candidate inner_val_f1.
    """
    scores = {c.name: read_inner_val_f1(c.run_dir, fold_id) for c in candidates}
    # Argmax with NaN-safety: NaN is treated as -inf for argmax purposes.
    best_name = max(scores.keys(),
                    key=lambda n: scores[n] if not np.isnan(scores[n]) else -np.inf)
    chosen = next(c for c in candidates if c.name == best_name)
    return chosen, {"inner_val_f1": scores, "selected": best_name}


def hard_predictions_for_fold(
    candidates: list[LambdaCandidate],
    fold_id: str,
) -> tuple[pd.DataFrame, dict]:
    """Load the chosen candidate's saved per-spectrum predictions for this fold.

    Returns (predictions_df, selection_info).
    """
    chosen, info = hard_select(candidates, fold_id)
    p = chosen.run_dir / f"predictions_fold_{fold_id}.parquet"
    df = pd.read_parquet(p)
    return df, info


def soft_predictions_for_fold(
    candidates: list[LambdaCandidate],
    fold_id: str,
    temperature: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """Softmax-weighted average of per-spectrum probas across candidates.

    Weights = softmax(inner_val_f1 / temperature). Lower temperature ->
    more concentrated on the best candidate (approaches `hard` as T -> 0).
    """
    scores = np.array([read_inner_val_f1(c.run_dir, fold_id) for c in candidates])
    # Replace NaN with the minimum non-NaN -> effectively weight ~0.
    if np.any(np.isnan(scores)):
        if np.all(np.isnan(scores)):
            scores = np.zeros_like(scores)  # uniform fallback
        else:
            scores = np.where(np.isnan(scores), np.nanmin(scores), scores)
    scaled = scores / max(temperature, 1e-8)
    weights = np.exp(scaled - scaled.max())
    weights = weights / weights.sum()

    # Load each candidate's predictions for this fold, merge on (spectrum_id, file_id),
    # then weighted-average the proba columns.
    frames = []
    for c in candidates:
        p = c.run_dir / f"predictions_fold_{fold_id}.parquet"
        frames.append(pd.read_parquet(p))

    base = frames[0][["spectrum_id", "file_id", "subclass", "primary_true", "fold_id"]].copy()
    accum = np.zeros((len(base), len(PROBA_COLS)), dtype=np.float64)
    for i, (w, f) in enumerate(zip(weights, frames)):
        # Align by (spectrum_id, file_id) merge
        merged = base.merge(f[["spectrum_id", "file_id"] + PROBA_COLS],
                            on=["spectrum_id", "file_id"], how="inner")
        if len(merged) != len(base):
            raise ValueError(
                f"soft-select merge dropped rows on fold {fold_id} "
                f"adding candidate {c.name}: {len(base)} -> {len(merged)}"
            )
        accum += w * merged[PROBA_COLS].to_numpy()

    # Renormalize defensively
    row_sum = accum.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    accum = accum / row_sum

    out = base.copy()
    for k, c in enumerate(PROBA_COLS):
        out[c] = accum[:, k]
    info = {
        "inner_val_f1": {c.name: float(s) for c, s in zip(candidates, scores)},
        "weights": {c.name: float(w) for c, w in zip(candidates, weights)},
        "temperature": temperature,
    }
    return out, info


def router_predictions_for_fold(
    candidates: list[LambdaCandidate],
    fold_id: str,
    signal: str = "max_proba",
) -> tuple[pd.DataFrame, dict]:
    """For each test FILE, pick the candidate whose file-level confidence is highest.

    Confidence signal options:
      "max_proba"   — mean of max(p) across the file's spectra (original).
      "margin"      — mean of (max(p) − second_max(p)) across spectra.
                      Calibration-INVARIANT for well-trained models because
                      shifting all logits by a constant cancels out.

    Aggregating to file level before routing means a single noisy spectrum
    doesn't flip the choice.
    """
    if signal not in ("max_proba", "margin"):
        raise ValueError(f"unknown signal: {signal}")
    frames = {c.name: pd.read_parquet(c.run_dir / f"predictions_fold_{fold_id}.parquet")
              for c in candidates}
    # Compute file-level mean confidence per candidate
    file_max_proba: dict[str, pd.DataFrame] = {}
    for name, df in frames.items():
        proba = df[PROBA_COLS].to_numpy()
        if signal == "max_proba":
            score_per_spectrum = proba.max(axis=1)
        else:  # margin
            sorted_p = np.sort(proba, axis=1)
            score_per_spectrum = sorted_p[:, -1] - sorted_p[:, -2]
        df = df.assign(_score=score_per_spectrum)
        agg = df.groupby("file_id", as_index=False)["_score"].mean()
        agg = agg.rename(columns={"_score": f"max_proba_{name}"})
        file_max_proba[name] = agg

    # Combine into a single DataFrame keyed on file_id with one column per candidate.
    combined = file_max_proba[candidates[0].name]
    for c in candidates[1:]:
        combined = combined.merge(file_max_proba[c.name], on="file_id", how="inner")

    # Pick best candidate per file
    score_cols = [f"max_proba_{c.name}" for c in candidates]
    combined["_best_candidate"] = combined[score_cols].idxmax(axis=1)
    combined["_best_candidate"] = combined["_best_candidate"].str.replace("max_proba_", "")

    # Per file, take that candidate's predictions for the file's spectra.
    # Reassemble the per-spectrum predictions parquet by routing per file.
    out_chunks = []
    for _, row in combined.iterrows():
        fid = row["file_id"]
        chosen_name = row["_best_candidate"]
        chosen_df = frames[chosen_name]
        chunk = chosen_df[chosen_df["file_id"] == fid].copy()
        chunk["_routed_via"] = chosen_name
        out_chunks.append(chunk)
    out = pd.concat(out_chunks, axis=0, ignore_index=True).drop(columns=["_routed_via"])

    info = {
        "routing": dict(zip(combined["file_id"].tolist(),
                            combined["_best_candidate"].tolist())),
    }
    return out, info


def per_strain_parent_recall_from_predictions(pred_df: pd.DataFrame) -> float:
    """File-level soft-vote -> parent class -> recall for the fold (single strain)."""
    fids = pred_df["file_id"].unique()
    correct = 0
    for fid in fids:
        sub = pred_df[pred_df["file_id"] == fid]
        proba = sub[PROBA_COLS].mean(axis=0).to_numpy()
        pred = PRIMARY_CLASSES[int(np.argmax(proba))]
        true = sub["primary_true"].iloc[0]
        if pred == true:
            correct += 1
    return float(correct / len(fids))
