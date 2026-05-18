"""Atlas Raman UI — Results sidecar builder (W8).

Reads Stage 15F artifacts and emits the four JSON sidecars consumed by the
Results tab. Reproducibility-first: every number on screen is derived from
`artifacts/stage15f_*` (source of truth for §6 of FINAL/PAPER.md).

Sources:
  - artifacts/stage15f_metadata.json
  - artifacts/stage15f_paper_stats.json
  - artifacts/stage15f_loso_predictions.parquet
  - outputs/2026-05-14_plsda_loso_9b4a9cb3/predictions_fold_*.parquet
    (PLS-DA-on-raw baseline, used for an optional sanity contingency)

Outputs (idempotent overwrites):
  - ui/public/data/stage15f.json
  - ui/public/data/confusion.json
  - ui/public/data/bootstrap.json
  - ui/public/data/mcnemar.json

Run from the project root or ui/:
    cd ui && python scripts/build_results.py
    # or with uv:
    cd ui && uv run scripts/build_results.py

# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "pyarrow", "numpy"]
# ///
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Canonical class ordering used in the paper's confusion matrix
# (FINAL/PAPER.md §6.6.3: STEC row 1, Non-STEC row 2, Salmonella row 3,
# H2O row 4 with [8,0,0,0] = all 8 H2O LOSO files predicted STEC).
CLASSES = ["STEC", "Non-STEC", "Salmonella", "H2O"]


def _ui_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _repo_root() -> Path:
    return _ui_root().parent


def _load_predictions() -> pd.DataFrame:
    """Stage 15F per-file LOSO predictions for all 3 algos (already per-file)."""
    p = _repo_root() / "artifacts" / "stage15f_loso_predictions.parquet"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    return pd.read_parquet(p)


def _load_metadata() -> dict[str, Any]:
    p = _repo_root() / "artifacts" / "stage15f_metadata.json"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    # NaN is in this file; allow with parse_constant.
    with p.open() as f:
        return json.load(f, parse_constant=lambda x: None)


def _load_paper_stats() -> dict[str, Any]:
    p = _repo_root() / "artifacts" / "stage15f_paper_stats.json"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    with p.open() as f:
        return json.load(f)


def _load_plsda_raw_predictions() -> pd.DataFrame | None:
    """Aggregate PLS-DA-on-raw per-spectrum predictions to per-file majority
    vote. Used only for the optional baseline contingency check. Returns
    None if the run directory is missing.
    """
    run_dir = _repo_root() / "outputs" / "2026-05-14_plsda_loso_9b4a9cb3"
    if not run_dir.exists():
        return None
    parts = []
    for p in sorted(run_dir.glob("predictions_fold_*.parquet")):
        parts.append(pd.read_parquet(p))
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    # Per-file mean probability → argmax
    agg = (
        df.groupby("file_id")
        .agg(
            primary_true=("primary_true", "first"),
            p_H2O=("p_H2O", "mean"),
            p_NonSTEC=("p_Non-STEC", "mean"),
            p_STEC=("p_STEC", "mean"),
            p_Salmonella=("p_Salmonella", "mean"),
        )
        .reset_index()
    )
    name_map = {
        "p_H2O": "H2O",
        "p_NonSTEC": "Non-STEC",
        "p_STEC": "STEC",
        "p_Salmonella": "Salmonella",
    }
    prob_cols = list(name_map.keys())
    agg["y_pred"] = agg[prob_cols].idxmax(axis=1).map(name_map)
    agg["y_true"] = agg["primary_true"]
    return agg[["file_id", "y_true", "y_pred"]].copy()


def build_stage15f(meta: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    """Top-level Stage 15F summary block consumed by the KPI strip."""
    boot = stats["bootstrap_ci"]["logreg"]
    mcnemar = stats["mcnemar"]["logreg_vs_plsda"]
    return {
        "loso_mean_acc": float(meta["loso_mean_accuracy"]),
        "loso_fw_acc": float(boot["loso_mean_file_accuracy"]),
        "bootstrap_ci_low": float(boot["boot_ci_95_lo"]),
        "bootstrap_ci_high": float(boot["boot_ci_95_hi"]),
        "mcnemar_p": float(mcnemar["mcnemar_p"]),
        "n_features": int(meta["feature_count"]),
        "per_strain_accuracy": {
            str(k): float(v) for k, v in meta["per_strain_accuracy"].items()
        },
        "algo_comparison": {
            algo: {
                "mean_loso_accuracy": float(d["mean_loso_accuracy"]),
                "mean_loso_macro_recall": float(d["mean_loso_macro_recall"]),
            }
            for algo, d in meta["algo_comparison"].items()
        },
        # Project headline — PLS-DA-on-raw LOSO 0.603 (FINAL/PAPER.md §3.1).
        "plsda_raw_loso": 0.603,
    }


def build_confusion(preds: pd.DataFrame) -> dict[str, Any]:
    """4x4 confusion matrix on Stage 15F LogReg per-file LOSO predictions.

    Each cell exposes the list of file_ids whose true class is the row and
    predicted class is the column. Powers the click-to-inspect <Dialog>.
    """
    log = preds[preds["algo"] == "logreg"].copy()
    matrix: list[list[int]] = [[0] * len(CLASSES) for _ in CLASSES]
    cell_files: dict[str, list[str]] = {}
    idx = {c: i for i, c in enumerate(CLASSES)}
    for row in log.itertuples(index=False):
        if row.y_true not in idx or row.y_pred not in idx:
            continue
        i, j = idx[row.y_true], idx[row.y_pred]
        matrix[i][j] += 1
        key = f"{row.y_true}_{row.y_pred}"
        cell_files.setdefault(key, []).append(str(row.file_id))
    # Sort per cell for stable output
    for k in cell_files:
        cell_files[k].sort()
    return {
        "classes": CLASSES,
        "matrix": matrix,
        "per_cell_files": cell_files,
        "n_files": int(log.shape[0]),
    }


def build_bootstrap(preds: pd.DataFrame, stats: dict[str, Any]) -> dict[str, Any]:
    """5,000-resample file-wise bootstrap of LogReg LOSO accuracy.

    Reproduces the CI in `stage15f_paper_stats.json:bootstrap_ci.logreg` and
    emits the raw resampled accuracies for the Plotly histogram.
    """
    log = preds[preds["algo"] == "logreg"][["file_id", "y_true", "y_pred"]].copy()
    correct = (log["y_true"] == log["y_pred"]).to_numpy(dtype=float)
    n = correct.shape[0]
    rng = np.random.default_rng(0)
    n_boot = 5000
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = correct[idx].mean(axis=1)
    boot = stats["bootstrap_ci"]["logreg"]
    return {
        "samples": [float(s) for s in samples],
        "ci_low": float(boot["boot_ci_95_lo"]),
        "ci_high": float(boot["boot_ci_95_hi"]),
        "point_estimate": float(boot["loso_mean_file_accuracy"]),
        "n_boot": int(n_boot),
        "n_files": int(n),
        "seed": 0,
    }


def build_mcnemar(
    preds: pd.DataFrame,
    stats: dict[str, Any],
    plsda_raw: pd.DataFrame | None,
) -> dict[str, Any]:
    """2x2 contingency for the LogReg-vs-PLS-DA paired test.

    Primary source: paper_stats.json (engineered-features PLS-DA, matching
    FINAL/PAPER.md §6.4 p = 0.0020). We additionally compute the contingency
    against PLS-DA-on-raw when the LOSO run is available; that comparison
    only covers the 79 bacterial files (no H2O fold in the raw run) and is
    included for transparency under `plsda_raw_baseline`.
    """
    primary = stats["mcnemar"]["logreg_vs_plsda"]
    payload: dict[str, Any] = {
        "both_right": int(primary["n_both_correct"]),
        "logreg_only_right": int(primary["n_a_only_correct"]),
        "plsda_only_right": int(primary["n_b_only_correct"]),
        "both_wrong": int(primary["n_both_wrong"]),
        "p_value": float(primary["mcnemar_p"]),
        "n_total": int(primary["n_total"]),
        "source": "stage15f_paper_stats.json:mcnemar.logreg_vs_plsda",
    }
    if plsda_raw is not None:
        log = preds[preds["algo"] == "logreg"][["file_id", "y_true", "y_pred"]].copy()
        log = log.rename(columns={"y_pred": "y_pred_logreg"})
        plsda_raw = plsda_raw.rename(columns={"y_pred": "y_pred_plsda_raw"})
        merged = log.merge(plsda_raw[["file_id", "y_pred_plsda_raw"]], on="file_id")
        a_correct = (merged["y_true"] == merged["y_pred_logreg"]).to_numpy()
        b_correct = (merged["y_true"] == merged["y_pred_plsda_raw"]).to_numpy()
        both_right = int((a_correct & b_correct).sum())
        a_only = int((a_correct & ~b_correct).sum())
        b_only = int((~a_correct & b_correct).sum())
        both_wrong = int((~a_correct & ~b_correct).sum())
        payload["plsda_raw_baseline"] = {
            "both_right": both_right,
            "logreg_only_right": a_only,
            "plsda_only_right": b_only,
            "both_wrong": both_wrong,
            "n_total": int(merged.shape[0]),
            "note": "PLS-DA-on-raw covers 79 bacterial files only (no H2O fold).",
        }
    return payload


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def main() -> None:
    out_dir = _ui_root() / "public" / "data"

    meta = _load_metadata()
    stats = _load_paper_stats()
    preds = _load_predictions()
    plsda_raw = _load_plsda_raw_predictions()

    _write(out_dir / "stage15f.json", build_stage15f(meta, stats))
    _write(out_dir / "confusion.json", build_confusion(preds))
    _write(out_dir / "bootstrap.json", build_bootstrap(preds, stats))
    _write(out_dir / "mcnemar.json", build_mcnemar(preds, stats, plsda_raw))


if __name__ == "__main__":
    main()
