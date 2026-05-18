"""Fit the project-headline PLS-DA-on-raw-spectrum classifier on the full
QC-passing training set and save it as a deployable joblib.

The LOSO record is **0.603 file-weighted balanced accuracy** (project headline,
FINAL/PAPER.md §3.1). That measurement was per-fold inner-CV (HPO grid
[5,8,10,12,15,20,25,30] components, modal winner = 30). For deployment we
refit once on all 7,122 QC-passing spectra with n_components = 30, giving a
single shippable model.

Inputs (in repo `data_cache/`):
    spectra_array_preprocessed.npy   (7999, 987)  float32   arPLS+SG+crop+SNV
    spectra.parquet                  (7999,)               with primary_class
    qc_mask.npy                      (7999,) bool   True = QC pass (sum=7122)

Outputs (in repo `artifacts/`):
    plsda_raw_classifier.joblib   Pipeline(StandardScaler, PLSDA(n_comp=30))
    plsda_raw_metadata.json       training stats + classes + version pins
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import sklearn  # noqa: E402  (need version after sys.path fix)
from atlas.models_classical import make_plsda  # noqa: E402

CACHE_DIR = REPO_ROOT / "data_cache"
ARTIFACT_DIR = REPO_ROOT / "artifacts"

N_COMPONENTS = 30  # mode of per-fold LOSO winners (see PAPER.md §3.1)
SEED = 42


def main() -> int:
    t0 = time.perf_counter()
    print(f"sklearn version: {sklearn.__version__}")

    X_full = np.load(CACHE_DIR / "spectra_array_preprocessed.npy")
    mask = np.load(CACHE_DIR / "qc_mask.npy")
    spec_df = pd.read_parquet(CACHE_DIR / "spectra.parquet")
    if X_full.shape[0] != mask.shape[0] or X_full.shape[0] != len(spec_df):
        raise RuntimeError("cache shape mismatch — re-run preprocessing")

    X = X_full[mask].astype(np.float32, copy=False)
    y = spec_df.loc[mask, "primary_class"].to_numpy()
    print(f"Training set: X={X.shape}  y={y.shape}  classes={sorted(set(y))}")

    pipe = make_plsda({"n_components": N_COMPONENTS}, seed=SEED)
    pipe.fit(X, y)
    print(f"Fit done in {time.perf_counter() - t0:.1f}s")

    ARTIFACT_DIR.mkdir(exist_ok=True)
    out_joblib = ARTIFACT_DIR / "plsda_raw_classifier.joblib"
    joblib.dump(pipe, out_joblib, compress=3)
    print(f"Saved {out_joblib}  ({out_joblib.stat().st_size / 1024:.1f} KiB)")

    # Quick in-sample sanity check (NOT a generalization metric).
    pred = pipe.predict(X)
    in_sample_acc = float((pred == y).mean())
    classes_in_pipe = list(pipe.named_steps["clf"].classes_)
    proba_sample = pipe.predict_proba(X[:1])[0]
    print(f"In-sample accuracy: {in_sample_acc:.4f}")
    print(f"Classes (clf.classes_): {classes_in_pipe}")
    print(f"Sample predict_proba row: {proba_sample.tolist()}")

    metadata = {
        "name": "plsda_raw",
        "description": "PLS-DA on preprocessed raw spectrum (987 bins). "
        "Project-headline baseline (LOSO file-weighted balanced acc = 0.603).",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "seed": SEED,
        "n_components": N_COMPONENTS,
        "classes": classes_in_pipe,
        "n_train_spectra": int(X.shape[0]),
        "n_bins": int(X.shape[1]),
        "qc_mask_sum": int(mask.sum()),
        "in_sample_accuracy": in_sample_acc,
        "loso_headline": {
            "metric": "file_weighted_balanced_accuracy",
            "value": 0.6034,
            "source": "outputs/2026-05-14_plsda_loso_9b4a9cb3 (mean over 9 LOSO folds)",
            "note": "This is the *generalization* score under LOSO CV. The deployed "
            "joblib was refit on all 7,122 spectra, so it does NOT have a single "
            "LOSO number — use the cross-validated value as the operational expectation.",
        },
    }
    with open(ARTIFACT_DIR / "plsda_raw_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {ARTIFACT_DIR / 'plsda_raw_metadata.json'}")

    print(f"\nDONE in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
