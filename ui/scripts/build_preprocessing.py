"""Atlas Raman UI — Preprocessing pipeline sidecar builder (W12).

Picks ONE exemplar file per primary class (STEC / Non-STEC / Salmonella / H2O),
mean-pools its QC-passed pixel spectra, then walks every atomic step of the
preprocessing pipeline from `atlas/preprocess.py`, capturing the intermediate
result after each step. The arPLS-fitted baseline is also captured separately
so the UI can render "what arPLS removed".

Pipeline order (per `atlas/preprocess.py` and `FINAL/PAPER.md` §2.3):
    0. raw 2048-bin           (mean of QC-passed pixel spectra)
    1. cosmic-ray removal     -> `remove_cosmic_rays`
    2. arPLS baseline fit     -> internal `arpls(...)` call (so we keep baseline)
    3. baseline subtracted    -> raw_cosmic - baseline
    4. Savitzky-Golay smooth  -> `smooth_savgol`
    5. crop fingerprint+CH    -> `crop_two_regions`  (2048 -> 987 bins)
    6. SNV normalize          -> `snv`

Emits `ui/public/data/preprocessing.json` containing 4 exemplars, each with
all 7 stage traces + the fitted baseline, plus a static `pipeline` description
array consumed by the UI side panel.

Atlas/preprocess.py functions used:
    - remove_cosmic_rays (atomic, default knobs)
    - pybaselines.whittaker.arpls (called directly to capture baseline trace)
    - smooth_savgol (atomic, default knobs window=9, polyorder=3)
    - crop_two_regions (atomic, default 400-1800 + 2800-3050)
    - snv (atomic)

Run from the project root or `ui/`:
    cd ui && python scripts/build_preprocessing.py
    # or with uv:
    cd ui && uv run scripts/build_preprocessing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# This script lives at: <repo>/ui/scripts/build_preprocessing.py
SCRIPT_DIR = Path(__file__).resolve().parent
UI_DIR = SCRIPT_DIR.parent
REPO_ROOT = UI_DIR.parent
DATA_CACHE = REPO_ROOT / "data_cache"
OUT_PATH = UI_DIR / "public" / "data" / "preprocessing.json"

# Ensure the repo root is on sys.path so `import atlas.preprocess` works
# when this script is executed from anywhere.
sys.path.insert(0, str(REPO_ROOT))

# ruff: noqa: E402  -- import after sys.path tweak is intentional.
from atlas.preprocess import (  # noqa: E402
    crop_two_regions,
    remove_cosmic_rays,
    smooth_savgol,
    snv,
)
from pybaselines.whittaker import arpls  # noqa: E402


CLASSES = ("STEC", "Non-STEC", "Salmonella", "H2O")


def _round_list(arr: np.ndarray, decimals: int = 4) -> list[float]:
    """Convert a float array to a JSON-friendly rounded list."""
    rounded = np.round(np.asarray(arr, dtype=np.float64), decimals)
    rounded = np.nan_to_num(rounded, nan=0.0, posinf=0.0, neginf=0.0)
    return [float(x) for x in rounded.tolist()]


def _pick_exemplars(metadata: pd.DataFrame) -> list[tuple[str, str, str | None]]:
    """Pick first-alphabetical file_id per primary_class for stability.

    Returns list of (file_id, primary_class, subclass).
    """
    picks: list[tuple[str, str, str | None]] = []
    for klass in CLASSES:
        sub = metadata[metadata["primary_class"] == klass].sort_values("file_id")
        if sub.empty:
            raise SystemExit(f"no files in metadata for class {klass}")
        row = sub.iloc[0]
        subclass_val = row["subclass"]
        subclass: str | None
        if subclass_val is None or (
            isinstance(subclass_val, float) and np.isnan(subclass_val)
        ):
            subclass = None
        else:
            subclass = str(subclass_val)
        picks.append((str(row["file_id"]), klass, subclass))
    return picks


def _mean_spectrum_for_file(
    file_id: str,
    raw_arr: np.ndarray,
    qc_mask: np.ndarray,
    file_ids_per_pixel: np.ndarray,
) -> np.ndarray:
    """Mean of this file's QC-passed pixel spectra (fallback: all pixels)."""
    pixel_mask = file_ids_per_pixel == file_id
    combined = pixel_mask & qc_mask
    if combined.sum() == 0:
        combined = pixel_mask
    rows = np.asarray(raw_arr[combined], dtype=np.float64)
    return rows.mean(axis=0).astype(np.float32)


def _run_pipeline(
    raw: np.ndarray, wn: np.ndarray
) -> dict[str, np.ndarray]:
    """Apply each atomic step and capture intermediate traces.

    Returns dict with keys:
        raw           (2048)  -- input
        cosmic_ray    (2048)  -- after cosmic-ray removal
        baseline      (2048)  -- the arPLS baseline that was subtracted
        after_baseline(2048)  -- cosmic - baseline
        smoothed      (2048)  -- after Savitzky-Golay smoothing
        cropped       (987)   -- after fingerprint+C-H crop
        snv           (987)   -- after SNV normalization
    """
    s_raw = np.asarray(raw, dtype=np.float32)

    s_cosmic = remove_cosmic_rays(s_raw)

    # Capture the baseline so the UI can show what arPLS subtracted.
    # arpls() returns (baseline, params).
    baseline, _ = arpls(s_cosmic, lam=1e5, max_iter=50, diff_order=2)
    baseline = np.asarray(baseline, dtype=np.float32)
    s_after_baseline = (s_cosmic - baseline).astype(np.float32)

    s_smoothed = smooth_savgol(s_after_baseline, window=9, polyorder=3)

    s_cropped, _wn_cropped, _keep = crop_two_regions(s_smoothed, wn)
    s_snv = snv(s_cropped)

    return {
        "raw": s_raw,
        "cosmic_ray": s_cosmic,
        "baseline": baseline,
        "after_baseline": s_after_baseline,
        "smoothed": s_smoothed,
        "cropped": np.asarray(s_cropped, dtype=np.float32),
        "snv": np.asarray(s_snv, dtype=np.float32),
    }


PIPELINE_DESCRIPTION = [
    {
        "key": "raw",
        "label": "Raw 2048-bin",
        "description": (
            "Per-pixel intensity counts straight off the Raman microscope CCD. "
            "2048 bins spanning ~76-3499 cm-1."
        ),
        "algorithm": "-",
    },
    {
        "key": "cosmic_ray",
        "label": "Cosmic-ray removal",
        "description": (
            "Single-bin spikes from cosmic rays hitting the CCD get nuked via "
            "a MAD-robust median filter at z=5 sigma."
        ),
        "algorithm": "MAD median filter, z-threshold 5 sigma",
    },
    {
        "key": "after_baseline",
        "label": "arPLS baseline subtracted",
        "description": (
            "Fluorescence creates a slow-varying baseline that buries the Raman "
            "peaks. arPLS fits and subtracts a sparse Whittaker baseline."
        ),
        "algorithm": "pybaselines.whittaker.arpls, lambda=1e5, max_iter=50",
    },
    {
        "key": "smoothed",
        "label": "Savitzky-Golay smoothed",
        "description": (
            "A low-order polynomial smoother preserves Raman peak shape while "
            "killing high-frequency CCD noise."
        ),
        "algorithm": "window=9, polyorder=3",
    },
    {
        "key": "cropped",
        "label": "Cropped to fingerprint + C-H stretch",
        "description": (
            "Drops the silent 1800-2800 cm-1 region and noisy edges; keeps "
            "400-1800 (fingerprint) + 2800-3050 (C-H stretch)."
        ),
        "algorithm": "linear crop to 987 bins",
    },
    {
        "key": "snv",
        "label": "SNV normalized",
        "description": (
            "Standard Normal Variate: per-spectrum z-score. Corrects "
            "multiplicative scatter from focus + cell density variation."
        ),
        "algorithm": "x <- (x - mu) / sigma",
    },
]


def main() -> None:
    print(f"[build_preprocessing] repo root: {REPO_ROOT}")
    print(f"[build_preprocessing] data_cache: {DATA_CACHE}")
    print(f"[build_preprocessing] out: {OUT_PATH}")

    raw_arr = np.load(DATA_CACHE / "spectra_array.npy", mmap_mode="r")
    wn = np.load(DATA_CACHE / "wavenumber_axis.npy")
    qc_mask = np.load(DATA_CACHE / "qc_mask.npy").astype(bool)
    metadata = pd.read_parquet(DATA_CACHE / "metadata.parquet")
    spectra_meta = pd.read_parquet(DATA_CACHE / "spectra.parquet")
    file_ids_per_pixel = spectra_meta["file_id"].to_numpy()

    assert raw_arr.shape[1] == wn.shape[0], (
        f"raw_arr cols {raw_arr.shape[1]} != wn size {wn.shape[0]}"
    )
    assert raw_arr.shape[0] == qc_mask.shape[0], "qc_mask row mismatch"
    print(
        f"[build_preprocessing] loaded raw={raw_arr.shape} wn={wn.shape} "
        f"qc_pass={int(qc_mask.sum())}/{raw_arr.shape[0]}"
    )

    picks = _pick_exemplars(metadata)
    print(
        f"[build_preprocessing] picks: "
        + ", ".join(f"{c}={fid}" for fid, c, _ in picks)
    )

    exemplars: list[dict] = []
    for file_id, primary_class, subclass in picks:
        mean_raw = _mean_spectrum_for_file(
            file_id, raw_arr, qc_mask, file_ids_per_pixel
        )
        stages = _run_pipeline(mean_raw, wn)

        # The wn axis used by stages "raw" through "smoothed" is `wn` itself
        # (2048 bins). The "cropped" + "snv" stages use the cropped axis.
        _cropped_check, wn_cropped, _ = crop_two_regions(mean_raw, wn)

        # Raw-space stages carry large absolute counts (~10^3-10^4) so 2
        # decimal places of precision are plenty for visualization. Cropped +
        # SNV stages are normalized; keep 4 decimals there.
        exemplars.append(
            {
                "file_id": file_id,
                "primary_class": primary_class,
                "subclass": subclass,
                "wn_raw": _round_list(wn, decimals=2),
                "wn_cropped": _round_list(wn_cropped, decimals=2),
                "stages": {
                    "raw": _round_list(stages["raw"], decimals=2),
                    "cosmic_ray": _round_list(stages["cosmic_ray"], decimals=2),
                    "baseline": _round_list(stages["baseline"], decimals=2),
                    "after_baseline": _round_list(
                        stages["after_baseline"], decimals=2
                    ),
                    "smoothed": _round_list(stages["smoothed"], decimals=2),
                    "cropped": _round_list(stages["cropped"], decimals=4),
                    "snv": _round_list(stages["snv"], decimals=4),
                },
            }
        )
        print(
            f"[build_preprocessing]   {primary_class:10s} {file_id} done "
            f"(raw_bins={len(stages['raw'])} cropped_bins={len(stages['cropped'])})"
        )

    payload = {
        "exemplars": exemplars,
        "pipeline": PIPELINE_DESCRIPTION,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, separators=(",", ":")))
    size_kb = OUT_PATH.stat().st_size / 1024
    print(
        f"[build_preprocessing] done -> {OUT_PATH.name} "
        f"({size_kb:.1f} KB, {len(exemplars)} exemplars)"
    )


if __name__ == "__main__":
    main()
