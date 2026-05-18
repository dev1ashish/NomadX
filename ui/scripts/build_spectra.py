"""Atlas Raman UI — W3 sidecar builder for the Spectrum Explorer tab.

Reads from the project root `data_cache/` (see plan/ui/ULTRAPLAN.md §W3 +
§6 worker rules) and emits per-file mean-spectrum JSON sidecars under
`ui/public/data/spectra/`, plus a top-level index at
`ui/public/data/spectra_index.json`.

What this script does (in plain English):
  1. Load the global raw and preprocessed spectra arrays (one row per pixel).
  2. Load the QC pass mask so we mean only over QC-passed pixels.
  3. Load metadata.parquet (one row per file, 87 entries) and spectra.parquet
     (one row per pixel — gives us the pixel -> file_id mapping that lets us
     slice contiguous pixel ranges per file).
  4. For each of the 87 files, compute a mean spectrum across that file's
     QC-passed pixels in both raw and preprocessed spaces.
  5. Round floats to 4 decimals (keeps each JSON well under 40 KB).
  6. Write `<file_id>.json` and a single `spectra_index.json`.

Usage:
    cd ui && python scripts/build_spectra.py

The script is idempotent: re-running overwrites existing sidecars.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# This script lives at: <repo>/ui/scripts/build_spectra.py
# REPO_ROOT is two levels up.
SCRIPT_DIR = Path(__file__).resolve().parent
UI_DIR = SCRIPT_DIR.parent
REPO_ROOT = UI_DIR.parent
DATA_CACHE = REPO_ROOT / "data_cache"
OUT_DIR = UI_DIR / "public" / "data"
SPECTRA_OUT = OUT_DIR / "spectra"


def _round_list(arr: np.ndarray, decimals: int = 4) -> list[float]:
    """Convert a float array to a plain Python list rounded to `decimals` places.

    Using `float(round(x, n))` per element guarantees clean JSON output without
    scientific notation (e.g. `0.0001` not `1e-04`) and trims trailing zeros
    via json's default float repr.
    """
    rounded = np.round(arr.astype(np.float64), decimals)
    # Replace NaN/Inf with 0 so the JSON stays valid.
    rounded = np.nan_to_num(rounded, nan=0.0, posinf=0.0, neginf=0.0)
    return [float(x) for x in rounded.tolist()]


def main() -> None:
    print(f"[build_spectra] repo root: {REPO_ROOT}")
    print(f"[build_spectra] data_cache: {DATA_CACHE}")
    print(f"[build_spectra] out dir:   {OUT_DIR}")

    # --- Load inputs --------------------------------------------------------
    raw_arr = np.load(DATA_CACHE / "spectra_array.npy", mmap_mode="r")
    pp_arr = np.load(DATA_CACHE / "spectra_array_preprocessed.npy", mmap_mode="r")
    wn_raw = np.load(DATA_CACHE / "wavenumber_axis.npy")
    wn_pp = np.load(DATA_CACHE / "wavenumber_axis_preprocessed.npy")
    qc_mask = np.load(DATA_CACHE / "qc_mask.npy").astype(bool)
    metadata = pd.read_parquet(DATA_CACHE / "metadata.parquet")
    spectra_meta = pd.read_parquet(DATA_CACHE / "spectra.parquet")

    n_pix = raw_arr.shape[0]
    assert pp_arr.shape[0] == n_pix, (
        f"raw rows {n_pix} != preprocessed rows {pp_arr.shape[0]}"
    )
    assert qc_mask.shape[0] == n_pix, (
        f"qc_mask rows {qc_mask.shape[0]} != spectra rows {n_pix}"
    )
    assert wn_raw.shape[0] == raw_arr.shape[1], "wn_raw size mismatch"
    assert wn_pp.shape[0] == pp_arr.shape[1], "wn_pp size mismatch"
    assert len(metadata) == 87, f"expected 87 files in metadata, got {len(metadata)}"
    assert len(spectra_meta) == n_pix, (
        f"spectra.parquet rows {len(spectra_meta)} != npy rows {n_pix}"
    )

    print(
        f"[build_spectra] loaded raw={raw_arr.shape} pp={pp_arr.shape} "
        f"qc_pass={int(qc_mask.sum())}/{n_pix} files={len(metadata)}"
    )

    # --- Output dirs --------------------------------------------------------
    SPECTRA_OUT.mkdir(parents=True, exist_ok=True)

    # --- Per-file mean spectra ---------------------------------------------
    # spectra.parquet's file_id column is contiguous per file in the same order
    # as metadata. Build a (file_id -> [start, end)) row-range map.
    file_ids_per_pixel = spectra_meta["file_id"].to_numpy()

    index_entries: list[dict] = []
    total_size_bytes = 0
    sample_first_size: int | None = None

    for file_idx, row in metadata.iterrows():
        file_id = str(row["file_id"])
        primary_class = str(row["primary_class"])
        subclass_val = row["subclass"]
        subclass: str | None
        if subclass_val is None or (isinstance(subclass_val, float) and np.isnan(subclass_val)):
            subclass = None
        else:
            subclass = str(subclass_val)

        # Slice pixels belonging to this file
        pixel_mask = file_ids_per_pixel == file_id
        n_total = int(pixel_mask.sum())
        # Combine with QC mask
        combined = pixel_mask & qc_mask
        n_kept = int(combined.sum())

        if n_kept == 0:
            # Extremely defensive: fall back to all pixels of this file so we
            # never emit an empty spectrum. (Should not happen given QC stats.)
            print(
                f"[build_spectra]  WARN {file_id}: 0 QC-passed pixels of "
                f"{n_total}; falling back to all pixels"
            )
            combined = pixel_mask

        # NB: fancy indexing on a memmap loads into RAM only the selected rows.
        raw_rows = np.asarray(raw_arr[combined], dtype=np.float64)
        pp_rows = np.asarray(pp_arr[combined], dtype=np.float64)
        mean_raw = raw_rows.mean(axis=0)
        mean_pp = pp_rows.mean(axis=0)

        payload = {
            "file_id": file_id,
            "primary_class": primary_class,
            "subclass": subclass,
            "n_pixels": n_total,
            "n_qc_pass": n_kept,
            "wn_raw": _round_list(wn_raw, decimals=2),
            "wn_pp": _round_list(wn_pp, decimals=2),
            "mean_raw": _round_list(mean_raw, decimals=4),
            "mean_pp": _round_list(mean_pp, decimals=4),
        }

        out_path = SPECTRA_OUT / f"{file_id}.json"
        out_path.write_text(json.dumps(payload, separators=(",", ":")))
        size = out_path.stat().st_size
        total_size_bytes += size
        if sample_first_size is None:
            sample_first_size = size
            print(f"[build_spectra]  first file {file_id} -> {size / 1024:.1f} KB")

        index_entries.append(
            {
                "file_id": file_id,
                "primary_class": primary_class,
                "subclass": subclass,
            }
        )

        if (file_idx + 1) % 20 == 0 or (file_idx + 1) == len(metadata):
            print(f"[build_spectra]  wrote {file_idx + 1}/{len(metadata)}")

    # --- Index file ---------------------------------------------------------
    index_path = OUT_DIR / "spectra_index.json"
    index_path.write_text(json.dumps(index_entries, separators=(",", ":")))
    total_size_bytes += index_path.stat().st_size

    print(
        f"[build_spectra] done. {len(index_entries)} files -> "
        f"{SPECTRA_OUT} (total {total_size_bytes / 1024:.1f} KB)"
    )


if __name__ == "__main__":
    main()
