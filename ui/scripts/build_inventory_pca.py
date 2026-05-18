"""Atlas Raman UI — Inventory PCA sidecar builder (W11).

Computes a 3-component PCA over the full 259-feature engineered design
matrix at the FILE level (87 × 259) and emits
`ui/public/data/inventory_pca.json` for the Inventory tab's 3D
feature-space scatter.

Source caches (project root `data_cache/`):
  - band_features.parquet      (7,122 × 166, per-pixel)
  - spectral_features.parquet  (7,122 × 51,  per-pixel)
  - unmix_features.parquet     (87 × 33,    per-file — drop
                                `mcr_residual_norm_mean` per W5)
  - spatial_features.parquet   (87 × 10,    per-file)
  - spectra.parquet            (7,999 rows, used for pixel → file_id map)
  - qc_mask.npy                (7,999 bool, 7,122 pass)
  - metadata.parquet           (87 rows, file_id → primary_class/subclass)

Per-pixel parquets are mean-pooled to file level via QC-masked groupby
file_id (same pattern as `build_features.py`). The combined 87 × 259
matrix is standardized with `StandardScaler` and reduced with
`PCA(n_components=3)`. Per-pixel QC pass rate is recomputed from
`qc_mask.npy` so the sidecar is self-contained.

Sidecar contract (mirrored in `ui/components/plots/InventoryFeatureSpace3D.tsx`):

    {
      "files": [
        {"file_id": str, "primary_class": str, "subclass": str|null,
         "pc1": float, "pc2": float, "pc3": float,
         "n_pixels": int, "qc_pass_rate": float},
        ...  # 87 entries
      ],
      "variance_explained": [float, float, float],
      "n_features": 259
    }

PCs rounded to 4 decimals; QC pass rate to 4 decimals.

Plan reference: `plan/ui/ULTRAPLAN.md` §4 W11 (UI inventory PCA addition).

Usage (idempotent):
    cd ui && python scripts/build_inventory_pca.py
    # or:
    cd ui && uv run scripts/build_inventory_pca.py

# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn"]
# ///
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


UI_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = UI_DIR.parent
DATA_CACHE = REPO_ROOT / "data_cache"
OUT_PATH = UI_DIR / "public" / "data" / "inventory_pca.json"

EXPECTED_FEATURES = 259


def load_file_level_design() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return:
      - design: 87 × 259 file-level feature matrix (file_id index)
      - meta:   87-row metadata (file_id, primary_class, subclass, n_pixels)
      - qc_pass_rate: per-file QC pass rate (file_id index, float)
    """

    metadata = pd.read_parquet(DATA_CACHE / "metadata.parquet").set_index("file_id")

    qc_mask = np.load(DATA_CACHE / "qc_mask.npy").astype(bool)
    spectra = pd.read_parquet(DATA_CACHE / "spectra.parquet")
    if len(spectra) != qc_mask.size:
        raise RuntimeError(
            f"qc_mask ({qc_mask.size}) and spectra ({len(spectra)}) mismatch"
        )

    # Per-file QC pass rate from raw mask: kept / total.
    spectra_with_qc = spectra.assign(__qc__=qc_mask)
    qc_grp = spectra_with_qc.groupby("file_id")["__qc__"]
    qc_pass_rate = (qc_grp.sum() / qc_grp.size()).astype(float)

    spectra_qc = spectra.loc[qc_mask].reset_index(drop=True)

    # ---- per-pixel parquets -> file-level mean pool ----
    band = pd.read_parquet(DATA_CACHE / "band_features.parquet")
    spectral = pd.read_parquet(DATA_CACHE / "spectral_features.parquet")
    if len(band) != len(spectra_qc) or len(spectral) != len(spectra_qc):
        raise RuntimeError(
            "per-pixel cache row counts don't match QC-passed spectra; "
            f"band={len(band)} spectral={len(spectral)} qc={len(spectra_qc)}"
        )

    file_ids = spectra_qc["file_id"].to_numpy()
    band_num = band.apply(pd.to_numeric, errors="coerce")
    spectral_num = spectral.apply(pd.to_numeric, errors="coerce")
    band_file = band_num.groupby(file_ids).mean()
    spectral_file = spectral_num.groupby(file_ids).mean()
    band_file.index.name = "file_id"
    spectral_file.index.name = "file_id"

    # ---- per-file parquets ----
    unmix = pd.read_parquet(DATA_CACHE / "unmix_features.parquet")
    # Drop the residual column (not a per-component feature; matches W5).
    if "mcr_residual_norm_mean" in unmix.columns:
        unmix = unmix.drop(columns=["mcr_residual_norm_mean"])
    spatial = pd.read_parquet(DATA_CACHE / "spatial_features.parquet")

    file_order = metadata.index
    band_file = band_file.reindex(file_order)
    spectral_file = spectral_file.reindex(file_order)
    unmix = unmix.reindex(file_order)
    spatial = spatial.reindex(file_order)

    design = pd.concat([band_file, spectral_file, unmix, spatial], axis=1)
    design = design.loc[:, ~design.columns.duplicated()]

    if design.shape != (87, EXPECTED_FEATURES):
        raise SystemExit(
            f"design shape {design.shape} != (87, {EXPECTED_FEATURES}); "
            "upstream caches changed — update EXPECTED_FEATURES or fix join."
        )

    qc_pass_rate = qc_pass_rate.reindex(file_order).fillna(0.0)

    return design, metadata, qc_pass_rate


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    design, metadata, qc_pass_rate = load_file_level_design()

    # Median-impute any NaN slots (a handful of bio_* / mcr_* features have
    # NaN for select files — leave-one-out would re-fit anyway; for the
    # global-fit scatter we just need stable PCs).
    medians = design.median(numeric_only=True)
    design_filled = design.fillna(medians)
    # If any column was entirely NaN (shouldn't happen), drop it explicitly.
    all_nan_cols = design_filled.columns[design_filled.isna().any()].tolist()
    if all_nan_cols:
        raise RuntimeError(
            f"columns still NaN after median fill: {all_nan_cols[:5]} ..."
        )

    X = design_filled.to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=3, random_state=0)
    pcs = pca.fit_transform(Xs)
    variance = pca.explained_variance_ratio_.tolist()

    files_out: list[dict] = []
    for i, file_id in enumerate(design_filled.index):
        row = metadata.loc[file_id]
        subclass = row["subclass"]
        if subclass is None or (isinstance(subclass, float) and np.isnan(subclass)):
            subclass_val: str | None = None
        else:
            subclass_val = str(subclass)
        files_out.append(
            {
                "file_id": str(file_id),
                "primary_class": str(row["primary_class"]),
                "subclass": subclass_val,
                "pc1": round(float(pcs[i, 0]), 4),
                "pc2": round(float(pcs[i, 1]), 4),
                "pc3": round(float(pcs[i, 2]), 4),
                "n_pixels": int(row["n_pixels"]),
                "qc_pass_rate": round(float(qc_pass_rate.loc[file_id]), 4),
            }
        )

    payload = {
        "files": files_out,
        "variance_explained": [round(float(v), 4) for v in variance],
        "n_features": int(design_filled.shape[1]),
    }

    if len(files_out) != 87:
        raise SystemExit(f"expected 87 files, got {len(files_out)}")
    if len(payload["variance_explained"]) != 3:
        raise SystemExit("variance_explained must have length 3")

    with OUT_PATH.open("w") as fh:
        json.dump(payload, fh, indent=2)

    size_kb = OUT_PATH.stat().st_size / 1024
    var_pct = [round(100 * v, 1) for v in variance]
    print(
        f"wrote {OUT_PATH.relative_to(REPO_ROOT)} — "
        f"{len(files_out)} files, "
        f"variance explained {var_pct[0]}% / {var_pct[1]}% / {var_pct[2]}% "
        f"(total {sum(var_pct):.1f}%), "
        f"{size_kb:.1f} KB"
    )


if __name__ == "__main__":
    main()
