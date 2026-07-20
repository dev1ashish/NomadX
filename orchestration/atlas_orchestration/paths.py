"""Filesystem anchors for the orchestration layer.

Everything is resolved relative to the repo root so `dagster dev` works no
matter the cwd. NOTHING here is a data value — these are just locations. All
displayed numbers/schemas are read from the files these point at, at
materialization time.
"""

from __future__ import annotations

from pathlib import Path

# orchestration/atlas_orchestration/paths.py -> repo root is two parents up
PKG_DIR = Path(__file__).resolve().parent
ORCH_DIR = PKG_DIR.parent
REPO_ROOT = ORCH_DIR.parent

DATA_CACHE = REPO_ROOT / "data_cache"
ARTIFACTS = REPO_ROOT / "artifacts"
CONTRACT_DIR = ORCH_DIR / "contract"

# --- raw / ingest stage ---
METADATA_PARQUET = DATA_CACHE / "metadata.parquet"
SPECTRA_PARQUET = DATA_CACHE / "spectra.parquet"
SPECTRA_ARRAY = DATA_CACHE / "spectra_array.npy"
WN_AXIS = DATA_CACHE / "wavenumber_axis.npy"
BUILD_LOG = DATA_CACHE / "build.log"

# --- quality stage ---
QC_MASK = DATA_CACHE / "qc_mask.npy"
QC_INFO = DATA_CACHE / "qc_info.json"

# --- preprocess stage ---
SPECTRA_PREP = DATA_CACHE / "spectra_array_preprocessed.npy"
WN_AXIS_PREP = DATA_CACHE / "wavenumber_axis_preprocessed.npy"

# --- feature stage (the feature store) ---
BAND_FEATURES = DATA_CACHE / "band_features.parquet"
SPECTRAL_FEATURES = DATA_CACHE / "spectral_features.parquet"
UNMIX_FEATURES = DATA_CACHE / "unmix_features.parquet"
SPATIAL_FEATURES = DATA_CACHE / "spatial_features.parquet"

# --- serving (downstream of the handoff; shown for lineage context) ---
PLSDA_META = ARTIFACTS / "plsda_raw_metadata.json"
STAGE15F_META = ARTIFACTS / "stage15f_metadata.json"


def rel(p: Path) -> str:
    """Repo-relative string for display."""
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)
