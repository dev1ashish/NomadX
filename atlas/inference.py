"""Stage 15F production inference API.

Loads the 5 artifacts produced by `scripts/run_stage15f_final.py` and exposes
two prediction entry points:

    predict_from_xls(path)               — parse a raw Atlas .xls / .txt → label
    predict_from_array(X, wn)            — caller already has preprocessed spectra

Both return a dict:
    {'class': str,
     'probabilities': dict[str, float],
     'spectrum_mean': np.ndarray,      # mean preprocessed spectrum across pixels
     'wn': np.ndarray,                 # preprocessed wavenumber axis
     'feature_values': dict[str, float]}  # values for the model's input features

Artifact paths are resolved from `ATLAS_ARTIFACTS_DIR` env var if set, otherwise
the `artifacts/` directory at the repo root.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from atlas.band_features import feature_frame
from atlas.io import CANONICAL_WN, FileRecord, parse_file
from atlas.preprocess import preprocess_matrix
from atlas.spatial_features import feature_frame_spatial
from atlas.spectral_features import (
    LPS_REGION_FOR_SAM,
    dwt_features,
    transform_roi_pca,
    transform_sam,
)
from atlas.unmix_features import mcr_concentration_summary


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _artifacts_dir() -> Path:
    env = os.environ.get("ATLAS_ARTIFACTS_DIR")
    if env:
        return Path(env)
    return _REPO_ROOT / "artifacts"


# ---------------------------------------------------------------------------
# Artifact loading (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_artifacts() -> dict[str, Any]:
    d = _artifacts_dir()
    classifier = joblib.load(d / "stage15f_classifier.joblib")
    feature_columns = json.loads((d / "stage15f_feature_columns.json").read_text())
    mcr = joblib.load(d / "stage15f_mcr_global.joblib")
    roi_pca = joblib.load(d / "stage15f_roi_pca.joblib")
    sam_templates = joblib.load(d / "stage15f_sam_templates.joblib")
    metadata = json.loads((d / "stage15f_metadata.json").read_text())
    return dict(
        classifier=classifier,
        feature_columns=feature_columns,
        mcr=mcr,
        roi_pca=roi_pca,
        sam_templates=sam_templates,
        metadata=metadata,
    )


def model_metadata() -> dict[str, Any]:
    """Return the metadata JSON saved at training time. Cached."""
    return _load_artifacts()["metadata"]


# ---------------------------------------------------------------------------
# Feature builders shared by both entry points
# ---------------------------------------------------------------------------

def _build_per_pixel_features(X_pp: np.ndarray, wn: np.ndarray) -> pd.DataFrame:
    """Run band + spectral DWT/PCA/SAM feature extraction on preprocessed spectra.

    PCA + SAM use frozen artifacts (no train-time leakage).
    """
    art = _load_artifacts()
    band_df = feature_frame(X_pp, wn)                   # (N_pix, 166)

    # Spectral: DWT (label-free) + PCA (frozen) + SAM (frozen)
    dwt = dwt_features(X_pp, wavelet="db4", max_level=6)
    roi = transform_roi_pca(X_pp, wn, art["roi_pca"])
    sam = transform_sam(X_pp, art["sam_templates"], wn=wn)
    spectral_cols = {**dwt, **roi, **sam}
    spectral_df = pd.DataFrame(spectral_cols)
    return pd.concat([band_df, spectral_df], axis=1)


def _build_file_level_row(X_pp: np.ndarray, wn: np.ndarray,
                          file_id: str = "uploaded") -> pd.DataFrame:
    """Assemble the 1-row file-level feature DataFrame the classifier expects."""
    art = _load_artifacts()
    n_pix = X_pp.shape[0]
    pix_df = _build_per_pixel_features(X_pp, wn)
    pix_df["file_id"] = file_id
    file_means = pix_df.groupby("file_id").mean(numeric_only=True)

    # MCR concentrations (frozen) → per-file summary
    X_offset = X_pp - X_pp.min()
    mcr_C = art["mcr"].transform(X_offset)
    mcr_df = mcr_concentration_summary(mcr_C, np.array([file_id] * n_pix))

    # Spatial features need a spec_df-like input
    spec_like = pd.DataFrame({"file_id": [file_id] * n_pix})
    spat_df = feature_frame_spatial(X_pp, wn, spec_like)

    out = file_means.join(mcr_df, how="left").join(spat_df, how="left")
    return out


def _predict_row(file_row: pd.DataFrame, wn: np.ndarray,
                 spectrum_mean: np.ndarray) -> dict[str, Any]:
    """Run the saved classifier on a single file-level row. Returns the
    standard prediction dict described in the module docstring."""
    art = _load_artifacts()
    feats: list[str] = art["feature_columns"]
    missing = [c for c in feats if c not in file_row.columns]
    if missing:
        # The training script only saves columns present at fit time; if any are
        # missing in inference (very rare — only on degenerate uploads), fill
        # with 0 (post-StandardScaler equivalent to "no info").
        for c in missing:
            file_row[c] = 0.0
    X = file_row[feats].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    clf = art["classifier"]
    pred = clf.predict(X)[0]
    proba = None
    classes = list(getattr(clf, "classes_", []))
    if hasattr(clf, "predict_proba"):
        try:
            p = clf.predict_proba(X)[0]
            proba = {str(c): float(p[i]) for i, c in enumerate(classes)}
        except Exception:
            proba = None
    return {
        "class": str(pred),
        "probabilities": proba or {},
        "spectrum_mean": spectrum_mean.astype(np.float32),
        "wn": wn.astype(np.float32),
        "feature_values": {c: float(file_row.iloc[0][c]) for c in feats},
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_from_array(intensities: np.ndarray, wn: np.ndarray,
                       *, preprocess: bool = True) -> dict[str, Any]:
    """Predict from a (N_pixels, B) intensity array + matching wavenumber axis.

    If `preprocess=True`, runs the canonical preprocess pipeline (arPLS + SG + crop + SNV)
    before feature extraction. Set to False if the caller already preprocessed.
    """
    X = np.asarray(intensities, dtype=np.float32)
    wn_arr = np.asarray(wn, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    if preprocess:
        X_pp, wn_pp, _ = preprocess_matrix(X, wn_arr, progress=False)
    else:
        X_pp, wn_pp = X, wn_arr
    spectrum_mean = X_pp.mean(axis=0)
    file_row = _build_file_level_row(X_pp, wn_pp, file_id="uploaded")
    return _predict_row(file_row, wn_pp, spectrum_mean)


@lru_cache(maxsize=1)
def _load_plsda_raw() -> dict[str, Any]:
    """Load the PLS-DA-on-raw-spectrum project-headline classifier.

    Separate from `_load_artifacts()` because PLS-DA-raw doesn't need the
    Stage 15F feature engineering chain (MCR / SAM / PCA / etc.).
    """
    d = _artifacts_dir()
    classifier = joblib.load(d / "plsda_raw_classifier.joblib")
    metadata = json.loads((d / "plsda_raw_metadata.json").read_text())
    return dict(classifier=classifier, metadata=metadata)


def predict_from_array_plsda(intensities: np.ndarray, wn: np.ndarray,
                             *, preprocess: bool = True) -> dict[str, Any]:
    """Run the PLS-DA-on-raw-spectrum model and aggregate to a file-level prob.

    Pipeline: (optional preprocess) -> per-pixel predict_proba -> mean across
    pixels -> argmax. Matches the file-level scoring used during the LOSO
    bake-off (atlas.evaluate.file_aggregate_softvote, 2026-05-14 run).

    Returns the same shape as predict_from_array() so callers can switch
    between models with one keyword.
    """
    X = np.asarray(intensities, dtype=np.float32)
    wn_arr = np.asarray(wn, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    if preprocess:
        X_pp, wn_pp, _ = preprocess_matrix(X, wn_arr, progress=False)
    else:
        X_pp, wn_pp = X, wn_arr
    spectrum_mean = X_pp.mean(axis=0)

    clf = _load_plsda_raw()["classifier"]
    scaler = clf.named_steps["scaler"]
    plsda = clf.named_steps["clf"]
    classes = list(plsda.classes_)
    proba_per_pixel = clf.predict_proba(X_pp)        # (N_pix, K)
    file_proba = proba_per_pixel.mean(axis=0)        # soft-vote across pixels
    top_idx = int(np.argmax(file_proba))

    # ---- Spectral loadings (interpretability surface for the UI) -----------
    # The full PLS-DA chain is linear in the standardized input space:
    #   log_odds_k = (x_std - pls.x_mean_) @ (pls.x_rotations_ @ logreg.coef_[k])
    # So `w_k = pls.x_rotations_ @ logreg.coef_[k]` is the per-wavenumber
    # sensitivity for class k — "how much each bin pushes the prediction
    # toward class k". Same for every input (global).
    #
    # Per-file per-bin contribution to the predicted class log-odds:
    #   contrib[b] = (x_std_mean[b] - pls.x_mean_[b]) * w_pred[b]
    # — this is what made the model say what it said for THIS file.
    rotations = np.asarray(plsda.x_rotations_)                # (B, n_components)
    coef = np.asarray(plsda.logreg_.coef_)                    # (K, n_components)
    loadings = rotations @ coef.T                             # (B, K)
    X_std = scaler.transform(X_pp)                            # (N_pix, B)
    x_std_mean = X_std.mean(axis=0)                           # (B,)
    pls_x_mean = np.asarray(getattr(plsda, "x_mean_", np.zeros_like(x_std_mean)))
    centered = x_std_mean - pls_x_mean                        # (B,)
    contribution_predicted = centered * loadings[:, top_idx]  # (B,)
    loadings_per_class = {
        str(c): loadings[:, i].astype(np.float32).tolist()
        for i, c in enumerate(classes)
    }

    return {
        "class": str(classes[top_idx]),
        "probabilities": {str(c): float(file_proba[i]) for i, c in enumerate(classes)},
        "spectrum_mean": spectrum_mean.astype(np.float32),
        "wn": wn_pp.astype(np.float32),
        # PLS-DA-raw has no engineered features; keep the key for shape parity.
        "feature_values": {},
        # Interpretability surfaces (PLS-DA-only — empty/absent on /predict).
        "loadings_per_class": loadings_per_class,
        "contribution_for_predicted": contribution_predicted.astype(np.float32).tolist(),
    }


def predict_from_xls(path: str | Path) -> dict[str, Any]:
    """Parse a raw Atlas .xls/.txt file → preprocess → predict.

    The file's path relative root is inferred — passing a file from outside
    `Atlas Data/` is fine; we only need the class folder for `parse_file`'s
    bookkeeping, which we synthesize via a temporary structure.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    # parse_file requires a `data_root` that is the parent of the class folder
    # so it can derive primary_class/subclass. For inference we don't care
    # about the class; we just need the spectra. Use a permissive synthetic root.
    rec = _parse_inference_file(p)
    if not rec.is_valid or rec.intensities is None:
        raise ValueError(f"failed to parse {p}: errors={rec.fatal_errors}")

    X_raw = rec.intensities.astype(np.float32)
    wn = CANONICAL_WN
    X_pp, wn_pp, _ = preprocess_matrix(X_raw, wn, progress=False)
    spectrum_mean = X_pp.mean(axis=0)
    file_row = _build_file_level_row(X_pp, wn_pp, file_id=p.stem)
    return _predict_row(file_row, wn_pp, spectrum_mean)


def _parse_inference_file(path: Path) -> FileRecord:
    """Wrap `atlas.io.parse_file` to tolerate arbitrary upload locations.

    parse_file expects `data_root` such that `path` lives under
    `data_root/<ClassFolder>/...`. For uploaded files outside `Atlas Data/`,
    we monkey-build a FileRecord by re-using parse_file with a synthetic
    data_root (the file's parent), and fall back to bypassing classification
    if the path lookup fails.
    """
    # Try parsing under the canonical Atlas Data/ root if applicable
    try:
        atlas_root = _REPO_ROOT / "Atlas Data"
        if atlas_root in path.parents:
            return parse_file(path, atlas_root)
    except Exception:
        pass

    # Fallback: use a synthetic 2-level structure so parse_file accepts the path
    # We don't actually use primary_class/subclass for inference.
    parent = path.parent
    # parse_file expects path.relative_to(data_root) with ≥2 parts and a
    # recognized top-level CLASS folder. We satisfy this by creating a
    # symlink-shaped fake root in-memory: data_root = parent.parent if parent
    # is inside a CLASS_MAP folder, else use parent and patch via try/except.
    try:
        return parse_file(path, parent.parent)
    except Exception:
        return _parse_inference_file_minimal(path)


def _parse_inference_file_minimal(path: Path) -> FileRecord:
    """Last-resort: parse the file ignoring class layout (for arbitrary uploads).

    Re-implements just enough of parse_file to get x/y/intensities. Used when
    the uploaded file isn't under Atlas Data/.
    """
    # Re-use parse_file's logic by stubbing in a CLASS_MAP-compliant parent.
    # Simpler: shell-parse the file ourselves.
    from atlas.io import N_BINS, _to_float_safe, _strip_commas, _parse_header_lines

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    _header, body_start = _parse_header_lines(lines)
    wn_row = lines[body_start].split("\t")
    wn_values_raw = [t for t in wn_row[2:] if t.strip() != ""]
    wn_native = np.array([_to_float_safe(t) for t in wn_values_raw], dtype=np.float64)
    pixel_lines = lines[body_start + 1:]
    xs: list[float] = []
    ys: list[float] = []
    raws: list[np.ndarray] = []
    for ln in pixel_lines:
        if not ln.strip():
            continue
        toks = ln.split("\t")
        if len(toks) < 3:
            continue
        try:
            xs.append(float(_strip_commas(toks[0])))
            ys.append(float(_strip_commas(toks[1])))
        except ValueError:
            continue
        ints_tokens = toks[2: 2 + N_BINS]
        if len(ints_tokens) < N_BINS:
            xs.pop(); ys.pop()
            continue
        arr = np.fromiter(
            (_to_float_safe(t) for t in ints_tokens),
            dtype=np.float64, count=N_BINS,
        )
        if not np.all(np.isfinite(arr)):
            xs.pop(); ys.pop()
            continue
        raws.append(arr)
    if not raws:
        raise ValueError(f"no valid pixel rows in {path}")
    raw_mat = np.asarray(raws, dtype=np.float32)
    interp = np.empty((raw_mat.shape[0], CANONICAL_WN.size), dtype=np.float32)
    for i in range(raw_mat.shape[0]):
        interp[i] = np.interp(CANONICAL_WN, wn_native, raw_mat[i]).astype(np.float32)
    rec = FileRecord(
        file_id=path.stem, file_path=str(path),
        primary_class="unknown", subclass=None,
        n_pixels=int(interp.shape[0]),
        grid_nx=len(set(xs)), grid_ny=len(set(ys)),
        header_numx=None, header_numy=None,
        xsize=None, ysize=None,
        laser=None, exposure_ms=None,
        acquisition_date=None, ac_calibration_date=None,
        wn_start=float(wn_native[0]), wn_end=float(wn_native[-1]),
        is_complete_scan=True,
        file_sha256="", file_mtime=0.0, file_size_bytes=path.stat().st_size,
    )
    rec.x_um = np.asarray(xs, dtype=np.float32)
    rec.y_um = np.asarray(ys, dtype=np.float32)
    rec.intensities = interp
    return rec
