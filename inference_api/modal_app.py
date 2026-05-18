"""Atlas Raman inference -- Modal serverless endpoint.

Wraps atlas.inference.predict_from_xls() 1:1 as a POST /predict endpoint.
Deploys to Modal with `modal deploy inference_api/modal_app.py`.

Verified against Modal docs (https://modal.com/docs/guide/webhooks, fetched
2026-05-19): `@modal.fastapi_endpoint` is the current decorator. Quote
from the docs:

    "The easiest way to make a Python function addressable over the web
    uses the @modal.fastapi_endpoint decorator [...] Note: Prior to
    v0.73.82, this function was named @modal.web_endpoint."

The decorator accepts `method` ("GET" default) and `docs` (enables
FastAPI's interactive /docs page).
"""
from __future__ import annotations

from pathlib import Path

import modal
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

REPO_ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # EXACT pins from FINAL/requirements.txt — match the training env so
        # the saved joblib estimators unpickle without InconsistentVersion
        # warnings and predictions are bit-stable.
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scipy==1.13.1",
        "scikit-learn==1.5.1",
        "pybaselines==1.1.0",
        "PyWavelets>=1.5,<2",
        "pyMCR>=0.5,<1",
        "joblib>=1.3",
        "tqdm>=4.65",
        # Runtime-only deps (not used at training time)
        "fastapi[standard]>=0.110",
    )
    .add_local_dir(str(REPO_ROOT / "atlas"),     remote_path="/root/atlas")
    .add_local_dir(str(REPO_ROOT / "artifacts"), remote_path="/root/artifacts")
)

app = modal.App("atlas-inference", image=image)


CORS_ORIGIN_REGEX = r"https?://(localhost(:\d+)?|.*\.vercel\.app|.*\.modal\.run)"


# Training parsed files with `pixel_cap=200` (atlas/io.py:164). The inference
# fallback `_parse_inference_file_minimal` (atlas/inference.py:240) does NOT
# apply the cap, which causes a train/inference distribution shift on mosaic
# files (R364 = 324 px, R370 = 720 px → model sees a different feature
# vector than the one it learned from). We replicate the cap here.
PIXEL_CAP = 200
# Abstain when the top class is below this — the production model has
# bootstrap CI [0.345, 0.552], so any single low-confidence call should
# be flagged rather than rendered as a confident banner.
ABSTAIN_THRESHOLD = 0.55


@app.function(
    memory=2048,
    cpu=2.0,
    min_containers=0,
    scaledown_window=300,
)
@modal.fastapi_endpoint(method="POST", docs=True)
async def predict(file: UploadFile = File(...)):
    """POST /predict -- multipart upload of an Atlas .xls/.txt file.

    Pipeline: parse → cap to 200 pixels (matches training) → preprocess →
    Stage 15F feature extraction → LogReg-L2 predict.
    """
    import os
    import sys
    import tempfile
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, "/root")
    os.environ["ATLAS_ARTIFACTS_DIR"] = "/root/artifacts"

    from atlas.inference import _parse_inference_file, predict_from_array
    from atlas.io import CANONICAL_WN

    contents = await file.read()
    suffix = ".xls"
    if file.filename and file.filename.lower().endswith(".txt"):
        suffix = ".txt"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    # Parse first so we can subsample to match training distribution.
    rec = _parse_inference_file(tmp_path)
    if not rec.is_valid or rec.intensities is None:
        return JSONResponse(
            {"error": f"failed to parse file: {rec.fatal_errors}"},
            status_code=400,
            headers={"Access-Control-Allow-Origin": "*"},
        )

    X = rec.intensities.astype(np.float32)
    n_pix_raw = int(X.shape[0])

    # Deterministic 200-pixel cap — same seed every call so re-predictions
    # are stable. 200 random pixels is enough for file-level mean features
    # (law of large numbers) and matches the training distribution exactly.
    if n_pix_raw > PIXEL_CAP:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(n_pix_raw, size=PIXEL_CAP, replace=False)
        idx.sort()
        X = X[idx]

    result = predict_from_array(X, CANONICAL_WN, preprocess=True)

    # Compute abstention + top-2 for the UI.
    probs = result["probabilities"]
    sorted_probs = sorted(probs.items(), key=lambda kv: -kv[1])
    top_class = result["class"]
    top_prob = float(probs.get(top_class, 0.0))

    payload = {
        "class": result["class"],
        "probabilities": probs,
        "spectrum_mean": result["spectrum_mean"].tolist(),
        "wn": result["wn"].tolist(),
        "feature_values": result["feature_values"],
        # Extra fields the UI uses for abstention rendering.
        "abstain": top_prob < ABSTAIN_THRESHOLD,
        "top2": [
            {"class": k, "prob": float(v)} for k, v in sorted_probs[:2]
        ],
        "n_pixels_input": n_pix_raw,
        "n_pixels_used": int(X.shape[0]),
        "model": "logreg_stage15f",
    }
    return JSONResponse(
        payload,
        headers={"Access-Control-Allow-Origin": "*"},
    )


@app.function(
    memory=2048,
    cpu=2.0,
    min_containers=0,
    scaledown_window=300,
)
@modal.fastapi_endpoint(method="POST", docs=True)
async def predict_plsda(file: UploadFile = File(...)):
    """POST /predict_plsda -- PLS-DA on raw preprocessed spectrum.

    Same parse / pixel-cap / preprocess as /predict, but runs the
    project-headline PLS-DA classifier (LOSO file-weighted balanced acc =
    0.603). Returns the same payload shape as /predict; `feature_values`
    is intentionally empty because PLS-DA-raw doesn't use engineered features.
    """
    import os
    import sys
    import tempfile
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, "/root")
    os.environ["ATLAS_ARTIFACTS_DIR"] = "/root/artifacts"

    from atlas.inference import _parse_inference_file, predict_from_array_plsda
    from atlas.io import CANONICAL_WN

    contents = await file.read()
    suffix = ".xls"
    if file.filename and file.filename.lower().endswith(".txt"):
        suffix = ".txt"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    rec = _parse_inference_file(tmp_path)
    if not rec.is_valid or rec.intensities is None:
        return JSONResponse(
            {"error": f"failed to parse file: {rec.fatal_errors}"},
            status_code=400,
            headers={"Access-Control-Allow-Origin": "*"},
        )

    X = rec.intensities.astype(np.float32)
    n_pix_raw = int(X.shape[0])

    if n_pix_raw > PIXEL_CAP:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(n_pix_raw, size=PIXEL_CAP, replace=False)
        idx.sort()
        X = X[idx]

    result = predict_from_array_plsda(X, CANONICAL_WN, preprocess=True)

    probs = result["probabilities"]
    sorted_probs = sorted(probs.items(), key=lambda kv: -kv[1])
    top_class = result["class"]
    top_prob = float(probs.get(top_class, 0.0))

    payload = {
        "class": result["class"],
        "probabilities": probs,
        "spectrum_mean": result["spectrum_mean"].tolist(),
        "wn": result["wn"].tolist(),
        "feature_values": result["feature_values"],
        "abstain": top_prob < ABSTAIN_THRESHOLD,
        "top2": [
            {"class": k, "prob": float(v)} for k, v in sorted_probs[:2]
        ],
        "n_pixels_input": n_pix_raw,
        "n_pixels_used": int(X.shape[0]),
        "model": "plsda_raw",
        # PLS-DA interpretability surfaces. Each list is length 987 (matches `wn`).
        "loadings_per_class": result["loadings_per_class"],
        "contribution_for_predicted": result["contribution_for_predicted"],
    }
    return JSONResponse(
        payload,
        headers={"Access-Control-Allow-Origin": "*"},
    )


@app.function()
@modal.fastapi_endpoint(method="OPTIONS")
def predict_plsda_preflight():
    """CORS preflight for POST /predict_plsda."""
    return JSONResponse(
        {"ok": True},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "86400",
        },
    )


@app.function()
@modal.fastapi_endpoint(method="OPTIONS")
def predict_preflight():
    """CORS preflight for POST /predict."""
    return JSONResponse(
        {"ok": True},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "86400",
        },
    )


@app.function()
@modal.fastapi_endpoint(method="GET")
def healthz():
    """GET /healthz -- liveness probe."""
    return JSONResponse(
        {"ok": True, "service": "atlas-inference"},
        headers={"Access-Control-Allow-Origin": "*"},
    )


# Silence unused-import warnings — the imports are needed for type hints
# even if CORS middleware isn't directly attached (Modal's fastapi_endpoint
# constructs the FastAPI app per-decorator, so middleware is per-response).
_ = (CORSMiddleware, CORS_ORIGIN_REGEX)
