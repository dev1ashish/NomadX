"""Build `ui/public/data/mcr_components.json` from the Stage 15F MCR-ALS fit.

Sources
-------
- `artifacts/stage15f_mcr_global.joblib` — atlas.unmix_features.MCRALSWrapper
  (custom class; NOT sklearn). Exposes `.pure_spectra` (K, B) and
  `.transform(X) -> (N, K)`. Fitted on QC-passed preprocessed spectra
  (X[qc_mask] - X[qc_mask].min()) per `atlas/inference.py:114`.
- `data_cache/spectra_array_preprocessed.npy` (7999, 987)
- `data_cache/wavenumber_axis_preprocessed.npy` (987,)
- `data_cache/qc_mask.npy` (7999,)
- `data_cache/spectra.parquet` — pixel-level `file_id`, `primary_class`.
- `data_cache/metadata.parquet` — file-level mapping (cross-check only).

Output
------
`ui/public/data/mcr_components.json` (≪500 KB):

```
{
  "wn": [...987...],
  "components": [
    { "k": 0, "label": "C1 — ...", "spectrum": [...987...],
      "global_d_stec_nonstec": ... },
    ...
  ],
  "per_class_mean_C": { "STEC": [c1..c7], ... },
  "examples": [
    { "primary_class": "STEC", "file_id": "...",
      "observed_mean": [...987...],
      "component_weights": [w1..w7] }, ...
  ]
}
```

Run:
    cd ui && python scripts/build_mcr.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---- Paths ----
HERE = Path(__file__).resolve()
UI_DIR = HERE.parent.parent          # NomadX/ui/
REPO = UI_DIR.parent                 # NomadX/
ART = REPO / "artifacts"
CACHE = REPO / "data_cache"
OUT = UI_DIR / "public" / "data" / "mcr_components.json"

# atlas package import (MCRALSWrapper lives in atlas.unmix_features)
sys.path.insert(0, str(REPO))

CLASS_ORDER = ["STEC", "Non-STEC", "Salmonella", "H2O"]
ROUND = 4


def _round(x: float) -> float:
    return float(round(float(x), ROUND))


def _round_arr(a: np.ndarray) -> list[float]:
    return [_round(v) for v in np.asarray(a, dtype=float).ravel().tolist()]


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d. Sign: positive when mean(a) > mean(b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return 0.0
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    pooled = np.sqrt((va + vb) / 2.0)
    if not np.isfinite(pooled) or pooled <= 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def _component_label(k: int, d_value: float, is_top_abs: bool) -> str:
    """Per-component label.

    The saved global joblib was refit at K=7 — its component ordering does
    NOT match the K=8 fit cached in `unmix_features.parquet` that produced
    the paper's `mcr_C6_mean` d=-1.23 headline. We therefore label each
    component with its own (saved-fit) d and flag the largest-|d| component
    as the saved-fit STEC↔Non-STEC peak.
    """
    one_indexed = k + 1
    if is_top_abs:
        return (
            f"C{one_indexed} — top |d| in saved K=7 fit "
            f"(d={d_value:+.2f} STEC vs Non-STEC)"
        )
    return f"C{one_indexed} — pure component"


def main() -> None:
    # ---- Load artifacts ----
    print(f"[build_mcr] Loading MCR wrapper from {ART / 'stage15f_mcr_global.joblib'}")
    mcr = joblib.load(ART / "stage15f_mcr_global.joblib")
    pure = np.asarray(mcr.pure_spectra, dtype=np.float64)   # (K, B)
    K, B = pure.shape
    print(f"[build_mcr]  pure_spectra shape: ({K}, {B})")
    if K != 7:
        raise RuntimeError(f"Expected K=7 components, got K={K}")

    X = np.load(CACHE / "spectra_array_preprocessed.npy")    # (7999, B)
    wn = np.load(CACHE / "wavenumber_axis_preprocessed.npy") # (B,)
    qc = np.load(CACHE / "qc_mask.npy")                      # (7999,)
    specs = pd.read_parquet(CACHE / "spectra.parquet")       # (7999, ...)
    if X.shape != (specs.shape[0], B):
        raise RuntimeError(
            f"shape mismatch X={X.shape} vs specs={specs.shape} / B={B}"
        )
    if X.shape[0] != qc.shape[0]:
        raise RuntimeError(f"qc mask {qc.shape} doesn't align with X {X.shape}")

    # ---- QC subset (matches the original fit on 7122 pixels) ----
    X_qc = X[qc]
    specs_qc = specs.loc[qc].reset_index(drop=True)
    if X_qc.shape[0] != mcr.concentrations.shape[0]:
        # Soft warning — re-transform may give slightly different C than the
        # fit-time one, but we still proceed.
        print(
            f"[build_mcr] WARN: X_qc rows {X_qc.shape[0]} != "
            f"mcr.concentrations rows {mcr.concentrations.shape[0]}"
        )

    # The transform contract requires shifted (non-negative) input
    # (see atlas/inference.py:113 — `X_offset = X_pp - X_pp.min()`).
    # For the global fit cache, the offset was computed over QC-passed pixels.
    X_offset = X_qc - X_qc.min()
    print(f"[build_mcr] Calling mcr.transform on X_offset shape {X_offset.shape}")
    C_all = mcr.transform(X_offset)                          # (N_qc, K)
    print(f"[build_mcr]  concentrations shape: {C_all.shape}")

    # ---- Per-class mean concentrations (file-weighted) ----
    # First aggregate to per-file C (mean of pixels), then mean per class.
    # This matches `mcr_C{k}_mean` in `mcr_concentration_summary`.
    df_pix = specs_qc[["file_id", "primary_class"]].copy()
    for k in range(K):
        df_pix[f"C{k + 1}"] = C_all[:, k]
    per_file = df_pix.groupby("file_id", as_index=False).agg(
        {**{f"C{k + 1}": "mean" for k in range(K)}, "primary_class": "first"}
    )

    per_class_mean_C: dict[str, list[float]] = {}
    for cls in CLASS_ORDER:
        sub = per_file[per_file["primary_class"] == cls]
        means = [_round(sub[f"C{k + 1}"].mean()) for k in range(K)]
        per_class_mean_C[cls] = means

    # ---- Cohen's d per component (STEC vs Non-STEC, file-level) ----
    stec_files = per_file[per_file["primary_class"] == "STEC"]
    nonstec_files = per_file[per_file["primary_class"] == "Non-STEC"]
    d_per_k: list[float] = []
    for k in range(K):
        col = f"C{k + 1}"
        d_per_k.append(
            _cohens_d(stec_files[col].values, nonstec_files[col].values)
        )

    top_abs_k = int(np.argmax(np.abs(d_per_k)))
    components = []
    for k in range(K):
        components.append(
            {
                "k": k,
                "label": _component_label(k, d_per_k[k], k == top_abs_k),
                "spectrum": _round_arr(pure[k]),
                "global_d_stec_nonstec": _round(d_per_k[k]),
            }
        )

    # ---- Representative file per class (first by file_id) ----
    examples = []
    for cls in CLASS_ORDER:
        sub_pix = specs_qc[specs_qc["primary_class"] == cls]
        if sub_pix.empty:
            continue
        file_id = sorted(sub_pix["file_id"].unique())[0]
        mask = (specs_qc["file_id"] == file_id).values
        observed_mean = X_qc[mask].mean(axis=0)         # (B,)
        # Per-file mean of transformed concentrations (matches `mcr_C*_mean`).
        weights = C_all[mask].mean(axis=0)              # (K,)
        examples.append(
            {
                "primary_class": cls,
                "file_id": file_id,
                "observed_mean": _round_arr(observed_mean),
                "component_weights": _round_arr(weights),
            }
        )

    payload = {
        "wn": _round_arr(wn),
        "components": components,
        "per_class_mean_C": per_class_mean_C,
        "examples": examples,
        "meta": {
            "saved_K": K,
            "paper_K": 8,
            "paper_headline_feature": "mcr_C6_mean",
            "paper_headline_d_stec_nonstec": -1.23,
            "saved_top_abs_k": top_abs_k,
            "note": (
                "The saved global joblib was refit at K=7 — its component "
                "ordering does NOT match the K=8 fit cached in "
                "data_cache/unmix_features.parquet that drives the paper's "
                "mcr_C6_mean d=-1.23 headline. Plan/15 §6.7: 0 MCR features "
                "survived per-fold MI selection in Stage 15F."
            ),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    size_kb = OUT.stat().st_size / 1024
    print(
        f"[build_mcr] wrote {OUT}  ({size_kb:.1f} KB, "
        f"{len(components)} components, {len(examples)} examples)"
    )
    if size_kb > 500:
        print(f"[build_mcr] WARN: sidecar exceeds 500 KB target ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
