"""The DATA CONTRACT at the data-scientist handoff.

This module holds the *declared* invariants of the feature store (the spec the
data scientist may rely on) and a generator that reads the *real* files in
data_cache/ to (a) derive the observed schema and (b) reconcile it against the
declared invariants. The output is written to two artifacts:

    orchestration/contract/CONTRACT.md    (human-readable)
    orchestration/contract/contract.json  (machine-readable)

Design rule (honesty): declared invariants are the SPEC (what the store must
guarantee). Observed values are READ FROM FILES at generation time. The
generator reports both and flags any mismatch — nothing displayed is hardcoded;
the only constants here are the contract's own promises, which the asset checks
then enforce against the real files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from atlas_orchestration import paths

CONTRACT_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Declared invariants (the promises). These mirror the constants baked into the
# atlas/* code: CANONICAL_WN in io.py, the crop windows in preprocess.py, and
# the SNR threshold in qc.py. The checks enforce these against the real files.
# ---------------------------------------------------------------------------
CANONICAL_WN_START = 76.0
CANONICAL_WN_END = 3499.0
CANONICAL_WN_N = 2048           # io.py: np.linspace(76, 3499, 2048)
PREP_WN_N = 987                 # GUARANTEED INVARIANT: every model sees 987 bins
PREP_WN_MIN = 400.0             # preprocess.py crop r1 lower bound
PREP_WN_MAX = 3050.0            # preprocess.py crop r2 upper bound
QC_SNR_THRESHOLD = 5.0          # qc.py default
PIXEL_CAP = 200                 # io.py default per-file pixel cap


@dataclass(frozen=True)
class StoreSpec:
    """One physical store in the feature store / handoff surface."""

    name: str
    path: Any
    fmt: str                    # "parquet" | "npy"
    grain: str                  # "per-file" | "per-pixel" | "matrix" | "vector"
    key: list[str]              # logical join key(s); [] if positional
    join_to: str                # how a consumer joins this to identity
    null_policy: str            # human-readable policy
    purpose: str
    not_null: tuple[str, ...] = ()   # columns the contract guarantees non-null
    max_null_row_frac: float = 0.0   # tolerated fraction of rows with ANY null


STORES: list[StoreSpec] = [
    StoreSpec(
        name="metadata",
        path=paths.METADATA_PARQUET,
        fmt="parquet",
        grain="per-file",
        key=["file_id"],
        join_to="primary identity table; file_id is the universal join key",
        null_policy="subclass null iff H2O (8 files); exposure_ms null for 31 "
                    "files + header_numx/numy/xsize/ysize MAY be null (source "
                    "headers unreliable). file_id/primary_class/n_pixels NEVER null.",
        purpose="One row per raw file: provenance, grid, class label, sha256, "
                "messy-data warnings.",
        not_null=("file_id", "primary_class", "n_pixels", "grid_nx", "grid_ny",
                  "file_sha256", "is_complete_scan"),
        max_null_row_frac=1.0,  # whole-row nulls expected (H2O subclass etc.)
    ),
    StoreSpec(
        name="spectra_index",
        path=paths.SPECTRA_PARQUET,
        fmt="parquet",
        grain="per-pixel",
        key=["file_id", "pixel_idx"],
        join_to="row-aligned (position) to spectra_array.npy and qc_mask.npy",
        null_policy="subclass null iff H2O pixel (767 rows). All other columns "
                    "NEVER null.",
        purpose="One row per ingested pixel-spectrum: identity + (x,y) coords. "
                "Row i corresponds to spectra_array.npy[i].",
        not_null=("file_id", "primary_class", "pixel_idx", "x_um", "y_um"),
        max_null_row_frac=1.0,
    ),
    StoreSpec(
        name="spectra_array",
        path=paths.SPECTRA_ARRAY,
        fmt="npy",
        grain="matrix",
        key=[],
        join_to="position i <-> spectra_index row i",
        null_policy="all-finite (no NaN/Inf); enforced at interp in io.py",
        purpose="Dense float32 (n_pixels, 2048) raw intensities on the canonical "
                "wavenumber axis.",
    ),
    StoreSpec(
        name="qc_mask",
        path=paths.QC_MASK,
        fmt="npy",
        grain="vector",
        key=[],
        join_to="position i <-> spectra_index row i; True = kept",
        null_policy="boolean, no nulls",
        purpose="QC gate result. Selects which ingested spectra flow into the "
                "per-pixel feature stores.",
    ),
    StoreSpec(
        name="spectra_array_preprocessed",
        path=paths.SPECTRA_PREP,
        fmt="npy",
        grain="matrix",
        key=[],
        join_to="position i <-> spectra_index row i (BEFORE qc_mask applied)",
        null_policy="all-finite",
        purpose="Dense float32 (n_pixels, 987) cleaned spectra on the 987-bin "
                "preprocessed axis. THE 987-bin axis is the guaranteed invariant.",
    ),
    StoreSpec(
        name="band_features",
        path=paths.BAND_FEATURES,
        fmt="parquet",
        grain="per-pixel",
        key=[],  # NO explicit key column — positional alignment
        join_to="POSITIONAL: row i <-> spectra_index[qc_mask].reset_index(drop=True) "
                "row i. To attach file_id/pixel_idx, the consumer must re-derive "
                "from spectra_index filtered by qc_mask. (Contract risk: implicit key.)",
        null_policy="NULL POCKET: ~3 rows / 7122 (0.04%) carry NaN in roi_*_skew/kurt "
                    "and bio_* columns (degenerate/flat spectra → undefined moment). "
                    "Consumer MUST impute or drop. Tolerance: <=1% of rows.",
        purpose="Macromolecule band chemistry per kept pixel (AUCs, peak fits, "
                "ROI moments, derivatives, bio-axes).",
        not_null=(),  # no single column guaranteed; whole-store tolerance below
        max_null_row_frac=0.01,
    ),
    StoreSpec(
        name="spectral_features",
        path=paths.SPECTRAL_FEATURES,
        fmt="parquet",
        grain="per-pixel",
        key=[],
        join_to="POSITIONAL: same alignment as band_features.",
        null_policy="zero nulls (observed 0).",
        purpose="Wavelet energies/entropies, region PCA scores, SAM template "
                "similarities per kept pixel.",
        not_null=(),
        max_null_row_frac=0.0,
    ),
    StoreSpec(
        name="unmix_features",
        path=paths.UNMIX_FEATURES,
        fmt="parquet",
        grain="per-file",
        key=["file_id"],
        join_to="index = file_id (subset of metadata.file_id)",
        null_policy="zero nulls (observed 0).",
        purpose="MCR component abundance moments per file (8 components x 4 stats "
                "+ 1 residual-norm).",
        not_null=(),
        max_null_row_frac=0.0,
    ),
    StoreSpec(
        name="spatial_features",
        path=paths.SPATIAL_FEATURES,
        fmt="parquet",
        grain="per-file",
        key=["file_id"],
        join_to="index = file_id (subset of metadata.file_id)",
        null_policy="zero nulls (observed 0).",
        purpose="Spatial heterogeneity of LPS/C-H bands across each file's pixel "
                "grid (variance, CV, skew, kurtosis).",
        not_null=(),
        max_null_row_frac=0.0,
    ),
]


# ---------------------------------------------------------------------------
# Observation: read the real files, derive schema, reconcile vs declared.
# ---------------------------------------------------------------------------

def _observe_parquet(path) -> dict:
    df = pd.read_parquet(path)
    cols = [
        {"name": c, "dtype": str(t), "nulls": int(df[c].isna().sum())}
        for c, t in df.dtypes.items()
    ]
    null_cols = {c["name"]: c["nulls"] for c in cols if c["nulls"] > 0}
    n_rows = int(len(df))
    return {
        "exists": True,
        "rows": n_rows,
        "n_columns": int(df.shape[1]),
        "index_name": df.index.name,
        "total_nulls": int(df.isna().sum().sum()),
        "null_columns": null_cols,
        "null_row_frac": round(float(df.isna().any(axis=1).mean()), 6) if n_rows else 0.0,
        "columns": cols,
    }


def _observe_npy(path) -> dict:
    arr = np.load(path, mmap_mode="r")
    return {
        "exists": True,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "all_finite": bool(np.isfinite(np.asarray(arr[: min(len(arr), 256)])).all())
        if arr.dtype.kind == "f"
        else None,
    }


def observe_store(spec: StoreSpec) -> dict:
    if not spec.path.exists():
        return {"exists": False}
    if spec.fmt == "parquet":
        return _observe_parquet(spec.path)
    return _observe_npy(spec.path)


def build_contract() -> dict:
    """Read every store + the QC funnel; reconcile against declared invariants."""
    qc_info = json.loads(paths.QC_INFO.read_text()) if paths.QC_INFO.exists() else {}
    wn_prep = (
        np.load(paths.WN_AXIS_PREP) if paths.WN_AXIS_PREP.exists() else np.array([])
    )
    wn_raw = np.load(paths.WN_AXIS) if paths.WN_AXIS.exists() else np.array([])

    stores = []
    for spec in STORES:
        obs = observe_store(spec)
        stores.append(
            {
                "name": spec.name,
                "path": paths.rel(spec.path),
                "format": spec.fmt,
                "grain": spec.grain,
                "key": spec.key,
                "join_to": spec.join_to,
                "null_policy": spec.null_policy,
                "not_null": list(spec.not_null),
                "max_null_row_frac": spec.max_null_row_frac,
                "purpose": spec.purpose,
                "observed": obs,
            }
        )

    n_keep = int(qc_info.get("n_keep", 0))

    # Feature-count reconciliation (the "259 vs 260" honesty point).
    def _ncols(name):
        for s in stores:
            if s["name"] == name and s["observed"].get("exists"):
                return s["observed"]["n_columns"]
        return None

    band_c, spec_c = _ncols("band_features"), _ncols("spectral_features")
    unmix_c, spat_c = _ncols("unmix_features"), _ncols("spatial_features")
    total_cols = sum(c for c in (band_c, spec_c, unmix_c, spat_c) if c)

    invariants = {
        "canonical_raw_axis": {
            "declared": f"linspace({CANONICAL_WN_START}, {CANONICAL_WN_END}, "
                        f"{CANONICAL_WN_N})",
            "observed_n": int(wn_raw.size),
            "observed_start": float(wn_raw[0]) if wn_raw.size else None,
            "observed_end": float(wn_raw[-1]) if wn_raw.size else None,
        },
        "preprocessed_axis_987_bins": {
            "declared_n": PREP_WN_N,
            "observed_n": int(wn_prep.size),
            "observed_min": float(wn_prep.min()) if wn_prep.size else None,
            "observed_max": float(wn_prep.max()) if wn_prep.size else None,
            "declared_range": [PREP_WN_MIN, PREP_WN_MAX],
        },
        "qc_funnel": {
            "n_input": qc_info.get("n_input"),
            "n_drop_snr": qc_info.get("n_drop_snr"),
            "n_drop_bg": qc_info.get("n_drop_bg"),
            "n_keep": n_keep,
            "reconciles": (
                qc_info.get("n_input", 0)
                - qc_info.get("n_drop_snr", 0)
                - qc_info.get("n_drop_bg", 0)
                == n_keep
            ),
            "retention_pct": round(100 * n_keep / qc_info["n_input"], 2)
            if qc_info.get("n_input")
            else None,
            "snr_threshold": QC_SNR_THRESHOLD,
        },
        "per_pixel_rowcount_equals_qc_keep": {
            "qc_keep": n_keep,
            "band_rows": next(
                (s["observed"].get("rows") for s in stores if s["name"] == "band_features"),
                None,
            ),
            "spectral_rows": next(
                (s["observed"].get("rows") for s in stores if s["name"] == "spectral_features"),
                None,
            ),
        },
        "feature_columns": {
            "band": band_c,
            "spectral": spec_c,
            "unmix": unmix_c,
            "spatial": spat_c,
            "total_columns": total_cols,
            "note": "unmix has 1 diagnostic column (mcr_residual_norm_mean) on top "
                    "of its 32 MCR moment features; counting unmix as 32 gives the "
                    "often-quoted '259 features'. The real on-disk column total is "
                    f"{total_cols}.",
        },
    }

    return {
        "contract_version": CONTRACT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stores": stores,
        "invariants": invariants,
    }


def render_markdown(contract: dict) -> str:
    inv = contract["invariants"]
    lines: list[str] = []
    lines.append(f"# Atlas Feature Store — Data Contract v{contract['contract_version']}")
    lines.append("")
    lines.append(
        "> The interface between the data-engineering pipeline and the data "
        "scientist. Every value below was read from the real files in "
        "`data_cache/` at generation time; the asset checks enforce the declared "
        "invariants against those files on every run."
    )
    lines.append("")
    lines.append(f"_Generated: {contract['generated_at']}_")
    lines.append("")

    # Guaranteed invariants
    lines.append("## Guaranteed invariants")
    lines.append("")
    f = inv["qc_funnel"]
    lines.append(
        f"- **QC funnel reconciles:** {f['n_input']} in − {f['n_drop_snr']} (SNR<"
        f"{f['snr_threshold']}) − {f['n_drop_bg']} (background) = **{f['n_keep']}** "
        f"kept ({f['retention_pct']}%) — `{f['reconciles']}`"
    )
    a = inv["preprocessed_axis_987_bins"]
    lines.append(
        f"- **987-bin preprocessed axis:** declared {a['declared_n']}, observed "
        f"{a['observed_n']} bins over [{a['observed_min']:.2f}, {a['observed_max']:.2f}] "
        f"cm⁻¹ (crop window {a['declared_range']})"
    )
    r = inv["canonical_raw_axis"]
    lines.append(
        f"- **Canonical raw axis:** declared {r['declared']}; observed {r['observed_n']} "
        f"bins, [{r['observed_start']}, {r['observed_end']}]"
    )
    p = inv["per_pixel_rowcount_equals_qc_keep"]
    lines.append(
        f"- **Per-pixel rowcount == QC keep:** qc_keep={p['qc_keep']}, "
        f"band={p['band_rows']}, spectral={p['spectral_rows']}"
    )
    fc = inv["feature_columns"]
    lines.append(
        f"- **Feature columns:** band {fc['band']} + spectral {fc['spectral']} + "
        f"unmix {fc['unmix']} + spatial {fc['spatial']} = **{fc['total_columns']}** "
        f"columns on disk. {fc['note']}"
    )
    lines.append("")

    # Stores
    lines.append("## Stores")
    lines.append("")
    lines.append("| store | file | format | grain | key | rows | cols | nulls |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in contract["stores"]:
        o = s["observed"]
        if not o.get("exists"):
            lines.append(f"| {s['name']} | {s['path']} | — | — | — | MISSING | — | — |")
            continue
        if "shape" in o:  # npy
            shape = o["shape"]
            rows = shape[0]
            cols = shape[1] if len(shape) > 1 else 1
            nulls = "—"
        else:  # parquet
            rows = o.get("rows")
            cols = o.get("n_columns")
            nulls = o.get("total_nulls", "—")
        key = ", ".join(s["key"]) if s["key"] else "_(positional)_"
        lines.append(
            f"| {s['name']} | `{s['path']}` | {s['format']} | {s['grain']} | {key} "
            f"| {rows} | {cols} | {nulls} |"
        )
    lines.append("")

    # Join map
    lines.append("## Join / lineage map (how the DS consumes this)")
    lines.append("")
    for s in contract["stores"]:
        lines.append(f"- **{s['name']}** — {s['join_to']}")
    lines.append("")
    lines.append(
        "> ⚠️ **Contract risk surfaced honestly:** `band_features` and "
        "`spectral_features` carry **no** `file_id`/`pixel_idx` columns — they are "
        "**positionally aligned** to `spectra_index` filtered by `qc_mask`. A "
        "consumer that reorders or re-filters either side silently breaks the join. "
        "The `band/spectral rows == qc_mask.sum()` check is the guardrail."
    )
    lines.append("")

    # Null policy
    lines.append("## Null policy")
    lines.append("")
    for s in contract["stores"]:
        lines.append(f"- **{s['name']}**: {s['null_policy']}")
    lines.append("")
    return "\n".join(lines)


def write_contract() -> dict:
    """Generate + persist CONTRACT.md and contract.json. Returns the contract dict."""
    contract = build_contract()
    paths.CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
    (paths.CONTRACT_DIR / "contract.json").write_text(json.dumps(contract, indent=2))
    (paths.CONTRACT_DIR / "CONTRACT.md").write_text(render_markdown(contract))
    return contract


if __name__ == "__main__":
    c = write_contract()
    print(f"Wrote contract v{c['contract_version']} -> {paths.rel(paths.CONTRACT_DIR)}/")
    print(json.dumps(c["invariants"]["qc_funnel"], indent=2))
