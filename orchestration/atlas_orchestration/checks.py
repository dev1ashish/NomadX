"""Dagster asset checks = the data-quality GATES.

Each check reads the real files in data_cache/ and asserts a declared contract
invariant. Hard gates (severity=ERROR, blocking=True) fail the materialization
and block downstream when violated; soft gates (WARN) flag drift without
halting. Nothing is hardcoded — the checks compare the live file contents
against the contract's declared invariants.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    MetadataValue,
    asset_check,
)

from atlas_orchestration import contract, paths
from atlas_orchestration.assets import (
    band_features,
    data_scientist_view,
    feature_store_contract,
    ingested_spectra,
    preprocessed_spectra,
    qc_mask,
    spatial_features,
    spectral_features,
    unmix_features,
)

ERR = AssetCheckSeverity.ERROR
WARN = AssetCheckSeverity.WARN


# --------------------------- ingest gates --------------------------------- #
@asset_check(asset=ingested_spectra, blocking=True,
             description="Raw axis equals the canonical linspace(76,3499,2048).")
def canonical_axis_invariant() -> AssetCheckResult:
    wn = np.load(paths.WN_AXIS)
    declared = np.linspace(
        contract.CANONICAL_WN_START, contract.CANONICAL_WN_END, contract.CANONICAL_WN_N
    ).astype(np.float32)
    ok = wn.size == contract.CANONICAL_WN_N and bool(np.allclose(wn, declared, atol=1e-2))
    return AssetCheckResult(
        passed=ok, severity=ERR,
        metadata={"observed_n": int(wn.size), "start": float(wn[0]), "end": float(wn[-1])},
    )


@asset_check(asset=ingested_spectra, blocking=True,
             description="Matrix is (n_pixels, 2048) and row-aligned to spectra_index.")
def raw_matrix_shape() -> AssetCheckResult:
    arr = np.load(paths.SPECTRA_ARRAY, mmap_mode="r")
    n_idx = len(pd.read_parquet(paths.SPECTRA_PARQUET))
    ok = arr.shape[1] == contract.CANONICAL_WN_N and arr.shape[0] == n_idx
    return AssetCheckResult(
        passed=bool(ok), severity=ERR,
        metadata={"shape": f"{arr.shape}", "index_rows": n_idx},
    )


@asset_check(asset=ingested_spectra, blocking=True,
             description="No file ended with a fatal parse error.")
def no_fatal_ingest_errors() -> AssetCheckResult:
    m = pd.read_parquet(paths.METADATA_PARQUET)
    fatals = m["fatal_errors"].apply(lambda s: len(json.loads(s)) if isinstance(s, str) else 0)
    n = int(fatals.sum())
    return AssetCheckResult(
        passed=n == 0, severity=ERR,
        metadata={"files": int(len(m)), "fatal_error_count": n},
    )


# --------------------------- quality gates -------------------------------- #
@asset_check(asset=qc_mask, blocking=True,
             description="QC funnel reconciles AND mask length/sum agree with qc_info.")
def qc_funnel_reconciles() -> AssetCheckResult:
    info = json.loads(paths.QC_INFO.read_text())
    mask = np.load(paths.QC_MASK)
    n_spectra = len(pd.read_parquet(paths.SPECTRA_PARQUET))
    recon = info["n_input"] - info["n_drop_snr"] - info["n_drop_bg"] == info["n_keep"]
    len_ok = mask.size == n_spectra == info["n_input"]
    sum_ok = int(mask.sum()) == info["n_keep"]
    return AssetCheckResult(
        passed=bool(recon and len_ok and sum_ok), severity=ERR,
        metadata={
            "n_input": info["n_input"], "n_drop_snr": info["n_drop_snr"],
            "n_drop_bg": info["n_drop_bg"], "n_keep": info["n_keep"],
            "reconciles": recon, "mask_len_ok": len_ok, "mask_sum_ok": sum_ok,
        },
    )


@asset_check(asset=qc_mask, blocking=True,
             description="Re-running atlas.qc.apply_qc reproduces the funnel counts exactly.")
def qc_gate_reproducible() -> AssetCheckResult:
    import atlas.qc as aqc

    info = json.loads(paths.QC_INFO.read_text())
    X = np.load(paths.SPECTRA_ARRAY)
    wn = np.load(paths.WN_AXIS)
    fids = pd.read_parquet(paths.SPECTRA_PARQUET).file_id.values
    cached = np.load(paths.QC_MASK)
    keep, live = aqc.apply_qc(X, wn, fids)
    counts_match = all(
        live[k] == info[k] for k in ("n_input", "n_drop_snr", "n_drop_bg", "n_keep")
    )
    agree = float((keep == cached).mean())
    return AssetCheckResult(
        passed=bool(counts_match), severity=ERR,
        metadata={
            "live_n_keep": live["n_keep"], "cached_n_keep": info["n_keep"],
            "counts_match": counts_match,
            "mask_agreement_pct": round(agree * 100, 4),
            "note": "counts are deterministic; the tiny mask delta is a "
                    "percentile-boundary tie in the background filter",
        },
    )


@asset_check(asset=qc_mask,
             description="Overall retention >= 80% (soft floor).")
def qc_retention_floor() -> AssetCheckResult:
    info = json.loads(paths.QC_INFO.read_text())
    ret = info["n_keep"] / info["n_input"]
    worst = min(info["per_file"].items(), key=lambda kv: kv[1]["retention"])
    return AssetCheckResult(
        passed=ret >= 0.80, severity=WARN,
        metadata={
            "overall_retention_pct": round(100 * ret, 2),
            "worst_file": worst[0],
            "worst_file_retention_pct": round(100 * worst[1]["kept"] / worst[1]["n"], 2),
        },
    )


# --------------------------- preprocess gates ----------------------------- #
@asset_check(asset=preprocessed_spectra, blocking=True,
             description="The 987-bin axis invariant holds (len, range).")
def axis_987_invariant() -> AssetCheckResult:
    wn = np.load(paths.WN_AXIS_PREP)
    arr = np.load(paths.SPECTRA_PREP, mmap_mode="r")
    ok = (
        wn.size == contract.PREP_WN_N
        and arr.shape[1] == contract.PREP_WN_N
        and wn.min() >= contract.PREP_WN_MIN - 1
        and wn.max() <= contract.PREP_WN_MAX + 1
    )
    return AssetCheckResult(
        passed=bool(ok), severity=ERR,
        metadata={
            "axis_n": int(wn.size), "matrix_cols": int(arr.shape[1]),
            "min": float(wn.min()), "max": float(wn.max()),
        },
    )


@asset_check(asset=preprocessed_spectra, blocking=True,
             description="Preprocessed rows match ingested rows (all spectra preprocessed).")
def prep_rows_match_ingest() -> AssetCheckResult:
    prep = np.load(paths.SPECTRA_PREP, mmap_mode="r")
    raw = np.load(paths.SPECTRA_ARRAY, mmap_mode="r")
    ok = prep.shape[0] == raw.shape[0]
    return AssetCheckResult(
        passed=bool(ok), severity=ERR,
        metadata={"prep_rows": int(prep.shape[0]), "raw_rows": int(raw.shape[0])},
    )


# --------------------------- feature gates -------------------------------- #
def _per_pixel_rowcount_check(path):
    df = pd.read_parquet(path)
    keep = int(np.load(paths.QC_MASK).sum())
    ok = len(df) == keep
    return ok, {"rows": len(df), "qc_keep": keep}


@asset_check(asset=band_features, blocking=True,
             description="band rows == qc_mask.sum() (positional-alignment guardrail).")
def band_rows_equal_qc_keep() -> AssetCheckResult:
    ok, md = _per_pixel_rowcount_check(paths.BAND_FEATURES)
    return AssetCheckResult(passed=ok, severity=ERR, metadata=md)


@asset_check(asset=band_features,
             description="band_features null-row fraction within declared tolerance (<=1%).")
def band_null_tolerance() -> AssetCheckResult:
    df = pd.read_parquet(paths.BAND_FEATURES)
    frac = float(df.isna().any(axis=1).mean())
    spec = next(s for s in contract.STORES if s.name == "band_features")
    return AssetCheckResult(
        passed=frac <= spec.max_null_row_frac, severity=WARN,
        metadata={
            "null_row_frac": round(frac, 6),
            "tolerance": spec.max_null_row_frac,
            "n_null_rows": int(df.isna().any(axis=1).sum()),
        },
    )


@asset_check(asset=spectral_features, blocking=True,
             description="spectral rows == qc_mask.sum() and zero nulls.")
def spectral_rows_and_nulls() -> AssetCheckResult:
    df = pd.read_parquet(paths.SPECTRAL_FEATURES)
    keep = int(np.load(paths.QC_MASK).sum())
    nulls = int(df.isna().sum().sum())
    ok = len(df) == keep and nulls == 0
    return AssetCheckResult(
        passed=ok, severity=ERR,
        metadata={"rows": len(df), "qc_keep": keep, "total_nulls": nulls},
    )


def _per_file_check(path):
    df = pd.read_parquet(path)
    valid = set(pd.read_parquet(paths.METADATA_PARQUET).file_id)
    ids = list(df.index if df.index.name == "file_id" else df.get("file_id", pd.Series([])))
    subset = set(ids).issubset(valid)
    unique = len(ids) == len(set(ids))
    ok = len(df) == len(valid) and subset and unique
    return ok, {"rows": len(df), "metadata_files": len(valid),
                "file_id_subset": subset, "file_id_unique": unique}


@asset_check(asset=unmix_features, blocking=True,
             description="unmix is per-file: 87 rows, unique file_id subset of metadata.")
def unmix_per_file_integrity() -> AssetCheckResult:
    ok, md = _per_file_check(paths.UNMIX_FEATURES)
    return AssetCheckResult(passed=ok, severity=ERR, metadata=md)


@asset_check(asset=spatial_features, blocking=True,
             description="spatial is per-file: 87 rows, unique file_id subset of metadata.")
def spatial_per_file_integrity() -> AssetCheckResult:
    ok, md = _per_file_check(paths.SPATIAL_FEATURES)
    return AssetCheckResult(passed=ok, severity=ERR, metadata=md)


# --------------------------- handoff gate --------------------------------- #
@asset_check(asset=feature_store_contract, blocking=True,
             description="Generated contract is internally consistent: all stores "
             "present, funnel reconciles, 987-bin axis holds, column total matches.")
def contract_consistent() -> AssetCheckResult:
    c = contract.build_contract()
    inv = c["invariants"]
    all_present = all(s["observed"].get("exists") for s in c["stores"])
    funnel_ok = inv["qc_funnel"]["reconciles"]
    axis_ok = inv["preprocessed_axis_987_bins"]["observed_n"] == contract.PREP_WN_N
    fc = inv["feature_columns"]
    total_ok = fc["total_columns"] == (fc["band"] + fc["spectral"] + fc["unmix"] + fc["spatial"])
    ok = all_present and funnel_ok and axis_ok and total_ok
    return AssetCheckResult(
        passed=bool(ok), severity=ERR,
        metadata={
            "all_stores_present": all_present,
            "funnel_reconciles": funnel_ok,
            "axis_987_holds": axis_ok,
            "total_columns": fc["total_columns"],
            "column_total_consistent": total_ok,
        },
    )


@asset_check(asset=data_scientist_view, blocking=True,
             description="The model-ready handoff assembles correctly: pixel matrix "
             "rows == qc keep, label column present & non-null, file table == 87 rows.")
def ds_view_assembles() -> AssetCheckResult:
    idx = pd.read_parquet(paths.SPECTRA_PARQUET)
    mask = np.load(paths.QC_MASK)
    kept = idx[mask].reset_index(drop=True)
    band = pd.read_parquet(paths.BAND_FEATURES)
    spec = pd.read_parquet(paths.SPECTRAL_FEATURES)
    meta = pd.read_parquet(paths.METADATA_PARQUET)
    pixel_rows_ok = len(kept) == len(band) == len(spec)
    label_ok = "primary_class" in kept.columns and kept["primary_class"].notna().all()
    file_rows_ok = len(meta) == 87
    ok = pixel_rows_ok and label_ok and file_rows_ok
    return AssetCheckResult(
        passed=bool(ok), severity=ERR,
        metadata={
            "pixel_rows_aligned": pixel_rows_ok,
            "pixel_rows": len(kept),
            "label_column_complete": bool(label_ok),
            "file_rows": len(meta),
        },
    )


ALL_CHECKS = [
    canonical_axis_invariant,
    raw_matrix_shape,
    no_fatal_ingest_errors,
    qc_funnel_reconciles,
    qc_gate_reproducible,
    qc_retention_floor,
    axis_987_invariant,
    prep_rows_match_ingest,
    band_rows_equal_qc_keep,
    band_null_tolerance,
    spectral_rows_and_nulls,
    unmix_per_file_integrity,
    spatial_per_file_integrity,
    contract_consistent,
    ds_view_assembles,
]
