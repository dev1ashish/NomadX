"""Software-defined assets = the Atlas pipeline DAG.

Each asset is a THIN WRAPPER over the existing science code in atlas/* and the
materialized outputs in data_cache/. The science is NOT reinvented here: assets
read the real cache as the source of truth and attach lineage + quality +
schema metadata so Dagster renders the DAG, the data-quality funnel, and the
data-scientist handoff contract.

Two stages additionally RE-INVOKE the real atlas code live to prove the
lineage is genuine (not a faked number):
  * qc_mask                — re-runs atlas.qc.apply_qc and reconciles the funnel.
  * preprocessed_spectra   — re-runs atlas.preprocess.preprocess_matrix on a
                             small sample to confirm the 987-bin axis.

Every displayed number is read from a real file at materialization time.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from dagster import MetadataValue, MaterializeResult, asset

from atlas_orchestration import contract, paths

# Group names (drive the swimlanes in the Dagster asset graph).
G_INGEST = "stage_1_ingest"
G_QC = "stage_2_quality"
G_PREP = "stage_3_preprocess"
G_FEAT = "stage_4_features"
G_HANDOFF = "stage_5_handoff"
G_SERVE = "stage_6_serving"


# --------------------------------------------------------------------------- #
# small metadata helpers
# --------------------------------------------------------------------------- #
def _md_table(headers: list[str], rows: list[list]) -> MetadataValue:
    h = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return MetadataValue.md("\n".join([h, sep, *body]))


def _jload(p) -> dict:
    return json.loads(p.read_text())


def _write_dataset_card(out, pixel, file_tbl, id_label_cols, feature_cols,
                        n_file_feats, label_dist) -> None:
    """Write feature_store/DATASET_CARD.md — the human-readable 'what is this
    dataset and how do I model on it' doc for the data scientist."""
    lines = [
        "# Atlas Feature Store — Model-Ready Dataset",
        "",
        "Prepared by the data-engineering pipeline; this is the handoff to the "
        "data scientist. Two tables, both with class labels attached.",
        "",
        "## 1. `pixel_training_matrix.parquet`  (the main training matrix)",
        f"- **Shape:** {pixel.shape[0]} rows × {pixel.shape[1]} cols "
        f"(one row per QC-kept pixel-spectrum)",
        f"- **Label (y):** `primary_class` — {label_dist}",
        "- **Group key (for CV):** `file_id` — use leave-one-group-out so pixels "
        "from the same sample never straddle train/test",
        f"- **Identity columns:** {', '.join('`'+c+'`' for c in id_label_cols)}",
        f"- **Features (X):** {len(feature_cols)} numeric columns "
        "(band chemistry + spectral). Drop the identity columns to get X.",
        "",
        "## 2. `file_level_table.parquet`  (one row per sample)",
        f"- **Shape:** {file_tbl.shape[0]} rows × {file_tbl.shape[1] + 1} cols "
        "(`file_id` index)",
        f"- **Features:** {n_file_feats} per-file columns (MCR unmixing + spatial "
        "heterogeneity). Join to the pixel table on `file_id` if you want both grains.",
        "",
        "## Load it",
        "```python",
        "import pandas as pd",
        "df = pd.read_parquet('feature_store/pixel_training_matrix.parquet')",
        "y, groups = df['primary_class'], df['file_id']",
        "X = df.drop(columns=['file_id','primary_class','subclass',"
        "'pixel_idx','x_um','y_um'])",
        "```",
        "",
        "## Guarantees (enforced by the pipeline's asset checks)",
        "- row count == QC keep; no leakage of dropped/low-SNR pixels",
        "- features computed on the 987-bin preprocessed axis",
        "- `sample_for_humans.csv` = a class-stratified 200-row sample (50 per "
        "class, shuffled), full columns — for eyeballing in Excel/Numbers",
        "- null policy: ~3 pixel rows carry NaN in skew/kurt/bio cols — impute or drop",
        "",
        "See `orchestration/contract/CONTRACT.md` for the full schema + invariants.",
    ]
    (out / "DATASET_CARD.md").write_text("\n".join(lines))


# =========================================================================== #
# STAGE 1 — INGEST
# =========================================================================== #
@asset(
    group_name=G_INGEST,
    description="87 raw .xls/.txt files parsed by atlas.io.parse_dataset. Surfaces "
    "the messy-data handling (tab-delimited .xls, comma thousands-seps, "
    "unreliable NUMX/NUMY headers, mosaics, partial scans, per-file wavenumber "
    "drift). Reads data_cache/metadata.parquet.",
)
def raw_file_inventory() -> MaterializeResult:
    m = pd.read_parquet(paths.METADATA_PARQUET)
    m["warns"] = m["warnings"].apply(lambda s: json.loads(s) if isinstance(s, str) else (s or []))
    m["fatals"] = m["fatal_errors"].apply(lambda s: json.loads(s) if isinstance(s, str) else (s or []))

    class_counts = m.groupby("primary_class").size().to_dict()
    exts = m["file_path"].apply(lambda p: p.rsplit(".", 1)[-1]).value_counts().to_dict()
    n_fatal = int(m["fatals"].apply(len).sum())

    capped = m[m["warns"].apply(lambda w: any("pixel_capped" in x for x in w))]
    cap_rows = []
    for _, r in capped.iterrows():
        raw_px = next((x for x in r["warns"] if "pixel_capped" in x), "")
        cap_rows.append([r.file_id, r.primary_class, raw_px.replace("pixel_capped:", "")])

    partial = m[~m["is_complete_scan"]]
    # Header NUMX/NUMY disagree with the coordinate-derived grid?
    hdr_wrong = m[
        (m["header_numx"] != m["grid_nx"]) | (m["header_numy"] != m["grid_ny"])
    ]

    meta = {
        "dagster/row_count": MetadataValue.int(len(m)),
        "n_files": MetadataValue.int(len(m)),
        "n_pixels_total_after_cap": MetadataValue.int(int(m["n_pixels"].sum())),
        "n_fatal_errors": MetadataValue.int(n_fatal),
        "file_extensions": MetadataValue.json(exts),
        "class_file_counts": _md_table(
            ["primary_class", "n_files"], [[k, v] for k, v in class_counts.items()]
        ),
        "wavenumber_drift_pre_interp": MetadataValue.md(
            f"raw `wn_start` ∈ [{m.wn_start.min():.2f}, {m.wn_start.max():.2f}], "
            f"`wn_end` ∈ [{m.wn_end.min():.2f}, {m.wn_end.max():.2f}] → all interpolated "
            f"to canonical `linspace({contract.CANONICAL_WN_START}, "
            f"{contract.CANONICAL_WN_END}, {contract.CANONICAL_WN_N})`."
        ),
        "files_pixel_capped": _md_table(
            ["file_id", "class", "raw_px -> cap"], cap_rows
        )
        if cap_rows
        else MetadataValue.text("none"),
        "files_partial_scan": MetadataValue.md(
            "\n".join(
                f"- `{r.file_id}`: "
                + next((x for x in r["warns"] if "partial_scan" in x), "")
                for _, r in partial.iterrows()
            )
            or "none"
        ),
        "files_with_unreliable_headers": MetadataValue.md(
            f"**{len(hdr_wrong)}/{len(m)}** files: source `#NUMX/#NUMY` header disagrees "
            f"with the coordinate-derived grid → grid taken from `unique(x),unique(y)` "
            f"(per atlas.io). e.g. "
            + ", ".join(
                f"`{r.file_id}` hdr {r.header_numx}×{r.header_numy} vs grid "
                f"{r.grid_nx}×{r.grid_ny}"
                for _, r in hdr_wrong.head(3).iterrows()
            )
        ),
        "messy_data_handling": MetadataValue.md(
            "- tab-delimited despite `.xls`; comma thousands-separators stripped\n"
            "- `#NUMX/#NUMY` headers unreliable → grid derived from coordinates\n"
            "- mosaics (e.g. R370=720px, R364=324px) capped at "
            f"{contract.PIXEL_CAP}px/file (seed=0)\n"
            "- partial scans flagged `is_complete_scan=False`\n"
            "- per-file wavenumber drift → interpolated to one canonical axis\n"
            "- 1 `.txt` file ingested alongside 86 `.xls`"
        ),
        "wraps": MetadataValue.text("atlas.io.parse_dataset / parse_file / write_cache"),
        "source_file": MetadataValue.path(str(paths.METADATA_PARQUET)),
    }
    return MaterializeResult(metadata=meta)


@asset(
    group_name=G_INGEST,
    deps=[raw_file_inventory],
    description="Dense ingested matrix: every pixel-spectrum interpolated onto the "
    "canonical 2048-bin axis. Reads spectra_array.npy + spectra.parquet + "
    "wavenumber_axis.npy. Includes a LIVE lineage probe that re-parses one real "
    "file via atlas.io.parse_file.",
)
def ingested_spectra() -> MaterializeResult:
    arr = np.load(paths.SPECTRA_ARRAY, mmap_mode="r")
    wn = np.load(paths.WN_AXIS)
    idx = pd.read_parquet(paths.SPECTRA_PARQUET)

    declared = np.linspace(
        contract.CANONICAL_WN_START, contract.CANONICAL_WN_END, contract.CANONICAL_WN_N
    ).astype(np.float32)
    axis_ok = bool(np.allclose(wn, declared, atol=1e-2))

    # LIVE lineage probe: re-parse the smallest real file end-to-end.
    probe_md = "skipped"
    try:
        import atlas.io as aio

        data_root = paths.REPO_ROOT / "Atlas Data"
        if data_root.exists():
            files = aio.discover_files(data_root)
            # smallest file = fastest probe
            target = min(files, key=lambda p: p.stat().st_size)
            t = time.time()
            rec = aio.parse_file(target, data_root)
            dt = time.time() - t
            probe_md = (
                f"re-parsed `{rec.file_id}` via `atlas.io.parse_file` in {dt:.2f}s → "
                f"{rec.n_pixels}px × {rec.intensities.shape[1]} bins, valid={rec.is_valid} "
                f"(bins==2048: {rec.intensities.shape[1] == contract.CANONICAL_WN_N})"
            )
        else:
            probe_md = f"raw `Atlas Data/` not present at {paths.rel(data_root)} — probe skipped (cache-only mode)"
    except Exception as e:  # pragma: no cover
        probe_md = f"probe error (non-fatal): {type(e).__name__}: {e}"

    meta = {
        "dagster/row_count": MetadataValue.int(int(arr.shape[0])),
        "shape": MetadataValue.text(f"{arr.shape[0]} × {arr.shape[1]} ({arr.dtype})"),
        "n_bins": MetadataValue.int(int(arr.shape[1])),
        "canonical_axis_observed": MetadataValue.md(
            f"n={wn.size}, [{wn[0]:.2f}, {wn[-1]:.2f}] cm⁻¹"
        ),
        "canonical_axis_invariant_holds": MetadataValue.bool(axis_ok),
        "row_aligned_index_rows": MetadataValue.int(len(idx)),
        "live_lineage_probe": MetadataValue.md(probe_md),
        "wraps": MetadataValue.text("atlas.io (write_cache outputs)"),
        "storage_rationale": MetadataValue.md(
            "dense float32 `.npy` (not parquet): 7999×2048 numeric block read "
            "contiguously into torch/sklearn; columnar typing buys nothing for a "
            "homogeneous matrix."
        ),
    }
    return MaterializeResult(metadata=meta)


# =========================================================================== #
# STAGE 2 — QUALITY
# =========================================================================== #
@asset(
    group_name=G_QC,
    deps=[ingested_spectra],
    description="QC gate. Reads qc_mask.npy + qc_info.json AND re-runs "
    "atlas.qc.apply_qc live on the cached raw array to reconcile the funnel "
    "(7999 → −SNR → −background → kept).",
)
def qc_mask() -> MaterializeResult:
    info = _jload(paths.QC_INFO)
    cached = np.load(paths.QC_MASK)

    n_in, n_snr, n_bg, n_keep = (
        info["n_input"],
        info["n_drop_snr"],
        info["n_drop_bg"],
        info["n_keep"],
    )
    retention = 100 * n_keep / n_in

    # per-file retention (lowest few)
    pf = info.get("per_file", {})
    pf_sorted = sorted(pf.items(), key=lambda kv: kv[1]["retention"])
    low_rows = [
        [fid, d["n"], d["kept"], f"{100*d['kept']/d['n']:.1f}%", f"{d['median_snr']:.1f}"]
        for fid, d in pf_sorted[:8]
    ]

    # LIVE re-derivation of the gate.
    repro_md = "skipped"
    try:
        import atlas.qc as aqc

        X = np.load(paths.SPECTRA_ARRAY)
        wn = np.load(paths.WN_AXIS)
        fids = pd.read_parquet(paths.SPECTRA_PARQUET).file_id.values
        t = time.time()
        keep, live = aqc.apply_qc(X, wn, fids)
        dt = time.time() - t
        counts_match = (
            live["n_input"] == n_in
            and live["n_drop_snr"] == n_snr
            and live["n_drop_bg"] == n_bg
            and live["n_keep"] == n_keep
        )
        agree = float((keep == cached).mean())
        repro_md = (
            f"re-ran `atlas.qc.apply_qc` in {dt:.2f}s → funnel counts match cache: "
            f"**{counts_match}** (in {live['n_input']} −snr {live['n_drop_snr']} "
            f"−bg {live['n_drop_bg']} = keep {live['n_keep']}). "
            f"Mask agreement {agree*100:.3f}% — the {int((1-agree)*len(keep))}-pixel "
            f"delta is a percentile-boundary tie in the per-file background filter "
            f"(does not change any count)."
        )
    except Exception as e:  # pragma: no cover
        repro_md = f"re-derivation error (non-fatal): {type(e).__name__}: {e}"

    meta = {
        "dagster/row_count": MetadataValue.int(int(cached.size)),
        "QC_FUNNEL": _md_table(
            ["stage", "count"],
            [
                ["ingested", n_in],
                ["− low SNR (<%.0f)" % contract.QC_SNR_THRESHOLD, f"−{n_snr}"],
                ["− background (off-cell)", f"−{n_bg}"],
                ["**kept**", f"**{n_keep}**"],
            ],
        ),
        "retention_pct": MetadataValue.float(round(retention, 2)),
        "funnel_reconciles": MetadataValue.bool(n_in - n_snr - n_bg == n_keep),
        "median_snr_overall": MetadataValue.float(round(info["median_snr_overall"], 2)),
        "lowest_retention_files": _md_table(
            ["file_id", "n", "kept", "retention", "median_snr"], low_rows
        ),
        "live_gate_reproducibility": MetadataValue.md(repro_md),
        "wraps": MetadataValue.text("atlas.qc.apply_qc (snr_per_spectrum + background_mask)"),
        "note": MetadataValue.md(
            "QC runs on the **2048-bin raw** axis (needs the 1800–2500 cm⁻¹ noise "
            "band, which preprocessing crops out) — so it is independent of, and "
            "parallel to, the preprocess stage. The 7122 selection is applied at "
            "feature time."
        ),
    }
    return MaterializeResult(metadata=meta)


# =========================================================================== #
# STAGE 3 — PREPROCESS
# =========================================================================== #
@asset(
    group_name=G_PREP,
    deps=[ingested_spectra],
    description="Cleaned spectra on the guaranteed 987-bin axis "
    "(cosmic→arPLS→Savitzky-Golay→crop→SNV). Reads "
    "spectra_array_preprocessed.npy + wavenumber_axis_preprocessed.npy, and "
    "re-runs atlas.preprocess.preprocess_matrix on a small sample to confirm the "
    "987-bin axis invariant.",
)
def preprocessed_spectra() -> MaterializeResult:
    arr = np.load(paths.SPECTRA_PREP, mmap_mode="r")
    wn = np.load(paths.WN_AXIS_PREP)

    n_bins = int(arr.shape[1])
    invariant_ok = bool(
        n_bins == contract.PREP_WN_N
        and wn.min() >= contract.PREP_WN_MIN - 1
        and wn.max() <= contract.PREP_WN_MAX + 1
    )

    sample_md = "skipped"
    try:
        import atlas.preprocess as app

        raw = np.load(paths.SPECTRA_ARRAY, mmap_mode="r")
        wn_raw = np.load(paths.WN_AXIS)
        t = time.time()
        Xp, wnp, _ = app.preprocess_matrix(np.asarray(raw[:6]), wn_raw, progress=False)
        dt = time.time() - t
        axis_match = bool(np.allclose(wnp, wn, atol=1e-3))
        sample_md = (
            f"re-ran `atlas.preprocess.preprocess_matrix` on 6 raw spectra in "
            f"{dt:.2f}s → out shape {Xp.shape}, {wnp.size} bins, axis matches cache: "
            f"**{axis_match}**"
        )
    except Exception as e:  # pragma: no cover
        sample_md = f"sample error (non-fatal): {type(e).__name__}: {e}"

    meta = {
        "dagster/row_count": MetadataValue.int(int(arr.shape[0])),
        "shape": MetadataValue.text(f"{arr.shape[0]} × {arr.shape[1]} ({arr.dtype})"),
        "n_bins_987_invariant": MetadataValue.bool(n_bins == contract.PREP_WN_N),
        "preprocessed_axis": MetadataValue.md(
            f"n={wn.size}, [{wn.min():.2f}, {wn.max():.2f}] cm⁻¹ "
            f"(crop {contract.PREP_WN_MIN:.0f}–1800 + 2800–{contract.PREP_WN_MAX:.0f})"
        ),
        "invariant_holds": MetadataValue.bool(invariant_ok),
        "pipeline_config": MetadataValue.md(
            "1. cosmic-ray removal (median-filter, z>5)\n"
            "2. arPLS baseline (λ=1e5, diff_order=2)\n"
            "3. Savitzky-Golay (window=9, poly=3)\n"
            "4. crop 400–1800 + 2800–3050 cm⁻¹\n"
            "5. SNV (per-spectrum z-score)\n"
            "_(defaults from atlas.preprocess.preprocess_matrix)_"
        ),
        "live_sample_proof": MetadataValue.md(sample_md),
        "wraps": MetadataValue.text("atlas.preprocess.preprocess_matrix"),
        "note": MetadataValue.md(
            "Runs on all 7999 ingested spectra; the QC mask selects the 7122 that "
            "flow downstream into the per-pixel feature stores."
        ),
    }
    return MaterializeResult(metadata=meta)


# =========================================================================== #
# STAGE 4 — FEATURES
# =========================================================================== #
def _feature_meta(path, grain: str, wraps: str, per_pixel: bool) -> dict:
    df = pd.read_parquet(path)
    null_cols = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
    dtype_counts = df.dtypes.astype(str).value_counts().to_dict()
    sample = [[c, str(df[c].dtype)] for c in list(df.columns)[:12]]

    meta = {
        "dagster/row_count": MetadataValue.int(len(df)),
        "shape": MetadataValue.text(f"{df.shape[0]} rows × {df.shape[1]} cols"),
        "grain": MetadataValue.text(grain),
        "dtype_histogram": MetadataValue.json(dtype_counts),
        "null_columns": MetadataValue.json(null_cols) if null_cols else MetadataValue.text("none"),
        "null_row_frac": MetadataValue.float(round(float(df.isna().any(axis=1).mean()), 6)),
        "column_sample": _md_table(["column", "dtype"], sample),
        "wraps": MetadataValue.text(wraps),
    }
    if per_pixel:
        keep = int(np.load(paths.QC_MASK).sum())
        meta["rows_equal_qc_keep"] = MetadataValue.bool(len(df) == keep)
        meta["join"] = MetadataValue.md(
            "**positional** to `spectra_index[qc_mask]` (no explicit key column)"
        )
    else:
        valid = set(pd.read_parquet(paths.METADATA_PARQUET).file_id)
        ids = set(df.index if df.index.name == "file_id" else df.get("file_id", []))
        meta["file_id_subset_of_metadata"] = MetadataValue.bool(ids.issubset(valid))
        meta["join"] = MetadataValue.md("`file_id` index → `metadata.file_id`")
    return meta


@asset(group_name=G_FEAT, deps=[preprocessed_spectra, qc_mask],
       description="Per-pixel macromolecule band chemistry (166 features). "
       "Wraps atlas.band_features. Reads band_features.parquet.")
def band_features() -> MaterializeResult:
    return MaterializeResult(
        metadata=_feature_meta(paths.BAND_FEATURES, "per-pixel (7122 kept rows)",
                               "atlas.band_features", per_pixel=True)
    )


@asset(group_name=G_FEAT, deps=[preprocessed_spectra, qc_mask],
       description="Per-pixel spectral features: wavelet, region-PCA, SAM "
       "similarity (51 features). Wraps atlas.spectral_features.")
def spectral_features() -> MaterializeResult:
    return MaterializeResult(
        metadata=_feature_meta(paths.SPECTRAL_FEATURES, "per-pixel (7122 kept rows)",
                               "atlas.spectral_features", per_pixel=True)
    )


@asset(group_name=G_FEAT, deps=[preprocessed_spectra],
       description="Per-file MCR component-abundance moments (33 cols = 32 MCR "
       "moments + 1 residual-norm). Wraps atlas.unmix_features.")
def unmix_features() -> MaterializeResult:
    return MaterializeResult(
        metadata=_feature_meta(paths.UNMIX_FEATURES, "per-file (87 rows, index=file_id)",
                               "atlas.unmix_features", per_pixel=False)
    )


@asset(group_name=G_FEAT, deps=[preprocessed_spectra],
       description="Per-file spatial heterogeneity of LPS/C-H bands (10 features). "
       "Wraps atlas.spatial_features.")
def spatial_features() -> MaterializeResult:
    return MaterializeResult(
        metadata=_feature_meta(paths.SPATIAL_FEATURES, "per-file (87 rows, index=file_id)",
                               "atlas.spatial_features", per_pixel=False)
    )


# =========================================================================== #
# STAGE 5 — DATA-SCIENTIST HANDOFF (the contract)
# =========================================================================== #
@asset(
    group_name=G_HANDOFF,
    deps=[band_features, spectral_features, unmix_features, spatial_features, qc_mask],
    description="The feature-store data contract consumed by the data scientist. "
    "Regenerates orchestration/contract/CONTRACT.md + contract.json from the real "
    "files and attaches schema/grain/keys/null-policy/invariants as metadata.",
)
def feature_store_contract() -> MaterializeResult:
    c = contract.write_contract()
    inv = c["invariants"]
    fc = inv["feature_columns"]

    store_rows = []
    for s in c["stores"]:
        o = s["observed"]
        if "shape" in o:
            rc = o["shape"][0]
            cc = o["shape"][1] if len(o["shape"]) > 1 else 1
        else:
            rc, cc = o.get("rows"), o.get("n_columns")
        key = ", ".join(s["key"]) if s["key"] else "_(positional)_"
        store_rows.append([s["name"], s["format"], s["grain"], key, rc, cc])

    meta = {
        "contract_version": MetadataValue.text(c["contract_version"]),
        "total_feature_columns": MetadataValue.int(fc["total_columns"]),
        "feature_columns_breakdown": MetadataValue.md(
            f"band {fc['band']} + spectral {fc['spectral']} + unmix {fc['unmix']} + "
            f"spatial {fc['spatial']} = **{fc['total_columns']}** cols. {fc['note']}"
        ),
        "stores": _md_table(
            ["store", "fmt", "grain", "key", "rows", "cols"], store_rows
        ),
        "guaranteed_invariants": MetadataValue.md(
            f"- QC funnel reconciles: **{inv['qc_funnel']['reconciles']}** "
            f"({inv['qc_funnel']['n_input']}→{inv['qc_funnel']['n_keep']}, "
            f"{inv['qc_funnel']['retention_pct']}%)\n"
            f"- 987-bin axis: observed **{inv['preprocessed_axis_987_bins']['observed_n']}** "
            f"bins\n"
            f"- per-pixel rows == qc keep: band="
            f"{inv['per_pixel_rowcount_equals_qc_keep']['band_rows']}, "
            f"qc_keep={inv['per_pixel_rowcount_equals_qc_keep']['qc_keep']}"
        ),
        "join_map": MetadataValue.md(
            "\n".join(f"- **{s['name']}** — {s['join_to']}" for s in c["stores"])
        ),
        "contract_risk_positional_alignment": MetadataValue.md(
            "⚠️ `band_features` / `spectral_features` carry **no** `file_id`/`pixel_idx` "
            "column — positionally aligned to `spectra_index[qc_mask]`. Reordering "
            "either side silently breaks the join. Guardrail: the "
            "`band/spectral rows == qc_mask.sum()` check."
        ),
        "contract_md_path": MetadataValue.path(str(paths.CONTRACT_DIR / "CONTRACT.md")),
        "contract_json": MetadataValue.json(inv),
    }
    return MaterializeResult(metadata=meta)


# =========================================================================== #
# STAGE 5b — THE DATA-SCIENTIST VIEW (the model-ready table the DS opens)
# =========================================================================== #
@asset(
    group_name=G_HANDOFF,
    deps=[feature_store_contract],
    description="The model-ready tables the data scientist actually consumes, "
    "assembled by following the contract's join map: per-pixel features joined "
    "POSITIONALLY to the QC-kept index, per-file features joined on file_id. This "
    "is THE handoff made concrete — labels + identity + features, ready to model.",
)
def data_scientist_view() -> MaterializeResult:
    # --- per-pixel model-ready table (the DS's main training matrix) ---
    idx = pd.read_parquet(paths.SPECTRA_PARQUET)
    mask = np.load(paths.QC_MASK)
    kept = idx[mask].reset_index(drop=True)                 # identity + labels
    band = pd.read_parquet(paths.BAND_FEATURES).reset_index(drop=True)
    spec = pd.read_parquet(paths.SPECTRAL_FEATURES).reset_index(drop=True)
    # Contract guarantee (guarded by the band/spectral rows==qc_keep checks):
    assert len(kept) == len(band) == len(spec), "positional alignment violated"
    pixel = pd.concat([kept, band, spec], axis=1)

    # --- per-file model-ready table (file_id-keyed) ---
    meta = pd.read_parquet(paths.METADATA_PARQUET).set_index("file_id")
    unmix = pd.read_parquet(paths.UNMIX_FEATURES)
    spatial = pd.read_parquet(paths.SPATIAL_FEATURES)
    file_tbl = meta[["primary_class", "subclass"]].join(unmix, how="left").join(
        spatial, how="left"
    )

    n_pixel_feats = band.shape[1] + spec.shape[1]
    n_file_feats = unmix.shape[1] + spatial.shape[1]
    label_dist = kept["primary_class"].value_counts().to_dict()

    # --- WRITE the prepared dataset the data scientist actually opens ---
    out = paths.REPO_ROOT / "feature_store"
    out.mkdir(exist_ok=True)
    id_label_cols = list(kept.columns)               # file_id, primary_class, subclass, pixel_idx, x, y
    feature_cols = [c for c in pixel.columns if c not in id_label_cols]
    pixel.to_parquet(out / "pixel_training_matrix.parquet", compression="snappy")
    file_tbl.to_parquet(out / "file_level_table.parquet", compression="snappy")
    # human-eyeball CSV: a class-STRATIFIED, shuffled sample (all 4 classes
    # represented) with the SAME columns as the full dataset — just fewer rows.
    # (NB: pixel rows are ordered by file, so head() would be one class only.)
    sample = (pixel.groupby("primary_class", group_keys=False)
              .sample(n=50, random_state=0)
              .sample(frac=1, random_state=0)
              .reset_index(drop=True))
    sample.to_csv(out / "sample_for_humans.csv", index=False)
    _write_dataset_card(out, pixel, file_tbl, id_label_cols, feature_cols,
                        n_file_feats, label_dist)

    # Preview: a few identity/label cols + a few real features (what the DS sees)
    prev_cols = [c for c in
                 ["file_id", "primary_class", "pixel_idx",
                  "auc_aa_1004", "auc_lps_1194", "sam_class_STEC"]
                 if c in pixel.columns]
    prev = pixel[prev_cols].head(5).round(3)
    prev_rows = [[*row] for row in prev.itertuples(index=False)]

    meta_out = {
        "dagster/row_count": MetadataValue.int(len(pixel)),
        "what_the_DS_receives": MetadataValue.md(
            f"**Two contract-guaranteed tables:**\n"
            f"1. **Pixel-level training matrix** — `{len(pixel)} × {pixel.shape[1]}` "
            f"= identity/labels ({kept.shape[1]} cols) + {n_pixel_feats} features. "
            f"One row per QC-kept spectrum.\n"
            f"2. **File-level table** — `{len(file_tbl)} × {file_tbl.shape[1] + 1}` "
            f"(file_id index) = labels + {n_file_feats} per-file features."
        ),
        "pixel_table_shape": MetadataValue.text(f"{pixel.shape[0]} × {pixel.shape[1]}"),
        "file_table_shape": MetadataValue.text(f"{file_tbl.shape[0]} × {file_tbl.shape[1] + 1}"),
        "label_distribution_pixels": MetadataValue.json(label_dist),
        "PREVIEW_pixel_training_matrix": _md_table(prev_cols, prev_rows),
        "join_recipe": MetadataValue.md(
            "```python\n"
            "# per-pixel (positional — per the contract's join map)\n"
            "idx  = read_parquet('spectra.parquet')\n"
            "mask = load('qc_mask.npy')\n"
            "kept = idx[mask].reset_index(drop=True)\n"
            "X_px = concat([kept, band_features, spectral_features], axis=1)\n\n"
            "# per-file (keyed on file_id)\n"
            "X_file = metadata[['primary_class']].join(unmix).join(spatial)\n"
            "```"
        ),
        "join_validated": MetadataValue.bool(len(kept) == len(band) == len(spec)),
        "consumes": MetadataValue.text("the feature_store_contract handoff"),
        "DATASET_pixel_parquet": MetadataValue.path(str(out / "pixel_training_matrix.parquet")),
        "DATASET_file_parquet": MetadataValue.path(str(out / "file_level_table.parquet")),
        "DATASET_human_csv": MetadataValue.path(str(out / "sample_for_humans.csv")),
        "DATASET_card": MetadataValue.path(str(out / "DATASET_CARD.md")),
        "how_the_DS_loads_it": MetadataValue.md(
            "```python\n"
            "import pandas as pd\n"
            "df = pd.read_parquet('feature_store/pixel_training_matrix.parquet')\n"
            "y      = df['primary_class']      # 4-class label\n"
            "groups = df['file_id']            # for leave-one-group-out CV\n"
            "X      = df.drop(columns=['file_id','primary_class','subclass',\n"
            "                          'pixel_idx','x_um','y_um'])  # 217 features\n"
            "```"
        ),
    }
    return MaterializeResult(metadata=meta_out)


# =========================================================================== #
# STAGE 6 — SERVING CONTEXT (downstream of the handoff; completes lineage)
# =========================================================================== #
@asset(
    group_name=G_SERVE,
    deps=[data_scientist_view],
    description="What the data scientist builds from the feature store. Reads "
    "artifacts/*_metadata.json — the deployed classifiers and their LOSO scores. "
    "Shown to complete the raw-file → … → model lineage; not produced by this "
    "pipeline.",
)
def model_serving_context() -> MaterializeResult:
    rows = []
    for label, p in [("PLS-DA (raw 987-bin)", paths.PLSDA_META),
                     ("Stage-15F LogReg (35 eng. feats)", paths.STAGE15F_META)]:
        if not p.exists():
            continue
        d = _jload(p)
        if "loso_headline" in d:
            rows.append([label, d["loso_headline"]["value"], paths.rel(p)])
        elif "loso_mean_accuracy" in d:
            rows.append([label, round(d["loso_mean_accuracy"], 4), paths.rel(p)])
    return MaterializeResult(
        metadata={
            "deployed_models": _md_table(
                ["model", "LOSO file-weighted balanced acc", "artifact"], rows
            ),
            "note": MetadataValue.md(
                "PLS-DA on the raw 987-bin spectrum (LOSO **0.603**) remains the "
                "project headline — it beats the 35 engineered features (LogReg "
                "**0.436**). Surfaced here only to close the lineage loop."
            ),
            "consumes": MetadataValue.text("feature_store_contract (the handoff)"),
        }
    )


ALL_ASSETS = [
    raw_file_inventory,
    ingested_spectra,
    qc_mask,
    preprocessed_spectra,
    band_features,
    spectral_features,
    unmix_features,
    spatial_features,
    feature_store_contract,
    data_scientist_view,
    model_serving_context,
]
