# Atlas Feature Store — Data Contract v1.0.0

> The interface between the data-engineering pipeline and the data scientist. Every value below was read from the real files in `data_cache/` at generation time; the asset checks enforce the declared invariants against those files on every run.

_Generated: 2026-06-02T12:28:15.385491+00:00_

## Guaranteed invariants

- **QC funnel reconciles:** 7999 in − 6 (SNR<5.0) − 871 (background) = **7122** kept (89.04%) — `True`
- **987-bin preprocessed axis:** declared 987, observed 987 bins over [400.41, 3049.18] cm⁻¹ (crop window [400.0, 3050.0])
- **Canonical raw axis:** declared linspace(76.0, 3499.0, 2048); observed 2048 bins, [76.0, 3499.0]
- **Per-pixel rowcount == QC keep:** qc_keep=7122, band=7122, spectral=7122
- **Feature columns:** band 166 + spectral 51 + unmix 33 + spatial 10 = **260** columns on disk. unmix has 1 diagnostic column (mcr_residual_norm_mean) on top of its 32 MCR moment features; counting unmix as 32 gives the often-quoted '259 features'. The real on-disk column total is 260.

## Stores

| store | file | format | grain | key | rows | cols | nulls |
|---|---|---|---|---|---|---|---|
| metadata | `data_cache/metadata.parquet` | parquet | per-file | file_id | 87 | 23 | 39 |
| spectra_index | `data_cache/spectra.parquet` | parquet | per-pixel | file_id, pixel_idx | 7999 | 6 | 767 |
| spectra_array | `data_cache/spectra_array.npy` | npy | matrix | _(positional)_ | 7999 | 2048 | — |
| qc_mask | `data_cache/qc_mask.npy` | npy | vector | _(positional)_ | 7999 | 1 | — |
| spectra_array_preprocessed | `data_cache/spectra_array_preprocessed.npy` | npy | matrix | _(positional)_ | 7999 | 987 | — |
| band_features | `data_cache/band_features.parquet` | parquet | per-pixel | _(positional)_ | 7122 | 166 | 26 |
| spectral_features | `data_cache/spectral_features.parquet` | parquet | per-pixel | _(positional)_ | 7122 | 51 | 0 |
| unmix_features | `data_cache/unmix_features.parquet` | parquet | per-file | file_id | 87 | 33 | 0 |
| spatial_features | `data_cache/spatial_features.parquet` | parquet | per-file | file_id | 87 | 10 | 0 |

## Join / lineage map (how the DS consumes this)

- **metadata** — primary identity table; file_id is the universal join key
- **spectra_index** — row-aligned (position) to spectra_array.npy and qc_mask.npy
- **spectra_array** — position i <-> spectra_index row i
- **qc_mask** — position i <-> spectra_index row i; True = kept
- **spectra_array_preprocessed** — position i <-> spectra_index row i (BEFORE qc_mask applied)
- **band_features** — POSITIONAL: row i <-> spectra_index[qc_mask].reset_index(drop=True) row i. To attach file_id/pixel_idx, the consumer must re-derive from spectra_index filtered by qc_mask. (Contract risk: implicit key.)
- **spectral_features** — POSITIONAL: same alignment as band_features.
- **unmix_features** — index = file_id (subset of metadata.file_id)
- **spatial_features** — index = file_id (subset of metadata.file_id)

> ⚠️ **Contract risk surfaced honestly:** `band_features` and `spectral_features` carry **no** `file_id`/`pixel_idx` columns — they are **positionally aligned** to `spectra_index` filtered by `qc_mask`. A consumer that reorders or re-filters either side silently breaks the join. The `band/spectral rows == qc_mask.sum()` check is the guardrail.

## Null policy

- **metadata**: subclass null iff H2O (8 files); exposure_ms null for 31 files + header_numx/numy/xsize/ysize MAY be null (source headers unreliable). file_id/primary_class/n_pixels NEVER null.
- **spectra_index**: subclass null iff H2O pixel (767 rows). All other columns NEVER null.
- **spectra_array**: all-finite (no NaN/Inf); enforced at interp in io.py
- **qc_mask**: boolean, no nulls
- **spectra_array_preprocessed**: all-finite
- **band_features**: NULL POCKET: ~3 rows / 7122 (0.04%) carry NaN in roi_*_skew/kurt and bio_* columns (degenerate/flat spectra → undefined moment). Consumer MUST impute or drop. Tolerance: <=1% of rows.
- **spectral_features**: zero nulls (observed 0).
- **unmix_features**: zero nulls (observed 0).
- **spatial_features**: zero nulls (observed 0).
