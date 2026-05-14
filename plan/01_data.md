# 01 — Data

> **Mutability:** stable. Update only when the source data changes.
> **Last verified:** 2026-05-14 (parser exit 0 on all 87 files).

## Inventory (87 files total)

| Path | Files | Notes |
|---|---|---|
| `H20/*.xls` | 8 | water blanks, no subclass |
| `STEC/{O103H2,O121H19,O157H7}/*.xls` | 9 + 9 + 9 = **27** | |
| `Non STEC/{83972,ATCC25922,K-12}/*.xls` | 8 + 9 + 8 = **25** | |
| `Salmonella/{Dublin,Heidelburg,Typhimurium}/*.xls` | 9 + 9 + 9 = **27** | one Heidelburg file is `.txt` |
| `Salmonella/Heidelburg/R427_*.txt` | (counted above) | identical tab-delim format; parser globs both extensions |

**Total spectra:** ~9,500–10,500 raw. 7,999 after 200-px-per-file cap. 7,122 after QC.

## Per-file format

- Tab-delimited ASCII despite `.xls` extension.
- ~44 metadata lines prefixed with `#KEY=\tVALUE`.
- One wavenumber row (2 blank cells + 2048 wavenumber values).
- N pixel rows, each: `x_um \t y_um \t intensity_0 \t ... \t intensity_2047`.
- Intensity values use **comma thousands-separators** (e.g. `1,034.00`). Must strip before float-cast.

## Critical edge cases

1. **`#NUMX` and `#NUMY` headers are wrong for early-batch files** (R357–R371, Feb–early March). R357 declares `NUMX=10, NUMY=2` but contains a 22×17 grid (374 pixels). **Always derive grid dims from `unique(x_um) × unique(y_um)`**; treat header values as documentation only.
2. **Multi-map mosaics:** R364 (STEC, 324 rows = 9 tiled maps) and R370 (Salmonella Dublin, 720 rows = 9 tiled maps). No per-map separator. Treat as one big spatial scan.
3. **R371 (Salmonella Typhimurium) has a partial scan**: 351 of expected 360 pixels. Included with `is_complete_scan=False`.
4. **Wavenumber axis drifts ~0.05 cm⁻¹ across calibration batches.** All files have 2048 points, but `wn[0]` varies 75.87–75.92 depending on `#AC` calibration date. The parser interpolates every spectrum onto a single canonical axis (`linspace(76.0, 3499.0, 2048)`).
5. **Heidelburg vs Heidelberg** — folder name is "Heidelburg" (likely typo for the German city). Kept as-is in subclass labels; mention in final README.

## Canonical artifacts (post-parse)

| Path | Shape / type | Purpose |
|---|---|---|
| `data_cache/spectra.parquet` | (N, 6) — `file_id, primary_class, subclass, pixel_idx, x_um, y_um` | long-form pixel index, row-aligned with the array |
| `data_cache/spectra_array.npy` | float32 `(N, 2048)` | raw intensities on canonical wn axis |
| `data_cache/wavenumber_axis.npy` | float32 `(2048,)` | canonical axis `linspace(76, 3499, 2048)` |
| `data_cache/metadata.parquet` | (87, 23) | per-file header + provenance (sha256, mtime, grid, dates) |
| `data_cache/build.log` | JSONL | per-file parse audit trail |
| `data_cache/spectra_array_preprocessed.npy` | float32 `(N, 987)` | after preprocess pipeline (cropped + SNV) |
| `data_cache/wavenumber_axis_preprocessed.npy` | float32 `(987,)` | wn after crop (400–1800 + 2800–3050) |
| `data_cache/qc_mask.npy` | bool `(N,)` | True = passed QC |
| `data_cache/qc_info.json` | dict | per-file retention rates |
