# Atlas Raman — Data-Engineering Pipeline (Dagster)

The **orchestration / data-contract / observability** layer over the existing
Atlas Raman science code. It models how raw, messy source files flow to a
clean, contract-guaranteed feature store that the data scientist consumes.

> **Honesty note.** This layer does **not** reinvent the science. Every asset is
> a thin wrapper that calls the existing `atlas/*` functions and/or reads the
> already-materialized `data_cache/` outputs. Every number and schema in the UI
> is read from a **real file** at run time — nothing is hardcoded. Two stages
> additionally **re-invoke the real `atlas` code live** to prove the lineage is
> genuine (see *Lineage proof* below).

---

## One command

```bash
./orchestration/demo_pipeline.sh        # → http://127.0.0.1:3333
```

Then in the Dagster UI:
- **Assets** tab → the DAG + lineage. Click any asset for its schema, row
  counts, and quality metadata.
- **Checks** → the data-quality gates (all green = the contract holds).
- **Materialize all** → re-runs the whole pipeline against `data_cache/`
  (~6 s; idempotent — reads the cache, attaches fresh metadata, re-runs the
  gates).

> Use `127.0.0.1`, not `localhost` — a Next.js dev server commonly squats on
> IPv6 `[::1]:3000`. Override the port with `PORT=4444 ./orchestration/demo_pipeline.sh`.

Isolation: runs in a dedicated **`.venv-dagster`** so the science `.venv` is
untouched (Dagster needs `antlr4>=4.10`; the science env pins `omegaconf`'s
`antlr4==4.9.*` — irreconcilable in one env, so they're separated).

---

## The DAG (real numbers, verified against the files)

```
 stage 1 INGEST          stage 2 QUALITY        stage 3 PREPROCESS
 raw_file_inventory  ┌─► qc_mask                preprocessed_spectra
   87 files          │     7999 −6 SNR −871 bg    7999 × 987 (float32)
   (86 .xls + 1 .txt)│     = 7122 kept (89.0%)    987-bin axis INVARIANT
        │            │     re-runs atlas.qc       re-runs atlas.preprocess
        ▼            │     live (counts match)    on a sample (987 bins)
 ingested_spectra ───┤            │                       │
   7999 × 2048       │            └───────────┬───────────┘
   canonical axis    │                        ▼  features = preprocessed[qc_mask]
   re-parses 1 file  │     stage 4 FEATURES
   live via atlas.io │     band_features    7122 × 166   per-pixel (positional)
                     │     spectral_features 7122 × 51    per-pixel (positional)
                     │     unmix_features      87 × 33    per-file (file_id)
                     │     spatial_features     87 × 10    per-file (file_id)
                     │            │
                     ▼            ▼  stage 5 HANDOFF
              feature_store_contract  → CONTRACT.md + contract.json
                     │                  260 cols, join map, null policy, invariants
                     ▼  stage 6 SERVING (context, not produced here)
              model_serving_context     PLS-DA 0.603 / LogReg 0.436 (LOSO)
```

---

## Data-engineering concepts on display

| Concept | Where it shows up |
|---|---|
| **Explicit DAG + dependencies** | the asset graph; edges are real `deps=[...]` |
| **Data contract @ handoff** | `feature_store_contract` asset + `orchestration/contract/CONTRACT.md` (schema, dtypes, grain, join keys, null policy, invariants) |
| **Quality gates w/ observable metrics** | 14 Dagster **asset checks**; the ingest+QC funnel, per-file retention, schema/range/rowcount checks |
| **Idempotent / cacheable / reproducible** | assets read the cache as source-of-truth; `sha256` per file in metadata; QC/preprocess re-derive deterministically |
| **Lineage** | raw file → `atlas.io` → cache → `atlas.qc`/`preprocess` → features → contract → model, wired through the graph |
| **Storage choices** | `.npy` (dense float32 matrices) vs `.parquet` (typed columnar features) — rationale in each asset's metadata |

### The data-quality gates (asset checks)

Hard gates (ERROR, **blocking** — fail the run if violated):
`canonical_axis_invariant`, `raw_matrix_shape`, `no_fatal_ingest_errors`,
`qc_funnel_reconciles`, `qc_gate_reproducible`, `axis_987_invariant`,
`prep_rows_match_ingest`, `band_rows_equal_qc_keep`, `spectral_rows_and_nulls`,
`unmix_per_file_integrity`, `spatial_per_file_integrity`, `contract_consistent`.
Soft gates (WARN): `qc_retention_floor`, `band_null_tolerance`.

### Lineage proof (the receipt)

```bash
PYTHONPATH=. .venv-dagster/bin/python orchestration/verify_lineage.py
```
Re-runs `atlas.io.parse_file`, `atlas.qc.apply_qc`, and
`atlas.preprocess.preprocess_matrix` against the cache and confirms they
reproduce it. (The full preprocess is ~30 min of arPLS fits, so the live
re-runs are on cheap slices; the QC re-run is full and reproduces the funnel
counts exactly.)

---

## Honesty notes — three things the source brief got slightly wrong

These are surfaced in the contract, not hidden:

1. **260 columns, not 259.** `166 + 51 + 33 + 10 = 260` on disk. The "259"
   figure counts `unmix` as 32 (it omits the `mcr_residual_norm_mean` diagnostic
   column).
2. **Per-pixel features have no join key.** `band_features` / `spectral_features`
   carry **no** `file_id`/`pixel_idx` column — they are **positionally aligned**
   to `spectra_index[qc_mask]`. Only `unmix`/`spatial` key on `file_id`. This is
   a real contract *risk*; the `*_rows_equal_qc_keep` checks guard it.
3. **QC ⟂ preprocess (parallel, not sequential).** QC runs on the 2048-bin raw
   axis (it needs the 1800–2500 cm⁻¹ noise band that preprocessing crops out).
   The 7122 selection is applied at *feature* time, not before preprocessing.

And one the brief got right that's worth stating: the QC live re-run reproduces
the funnel **counts** exactly but the boolean mask differs on **2 / 7999**
pixels — a percentile-boundary tie in the background filter, not a bug.

---

## Layout

```
orchestration/
  demo_pipeline.sh                  one command → UI (port 3333, .venv-dagster)
  verify_lineage.py                 standalone "the cache came from atlas/*" proof
  contract/
    CONTRACT.md                     generated, human-readable data contract
    contract.json                   generated, machine-readable contract
  atlas_orchestration/
    definitions.py                  Dagster entrypoint (Definitions)
    assets.py                       10 software-defined assets (wrap atlas/* + cache)
    checks.py                       14 asset checks (the quality gates)
    contract.py                     declared invariants + contract generator
    paths.py                        file locations (no data values)
```
