# 00 — Status

> **Mutability:** mutable. Rewrite freely.
> **Last updated:** 2026-05-14.

## Phase

| Phase | State | Notes |
|---|---|---|
| 0. Discovery | done | Raman hyperspectral maps, tab-text `.xls` files. |
| 1. Subagent research (5 in parallel) | done | data-engineer, data-scientist, ml-engineer, ai-engineer, mlops-engineer all returned. |
| 2. Synthesis into master plan | done | Now split across `plan/`. |
| 3. User approval of plan | done | Open decisions resolved — see `02_decisions.md`. |
| 4. Implementation | in progress | Steps 1–3 done (parser, EDA, preprocess+QC). |

## What's done

- `atlas/io.py` — parses tab-delimited .xls/.txt, strips comma thousands, derives grid from coord uniqueness, interpolates to canonical wn axis.
- `scripts/build_dataset.py` — discovers files, runs the parser, caches to `data_cache/`. Idempotent via (mtime, sha256).
- Raw cache: 87/87 files parse with 0 fatal errors. 7,999 spectra @ 2048 bins.
- `notebooks/atlas_driver.ipynb` — EDA blocks 1–9 (1–8 on raw+SNV, 9 on full preprocessing).
- `outputs/eda/` and `outputs/eda_v2/` — 13 + 7 plot/csv artifacts.
- `atlas/preprocess.py` — cosmic-ray removal → arPLS baseline → Sav-Gol smoothing → crop → SNV → optional 2nd derivative.
- `atlas/qc.py` — SNR ≥ 5 + per-file background pixel detection.
- `scripts/preprocess_dataset.py` — caches preprocessed array. 7,122/7,999 spectra retained after QC.

## What's next (in order)

1. ✅ **`atlas/splits.py`** — StratifiedGroupKFold(5) on file_id (Protocol A) + Leave-One-Strain-Out (Protocol B, 9 folds). Splits cached at `data_cache/splits/protocol_{a,b}.json`. Smoke check passes.
2. ✅ **`atlas/models_classical.py` + `atlas/evaluate.py`** — 5 of 6 classical models trained under both protocols (XGBoost deferred until `brew install libomp`). Results in `outputs/runs.jsonl` and per-run dirs.
3. ⏳ `atlas/models_cnn.py` + `atlas/train.py` — small CNN variant under both protocols. Will use MPS auto-detect per memory preference.
4. ⏳ `atlas/models_transformer.py` — small 1D-Transformer (~200K params) under both protocols.
5. ⏳ **Memorization probe** — train CNN to predict `file_id`; if accuracy materially > chance, encoder is leaking acquisition signature → enable DANN. The classical LOSO crater strongly suggests this probe will fire.
6. ⏳ Final plots, README narrative, `make verify`, CI green.

## Open items / TODOs

- **Install libomp + re-run XGBoost.** `brew install libomp` then `python scripts/run_classical.py --models xgb --protocols group_kfold loso`. Will fill the missing row in `08_expectations.md`.
- Tweak arPLS crop start to ~450 cm⁻¹ (currently 400) — boundary artifact at left edge of preprocessed spectra. See [findings.md §arpls-boundary](07_findings.md#arpls-boundary-artifact).
- Re-run Block 9 after crop tweak.
- **Run a bacteria-only ANOVA (3 classes, excluding H₂O)** to surface within-bacterial discriminative bins. Current 4-class ANOVA bins are about water-vs-bacteria, per [findings.md §anova-bins-vs-stec-discriminative-bands](07_findings.md#anova-bins-vs-stec-discriminative-bands).
