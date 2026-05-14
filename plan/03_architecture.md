# 03 — Architecture

> **Mutability:** stable. Changes here imply a `10_decision_log.md` entry.
> **Last verified:** 2026-05-14.

## Pipeline overview

```
raw .xls/.txt files                          ┐
   │ parse + interp to canonical wn axis     │  one-time, cached
   ▼                                         │
data_cache/spectra.parquet  (long-form)      │
data_cache/metadata.parquet  (per file)      ┘
   │
   │ preprocess: cosmic-ray → arPLS baseline → Savitzky-Golay → crop → SNV
   │ QC: SNR ≥ 5, per-file 10th-pct background filter
   ▼
Preprocessed spectrum matrix  (N ~7K × ~1000 bins after cropping)
   │
   │ group-aware splits (file_id) — both protocols share the same artifacts
   ▼
┌─ Protocol A: StratifiedGroupKFold(5) by primary class
└─ Protocol B: Leave-One-Strain-Out (9 folds, one per bacterial subclass)
   │
   ├─ Classical: LogReg, LinearSVM, RBF-SVM, RF, XGBoost, PLS-DA
   ├─ CNN:        small 1D-CNN (medium variant only if compute permits)
   └─ Transformer: small 1D-Transformer (~200K params)
   │
   │ per-spectrum predict_proba → per-file aggregation via soft vote
   ▼
Evaluation: macro-F1 (primary), balanced acc, per-class P/R/F1,
            per-subclass recall, 4×4 + 10×10 confusion matrices,
            calibration (Brier + reliability), LOSO summary table
   │
   ▼
outputs/<run_id>/  {config.resolved.yaml, metrics.json, *.png/*.csv, model.{joblib|pt}, predictions.parquet}
```

## Repo layout

```
NomadX/
├── atlas/                          # importable library
│   ├── __init__.py
│   ├── config.py                   # dataclass + omegaconf loader
│   ├── io.py                       ✅ parse .xls/.txt, write/read cache
│   ├── preprocess.py               ✅ cosmic-ray, arPLS, Sav-Gol, crop, SNV, derivative
│   ├── qc.py                       ✅ SNR + background pixel filter
│   ├── splits.py                   ⏳ StratifiedGroupKFold + LOSO
│   ├── features.py                 ⏳ PCA wrapper, optional PLS transformer
│   ├── models_classical.py         ⏳ LogReg, SVM, RF, XGB, PLS-DA pipelines
│   ├── models_cnn.py               ⏳ CNN1D (small + medium variants)
│   ├── models_transformer.py       ⏳ small 1D-Transformer
│   ├── train.py                    ⏳ unified training (classical + CNN/transformer)
│   ├── evaluate.py                 ⏳ FoldResult / ModelResult, aggregation
│   └── plots.py                    ⏳ spectrum overlays, confusion, embeddings
├── configs/                        ⏳ one YAML per experiment
├── scripts/
│   ├── build_dataset.py            ✅ parse → cache (idempotent via mtime+sha)
│   ├── preprocess_dataset.py       ✅ preprocess + cache
│   ├── build_eda_notebook.py       ✅ build atlas_driver.ipynb from cell list
│   └── run_experiments.py          ⏳ train + eval + write outputs
├── notebooks/
│   └── atlas_driver.ipynb          ✅ EDA blocks 1–9
├── tests/test_smoke.py             ⏳ synthetic-data end-to-end smoke
├── expected/smoke_metrics.json     ⏳ regression tolerance file
├── outputs/                        # gitignored; per-run artifacts
│   ├── eda/                        ✅ blocks 1-8 plots (raw+SNV baseline)
│   └── eda_v2/                     ✅ block 9 plots (full preprocessing)
├── data_cache/                     # gitignored; parsed parquet + arrays
├── .github/workflows/ci.yml        ⏳ ruff + mypy + smoke
├── pyproject.toml                  ⏳ deps + tool config
├── requirements.txt                ✅ pinned lock
├── Makefile                        ⏳ build / verify / clean targets
├── README.md                       ⏳ final write-up
└── plan/                           ✅ this index
```

## Module-by-module spec

### A. Ingestion (`atlas/io.py`, `scripts/build_dataset.py`) — ✅ done

- Schema (long-form, one row = one pixel-spectrum): `file_id (cat) | primary_class (cat) | subclass (cat|null) | pixel_idx (int32) | x_um (f32) | y_um (f32)`. Intensities live in the row-aligned npy.
- Metadata table (`data_cache/metadata.parquet`): one row per file with parsed header — laser, exposure, NUMX/NUMY (header + actual grid_nx/grid_ny), XSIZE/YSIZE, acquisition_date, ac_calibration_date, file_sha256, file_mtime, is_complete_scan, wn_start, wn_end. Used for batch-effect / drift analysis and idempotent rebuilds.
- Canonical wn axis: `linspace(76.0, 3499.0, 2048)` written to `data_cache/wavenumber_axis.npy`. Every spectrum linearly interpolated at parse time.
- Companion array `data_cache/spectra_array.npy` (float32 `(N, 2048)`), row-aligned with parquet.
- Idempotency: skip files whose `(mtime, sha256)` match the cache.
- Validation per file:
    - FATAL → file excluded: 0 rows, NaN in intensities, axis ≠ 2048 pts, axis not monotonic, post-interp NaN.
    - WARNING → file included with flag: row count ≠ `grid_nx*grid_ny`, pixel-capped, missing header fields.
- Build log → JSONL at `data_cache/build.log`.

### B. Preprocessing + QC (`atlas/preprocess.py`, `atlas/qc.py`) — ✅ done

Pipeline order (per spectrum):
1. **Cosmic-ray removal** — median-filter (window=7) Z-score outlier replacement on positive spikes.
2. **Baseline** — arPLS via `pybaselines.whittaker.arpls(lam=1e5, max_iter=50, diff_order=2)`.
3. **Smoothing** — Savitzky-Golay (window=9, polyorder=3).
4. **Crop** — fingerprint **400–1800 cm⁻¹** + C–H **2800–3050 cm⁻¹**. ~987 bins.
5. **SNV** — z-score per spectrum.
6. **(Classical only)** — optional concat of 2nd-derivative spectrum (`savgol_filter(..., deriv=2)`). CNN consumes SNV-only.

QC filter (`atlas/qc.py`):
- **SNR** = mean(900–1700) / std(1800–2500). Drop pixels with SNR < 5.
- **Background** — per-file 10th-percentile of integrated fingerprint intensity AND low spectral MAD → mark as background candidate.

Acceptance criteria for the pipeline (must pass before modeling): no NaN/Inf, ≥50% pixel retention per file, primary-class PCA silhouette ≥ 0.30 (currently failing — see `07_findings.md`), ≥50 ANOVA-significant bins (FDR q=0.05).

### C. Splits (`atlas/splits.py`) — ⏳ next

Two protocols, both file-level (no pixel ever crosses train/test):
- **Protocol A — StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)**: `groups=file_id`, stratify on `primary_class`. With 87 files this gives ~17–18 test files per fold.
    - **H₂O pre-balancing.** With only 8 H₂O files, the auto-stratifier produces empty H₂O test folds for at least one fold (sklearn issue #33085). Mitigation: pre-assign H₂O files to test folds (round-robin so each fold gets 1 or 2), then run `StratifiedGroupKFold` on the remaining 79 bacterial files, then merge.
    - **Stable row order.** Sort the file-level dataframe by `file_id` ASCII order before splitting; record `sklearn.__version__` and seed in split artifact metadata. Reason: splitter assigns folds based on input row order.
    - **Calibration-date sanity check.** Per fold, log a warning (do not fail) if any `ac_calibration_date` appears in both train and test. With 12 dates and the same-date batch ratio of 0.893 ([07_findings.md §batch-effect](07_findings.md#batch-effect)), some cross-fold contamination is unavoidable but must be visible.
- **Protocol B — LOSO**: 9 iterations, one per bacterial subclass. Hold out all files of that subclass; train on the other 8 subclasses + H₂O. H₂O is permanently in training — too few files (8) for class-level holdout.
- **Inner validation for HPO**: within each outer-train fold, `StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=derived)`, fold 0 fixed as inner-val. Documented compromise: single-fold inner validation variance is 2–4× nested CV variance — risks ~2–5% suboptimal HPO selection. Acceptable for budget; pre-registered in [08_expectations.md](08_expectations.md).
- All scaling/PCA fit on inner-train only.

Split artifacts: two master JSON files at `data_cache/splits/`:
- `protocol_a.json` — 5 folds × `{train_file_ids, test_file_ids, train_row_indices, test_row_indices, n_train_spectra, n_test_spectra, test_class_dist}`.
- `protocol_b.json` — 9 folds keyed by held-out subclass.
- Both contain a `meta` block: `cache_hash` (sha256 of `qc_mask.npy` + `primary_class` + `subclass` + seed), `seed`, `sklearn_version`, `created_at`, `n_spectra_total`, `n_spectra_qc`, `n_files`.

Row indices in artifacts are resolved against the 7,999-row preprocessed array (the qc_mask filter is already applied — only QC-passing rows appear in any fold). Persisting BOTH file_ids and row_indices is intentional redundancy — derivation at consume time would re-execute QC logic in every consumer.

Smoke check: load both protocol files, assert (a) zero pixel index appears in both train and test of any fold, (b) zero file_id appears in both train and test of any fold, (c) every fold's union of train+test row indices is a strict subset of the QC-passing index set.

### D. Classical models (`atlas/models_classical.py`) — ⏳

| Model | Pipeline | Hyperparameter search |
|---|---|---|
| LogReg | StdScaler → PCA(n∈{50,100,150}) → LogReg(multinom, balanced) | C ∈ log-uniform [1e-3, 10] |
| Linear SVM | StdScaler → PCA(n∈{50,100,150}) → LinearSVC(balanced) | C ∈ log-uniform [1e-3, 10] |
| RBF SVM | StdScaler → PCA(n∈{50,100,150}) → SVC(rbf, balanced, prob) | C ∈ log-uniform [1, 1000], gamma ∈ log-uniform [1e-4, 1e-1], 20 trials |
| Random Forest | RF(balanced, n_jobs=-1), no PCA | n_estimators ≥ 300, max_features ∈ {sqrt, 0.1}, max_depth ∈ {None, 20, 40}, min_samples_leaf ∈ {1, 3, 5}, 12-point grid |
| XGBoost | StdScaler → XGB (sample_weight=`balanced`) | n_estimators ∈ [200, 1000] w/ early-stop patience 30, max_depth ∈ {3..7}, lr log-uniform [1e-2, 3e-1], reg_lambda log-uniform [1e-1, 1e1], subsample ∈ [0.6, 1.0], colsample_bytree ∈ [0.5, 0.9], 20 trials |
| PLS-DA | StdScaler → PLS(n) → LogReg | n_components ∈ {5, 8, 10, 12, 15, 20, 25, 30} |

- **PCA cap = 100 components (with grid {50,100,150}), NOT 99% variance.** 99%-var retains noise-laden trailing PCs that hurt linear models — reconstruction target ≠ classification target.
- **2nd-derivative concat is dropped.** arPLS+Sav-Gol+SNV already removed baseline and smoothed noise; 2nd derivative on smoothed-SNV data adds noise-dominated features.
- **Class weighting:** `class_weight="balanced"` on sklearn models; `sample_weight=compute_sample_weight("balanced", y_train)` on XGBoost (`scale_pos_weight` is binary-only).
- **Random state:** every Pipeline step's `random_state` and XGBoost `seed` derive from `fold_seed = (master_seed * 31337 + fold_idx) % 2**31`.
- All scaling/PCA fit inside CV folds.

### E. 1D-CNN (`atlas/models_cnn.py`) — ⏳

Two variants; **small is default**.

**Small (~110K params, default):**
- Input: (B, 1, ~1000). InstanceNorm1d at input.
- 4 conv stages, channels 16→32→48→64, kernels 15-7-7-5, GELU+BN, MaxPool/2 between stages 1–3, dilation 2 in stage 3.
- GAP head → Linear(64→32→4).

**Medium (~450K params):** channels 32→64→96→128, dilation 2 and 4, GAP⊕GMP concat, 2-layer MLP head. Only if compute permits.

Training:
- Loss: CrossEntropy with per-fold class weights, label smoothing 0.05.
- Optimizer: AdamW lr=3e-4, wd=1e-4. Cosine annealing + 3-epoch linear warmup, 60 epochs total.
- Batch size 32 (CPU) / 128 (GPU). Early-stop on val macro-F1, patience 10, restore best.
- Grad clip 1.0. Seed everything; document MPS non-determinism in README.

Augmentation (training only):

| Op | Range | p |
|---|---|---|
| Gaussian noise | σ ∈ [0.005, 0.03]·spec_std | 0.5 |
| Multiplicative scale | [0.9, 1.1] | 0.4 |
| Wavenumber shift | ±3 bins | 0.4 |
| Sinusoidal baseline | A ∈ [0, 0.05]·max, period [200, 800] bins | 0.3 |
| Mixup | α=0.2, any class | 0.3 |

Optional (flag-controlled):
- **Subclass auxiliary head** (9 subclasses, water masked): `L = 1.0·L_primary + 0.3·L_sub`.
- **DANN domain adversary** on `file_id`: only if memorization probe fails.

Inference: per-pixel softmax → per-file soft-vote; flag files with <50 QC-passing pixels as `low_support`. Temperature scaling on val.

### F. Transformer (`atlas/models_transformer.py`) — ⏳

Small 1D-Transformer (~200K params). Patch-embed the spectrum (~20-bin patches), positional encodings, 4–6 encoder blocks, `[CLS]` token → classifier head. Not expected to beat CNN; included for honest benchmarking.

### G. Evaluation (`atlas/evaluate.py`) — ⏳

Shared result schema:

```python
@dataclass
class FoldResult:
    fold_id: int | str
    protocol: str  # "group_kfold" | "loso"
    model_name: str
    # spectrum-level + file-level + per-subclass + confusion 4x4 and 10x10
    # calibration: brier_score, reliability points
    # metadata: n_train, n_test, training_time_s
    # predictions: path to parquet shard with columns
    #   [spectrum_id, file_id, subclass_true, primary_true, p_class0..p_class3, fold_id]

@dataclass
class ModelResult:
    folds: list[FoldResult]
    # mean ± SD (label as SD, not CI) of macro-F1, balanced-acc, brier at spectrum and file levels
```

**Statistical reporting rules:**

- **Effective N for significance is the number of files (87), not the number of spectra (7,122).** Within-file cosine similarity = 0.997 ([07_findings.md §interfile-similarity-high](07_findings.md#interfile-similarity-high)); pixels in one file are near-duplicates and inflate pixel-level CIs by ~9× (`√(7122/87)`).
- **Across-fold summary: mean ± SD** — explicitly labelled SD, not CI. If a CI is forced, use t-interval with df=4 (t=2.776 at 95%), report n_folds=5 alongside.
- **LOSO summary: mean + range (min–max)**, not SD or CI. Each held-out subclass is one observation; with 9 folds there is no distribution to summarize.
- **Minimum detectable F1 difference: ≈0.08.** Smaller deltas are noise at this sample size; do not crown a winner on a margin below that.
- **Headline 4×4 / 10×10 confusion = sum of counts across folds** (preserves class-imbalance signal). Per-fold normalized confusions reported separately with mean ± SD on the diagonal.
- **Multi-class Brier = proper formulation** (sum over K classes of (p_k − y_k)², averaged over samples). NOT averaged one-vs-rest. Computed per spectrum then averaged to file level by mean over a file's spectra.
- **Reliability diagrams: 10 quantile (equal-frequency) bins**, not equal-width — class-skewed probability distributions empty the tails of equal-width bins.

Predictions persisted as parquet per fold under `outputs/<run_id>/predictions_fold{i}.parquet`, keyed on `spectrum_id` and `file_id` columns (NOT positional arrays). Critical for cross-family comparability when CNN/Transformer arrive.

Reports produced:
- 4×4 confusion (spectrum-level and file-level), counts and normalized.
- 10×10 confusion (9 subclasses + H₂O × 4 predicted primary classes mapped back).
- LOSO summary table (9 rows × {held-out subclass, n_files, n_spectra, fold macro-F1, parent-class recall, soft-vote file-level recall, Δ vs GroupKFold baseline}).
- Reliability diagrams + Brier scores.
- Per-subclass recall under both protocols.

### H. MLOps (`atlas/config.py`, `configs/`, `scripts/run_experiments.py`) — ⏳

- **Config:** omegaconf-loaded YAML, one file per experiment. CLI overrides via `--override splits.n_folds=3 model.hyperparameters.C=10.0`.
- **Seed propagation:** one master seed → `random`, `numpy`, `torch` (CPU+MPS+CUDA), every sklearn `random_state`, DataLoader generator. Per-fold seed = `(master_seed * 31337 + fold_idx) % 2**31`.
- **Thread pinning:** runner sets `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` before importing numpy/sklearn/xgboost. Without this, FP-reduction order varies between runs of identical code and the same seed produces slightly different metrics.
- **Experiment log:** append-only JSONL at `outputs/runs.jsonl` with `{run_id, timestamp, config_hash, git_sha, platform, metrics{}, artifacts[], duration_s}`.
- **Per-run artifacts** in `outputs/<run_id>/`: `config.resolved.yaml`, `metrics.json`, `confusion_matrix.png`, `loso_table.csv`, `per_subclass_metrics.csv`, `model.{joblib|pt}`, `predictions.parquet`.
- **CLI:** `python scripts/run_experiments.py --config <yaml> [--override ...] [--protocol groupkfold|loso] [--smoke-test] [--output outputs/]`.
- **Verify target:** `make verify` runs smoke on synthetic data, asserts metrics within ±0.005 of `expected/smoke_metrics.json`.
- **CI:** ruff + mypy + import-smoke + `--smoke-test` (no real-data training in CI; synthetic only).
- **Dependencies** pinned in `pyproject.toml` → `requirements.txt`: numpy 1.26.4, pandas 2.2.2, scipy 1.13.1, scikit-learn 1.5.1, pyarrow 16.1.0, pybaselines 1.1.0, torch 2.3.1, xgboost 2.1.0, omegaconf 2.3.0, matplotlib 3.9.1.
