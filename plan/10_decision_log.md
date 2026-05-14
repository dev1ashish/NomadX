# 10 — Decision log (audit trail)

> **Mutability:** append-only. Never edit historical entries; correct via a new entry with a date.
> **Purpose:** provenance for every decision that's now baked into `02_decisions.md` or `03_architecture.md`. If a future session questions a choice, the rationale lives here.

---

## 2026-05-14 — Initial decision sweep (planning phase)

- **Deliverable = repo + driver notebook.**
- **Modeling = classical baseline + 1D-CNN.** Later (same day) extended to also include a small 1D-Transformer.
- **Subclass treatment = stratify + per-subclass metrics + LOSO.**
- **Spectra count revised down from ~30K to ~10K** based on actual file inspection (most files are 9×9 grids, not the 100×100 the agents initially assumed).
- **`.txt` file confirmed at `Salmonella/Heidelburg/R427_*.txt`** (not Non-STEC as one agent guessed).
- **Canonical wavenumber axis = `linspace(76, 3499, 2048)`,** all spectra interpolated at parse time.
- **Spectral crop = fingerprint 400–1800 cm⁻¹ + C–H stretch 2800–3050 cm⁻¹** (~1100 bins total). User deferred to recommendation. Later validated empirically — C-H stretch carries the strongest 4-class signal (see `07_findings.md#anova-c-h-stretch`).
- **Per-file pixel cap = 200 pixels/file** (random subsample at parse time). User deferred to recommendation. Prevents R364 (324 px) and R370 (720 px) from dominating their subclasses.
- **CNN device = auto-detect** (MPS if available else CPU), pinned in resolved config per run.
- **R371 partial scan = include with `is_complete_scan=False` flag** (351/360 px).
- **`.txt` file at Heidelburg = include.** Confirmed identical format; the parser globs both `*.xls` and `*.txt`.
- **Primary headline metric = macro-F1.** Balanced accuracy also reported.
- **PLS-DA = available but not in the default headline results.** Implement it; ship a separate `configs/extra_pls.yaml`; mention in README under "additional models".
- **Heidelburg subclass label = keep as-is** (do not normalize to "Heidelberg"); note the likely typo in README.
- **DANN domain adversary = off by default.** Enable only if the memorization probe shows file-ID leakage.
- **Small 1D-Transformer added as a third model arm** (after classical + CNN). Patch-embed the spectrum (~20-bin patches), positional encodings, 4–6 encoder blocks, `[CLS]` token → classifier head. Not expected to beat the CNN; included for honest benchmarking and as a credibility move.

## 2026-05-14 — Parser ships

- **Partial-scan detection switched from `#NUMX`*`#NUMY` (header) to `grid_nx`*`grid_ny` (coord-derived).** Early-batch headers are unreliable; the header-based check missed R371. Now compares actual pixel count against the size of the coord-uniqueness grid. R371 correctly flagged after the change (351/360).

## 2026-05-14 — EDA Block 9 + preprocessing pipeline

- **ANOVA discriminative plot switched to plotting `log10(F)` instead of `-log10(p)`.** Top bins underflow `p` to 0 → `inf` gaps in the line. F-statistic ranking is identical.
- **arPLS API correction.** Originally specified `arpls(lam=1e5, p=0.01, max_iter=50)` — but `p` is not a parameter of `arpls`; it belongs to `asls`. arpls auto-computes asymmetry via the iteratively reweighted PLS scheme. Now using `arpls(lam=1e5, max_iter=50, diff_order=2)`.

## 2026-05-14 — Plan restructuring

- **PLAN.md split into `plan/` directory.** Reason: the single file was mixing 4 kinds of content (stable design, mutable status, append-only findings, append-only audit log), making it hard to reason about which sections were authoritative at any moment. Now: `00_status.md` is mutable; `01_data.md` / `02_decisions.md` / `03_architecture.md` / etc. are stable; `07_findings.md` / `08_expectations.md` / `10_decision_log.md` are append-only. `plan/README.md` is the entry index. Old `PLAN.md` at root is now a one-line redirect.

## 2026-05-14 — Pre-build adjustments to §C (splits) and §D (classical)

Pre-build research pass dispatched 7 subagents (ml-engineer, machine-learning-engineer, data-scientist, ai-engineer, mlops-engineer, data-engineer, data-analyst) and 5 web searches (recent Raman bacterial benchmarks, StratifiedGroupKFold gotchas, LOSO conventions in spectroscopy, PLS-DA vs SVM head-to-head, multi-class Brier reporting). Synthesis produced 7 specific adjustments to `03_architecture.md` before any model code lands.

**§C (splits) — 4 adjustments:**

1. **Manual H₂O balancing across StratifiedGroupKFold folds.** Pre-assign 1–2 H₂O files to each test fold then fill remaining slots via the splitter. Reason: scikit-learn issue #33085 documents degenerate empty-class folds when group counts are small and class distribution is uneven; with only 8 H₂O files in 87 total, at least one fold will land 0 H₂O test files, producing undefined H₂O recall and uncomputable macro-F1 averaging. Fold-count change (e.g. drop to 4) is NOT the fix — 4 starves the training set, 10 makes the issue worse.

2. **Stable row order before splitting + recorded splitter version.** Sort the file-level dataframe by `file_id` ASCII order before calling `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, and write `sklearn.__version__` + the seed into split-artifact metadata. Reason: `StratifiedGroupKFold` assigns groups to folds based on input row order; if `spectra.parquet` is ever rebuilt (new file, different parse order) fold membership changes silently even with the same `random_state`.

3. **`ac_calibration_date` as a sanity-check, not a grouping variable.** Per-fold, assert no calibration date appears in BOTH train and test. With 12 distinct dates and same-date batch ratio = 0.893 ([findings.md §batch-effect](07_findings.md#batch-effect)), some date-pairs will cross-fold; log a warning rather than failing the build. Using calibration_date AS a grouping variable would over-constrain the splitter past the point of feasibility (some dates have only 1–2 files; not enough headroom for 5-fold balance).

4. **Split-artifact format: one master JSON per protocol; persist BOTH file_ids AND resolved row-indices.** Two files: `data_cache/splits/protocol_a.json` and `data_cache/splits/protocol_b.json`. Each fold stores `train_file_ids`, `test_file_ids`, `train_row_indices`, `test_row_indices` (resolved against the 7,999-row preprocessed array via qc_mask), `n_train_spectra`, `n_test_spectra`, `test_class_dist`. Reason: deriving row indices at consume-time re-executes QC logic in every consumer; the redundancy IS the reproducibility guarantee. JSON for human readability; size cost trivial (~500 KB total).

**§D (classical models) — 3 adjustments:**

5. **Cap PCA at 100 components (not 99% variance).** Treat n_components as a small grid: {50, 100, 150}. Reason: 99% var on 987 SNV-scaled bins retains 80–200 components including noise-laden trailing PCs that hurt linear models. 99% var is a reconstruction target, not a classification target. Applies to LogReg, LinearSVM, RBF-SVM.

6. **Drop the optional 2nd-derivative concat for classical models.** Reason: arPLS+Sav-Gol+SNV already removed baseline and smoothed noise. 2nd derivative on smoothed-SNV data amplifies edge noise and adds 987 redundant features approximately negative-of-curvature of the existing features. Empirically won't help; mathematically adds noise. Removes the optional flag from §B.6.

7. **Expand PLS-DA n_components grid: {5, 8, 10, 12, 15, 20, 25, 30}.** Reason: original {5,10,20,30} misses the 8–15 sweet spot for bacterial Raman per web research (multiple SERS PLS-DA studies report optima in that band).

**§G (evaluation) — clarifications layered on top of these adjustments:**

- **Significance and CIs on macro-F1 are reported at file level, not pixel level.** Effective N = 87 (number of files), not 7,122 (number of spectra). Within-file cosine similarity = 0.997 ([findings.md §interfile-similarity-high](07_findings.md#interfile-similarity-high)) means pixels in one file are near-duplicates; pixel-level CIs are wrong by ~9× (√(7122/87)).
- **Across-fold summary = mean ± SD,** explicitly labelled SD not CI. With n=5 folds any CI is theatre; if reviewer pressure forces one, use t-interval with df=4 (t=2.776 at 95%). LOSO summary = mean + range (min–max), no SD.
- **Minimum detectable F1 difference at this scale: ~0.08.** Anything smaller is noise; do not call a winner on margins below that.
- **Headline 4×4 / 10×10 confusion = sum of counts across folds.** Per-fold normalized confusions reported separately with mean ± SD on the diagonal.
- **Brier score = proper multiclass formulation** (sum over K classes of (p_k − y_k)², averaged over samples), NOT averaged one-vs-rest. Reliability diagrams use 10 quantile bins (equal-frequency), not equal-width.

**§H (mlops) — clarifications:**

- Per-fold seed derivation: `fold_seed = (master_seed * 31337 + fold_index) % 2**31`. Threaded into every Pipeline step's `random_state` (sklearn) and XGBoost `seed`.
- Pin `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` in the runner. Reason: without thread-count pinning, FP-reduction order in RF/XGB and BLAS calls varies between runs and the same seed produces slightly different metrics.
- Predictions persisted as parquet per fold keyed on `spectrum_id` and `file_id` columns (NOT positional arrays). Critical for later CNN/Transformer compatibility where DataLoader ordering is non-trivial.

**Not adopted from research:**

- ai-engineer suggested one-config-per-experiment vs matrix-style. mlops-engineer pushed matrix. Deferred — building classical first with per-file YAMLs (`logreg_groupkfold.yaml`, etc.) is fine for 12 experiments; can refactor to matrix later if it gets painful.
- data-analyst suggested cutting the 9-subclass recall heatmap as "bait." Will keep computing per-subclass recall in `FoldResult` (it's cheap and may surface a useful finding), but the headline plot set in the final README will be the 6 they recommended.

## 2026-05-14 — CNN small-variant architectural fixes mid-session

Two corrections to plan/03_architecture.md sec E (small variant) caught during the first CNN training pass.

### 1. Channel widening: 16/32/48/64 → 32/64/96/128

The §E spec lists channels `16 → 32 → 48 → 64` AND a `~110K params` target in the same paragraph. Those two facts are internally inconsistent: with kernels 15-7-7-5 the literal channel widths arithmetic-out to **33K params, not 110K** (precise count 32,628). 33K can't be reconciled with 110K by tweaking kernel sizes either.

**Empirical confirmation that 33K underfits.** Protocol A fold 0, no augmentation, no label smoothing, no class weighting, 120 epochs (well past the spec's 60-epoch budget): `train_acc` plateaus at 0.74 and `val_macro_f1` ceiling is 0.56. Classical PLS-DA on the same fold hits file-F1 = 0.951. The CNN was not capable of fitting its own training data, let alone generalizing.

**Fix.** Double every channel width to `32 → 64 → 96 → 128`. With kernels 15-7-7-5 unchanged, the literal count comes to **124,484 params** — matches the spec's `~110K` target within ~13%. The medium variant remains differentiated by its extra dilation (2 and 4 vs single dilation=2), GAP⊕GMP concat, and 2-layer MLP head, so the small/medium gap is preserved.

Reading the doc charitably: the channel-width tuple was written down at planning time before anyone actually counted parameters. The `~110K` figure was the intended target; the listed widths were aspirational and off by ~3.4×. Going to `32/64/96/128` follows the intent.

This is NOT a step toward the medium variant. The user's standing instruction "expand only if small CNN looks promising" was about a structural jump (dilation, dual-pool head, MLP). This is a numerical correction to the small variant itself.

### 2. Per-bin standardize at input

SNV-preprocessed spectra have **per-row** mean = 0 and std = 1 by construction, but **per-bin** mean ranges -0.46 to +3.84 and per-bin std ranges 0.05 to 0.39 across the 987 bins. Classical models pipe through `StandardScaler` in their sklearn `Pipeline` before PCA + LogReg; this removes the per-bin bias and re-equalizes per-bin variance so each bin contributes comparably to the linear (or kernel) decision boundary.

The §E spec describes "InstanceNorm1d at input" only — InstanceNorm normalizes **per-spectrum-across-L**, which is a near-noop on data that's already SNV-normalized. There's no per-BIN normalization in the spec'd CNN front-end.

**Empirical confirmation that per-bin standardize is the missing piece.** Same fold-0 / no-aug / 60-epoch test:

| Front-end | train_acc @ epoch 60 | val_macro_f1 best |
|---|---|---|
| 33K params, InstanceNorm only | 0.43 | 0.40 |
| 124K params, InstanceNorm only | 0.69 | 0.51 |
| 124K params, InstanceNorm + per-bin standardize | **0.88** | **0.56** |

Per-bin standardize is what turns "model can't fit train" into "model fits train but generalization is the question." That's the regime where pre-registered expectations actually apply.

**Implementation.** `SmallCNN1D` carries `(input_mu, input_sd)` as `register_buffer` tensors. `train.train_cnn_fold` fits them on the outer-train per fold and calls `model.set_input_stats(mu, sd)`; `state_dict()` then captures the buffers so the memprobe v2 (which loads the encoder via `load_state_dict`) inherits the same input pipeline automatically. No external sidecar files.

### What this changes about already-pre-registered expectations

The CNN expectations in [08_expectations.md](08_expectations.md) were registered against a "~110K param, well-fit" model. The 33K version registered against the same numbers was a guarantee of below-floor failure (Protocol A actual 0.40 file-F1 vs predicted 0.92–0.98). With both fixes applied, the registered ranges are the right thing to compare against.

Lesson logged so it's not repeated for the Transformer session: **fit a single training fold without augmentation as a sanity check before launching the full sweep.** If the model can't fit train, no amount of regularization sweep or LR tuning will help. This costs one extra fold (~1 minute on MPS) and would have caught the 33K problem in the first 5 minutes.

## 2026-05-14 — XGBoost spec cheapened mid-session

**Original spec** (from machine-learning-engineer subagent recommendation): `n_estimators: 200–1000`, `max_depth: 3–7`, 20 trials, `n_jobs=1`. **Problem:** ran ~19 min on Protocol A fold 0 without finishing; projected ~2 hours per protocol. **Root cause:** I copied the subagent's "n_estimators 200–1000 with early stopping" recommendation but never implemented early stopping in `make_xgb`, so every random-search trial trained all N trees. Combined with `n_jobs=1` (mlops-engineer's reproducibility advice over-applied), each fold took forever.

**Re-spec post-RF-postmortem:** `n_estimators: 100–300`, `max_depth: 3–6`, 10 trials, `n_jobs=4`. Resulting GroupKFold runtime: **7.4 min** (~16× speedup). Result quality: file macro-F1 = 0.796 ± 0.103 — lands between LinearSVM (0.78) and RBF-SVM (0.83), as expected for this dataset.

**Lesson recorded for future model specs:** when LOSO crater + biology-ceiling has already been established, subsequent classical models should be sized to *answer the experiment cheaply*, not to "try to win." The XGBoost run cost was ~2 hours of wall time for ~0.04 macro-F1 difference vs a 7-minute cheaper run — clearly the wrong trade-off given everything we already knew. Memorialized so we don't repeat it for the CNN session: don't auto-train the medium variant unless the small variant suggests it's worth it.
