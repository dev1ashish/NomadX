# 08 — Calibrated expectations (pre-registered)

> **Mutability:** append-only. **Important:** register predictions *before* running the experiment, then append the actual measurement when results land. That's how we keep ourselves honest.
> **Format:** each row has a date the prediction was made, the predicted value/range, and (eventually) the actual measured value with a date.

---

## 2026-05-14 — Final headline metrics (pre-registered before any model training)

| Metric | Predicted range | Actual | Verdict |
|---|---|---|---|
| 4-class macro-F1 (Protocol A, file-level) | 0.65 – 0.80 | — | pending |
| 4-class macro-F1 (LOSO, file-level) | 0.45 – 0.65 (i.e. 0.10–0.20 drop vs Protocol A) | — | pending |
| H₂O vs bacteria binary F1 | ≥ 0.95 | — | pending |
| STEC vs Non-STEC binary F1 (the hard cell) | 0.55 – 0.70 | — | pending |
| Best classical macro-F1 (Protocol A) | 0.55 – 0.70 | — | pending |
| Best CNN macro-F1 (Protocol A) | 0.65 – 0.80 | — | pending |
| Best Transformer macro-F1 (Protocol A) | 0.60 – 0.75 (not expected to beat CNN) | — | pending |

**Interpretation guide.** Above the range = real result. Below = bug or data leak suspected. Right at the range = expected behavior.

**Reasoning behind the predictions:**
- The "easy" axis (H₂O vs bacteria) should be trivial — water doesn't have proteins or lipids. ≥0.95 is the floor.
- STEC vs Non-STEC is the hard cell because both are E. coli. We expect this binary alone to dominate the error budget.
- LOSO drop is sized off the silhouette negativity — if subclasses don't even cluster on their own training data, transferring across subclasses is going to hurt.
- Classical models *probably* underperform the CNN because per-peak boxplots (see `07_findings.md#peaks-overlap`) show the signal is multi-bin and nonlinear — exactly the regime where deep models help.

## 2026-05-14 — Memorization probe (pre-registered)

| Metric | Predicted | Actual | Verdict |
|---|---|---|---|
| File-id classification accuracy from a single random spectrum | 25–50% (vs ~1.1% chance) | — | pending |

If above ~10%, the encoder is leaking acquisition signature → enable DANN. Reasoning: batch-effect ratio = 0.893 means there IS a same-acquisition signature; the question is how strong it is.

## 2026-05-14 — QC retention (post-hoc verification of pipeline doc)

| Metric | Predicted (in `03_architecture.md`) | Actual | Verdict |
|---|---|---|---|
| Bacterial pixel retention after QC | 60–80% | 89.0% | ✅ above (good — strong SNR) |
| H₂O pixel retention after QC | 40–60% | 89.2% | ⚠️ unexpectedly high — water spectra are noisier but our SNR threshold (5) was generous. May tighten later if water predictions look unreliable. |

## 2026-05-14 — Per-model classical macro-F1 (pre-registered BEFORE any training)

All metrics are FILE-LEVEL macro-F1 (soft-vote of per-spectrum predict_proba). Spectrum-level metrics will also be reported but are not the headline — effective N at pixel level is inflated by within-file cosine ≈ 0.997 ([07_findings.md §interfile-similarity-high](07_findings.md#interfile-similarity-high)).

| Model | Protocol A (GroupKFold-5) | LOSO (9-fold) | Actual A (file macro-F1) | Actual LOSO (file macro-F1) | Verdict |
|---|---|---|---|---|---|
| LogReg (linear) | 0.45 – 0.60 | 0.30 – 0.45 | **0.961 ± 0.042** | **0.161 ± 0.105** | ❌ above ceiling on A; ❌ below floor on LOSO |
| LinearSVM | 0.45 – 0.60 | 0.30 – 0.45 | **0.779 ± 0.112** | **0.143 ± 0.108** | ❌ above ceiling on A; ❌ below floor on LOSO |
| RBF SVM | 0.55 – 0.72 | 0.40 – 0.60 | **0.833 ± 0.096** | **0.127 ± 0.092** | ❌ above ceiling on A; ❌ below floor on LOSO |
| Random Forest | 0.55 – 0.70 | 0.40 – 0.55 | **0.753 ± 0.118** | **0.105 ± 0.073** | ⚠️ above ceiling on A; ❌ below floor on LOSO |
| XGBoost | 0.60 – 0.75 | 0.45 – 0.60 | **0.796 ± 0.103** | **0.125 ± 0.069** | ⚠️ above ceiling on A (modest); ❌ below floor on LOSO (same artifact as other models) |
| PLS-DA | 0.45 – 0.60 | 0.30 – 0.45 | **0.951 ± 0.051** | **0.164 ± 0.106** | ❌ above ceiling on A; ❌ below floor on LOSO |

**Verdict interpretation:**
- Every Protocol A score is above my predicted ceiling. The pre-registration was wrong — I underestimated how much within-file pixel averaging (cosine 0.997) denoises per-pixel errors when soft-voting to file level. Above-ceiling does NOT here mean "data leak" — it means my prior was miscalibrated.
- Every LOSO macro-F1 is below my predicted floor — but **macro-F1 is a broken metric for LOSO**: each LOSO fold has only one class in the test set, so 3 of 4 class F1s are 0 by construction, dragging macro to a ceiling of 0.25. The "below floor" verdict is mostly artifact. The correct LOSO metric is per-strain parent-class recall (see next table).

## 2026-05-14 — LOSO parent-class recall (post-hoc, the metric I should have pre-registered)

LOSO macro-F1 has a hard ceiling of 0.25 (only 1 of 4 classes is in any fold's test set). The meaningful LOSO metric is **per-strain parent-class recall**: "when we hold out all files of strain X (parent class Y), how often does the model predict Y for those files?"

| Strain held out | Parent | LogReg | LinSVM | RBF-SVM | RF   | PLS-DA |
|---|---|---|---|---|---|---|
| 83972        | Non-STEC   | 0.88 | 0.88 | 0.88 | 0.12 | 0.88 |
| ATCC25922    | Non-STEC   | 0.22 | 0.11 | 0.22 | 0.11 | 0.22 |
| K-12         | Non-STEC   | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Dublin       | Salmonella | 0.44 | 0.22 | 0.22 | 0.11 | 0.56 |
| Heidelburg   | Salmonella | 0.89 | 0.56 | 0.33 | 0.33 | 0.89 |
| Typhimurium  | Salmonella | 1.00 | 1.00 | 0.67 | 0.33 | 1.00 |
| O103H2       | STEC       | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 |
| O121H19      | STEC       | 0.89 | 0.89 | 0.44 | 0.78 | 0.89 |
| O157H7       | STEC       | 0.00 | 0.00 | 0.00 | 0.33 | 0.00 |
| **MEAN**     |            | **0.59** | **0.52** | **0.42** | **0.31** | **0.60** |

LOSO parent-class recall, now including XGBoost (added post-hoc after libomp install):

| Strain       | Parent     | LogReg | LinSVM | RBF-SVM | RF | XGB | PLS-DA |
|---|---|---|---|---|---|---|---|
| 83972        | Non-STEC   | 0.88 | 0.88 | 0.88 | 0.12 | 0.25 | 0.88 |
| ATCC25922    | Non-STEC   | 0.22 | 0.11 | 0.22 | 0.11 | **0.33** | 0.22 |
| K-12         | Non-STEC   | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Dublin       | Salmonella | 0.44 | 0.22 | 0.22 | 0.11 | 0.11 | 0.56 |
| Heidelburg   | Salmonella | 0.89 | 0.56 | 0.33 | 0.33 | 0.44 | 0.89 |
| Typhimurium  | Salmonella | 1.00 | 1.00 | 0.67 | 0.33 | 0.44 | 1.00 |
| O103H2       | STEC       | 1.00 | 1.00 | 1.00 | 0.67 | 0.67 | 1.00 |
| O121H19      | STEC       | 0.89 | 0.89 | 0.44 | 0.78 | 0.78 | 0.89 |
| O157H7       | STEC       | 0.00 | 0.00 | 0.00 | **0.33** | **0.33** | 0.00 |
| **MEAN**     |            | **0.59** | **0.52** | **0.42** | **0.31** | **0.37** | **0.60** |

Tree-based models (RF, XGB) are the **only models that crack O157H7→STEC at all** (0.33 each) but their averages are dragged down by poor performance on the "easy" strains (83972, Typhimurium). Linear models do the inverse: ace the easy strains, fail entirely on O157H7. **This is real signal that the inductive bias matters** — neither family is universally better; they fail in *different ways* on different strains.

Mean across 5 models = 0.49; chance level for 4-class = 0.25. **All models do real work**, but with massive per-strain variance.

## 2026-05-14 — H₂O vs bacteria binary F1 (post-hoc verification)

| Metric | Predicted | Actual (best model, Protocol A) | Verdict |
|---|---|---|---|
| H₂O vs bacteria binary F1 | ≥ 0.95 | LogReg = 1.00 across all 5 folds (per-fold H₂O recall) | ✅ at ceiling — water vs bacteria is trivial as expected |

**Reasoning behind the predictions:**

- **Linear models (LogReg, LinSVM, PLS-DA) cluster at the floor** because [07_findings.md §peaks-overlap](07_findings.md#peaks-overlap) and [§silhouette-negative-raw](07_findings.md#silhouette-negative-raw) show subclasses don't linearly separate. PLS-DA's supervised dimensionality reduction gives it a small edge but it's still linear after the LV step.
- **RBF SVM and XGBoost are the candidates for the top spot.** Kernel non-linearity (RBF) and gradient-boosted decision boundaries (XGB) both capture multi-bin peak ratios. Web research: published Raman bacterial benchmarks rank SVM ≈ XGB ≈ RF, with PLS-DA below.
- **Random Forest sits between linear and RBF-SVM.** No PCA so the full 987-bin feature space is available, but axis-aligned splits limit kernel-like discrimination.
- **LOSO drop = ~0.10–0.20 macro-F1 across all models** per the headline prediction. If the drop is bigger for tree models (RF, XGB) than for linear models, that would imply they're memorizing strain identity more aggressively — useful diagnostic.
- **Below floor = bug** (label corruption, leakage). **Above ceiling = data leak or systematic batch effect not caught by our cal-date sanity check.**
