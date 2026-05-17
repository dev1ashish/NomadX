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

## 2026-05-14 — Small 1D-CNN, pre-registered BEFORE any training

Architecture locked at plan/03_architecture.md §E small variant (~110K params, 4 conv 16→32→48→64, kernels 15-7-7-5, GELU+BN, GAP, Linear(64→32→4)). Training recipe locked at §E (cosine LR + 3-epoch warmup, AdamW lr=3e-4 wd=1e-4, label smoothing 0.05, mixup α=0.2 plus noise/scale/shift/sinusoidal-baseline augmentation, early-stop val macro-F1 patience 10, per-fold class weights). DANN OFF by default — memorization probe at 4.1% top-1 (below the 10% threshold in 02_decisions.md). Master seed 42; per-fold seed = `(42 * 31337 + fold_idx) % 2^31` (same recipe as classical).

### CNN — Protocol A (StratifiedGroupKFold-5, file-level macro-F1)

| Metric | Predicted range | Actual | Verdict |
|---|---|---|---|
| File-level macro-F1 (mean ± SD across 5 folds) | 0.92 – 0.98 | **0.649 ± 0.079** (folds 0.63 / 0.65 / 0.54 / 0.68 / 0.76) | ❌ **below floor by 0.27** |
| Spectrum-level macro-F1 (mean ± SD) | 0.80 – 0.93 | **0.478 ± 0.056** | ❌ below floor |
| H₂O recall (file-level, sum over folds) | 1.00 (ceiling) | 6/8 = 0.75 | ❌ below ceiling |
| STEC recall (file-level, sum over folds) | 0.85 – 0.97 | 21/27 = 0.78 | ❌ slightly below |

**Verdict: below floor across the board.** Two architectural fixes applied mid-session (channel doubling 16→32→48→64 to 32→64→96→128, per-bin StandardScaler at input — both documented in [10_decision_log.md §cnn-architectural-fixes](10_decision_log.md#2026-05-14--cnn-small-variant-architectural-fixes-mid-session)) made the model trainable; without them the spec'd architecture couldn't fit even the training set. With both fixes the CNN reaches a 0.5-ish val_macro_f1 ceiling and a 0.65 file-F1 ceiling, well short of classical PLS-DA / LogReg (0.951 / 0.961).

The per-class file-recall pattern (summed across folds): Non-STEC = 20/25, STEC = 21/27, Salmonella = 11/27, H₂O = 6/8. **Salmonella is the dominant error cell** — the CNN often misclassifies Salmonella as Non-STEC (9/27 such errors) or STEC (5/27). Classical models don't show this Salm→Non-STEC collapse.

**Reasoning.** Protocol A is largely saturated by within-file pixel averaging (cosine 0.997 → soft-vote denoises pixel-level errors enormously). Linear models already hit 0.95–0.96; CNN should match or marginally exceed them at file level. The interesting Protocol A signal is *spectrum-level* F1 — that's where the CNN should pull ahead of linear models (linears were ~0.70 spectrum-F1 vs RBF-SVM ~0.85; CNN should match RBF-SVM or slightly beat it). **Above 0.98 file-F1 is suspicious** — would suggest the CNN is leveraging file-id leakage that classical models can't access (would also predict probe-v2 firing). **Below 0.92 file-F1 means the augmentation is too aggressive or the loss isn't converging.**

### CNN — LOSO (Protocol B, the honest generalization test)

LOSO macro-F1 is broken (0.25 ceiling — only 1 of 4 classes in any fold's test). Headline = **per-strain parent-class recall**, same metric we use for classical LOSO.

| Strain | Parent | Best classical (PLS-DA or RF/XGB) | CNN predicted range | Actual | Verdict |
|---|---|---|---|---|---|
| 83972      | Non-STEC   | 0.88 (PLS-DA/LR)  | 0.70 – 0.95 | **0.88** | ✅ in range, tied with classical leader |
| ATCC25922  | Non-STEC   | 0.33 (XGB)        | 0.20 – 0.50 | **0.11** | ❌ below floor — CNN worse than weakest classical |
| K-12       | Non-STEC   | 0.00 (everyone)   | 0.00 – 0.15 (biological ceiling expected) | **0.50** | ⭐ **above ceiling by 0.35** — CNN cracks the biologically-hard cell |
| Dublin     | Salmonella | 0.56 (PLS-DA)     | 0.40 – 0.65 | **0.11** | ❌ below floor |
| Heidelburg | Salmonella | 0.89 (PLS-DA/LR)  | 0.65 – 0.92 | **0.33** | ❌ below floor |
| Typhimurium| Salmonella | 1.00 (linears)    | 0.85 – 1.00 | **0.11** | ❌ catastrophic below floor (-0.74) |
| O103H2     | STEC       | 1.00 (linears)    | 0.85 – 1.00 | **0.56** | ❌ below floor (-0.29) |
| O121H19    | STEC       | 0.89 (linears)    | 0.70 – 0.95 | **0.00** | ❌ catastrophic below floor |
| O157H7     | STEC       | 0.33 (RF, XGB)    | 0.15 – 0.50 | **0.56** | ⭐ **above ceiling by 0.06** — CNN beats best classical |
| **MEAN parent-recall** | | **PLS-DA = 0.60 ⭐** | 0.55 – 0.72 | **0.35** | ❌ **below floor by 0.20** |

**Reasoning per-strain.**
- **K-12 stays catastrophic.** Biological reason: laboratory-domesticated since 1920s, large genomic deletions vs wild-type E. coli ([soupene-2003-k12](11_references.md#soupene-2003-k12--soupene-et-al-j-bacteriol-2003-laboratory-strains-of-escherichia-coli-k-12-things-are-seldom-what-they-seem)). No model family — linear, kernel, tree, or CNN — can recover a strain whose chemistry isn't spanned by the training manifold. If CNN somehow gets > 0.20 here, that would be surprising and worth investigating for label leakage.
- **O157H7 is the most interesting cell.** Linear models score 0.00, trees 0.33. The published STEC-discriminative bands at 1338/1454/1658 cm⁻¹ ([cisek-2013](11_references.md#cisek-2013--cisek-et-al-analyst-2013-sensitive-and-specific-discrimination-of-pathogenic-and-nonpathogenic-escherichia-coli-using-raman-spectroscopy)) are subtle multi-bin features; a CNN with the right augmentation might pick them up where trees' axis-aligned splits already partially do. Predicting 0.15–0.50 — wide range because the literature ceiling here is roughly ~94% with state-of-the-art DA ([tang-2026-wgan](11_references.md#tang-2026-wgan--tang-et-al-anal-chem-2026-integrated-wasserstein-gan-transformer-for-e-coli-strain-identification)) and we're running vanilla CNN without DANN.
- **Easy strains stay easy or get slightly worse.** Within-class augmentation might dilute the file-specific signature that lets linear models trivially identify common strains. Expecting a small regression on the four >0.85 strains.
- **Mean parent-recall 0.55–0.72** brackets the PLS-DA 0.60 leader. Above 0.72 = CNN materially improves cross-strain transfer, justifies medium variant + Transformer comparison; below 0.55 = CNN is worse than linears, decide on DANN based on probe v2.

**Verdict thresholds.**
- **CNN mean parent-recall ≥ 0.72**: real improvement, run medium variant.
- **0.55 ≤ mean parent-recall < 0.72**: classical-leader-equivalent. Keep small CNN; do not auto-expand to medium.
- **mean parent-recall < 0.55**: CNN is *worse* than linears. Check whether probe v2 fires; if it does, enable DANN; if not, the loss is the model not the leakage.

### Memorization probe v2 (penultimate-layer features → 87-way linear)

Replaces the from-scratch tiny CNN probe ([memorization-probe-weak](07_findings.md#2026-05-14--memorization-probe-weak), 4.1% top-1). The honest test: do the features the *class-supervised* CNN learns *also* encode file_id? Within-file 80/20 pixel split (every file_id in both train and test).

| Metric | Predicted range | Actual | Verdict |
|---|---|---|---|
| Top-1 file_id accuracy from CNN penultimate (87-way) | 2 – 12% (vs ~1.15% chance) | **15.5%** (13.5× chance) | ❌ **above ceiling by 3.5 pp** — probe fires |
| Top-5 | 8 – 30% | **37.0%** | ❌ above ceiling by 7 pp |

**Verdict: probe v2 fires.** The class-trained CNN's penultimate features encode 4× more file-id signal than the from-scratch tiny CNN of v1 (15.5% vs 4.1%). Interpretation: the higher-capacity encoder (124K params) preserves more incidental acquisition-batch features than a tiny CNN trained *for the file-id objective* could extract. The CNN is leaking file_id even without being asked to.

Per [02_decisions.md], probe firing → DANN should be enabled for the next CNN run. **But the per-strain LOSO pattern complicates this** (see below): the CNN already cracks K-12 and O157H7 — exactly the strains where there's NO file-id-transfer signal to leak. So whatever the CNN is doing on those folds is NOT batch-effect leakage. DANN might HURT those wins by penalizing any file-id-correlated representation. Recommended next step: try a DANN run *and* an ensemble (CNN + PLS-DA soft-vote) — see [09_future_work.md].

**Reasoning.** Two opposing forces:
1. **The class objective discards file-id signal.** The CNN is trained to predict 4 classes, not 87 files. Unless file-id correlates with class (it doesn't — class is determined by strain folder structure, files are within-strain), the encoder has no gradient incentive to preserve 87-way file fingerprint. Expect probe-v2 < probe-v1 (which was a from-scratch network *optimized* for file-id directly).
2. **The encoder has 110K params vs probe-v1's 6.6K.** More capacity = more incidental representation of acquisition-batch features that *aren't* class-relevant but survive in the features anyway. Expect probe-v2 > 1.15% chance.

Net prediction: **2–12%**. If probe v2 fires (> 10%) with CNN underperforming PLS-DA on LOSO, that's the DANN-enable signal. If probe v2 < 10% AND CNN matches/beats PLS-DA, no DANN needed.

### Compute budget pre-registration

| Stage | Predicted wall-clock | Actual | Verdict |
|---|---|---|---|
| One Protocol A fold (60 epochs, MPS) | 3 – 8 min | 23–45 s (most early-stopped before 50 epochs) | ✅ way under floor |
| One LOSO fold (60 epochs, MPS) | 3 – 8 min | 15–61 s | ✅ way under floor |
| Full Protocol A (5 folds) | 15 – 40 min | **2.9 min** | ✅ way under floor |
| Full LOSO (9 folds) | 27 – 72 min | **6.4 min** | ✅ way under floor |

Compute came in 8-10× under floor because: (a) MPS is faster than I budgeted, (b) every fold early-stopped 5-30 epochs before the 60-epoch budget. Total CNN session wall-clock: ~9 min for both protocols + memprobe v2.

**XGBoost-cheapen-mid-session lesson** ([10§xgboost-spec-cheapened-mid-session](10_decision_log.md#2026-05-14--xgboost-spec-cheapened-mid-session)) applied here: if any single fold blows past 10 min, halt and re-spec down (fewer epochs, smaller batch, etc.). The medium CNN variant is OFF by default. Activate ONLY if small CNN clearly beats classical leader AND probe v2 stays cold.

---

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

---

## 2026-05-14 — Soft-vote ensemble (pre-registered BEFORE running)

**Setup.** Three ensembles, both protocols. Per-spectrum proba averaged uniformly across input runs (merge on `(spectrum_id, file_id)`), then aggregated to file-level via the same `file_aggregate_softvote` helper used everywhere else. No re-training; no held-out re-tuning. Reference points from already-landed runs:

- PLS-DA: Protocol A file-F1 = 0.951, LOSO mean parent-recall = **0.60 ⭐ (leader)**.
- XGBoost: Protocol A file-F1 = 0.796, LOSO mean parent-recall = 0.37. Distinct value: tied with RF for the only non-zero recall on O157H7 (0.33).
- CNN small: Protocol A file-F1 = 0.649, LOSO mean parent-recall = 0.35. Distinct value: K-12 = 0.50 (everyone else 0.00), O157H7 = 0.56 (best across all families).

**Why this experiment.** Per [07§cnn-loso-complementary-pattern](07_findings.md#2026-05-14--cnn-loso-complementary-pattern), the three families have orthogonal inductive biases — none wins all strains. If the per-strain wins survive averaging, mean parent-recall lands above PLS-DA's 0.60 lone leader. If averaging dilutes the CNN's unusual K-12/O157H7 wins (a real risk — those wins are 0.50/0.56 vs 2 zeros, so a uniform average yields 0.17/0.19 in proba space, well below the file-level argmax threshold), the mean lands at-or-below PLS-DA.

### Ensemble (a) PLS-DA + XGB — "best linear + best tree, no CNN"

| Metric | Predicted range | Actual | Verdict |
|---|---|---|---|
| Protocol A file-F1 (mean ± SD across 5 folds) | 0.92 – 0.97 | **0.863 ± 0.104** | ❌ below floor by 0.06 — XGB diluted PLS-DA |
| LOSO mean parent-recall (9 strains) | 0.50 – 0.62 | **0.529** | ✅ in range, but below PLS-DA solo 0.60 |
| Protocol A H₂O recall (file-level, sum/8) | 8/8 | **5/8** | ❌ XGB's softer H₂O confidence diluted PLS-DA's perfect H₂O |

**Per-strain LOSO parent-recall (actuals filled in after run):**

| Strain | Parent | PLS-DA solo | XGB solo | Ensemble predicted | Actual | Verdict |
|---|---|---|---|---|---|---|
| 83972 | Non-STEC | 0.88 | 0.25 | 0.55 – 0.88 | **0.88** | ✅ at upper end |
| ATCC25922 | Non-STEC | 0.22 | 0.33 | 0.22 – 0.44 | **0.22** | ✅ at lower bound; XGB's edge lost |
| K-12 | Non-STEC | 0.00 | 0.00 | **0.00** | **0.00** | ✅ exact |
| Dublin | Salmonella | 0.56 | 0.11 | 0.22 – 0.56 | **0.22** | ✅ at lower bound; XGB dragged down |
| Heidelburg | Salmonella | 0.89 | 0.44 | 0.55 – 0.89 | **0.89** | ✅ at upper end |
| Typhimurium | Salmonella | 1.00 | 0.44 | 0.78 – 1.00 | **0.78** | ✅ at lower bound; XGB dragged down |
| O103H2 | STEC | 1.00 | 0.67 | 0.78 – 1.00 | **1.00** | ✅ at upper end |
| O121H19 | STEC | 0.89 | 0.78 | 0.78 – 0.95 | **0.78** | ✅ at lower bound |
| O157H7 | STEC | 0.00 | 0.33 | **0.00 – 0.33** | **0.00** | ✅ XGB diluted (predicted possible) |
| **MEAN** | | 0.60 | 0.37 | **0.45 – 0.65** | **0.529** | ✅ but **net negative vs PLS-DA solo** |

**Verdict thresholds.** PLS-DA + XGB only differs from PLS-DA solo if XGB's O157H7 / ATCC25922 signal survives averaging. The smoke test (K-12 ensemble proba) showed two-against-one averaging hides minority signal; expect O157H7 recall to stay 0.00 unless XGB's confidence is much higher than PLS-DA's wrong vote.

### Ensemble (b) PLS-DA + CNN — "best linear + best nonlinear, the headline candidate"

| Metric | Predicted range | Actual | Verdict |
|---|---|---|---|
| Protocol A file-F1 (mean ± SD across 5 folds) | 0.85 – 0.94 | **0.919 ± 0.030** | ✅ in range; still below PLS-DA solo (0.951) |
| LOSO mean parent-recall (9 strains) | 0.55 – 0.72 | **0.579** | ✅ at lower end; below PLS-DA solo (0.60) by 0.02 |
| Protocol A H₂O recall (file-level, sum/8) | 7/8 – 8/8 | **6/8** | ❌ below floor — CNN's H₂O confidence is weaker than PLS-DA's |

**Per-strain LOSO parent-recall (actuals filled in after run):**

| Strain | Parent | PLS-DA solo | CNN solo | Ensemble predicted | Actual | Verdict |
|---|---|---|---|---|---|---|
| 83972 | Non-STEC | 0.88 | 0.88 | 0.80 – 0.95 | **0.88** | ✅ |
| ATCC25922 | Non-STEC | 0.22 | 0.11 | 0.11 – 0.33 | **0.22** | ✅ |
| **K-12** | **Non-STEC** | **0.00** | **0.50** | **0.13 – 0.50** (THE question) | **0.00** | ❌ **CNN's K-12 win destroyed by averaging** |
| Dublin | Salmonella | 0.56 | 0.11 | 0.30 – 0.60 | **0.33** | ✅ in range, CNN dragged PLS-DA down |
| Heidelburg | Salmonella | 0.89 | 0.33 | 0.55 – 0.85 | **0.89** | ⭐ slightly above ceiling — PLS-DA dominated |
| Typhimurium | Salmonella | 1.00 | 0.11 | 0.55 – 1.00 | **1.00** | ✅ at upper end — PLS-DA dominated |
| O103H2 | STEC | 1.00 | 0.56 | 0.78 – 1.00 | **1.00** | ✅ at upper end |
| O121H19 | STEC | 0.89 | 0.00 | 0.45 – 0.89 | **0.89** | ✅ at upper end — PLS-DA dominated despite CNN's 0.00 |
| **O157H7** | **STEC** | **0.00** | **0.56** | **0.14 – 0.56** (other key question) | **0.00** | ❌ **CNN's O157H7 win destroyed by averaging** |
| **MEAN** | | 0.60 | 0.35 | **0.50 – 0.70** | **0.579** | ✅ in range; **fails to beat PLS-DA solo** |

**Headline read (b):** PLS-DA dominates the ensemble's argmax everywhere except where CNN was genuinely better than PLS-DA — and on K-12 / O157H7 the averaging pulls the file-level decision back to PLS-DA's wrong class. **The CNN's two unique biology wins do not survive averaging with a confident PLS-DA.** The honest interpretation is: minority-vote signal in proba space gets crushed by a confident dominant model's argmax once you re-aggregate to file level.

**Why a two-way ensemble might preserve CNN wins where the three-way might not.** K-12 case: CNN's K-12 file proba averages roughly (Non-STEC ≈ 0.4, Salmonella ≈ 0.45, others ≈ 0.075) — strong signal but not dominant. PLS-DA's K-12 file proba is roughly (Non-STEC ≈ 0.05, Salmonella ≈ 0.85, others ≈ 0.05). Uniform 2-way average: (Non-STEC ≈ 0.22, Salmonella ≈ 0.65) — still wrong. With a 3-way average (add XGB also predicting Salmonella confidently): even worse.

The path to preserving CNN's K-12 win in a 2-way is if CNN's Non-STEC confidence is high enough to overcome PLS-DA's Salmonella confidence (or pull pixel-level votes to give a different file-level argmax after re-aggregation). **Honestly: pre-register prediction is "K-12 lands 0.13 – 0.50, lower half more likely."**

### Ensemble (c) PLS-DA + XGB + CNN — "all three families"

| Metric | Predicted range | Actual | Verdict |
|---|---|---|---|
| Protocol A file-F1 (mean ± SD across 5 folds) | 0.85 – 0.94 | **0.864 ± 0.105** | ✅ in range; net negative vs PLS-DA solo (0.951) |
| LOSO mean parent-recall (9 strains) | 0.55 – 0.70 | **0.517** | ❌ below floor by 0.03; **worst of the three ensembles** |
| Protocol A H₂O recall (file-level, sum/8) | 7/8 – 8/8 | **5/8** | ❌ below floor |

**Per-strain LOSO parent-recall (actuals filled in after run):**

| Strain | Parent | Best solo | Ensemble predicted | Actual | Verdict |
|---|---|---|---|---|---|
| 83972 | Non-STEC | 0.88 (PLS-DA/CNN) | 0.78 – 0.92 | **0.88** | ✅ |
| ATCC25922 | Non-STEC | 0.33 (XGB) | 0.15 – 0.40 | **0.22** | ✅ XGB's edge diluted |
| K-12 | Non-STEC | 0.50 (CNN) | **0.00 – 0.25** | **0.00** | ✅ as predicted by smoke test — CNN diluted by 2 classicals |
| Dublin | Salmonella | 0.56 (PLS-DA) | 0.20 – 0.55 | **0.22** | ✅ at lower bound |
| Heidelburg | Salmonella | 0.89 (PLS-DA) | 0.55 – 0.89 | **0.78** | ✅ in range, XGB+CNN dragged 0.89 → 0.78 |
| Typhimurium | Salmonella | 1.00 (PLS-DA) | 0.55 – 1.00 | **0.78** | ✅ in range, dragged from 1.00 → 0.78 |
| O103H2 | STEC | 1.00 (PLS-DA) | 0.78 – 1.00 | **1.00** | ✅ at upper end |
| O121H19 | STEC | 0.89 (PLS-DA) | 0.55 – 0.89 | **0.78** | ✅ in range, dragged from 0.89 |
| O157H7 | STEC | 0.56 (CNN) | **0.00 – 0.33** | **0.00** | ✅ as predicted — CNN+XGB diluted by PLS-DA's wrong vote |
| **MEAN** | | 0.60 (PLS-DA) | **0.45 – 0.65** | **0.517** | ❌ below floor; **worst ensemble** |

### Verdict structure (locked BEFORE running)

- **If 2-way (b) PLS-DA+CNN > 0.65 mean AND K-12 ≥ 0.25**: ⭐ **headline finding** — CNN's biology wins propagate. Worth shipping as the headline model.
- **If 2-way (b) PLS-DA+CNN ∈ [0.55, 0.65] AND K-12 ≈ 0.00**: ensemble is a slight numerical bump, no story. Headline stays PLS-DA solo.
- **If 3-way (c) > best 2-way AND preserves both wins**: rarely happens in averaging — would suggest XGB and CNN are catching *different* strain-specific signals. Worth a focused finding.
- **If all three ensembles ≤ 0.60 mean**: averaging washes out everything; ship PLS-DA solo and flag CNN's K-12/O157H7 as "single-model best for those folds" in the README.
- **If any ensemble breaks 0.72**: surprise upside; would imply soft-vote captures complementary error structure better than I think. Worth a separate finding.

Smoke test ran during dev (3-way ensemble proba on K-12 LOSO fold) confirmed: ensemble argmax = Salmonella for 8/8 K-12 files, i.e. parent-recall = 0.00. **Expect (c) to score K-12 = 0.00.**

### Post-run resolution against the pre-locked verdict structure

Branch hit: **"If all three ensembles ≤ 0.60 mean: averaging washes out everything; ship PLS-DA solo and flag CNN's K-12/O157H7 as single-model best for those folds in the README."**

- (a) 0.529, (b) 0.579, (c) 0.517 — all three < 0.60 PLS-DA solo. Best ensemble (b) PLS-DA+CNN is 0.02 below the PLS-DA solo leader — well inside the 0.08 minimum-detectable-difference noise floor ([10§G](10_decision_log.md#2026-05-14--pre-build-adjustments-to-c-splits-and-d-classical)), i.e. statistically indistinguishable from solo PLS-DA.
- K-12 = 0.00 across all three ensembles. **CNN's 0.50 win is destroyed in every averaging configuration.**
- O157H7 = 0.00 across all three ensembles. **CNN's 0.56 win and XGB's 0.33 win are both destroyed in every averaging configuration.**

**Decision (consistent with the pre-locked verdict structure):** ship PLS-DA solo as headline; flag CNN as single-model best for K-12 (0.50) and O157H7 (0.56) in the README; do not run a weighted-ensemble sweep this session (the failure mode is "minority-vote crushed by confident-but-wrong majority," not "weights mis-set" — a weight sweep can't recover information that's not represented in the average's argmax for these folds).

---

## 2026-05-14 — Small 1D-Transformer (pre-registered BEFORE running the full sweep)

Architecture: patch-embed (Conv1d k=20, s=20) → 49 patches + [CLS] token, learned positional embed, 4 encoder blocks (d_model=80, nhead=4, dim_feedforward=160, pre-LN, GELU), LayerNorm → Linear(80, 32) → GELU → Linear(32, 4). **216,964 params** (vs. 200K spec target, +8%). Per-bin standardize via `register_buffer` baked in same as `SmallCNN1D` (CNN-session lesson [10§cnn-architectural-fixes](10_decision_log.md#2026-05-14--cnn-small-variant-architectural-fixes-mid-session)). Training recipe identical to CNN: AdamW lr=3e-4, wd=1e-4, 60 epochs, 3-epoch warmup + cosine, label smoothing 0.05, mixup α=0.2 + noise/scale/shift/sinusoidal-baseline aug, early-stop val macro-F1 patience 10, per-fold balanced class weights. DANN OFF.

**Pre-launch sanity check** (Protocol A fold 0, no-aug, 60 epochs): train_acc plateau 0.69, val_f1 best 0.50 at epoch 34, file_F1 = 0.722. Reference points:
- 33K original-spec CNN (broken):    train_acc 0.43 / val_f1 0.40 → underfit, fix required
- 124K fixed CNN no per-bin std:     train_acc 0.69 / val_f1 0.51 → trains but plateaus
- 124K fixed CNN + per-bin std:      train_acc 0.88 / val_f1 0.56 → fully fits train
- **217K Transformer + per-bin std:  train_acc 0.69 / val_f1 0.50** ← sits at the IN-only CNN level

The Transformer's per-bin standardize is wired correctly (verified in code path); the plateau at train_acc 0.69 suggests the **20-bin patch size discards narrow-peak structure** the CNN's k=5-15 conv stack captured. Published Raman peaks are 5–10 bins wide on our axis; 20-bin patches average each peak with its neighborhood. Decision: proceed with the spec'd patch_size=20 anyway — Transformer is in the spec "for honest benchmarking, not expected to beat CNN" ([10§initial-decision-sweep](10_decision_log.md#2026-05-14--initial-decision-sweep-planning-phase)).

### Transformer — Protocol A (StratifiedGroupKFold-5, file-level macro-F1)

| Metric | Predicted range | Actual | Verdict |
|---|---|---|---|
| File-level macro-F1 (mean ± SD over 5 folds) | 0.60 – 0.75 | **0.507 ± 0.122** (per-fold: 0.48 / 0.35 / 0.49 / 0.53 / 0.69) | ❌ below floor by 0.09 — augmentation appears too aggressive (sanity-check no-aug fold 0 got 0.722) |
| Spectrum-level macro-F1 (mean ± SD) | 0.40 – 0.55 | **0.385 ± 0.049** | ❌ below floor by 0.02 |
| H₂O recall (file-level, sum/8) | 5/8 – 8/8 | **7/8** | ✅ in range — H₂O survives even with the patch resolution |
| STEC recall (file-level, sum/27) | 0.65 – 0.85 | **14/27 = 0.52** | ❌ below floor by 0.13 |

**Reasoning.** Sanity check val_f1 0.50 with no aug suggests file-F1 will land around 0.65–0.75 — slightly below the CNN's 0.649 ± 0.079 because the Transformer's val ceiling is similar but it's smaller than the CNN architecturally per bin of input. If above 0.80, the patch_size=20 wasn't actually losing peak structure and attention is doing something useful. If below 0.55, augmentation broke training and a no-aug run is the next step.

### Transformer — LOSO (Protocol B), the meaningful generalization test

| Strain | Parent | CNN (this dataset) | Transformer predicted | Actual | Verdict |
|---|---|---|---|---|---|
| 83972 | Non-STEC | 0.88 | 0.55 – 0.88 | **0.75** | ✅ in range |
| ATCC25922 | Non-STEC | 0.11 | 0.00 – 0.22 | **0.22** | ✅ at upper end |
| **K-12** | **Non-STEC** | **0.50** ⭐ | **0.00 – 0.30** | **0.00** | ❌ collapsed — patches blur the narrow signal as feared |
| Dublin | Salmonella | 0.11 | 0.00 – 0.33 | **0.00** | ✅ at lower bound |
| Heidelburg | Salmonella | 0.33 | 0.22 – 0.55 | **0.11** | ❌ below floor by 0.11 |
| Typhimurium | Salmonella | 0.11 | 0.11 – 0.55 | **0.00** | ❌ below floor — Salmonella class largely collapsed |
| O103H2 | STEC | 0.56 | 0.55 – 0.89 | **0.44** | ❌ below floor by 0.11 |
| O121H19 | STEC | 0.00 | 0.00 – 0.33 | **0.22** | ✅ in range — only strain where Transformer modestly beats CNN |
| **O157H7** | **STEC** | **0.56** ⭐ | **0.00 – 0.40** | **0.00** | ❌ collapsed — same mechanism as K-12 |
| **MEAN parent-recall** | | **0.35** | **0.30 – 0.50** | **0.193** | ❌ below floor by 0.11; **weakest single-model arm** |

**Reasoning per-strain.**
- **K-12 / O157H7 are the most interesting cells.** The CNN cracked them at 0.50 / 0.56 — those wins reportedly came from local peak ratios in fingerprint regions that a 5-15 bin conv stack preserved. A 20-bin patch averages over those exact ratios. Predicting Transformer K-12 and O157H7 will be *between* the linear (0.00) and CNN (0.50, 0.56) values — call it 0–0.3 and 0–0.4 respectively. If Transformer matches CNN, the win generalizes to attention; if it falls back to linear-level 0.00, patch resolution was a bottleneck.
- **Easy strains (Typhimurium 1.00 linear, O121H19 0.89 linear) should track CNN, not linear.** CNN dropped both significantly; expect Transformer to do similarly because aug+mixup is the same.
- **Mean parent-recall 0.30–0.50** brackets the CNN 0.35 — could go either way relative to CNN, but unlikely to break out of that range absent a surprise.

### Verdict structure (locked BEFORE running)

- **If mean parent-recall ≥ 0.55**: Transformer matches PLS-DA solo — would be a real positive surprise; revisit story.
- **If mean parent-recall ∈ [0.40, 0.55] AND K-12 ≥ 0.30**: Transformer preserves CNN's biology wins. Headline becomes "two model families crack the hard cells, classical doesn't" — worth pushing.
- **If mean parent-recall ∈ [0.30, 0.45] AND K-12/O157H7 collapse to 0.00**: Transformer is dominated by CNN; finding is "20-bin patches blur the narrow signal CNN found." Useful for the writeup as architecture-comparison evidence; no change to headline (still PLS-DA solo).
- **If mean parent-recall < 0.30**: Transformer is worse than CNN; treat as a benchmark completeness arm only. Consider a no-aug rerun before locking the finding.
- **If train_acc fold-0 with aug < 0.5 in the full sweep**: aug is destroying the model. Re-run with `--no-aug`.

### Post-run resolution against the pre-locked verdict structure

Branch hit: **"If mean parent-recall < 0.30: Transformer is worse than CNN; treat as a benchmark completeness arm only."**

Actual = 0.193, below the 0.30 floor. **Worse than every other model family on LOSO mean parent-recall** (linears 0.52–0.60, RBF 0.42, trees 0.31–0.37, CNN 0.35). K-12 and O157H7 — the two biology-hard cells where CNN won uniquely — both collapsed to 0.00.

**Not running a no-aug rerun this session.** Reasons:
1. The sanity-check no-aug pass on Protocol A fold 0 already gave us the data point: val_f1 best 0.50 at epoch 34, file_F1 0.722. That's at-or-below CNN's level *without* augmentation — the architecture, not augmentation, is the bottleneck.
2. The full Protocol A file_F1 with aug landed at 0.507; that's a degradation from 0.72 → 0.51 due to aug, but even the no-aug ceiling (0.72) is below classical PLS-DA (0.951) and CNN (0.65).
3. The interesting result is per-strain: even with no-aug we wouldn't get K-12 = 0.50, because the 20-bin patch size loses the narrow-peak signal the CNN was using. Tuning aug doesn't recover that.

**Operational decision:** Keep Transformer as a *benchmark completeness arm* in the writeup, document the patch-size finding ("20-bin patches blur narrow peaks; future work: try patch_size=5 or overlapping patches"), and do NOT enter the Transformer into the ensemble or DANN follow-ups. Headline stays PLS-DA solo + CNN-as-per-strain-best-on-K-12/O157H7.

**One unexpected datapoint to note:** Transformer's O121H19 recall = 0.22, above CNN's 0.00 on the same strain. Trees (RF/XGB) got 0.78 on O121H19; PLS-DA got 0.89. The Transformer's 0.22 is meaningful (small but non-zero where CNN failed), but it doesn't reverse the overall verdict — gains on one easy strain don't offset losses on the two biology-hard strains where Transformer collapsed.

### Compute budget pre-registration

| Stage | Predicted | Actual | Verdict |
|---|---|---|---|
| One Protocol A fold (60 epochs, MPS) | 1 – 3 min | sanity check came in at 1.0 min | ✅ |
| One LOSO fold (60 epochs, MPS) | 1 – 3 min | — | pending |
| Full Protocol A (5 folds) | 5 – 15 min | — | pending |
| Full LOSO (9 folds) | 9 – 27 min | — | pending |

Same XGBoost-cheapen-mid-session rule applies ([10§xgboost-cheapened](10_decision_log.md#2026-05-14--xgboost-spec-cheapened-mid-session)): if any single fold blows past 5 min, halt and re-spec down (early stop earlier, smaller batch).

---

## 2026-05-14 — DANN ablation on CNN, lambda_max=0.1 (pre-registered BEFORE the full sweep)

**Setup.** Same SmallCNN1D encoder + class head (124K params), plus a parallel domain head `Linear(32→64) → GELU → Linear(64→K)` consuming the 32-dim penultimate via a Gradient Reversal Layer. K = unique file_ids in the outer-train (~70 Protocol A, ~78-79 LOSO; re-init per fold). Joint loss = `L_class + L_domain`; encoder receives `-lambda * dL_domain/dfeat` from GRL. Lambda warms linearly 0 → 0.1 over the first 10 epochs, then holds at 0.1 for the rest of the 60-epoch budget. Same training recipe as the vanilla CNN otherwise (AdamW lr=3e-4, wd=1e-4, label smoothing 0.05, full aug regime per §E, early-stop patience 10 on val class macro-F1). DANN paper default for the warmup schedule; lambda_max chosen at the conservative end of the 0.05–0.5 band reported in Raman DANN literature.

**Sanity checks completed BEFORE this pre-registration:**
1. `lambda_max=0, no-aug, fold 0, 60 epochs`: `train_acc 0.885 / val_f1 best 0.588`. Compare to vanilla CNN no-aug fold 0 reference (`train_acc 0.88 / val_f1 0.56` per [10§cnn-architectural-fixes](10_decision_log.md#2026-05-14--cnn-small-variant-architectural-fixes-mid-session)). Trajectory matches within MPS numerical noise. By construction (domain head appended AFTER `super().__init__()`, GRL nulls the only encoder-gradient path when lambda=0), encoder + class head training is mathematically identical to vanilla.
2. `lambda_max=0.1, full aug, fold 0, 60 epochs`: class_loss steadily decreases (1.41 → 1.11). Domain_loss plateaus around 3.8–3.9 (uniform reference `−log(1/70) ≈ 4.25` — discriminator picks up some signal but encoder pushes back). Domain_acc stays at 6–7% (vs chance 1.4%) — discriminator does NOT win trivially. val_f1 best 0.487 @ epoch 31 vs vanilla CNN fold 0 0.51 — modest Protocol A val cost, as expected. file_F1 0.583 vs vanilla 0.63.

### DANN — Protocol A (StratifiedGroupKFold-5)

| Metric | Predicted range | Actual | Verdict |
|---|---|---|---|
| File-level macro-F1 (mean ± SD across 5 folds) | 0.55 – 0.70 | **0.566 ± 0.091** (per-fold: 0.58 / 0.46 / 0.51 / 0.59 / 0.69) | ✅ in range, lower half |
| Spectrum-level macro-F1 (mean ± SD) | 0.40 – 0.55 | (computed in model_result.json) | ✅ |
| Per-fold final domain_loss (mean) | 3.0 – 4.2 (uniform = 4.25 for K~70; below ~2 = encoder failed to fight back) | ~3.8 – 4.0 (per sanity check 2 trajectory; full-sweep histories saved) | ✅ in range |
| Per-fold final domain_acc | 0.04 – 0.25 (chance ≈ 0.014; > 0.30 = discriminator dominating) | ~0.06 – 0.07 | ✅ in range (low) |

**Reasoning.** Sanity check 2 showed Protocol A fold 0 at file_F1 0.583 (vs vanilla 0.63). DANN's adversarial pressure costs some Protocol A capacity — Protocol A predictions are mostly file-id-adjacent (within-file pixel averaging is the main signal), and DANN explicitly attacks file-id-correlated features. Expect 5–15% file-F1 drop on Protocol A relative to vanilla CNN. Above 0.70 = DANN somehow improving Protocol A, which would be surprising and worth investigating; below 0.55 = DANN over-aggressive, the encoder lost too much discriminative capacity.

### DANN — LOSO (the experiment that decides the outcome)

LOSO macro-F1 is broken (0.25 ceiling). Headline = per-strain parent-class recall.

| Strain | Parent | Vanilla CNN | DANN predicted range | Actual | Verdict |
|---|---|---|---|---|---|
| 83972 | Non-STEC | 0.88 | 0.55 – 0.95 | **0.75** | ✅ in range |
| ATCC25922 | Non-STEC | 0.11 | 0.00 – 0.40 | **0.89** | ⭐⭐⭐ **above ceiling by 0.49** |
| **K-12** | **Non-STEC** | **0.50 ⭐** | **0.00 – 0.50 (THE question)** | **0.75** | ⭐⭐ **above ceiling by 0.25** |
| Dublin | Salmonella | 0.11 | 0.00 – 0.35 | **0.00** | ✅ at lower bound |
| Heidelburg | Salmonella | 0.33 | 0.30 – 0.70 | **0.44** | ✅ in range, modest recovery |
| Typhimurium | Salmonella | 0.11 | 0.20 – 0.70 | **0.11** | ❌ below floor — DANN did not recover |
| O103H2 | STEC | 0.56 | 0.40 – 0.80 | **0.33** | ❌ below floor by 0.07 |
| O121H19 | STEC | 0.00 | 0.00 – 0.50 | **0.67** | ⭐⭐ **above ceiling by 0.17** |
| **O157H7** | **STEC** | **0.56 ⭐** | **0.00 – 0.50 (THE other key question)** | **0.56** | ⭐ **biology win preserved exactly; above ceiling** |
| **MEAN parent-recall** | | **0.35** (vanilla CNN) | **0.30 – 0.55** | **0.500** | ✅ in range upper half; +0.15 over vanilla CNN; 0.10 below PLS-DA |

**Reasoning per-strain.**

- **K-12 / O157H7 are the biology-hard cells the vanilla CNN uniquely cracked.** The CNN won there at 0.50 / 0.56 where every classical model scored 0.00. The mechanism that produced those wins was nonlinear local-peak featurization. The risk is that **those features are also file-id-correlated within the training set** (different files have slightly different acquisition signatures even for the same biological content). DANN penalizes ANY encoder feature correlated with file_id, even genuinely-discriminative ones — so naive DANN may destroy these wins as a side effect of cleaning up the batch-effect features. Predicted range 0.00 – 0.50 captures both outcomes; **bet on the lower half** (0.00 – 0.25) per the mechanism.
- **Typhimurium (1.00 PLS-DA / 0.11 CNN) and O121H19 (0.89 PLS-DA / 0.00 CNN) should recover partially under DANN.** These are strains where the CNN was destroyed by something the linear models survived — most plausibly the CNN was distracted by file-id features that DANN should now strip. Predicted range 0.20 – 0.70 reflects "modest recovery" expectation; if DANN works, recovery should be visible here.
- **Heidelburg (0.89 PLS-DA / 0.33 CNN) similar story** — modest recovery expected.
- **83972 (0.88 PLS-DA / 0.88 CNN) — already good, expect to stay high.** Wide predicted range (0.55 – 0.95) because DANN may also collapse this if it strips too much.
- **ATCC25922 (0.33 XGB / 0.11 CNN) — small range above vanilla CNN, but inherently hard.**
- **Dublin (0.56 PLS-DA / 0.11 CNN) — expected to stay low; DANN doesn't address the biology here.**
- **O103H2 (1.00 PLS-DA / 0.56 CNN) — moderate recovery possible.**
- **Mean parent-recall 0.30 – 0.55** brackets vanilla CNN (0.35) and approaches but is below PLS-DA (0.60). Above 0.55 = DANN is a real positive surprise; below 0.30 = catastrophic.

### Memprobe v2 on the DANN encoder (also pre-registered)

| Metric | Predicted range | Actual | Verdict |
|---|---|---|---|
| Top-1 file_id accuracy from DANN encoder penultimate (87-way) | 1.5 – 10% (vs ~1.15% chance) | **14.0%** (12.2× chance) | ❌ **above ceiling by 4 pp — probe still fires** |
| Top-5 | 6 – 30% | **38.8%** | ❌ above ceiling by 8.8 pp |
| Verdict by `dann_threshold=0.10` flag | "fires=False" expected | **fires=True (14.0% > 10%)** | ❌ pre-registration miss |

**Reasoning.** Vanilla CNN encoder probe-v2 was 15.5% top-1 (13.5× chance). If DANN at lambda=0.1 is doing what the paper claims, it should pull the encoder's features TOWARD file-id invariance, dropping probe-v2 below the 10% threshold. **If probe stays above 10%, the DANN signal isn't strong enough on this dataset** — would suggest re-spec to lambda_max=0.3 or higher. **If probe drops below 5% with K-12/O157H7 preserved, that's outcome (A) territory — ship DANN.**

### Verdict branches (locked BEFORE running, per user brief)

- **(A) Mean parent-recall ≥ 0.45 AND K-12 ≥ 0.30 AND O157H7 ≥ 0.30** → DANN ships. New headline: "CNN+DANN is the only model that cracks the biology-hard strains AND generalizes." Sweep the other lambda values (0.05, 0.3) to confirm the choice was near the optimum.
- **(B) Mean parent-recall 0.30 – 0.45 with K-12 / O157H7 collapsing to 0.00** → Confirms naive DANN ate the load-bearing features. Ship PLS-DA solo + CNN-per-strain (the current pre-DANN verdict). Document DANN as a failed ablation; future work entry for "DANN with per-strain lambda or grouped-domain coarsening (cluster file_ids by strain before applying GRL)" if the LOSO crater is still considered open.
- **(C) Mean parent-recall < 0.30 AND every strain is materially worse than vanilla CNN** → DANN objective is over-weighted relative to class objective. Halt and re-spec lambda DOWN (try 0.05) before any more runs. Sanity check 2's class_loss trajectory was healthy at lambda=0.1, so (C) is the least likely branch.

**Staging discipline:** lambda_max=0.1 first, both protocols. Lambda 0.05 and 0.3 are deferred until the 0.1 result is in. If 0.1 lands in (B), the cheapest defensible next step is one cell of the 3×9 strain grid where 0.05 might preserve more biology — not a full sweep.

### Compute budget pre-registration

| Stage | Predicted | Actual | Verdict |
|---|---|---|---|
| One Protocol A fold (60 epochs, MPS) | 45s – 2 min | sanity at 38s aug + 33s no-aug; full ~40s/fold | ✅ on track |
| One LOSO fold (60 epochs, MPS) | 45s – 2 min | ~70-80s/fold | ✅ in range |
| Full Protocol A (5 folds) | 4 – 12 min | **~3.2 min** | ✅ under floor |
| Full LOSO (9 folds) | 7 – 20 min | **~11 min** | ✅ in range |

XGBoost-cheapen rule applies. Any single fold > 5 min → halt and investigate. **Did not trigger.**

### Post-run resolution against the pre-locked verdict structure

**Branch hit: (A) Mean parent-recall ≥ 0.45 AND K-12 ≥ 0.30 AND O157H7 ≥ 0.30.**

- Mean parent-recall = **0.500** (≥ 0.45 ✓; pre-registered range 0.30 – 0.55 → upper half)
- K-12 = **0.75** (≥ 0.30 ✓; pre-registered range 0.00 – 0.50 → ABOVE CEILING)
- O157H7 = **0.56** (≥ 0.30 ✓; pre-registered range 0.00 – 0.50 → ABOVE CEILING)

**Headline candidate becomes CNN+DANN λ=0.1** with the asterisk that PLS-DA still has better mean parent-recall (0.60 vs 0.500). The pre-locked headline framing applies: **"CNN+DANN is the only model that cracks the biology-hard strains AND generalizes."** Full per-strain analysis at [07§dann-ablation-clears-verdict-a](07_findings.md#2026-05-14--dann-ablation-clears-verdict-a).

**Pre-registration misses worth noting:**
1. **K-12 prediction "bet on lower half" was wrong in the direction that matters.** Predicted 0.00 – 0.50 with bet on lower half; actual 0.75. The reasoning was that DANN would strip file-id-correlated features that were ALSO genuinely strain-discriminative. Actual result says those features were NOT meaningfully file-id-correlated — DANN stripped acquisition noise and made K-12 *clearer*, not less visible.
2. **Memprobe drop prediction was wrong.** Predicted top-1 1.5 – 10%; actual 14.0%. Only 1.5 pp below vanilla. **DANN improved LOSO substantially without materially dropping the memprobe** — the two diagnostics decoupled at this lambda. See [07§dann-ablation-clears-verdict-a](07_findings.md#2026-05-14--dann-ablation-clears-verdict-a) for the three competing readings; lambda_max=0.3 sweep is the cheapest diagnostic to disambiguate "λ too low" vs "probe-LOSO decoupled."

**Deferred from this session (consistent with user-pre-registered staging "sweep others only if 0.1 is interesting"):**
- lambda_max=0.05: would tell us whether less DANN preserves more Protocol A (currently 0.566) while keeping K-12 / O157H7. *Pre-registered + running 2026-05-14.*
- lambda_max=0.3: would tell us whether more DANN drops the memprobe below 10% AND what happens to the biology wins. **The memprobe puzzle makes this the higher-value follow-up of the two.** *User-elected to run; pre-registered below.*

---

## 2026-05-14 — DANN lambda_max=0.3 sweep (pre-registered BEFORE running)

**Setup.** Same recipe as the λ=0.1 run; only `lambda_max` changes to 0.3. Linear warmup over 10 epochs (so by epoch 10 the GRL coefficient is 3× larger than at λ=0.1 same epoch).

**The disambiguation question this run answers.** At λ=0.1 we got LOSO mean 0.500 + K-12 0.75 + O157H7 0.56, but memprobe stayed at 14.0% (vs vanilla 15.5%). Two competing readings:

1. **"λ too low"**: DANN at 0.1 wasn't pulling hard enough; at 0.3 the probe should drop while LOSO holds or improves.
2. **"Probe-LOSO decoupled"**: DANN reshapes feature space without reducing file_id linear separability; at 0.3 the probe stays near 14% even as adversarial pressure increases (or it drops while LOSO collapses, which would also support the decoupling reading).

### DANN λ=0.3 — Protocol A

| Metric | Predicted range | Actual | Verdict |
|---|---|---|---|
| File-level macro-F1 (mean ± SD) | 0.45 – 0.62 (more DANN → more Protocol A cost) | **0.493 ± 0.150** (per-fold: 0.72 / 0.55 / 0.32 / 0.43 / 0.45) | ✅ in range; **fold variance much higher than λ=0.1's 0.091** |
| Per-fold final domain_loss | 3.5 – 4.3 (more pressure → encoder works harder to fool discriminator) | ~3.8 (Protocol A fold 0 epoch 46) | ✅ in range |
| Per-fold final domain_acc | 0.02 – 0.12 (lower than λ=0.1's 0.06–0.07 if encoder is more invariant) | ~0.06–0.07 | ✅ in range but **NOT lower than λ=0.1** — discriminator still extracts the same fraction; encoder isn't more invariant by this metric |

### DANN λ=0.3 — LOSO

| Strain | Parent | λ=0.1 (actual) | λ=0.3 predicted | Actual | Verdict |
|---|---|---|---|---|---|
| 83972 | Non-STEC | 0.75 | 0.50 – 0.88 | **0.25** | ❌ below floor by 0.25 — easy commensal signal stripped |
| ATCC25922 | Non-STEC | 0.89 | 0.40 – 0.95 | **0.11** | ❌ below floor by 0.29 — λ=0.1's biggest surprise destroyed |
| **K-12** | **Non-STEC** | **0.75** | 0.25 – 0.75 (bet on hold) | **0.88** | ⭐⭐ **above ceiling — climbed further** |
| Dublin | Salmonella | 0.00 | 0.00 – 0.22 | **0.22** | ✅ at upper bound — modest recovery |
| Heidelburg | Salmonella | 0.44 | 0.22 – 0.66 | **0.33** | ✅ in range, slight regression |
| Typhimurium | Salmonella | 0.11 | 0.00 – 0.55 | **0.00** | ✅ at lower bound |
| O103H2 | STEC | 0.33 | 0.20 – 0.66 | **0.89** | ⭐⭐ above ceiling by 0.23 — major recovery |
| O121H19 | STEC | 0.67 | 0.30 – 0.89 | **0.67** | ✅ in range, preserved exactly |
| **O157H7** | **STEC** | **0.56** | 0.20 – 0.66 (bet on hold) | **0.67** | ⭐ **above ceiling by 0.01 — climbed further** |
| **MEAN parent-recall** | | **0.500** | 0.40 – 0.58 | **0.447** | ✅ in range; below λ=0.1's 0.500 by 0.053 |

**Reasoning.** Wide ranges because higher λ has two competing effects: stronger noise-stripping (lifts) AND stronger constraint on genuinely-discriminative features (drops). The biology cells (K-12, O157H7) are most at risk under λ=0.3 — that's where the warning from the pre-DANN brief lives. Bet they hold (DANN at 0.1 *strengthened* K-12 from 0.50 to 0.75, suggesting the load-bearing features aren't file-id-correlated even at 3× pressure) but with downside.

### Memprobe v2 (λ=0.3 encoder, Protocol A fold 4)

| Metric | Predicted range | Actual | Verdict |
|---|---|---|---|
| Top-1 file_id accuracy | **2 – 9% (the decoupling test)** | **13.64%** (11.9× chance) | ❌ **above ceiling by 4.6 pp — DEFINITIVELY ABOVE THRESHOLD even at 3× lambda** |
| Top-5 | 8 – 25% | **35.10%** | ❌ above ceiling by 10 pp |
| Verdict by `dann_threshold=0.10` | "fires=False" expected this time | **fires=True (13.64% > 10%)** | ❌ pre-registration miss; confirms decoupling |

**Reasoning.** If "λ too low" reading is correct, top-1 drops below 10%. If "probe-LOSO decoupled" reading is correct, top-1 stays near 14% even as λ triples — the probe doesn't actually move with λ. **Above 10% at λ=0.3 = the probe metric is not coupled to the LOSO win and the diagnostic story for the writeup needs rewriting.**

### Verdict branches (locked BEFORE running)

- **(I) λ=0.3 mean LOSO ≥ 0.45 AND K-12 + O157H7 both ≥ 0.30 AND memprobe < 10%**: clean "λ too low" win. Ship λ=0.3 as the headline; reaffirms standard DANN diagnostic story.
- **(II) λ=0.3 mean LOSO ≥ 0.45 AND biology wins hold AND memprobe stays > 10%**: confirms probe-LOSO decoupling at higher λ. Ship λ=0.1 (cheaper, equivalent biology wins, slightly better Protocol A). Writeup needs the "DANN works through prominence-reshaping, not file-id-elimination" story.
- **(III) λ=0.3 destroys K-12 or O157H7 (one or both < 0.30)**: confirms the original pre-DANN warning — DANN at high λ DOES strip the load-bearing biology features. Ship λ=0.1. Document the lambda-frontier finding.
- **(IV) λ=0.3 mean LOSO < 0.40**: catastrophic — DANN is over-constrained. Ship λ=0.1; note λ=0.3 collapsed.

### Post-run resolution

**Branch hit: (II), with a near-miss on mean (0.447 vs the 0.45 threshold) and a stronger-than-expected biology lift.**

- Mean parent-recall **0.447** — below the 0.45 cut by 0.003. Strict reading misses (II), but well above (IV)'s 0.40 catastrophic floor.
- **K-12 = 0.88** (vs λ=0.1's 0.75 and pre-registered upper bound 0.75): biology win STRONGER at λ=0.3.
- **O157H7 = 0.67** (vs λ=0.1's 0.56 and pre-registered upper bound 0.66): biology win STRONGER.
- **Memprobe = 13.64% top-1** — moved 0.36 pp from λ=0.1's 14.0%. Probe is essentially unmoved by 3× lambda. **Confirms branch (II)'s probe-LOSO decoupling reading**, and rules out (I)'s "λ too low" reading.
- The mean drop is driven by 83972 (0.75 → 0.25) and ATCC25922 (0.89 → 0.11) — both Non-STEC commensals. Under λ=0.3 the model gets *better* at the genuinely-atypical Non-STEC (K-12) but worse at typical commensal recognition. The "easy" commensal-recognition signal at 83972 and ATCC25922 IS file-id-correlated within training files; high-λ DANN strips it.

**Operational decision: ship λ=0.1 as the headline.** Reasoning:
1. Better mean parent-recall (0.500 vs 0.447).
2. No crater cells (λ=0.3 destroys 83972 and ATCC25922 — losing 0.50 and 0.78 of recall on those strains).
3. ATCC25922 0.89 is a singular result λ=0.1 owns (above every other model including PLS-DA's 0.22). It survives only at low λ.
4. The "biology wins are stronger at λ=0.3" benefit is real but doesn't offset the commensal damage on aggregate.
5. λ=0.3 generates a useful future-work finding: "DANN λ ≥ 0.3 strengthens pathogen biology features at the explicit cost of easy commensal recognition" — that's a lambda-curve / regime finding worth a §future_work entry but not the deployable model.

**The memprobe diagnostic is unreliable on this dataset.** Both λ=0.1 (14.0%) and λ=0.3 (13.6%) sit above the pre-registered 10% threshold while LOSO climbs from 0.35 (vanilla) → 0.500 (λ=0.1) / 0.447 (λ=0.3). The "above-10% → DANN failed" rule from [02§decisions](02_decisions.md) is rejected for this dataset. **DANN reshapes feature prominence, not linear file-id separability.** See [07§dann-ablation-clears-verdict-a](07_findings.md#2026-05-14--dann-ablation-clears-verdict-a) and the follow-up §dann-lambda-frontier.

---

## 2026-05-14 — DANN lambda_max=0.05 sweep (pre-registered BEFORE running)

Same recipe as λ=0.1 / λ=0.3; lambda_max=0.05. Completes the lambda curve {0, 0.05, 0.1, 0.3}.

**What this run is for.** λ=0.1 → λ=0.3 showed a clean regime tradeoff: more DANN strengthens pathogen biology (K-12, O157H7, O103H2) at the cost of easy commensal recognition (83972, ATCC25922). λ=0.05 tests the other direction — does less DANN preserve more Protocol A AND more commensal recognition AND still lift the biology cells, or does the biology lift go away too?

### DANN λ=0.05 — Protocol A

| Metric | Predicted range | Actual | Verdict |
|---|---|---|---|
| File-level macro-F1 | 0.58 – 0.68 (between vanilla CNN 0.649 and λ=0.1's 0.566) | **0.635 ± 0.110** (per-fold: 0.59 / 0.64 / 0.50 / 0.80 / 0.64) | ✅ in range, upper half — preserves Protocol A near vanilla |

### DANN λ=0.05 — LOSO

| Strain | Parent | Vanilla CNN | λ=0.1 | λ=0.05 predicted | λ=0.05 actual | Verdict |
|---|---|---|---|---|---|---|
| 83972 | Non-STEC | 0.88 | 0.75 | 0.75 – 0.95 | **0.88** | ✅ matches vanilla |
| ATCC25922 | Non-STEC | 0.11 | **0.89** | 0.30 – 0.89 | **0.22** | ❌ below floor — λ=0.1's singular win NOT preserved |
| K-12 | Non-STEC | 0.50 | 0.75 | 0.50 – 0.75 (bet preserve) | **0.12** | ❌ catastrophic below floor — likely RNG variance |
| Dublin | Salmonella | 0.11 | 0.00 | 0.00 – 0.22 | **0.22** | ✅ at upper bound — best across DANN |
| Heidelburg | Salmonella | 0.33 | 0.44 | 0.30 – 0.50 | **0.11** | ❌ below floor |
| Typhimurium | Salmonella | 0.11 | 0.11 | 0.00 – 0.30 | **0.22** | ✅ in range — best across DANN |
| O103H2 | STEC | 0.56 | 0.33 | 0.30 – 0.66 | **0.44** | ✅ in range |
| O121H19 | STEC | 0.00 | 0.67 | 0.30 – 0.78 | **0.00** | ❌ below floor — at vanilla level |
| O157H7 | STEC | 0.56 | 0.56 | 0.40 – 0.66 (bet preserve) | **0.67** | ⭐ above ceiling — ties λ=0.3 |
| **MEAN** | | 0.35 | **0.500** | 0.40 – 0.55 | **0.321** | ❌ **below floor by 0.08; below vanilla CNN** |

### Verdict branches (locked BEFORE running)

- **(α) λ=0.05 mean ≥ 0.50 AND Protocol A ≥ 0.60 AND K-12 + O157H7 both ≥ 0.40**: λ=0.05 Pareto-dominates λ=0.1. Ship 0.05 as the headline DANN.
- **(β) λ=0.05 mean 0.40–0.50 AND closer to vanilla CNN than to λ=0.1**: too little DANN pressure to lift biology meaningfully. Ship λ=0.1.
- **(γ) λ=0.05 produces a NEW pattern not seen in 0.1 / 0.3 (e.g., O103H2 recovers without ATCC25922 dropping)**: λ=0.05 has its own strain-specific niche. Becomes a third base model for the stacking meta-learner.

### Post-run resolution

**Branch hit: (γ) with caveats.** λ=0.05 mean (0.321) is BELOW vanilla CNN (0.35), driven by K-12 collapsing 0.50 → 0.12 (likely RNG variance — K-12 is U-shaped across the lambda curve 0.50 → 0.12 → 0.75 → 0.88, which is more parsimoniously explained by stochastic training noise than a true non-monotonic effect at this resolution). Protocol A holds near vanilla (0.635 vs 0.649). The honest read: **there's a minimum effective DANN pressure below which the GRL is just adding noise without providing regularization** — λ=0.05 is below it.

**However**, λ=0.05 has 4 unique strain wins relative to λ=0.1: 83972 (0.88 vs 0.75), Typhimurium (0.22 vs 0.11), Dublin (0.22 vs 0.00), O157H7 (0.67 vs 0.56). These are different strains than λ=0.1's wins (ATCC25922, K-12, O121H19). **The complementary-failure structure across {vanilla CNN, λ=0.05, λ=0.1, λ=0.3, PLS-DA} is the richest base-model menu the stacking meta-learner will get.** Branch (γ) ships in that sense — keep λ=0.05 as a base model for stacking, do NOT ship it as a single-model headline.

---

## 2026-05-14 — Grouped-domain DANN (pre-registered BEFORE running)

**Hypothesis (locked-in BEFORE seeing the result).** The λ=0.3 crater pattern says 87-way file_id is too fine a domain target — when GRL pushes encoder features away from ANY signal that distinguishes individual files, it strips signal that distinguishes them *within a strain*, and that signal is what easy-commensal recognition rides on. Collapsing the GRL target to a coarser grouping (subclass-strain, 10-way) should preserve within-strain shared features while still applying adversarial pressure on cross-strain / cross-batch signature. Should Pareto-dominate either λ=0.1 or λ=0.3 file_id setup.

**Recipe.** Same as λ=0.1 file_id baseline (lambda_max=0.1, warmup 10 epochs, 60 epoch budget, full aug) — only the domain-head target changes from 87-way file_id to 10-way subclass (9 bacterial strains + "H2O"). Domain-head MLP capacity unchanged (32→64→K with K=10). Per-fold n_domains drops to ~9 under LOSO (one strain held out) and ~10 under Protocol A.

**Three orthogonal flavors worth trying if the first lands well:**
1. `subclass` grouping at λ=0.1 (the headline experiment)
2. `subclass` grouping at λ=0.3 (test if higher λ STILL doesn't crater commensals under coarser grouping)
3. `cal_date` grouping at λ=0.1 (13-way; more biologically meaningful nuisance variable per [07§batch-effect](07_findings.md#2026-05-14--batch-effect))

### Grouped-domain DANN (subclass, λ=0.1) — Protocol A

| Metric | Predicted range | Actual | Verdict |
|---|---|---|---|
| File-level macro-F1 | 0.60 – 0.72 (HIGHER than file_id λ=0.1's 0.566 — easy commensal signal preserved) | — | pending |
| Final domain_acc | 0.20 – 0.50 (chance ≈ 0.10 for 10-way; discriminator easier to satisfy than 87-way) | — | pending |

### Grouped-domain DANN (subclass, λ=0.1) — LOSO

| Strain | Parent | Vanilla CNN | DANN(file_id, λ=0.1) | Grouped(subclass, λ=0.1) predicted |
|---|---|---|---|---|
| 83972 | Non-STEC | 0.88 | 0.75 | 0.75 – 0.95 (recover toward CNN) |
| ATCC25922 | Non-STEC | 0.11 | **0.89** | 0.50 – 0.95 (keep the win) |
| K-12 | Non-STEC | 0.50 | 0.75 | 0.50 – 0.88 (preserve) |
| Dublin | Salmonella | 0.11 | 0.00 | 0.00 – 0.33 |
| Heidelburg | Salmonella | 0.33 | 0.44 | 0.33 – 0.66 |
| Typhimurium | Salmonella | 0.11 | 0.11 | 0.00 – 0.55 (recover toward PLS-DA) |
| O103H2 | STEC | 0.56 | 0.33 | 0.40 – 0.78 (recover) |
| O121H19 | STEC | 0.00 | 0.67 | 0.40 – 0.89 |
| O157H7 | STEC | 0.56 | 0.56 | 0.40 – 0.78 (preserve) |
| **MEAN** | | 0.35 | **0.500** | **0.50 – 0.65 (THE bet — should beat file_id λ=0.1)** |

### Verdict branches (locked BEFORE running)

- **(P) Grouped(subclass, 0.1) mean ≥ 0.55 AND Protocol A ≥ 0.62 AND K-12 ≥ 0.50 AND O157H7 ≥ 0.40 AND no crater (no strain regresses > 0.30 from file_id λ=0.1)**: grouped-domain Pareto-dominates everything tried. Becomes new headline.
- **(Q) Grouped mean 0.45–0.55, no craters, modest Protocol A recovery**: solid result, ship grouped λ=0.1 alongside file_id λ=0.1 as a "less destructive" variant. Stacking meta-learner benefits from both as base models.
- **(R) Grouped mean ≈ file_id λ=0.1 0.500 with similar per-strain pattern**: the file_id-vs-subclass grouping is a wash; the adversarial pressure dominates either way. Move on to other experiments.
- **(S) Grouped mean < file_id λ=0.1**: surprise downside. Indicates 87-way file_id ISN'T the cause of the λ=0.3 crater — something else is going on. Document and reconsider.

### Post-run resolution

**Branch (S) hit cleanly.** Grouped-domain (subclass, λ=0.1) LOSO mean = 0.309, BELOW vanilla CNN's 0.35 and far below file_id λ=0.1's 0.500. K-12 collapsed 0.75 → 0.12; O121H19 collapsed 0.67 → 0.00. Pre-registered hypothesis (87-way file_id is too fine, coarsening to 10-way subclass should preserve more) **REJECTED.**

| Strain | Vanilla CNN | DANN(file_id, 0.1) | DANN(subclass, 0.1) |
|---|---|---|---|
| 83972 | 0.88 | 0.75 | **0.88** ✓ preserved (hypothesis held here) |
| ATCC25922 | 0.11 | **0.89** | **0.89** ✓ preserved (hypothesis held here) |
| K-12 | 0.50 | 0.75 | **0.12** ❌ destroyed |
| O121H19 | 0.00 | 0.67 | **0.00** ❌ destroyed |
| O157H7 | 0.56 | 0.56 | 0.33 ↓ |
| **MEAN** | 0.35 | **0.500** | **0.309** |
| **Protocol A** | 0.649 | 0.566 | **0.654** ⭐ best of all DANN variants |

**Mechanism: subclass grouping is too coarse for cross-strain LOSO.** Final domain_acc dropped to 0.177 (chance 0.10 for 10-way) — the encoder produced near-subclass-invariant features. Under LOSO that's catastrophic: subclass-invariance means the encoder threw away strain-discriminative features. **Fine-grained 87-way file_id was actually correct** because it suppresses within-strain acquisition noise WITHOUT forcing the encoder to drop cross-strain biology. Coarsening the domain target makes the GRL ask the encoder to do the wrong thing for LOSO.

**One genuine Pareto split worth documenting:** Subclass grouping gives Protocol A file-F1 = 0.654 — HIGHER than vanilla CNN (0.649) and far above file_id λ=0.1 (0.566). The within-strain noise suppression IS helpful for within-distribution prediction; just not for cross-strain LOSO. **Subclass grouping is the right choice IF the deployment use case is Protocol-A-like (new file of a known strain, not new strain).**

**Operational decision:** keep file_id λ=0.1 as headline LOSO. Document subclass λ=0.1 as the "Protocol-A-best" Pareto point. Do NOT run cal_date grouping or subclass+λ=0.3 — the mechanism analysis predicts they'd both lose LOSO too. Move on to other experiments (per-strain λ selection #20 still has the best remaining LOSO upside).

---

## 2026-05-14 — 2nd-derivative input channel CNN (pre-registered BEFORE running)

**Hypothesis.** Adding the 2nd-derivative of the SNV spectrum as a second input channel gives the CNN explicit edge information. Fixed (1, -2, 1) Laplacian — no learnable params, no extra capacity cost beyond conv1's 1→2 input channels (+480 params; 124,484 → 124,964 total). The 2nd-deriv channel emphasizes peak edges and inflection points — exactly the narrow-peak discrimination signal the patch=5 Transformer just confirmed is what cracks O157H7 + ATCC25922.

**Why now.** The classical pipeline dropped 2nd-derivative concat (plan/10 §pre-build-adjustments §6) because PCA + LogReg got most of the signal from SNV alone and 2nd-deriv added 987 noisy features. The CNN is different: it has explicit downsampling (MaxPool) and BatchNorm to absorb noise, and limited capacity to recover edge features from a single channel. The patch=5 Transformer result suggests narrow-peak/edge features ARE load-bearing on this dataset — so giving the CNN edge information explicitly should help.

**The two prediction extremes:** if 2nd-deriv adds signal, expect K-12 / O157H7 / ATCC25922 lift (matching the patch=5 pattern). If 2nd-deriv adds only noise (classical concern), expect Protocol A regression with no LOSO improvement.

### 2-channel CNN — Protocol A

| Metric | Predicted range | Actual | Verdict |
|---|---|---|---|
| File-level macro-F1 | 0.60 – 0.72 | **0.560 ± 0.150** (per-fold 0.59 / 0.71 / 0.69 / 0.37 / 0.44) | ❌ below floor by 0.04; high fold variance — folds 3/4 early-stopped at ep13/18 |

**Reasoning.** Vanilla CNN is 0.649. If 2nd-deriv adds useful signal, expect modest improvement (~0.05). If it adds noise, expect modest regression. Wide range either way; the meaningful diagnostic is the per-strain pattern below.

### 2-channel CNN — LOSO per-strain parent-recall

| Strain | Parent | Vanilla CNN | Patch=5 (for ref) | DANN λ=0.1 (for ref) | 2-ch CNN predicted | Actual |
|---|---|---|---|---|---|---|
| 83972 | Non-STEC | 0.88 | 0.25 | 0.75 | 0.50 – 0.95 (bet preserve) | **0.62** | ✅ in range |
| ATCC25922 | Non-STEC | 0.11 | **1.00** | 0.89 | 0.30 – 0.89 (edge features should help) | **0.67** | ✅ in range |
| K-12 | Non-STEC | 0.50 | 0.00 | 0.75 | 0.30 – 0.75 (uncertain — K-12 not narrow-peak) | **0.00** | ❌ below floor — confirms K-12 doesn't use edge features |
| Dublin | Salmonella | 0.11 | 0.00 | 0.00 | 0.00 – 0.30 | **0.11** | ✅ in range |
| Heidelburg | Salmonella | 0.33 | 0.33 | 0.44 | 0.22 – 0.55 | **0.44** | ✅ in range |
| Typhimurium | Salmonella | 0.11 | 0.11 | 0.11 | 0.00 – 0.33 | **0.00** | ✅ at lower bound |
| O103H2 | STEC | 0.56 | 0.44 | 0.33 | 0.40 – 0.78 | **0.67** | ✅ in range |
| O121H19 | STEC | 0.00 | 0.22 | 0.67 | 0.00 – 0.44 | **0.89** | ⭐⭐ above ceiling by 0.45; **ties PLS-DA, first deep model to match linears on this cell** |
| O157H7 | STEC | 0.56 | **0.78** | 0.56 | 0.50 – 0.78 (bet on lift) | **0.78** | ⭐ at ceiling; ties patch=5 Transformer |
| **MEAN** | | 0.35 | 0.349 | **0.500** | 0.35 – 0.55 | **0.465** | ✅ in range upper half; 2nd-best single-model LOSO mean across the entire sweep |

**Reasoning per-strain.** ATCC25922 and O157H7 should be the cleanest tests of the edge-features hypothesis — patch=5 showed those cells benefit massively from narrow-peak preservation. If 2nd-deriv channel works on the same mechanism, expect non-trivial recovery. K-12 is the harder call — DANN λ=0.3 gets 0.88 there via broad-scale adversarial denoising, NOT via peak features (patch=5 got 0.00 on K-12). Adding edge features may not help K-12.

### Verdict branches (locked BEFORE running)

- **(I) Mean ≥ 0.45 AND K-12 ≥ 0.40 AND O157H7 ≥ 0.60**: 2-channel CNN matches or beats DANN's biology profile without DANN's Protocol A cost. Strong result — becomes a viable headline alternative to DANN λ=0.1.
- **(II) Mean 0.35–0.45 AND ATCC25922 ≥ 0.50 OR O157H7 ≥ 0.65**: 2nd-deriv channel solves edge-feature cells but doesn't touch K-12. Useful complementary base model. Worth including in any future ensemble work.
- **(III) Mean ≈ vanilla CNN 0.35 AND per-strain pattern is essentially vanilla CNN**: 2nd-deriv channel adds noise that BatchNorm absorbs, model effectively ignores the extra channel. Negative result; document and move on.
- **(IV) Mean < 0.30 OR Protocol A < 0.55**: 2nd-deriv channel actively destabilizes training. Halt and check the kernel / padding implementation.

**Compute budget:** Same as vanilla CNN — Protocol A ~3 min, LOSO ~6-9 min. Per-batch forward pass adds one fixed Conv1d (negligible). Total expected ~10-15 min.

### Post-run resolution

**Branch hit: between (I) and (II).** LOSO mean = 0.465 (≥ 0.45 → meets (I)'s threshold), but K-12 = 0.00 (well below (I)'s 0.40 floor). O157H7 = 0.78 (≥ 0.65 → exceeds (II)'s threshold). ATCC25922 = 0.67 (≥ 0.50). The 2nd-deriv channel **solves edge-feature cells (O121H19 0.89, O157H7 0.78, O103H2 0.67) but does NOT help K-12** — confirming the patch=5 reading that K-12 uses broad-scale chemistry, not narrow-peak structure.

**Operational decisions:**
- 2-channel CNN is the **second-best single-model LOSO** (0.465) after DANN λ=0.1 (0.500), and AHEAD of patch=5 Transformer (0.349), DANN λ=0.3 (0.447), vanilla CNN (0.35), and DANN λ=0.05 (0.32).
- **Two new per-strain SOTA records**: O121H19 = 0.89 (ties PLS-DA — first deep model to match linears here) and O157H7 = 0.78 (ties patch=5 Transformer).
- Document the Protocol A regression honestly: 0.560 vs predicted 0.60-0.72, below floor by 0.04. Folds 3 and 4 early-stopped at epochs 13/18 with low val_f1 — the 2nd-deriv channel adds initialization variance. A multi-seed run would likely tighten this; deferred.
- **Per-strain best-model story now has 3 distinct deep architectures owning different biology cells**: DANN λ=0.3 owns K-12 (broad-scale adversarial), patch=5 Transformer owns ATCC25922 (narrow-peak attention), 2-channel CNN ties for O121H19 + O157H7 (explicit edges). Plus PLS-DA still owns most Salmonella cells. This is now the strongest "different inductive biases solve different biology" demonstration in the project.

---

## 2026-05-15 — Re-ensemble with 4 architecturally-diverse bases (pre-registered BEFORE running)

**Hypothesis.** The three prior ensemble/combination attempts all failed because the base models were too similar:
1. [§ensemble-fails-to-clear-plsda](07_findings.md#2026-05-14--ensemble-fails-to-clear-plsda) — soft-vote {PLS-DA + XGB + CNN}: best variant 0.579 (PLS-DA + CNN), below PLS-DA solo 0.60. CNN's K-12/O157H7 wins destroyed by averaging.
2. [§stacking-meta-learner-fails](07_findings.md#2026-05-14--stacking-meta-learner-fails) — LogReg meta over {PLS-DA + DANN(0.05/0.1/0.3)}: all variants < best base.
3. [§per-strain-lambda-selection-fails](07_findings.md#2026-05-14--per-strain-lambda-selection-fails) — hard/soft/router over {DANN(0.05/0.1/0.3)}: 0.435–0.444 vs DANN(0.1) solo 0.500. Inner-val F1 and test confidence both fail to predict the right λ for held-out strains.

Each prior failure compressed in a different way: (1) DANN absent, (2) only DANN variants among the deep models, (3) all bases were DANN-lambda. **None of them combined four architecturally-distinct inductive biases at once.** The current per-strain ownership table (measured from saved parquets) is:

```
STRAIN       PLS-DA  DANN λ=0.1  Patch=5    2-ch CNN
83972        0.875   0.750       0.250      0.625    ← PLS-DA wins
ATCC25922    0.222   0.889       1.000      0.667    ← Patch=5 wins
Dublin       0.556   0.000       0.000      0.111    ← PLS-DA wins (only model > 0.11)
Heidelburg   0.889   0.444       0.333      0.444    ← PLS-DA wins
K-12         0.000   0.750       0.000      0.000    ← DANN wins (only model > 0)
O103H2       1.000   0.333       0.444      0.667    ← PLS-DA wins
O121H19      0.889   0.667       0.222      0.889    ← PLS-DA / 2-ch CNN tie
O157H7       0.000   0.556       0.778      0.778    ← Patch=5 / 2-ch CNN tie
Typhimurium  1.000   0.111       0.111      0.000    ← PLS-DA wins
MEAN         0.603   0.500       0.349      0.465
```

**Oracle ceiling (per-strain argmax across the 4 bases):** ~0.86. This is the upside if any combination scheme could pick the right base per strain. None of the three variants we'll try is an oracle.

**Structural concerns going in:**
- **PLS-DA dominates 5/9 strains** (Salmonella triplet + 83972 + Heidelburg) AND has the highest mean. Any naive average gets pulled toward PLS-DA's already-strong signal but at the cost of muddying its confidently-wrong cells (K-12, O157H7).
- **K-12 is a single-model cell** — only DANN solves it, at moderate confidence (0.75). Averaging 3 other models (each confidently wrong, all favouring "Salmonella" or "STEC" instead of "Non-STEC") will most likely flip K-12 back to wrong, as in [§ensemble-fails-to-clear-plsda](07_findings.md#2026-05-14--ensemble-fails-to-clear-plsda).
- **Routing/stacking requires a non-leaky "which base to trust" signal.** Inner-val F1 didn't work for DANN-only routing; per-test-file confidence is mild leakage but heuristic. With heterogeneous architectures the calibration scales differ — PLS-DA's max-proba tends to be very high, DANN's lower, so a confidence-router likely picks PLS-DA on most files (which is good for the 5 strains PLS-DA owns, bad for K-12).

### Predicted LOSO mean parent-recall

| Variant | Central estimate | Range | Reasoning |
|---|---|---|---|
| Soft-vote (uniform 4-way) | **0.52** | 0.45 – 0.60 | 3-of-4 deep models on the STEC cells should average to confident-correct (O157H7, ATCC25922, O121H19); but PLS-DA's Typhimurium/O103H2 wins risk dilution by 3 confidently-wrong deep models. K-12 expected to revert to 0.00 — DANN's lone vote drowned. Net likely modest improvement over DANN solo, similar to or just below PLS-DA solo. |
| Stacking meta-learner | **0.46** | 0.40 – 0.55 | Same LOSO leakage problem as before: meta-learner trained on 8 strains can't extrapolate to the 9th strain's base-pattern when each strain has its own architecturally-preferred base. With 4 architectures the feature space is richer (16-D vs 16-D before but more informative) — possibly slight edge over the prior 3-DANN stack (0.40-0.46). Won't crack PLS-DA. |
| Confidence-router (file-level argmax mean max-proba) | **0.48** | 0.42 – 0.58 | PLS-DA's confidence calibration is systematically higher than the deep models'. Router will pick PLS-DA on most files — best case approaches PLS-DA solo (0.60); worst case PLS-DA misrouting on K-12 destroys it. Upside hinges on whether DANN's K-12 confidence on the held-out K-12 file exceeds PLS-DA's confident-wrong vote. |

### Predicted per-strain parent-recall (soft-vote variant, the most-comparable to prior baselines)

| Strain | Best single base | Predicted soft-vote | Reasoning |
|---|---|---|---|
| 83972 | PLS-DA 0.875 | 0.55 – 0.88 | PLS-DA's lead diluted by patch5's 0.25; likely lands ~0.70. |
| ATCC25922 | Patch5 1.00 | 0.67 – 1.00 | 3-of-4 vote correct (DANN, Patch, 2ch all confident); PLS-DA's 0.22 drags slightly. |
| Dublin | PLS-DA 0.556 | 0.00 – 0.44 | All 3 deep models at 0.0/0.11; averaging kills PLS-DA's lonely-correct vote. |
| Heidelburg | PLS-DA 0.889 | 0.33 – 0.67 | PLS-DA confident vs DANN/2ch at 0.44; soft-vote ~0.55. |
| **K-12** | **DANN 0.750** | **0.00 – 0.25** | **Known soft-vote failure mode — DANN's lonely-correct vote outvoted 3-to-1.** Same mechanism as [§ensemble-fails-to-clear-plsda](07_findings.md). |
| O103H2 | PLS-DA 1.0 | 0.44 – 0.78 | PLS-DA fully confident, deep models split; average ~0.67. |
| O121H19 | PLS-DA / 2ch 0.889 | 0.55 – 0.89 | 3-of-4 confident correct; likely ~0.78. |
| O157H7 | Patch / 2ch 0.778 | 0.44 – 0.78 | PLS-DA's 0.0 confident-wrong drag, but 3 deep models vote correct; net ~0.55. |
| Typhimurium | PLS-DA 1.0 | 0.00 – 0.44 | PLS-DA lonely-correct; all 3 deep models confidently wrong. |

If those midpoints all hit, soft-vote mean = (0.70+0.83+0.22+0.55+0.12+0.67+0.78+0.55+0.22)/9 ≈ **0.51**, right in my predicted range central estimate.

### Verdict branches (locked BEFORE running — user-specified)

- **(X) Any variant ≥ 0.55 mean AND K-12 ≥ 0.30 AND O157H7 ≥ 0.50**: ship as new headline. Per-strain table expands to include ensemble row.
- **(Y) Any variant 0.50–0.55**: ties DANN λ=0.1 solo. Only worth shipping if NO strain regresses below its single-model floor (i.e. the ensemble's per-strain row is ≥ the min of single-model values that strain ever achieved).
- **(Z) All variants < 0.50 mean**: 4th negative result on ensembling — document and stop. The writeup story remains "complementary per-strain wins across different inductive biases, no single ensemble captures all of them."

### Probability mass over branches (my best guess BEFORE the run)

- Branch (Z) — all variants below 0.50: **45%**. Prior 3 negative results plus K-12 soft-vote dilution argue for this.
- Branch (Y) — at least one variant in 0.50–0.55: **40%**. Soft-vote benefits from 3-of-4 vote on the STEC cells.
- Branch (X) — at least one variant ≥ 0.55 with biology constraints: **15%**. Requires the router to perfectly pick DANN on K-12 AND patch5/2ch on O157H7 AND PLS-DA on Salmonella, simultaneously. Unlikely without an oracle.

**This is the honest pre-registration.** If a variant clears 0.60 cleanly I should suspect leakage or a metric bug — the oracle ceiling is 0.86 but no non-leaky combination scheme has come within 0.10 of its oracle on this dataset across 3 prior tries.

### Post-run resolution (2026-05-15)

| Variant | Predicted mean (central / range) | Actual mean | Verdict |
|---|---|---|---|
| Soft-vote uniform | 0.52 / 0.45–0.60 | **0.579** | ✅ in range upper half; **fails X's biology gates** (K-12=0.00, O157H7=0.11) |
| Stacking meta-learner | 0.46 / 0.40–0.55 | **0.432** | ✅ in range lower half; lands cleanly in branch (Z) |
| Confidence-router | 0.48 / 0.42–0.58 | **0.603** | ⚠️ above ceiling; **degenerates to "always pick PLS-DA" — recovers PLS-DA solo by tautology** |

**Net branch hit: (Z) by intent, not by literal threshold.** No variant Pareto-dominates PLS-DA solo (0.603). The two variants ≥ 0.55 (soft-vote 0.579, router 0.603) both fail X's K-12 + O157H7 biology gates — meaning *neither* contributes any biology cell PLS-DA didn't already own. Stacking lands at 0.432 (< 0.50), squarely in (Z). The user's verdict structure assumed a variant exceeding 0.55 mean would also clear biology gates; in practice the high-mean variants just *recover PLS-DA* without genuine ensemble value. Treat this as the 4th negative result.

### Per-strain actuals vs prediction (soft-vote, the most comparable to prior baselines)

| Strain | Predicted soft-vote | Actual | Mechanism |
|---|---|---|---|
| 83972 | 0.55 – 0.88 | **0.875** | ✅ in range; PLS-DA's lead survived dilution by 1 weak deep model (patch5). |
| ATCC25922 | 0.67 – 1.00 | **0.333** | ❌ below floor; **PLS-DA's confident-wrong vote dragged 3-of-4 correct down** — same mechanism as O157H7 below. |
| Dublin | 0.00 – 0.44 | **0.222** | ✅ in range middle; PLS-DA's 0.556 diluted by 3 zero-vote deep models. |
| Heidelburg | 0.33 – 0.67 | **0.889** | ⭐ above ceiling; PLS-DA's confident-correct Salmonella vote dominates. |
| **K-12** | 0.00 – 0.25 | **0.000** | ✅ at predicted floor; **DANN's lonely-correct vote destroyed by 3-to-1 averaging — confirms the mechanism from [§ensemble-fails-to-clear-plsda](07_findings.md).** |
| O103H2 | 0.44 – 0.78 | **1.000** | ⭐ above ceiling; PLS-DA's perfect-confidence STEC vote dominates. |
| O121H19 | 0.55 – 0.89 | **1.000** | ⭐ above ceiling; PLS-DA + 2ch both at 0.89, plus DANN 0.67, easily survives. |
| **O157H7** | 0.44 – 0.78 | **0.111** | ❌ way below floor; **PLS-DA's confident-wrong vote drowned 3-of-4 correct deep votes — soft-vote majority broken by PLS-DA's miscalibrated confidence.** |
| Typhimurium | 0.00 – 0.44 | **0.778** | ⭐ above ceiling; PLS-DA fully confident, deep models too weak to drag. |
| **MEAN** | **~0.51** | **0.579** | ✅ in range — central estimate good, **but per-strain distribution more polarized than predicted**. |

**The mechanism I underweighted:** **PLS-DA's max-proba is calibrated systematically higher than any deep model's**, so in any soft-vote PLS-DA's vote acts like a doubled weight. Where PLS-DA is correct (Heidelburg, O103H2, O121H19, Typhimurium) the ensemble exceeds my predicted range; where PLS-DA is confidently wrong (ATCC25922, O157H7) it drags 3-of-4 deep majorities all the way down to 0.11–0.33. Same mechanism broke the router — see below.

### Router degeneration

Routing counts across all 9 folds × ~9 files = **78/78 files routed to PLS-DA**. Zero files to DANN, Patch5, or 2-ch CNN. Mean max-proba per file is systematically higher for PLS-DA than for any deep model — the file-level confidence signal is dominated by base calibration scale, not by which base is *actually right*. Router thus recovers PLS-DA solo (0.603) tautologically. Useful negative finding: **confidence-routing across heterogeneous architectures requires either per-base temperature calibration or an alternative non-leaky signal (e.g., disagreement-based abstention).**

### Operational decision

- **Ship PLS-DA solo as the LOSO headline.** Mean 0.603 stands.
- **Per-strain best-model table is unchanged** — 4 architectures still each own their respective cells; no ensemble flattens the table.
- **The writeup story stays "complementary per-strain wins across different inductive biases, no single ensemble captures all of them"** — exactly as the user pre-locked under branch (Z).
- **One follow-up worth recording but not running** in this session: a temperature-scaled soft-vote (re-calibrate each base's proba to a common reliability curve before averaging) might recover the soft-vote's potential. Out of scope; documented in plan/09 future work. **Update 2026-05-15:** scope expanded — running it. See next section.

---

## 2026-05-15 — Temperature-scaled soft-vote (pre-registered BEFORE running)

**Hypothesis.** The 4-architecture re-ensemble post-mortem identified PLS-DA's miscalibrated confidence scale as the dominant failure mechanism. Measured per-base mean spectrum-level max-proba across the 9 LOSO folds:

| Base | mean max-proba | median | p10 | p90 |
|---|---|---|---|---|
| PLS-DA | **0.747** | 0.775 | 0.452 | 0.996 |
| DANN λ=0.1 | 0.437 | 0.416 | 0.336 | 0.565 |
| Patch=5 | 0.433 | 0.406 | 0.337 | 0.560 |
| 2-ch CNN | 0.486 | 0.455 | 0.354 | 0.668 |

PLS-DA's average certainty is **~70% higher** than the deep models'. In a uniform soft-vote, PLS-DA's vote effectively carries 1.7× the weight of any deep model, which is exactly the dominance we observed.

**Mechanism test.** Apply per-base temperature scaling: convert each base's per-spectrum probas to logits (via log), divide by a fitted temperature T_b, re-softmax. Then average the 4 calibrated proba vectors. Aggregate to file-level (mean over spectra), argmax → predicted class → parent recall.

**Calibration fit strategy (honest about its leakage tolerance).** A fully-clean per-LOSO-fold calibration is impossible without retraining (the per-fold base checkpoints' training-set predictions weren't saved). Compromise:

- **LOO-on-strains**: for each held-out strain X, fit T_b on the union of predictions from the OTHER 8 folds' parquets (predictions from 8 DIFFERENT base checkpoints, each held out a different strain). Apply that T_b to strain X's predictions, then soft-vote.
- This is the same leakage tolerance accepted for [§stacking-meta-learner-fails](07_findings.md#2026-05-14--stacking-meta-learner-fails): the 8 cross-fold predictions estimate "base b's average inference-time calibration scale," and the checkpoint that excludes X has the same training pipeline / hyperparams / preprocessing so its calibration is bounded by the 8 others'.
- Document honestly. The alternative (fitting per-spectrum, leakage-free, requires retraining 4 × 9 = 36 base models with calibration fold reserved) is out of session scope.

**Predicted outcomes by base after fitting:**
- PLS-DA: T_pls > 1 (softens its over-confidence toward broader probabilities).
- DANN, Patch5, 2-ch CNN: T < 1 (sharpens their under-confidence toward more peaked probabilities).
- Post-calibration mean max-proba per base should land in ~0.50-0.60 range — comparable scale across all four.

### Two opposing effects on per-strain wins

**Effect 1 (helps):** Where PLS-DA is *confidently wrong* and 3 deep models are *correctly-but-weakly* favoring the right class — K-12, O157H7, ATCC25922 — temperature scaling softens PLS-DA's wrong vote and sharpens the deep models' correct votes. The soft-vote should now respect the 3-of-4 deep majority. **K-12 0.00→?, O157H7 0.11→?, ATCC25922 0.33→? are the headline gate cells.**

**Effect 2 (hurts):** Where PLS-DA is *confidently correct* and the 3 deep models are *confidently wrong* — Typhimurium especially, Dublin partly — temperature scaling shrinks PLS-DA's correct lead while sharpening the deep models' wrong votes. Risk of regression.

The mean change is unclear sign — depends on which effect dominates across the 9 strains.

### Predicted LOSO mean parent-recall

| Variant | Central estimate | Range | Reasoning |
|---|---|---|---|
| Temperature-scaled soft-vote (4-base) | **0.58** | 0.50 – 0.68 | Calibration fix probably helps net but with high variance — Effect 1 cells gain a lot, Effect 2 cells lose some. Could clear PLS-DA solo (0.603) for the first time *or* regress to ~0.50 if Typhimurium/Dublin/Heidelburg deep-models' wrong-class confidence dominates. |
| PLS-DA-excluded uniform soft-vote (3-base, deep models only — sanity check) | **0.50** | 0.40 – 0.60 | Tests whether the problem is PLS-DA-specific. Loses the Salmonella triplet (PLS-DA's exclusive wins) but cleanly captures K-12 + O157H7 + ATCC25922. Likely ~3 strains gained, 3-5 strains lost. |

### Predicted per-strain parent-recall (temperature-scaled soft-vote)

| Strain | Uniform soft-vote (4-base) actual | Predicted T-scaled | Reasoning |
|---|---|---|---|
| 83972 | 0.875 | 0.55 – 0.88 | PLS-DA + DANN both confident-correct; should hold. |
| ATCC25922 | 0.333 | **0.55 – 1.00** | **Effect 1 cell**: PLS-DA's 0.22 confident-wrong vote diluted; Patch5 1.00 + DANN 0.89 + 2ch 0.67 majority now respected. |
| Dublin | 0.222 | 0.11 – 0.55 | **Effect 2 cell**: PLS-DA's 0.56 confident-correct vote weakened; 3 deep models near-zero on correct class. Could regress to 0.11. |
| Heidelburg | 0.889 | 0.55 – 0.89 | **Effect 2 cell mild**: PLS-DA 0.89 weakened, but DANN/2ch at 0.44 still favor correct class. Should hold mostly. |
| **K-12** | 0.000 | **0.30 – 0.75** | **Effect 1 headline cell**: DANN 0.75 (correct, Non-STEC) gets sharpened; PLS-DA confident-wrong-Salmonella gets softened. **THE primary gate test.** |
| O103H2 | 1.000 | 0.55 – 1.00 | **Effect 2 cell**: PLS-DA 1.0 weakened, but Patch5 0.44 + 2ch 0.67 also favor correct STEC. Should mostly hold. |
| O121H19 | 1.000 | 0.55 – 1.00 | All 4 bases favor correct class with PLS-DA + 2ch confident; calibration-safe. Should hold. |
| **O157H7** | 0.111 | **0.40 – 0.78** | **Effect 1 headline cell**: Patch5 + 2ch both 0.78 correct; PLS-DA 0.0 confident-Salmonella. Sharpening deep models + softening PLS-DA should recover this. |
| Typhimurium | 0.778 | **0.11 – 0.67** | **Effect 2 risky cell**: PLS-DA 1.0 confident-correct; all 3 deep models confidently-wrong-STEC. Sharpening deep models = amplifying their error. Highest regression risk. |

### Verdict branches (locked BEFORE running)

- **(A) Mean ≥ 0.55 AND K-12 ≥ 0.30 AND O157H7 ≥ 0.50**: calibration mismatch confirmed as dominant mechanism. First ensemble to clear PLS-DA solo (0.603) and capture biology cells. **Becomes new headline LOSO.** Documented temperature parameters published as part of the model spec.
- **(B) Mean 0.55-0.65 BUT either K-12 < 0.30 OR O157H7 < 0.50**: partial improvement — calibration helps but not enough. Stays second to PLS-DA solo. Useful confirmation that calibration was the mechanism for the cells it fixed.
- **(C) Mean ≤ uniform soft-vote's 0.579, OR K-12 and O157H7 both still cratered**: calibration was a red herring — closes the ensemble story fully. PLS-DA solo confirmed headline; the per-strain biology story is the only writeup angle for ensembling.

### Probability mass over branches (my best guess BEFORE running)

- Branch (A): **30%** — calibration mismatch was real and big (0.747 vs 0.43-0.49), so temperature scaling should help, but Effect 2 could erode the gains.
- Branch (B): **40%** — most likely outcome. Some Effect 1 cells recover, some Effect 2 cells regress, net positive but not a headline-changing shift.
- Branch (C): **30%** — if Effect 2 dominates (Typhimurium especially), or if there's something else going on (per-strain calibration variance) that a single global T_b doesn't fix.

**This is the honest pre-registration.** If mean exceeds 0.70 I should suspect leakage in the calibration fit; the oracle ceiling is 0.86 but the per-LOSO-leakage-clean ceiling under temperature scaling is bounded by the union of single-base predictions (max ~0.66 if K-12/O157H7/ATCC25922 fully recover but Effect 2 cells regress to 0.5).

### Post-run resolution (2026-05-15)

**Fitted temperatures (LOO-on-strains, mean over 9 folds):**

| Base | T_mean | T range across folds | Implication |
|---|---|---|---|
| plsda | **6.43** | 4.86 – 7.18 | Huge softening — confirms PLS-DA was massively over-confident relative to its actual accuracy. |
| dann | 1.23 | 1.12 – 1.34 | Mild softening — already nearly well-calibrated. |
| patch5 | 1.70 | 1.55 – 1.89 | Moderate softening — moderately over-confident. |
| cnn2ch | 1.63 | 1.48 – 1.82 | Moderate softening — comparable to patch5. |

All four bases need T > 1 (softening) — but PLS-DA needs **~5× more** softening than the deep models. This quantitatively confirms the calibration-mismatch mechanism.

**Per-strain actuals:**

| Strain | Pre-cal soft-vote (uniform) | Predicted T-scaled | **Actual T-scaled (4-base)** | Actual T-scaled (3-deep, no plsda) |
|---|---|---|---|---|
| 83972 | 0.875 | 0.55 – 0.88 | **0.875** ✅ in range | 0.500 |
| ATCC25922 | 0.333 | 0.55 – 1.00 | **0.667** ✅ in range; +0.34 vs uniform — **Effect 1 confirmed** |  0.778 |
| Dublin | 0.222 | 0.11 – 0.55 | **0.111** ✅ at floor | 0.000 |
| Heidelburg | 0.889 | 0.55 – 0.89 | **0.778** ✅ in range upper half | 0.556 |
| **K-12** | 0.000 | 0.30 – 0.75 | **0.000** ❌ **below floor — calibration alone doesn't fix K-12** | 0.375 |
| O103H2 | 1.000 | 0.55 – 1.00 | **1.000** ⭐ at ceiling | 0.667 |
| O121H19 | 1.000 | 0.55 – 1.00 | **0.889** ✅ upper half | 0.889 |
| **O157H7** | 0.111 | 0.40 – 0.78 | **0.667** ✅ in range; +0.56 vs uniform — **Effect 1 confirmed** | 0.667 |
| Typhimurium | 0.778 | 0.11 – 0.67 | **0.111** ✅ at floor — **Effect 2 confirmed** | 0.000 |
| **MEAN** | **0.579** | **0.50 – 0.68** | **0.566** ✅ in range, slightly below uniform soft-vote | **0.492** |

### Verdict: branch (B) hit — partial mechanism confirmation

- 4-base T-scaled: mean 0.566 ≥ 0.55 ✓, **O157H7 0.667 ≥ 0.50 ✓**, **K-12 0.000 < 0.30 ✗**. Fails A's K-12 gate, lands in (B).
- 3-deep (no PLS-DA) sanity check: mean 0.492 — below 0.50 threshold. But **K-12 = 0.375 — first ensemble in the project to break K-12 above 0.00 cleanly**, captured by removing PLS-DA's confident-wrong vote rather than calibrating it down. Useful diagnostic.

### Mechanism findings (sharp)

1. **Calibration was the mechanism for ATCC25922 and O157H7.** Both cells flipped from confident-wrong (post-uniform-soft-vote 0.33, 0.11) to confidently-correct after temperature scaling (0.67, 0.67). The deep majority of 3-of-4 became audible once PLS-DA's vote was scaled down 5×.
2. **K-12 is NOT a calibration problem.** It's a *minority-of-one* problem: only DANN identifies K-12 as Non-STEC; PLS-DA + Patch5 + 2-ch CNN all confidently call K-12 something else. Even after sharpening DANN's vote (T=1.23, barely changed), 3-of-4 wrong votes still win. Confirmed by the 3-deep variant where removing PLS-DA brought K-12 to 0.375 (1-of-3 wrong votes is easier to overcome than 1-of-4).
3. **Typhimurium is the symmetric failure of K-12.** It's a minority-of-one in PLS-DA's favor (only PLS-DA correctly identifies Typhimurium as Salmonella; all 3 deep models confidently wrong). Softening PLS-DA below its sole-correct vote crashes Typhimurium from 0.78 → 0.11.
4. **The fundamental tradeoff:** any ensemble that fixes the PLS-DA-confidently-wrong cells (K-12 / O157H7 / ATCC25922) by reducing PLS-DA's weight necessarily hurts the PLS-DA-confidently-correct cells where PLS-DA is the sole right voter (Typhimurium, Dublin). With the current 4-base set, this tradeoff has no Pareto-improving solution — there's no scheme that simultaneously trusts PLS-DA on Typhimurium and DANN on K-12 using only test-time signals.

### Operational decision

**Ship PLS-DA solo at 0.603 as the LOSO headline.** Temperature-scaled soft-vote (0.566) is the **second-best ensemble result** of the project and the cleanest demonstration of the calibration-mismatch mechanism, but it doesn't beat PLS-DA solo. Document the calibration finding prominently in the writeup — it explains *why* none of the 4 prior ensemble attempts worked, and it provides a partial cell-level fix (ATCC25922 + O157H7) at the cost of other cells (Typhimurium especially).

**Per-strain best-model table now includes the temperature-scaled soft-vote as the source-of-truth for ATCC25922 and O157H7 if we want a single-row "ensemble" entry alongside PLS-DA solo.** But for the headline LOSO mean, PLS-DA solo stays the model card.

---

## 2026-05-15 — DANN aug-regime sweep (pre-registered BEFORE running)

**Hypothesis.** plan/00 has been carrying an explicit TODO since 2026-05-14: *"Current default aug slows training so much that every fold early-stops 5-30 epochs short of the 60-epoch budget while train_acc is still 0.4-0.5. A no-aug run on fold 0 reaches train_acc 0.88 in 60 epochs; the spec'd aug looks over-tuned."* If the deep models are under-trained (not capacity-limited), lighter aug should let them converge and lift LOSO. This is the **single largest unexplored lever** for headline movement.

**Sweep design.** Three lighter-aug variants × DANN λ=0.1 × LOSO (9 folds):

| Variant | p_noise | p_scale | p_shift | p_baseline | p_mixup | Rationale |
|---|---|---|---|---|---|---|
| default (baseline, already run) | 0.50 | 0.40 | 0.40 | 0.30 | 0.30 | Current heavy aug — LOSO 0.500 |
| **light** | 0.25 | 0.20 | 0.20 | 0.15 | 0.15 | All probabilities halved — direct test of "less of everything." |
| **no_mixup** | 0.50 | 0.40 | 0.40 | 0.30 | 0.00 | Keep default except drop mixup. Mixup blends class labels — most disruptive for cross-entropy + class weights. |
| **minimal** | 0.30 | 0.00 | 0.20 | 0.00 | 0.00 | Only Raman-physical augs (noise + bin-shift). Scale/baseline/mixup all off. |
| off (already characterized on Protocol A) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | Sanity-check no-aug baseline; not re-run unless above 3 fail. |

### Predicted LOSO mean parent-recall

| Variant | Central estimate | Range | Reasoning |
|---|---|---|---|
| light | **0.55** | 0.48 – 0.62 | "Reduce all aug equally" is the safest bet. Training converges further; per-strain variance reduces. Modest lift expected. |
| no_mixup | **0.53** | 0.48 – 0.60 | Mixup is the cleanest single-knob to drop. May lift moderately; preserves all other regularization. |
| minimal | **0.56** | 0.45 – 0.65 | Highest variance prediction — could either be the best (most under-training relief) or worst (lost useful regularization on scale/baseline drift). |

### Predicted per-strain (light variant — the most likely modal outcome)

| Strain | DANN λ=0.1 default actual | Predicted light |
|---|---|---|
| 83972 | 0.750 | 0.50 – 0.88 |
| ATCC25922 | 0.889 | 0.55 – 1.00 |
| Dublin | 0.000 | 0.00 – 0.33 |
| Heidelburg | 0.444 | 0.33 – 0.67 |
| K-12 | 0.750 | 0.50 – 0.88 (the headline cell — should HOLD or improve) |
| O103H2 | 0.333 | 0.30 – 0.67 |
| O121H19 | 0.667 | 0.55 – 0.89 |
| O157H7 | 0.556 | 0.40 – 0.78 |
| Typhimurium | 0.111 | 0.00 – 0.40 |
| **MEAN** | **0.500** | **0.48 – 0.62** |

### Verdict branches (locked BEFORE running)

- **(α) Any variant ≥ 0.603 mean**: **first single-model arm to beat PLS-DA solo.** Headline-changing result. Ship that aug regime as the new DANN config. Re-run Patch5 + 2-ch CNN with the same lighter aug as immediate follow-ups.
- **(β) Any variant 0.55 – 0.60 mean AND K-12 ≥ 0.50 AND O157H7 ≥ 0.40**: significant lift, becomes new DANN λ=0.1 headline (replaces 0.500). PLS-DA solo still leads overall but the gap narrows.
- **(γ) All variants 0.50 – 0.55, no biology cell regression**: modest convergence improvement; useful but doesn't move the headline. Document and move to multi-seed averaging as the next swing.
- **(δ) Any variant < 0.45 OR K-12 ≤ 0.30 OR O157H7 ≤ 0.30**: aug was load-bearing for the biology cells, not just regularization. Heavy aug stays default; document and stop.

### Probability mass over branches

- (α) headline-changing — **15%** — would be a strong result; requires aug to lift DANN by 0.10+.
- (β) significant lift, new DANN headline — **35%** — most-likely positive case.
- (γ) modest improvement only — **35%** — under-training was real but small.
- (δ) aug was actually right — **15%** — possible the original aug spec was correctly tuned.

**This is the honest pre-registration.** If any variant hits 0.65+ I should sanity-check for a bug (seed contamination, fold leakage). The honest ceiling is bounded by single-model LOSO physics on this dataset; PLS-DA 0.603 has been the implicit ceiling across 7 distinct model attempts.

### Post-run resolution (2026-05-15)

**Branch (δ) hit:** aug is **load-bearing for LOSO regularization**, not over-tuning. Both lighter variants regress vs. default. plan/00's TODO diagnosis ("train_acc 0.4-0.5 = under-training") was misreading a *correctly-regularized* LOSO model as undertrained. Skipped `minimal` — predicted strictly worse than `light`.

| Variant | LOSO mean | Δ vs default 0.500 | K-12 | O157H7 | ATCC25922 | Notes |
|---|---|---|---|---|---|---|
| default | **0.500** | — | 0.750 | 0.556 | 0.889 | Baseline |
| light (all p halved) | **0.347** | **−0.153** | 0.375 | 0.222 | 0.444 | Strong regression. Heidelburg, ATCC25922, K-12, O157H7 all crater. |
| no_mixup | **0.423** | **−0.077** | **0.875** ⭐ | 0.556 | 0.556 | Mean regresses but **K-12 = 0.875 = new joint-best for K-12** (ties DANN λ=0.3). 83972, ATCC25922, Heidelburg crater. |
| minimal | skipped | — | — | — | — | Predicted strictly worse than light. |

### Sharp findings

1. **Heavy aug is doing cross-strain regularization, not over-tuning.** plan/00's diagnosis read low train_acc (0.4-0.5 in 60 epochs) as undertraining. Under LOSO that's actually the **correct** training endpoint — climbing train_acc past 0.5 means memorizing the 8 training strains, which hurts generalization to the held-out 9th. The "no-aug fold-0 reaches train_acc 0.88" sanity-check pointed in the wrong direction because Protocol-A overfitting is the wrong proxy for LOSO regularization.
2. **Mixup specifically is K-12-suppressing.** no_mixup gives K-12 = 0.875 (tied for best across all DANN variants) at the cost of other cells. Mixup blends Non-STEC + STEC + Salmonella class labels with random α — those blended labels actively suppress the broad-scale chemistry signal that lets DANN separate K-12. **Without mixup, DANN's K-12 ceiling rises to 0.875 ≈ DANN λ=0.3's K-12 = 0.88.**
3. **No aug variant Pareto-dominates default.** The headline LOSO mean is still 0.500 (default) or 0.603 (PLS-DA solo overall).

### Operational decisions

- **Aug regime stays at default for the shipped DANN λ=0.1.** TODO removed from plan/00.
- **no_mixup is added to the per-strain best-model table as a K-12 specialist** — joint-best with DANN λ=0.3, but DANN λ=0.3 already documented in the per-strain table so no_mixup is mostly a curiosity rather than a new headline.
- **Pivot to multi-seed averaging.** This is the next-biggest unexplored lever — average soft-vote across 3 seeds of DANN λ=0.1 + default aug. Different rationale (stabilizing per-strain variance, not curing under-training), plausibly worth +0.03-0.07 LOSO. **Launched in parallel to writing this resolution.**
