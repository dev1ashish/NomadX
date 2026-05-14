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
