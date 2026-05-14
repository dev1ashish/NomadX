# 07 — Empirical findings

> **Mutability:** append-only. Never edit historical entries; if something turns out wrong, append a correction.
> **Format:** each finding has a date, a short slug for anchor-linking, a one-line summary, and supporting numbers.

---

## 2026-05-14 — parser-clean

**All 87 files parse with 0 fatal errors.** 7,999 spectra cached at 2048 bins each on the canonical `linspace(76, 3499, 2048)` wn axis. 8 files trigger the 200-px cap (incl. R364, R370, R365, R371) — these are the high-density files predicted by `01_data.md`. R371 also flagged `is_complete_scan=False` (351/360 px on its 20×18 coord grid). Cache lives at `data_cache/`.

## 2026-05-14 — class-balance

Per-class file counts confirmed: STEC=27, Salmonella=27, Non-STEC=25, H₂O=8. Per-class spectra counts after the 200-px cap: STEC=2544, Salmonella=2544, Non-STEC=2144, H₂O=767. H₂O is ~3× smaller — confirms macro-F1 over accuracy as the headline metric.

## 2026-05-14 — pc1-dominates-raw

**On raw+SNV-only data, PC1 captures 80.1% of total variance.** Way too high for a "well-behaved" chemistry-driven dataset. Almost certainly PC1 is fluorescence baseline gradient, not biological signal. Confirms preprocessing (arPLS in particular) is essential, not optional.

## 2026-05-14 — silhouette-negative-raw

**Overall subclass silhouette in PCA-50 space (raw+SNV) = −0.254.** Only H₂O is positive (+0.71); every bacterial subclass is negative. LDA between/total variance ratio for subclass = 0.13 (only 13% of variance lies between subclasses). Means: bacterial subclasses are NOT cleanly separable by linear methods in this representation.

## 2026-05-14 — anova-c-h-stretch

**Top 30 most-discriminative bins (ANOVA F-statistic, raw+SNV) are all in the C-H stretch region 2880–2940 cm⁻¹.** Mutual information per bin agrees. Validates the spectral crop decision (fingerprint + C-H, not fingerprint-only) — C-H lipid signal is the *strongest* 4-class discriminator, not a secondary contribution.

## 2026-05-14 — interfile-similarity-high

**Inter-file cosine similarity ~0.99 across all class pairs (raw+SNV).** Mean intra-class file-to-file similarity: STEC 0.997, Non-STEC 0.991, Salmonella 0.994, H₂O 0.999. Bacterial spectra are very similar at gross-shape level; discrimination lives in subtle peak ratios.

## 2026-05-14 — preprocess-helped-modestly

**Full preprocessing moves PC1 80.1% → 69.5% and silhouette −0.254 → −0.229.** Modest improvement, not the dramatic one I predicted. Reasons:
1. arPLS boundary artifact at left edge of crop range (see `arpls-boundary-artifact`).
2. SNV after baseline removal is still dominated by remaining global shape variance.
3. Bacterial classes are genuinely close in feature space.

H₂O silhouette actually *dropped* (+0.71 → +0.48) — probably because SNV+crop normalized away the absolute-intensity difference that made water trivially separable in raw data.

## 2026-05-14 — arpls-boundary-artifact

**arPLS leaves a huge spike at the left edge of the crop range (~400–500 cm⁻¹).** Visible in all 4 class-mean preprocessed spectra. Not chemistry — it's the arPLS algorithm's boundary fit being unreliable. Easy fix: shift crop start from 400 → 450 cm⁻¹. Will apply in next preprocessing pass.

## 2026-05-14 — qc-retention

**QC keeps 7,122 / 7,999 spectra (~89% across all classes).** 6 dropped by SNR < 5 (per-class minimum); 871 flagged as background candidates (low integrated fingerprint intensity + low spectral MAD). Median overall SNR = 18. Per-class retention rates: STEC 89.0%, Non-STEC 89.0%, Salmonella 89.1%, H₂O 89.2%. Better than the 60–80% bacterial / 40–60% water expected in `03_architecture.md` §B.

## 2026-05-14 — batch-effect

**Same-calibration-date files cluster 11% tighter than random pairs** in PCA-50 space. Mean file-centroid distance: same-date 2.141, different-date 2.398 (ratio 0.893). 12 distinct calibration dates in the dataset. R1 in `06_risks.md` is realized. Must run the memorization probe before claiming model success; may need to enable DANN.

## 2026-05-14 — peaks-overlap

**At all 4 known biological Raman bands (1004 Phe, 1450 CH₂, 1660 amide-I, 2900 C-H), the per-class boxplots for STEC/Non-STEC/Salmonella overlap almost completely.** H₂O is slightly lower at each band but with significant overlap too. Confirms: no single peak separates the classes; discrimination signal is multi-bin and likely nonlinear. Linear models will underperform; CNN/RBF-SVM/XGB are mandatory.

## 2026-05-14 — splits-cal-date-overlap

**Protocol A folds have 7–9 calibration dates appearing in BOTH train and test (warning, not error).** Mathematically unavoidable: only 12 distinct calibration dates across 87 files, so at any 80/20 split most dates straddle the boundary. Per-fold breakdown: fold 0=8, fold 1=9, fold 2=9, fold 3=9, fold 4=7. Protocol B (LOSO) is much cleaner: 1–3 cal-date overlaps per fold, since strains naturally cluster by acquisition window. Implication for interpretation: if Protocol A macro-F1 is materially higher than LOSO, some of that gap is plausibly date-batch leakage rather than true generalization; the LOSO drop is the more honest generalization measure. Mitigation deferred — promoting cal_date to a strict grouping variable would over-constrain a 5-fold split on only 12 dates.

## 2026-05-14 — splits-h2o-balance-ok

**H₂O pre-balancing across Protocol A folds works as intended.** Round-robin assignment of 8 H₂O files into 5 folds: 3 folds get 2 H₂O test files, 2 folds get 1. Every test fold contains H₂O → macro-F1 averaging is computable on every fold, no degenerate empty-class folds. Without this pre-balance step the auto stratifier hits sklearn issue #33085 with this group/class shape.

## 2026-05-14 — splits-summary

**Protocol A:** 5 folds × 16–18 test files = 1258–1512 test spectra per fold. Per-fold class distribution stays close to global proportions: STEC 4–6, Non-STEC 4–6, Salmonella 4–6, H₂O 1–2 files. **Protocol B:** 9 LOSO folds × 8–9 test files = 576–756 test spectra per fold (one full bacterial subclass each). Splits cached at `data_cache/splits/protocol_{a,b}.json`. Smoke check passes: zero pixel or file appears in both train and test of any fold; every row index in any split is QC-passing.

## 2026-05-14 — anova-bins-vs-stec-discriminative-bands

**Our 4-class ANOVA top 30 bins (all in 2880–2940 cm⁻¹ C-H stretch) are about the water-vs-bacteria boundary, NOT the within-bacterial discrimination problem.** Published STEC-vs-Non-STEC discriminative bands per [`cisek-2013`](11_references.md#cisek-2013--cisek-et-al-analyst-2013-sensitive-and-specific-discrimination-of-pathogenic-and-nonpathogenic-escherichia-coli-using-raman-spectroscopy): **1658 cm⁻¹** (amide I, protein), **1454 cm⁻¹** (CH₂/CH₃ deformation, lipid), **1338 cm⁻¹** (CH₂ wagging / adenine ring). All in the fingerprint region, well inside our 400–1800 crop. Implication for model interpretation: when feature importance / saliency lands on 1338/1454/1658, that's biologically grounded; when it lands on C-H stretch 2900, the model is mostly using "is it water or bacteria?" — not useful for the hard cell. Recommend: run a second ANOVA EXCLUDING H₂O (just the 3 bacterial classes) to surface the genuine within-bacterial discriminative bins. Slot into next preprocessing/EDA pass.

## 2026-05-14 — k12-loso-failure-is-biology

**The complete failure on the K-12 LOSO fold (8/8 misclassified as Salmonella for LogReg, LinSVM, RBF-SVM) is biology, not a bug.** Per [`soupene-2003-k12`](11_references.md#soupene-2003-k12--soupene-et-al-j-bacteriol-2003-laboratory-strains-of-escherichia-coli-k-12-things-are-seldom-what-they-seem): K-12 has been laboratory-domesticated since the 1920s; it has accumulated large genomic deletions vs wild-type E. coli; it is missing common stress-response and surface-structure genes. K-12 is genuinely atypical vs the other Non-STEC training set (83972, ATCC25922). When you hold K-12 out, the remaining Non-STEC training files don't span the chemistry needed to recognize it. **In the final write-up:** flag K-12 as a known-atypical strain rather than treating it as a fair Non-STEC test case.

## 2026-05-14 — stec-virulence-is-plasmid-not-cellwall

**The STEC vs Non-STEC distinction is virulence-defined (Shiga toxin presence), not phylogenetic.** Per [`stec-virulence-overview`](11_references.md#stec-virulence-overview--virulence-2013-shiga-toxin-producing-escherichia-coli): stx genes are encoded on lysogenic phages (chromosomally integrated, horizontally mobile); most other STEC virulence markers (ehxA, katP, espP, stcE, subAB) are plasmid-encoded. The cell wall, ribosomes, core cytoplasm — the bulk of any Raman signal — are NOT meaningfully different between STEC and Non-STEC E. coli. **The within-E. coli classification problem we're being asked to solve is fundamentally harder than the cross-genus problem.** Sets a hard biological ceiling on what any label-free Raman model — classical or deep — can achieve on this cell. Expect this cell to dominate the error budget across every model we run.

## 2026-05-14 — classical-results-groupkfold-vs-loso

**Five classical models trained under both protocols. Headline numbers (file-level macro-F1):**

| Model     | Protocol A (GroupKFold) | LOSO macro-F1 | LOSO parent-recall |
|-----------|-------------------------|----------------|---------------------|
| LogReg    | 0.961 ± 0.042           | 0.161 ± 0.105  | **0.59**            |
| PLS-DA    | 0.951 ± 0.051           | 0.164 ± 0.106  | **0.60** ⭐         |
| RBF-SVM   | 0.833 ± 0.096           | 0.127 ± 0.092  | 0.42                |
| LinSVM    | 0.779 ± 0.112           | 0.143 ± 0.108  | 0.52                |
| RF        | 0.753 ± 0.118           | 0.105 ± 0.073  | 0.31                |

**LOSO macro-F1 is a broken metric** (ceiling 0.25 because only 1 of 4 classes is in any LOSO fold's test set). **The correct LOSO metric is per-strain parent-class recall.**

**Headline interpretation:**
- **Linear models (LogReg, PLS-DA) win both protocols.** Surprising — the EDA findings predicted linear methods would struggle. They don't, when judged on file-level soft-vote. Pixel-level macro-F1 *was* worse for linear models (~0.70 vs 0.85 for RBF-SVM); within-file averaging of ~80–200 near-duplicate pixels collapses pixel-level noise.
- **GroupKFold-to-LOSO drop is severe across all models.** ~0.55–0.65 drop in macro-F1, ~0.30–0.40 drop in equivalent file-level parent-recall. Most of the GroupKFold success was learning strain-specific signal (biological strain quirks AND/OR calibration-date batch effect). Held-out new strains expose this.
- **PLS-DA is the marginal leader on LOSO** by parent-class recall (0.60), barely above LogReg (0.59). Within the 0.08-F1 minimum-detectable-difference noise floor; not a meaningful gap.
- **RF is the weakest overall.** Likely because RF without PCA on 987 correlated bins splits at arbitrary axis-aligned thresholds rather than capturing low-dimensional manifold structure.

## 2026-05-14 — loso-per-strain-pattern

**Per-strain LOSO parent-recall reveals 4 distinct generalization regimes** (mean across 5 models in parentheses):

```
EASY (most models get >0.85):
  Typhimurium  (Salmonella)  0.80  -> common Salmonella serovar, well-represented by other Salmonella strains in training
  O103H2       (STEC)        0.93  -> covered by other STEC strains in training
  83972        (Non-STEC)    0.73  -> uropathogenic E. coli; other Non-STEC training files generalize to it (except RF)

MEDIUM (0.4-0.8):
  O121H19      (STEC)        0.78  -> partially covered by other STEC
  Heidelburg   (Salmonella)  0.60  -> mixed predictions across models
  Dublin       (Salmonella)  0.31  -> often misclassified as Non-STEC

HARD (recall < 0.3):
  ATCC25922    (Non-STEC)    0.18  -> often misclassified as STEC (intraspecies E. coli confusion)
  O157H7       (STEC)        0.13  -> nearly always misclassified as Non-STEC (the classic O157 vs commensal E. coli confusion)

CATASTROPHIC (recall = 0.00):
  K-12         (Non-STEC)    0.00  -> always misclassified as Salmonella; biological reason — K-12 is laboratory-domesticated [11_references.md#soupene-2003-k12]
```

**Three biologically-grounded observations:**

1. **O157H7 → Non-STEC misclassification is the predicted-by-biology error.** O157:H7 is a serotype of E. coli; non-STEC strains (83972, ATCC25922) are also E. coli. Without explicit Shiga-toxin presence detection (a single phage-encoded protein), Raman cannot distinguish them. RF is the only model getting any O157H7 recall (0.33) — interesting because it's also the worst average; RF's per-fold variance is doing real work occasionally.

2. **K-12 → Salmonella is a model-shape failure, NOT a chemistry failure.** K-12 is a lab strain with significant genomic divergence from other Non-STEC training files; the model has nothing in its non-STEC manifold to anchor K-12 to, so it defaults to whichever class is geographically nearest in feature space — apparently Salmonella in PCA-50.

3. **ATCC25922 (Non-STEC, lab strain) → STEC confusion is the inverse of O157H7's confusion.** Both are lab/clinical isolates with similar growth characteristics; the model has no way to distinguish pathogenic E. coli from atypical commensal E. coli at the chemistry level.

**STEC vs Non-STEC IS the dominant error cell, as predicted in [08_expectations.md](08_expectations.md).** Across all models under LOSO, the largest single confusion is in the E. coli binary — exactly what biology says we should expect.

## 2026-05-14 — memorization-probe-weak

**Tiny 6,615-param 1D-CNN predicting file_id from a single preprocessed spectrum.** Within-file 80/20 pixel split (so every file_id appears in both train and test). Trained 15 epochs on MPS, ~9 seconds.

Result: **top-1 test acc = 0.041 (3.5× chance of 0.0115)**, **top-5 test acc = 0.159 (2.8× chance of ~0.057)**. Plateaus around epoch 6.

Pre-registered DANN-enable threshold per [02_decisions.md](02_decisions.md) is 10% top-1. We are at 4.1% → **below threshold; DANN not auto-enabled for CNN session.** But signal exists — a file-id signature IS detectable, just not dominant. The CNN we'll train next session has 100K+ parameters; it WILL likely extract more file signature than this tiny probe did. Recommendation: re-run the probe using the trained CNN's penultimate-layer representation rather than a from-scratch tiny network — that's the more honest check.

## 2026-05-14 — cal-date-diagnostic-mixed-signal

**For each LOSO misclassification across the 5 classical models, computed "lift" of calibration-date overlap between the held-out file and the wrong predicted class's training files.** Lift > 1 means batch effect is correlated with the misclassification; lift ~ 0 means biology / feature space, not date, drove the error.

| Strain | True parent | Wrong pred | n_err (across 5 models) | Lift |
|---|---|---|---|---|
| ATCC25922 | Non-STEC | STEC | 33 | **2.89×** |
| Heidelburg | Salmonella | STEC | 8 | **1.81×** |
| O157H7 | STEC | Salmonella | 6 | **1.69×** |
| O121H19 | STEC | Non-STEC | 2 | 1.56× |
| Typhimurium | Salmonella | Non-STEC | 9 | 1.39× |
| 83972 | Non-STEC | Salmonella | 11 | 0.65× |
| K-12 | Non-STEC | Salmonella | **38** | **0.59×** |
| O157H7 | STEC | Non-STEC | **36** | **0.00×** |
| Dublin | Salmonella | Non-STEC | 28 | 0.00× |
| Heidelburg | Salmonella | Non-STEC | 10 | 0.00× |

**Key reads:**
- **The two biggest single errors are NOT batch-effect-driven.** O157H7 → Non-STEC (36 errors, lift 0.00) and K-12 → Salmonella (38 errors, lift 0.59 — *less* than random) cannot be explained by calibration-date leakage. These are pure biology / feature-space failures.
- **Some smaller errors ARE batch-driven.** ATCC25922 → STEC at lift 2.89× means the model's tendency to misclassify ATCC25922 as STEC really is partly a date-batch artifact. DANN would help these.
- **Dublin → Non-STEC (28 errors) is also pure biology / feature-space**, not date.

Combined with [memorization-probe-weak](#2026-05-14--memorization-probe-weak): **the LOSO crater is bigger than file-id / calibration-date leakage can fully explain.** Most of the gap is genuine cross-strain generalization difficulty, consistent with the biological ceiling argued in [stec-virulence-is-plasmid-not-cellwall](#2026-05-14--stec-virulence-is-plasmid-not-cellwall). DANN should be available as a flag but not enabled by default for the first CNN run.

## 2026-05-14 — xgboost-complementary-failure-mode

**XGBoost added to the classical sweep post-hoc** after `brew install libomp`. Protocol A file F1 = 0.796 ± 0.103, between LinearSVM and RBF-SVM. LOSO macro-F1 = 0.125 ± 0.069. **LOSO parent-class recall mean = 0.37**, fifth out of six classical models.

**Per-strain pattern is the most interesting result.** XGBoost (and RF) are the ONLY classical models to get any non-zero recall on the **O157H7 strain (parent: STEC)** under LOSO — both score 0.33 there, where linear models (LogReg/LinSVM/PLS-DA) and kernel SVM all score exactly 0.00. Conversely, tree models score badly on the "easy" strains 83972 and Typhimurium that linear models ace.

**Interpretation:** tree-based axis-aligned splits can pick up a small set of high-discriminator bins (possibly the published STEC bands at 1338/1454/1658 cm⁻¹) that survive cross-strain transfer; this gives them partial O157H7 recall. But they overfit to non-transferable strain-specific cues on the easier folds, dropping recall there. **Linear-vs-tree models fail in different ways on different strains** — neither family wins all strains. This suggests an ensemble (linear + tree averaging) might beat the best individual classical model on LOSO. Future work.

This also reinforces the earlier conclusion: most of the LOSO crater is biology, and DANN won't close the O157H7 gap (which is the largest single-strain failure). Trees show that a *small amount* of cross-strain signal exists; getting more of it requires either domain adaptation (DANN) for the biology-independent batch component OR a higher-capacity nonlinear model (CNN) for the biology-dependent within-E. coli component.

## 2026-05-14 — cnn-spec-underfit-and-fixes

**The §E small-CNN spec, taken literally, can't fit even the training data.** First Protocol A run (channels 16-32-48-64 per spec, InstanceNorm-only front end) produced mean file-F1 = 0.40 ± 0.19 with per-fold values [0.61, 0.50, 0.46, 0.14, 0.28] — well below the pre-registered floor of 0.92 and below RF's classical floor (0.75).

Diagnostic on fold 0 with no augmentation, no class weighting, no label smoothing, 120 epochs: train_acc plateau at 0.74, val_macro_f1 ceiling 0.56. The model wasn't undertraining — it was running out of fittable capacity.

**Two root causes, both architectural:**

1. **The spec's `~110K param` target and `channels 16-32-48-64` description are arithmetically inconsistent** — those channel widths with kernels 15-7-7-5 give 33K params, not 110K. Doubled to 32-64-96-128: 124K params, matches the target. See [10_decision_log.md §cnn-fixes](10_decision_log.md#2026-05-14--cnn-small-variant-architectural-fixes-mid-session).

2. **No per-bin normalization at the input.** SNV ensures per-row mean=0 and std=1 but per-BIN mean ranges -0.46 to +3.84 across the 987 bins. Classical models pipe through StandardScaler before PCA+LogReg; the §E spec only listed InstanceNorm1d (which is per-spectrum, near-noop on SNV data). Adding a per-bin standardize (fit on outer-train, baked into the model as a `register_buffer`) lifted train_acc from 0.43 to 0.88 in 60 epochs on the same fold.

With both fixes the model trains. Without them the experiment doesn't answer any question because the architecture isn't capable of fitting train data. Logged as a future-session prerequisite: **fit one fold no-aug as a sanity check before launching any new model arm's full sweep.**

## 2026-05-14 — cnn-protocol-a-underperforms-classical

**CNN (fixed architecture, default §E augmentation) on Protocol A: file-macro-F1 = 0.649 ± 0.079.** Per-fold: 0.63 / 0.65 / 0.54 / 0.68 / 0.76. Compared to classical leaderboard (file-F1):

```
PLS-DA     0.951 ± 0.051   ⭐ leader
LogReg     0.961 ± 0.042
RBF-SVM    0.833 ± 0.096
XGBoost    0.796 ± 0.103
LinSVM     0.779 ± 0.112
RandomFor  0.753 ± 0.118
CNN small  0.649 ± 0.079   ⬅ worst classical-family-level result
```

CNN is the worst headline-metric model on Protocol A. **The Salmonella class collapses in 4 of 5 folds** (Salmonella recall sequence: 0.50 / 0.17 / 0.00 / 0.50 / 0.83) — the model often predicts Non-STEC or STEC for Salmonella files. Classical models don't show this Salm→{Non-STEC, STEC} confusion at file level.

**Why is the CNN losing on Protocol A?** Protocol A is largely saturated by within-file pixel averaging (cosine 0.997 → file-level soft-vote denoises pixel-level noise). Classical models exploit this by being linear over a learned PCA(50-150) basis — their effective dimensionality matches the data's true low-dim manifold. The CNN has to LEARN that manifold from raw 987-bin SNV input with only 70 training files; 124K params + the spec'd augmentation regime overfits to file-specific signatures inside the training files (visible as bumpy per-fold val_macro_f1 trajectories: best inner-val F1 averages 0.51 ± 0.07 across folds) and the soft-vote can't denoise overfitting in the same way it denoises pixel-level noise.

## 2026-05-14 — cnn-loso-complementary-pattern

**CNN LOSO mean parent-class recall = 0.35**, vs PLS-DA 0.60. Below the classical floor, by 0.25.

But the per-strain breakdown is the *only* meaningful finding from this run:

| Strain | Parent | Linear leaders (LogReg/PLS-DA) | Trees (RF/XGB) | CNN small | Family with best recall |
|---|---|---|---|---|---|
| 83972      | Non-STEC   | 0.88 | 0.12–0.25 | **0.88** | linears = CNN |
| ATCC25922  | Non-STEC   | 0.22 | 0.11–0.33 | 0.11 | trees |
| **K-12**       | **Non-STEC**   | **0.00** | **0.00** | **0.50** | ⭐ **CNN only** |
| Dublin     | Salmonella | 0.56 (PLS-DA) | 0.11 | 0.11 | linears |
| Heidelburg | Salmonella | 0.89 | 0.33–0.44 | 0.33 | linears |
| Typhimurium| Salmonella | 1.00 | 0.33–0.44 | 0.11 | linears |
| O103H2     | STEC       | 1.00 | 0.67 | 0.56 | linears |
| O121H19    | STEC       | 0.89 | 0.78 | 0.00 | linears |
| **O157H7**     | **STEC**       | **0.00** | **0.33** | **0.56** | ⭐ **CNN > trees > linears** |

**Three patterns extending [§xgboost-complementary-failure-mode](07_findings.md#2026-05-14--xgboost-complementary-failure-mode):**

1. **CNN cracks K-12 (parent Non-STEC) at 0.50 recall where every classical model — linear, kernel, tree — scored exactly 0.00.** K-12 is the laboratory-domesticated strain that [soupene-2003-k12](11_references.md#soupene-2003-k12--soupene-et-al-j-bacteriol-2003-laboratory-strains-of-escherichia-coli-k-12-things-are-seldom-what-they-seem) flagged as genomically atypical. Pre-registered prediction was 0.00–0.15 ("biological ceiling"); the actual 0.50 is well above that ceiling. **The CNN's nonlinear featurization picks up something on K-12 that linear methods on PCA-50 simply cannot reach.** Plausible mechanism: K-12 may share local peak ratios with other Non-STEC strains in fingerprint regions that PCA collapses but a CNN preserves; what looks like "biology no model can solve" on linear features may turn out to be "no shared low-dim manifold linear methods can find, BUT shared local peaks a CNN can find."

2. **CNN cracks O157H7 (parent STEC) at 0.56 recall**, above trees (0.33) and linear models (0.00). O157H7 is the second-hardest "biology" fold (published-STEC bands at 1338/1454/1658, but the STEC vs Non-STEC distinction is virulence-defined and most of the Raman signal is phylogenetic, not virulence). This corroborates pattern 1: nonlinear features capture sub-peak structure that linears miss.

3. **CNN loses on the "easy" strains the linear models ace** — Typhimurium (1.00 → 0.11), Heidelburg (0.89 → 0.33), O121H19 (0.89 → 0.00), O103H2 (1.00 → 0.56). On these strains the held-out file simply *looks similar* to a training file via PCA distance; linear methods nail this; the CNN destroys this signal during augmentation (especially Beta(0.2, 0.2) mixup) and replaces it with whatever its 124K-param encoder happened to learn.

**Linear / tree / CNN are three different inductive biases.** None of them wins all strains. The cross-family ensemble (soft-vote across PLS-DA + XGB + CNN, or any 2 of the 3) likely scores higher mean parent-recall than any single family — [§future_work](09_future_work.md) entry recommended.

## 2026-05-14 — memprobe-v2-fires

**Class-trained CNN's penultimate features encode file_id at 15.5% top-1 (13.5× chance), top-5 = 37.0%.** Above the pre-registered 10% DANN threshold. Compares to memprobe v1 (from-scratch 6.6K-param tiny CNN trained directly for file_id) which got 4.1% top-1.

**Why is v2 4× higher than v1?** Capacity. The class-supervised 124K-param CNN, even though its training objective is 4-way class prediction, ends up incidentally encoding 87-way file-id features at higher fidelity than the tiny network *optimized for the file-id task* could extract. Files share calibration date, laser intensity, exposure, and operator — these correlate with class but also with sub-class file identity; the encoder's representation captures both.

**Decision: defer DANN.** Probe fires, so by the strict pre-registered rule DANN should be on for the next CNN run. But the per-strain pattern shows **the CNN already cracks K-12 and O157H7**, the strains where there's NO same-strain training file to leak from. Whatever the CNN learns about K-12 and O157H7 is by definition NOT batch-effect leakage. Naive DANN would penalize any encoder feature correlated with file_id, including the genuinely-cross-strain-discriminative features that produce those K-12 / O157H7 wins.

Recommend for the Transformer session: enable DANN as an *ablation arm* alongside vanilla, then compare the per-strain pattern on K-12 / O157H7 specifically. If DANN preserves K-12 and O157H7 recall AND boosts the others, ship DANN. If DANN crushes K-12 and O157H7, the right next step is an ensemble, not adaptation.

## 2026-05-14 — ensemble-fails-to-clear-plsda

**Soft-vote ensembles of three model families (PLS-DA + XGB + CNN, all pairwise combinations) under both protocols. None beats PLS-DA solo on LOSO mean parent-class recall.** Predicted in [08_expectations.md §ensemble](08_expectations.md#2026-05-14--soft-vote-ensemble-pre-registered-before-running); resolution: branch "all three ensembles ≤ 0.60 mean" hit.

Headline numbers:

| Ensemble | Protocol A file-F1 | LOSO mean parent-recall | K-12 | O157H7 |
|---|---|---|---|---|
| PLS-DA solo (leader) | **0.951 ± 0.051** | **0.60** ⭐ | 0.00 | 0.00 |
| CNN small solo | 0.649 ± 0.079 | 0.35 | **0.50** ⭐ | **0.56** ⭐ |
| XGB solo | 0.796 ± 0.103 | 0.37 | 0.00 | 0.33 |
| (a) PLS-DA + XGB | 0.863 ± 0.104 | 0.529 | 0.00 | 0.00 |
| (b) PLS-DA + CNN | 0.919 ± 0.030 | **0.579** | 0.00 | 0.00 |
| (c) PLS-DA + XGB + CNN | 0.864 ± 0.105 | 0.517 | 0.00 | 0.00 |

**The two biology-hard wins do not survive averaging.** Both CNN's K-12 (0.50) and CNN's O157H7 (0.56) — the wins that prompted this experiment — go to 0.00 in every ensemble. XGB's O157H7 = 0.33 also goes to 0.00. PLS-DA's wrong-class confidence on these strains is high enough that uniform averaging with a sharper-but-minority correct vote still produces file-level argmax = wrong-class.

**Why averaging crushes minority-vote signal here.** Take K-12 specifically:
- CNN K-12 file-proba: roughly (Non-STEC ≈ 0.40, Salmonella ≈ 0.45, others ≈ 0.075). Correct class is *runner-up*, not majority — CNN's K-12 win at the file level only emerges from per-spectrum aggregation favoring Non-STEC.
- PLS-DA K-12 file-proba: roughly (Non-STEC ≈ 0.05, Salmonella ≈ 0.85, others ≈ 0.05). Confidently wrong.
- Uniform 2-way average: (Non-STEC ≈ 0.22, Salmonella ≈ 0.65). Argmax = Salmonella. The CNN's lift on Non-STEC isn't large enough to overcome PLS-DA's confident Salmonella vote even when CNN is right.

Same mechanism for O157H7: CNN's STEC vote at ~0.45 averaged with PLS-DA's confident Non-STEC vote at ~0.80 yields ensemble Non-STEC argmax. **Weight tuning can't fix this** without breaking ensemble (a) and (b)'s simultaneous wins on the easy strains — PLS-DA needs to dominate those *and* be overridden on K-12/O157H7, which is just "predict CNN on those folds," which isn't an ensemble.

**Protocol A also regresses across the board.** PLS-DA solo Protocol A file-F1 = 0.951; the best ensemble (b) PLS-DA + CNN = 0.919; (a) and (c) come in at ~0.864. Two mechanisms:
1. H₂O recall drops from 8/8 (PLS-DA solo) to 5–6/8 in ensembles. CNN and XGB are less confident on H₂O than PLS-DA; averaging dilutes the perfect H₂O signal.
2. Salmonella files where the CNN was wrong [07§cnn-protocol-a-underperforms-classical](07_findings.md#2026-05-14--cnn-protocol-a-underperforms-classical) drag the (b) and (c) ensembles below PLS-DA's file-level argmax in fold 4 (file-F1 drops to 0.680).

**Pre-registered ranges were broadly accurate** — most actuals landed inside the predicted ranges. The two notable misses are (b)'s K-12 actual 0.00 vs predicted 0.13–0.50 (lower-half), and (a)/(b)/(c)'s O157H7 actual 0.00 vs predicted lower bounds 0.00 / 0.14 / 0.00. The pre-registration honestly captured "lower half more likely"; we got the lowest possible value for both biology-hard cells.

**Operational decision** (matches pre-locked verdict structure): **ship PLS-DA solo as the headline LOSO model.** Flag CNN as single-model-best on K-12 (0.50) and O157H7 (0.56) in the README — these are biologically meaningful per-strain wins that the ensemble cannot capture but a per-strain ensembling scheme (route-on-prediction-confidence, stacking with a meta-learner trained on disagreement structure) plausibly could. **Deferred to future work**, not this session.

**What this tells us about DANN, on the deferred decision in [§memprobe-v2-fires](#2026-05-14--memprobe-v2-fires).** The ensemble result rules out the "ensemble preserves biology wins → DANN unnecessary" path. We're left with: DANN might lift the CNN's mean by closing the easy-strain regression (Typhimurium 0.11, O121H19 0.00) without destroying its K-12/O157H7 wins. **That's the right next session.** A weighted ensemble where CNN is up-weighted on out-of-training-distribution test files (i.e. all LOSO folds, by construction) is a possible alternative, but it's basically "use CNN on LOSO and PLS-DA on Protocol A" — not a simple ensemble.

## 2026-05-14 — transformer-underperforms-cnn

**Small 1D-Transformer (~217K params, patch_size=20) trained under both protocols with the same recipe as the CNN. It is the weakest single-model arm in the sweep.** Pre-registered in [08§transformer](08_expectations.md#2026-05-14--small-1d-transformer-pre-registered-before-running-the-full-sweep); pre-locked verdict branch hit: "mean parent-recall < 0.30 → Transformer is worse than CNN; treat as benchmark completeness arm only."

Headline numbers:

| Model | Protocol A file-F1 | LOSO mean parent-recall | K-12 | O157H7 |
|---|---|---|---|---|
| PLS-DA (leader) | 0.951 ± 0.051 | **0.60** ⭐ | 0.00 | 0.00 |
| LogReg | 0.961 ± 0.042 | 0.59 | 0.00 | 0.00 |
| RBF-SVM | 0.833 ± 0.096 | 0.42 | 0.00 | 0.00 |
| XGBoost | 0.796 ± 0.103 | 0.37 | 0.00 | 0.33 |
| LinSVM | 0.779 ± 0.112 | 0.52 | 0.00 | 0.00 |
| Random Forest | 0.753 ± 0.118 | 0.31 | 0.00 | 0.33 |
| **CNN small** | 0.649 ± 0.079 | 0.35 | **0.50** ⭐ | **0.56** ⭐ |
| **Transformer small** | **0.507 ± 0.122** | **0.193** | 0.00 | 0.00 |

**Transformer is dead last on both axes.** Protocol A file-F1 is 0.10 below the CNN (which was already last among classical+CNN) and 0.44 below PLS-DA. LOSO mean parent-recall is 0.16 below the CNN, the previous weakest LOSO performer.

**The diagnostic finding: 20-bin patches blur the narrow-peak signal the CNN found.** Three pieces of evidence point at patch size as the culprit:

1. **Sanity check (no-aug, fold 0, 60 epochs) reached only train_acc 0.69 / val_f1 0.50.** Compared to the CNN's no-aug progression (33K underfit → 124K InstanceNorm-only 0.69/0.51 → 124K + per-bin standardize 0.88/0.56), the Transformer sits *at the InstanceNorm-only CNN level* despite carrying per-bin standardize from the start. The per-bin buffer is wired correctly (verified in code); the bottleneck is upstream of optimization. Confirmed via param count: 217K Transformer vs 124K CNN — more raw params, less effective capacity on this signal.

2. **K-12 and O157H7 — the strains the CNN uniquely cracked — both collapse to 0.00.** The mechanism the CNN was using on these biology-hard cells was almost certainly local peak ratios in the fingerprint region (~5-10 bin wide Raman peaks; CNN kernels 5-15 preserve them). A patch_size=20 strided Conv1d averages every peak with its 19-bin neighborhood before any attention pass sees it. The narrow-peak signature is gone before the encoder gets to look at it.

3. **The one place Transformer beats CNN is on a strain the CNN already failed on.** O121H19 (STEC) Transformer = 0.22 vs CNN = 0.00. Linear models got 0.89 on this strain; trees 0.78. The Transformer is finding *some* O121H19 signal but only enough to catch up partway. Doesn't reverse the overall ordering.

**Pre-launch sanity-check protocol paid off** ([10§cnn-architectural-fixes](10_decision_log.md#2026-05-14--cnn-small-variant-architectural-fixes-mid-session)). The 60-second no-aug fold-0 pass told us the architecture's val ceiling was ~0.50 before we burned 18 minutes on the full sweep. We chose to proceed to confirm the per-strain pattern; the result is informative for the writeup but doesn't change any deployment decision. **Lesson re-affirmed: always sanity-check a single fold no-aug before launching a new architecture's full sweep.**

**Augmentation is partly to blame on Protocol A.** No-aug fold 0 reached file_F1 0.722; full Protocol A with the §E aug regime got 0.507 (per-fold 0.48 / 0.35 / 0.49 / 0.53 / 0.69). A no-aug full sweep would land ~0.72 — better than the 0.507 actual but still below the CNN (0.649) and well below PLS-DA (0.951). Not pursued this session: the per-strain LOSO finding (patch resolution loses biology signal) is the load-bearing one, and tuning aug doesn't recover patch-blurred peaks.

**Operational decisions made consistent with the pre-locked verdict structure:**
- **Headline LOSO model remains PLS-DA solo + CNN flagged as per-strain best on K-12 (0.50) and O157H7 (0.56).**
- **Transformer documented as benchmark completeness arm** in the README; future work entry added: "try patch_size=5 or overlapping patches with the same architecture and recipe; expected to lift K-12 and O157H7 if the patch-blur hypothesis is correct."
- **Do NOT include Transformer in any downstream ensemble or DANN experiment** — it's strictly dominated by CNN on the per-strain pattern that motivates both.
- **DANN-as-ablation on CNN remains the next session** ([07§memprobe-v2-fires](#2026-05-14--memprobe-v2-fires)), unchanged by this finding.

Total wall-clock: ~25 min for the full sweep (Protocol A 7 min + LOSO 8 min + overhead). Came in under the pre-registered 5–15 / 9–27 minute budget.

## 2026-05-14 — dann-ablation-clears-verdict-a

**DANN (lambda_max=0.1, linear warmup over 10 epochs, GRL on the 32-dim penultimate, 87-way / per-fold-K-way file_id domain head) trained under both protocols with the same recipe as vanilla CNN. Pre-registered verdict branch (A) hit on the per-strain LOSO front; verdict branch on the memprobe pre-registration did NOT hit (probe stays above the 10% threshold).** Pre-registration at [08§dann-ablation](08_expectations.md#2026-05-14--dann-ablation-on-cnn-lambda_max01-pre-registered-before-the-full-sweep).

Headline numbers:

| Model | Protocol A file-F1 | LOSO mean parent-recall | K-12 | O157H7 |
|---|---|---|---|---|
| PLS-DA (LOSO mean leader) | 0.951 ± 0.051 | **0.60** ⭐ | 0.00 | 0.00 |
| LogReg | 0.961 ± 0.042 | 0.59 | 0.00 | 0.00 |
| LinSVM | 0.779 ± 0.112 | 0.52 | 0.00 | 0.00 |
| RBF-SVM | 0.833 ± 0.096 | 0.42 | 0.00 | 0.00 |
| XGBoost | 0.796 ± 0.103 | 0.37 | 0.00 | 0.33 |
| Random Forest | 0.753 ± 0.118 | 0.31 | 0.00 | 0.33 |
| CNN small (vanilla) | 0.649 ± 0.079 | 0.35 | **0.50** ⭐ | **0.56** ⭐ |
| Transformer small | 0.507 ± 0.122 | 0.193 | 0.00 | 0.00 |
| **CNN + DANN λ=0.1** | **0.566 ± 0.091** | **0.500** | **0.75** ⭐⭐ | **0.56** ⭐ |

**Per-strain LOSO comparison (vanilla CNN → DANN λ=0.1):**

| Strain | Parent | Best classical | Vanilla CNN | DANN predicted | DANN actual | Verdict |
|---|---|---|---|---|---|---|
| 83972 | Non-STEC | 0.88 (PLS-DA/LR) | 0.88 | 0.55 – 0.95 | **0.75** | ✅ in range, modest cost vs vanilla |
| ATCC25922 | Non-STEC | 0.33 (XGB) | 0.11 | 0.00 – 0.40 | **0.89** | ⭐⭐⭐ **above ceiling by 0.49 — best across ALL models** |
| **K-12** | **Non-STEC** | **0.00 (everyone)** | **0.50 (CNN only)** | 0.00 – 0.50 (bet lower) | **0.75** | ⭐⭐ **above ceiling by 0.25 — DANN improves on CNN's biology win** |
| Dublin | Salmonella | 0.56 (PLS-DA) | 0.11 | 0.00 – 0.35 | **0.00** | ✅ at lower bound |
| Heidelburg | Salmonella | 0.89 (PLS-DA/LR) | 0.33 | 0.30 – 0.70 | **0.44** | ✅ in range, modest recovery |
| Typhimurium | Salmonella | 1.00 (linears) | 0.11 | 0.20 – 0.70 | **0.11** | ❌ below floor — DANN didn't recover this strain |
| O103H2 | STEC | 1.00 (linears) | 0.56 | 0.40 – 0.80 | **0.33** | ❌ below floor by 0.07 |
| O121H19 | STEC | 0.89 (linears) | 0.00 | 0.00 – 0.50 | **0.67** | ⭐⭐ **above ceiling by 0.17 — major recovery** |
| **O157H7** | **STEC** | **0.33 (RF/XGB)** | **0.56 (CNN best)** | 0.00 – 0.50 (bet lower) | **0.56** | ⭐ **above ceiling — biology win preserved exactly** |
| **MEAN parent-recall** | | **PLS-DA 0.60** | **0.35** | 0.30 – 0.55 | **0.500** | ⭐ in range upper half; **+0.15 over vanilla CNN**; still 0.10 below PLS-DA |

**Three observations that change the headline:**

1. **DANN is the only model that crosses the biology-generalization Pareto frontier.** PLS-DA has the best LOSO mean (0.60) but zero biology wins (K-12, O157H7 both 0.00). Vanilla CNN cracks the biology cells but has a weak mean (0.35). **DANN does BOTH**: K-12 0.75 / O157H7 0.56 / mean 0.500. Per the pre-locked verdict structure, this is branch (A): "CNN+DANN is the only model that cracks the biology-hard strains AND generalizes."

2. **The pre-registered K-12 bet was wrong in the direction that matters.** I predicted K-12 = 0.00 – 0.50, "bet on lower half," reasoning that DANN would strip file-id-correlated features that were ALSO genuinely strain-discriminative. The actual K-12 = 0.75 says **the load-bearing features for K-12 were NOT the file-id-correlated ones.** DANN stripped acquisition noise, and the genuine peak-ratio signal got CLEARER. Same mechanism explains why ATCC25922 went 0.11 → 0.89: a strain where vanilla CNN was overwhelmed by acquisition noise, DANN cleaned up. Note the K-12 file-level probs are tight (Non-STEC ≈ 0.42, Salmonella ≈ 0.35 — close call) so the win is real but fragile.

3. **Three strains where DANN didn't help — and one where it cost.** Typhimurium stayed at 0.11 (vanilla CNN was 0.11; PLS-DA is 1.00 — gap unresolved). O103H2 dropped from 0.56 to 0.33. Dublin stayed at 0.00 (was 0.11). The DANN benefit isn't uniform — strains where vanilla CNN was hitting its biology-hard mechanism (K-12, ATCC25922, O121H19) gain a lot; strains where CNN was already at its ceiling and PLS-DA is stronger (Typhimurium, O103H2) don't recover and may even regress. **DANN is a per-strain regime change, not a universal lift.**

**The memprobe puzzle: probe stayed at 14.0% top-1 even with verdict (A) cleared.**

Memprobe v2 on the DANN encoder (Protocol A fold 4, same encoder snapshot used for the vanilla CNN baseline) = **14.0% top-1 (12.2× chance), top-5 = 38.8%**. Vanilla CNN was 15.5% / 37.0%. **DANN at λ=0.1 reduced file-id leakage by 1.5 percentage points — essentially nothing — but improved LOSO mean parent-recall by 0.15.** My pre-registered prediction ("fires=False expected") was wrong.

Three readings, in order of how likely:
1. **λ=0.1 was too low to suppress the probe.** During training, domain_acc held at ~6–7% (chance 1.4%, vs final domain_acc ~19% during the lambda=0 control run). The discriminator was beating the encoder, but the encoder was still leaking enough that a fresh post-hoc LogReg can recover 14% top-1. A larger λ (e.g. 0.3) would probably push the probe down further. **But** that doesn't explain the LOSO mechanism.
2. **The probe and the LOSO mechanism are decoupled.** DANN reshapes which directions in 32-dim feature space encode file-id, but a fresh LogReg can find a 14%-accuracy separating subspace. What changed isn't the *learnability* of file_id from features — it's the *prominence* of file-id directions relative to strain-discriminative directions. The class head (and a downstream LOSO test) sees the strain-discriminative directions first because they're high-variance now; the LogReg memprobe sees through this and finds whatever low-variance file-id remnant is still there. **This is the more interesting hypothesis and would change the diagnostic story for the writeup.**
3. **Encoder is from Protocol A fold 4 — 70 files seen in training, 17 in test.** The 17 unseen files weren't in DANN's adversarial gradient. The probe accuracy might be inflated by easy-to-identify unseen files. If anything, though, unseen files should be HARDER to identify, *lowering* memprobe — so this can't fully explain the result.

**Operational decision (consistent with verdict branch A):**
- **Headline LOSO model switches from PLS-DA solo to CNN+DANN λ=0.1**, with the asterisk that PLS-DA still has the better mean parent-recall (0.60 vs 0.500). The new headline framing: **"DANN is the only model that breaks the biology / generalization tradeoff."** Vanilla CNN cracked biology and lost generalization; PLS-DA solved generalization and missed biology entirely; DANN does both.
- **PLS-DA stays in the README** as the strongest-single-strain-mean classical baseline.
- **Vanilla CNN stays in the README** for the K-12 / O157H7 comparison (DANN's preservation of those wins is the proof point that DANN didn't destroy biology).
- **Document the memprobe decoupling honestly** — pre-registration miss, and possibly the more important diagnostic finding in the writeup.
- **DEFERRED for this session**: lambda_max=0.05 and lambda_max=0.3 sweeps. The user-pre-registered stage was "run 0.1 first under both protocols; sweep the others only if 0.1 is interesting." 0.1 is very interesting (cleared branch A) → sweep is justified. But the load-bearing question (does DANN clear branch A?) is already answered. lambda_max=0.3 in particular would test whether the memprobe puzzle is "λ too low" (reading 1) or "probe decoupled from LOSO" (reading 2) — the diagnostic worth running.

**Compute budget actuals:**
- Sanity 1 (lambda=0, no-aug, fold 0): 32s
- Sanity 2 (lambda=0.1, full aug, fold 0): 38s
- Full Protocol A (5 folds): ~3.2 min (well under pre-registered 4–12 min)
- Full LOSO (9 folds): ~11 min (well under pre-registered 7–20 min)
- Memprobe v2 on DANN encoder: 1.0s
- **Total session wall-clock: ~16 min.** Came in under everything.

## 2026-05-14 — dann-lambda-frontier

**Followed up the λ=0.1 result with a full λ=0.3 sweep + memprobe to disambiguate the memprobe decoupling.** Pre-registration at [08§lambda-0.3](08_expectations.md#2026-05-14--dann-lambda_max03-sweep-pre-registered-before-running). Two questions answered:

1. **The probe-LOSO decoupling is real.** Tripling λ moved the memprobe by 0.36 pp (14.0% → 13.6%). The 10% pre-registered threshold from [02§decisions](02_decisions.md) is **not** a reliable diagnostic for DANN success on this dataset.
2. **There's a clean regime tradeoff between lambda and per-strain pattern.** Higher λ strengthens pathogen biology (K-12, O157H7, O103H2) at the cost of easy commensal recognition (83972, ATCC25922).

**Per-strain head-to-head:**

| Strain | Parent | Vanilla CNN | DANN λ=0.1 | DANN λ=0.3 | Reading |
|---|---|---|---|---|---|
| 83972 | Non-STEC | 0.88 | 0.75 | **0.25** | Easy commensal signal — λ=0.3 strips it |
| ATCC25922 | Non-STEC | 0.11 | **0.89** | **0.11** | Singular to λ=0.1; collapses at λ=0.3 |
| **K-12** | **Non-STEC** | **0.50** | **0.75** | **0.88** | Monotonic in λ — atypical biology gets stronger with more pressure |
| Dublin | Salmonella | 0.11 | 0.00 | 0.22 | Slight recovery at λ=0.3 |
| Heidelburg | Salmonella | 0.33 | 0.44 | 0.33 | λ=0.1 sweet spot |
| Typhimurium | Salmonella | 0.11 | 0.11 | 0.00 | Unrecoverable by either |
| O103H2 | STEC | 0.56 | 0.33 | **0.89** | Non-monotonic; λ=0.3 unlocks this strain |
| O121H19 | STEC | 0.00 | **0.67** | **0.67** | λ=0.1 already maxes; λ=0.3 holds |
| **O157H7** | **STEC** | **0.56** | **0.56** | **0.67** | Monotonic — pathogen STEC biology gets stronger with λ |
| **MEAN parent-recall** | | **0.35** | **0.500** ⭐ | **0.447** | λ=0.1 has best mean |
| **Memprobe v2 top-1** | | 15.5% | 14.0% | **13.6%** | Probe is essentially unmoved by 3× λ |
| **Protocol A file-F1** | | 0.649 | 0.566 | **0.493** | Strict cost in λ |

**Three findings worth pulling out:**

1. **Monotonic-in-λ biology cells: K-12 (0.50→0.75→0.88), O157H7 (0.56→0.56→0.67).** These are the strains whose load-bearing features were NOT file-id-correlated; more DANN pressure → cleaner peak-ratio signal → stronger recall. **This is the core mechanism the writeup should foreground.**

2. **Non-monotonic / regime-switching cells: ATCC25922 (0.11→0.89→0.11), O103H2 (0.56→0.33→0.89).** Different features dominate at different λ. ATCC25922 wins at λ=0.1 because moderate denoising preserves its weak commensal signal; at λ=0.3 the signal collapses. O103H2 is the opposite — high λ unlocks something the moderate setting wasn't seeing yet. Both are signals of **per-strain optima**; a per-strain λ schedule (model selection by held-out fold) is a future-work entry.

3. **Cratered-at-λ=0.3 cells: 83972 (0.75→0.25), ATCC25922 (0.89→0.11).** These are the "easy" Non-STEC commensals. Their training-time recognition relies on file-id-correlated chemistry. High-λ DANN strips that, and the model loses commensal recognition entirely. **The headline implication:** DANN at λ=0.3 makes the model more careful about Non-STEC, hurting typical-commensal recognition (83972, ATCC25922) but helping atypical K-12. Beautiful biological / model-interaction story — the user's pre-DANN warning ("LOSO holds out the entire strain, so no in-distribution training file") gets resolved per-strain.

**Memprobe decoupling — final reading.** Tripling λ moved domain_acc during training from ~6-7% to ~6-7% (essentially the same) and memprobe top-1 from 14.0% to 13.6%. **Linear separability of file_id from the 32-dim feature space is preserved under DANN at both λ settings tested.** What DANN actually does is push file-id-encoding into low-variance directions while the high-variance directions encode strain. A LogReg probe finds whatever subspace works; DANN's effect doesn't register on linear-separability metrics. **The right diagnostic for "DANN is working" on this dataset is the LOSO per-strain pattern itself, not a from-scratch file-id classifier.**

**Operational decision: ship λ=0.1 as the headline DANN model.** λ=0.3 documented as the "high-pressure regime that strengthens pathogen biology at the cost of easy commensal recognition" — a future-work entry rather than a deployable model. PLS-DA stays as the strongest-LOSO-mean classical baseline in the README; vanilla CNN stays as the proof point that DANN preserved the K-12 + O157H7 biology wins it discovered.

**Compute actuals for λ=0.3:** Protocol A ~3.4 min, LOSO ~11.5 min, memprobe 0.6s. Same envelope as λ=0.1.
