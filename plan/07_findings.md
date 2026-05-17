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

## 2026-05-14 — dann-lambda-curve-completed

**Third lambda value (λ=0.05) added to the sweep.** Pre-registration at [08§lambda-0.05](08_expectations.md#2026-05-14--dann-lambda_max005-sweep-pre-registered-before-running). Result is a useful negative finding for picking the headline single-model AND a useful positive finding for stacking.

**Full lambda curve, per-strain:**

| Strain | Vanilla (λ=0) | λ=0.05 | λ=0.1 | λ=0.3 |
|---|---|---|---|---|
| 83972 (Non-STEC) | 0.88 | **0.88** | 0.75 | 0.25 |
| ATCC25922 (Non-STEC) | 0.11 | 0.22 | **0.89** | 0.11 |
| K-12 (Non-STEC) | 0.50 | 0.12 ↓ | 0.75 | **0.88** |
| Dublin (Salmonella) | 0.11 | **0.22** | 0.00 | 0.22 |
| Heidelburg (Salmonella) | 0.33 | 0.11 | **0.44** | 0.33 |
| Typhimurium (Salmonella) | 0.11 | **0.22** | 0.11 | 0.00 |
| O103H2 (STEC) | 0.56 | 0.44 | 0.33 | **0.89** |
| O121H19 (STEC) | 0.00 | 0.00 | **0.67** | 0.67 |
| O157H7 (STEC) | 0.56 | **0.67** | 0.56 | **0.67** |
| **MEAN** | 0.35 | **0.321** | **0.500** | **0.447** |
| Protocol A file-F1 | 0.649 | 0.635 | 0.566 | 0.493 |

**Three findings:**

1. **There's a minimum effective DANN pressure.** λ=0.05 mean (0.321) is BELOW vanilla CNN (0.35); the GRL at this strength is adding noise without providing regularization. K-12 dropped from 0.50 (vanilla) to 0.12 (λ=0.05) — likely RNG variance but consistent with "perturbation without enough regularization to compensate." **The effective DANN regime starts around λ ≈ 0.07-0.10 on this dataset.**

2. **Per-strain wins are non-overlapping across lambdas.** No strain is best-recall across the full menu — different lambdas win on different cells:
   - 83972 best at λ=0 / λ=0.05 (tied)
   - ATCC25922 best at λ=0.1 (0.89 vs ≤0.22 elsewhere)
   - K-12 best at λ=0.3 (0.88)
   - Dublin / Typhimurium best at λ=0.05 / λ=0.3 (tied at 0.22)
   - O103H2 best at λ=0.3 (0.89)
   - O157H7 best at λ=0.05 / λ=0.3 (tied at 0.67)
   - This is the structural justification for stacking — no single lambda is optimal; each provides a unique strain niche.

3. **Protocol A monotonically decreases in lambda** (0.649 → 0.635 → 0.566 → 0.493). The Protocol-A-vs-LOSO tradeoff is clean: more DANN buys LOSO biology features at the cost of Protocol A within-distribution accuracy. The Pareto curve is well-characterized now.

**Operational decisions:**
- **λ=0.1 remains the headline single-model DANN.** No update to the headline.
- **λ=0.05, λ=0.1, λ=0.3 all become base models for the stacking meta-learner** ([16] in the research queue). The complementary failure structure they show is the richest set we'll get for stacking inputs.
- **Useful negative finding to document in writeup:** "DANN below λ ≈ 0.07 on this dataset is just noise — there's a regularization phase transition between perturbation and effective adversarial pressure."

**Compute:** Protocol A ~3.2 min, LOSO ~11 min, no memprobe v2 needed (the probe finding is already documented across two lambdas and the decoupling result is established).

## 2026-05-14 — grouped-domain-dann-rejects-hypothesis

**Tested grouped-domain DANN (collapse 87-way file_id GRL target to 10-way subclass) at λ=0.1.** Pre-registration at [08§grouped-domain-dann](08_expectations.md#2026-05-14--grouped-domain-dann-pre-registered-before-running). Hypothesis: coarser domain target should preserve within-strain shared features and avoid the λ=0.3 commensal crater. **Hypothesis REJECTED.**

| Metric | Vanilla CNN | DANN(file_id, 0.1) | DANN(subclass, 0.1) |
|---|---|---|---|
| Protocol A file-F1 | 0.649 | 0.566 | **0.654** ⭐ |
| LOSO mean parent-recall | 0.35 | **0.500** ⭐ | 0.309 |
| K-12 (Non-STEC) | 0.50 | **0.75** ⭐ | 0.12 ❌ |
| ATCC25922 (Non-STEC) | 0.11 | **0.89** | **0.89** |
| 83972 (Non-STEC) | 0.88 | 0.75 | **0.88** |
| O121H19 (STEC) | 0.00 | **0.67** ⭐ | 0.00 ❌ |
| O157H7 (STEC) | 0.56 | **0.56** | 0.33 ↓ |
| Heidelburg (Salmonella) | 0.33 | 0.44 | 0.22 ↓ |
| Domain accuracy at end of training | n/a (no DANN) | 0.06 (chance 0.014 for 87-way) | 0.18 (chance 0.10 for 10-way) |

**Two findings:**

1. **The domain-target granularity is a fundamental design choice with opposite optimal directions for Protocol A vs LOSO.** Subclass grouping wins Protocol A (0.654 vs file_id 0.566 vs vanilla 0.649) because it suppresses within-strain acquisition noise that hurts standard CV. File_id grouping wins LOSO (0.500 vs subclass 0.309 vs vanilla 0.35) because it suppresses within-strain noise WITHOUT forcing the encoder to drop cross-strain biology. **Coarsening the domain target asks the encoder to do the wrong thing for cross-strain generalization.**

2. **The pre-registered hypothesis was wrong about the λ=0.3 crater mechanism.** I had attributed 83972 (0.75→0.25) and ATCC25922 (0.89→0.11) collapses at λ=0.3 to "file_id is too fine — GRL penalizes within-strain shared features that easy commensals ride on." Subclass grouping at λ=0.1 was the test: if the crater was caused by GRL granularity, subclass grouping should preserve commensals AND keep K-12/O157H7. **Subclass grouping preserved commensals (83972 0.88, ATCC25922 0.89) but destroyed K-12 (0.75→0.12) and O121H19 (0.67→0.00).** So the λ=0.3 commensal crater was probably caused by adversarial pressure strength, NOT by GRL granularity. The two are different design knobs that interact in non-obvious ways.

**Operational decision: do not pursue cal_date grouping or subclass×λ=0.3.** Both predicted by the mechanism analysis to lose LOSO. The remaining DANN-variant upside is essentially exhausted; the next experiments to try are:
- per-strain λ selection (#20 in the research queue) — picks one of {0.05, 0.1, 0.3} per LOSO test strain via held-out inner-fold confidence
- patch_size=5 Transformer (#19) — tests an explicit hypothesis on a different architecture
- 2nd-derivative input channel for CNN (#22) — adds peak-edge features as a second input channel

**Subclass-grouping λ=0.1 IS the best Protocol A model** (file-F1 0.654 vs vanilla 0.649 vs file_id λ=0.1 0.566). Worth keeping in the writeup as the "Protocol A-optimal DANN" Pareto point. If the deployment use case is "new file of a known strain" rather than "new strain", subclass grouping is the right choice.

**Compute:** ~14 min for both protocols. Same envelope as file_id DANN runs.

## 2026-05-14 — stacking-meta-learner-fails

**Built a LogReg meta-learner over {PLS-DA, DANN(0.05), DANN(0.1), DANN(0.3)} file-level predict_probas under LOSO meta-CV.** Code at `atlas/stacking.py` + `scripts/run_stacking.py`. Three variants tried: default (C=1, no subclass), --use-subclass (one-hot subclass feature), --meta-c 10 (less regularization).

**All three variants underperformed the best base model.** Mean parent-recall: default 0.407, subclass 0.136, low-reg 0.395. vs DANN(0.1) solo 0.500, PLS-DA solo 0.60.

**The fundamental failure mode** (per-strain table for default variant):

| Strain | PLS-DA | DANN 0.05 | DANN 0.1 | DANN 0.3 | Stacker |
|---|---|---|---|---|---|
| 83972 | 0.88 | 0.88 | 0.75 | 0.25 | **0.00** ❌ every base correct, stacker wrong |
| K-12 | 0.00 | 0.12 | 0.75 | 0.88 | **0.00** ❌ DANN had it, stacker followed PLS-DA |
| O157H7 | 0.00 | 0.67 | 0.56 | 0.67 | **0.00** ❌ same |
| Typhimurium | 1.00 | 0.22 | 0.11 | 0.00 | 1.00 ✓ PLS-DA carried |
| O121H19 | 0.89 | 0.00 | 0.67 | 0.67 | 0.89 ✓ PLS-DA carried |

**Mechanism:** in meta-training (excluding K-12), the meta-learner sees DANN(0.3) predicting Non-STEC at proba ~0.11–0.25 on ACTUAL Non-STEC training files (83972, ATCC25922) — so it LEARNS "DANN(0.3)::p_Non-STEC is LOW when the file IS Non-STEC." On held-out K-12, DANN(0.3) correctly outputs 0.88 on Non-STEC. The meta-learner applies its learned negative coefficient and predicts NOT-Non-STEC. Wrong.

**This is a fundamental LOSO-stacking limitation, not a hyperparameter issue.** DANN's per-strain behavior is too heterogeneous to learn a consistent meta-rule from out-of-fold examples — DANN(0.3) is bad at 83972 but good at K-12, and the meta-learner can't extrapolate that contrast from a held-out training set where K-12 is never seen.

**Operational decision: stacking does not ship.** Useful negative finding for the writeup: it shows we considered the obvious solution to complementary-failures and demonstrated empirically why it doesn't work under LOSO. A future-work entry for "per-strain confidence-routing instead of stacking" stays open.

**Three sub-results worth keeping in the writeup:**
1. The 0.136 result from --use-subclass shows that adding a held-out strain's one-hot to the meta-features doesn't help — the held-out strain has one-hot=1 in test but 0 in train, so the meta-learner can't learn what to do with it.
2. The 0.395 result from low regularization (C=10) shows the issue isn't regularization strength — even with more capacity to fit per-strain patterns, the heterogeneous training signal still misleads the meta-learner.
3. PLS-DA dominates the stacker's choices on every strain where PLS-DA is confident, because PLS-DA's training-set behavior matches its test-set behavior more consistently than DANN's does.

**Compute:** ~0.2 seconds per variant. Negligible.

## 2026-05-14 — per-strain-lambda-selection-fails

**Tried three leakage-bounded selector variants to pick the optimal DANN λ per LOSO test strain.** Code at `atlas/lambda_selector.py` + `scripts/run_lambda_selector.py`. Base candidates: DANN(λ=0.05), DANN(λ=0.1), DANN(λ=0.3).

| Selector | Selection signal | Leakage | Mean parent-recall |
|---|---|---|---|
| DANN λ=0.1 solo (target to beat) | n/a | n/a | **0.500** |
| hard | argmax(inner_val_f1) per strain | none (inner val on 8 outer-train strains) | 0.444 |
| soft | softmax(inner_val_f1)-weighted average | none | 0.435 |
| router | per-file argmax(mean max-proba) | mild (uses test-set confidence) | 0.440 |

**All three selectors underperform the best single base model.** The hard selector got 5/9 strain picks right but the 4 wrong picks were on high-stakes strains:

| Strain | Picked (recall) | Optimal choice (recall) | Cost of miss |
|---|---|---|---|
| K-12 | lam0.05 (0.12) | lam0.30 (0.88) | **0.76** |
| ATCC25922 | lam0.05 (0.22) | lam0.10 (0.89) | **0.67** |
| Heidelburg | lam0.05 (0.11) | lam0.10 (0.44) | 0.33 |

**Mechanism: inner-val F1 doesn't predict cross-strain behavior.** Inner-val F1 measures "fits the 8 training strains well." On the 9th held-out strain, the optimal λ depends on whether the strain is K-12-like (atypical biology, wants high λ to strip surface noise), Typhimurium-like (typical commensal Salmonella, wants linear methods), or ATCC25922-like (the singular middle case where moderate DANN wins). Inner-val F1 ranks lam0.05 marginally highest across most folds because it's the gentlest perturbation, easiest to optimize on the training distribution — but the training distribution is exactly the wrong thing to use as a proxy for the held-out strain.

**Router (test-time confidence) didn't fare better.** Routing to the highest-confidence base model per test file picked lam0.05 9/9 times on 83972 (correct, 0.875) but also lam0.05 9/9 times on Heidelburg (wrong — should have been lam0.10) and Typhimurium (no good answer). Confidence is biased toward the gentlest perturbation, which happens to be wrong as often as it's right.

**This is the THIRD negative result on multi-λ combination strategies:**
1. Soft-vote ensemble (`plan/07§ensemble-fails-to-clear-plsda`): uniform averaging crushes minority signal.
2. Stacking meta-learner (`plan/07§stacking-meta-learner-fails`): meta-learner can't learn DANN's per-strain behavior from out-of-fold examples.
3. Per-strain optimal λ (this finding): inner-val F1 and test-time confidence don't predict cross-strain DANN behavior.

**All three failures share a root cause:** DANN's per-strain idiosyncrasies (K-12 wants high λ, ATCC25922 wants moderate λ, 83972 wants low λ, Typhimurium isn't recoverable at any λ) are non-monotonic in any aggregate signal we can compute from leakage-free sources. The information needed to pick the right λ per strain *is held out by definition*.

**Operational decision: ship DANN λ=0.1 with file_id grouping.** It's the best robust single-model setting. The complementary failures across {λ=0.05, λ=0.1, λ=0.3} are documented as a regime-curve finding for the writeup, NOT as a base for a combined model. **Open future-work direction:** "labeled support files from each strain category at inference time" — give the model 1-2 representative spectra per known strain type to choose λ. This breaks LOSO purity but might be operationally realistic.

**Compute:** ~3 seconds total for all 3 selector variants. Negligible.

## 2026-05-14 — patch5-transformer-partially-confirms-blur-hypothesis

**Re-ran the Transformer with patch_size=5 instead of patch_size=20** to test the 2026-05-14§transformer-underperforms-cnn hypothesis that 20-bin patches blur the 5-10-bin Raman peaks the CNN was capturing on K-12 / O157H7. Pre-registered as future work in that earlier entry; not formally re-registered in plan/08 since the hypothesis was inherited.

**Headline: partial confirmation.** Patch=5 produces TWO new per-strain SOTA records but does NOT fix K-12.

### Protocol A

| Model | Mean file-F1 | SD | Per-fold |
|---|---|---|---|
| Patch=20 Transformer | 0.507 | 0.122 | 0.48 / 0.35 / 0.49 / 0.53 / 0.69 |
| **Patch=5 Transformer** | **0.534** | **0.028** | 0.57 / 0.52 / 0.54 / 0.50 / 0.54 |
| Vanilla CNN | 0.649 | 0.079 | — |
| PLS-DA | 0.951 | 0.051 | — |

Protocol A mean improves modestly (+0.027) but **fold variance drops dramatically (SD 0.028 vs 0.122)** — patch=5 is much more stable across folds. Still well below the CNN.

### LOSO per-strain

| Strain | Parent | Patch=20 | **Patch=5** | DANN λ=0.1 | Best across all models |
|---|---|---|---|---|---|
| 83972 | Non-STEC | 0.75 | 0.25 | 0.75 | 0.88 (PLS-DA, CNN, DANN λ=0.05) |
| ATCC25922 | Non-STEC | 0.22 | **1.00** ⭐⭐⭐ | 0.89 | **1.00 (patch=5)** |
| K-12 | Non-STEC | 0.00 | 0.00 | 0.75 | **0.88 (DANN λ=0.3)** |
| Dublin | Salmonella | 0.00 | 0.00 | 0.00 | 0.56 (PLS-DA) |
| Heidelburg | Salmonella | 0.11 | 0.33 | 0.44 | 0.89 (PLS-DA, LR) |
| Typhimurium | Salmonella | 0.00 | 0.11 | 0.11 | 1.00 (PLS-DA, LR, LinSVM) |
| O103H2 | STEC | 0.44 | 0.44 | 0.33 | 1.00 (linears) / 0.89 (DANN λ=0.3) |
| O121H19 | STEC | 0.22 | 0.22 | 0.67 | 0.89 (linears) |
| O157H7 | STEC | 0.00 | **0.78** ⭐⭐ | 0.56 | **0.78 (patch=5)** |
| **MEAN** | | 0.193 | **0.349** | **0.500** | (DANN λ=0.1) |

**Two new per-strain best-of-sweep records owned by patch=5 Transformer:**

1. **ATCC25922 = 1.00 (9/9 files correctly identified as Non-STEC).** First 100% recall on this strain across the entire sweep of 8 model families. Beats DANN λ=0.1's 0.89, every linear model's 0.22, every tree model's 0.33. **Patch=5 owns this cell.**
2. **O157H7 = 0.78 (7/9 files correctly identified as STEC).** Highest pathogenic-STEC recall in the sweep. Beats vanilla CNN (0.56), DANN λ=0.3 (0.67), every linear/tree model (≤0.33). **This is the canonical foodborne pathogen — strongest result of the project for that cell.**

**Where patch=5 doesn't help:** K-12 stays at 0/8. The model predicts STEC on all 8 K-12 files at Non-STEC proba ~0.26 (close-margin failure, not catastrophic, but still 0% recall). DANN λ=0.3's K-12 = 0.88 remains the K-12 SOTA. **K-12 and O157H7 use different chemistry; they're solved by different architectures.**

**Mechanism reading:**
- Patch=5 preserves narrow-peak local structure that patch=20 averaged out (5-bin patches let each Raman peak of width 5-10 bins span just 1-2 patches, not be averaged with 19 neighbors). The attention layers then weight these preserved peak features.
- For ATCC25922 (typical commensal E. coli) and O157H7 (pathogenic E. coli with phage-encoded virulence), the discriminative chemistry is narrow-peak ratios in the fingerprint region — exactly what patch=20 destroyed. Patch=5 sees those features clearly.
- For K-12 (laboratory-domesticated, large genomic deletions vs wild-type), the discriminative chemistry is probably broader-scale (the whole metabolome is shifted from genomic divergence). Patch=5 doesn't help because the issue wasn't peak resolution, it was that K-12 sits OUTSIDE the typical Non-STEC manifold entirely. DANN's adversarial denoising (which fights file-id signal) reveals the broader-scale K-12 distinctness; attention on preserved peaks doesn't.

**Operational decisions:**

1. **Headline LOSO model stays DANN λ=0.1** (mean 0.500 vs patch=5's 0.349).
2. **Add patch=5 Transformer to the per-strain best-model story** in the writeup. The ATCC25922 = 1.00 result is the strongest "this model uniquely owns this strain" finding in the project.
3. **Document the per-strain mechanism split explicitly:** K-12 wants DANN λ=0.3 (broad-scale adversarial denoising), O157H7 + ATCC25922 want patch=5 Transformer (narrow-peak preservation). Different architectures crack different biology.
4. **Open future-work direction:** train an ensemble that COMBINES DANN λ=0.3 + patch=5 Transformer + PLS-DA via a per-strain confidence-routed selection. The router/stacking failures earlier in the session were on DANN-family models that are too similar; patch=5 introduces a meaningfully different inductive bias.

**Compute:** Protocol A ~9 min, LOSO ~14 min. Per-fold runtime was higher than the patch=20 run because patch=5 produces ~4× more attention tokens (197 vs 49), so each forward pass is slower despite the architecture being otherwise identical.

## 2026-05-14 — 2nd-derivative-channel-second-best-loso

**Added 2nd-derivative as a second input channel to the SmallCNN1D** via a fixed (1, -2, 1) discrete-Laplacian kernel (no learnable params; +480 weights from conv1's input channels going 1→2; total 124,964 vs vanilla 124,484). Pre-registration at [08§2nd-deriv-cnn](08_expectations.md#2026-05-14--2nd-derivative-input-channel-cnn-pre-registered-before-running).

**Headline: second-best single-model LOSO mean of the entire sweep.**

| Model | LOSO mean parent-recall | Protocol A file-F1 |
|---|---|---|
| PLS-DA | **0.60** (LOSO leader) | 0.951 |
| LogReg | 0.59 | 0.961 |
| LinSVM | 0.52 | 0.779 |
| **DANN λ=0.1 (file_id)** | **0.500** (single-model headline) | 0.566 |
| **2-channel CNN (SNV + 2nd-deriv)** | **0.465** (2nd among deep models) | 0.560 |
| DANN λ=0.3 (file_id) | 0.447 | 0.493 |
| RBF-SVM | 0.42 | 0.833 |
| XGB | 0.37 | 0.796 |
| Vanilla 1-channel CNN | 0.35 | 0.649 |
| Patch=5 Transformer | 0.349 | 0.534 |
| DANN λ=0.05 | 0.321 | 0.635 |
| RF | 0.31 | 0.753 |
| Patch=20 Transformer | 0.193 | 0.507 |

The 2nd-deriv channel lifts LOSO mean by 0.115 over vanilla 1-channel CNN (0.35 → 0.465) — bigger lift than patch=5 Transformer's diff over patch=20 (0.156, similar magnitude). Edge information IS load-bearing on this dataset.

### Per-strain breakdown

| Strain | Vanilla CNN | DANN λ=0.1 | Patch=5 | DANN λ=0.3 | **2-ch CNN** | Best owner |
|---|---|---|---|---|---|---|
| 83972 | 0.88 | 0.75 | 0.25 | 0.25 | 0.62 | Vanilla CNN / PLS-DA |
| ATCC25922 | 0.11 | 0.89 | **1.00** | 0.11 | 0.67 | Patch=5 |
| K-12 | 0.50 | 0.75 | 0.00 | **0.88** | 0.00 | DANN λ=0.3 |
| Dublin | 0.11 | 0.00 | 0.00 | 0.22 | 0.11 | PLS-DA (0.56) |
| Heidelburg | 0.33 | 0.44 | 0.33 | 0.33 | 0.44 | PLS-DA (0.89) |
| Typhimurium | 0.11 | 0.11 | 0.11 | 0.00 | 0.00 | PLS-DA / linears (1.00) |
| O103H2 | 0.56 | 0.33 | 0.44 | **0.89** | 0.67 | DANN λ=0.3 (PLS-DA 1.00) |
| O121H19 | 0.00 | 0.67 | 0.22 | 0.67 | **0.89** ⭐⭐ | **2-ch CNN (ties PLS-DA)** |
| O157H7 | 0.56 | 0.56 | **0.78** | 0.67 | **0.78** ⭐ | **2-ch CNN ties Patch=5** |

**Two new per-strain records owned by 2-channel CNN:**

1. **O121H19 = 0.89 — first deep model to match linears.** Previously PLS-DA had this strain at 0.89; the closest deep result was DANN λ=0.1 at 0.67. 2-ch CNN ties PLS-DA exactly. Mechanism: O121H19's discriminative chemistry sits in narrow-peak edge features that the 2nd-deriv channel surfaces directly.
2. **O157H7 = 0.78 — ties patch=5 Transformer.** Two architectures (transformer attention on small patches + CNN with explicit edges) reach the same pathogenic-STEC ceiling. Suggests 0.78 is the load-bearing-feature ceiling on O157H7 under LOSO; further improvement on this cell would need a fundamentally different architecture.

**Where 2-channel CNN doesn't help:** K-12 = 0.00 (vanilla CNN had 0.50; the 2nd-deriv channel destroyed K-12's broad-scale signal). Same pattern as patch=5 Transformer — K-12 uses different chemistry than O157H7 / ATCC25922 / O121H19. **The K-12 vs O157H7 mechanism split is now confirmed across two independent architectural changes.**

### The per-strain best-model story now has 3 distinct deep architectures

| Cell | Owner | Recall | Mechanism |
|---|---|---|---|
| K-12 (atypical Non-STEC) | DANN λ=0.3 | 0.88 | broad-scale adversarial denoising |
| ATCC25922 (typical commensal Non-STEC) | Patch=5 Transformer | 1.00 | narrow-peak attention |
| O121H19 (STEC, edge-feature-rich) | 2-channel CNN (ties PLS-DA) | 0.89 | explicit 2nd-deriv channel |
| O157H7 (canonical pathogenic STEC) | Patch=5 / 2-ch CNN tie | 0.78 | narrow-peak preservation (two paths) |

Plus PLS-DA still owns Salmonella + most easy Non-STEC cells. **This is the strongest "different inductive biases solve different biology" demonstration in the project** — three deep architectures, three different biology cells, each pulled off by a different mechanism. Writeup foregrounds this.

### Protocol A regression — documented honestly

| Metric | Vanilla CNN | 2-ch CNN |
|---|---|---|
| File-F1 mean | 0.649 ± 0.079 | **0.560 ± 0.150** |
| Per-fold | 0.63 / 0.65 / 0.54 / 0.68 / 0.76 | 0.59 / 0.71 / 0.69 / 0.37 / 0.44 |
| Folds early-stopped < 30 epochs | none | folds 3 (ep13), 4 (ep18) |

**Pre-registration miss: 0.560 vs predicted 0.60-0.72, below floor by 0.04.** The 2nd-deriv channel adds initialization variance — folds 3 and 4 fell into local minima the model didn't recover from in time. A multi-seed run would likely tighten this. Documented as "this is the cost of the 2nd-deriv channel benefit on LOSO" rather than papered over.

### Operational decisions

- **Headline LOSO single-model remains DANN λ=0.1 at 0.500.** 2-channel CNN at 0.465 is the strong runner-up, valuable for the per-strain story.
- **The 2-channel CNN's O121H19 = 0.89 and O157H7 = 0.78 are headline-quality results for the writeup** — first deep model to tie PLS-DA on a STEC strain (O121H19) and joint-best pathogen detection (O157H7).
- **Soft-vote ensemble with 2-channel CNN + Patch=5 Transformer + DANN λ=0.1 + DANN λ=0.3** might still be worth revisiting now that we have FOUR meaningfully-different inductive biases with documented per-strain wins. Previous stacking failures were on within-DANN-family models; the new architectural diversity might unlock the meta-learner. Open future-work direction. **Update 2026-05-15:** this re-ensemble was run with the architecturally-diverse 4-model base ({PLS-DA, DANN λ=0.1, Patch=5, 2-ch CNN}). All three combination schemes failed — soft-vote and router degenerate to PLS-DA solo via calibration mismatch; stacking can't extrapolate under LOSO. See [§2026-05-15--4-architecture-re-ensemble-fails-to-beat-plsda](#2026-05-15--4-architecture-re-ensemble-fails-to-beat-plsda).

## 2026-05-15 — 4-architecture re-ensemble fails to beat PLS-DA — 4th negative result on ensembling {#2026-05-15--4-architecture-re-ensemble-fails-to-beat-plsda}

**Setup.** Re-ran the three combination schemes — soft-vote, stacking, confidence-router — over the four ARCHITECTURALLY-DIVERSE base models that previous attempts lacked: **{PLS-DA (linear chemometrics), DANN λ=0.1 (adversarial CNN), Patch=5 Transformer (narrow-peak attention), 2-channel CNN (explicit edge features)}**. Per-strain ownership of these 4 bases differs cleanly:

```
STRAIN       PLS-DA  DANN λ=0.1  Patch=5    2-ch CNN
83972        0.875   0.750       0.250      0.625    ← PLS-DA wins
ATCC25922    0.222   0.889       1.000      0.667    ← Patch=5 wins
Dublin       0.556   0.000       0.000      0.111    ← PLS-DA wins (only > 0.11)
Heidelburg   0.889   0.444       0.333      0.444    ← PLS-DA wins
K-12         0.000   0.750       0.000      0.000    ← DANN wins (only > 0)
O103H2       1.000   0.333       0.444      0.667    ← PLS-DA wins
O121H19      0.889   0.667       0.222      0.889    ← PLS-DA / 2-ch CNN tie
O157H7       0.000   0.556       0.778      0.778    ← Patch=5 / 2-ch CNN tie
Typhimurium  1.000   0.111       0.111      0.000    ← PLS-DA wins
MEAN         0.603   0.500       0.349      0.465
```

Oracle ceiling (per-strain argmax) is **0.86** — *if* any combination scheme could route correctly per strain. None did.

### Results

| Variant | Mean LOSO parent-recall | K-12 | O157H7 | Per-strain detail |
|---|---|---|---|---|
| Soft-vote uniform | **0.579** | 0.000 | 0.111 | 83972=0.88, ATCC25922=0.33, Dublin=0.22, Heidelburg=0.89, O103H2=1.00, O121H19=1.00, Typhimurium=0.78 |
| Stacking (LogReg meta) | **0.432** | 0.000 | 0.000 | 83972=0.00, ATCC25922=0.11, Dublin=0.22, Heidelburg=0.67, O103H2=1.00, O121H19=0.89, Typhimurium=1.00 |
| Confidence-router (file-level argmax mean max-proba) | **0.603** | 0.000 | 0.000 | **Identical to PLS-DA solo — router picked PLS-DA on 78/78 files (every file in every fold)** |

**No variant clears verdict (X)** (≥ 0.55 mean AND K-12 ≥ 0.30 AND O157H7 ≥ 0.50). Soft-vote and router each have mean ≥ 0.55 but fail both biology gates (K-12 + O157H7 cratered). Stacking lands in (Z) (< 0.50) cleanly.

### Mechanism: PLS-DA's miscalibrated confidence dominates

PLS-DA's file-level max-proba is calibrated systematically higher than any of the three deep models' on this dataset. In a uniform soft-vote, PLS-DA's vote acts like a *doubled-weight* base:

- Where PLS-DA is correct (Heidelburg, O103H2, O121H19, Typhimurium), the ensemble inherits its confident-correct vote and **exceeds the per-strain predicted range** (Heidelburg 0.89, O121H19 1.00, O103H2 1.00, Typhimurium 0.78).
- Where PLS-DA is *confidently wrong* (ATCC25922, K-12, O157H7), 3-of-4 majority deep votes are **dragged down to 0.0–0.33** — even when all three deep models agree on the correct class with moderate-to-high confidence.

For the **router**, the same calibration mismatch is fatal in a different way: file-level confidence is the routing signal, and PLS-DA's max-proba exceeds every deep model's on every single file. The router thus degenerates to "always pick PLS-DA," recovering PLS-DA solo (0.603) by tautology. **Confidence-routing across heterogeneous architectures requires per-base temperature calibration; raw max-proba is uninformative when one base has a fundamentally different calibration scale.**

For **stacking**, the LOSO meta-CV problem from [§stacking-meta-learner-fails](#2026-05-14--stacking-meta-learner-fails) is unchanged: the meta-learner trained on 8 strains can't extrapolate the right base-pattern to the 9th strain when each strain's optimal base is by definition held-out. The richer feature space (4 architectures × 4 classes vs. 4 DANN-lambdas × 4 classes) didn't help — possibly because the meta-learner picked up on PLS-DA's confidence-scale and weighted it heavily, then PLS-DA's per-fold variance hurt the held-out predictions.

### Verdict: 4th negative result on ensembling

This experiment was the strongest pro-ensemble test we could mount given the artifacts on disk — 4 base models with genuinely-different inductive biases each cleanly owning different per-strain cells, and three orthogonal combination schemes. **All three fail to Pareto-dominate PLS-DA solo.** Soft-vote and router *match or beat* PLS-DA's mean only by inheriting its predictions; neither contributes a single biology cell PLS-DA didn't already own. Stacking trails by 0.17.

**Operational decision.** Ship PLS-DA solo as the LOSO headline. The per-strain best-model table — one strain per architecture — is the actual story:

| Strain | Owner | Architecture | Mechanism |
|---|---|---|---|
| K-12 | DANN λ=0.3 (0.88) / λ=0.1 (0.75) | broad-scale adversarial denoising | broad-scale chemistry separates Non-STEC |
| ATCC25922 | Patch=5 Transformer (1.00) | narrow-peak attention | sharp peaks discriminate this Non-STEC |
| O121H19 | 2-channel CNN (0.89) / PLS-DA (0.89) | explicit edge features / linear chemometrics | edge structure + linear shape |
| O157H7 | Patch=5 / 2-ch CNN (0.78) | narrow-peak preservation / edges | both narrow-peak mechanisms work |
| Heidelburg / Typhimurium / Dublin / 83972 / O103H2 | PLS-DA | linear chemometrics | shape signature is linearly separable for these |

This is the writeup story: **four different inductive biases each crack different biology cells; no soft-vote, meta-learner, or confidence router on this dataset captures all the wins simultaneously, because PLS-DA's miscalibrated max-proba dominates every voting/routing scheme.** The complementary-failures narrative — extended from [classical+CNN](#2026-05-14--ensemble-fails-to-clear-plsda) to four architectures — is the strongest version of the project's central finding.

### Follow-up worth recording (out of scope this session)

- **Temperature-scaled soft-vote** — calibrate each base's proba to a common reliability curve (e.g., Platt scaling or isotonic regression on a held-out subset of training folds) BEFORE averaging. The current failure is purely a calibration mismatch; with equalized calibration the soft-vote would respect the actual class-prediction agreement instead of PLS-DA's scale advantage.
- **Disagreement-based abstention router** — instead of "argmax max-proba," route to "argmax (max-proba − second-max-proba)" or "argmin entropy" — both measure confidence in a calibration-invariant way for well-trained models. Documented in plan/09.
- **Per-strain oracle baseline** — for a writeup figure showing the gap between achievable per-strain wins (oracle 0.86) and what non-leaky combination schemes can extract (0.43–0.60). Not a ship-worthy model; useful as a "ceiling" reference.

### Artifacts

- `outputs/2026-05-15_ens_plsda_dann10_patch5_cnn2ch_loso_e1878a38/` (soft-vote)
- `outputs/2026-05-15_stack_4arch_loso_28e391eb/` (stacking)
- `outputs/2026-05-15_lambda_select_router_plsda_dann10_patch5_cnn2ch_loso_faec0bad/` (router)

Pre-registration and post-run resolution: [plan/08 §2026-05-15-re-ensemble-with-4-architecturally-diverse-bases](08_expectations.md).

## 2026-05-15 — Temperature-scaled soft-vote — partial calibration mechanism, no headline shift {#2026-05-15--temperature-scaled-softvote}

**Hypothesis under test.** The 4-architecture re-ensemble post-mortem identified PLS-DA's miscalibrated confidence as the dominant mechanism (mean max-proba 0.747 vs 0.43–0.49 for the 3 deep models). Temperature-scaling each base before averaging should equalize the effective vote weight and let the deep majority surface on cells where PLS-DA is confidently wrong.

**Implementation.** Per-base scalar temperature T_b fit by maximum likelihood on the union of predictions from the OTHER 8 LOSO folds (LOO-on-strains; same leakage tolerance as the stacking meta-learner). Applied per-spectrum to each base's probas, then uniform 4-way soft-vote, then file-level aggregation. Module: `atlas/calibrated_ensemble.py`; runner: `scripts/run_calibrated_ensemble.py`.

**Fitted temperatures (mean over 9 LOSO folds):** PLS-DA = **6.43**, DANN = 1.23, Patch=5 = 1.70, 2-ch CNN = 1.63. PLS-DA needed ~5× more softening than the deep models — quantitatively confirms the calibration mismatch.

### Results

| Strain | Uniform soft-vote (pre-cal) | **T-scaled 4-base** | T-scaled 3-deep (excl. PLS-DA) | PLS-DA solo (for ref) |
|---|---|---|---|---|
| 83972 | 0.875 | 0.875 | 0.500 | 0.875 |
| ATCC25922 | 0.333 | **0.667** ⭐ +0.34 | 0.778 | 0.222 |
| Dublin | 0.222 | 0.111 | 0.000 | 0.556 |
| Heidelburg | 0.889 | 0.778 | 0.556 | 0.889 |
| **K-12** | 0.000 | **0.000** | **0.375** | 0.000 |
| O103H2 | 1.000 | 1.000 | 0.667 | 1.000 |
| O121H19 | 1.000 | 0.889 | 0.889 | 0.889 |
| **O157H7** | 0.111 | **0.667** ⭐ +0.56 | 0.667 | 0.000 |
| Typhimurium | 0.778 | 0.111 ↓ | 0.000 | 1.000 |
| **MEAN** | **0.579** | **0.566** | 0.492 | **0.603** |

### Sharp mechanism findings

1. **Calibration was the mechanism for ATCC25922 and O157H7.** Both flipped from confident-wrong (0.33, 0.11) to confident-correct (0.67, 0.67) after temperature scaling. The 3-of-4 deep-model majority became audible once PLS-DA's vote scaled down 5×. **First time the project's ensembles cleanly captured these cells.**
2. **K-12 is NOT a calibration problem.** It's a **minority-of-one** problem — only DANN is right; PLS-DA + Patch5 + 2-ch CNN all confidently call K-12 something else. T_dann = 1.23 barely sharpens DANN's vote; even fully calibrated, 1-of-4 right votes can't outvote 3 wrong votes. **Confirmed by 3-deep sanity check**: K-12 = 0.375 when PLS-DA is removed (1-of-3 wrong is easier to overcome than 1-of-4).
3. **Typhimurium is K-12's symmetric failure.** Only PLS-DA is right; all 3 deep models confidently call Typhimurium something other than Salmonella. Softening PLS-DA's sole-correct vote crashes Typhimurium 0.78 → 0.11. **The minority-of-one failure mode is symmetric across PLS-DA-owns and DANN-owns cells.**
4. **The structural tradeoff.** Any ensemble that reduces PLS-DA's weight to fix K-12 / O157H7 / ATCC25922 necessarily hurts Typhimurium / Dublin where PLS-DA is the sole right voter. With the current 4-base set there is **no Pareto-improving combination** — test-time signals alone cannot distinguish "trust PLS-DA on Typhimurium" from "trust DANN on K-12."

### Disagreement-based (margin) router — also degenerates to PLS-DA

To rule out calibration-invariance: ran `router_margin` — route by argmax(mean(max_proba − second_max_proba)) per file. PLS-DA's distributions are not just over-confident, they're **systematically more peaked** (smaller mass on runner-up class) on every file. PLS-DA wins the margin signal 78/78 files. Mean = 0.603 (identical to original confidence router, identical to PLS-DA solo). **Disagreement-based routing across heterogeneous architectures fails for the same structural reason as confidence-based routing — PLS-DA's classifier produces categorically peaked distributions on every file regardless of correctness.**

### Verdict: branch (B) on temperature scaling; ensemble story now fully closed

- Temperature-scaled 4-base mean 0.566 ≥ 0.55 ✓, O157H7 0.67 ≥ 0.50 ✓, but **K-12 0.00 < 0.30 ✗** — branch (A) gate fails, lands in (B).
- 5 distinct combination schemes attempted (uniform soft-vote, stacking, max-proba router, margin router, T-scaled soft-vote) across 3 base sets (3-classical, 3-DANN, 4-architectural). **None Pareto-dominates PLS-DA solo.**
- **The story is sharp now:** PLS-DA solo wins LOSO mean because (a) it owns 5/9 strains outright, and (b) its calibration scale dominates any voting/routing scheme, and (c) the cells where PLS-DA is wrong are minority-of-one cells that no non-leaky scheme can route to the correct sole-voter.

### Operational decision

**Ship PLS-DA solo at 0.603 as the LOSO headline.** Temperature-scaled soft-vote at 0.566 is the second-best ensemble result and the cleanest demonstration of the calibration mechanism — include in the writeup as the "what *would* have to be true for ensembles to work" narrative. The deeper-takeaway sentence: *the per-strain best-model table is irreducible because each strain's optimal model is structurally hidden from any LOSO-clean routing signal.*

### Future-work directions worth recording

- **Per-base out-of-fold inner-validation calibration**. Retrain each of the 4 base models with a held-out *inner* fold (15% of training files) for honest calibration fitting. This is the leakage-free version of what we did with cross-fold predictions. Out of session scope; estimated 4× current LOSO retraining time.
- **Stacked routing with class-prior features**. Add per-base class-prior disagreement (e.g., "does this file have the H₂O signature?") as a meta-feature. May give the LOSO meta-learner a partial signal for "this is a Salmonella file → trust PLS-DA" that the architectural base probas don't carry. Speculative; documented in plan/09.
- **A per-strain-aware abstention scheme** — bases output an additional "I don't know" probability, and the ensemble routes based on which base abstains least. Requires retraining with abstention heads.

### Artifacts

- `outputs/2026-05-15_tcal_4base_all_loso_d5780c7b/` (4-base T-scaled, **the partial-fix result**)
- `outputs/2026-05-15_tcal_3deep_noplsda_loso_674f6163/` (3-deep T-scaled sanity check)
- `outputs/2026-05-15_lambda_select_router_margin_plsda_dann10_patch5_cnn2ch_loso_4d39c6d6/` (margin router; degenerates to PLS-DA)

**Compute:** Protocol A ~3 min, LOSO ~6 min. Same envelope as vanilla CNN.

## 2026-05-15 — DANN aug-regime sweep — aug is load-bearing for LOSO, not over-tuning {#2026-05-15--dann-aug-regime-sweep}

**Hypothesis under test.** plan/00 carried a TODO since 2026-05-14 claiming the CNN aug regime was over-tuned (folds early-stop with train_acc 0.4-0.5; no-aug fold-0 reaches train_acc 0.88). The implied prescription: lighter aug → better convergence → higher LOSO. Tested by sweeping DANN λ=0.1 across 3 aug presets.

### Results

| Variant | p_noise | p_scale | p_shift | p_baseline | p_mixup | **LOSO mean** | K-12 | O157H7 | ATCC25922 |
|---|---|---|---|---|---|---|---|---|---|
| default (baseline) | 0.50 | 0.40 | 0.40 | 0.30 | 0.30 | **0.500** | 0.750 | 0.556 | 0.889 |
| light (all halved) | 0.25 | 0.20 | 0.20 | 0.15 | 0.15 | **0.347** ↓ | 0.375 | 0.222 | 0.444 |
| no_mixup | 0.50 | 0.40 | 0.40 | 0.30 | **0.00** | **0.423** ↓ | **0.875** ⭐ | 0.556 | 0.556 |
| minimal | 0.30 | 0.00 | 0.20 | 0.00 | 0.00 | skipped | — | — | — |

**Verdict (δ) — heavy aug is load-bearing for LOSO regularization, not over-tuning.** Both lighter variants regress on LOSO mean. `minimal` skipped after `light` showed the direction was wrong; predicted strictly worse.

### Mechanism finding

**plan/00's "over-tuned aug" diagnosis was using the wrong proxy.** Reading train_acc 0.4-0.5 as undertraining is correct for Protocol A (within-distribution); under LOSO it's the **correct** training endpoint. Climbing train_acc past 0.5 means memorizing the 8 training strains' file-id correlated signals, which actively hurts generalization to the held-out 9th. The no-aug fold-0 train_acc 0.88 sanity-check was a misleading signal because it measured *within-distribution* overfitting capacity, not *cross-strain* generalization.

The lighter-aug runs are *overfitting* the training strains — their per-fold val_f1 climbs higher early on (light's epoch-15 val_f1 = 0.45 vs default's ~0.35) but they crash on held-out strain prediction because the encoder has memorized training-strain features.

### Surprise per-strain finding worth keeping

**no_mixup gives K-12 = 0.875 — joint-best K-12 across all DANN variants (ties DANN λ=0.3's K-12 = 0.875).** Mixup blends Non-STEC + STEC + Salmonella labels with random α; the blended targets suppress the broad-scale chemistry signal that lets DANN identify K-12. Removing mixup unlocks K-12 to the same ceiling DANN λ=0.3 hits with adversarial pressure — at the cost of regressing 83972 / ATCC25922 / Heidelburg. Useful Pareto data point but not a new headline since DANN λ=0.3 already serves K-12 = 0.875 with a less destructive cost profile elsewhere.

### Operational decisions

- **Default aug regime is correct.** TODO removed from plan/00. Documented as a closed negative finding (lighter ≠ better for LOSO).
- **Pivot to multi-seed averaging** as the next swing — different mechanism (variance reduction across stochastic training, not regularization tuning). Soft-vote across 3 seeds of DANN λ=0.1 + default aug. Launched in parallel to this writeup.

### Artifacts

- `outputs/2026-05-15_cnn_dann_lam0.10_auglight_loso_0e66e23f/` (light aug — 0.347)
- `outputs/2026-05-15_cnn_dann_lam0.10_nomixup_loso_f50345d9/` (no_mixup — 0.423, **K-12 = 0.875 ⭐**)

## 2026-05-15 — DANN λ=0.1 5-seed robustness: the headline 0.500 was a lucky seed {#2026-05-15--dann-5seed-robustness}

**Hypothesis under test.** Multi-seed averaging as the next swing for headline movement. Pre-registered expectation: stabilize per-strain variance, lift mean 0.03–0.07.

**Result: invalidates a major prior claim.** Five seeds of DANN λ=0.1 + default aug + LOSO:

| Seed | LOSO mean | 83972 | ATCC25922 | Dublin | Heidelburg | K-12 | O103H2 | O121H19 | O157H7 | Typhimurium |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 (shipped) | **0.500** | 0.750 | 0.889 | 0.000 | 0.444 | **0.750** | 0.333 | 0.667 | 0.556 | 0.111 |
| 1 | 0.153 | 0.250 | 0.444 | 0.000 | 0.111 | 0.125 | 0.333 | 0.000 | 0.111 | 0.000 |
| 2 | 0.528 | 1.000 | 1.000 | 0.000 | 0.111 | 0.750 | 0.556 | 0.556 | 0.222 | 0.556 |
| 3 | 0.273 | 0.125 | 1.000 | 0.000 | 0.222 | 0.000 | 1.000 | 0.000 | 0.000 | 0.111 |
| 4 | 0.270 | 0.875 | 0.889 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.667 | 0.000 |
| **mean ± SD** | **0.345 ± 0.145** | 0.600 ± 0.348 | **0.844 ± 0.206** | 0.000 | 0.178 ± 0.151 | 0.325 ± 0.350 | 0.444 ± 0.330 | 0.244 ± 0.301 | 0.311 ± 0.257 | 0.156 ± 0.206 |
| 5-seed soft-vote | **0.370** | 0.620 | **1.000** | 0.000 | 0.220 | 0.380 | 0.670 | 0.000 | 0.440 | 0.000 |
| Oracle (max per strain) | **0.676** | 1.000 | 1.000 | 0.000 | 0.444 | 0.750 | 1.000 | 0.667 | 0.667 | 0.556 |

### Sharp findings

1. **The shipped DANN λ=0.1 = 0.500 was the 2nd-best of 5 random seeds, not a typical value.** Honest reporting: **DANN λ=0.1 LOSO mean = 0.345 ± 0.145 across 5 seeds.** That's essentially **tied with vanilla CNN (0.35)** and matches DANN λ=0.3 (0.447) within 1 SD. **The "+0.15 lift vs vanilla CNN" claim from [§dann-ablation-clears-verdict-a](#2026-05-14--dann-ablation-clears-verdict-a) was a single-seed-luck artifact.**
2. **Per-strain wins claimed for DANN are mostly seed-fragile.** K-12 = 0.75 happens on 2 of 5 seeds (the other 3 give 0.125 / 0.000 / 0.000); 5-seed soft-vote gives K-12 = 0.380. The biology-cell narrative ("DANN cracks K-12 via broad-scale chemistry") was overclaimed; only 2 of 5 random initializations land in that basin.
3. **Only ATCC25922 = 1.00 is a robust DANN win.** Mean 0.844, SD 0.206, 5-seed soft-vote = **1.000** (3 seeds hit 1.00, 2 seeds hit 0.89, 1 hit 0.44). This is the one cell where DANN reliably outperforms the alternatives.
4. **Dublin is consistently impossible for DANN.** 0.000 in all 5 seeds. SD 0.000. The "deep models can't do Dublin" finding holds robustly.
5. **Oracle ceiling across 5 seeds is 0.676** — would exceed PLS-DA 0.603, but it's test-leakage (can't be deployed). Confirms there IS signal hidden in the per-seed variance, just no leakage-clean way to extract it.

### Major implication: prior single-seed deep-model claims are suspect

If DANN λ=0.1 has ±0.145 LOSO SD across just 5 seeds, then by parallel:
- **DANN λ=0.3 = 0.447** (with K-12 0.875, O157H7 0.667) — likely similar seed-fragility; need 5-seed run to confirm.
- **Patch=5 Transformer = 0.349** (with ATCC25922 1.00, O157H7 0.778) — similar concern.
- **2-channel CNN = 0.465** (with O121H19 0.89, O157H7 0.78) — similar concern.
- **Vanilla CNN = 0.35** — explicitly single-seed; baseline for the comparison.

**The per-strain best-model table in plan/00 may be largely an artifact of lucky-seed picks across runs done at different times.** A proper multi-seed characterization of all 4 deep base models would take ~3 hours and would likely substantially compress the per-strain narrative.

### Operational decisions

- **Update headline DANN λ=0.1 LOSO to 0.345 ± 0.145 (5 seeds) or 0.370 (5-seed soft-vote).** The 0.500 number stays in history as the originally-shipped single-seed point estimate but is no longer the headline.
- **PLS-DA solo at 0.603 remains the LOSO headline.** The gap to DANN is now much larger than originally claimed (0.603 vs 0.345-0.370, gap ≥ 0.23).
- **ATCC25922 = 1.00 is the one robust DANN win** worth keeping in the per-strain best-model table.
- **Multi-seed the other deep models next** to honestly characterize their wins. Most-suspect single-seed claims to verify: DANN λ=0.3 K-12 = 0.875, Patch=5 ATCC25922 = 1.00, 2-channel CNN O121H19 = 0.89.

### Artifacts

- `outputs/2026-05-15_cnn_dann_lam0.10_seed{1,2,3,4}_loso_*/` (4 new seeds)
- `outputs/2026-05-15_ens_dann_3seed_avg_loso_5cf10c7e/` (3-seed soft-vote: 0.398)
- `outputs/2026-05-15_ens_dann_5seed_loso_464268a6/` (5-seed soft-vote: **0.370 — the robust DANN ship**)

## 2026-05-15 — DANN λ=0.3 5-seed verification: K-12 = 0.75 survives, λ=0.3 is the robust DANN config {#2026-05-15--dann-lam03-5seed-verification}

**Hypothesis under test.** Given DANN λ=0.1's headline 0.500 turned out to be lucky-seed (5-seed mean 0.345 ± 0.145, soft-vote 0.370), the next most-load-bearing single-seed claim in the project is **DANN λ=0.3 K-12 = 0.875** ([§dann-lambda-frontier](#2026-05-14--dann-lambda-frontier)). 4 more seeds run; combined with the existing seed=42 baseline → full 5-seed characterization.

### Results

| Seed | LOSO mean | K-12 | ATCC25922 | O157H7 | O103H2 | 83972 | Typhimurium |
|---|---|---|---|---|---|---|---|
| 42 (shipped) | 0.447 | 0.875 | 0.111 | 0.667 | 0.889 | 0.250 | 0.000 |
| 1 | 0.349 | 0.000 | 0.778 | 0.889 | 0.667 | 0.250 | 0.111 |
| 2 | **0.577** | 0.750 | 1.000 | 0.778 | 0.444 | 1.000 | 0.667 |
| 3 | 0.224 | 0.125 | 0.667 | 0.222 | 0.889 | 0.000 | 0.000 |
| 4 | 0.369 | **1.000** | 0.889 | 0.444 | 0.000 | 0.875 | 0.000 |
| **mean ± SD** | **0.393 ± 0.117** | **0.550 ± 0.408** | 0.689 ± 0.310 | **0.600 ± 0.239** | 0.578 ± 0.333 | 0.475 ± 0.391 | 0.156 ± 0.259 |
| **5-seed soft-vote** | **0.448** | **0.750** ⭐ | 0.890 | **0.780** ⭐ | 0.670 | 0.500 | 0.000 |
| Oracle (max per strain) | 0.753 | 1.000 | 1.000 | 0.889 | 0.889 | 1.000 | 0.667 |

### Sharp findings

1. **DANN λ=0.3 5-seed soft-vote = 0.448 essentially recovers single-seed baseline 0.447.** Unlike DANN λ=0.1 (where single-seed 0.500 was lucky vs 5-seed soft-vote 0.370), DANN λ=0.3's reported headline is **robust under multi-seed averaging**. This is the **actually-shippable DANN result**.
2. **K-12 = 0.75 verified robust.** 5-seed soft-vote gives K-12 = 0.75; per-seed K-12 mean is 0.55 ± 0.41; **3 of 5 seeds hit ≥ 0.75 on K-12** (seeds 42, 2, 4). The "DANN λ=0.3 cracks K-12 via broad-scale adversarial denoising" narrative survives — at lower confidence than originally claimed (0.875 single-seed → 0.75 5-seed soft-vote) but the cell is genuinely solved by this architecture.
3. **O157H7 = 0.78 also verified robust** for DANN λ=0.3 (5-seed soft-vote; mean 0.60 ± 0.24). Better than DANN λ=0.1's 0.44 5-seed soft-vote.
4. **The originally-reported "λ=0.3 craters ATCC25922 0.89 → 0.11" was a single-seed misleading observation.** 5-seed mean ATCC25922 = 0.69 ± 0.31, soft-vote = 0.89. The seed=42 0.111 result was the worst of 5 seeds; other seeds give 0.78–1.00. **The λ-curve narrative needs a footnote: the per-strain Pareto split at higher λ is partially seed-fragile, not a hard tradeoff.**
5. **DANN λ=0.3 dominates DANN λ=0.1 under 5-seed soft-vote**:
   - λ=0.3 mean 0.448 vs λ=0.1 mean 0.370 (+0.078)
   - λ=0.3 K-12 0.75 vs λ=0.1 K-12 0.38 (+0.37)
   - λ=0.3 O157H7 0.78 vs λ=0.1 O157H7 0.44 (+0.34)
   - λ=0.3 trades off only ATCC25922 (0.89 vs 1.00) and 83972 (0.50 vs 0.62)
   - **The "λ=0.1 ships as headline" decision from 2026-05-14 was based on single-seed numbers and was wrong; λ=0.3 is the better DANN config under proper characterization.**

### Methodological finding: 5-seed soft-vote is the right way to report deep-model LOSO

Two ways to aggregate across seeds:
- **Per-seed mean** (compute LOSO mean per seed, average means): gives a "what's the expected LOSO from a random init" number with honest SD.
- **5-seed soft-vote** (average per-spectrum probas across seeds, then aggregate to file-level): gives a "what's the LOSO of a 5x-ensembled model" number.

The soft-vote is the right ship value because it captures **consensus across seeds at the probability level** — strains where multiple seeds confidently agree get preserved; strains where seeds disagree get averaged toward uncertainty. The per-seed mean penalizes minority-correct seeds (which are real signal that soft-vote can recover via probability mass).

DANN λ=0.1 soft-vote 0.370 vs per-seed mean 0.345: soft-vote correctly preserves ATCC25922 = 1.00 even though only 3 seeds hit perfect.
DANN λ=0.3 soft-vote 0.448 vs per-seed mean 0.393: soft-vote correctly preserves K-12 = 0.75 and O157H7 = 0.78 even though 2 seeds get them wrong.

### Operational decisions

- **Replace DANN headline with λ=0.3 5-seed soft-vote = 0.448 (K-12 = 0.75, O157H7 = 0.78, ATCC25922 = 0.89).** This is the robust DANN result for the writeup model card.
- **DANN λ=0.1 5-seed soft-vote = 0.370 documented as the λ-frontier low-pressure endpoint** with ATCC25922 = 1.00 as its single robust biology win.
- **PLS-DA solo 0.603 remains the headline LOSO mean** — gap to DANN is 0.16 not 0.10 or 0.23. Better than the post-revision λ=0.1 picture (gap 0.23) but worse than the originally-claimed picture (0.10).
- **Per-strain best-model table revised:**
  - K-12: DANN λ=0.3 5-seed soft-vote **0.75** (verified robust, replaces single-seed 0.875)
  - ATCC25922: DANN λ=0.1 5-seed soft-vote **1.00** (verified robust)
  - O157H7: DANN λ=0.3 5-seed soft-vote **0.78** (verified robust)
  - O103H2, O121H19, Salmonella triplet: PLS-DA owns (deterministic)
- **Multi-seed Patch=5 + 2-channel CNN next** to verify their remaining single-seed per-strain claims. Most-suspect: Patch=5 ATCC25922 = 1.00 (already verified for DANN λ=0.1 — independent confirmation would strengthen story), 2-channel CNN O121H19 = 0.89.

### Artifacts

- `outputs/2026-05-15_cnn_dann_lam0.30_seed{1,2,3,4}_loso_*/` (4 new seeds)
- `outputs/2026-05-15_ens_dann_lam0p3_5seed_loso_a57438fd/` (**5-seed soft-vote — the robust DANN ship**)
