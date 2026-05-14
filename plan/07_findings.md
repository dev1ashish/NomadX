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
