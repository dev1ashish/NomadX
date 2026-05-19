# Models Summary — NomadX Raman Classification

**Short version: no CNN or transformer. Classical ML only.**

## Models tried

- **PLS-DA** (Partial Least Squares Discriminant Analysis) — headline model
- **Logistic Regression (L2-penalized)** — engineered-feature model
- **CNN** — 1D conv net, multiple variants (vanilla, 2-channel SNV+2nd-derivative, DANN domain-adapted)
- **Transformer** — 1D patch-Transformer (~217K params), tested at patch=20 and patch=5
- Classical baselines: kNN, Random Forest, XGBoost, linear SVM (none beat PLS-DA)

## Task

4-class classification on Raman spectra: **STEC / Non-STEC / Salmonella / H₂O**.

Data: 87 files, 7,122 QC-passed spectra, 987 wavenumber bins.

## How classification is done

Step-by-step:

1. **Acquire** — Raman spectrum (Raman shift in cm⁻¹ vs intensity) from a sample on the slide.
2. **QC filter** — drop low-SNR / saturated spectra. Kept 7,122 of the raw set.
3. **Preprocess** —
   - Baseline subtraction (removes fluorescence background curve)
   - SNV normalization (standardizes intensity so files are comparable)
4. **Bin** — resample to 987 fixed wavenumbers.
5. **Featurize** — two paths:
   - (A) Use the raw 987 bins as-is.
   - (B) Engineer 35 features: peak fits (height, width, position), 1st/2nd derivatives, ROI moments over biology-meaningful bands (protein, nucleic acid, lipid, LPS).
6. **Classify** —
   - **PLS-DA** projects the spectrum into a few latent components that maximize separation between the 4 classes, then assigns the closest class.
   - **LogReg-L2** computes a weighted sum of features per class, softmax → probabilities, argmax → class.
7. **Validate with LOSO** = Leave-One-Sample-Out (hold out a whole *file*, train on the rest, repeat for every file). This is the strict, no-leakage number — within-file accuracy is much higher but doesn't generalize.

```
Raman spectrum  →  preprocess (baseline + SNV)  →  features  →  classifier  →  class
                                                     │
                            ┌────────────────────────┴────────────────────────┐
                       (A) raw 987 bins                              (B) 35 engineered features
                            ↓                                                  ↓
                          PLS-DA                                       LogReg-L2 (Stage 15F)
```

## Numbers

| Model     | Features                | LOSO accuracy                  |
| --------- | ----------------------- | ------------------------------ |
| **PLS-DA**| raw 987 bins            | **0.603** ← project best       |
| LogReg-L2 | 35 engineered features  | 0.448 (95% CI 0.345–0.552)     |
| PLS-DA    | same 35 engineered feats| lower than LogReg (McNemar p = 0.0020) |

## Translation

On the same engineered features, LogReg beats PLS-DA — but **PLS-DA on the raw spectrum still wins overall**. Feature engineering hit a plateau (Branch C).

## Why CNN and Transformer didn't win

**We DID build both. They just didn't beat PLS-DA on LOSO.** Here are the actual numbers:

| Model variant                       | Params | LOSO mean parent-recall   | vs PLS-DA 0.603         |
| ----------------------------------- | ------ | ------------------------- | ----------------------- |
| Vanilla CNN                         | ~124K  | 0.35                      | ❌ −0.25                 |
| 2-channel CNN (SNV + 2nd-deriv)     | ~124K  | 0.465                     | ❌ −0.14 (2nd-best deep) |
| DANN-adapted CNN (λ=0.1)            | ~124K  | 0.345 ± 0.145 (5-seed)    | ❌ −0.26                 |
| Patch=20 Transformer                | ~217K  | 0.193                     | ❌ −0.41 (weakest arm)   |
| Patch=5 Transformer                 | ~217K  | 0.349                     | ❌ −0.25                 |
| **PLS-DA (linear, ~thousand params)** | tiny | **0.603**                 | ✅ headline              |

### Why classical beat deep on this dataset

1. **Data is small.** 7,122 spectra across only 87 files. Deep models want 10×–100× more *biological diversity*, not more spectra-per-sample. PLS-DA's linear projection regularizes itself; CNN/Transformer have nothing to lean on.
2. **LOSO is brutal.** GroupKFold → LOSO drops macro-F1 by ~0.55–0.65 across every model. Deep models latch onto training-strain features (cell wall variation, calibration-date batch effects) that don't transfer to a held-out strain. Linear models can't fit those quirks in the first place.
3. **Transformer patches blur Raman peaks.** Raman peaks are 5–10 bins wide. Patch=20 averages each peak with its neighborhood *before* attention sees it — that's why patch=20 collapsed to 0.193. Patch=5 helped (0.349) but still below classical.
4. **DANN's K-12 win was a lucky-seed artifact.** Single-seed showed K-12 = 0.75; 5-seed honest characterization is 0.325 ± 0.41 — only 2 of 5 random inits land in that basin. The headline LOSO mean revised from 0.500 → 0.345 ± 0.145 after multi-seed.
5. **STEC vs Non-STEC virulence is mostly plasmid-encoded.** The Shiga toxin genes don't produce a strong cell-wall Raman signature. There may genuinely not be enough signal for a deep model to extract that PLS-DA isn't already finding.

### What the deep models DID prove

Even though none shipped, they cracked specific hard folds that PLS-DA misses:

- **DANN-CNN** → K-12 (Non-STEC) recall 0.75 (PLS-DA: 0.00)
- **2-channel CNN** → O121H19 (STEC) recall 0.89 (first deep model to tie PLS-DA)
- **Patch=5 Transformer** → ATCC25922 = 1.00 (9/9 correct, first 100% on this strain)
- **Patch=5 + 2ch-CNN** → O157H7 (canonical pathogenic STEC) recall 0.78

This is the writeup story: **"different inductive biases fail on different strains."** No single model wins all 9 LOSO folds — they're complementary failures, not redundant successes. The planned next step (plan/13) is SSL pretraining on unlabeled spectra to give a CNN/Transformer the data diversity it needs.

## Where accuracy is HIGH

- **Within-file accuracy** (train + test on the same sample) is very high — but that's leakage and not a real-world number.
- **H₂O vs anything** is easy in principle — water has no biological Raman signal, so it should be the cleanest class.
- **E. coli vs Salmonella** has a real spectral signal: different LPS structure and protein content. The engineered feature `spat_skew_lps_1117` (LPS band 1117 cm⁻¹ asymmetry) gives a strong split (Cohen's d = +0.725).
- **Macromolecule-grouped bands** (protein 1338 / 1454 / 1658 cm⁻¹) carry most of the discriminative signal — confirmed by mutual-information ranking.

## Why accuracy is LIMITED (~60%, not ~95%)

1. **LOSO is strict.** A whole file (= one physical sample, one acquisition session) is held out. Sample-to-sample variability in Raman is huge: focus drift, laser power, slide position, biological variation between cultures. Most published "high-accuracy" Raman numbers are within-file or random-split — not LOSO.
2. **STEC-default bias.** The model defaults to predicting STEC when uncertain. The real LOSO confusion matrix shows **all 8 held-out H₂O spectra got misclassified as STEC** — a clear class-prior / decision-boundary issue.
3. **STEC vs Non-STEC is genuinely hard.** Both are *E. coli*; they differ only in toxin-related genes whose Raman signature is subtle. Most of the LOSO error sits on this pair.
4. **Mixed samples lose another 10–20%.** Per Atlas briefing, real-world mixed cultures drop accuracy further than single-strain LOSO already shows.
5. **Feature-engineering plateau.** 35 hand-crafted features couldn't beat the raw 987 bins. MCR (matrix factorization) features looked great in global fit (d = −1.23) but **didn't survive per-fold mutual information** — the global fit was partly a leakage artifact.
6. **No deep model yet.** No CNN, no transformer, no self-supervised pretraining. That's the planned next step (plan/13, SSL pivot) — the expectation is that a CNN pretrained with SSL on unlabeled spectra should learn STEC/Non-STEC features that hand-engineering missed.

## TL;DR

- **PLS-DA on raw spectra = 0.603 LOSO** ships as the headline.
- CNN and Transformer **were built and tested** (vanilla CNN, 2-channel CNN, DANN-CNN, patch=20 Transformer, patch=5 Transformer) — best deep result 0.465, all below PLS-DA.
- Deep underperforms because the dataset is small (87 files), LOSO is strict, and Raman peaks are narrow (Transformer patches blur them).
- Each deep model cracks a different hard strain (DANN→K-12, Transformer→ATCC25922, 2ch-CNN→O121H19) but none wins overall.
- SSL-pretrained CNN/Transformer is the planned next move (plan/13).
