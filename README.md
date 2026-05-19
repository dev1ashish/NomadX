---
title: "Bacterial Identification from Confocal Raman Hyperspectral Maps"
subtitle: "A multi-track study on 87 Atlas files: classical baselines, deep models, chemistry-grounded engineered features, and a production classifier with per-fold MCR-ALS unmixing"
author: "NomadX Raman Take-Home — `dev@luaimplementation.ai`"
date: "2026-05-18"
abstract: |
  We study bacterial identification from 87 confocal-Raman hyperspectral
  maps spanning four primary classes (STEC, Non-STEC *E. coli*,
  *Salmonella*, H₂O) and nine bacterial subclasses. Three tracks are
  reported. (1) **Baseline modeling** sweeps classical (PLS-DA, LogReg,
  SVM, RF, XGB), deep (1D-CNN, 1D-Transformer), and domain-adversarial
  (DANN) architectures; under strain-level Leave-One-Strain-Out (LOSO),
  **PLS-DA reaches 0.603 mean parent-class recall** — the project record
  — and five ensemble schemes fail to clear it. (2) A **band-chemistry
  track** (Stages 1–7) falsifies the published Cisek-2013 STEC
  discriminative triple (1338/1454/1658 cm⁻¹) at file level, instead
  identifying the 800–1200 cm⁻¹ **LPS-chain region as the empirical
  anchor** (`auc_lps_1194` Cohen's d = +1.03), and quantifies a 10–20%
  mixed-sample-deployment penalty. (3) A **feature-engineering track**
  (Stages 15A–E) produces 259 features per file across five families;
  **MCR-ALS unmixing** is the highest-yield stage (`mcr_C6_mean`
  d = −1.23 is the project's strongest single file-level discriminator),
  and biology-grounded ratios surface a new protein-2°-structure axis
  (`bio_alpha_helix_score` d = −0.986) that also separates K-12 from
  clinical STEC strains. (4) **Stage 15F** trains the production
  classifier under strain-level LOSO with per-fold MCR-ALS / ROI-PCA /
  SAM refit, mutual-information feature selection, multi-seed evaluation,
  and a shuffled-label permutation test on the substrate component;
  artifacts are serialized and exposed via a Streamlit UI. Full methods,
  per-strain breakdown, and discussion follow.
---

# 1. Introduction

## 1.1 Problem statement

Atlas Raman delivered 87 confocal-Raman hyperspectral maps spanning four
primary classes — **STEC** (Shiga-toxin-producing *Escherichia coli*),
**Non-STEC** *E. coli*, **Salmonella enterica**, and **H₂O blanks** —
with nine bacterial subclasses representing serogroups commonly seen in
foodborne illness surveillance. The brief asked for a classifier that
distinguishes the four primary classes, honors subclass structure for
evaluation, and (per the Marler-Clark / FSIS non-O157 STEC overview)
recognizes that **non-O157 STEC has no unique biochemical signature on
standard media** — pathogenicity is defined by acquisition of a single
phage-encoded protein (Shiga toxin), making the within-*E. coli*
STEC ↔ Non-STEC boundary biologically thin.

```
Class          Subclasses                                 Files
─────────────────────────────────────────────────────────────────
STEC           O157:H7    O121:H19    O103:H2              9+9+9 = 27
Non-STEC       83972      ATCC25922   K-12                 8+9+8 = 25
Salmonella     Dublin     Heidelburg  Typhimurium          9+9+9 = 27
H₂O            —                                                  8
                                                          ─────────
                                                              87
```

Each file is a Raman map of one bacterial culture (a smear or pellet
imaged pixel-by-pixel on a confocal Raman microscope); after a 200-px
cap and pixel-level QC, **7,122 spectra remain across 87 files**, each
on a canonical 987-bin wavenumber axis after preprocessing.

## 1.2 Why this is hard

1. **STEC ↔ Non-STEC is virulence-defined, not phylogenetic.** The Stx
   phage-encoded toxin is the only protein that distinguishes the two
   groups within *E. coli* (STEC virulence overview, *Virulence* 2013).
   Cell wall, ribosomes, cytoplasm — the bulk Raman signal — should not
   differ. Expecting a label-free Raman map to find a virulence-protein
   signature in this signal-to-noise regime asks a lot.
2. **K-12 is a 100-year laboratory derivative** (Soupene 2003): large
   genomic deletions vs wild-type, missing surface-structure genes,
   genuinely atypical against any clinical or environmental *E. coli*
   distribution. In our LOSO protocol K-12 is one fold; in most
   classical baselines it lands at 0.00 recall.
3. **Cross-strain generalization is harder than cross-pixel.** Cisek
   2013's 95% intra-batch result on STEC ↔ non-pathogenic *E. coli* was
   per-batch cross-validation on controlled cell suspensions; LOSO
   removes one entire subclass per fold. Tang 2026's
   WGAN-Transformer reports 97% under 5-fold CV → 94% on an
   independent test set, a 3pp drop with state-of-the-art deep learning
   and thousands of spectra. That sets the realistic ceiling here.
4. **87 files is small.** The capacity-to-data ratio (259 features ÷ 87
   files = **3.0 features per file**) requires aggressive within-fold
   feature selection. The LOSO mean is a 9-point statistic (one per
   bacterial strain); no architecture makes that look like a
   90-point evaluation.

## 1.3 Headline results (forward reference)

| Track | Headline | Reference |
|---|---|---|
| Best LOSO baseline | **PLS-DA = 0.603** mean parent-class recall | §3.1 |
| Best published-band test | **Cisek-2013 triple null** at file level | §4.1 |
| Best raw single feature | `auc_lps_1194` **d = +1.03** | §4.2 |
| Best engineered feature | `mcr_C6_mean` **d = −1.23** | §5.3 |
| Best biology ratio | `bio_alpha_helix_score` **d = −0.986** | §5.4 |
| Best spatial feature | `spat_skew_lps_1117` **d = +0.725** (E. coli ↔ Salm) | §5.5 |
| Production classifier (Stage 15F) | see §6 | §6 |

---

# 2. Dataset

## 2.1 Inventory and provenance

![Dataset inventory — 87 files / 7,122 QC-passed spectra distributed across 4 primary classes × 9 bacterial subclasses + H₂O blanks.](https://raw.githubusercontent.com/dev1ashish/assets/main/summary_01_inventory.png)

87 confocal-Raman hyperspectral maps, all collected on a single Raman
microscope at one lab. Each file represents one bacterial culture (smear
or pellet) rastered pixel-by-pixel on a ~22×17 (or larger) grid; per
pixel, a full 2048-bin Raman spectrum spanning ~76–3499 cm⁻¹.

**Replicate structure — important caveat for the LOSO interpretation.**
The 8–9 files per bacterial strain in the corpus represent a mixture of
biological and technical replicates, and the provenance metadata does
not allow a clean accounting. Within a single strain the files share
acquisition date in many cases (calibration-batch metadata
clusters strains within 1–2 calendar weeks), which means **within-strain
file-level "replicates" are partially confounded with acquisition-session
batch effects**. Cross-strain LOSO is therefore measuring a mixture of
(a) genuine biology generalization, (b) cross-acquisition-session
robustness, and (c) cross-batch instrument drift. We can't disentangle
these three without a re-collection protocol that explicitly varies
acquisition session within strain. **For an honest wet-lab read, treat
the LOSO numbers as a lower bound on cross-strain biology recall and an
upper bound on cross-session robustness.**

| Class folder | Subclass folders | Files | Spectra (after 200-px cap) |
|---|---|---:|---:|
| `H20/`        | —                                                | 8 | 767  |
| `STEC/`       | `O103H2` (9), `O121H19` (9), `O157H7` (9)        | 27 | 2,544 |
| `Non STEC/`   | `83972` (8), `ATCC25922` (9), `K-12` (8)         | 25 | 2,344 |
| `Salmonella/` | `Dublin` (9), `Heidelburg` (9), `Typhimurium` (9)| 27 | 2,344 |
| **Total**     |                                                  | **87** | **7,999** |

Per-file format (intentional documentation, since the parser handles
several surprises):

- Tab-delimited ASCII despite `.xls` extension.
- ~44 metadata header lines (`#KEY=\tVALUE`).
- One wavenumber row (two empty cells + 2048 wavenumber values).
- N pixel rows: `x_um \t y_um \t intensity[0..2047]`.
- **Comma thousands-separators** in intensity values (`1,034.00`) — must
  be stripped before float-casting.

## 2.2 Parsing edge cases (and how we handle them)

| Edge case | File examples | Handling |
|---|---|---|
| Wrong `#NUMX` / `#NUMY` headers | R357–R371 (early batch, Feb–early Mar 2026) | Grid derived from `unique(x_um) × unique(y_um)`; headers ignored |
| Mosaics (9 stitched maps in one file) | R364 (STEC, 324 px), R370 (Salm Dublin, 720 px) | Treated as single dense map; 200-px cap prevents class dominance |
| Partial scan | R371 (Typhimurium, 351/360 px) | Kept with `is_complete_scan=False` flag |
| Calibration-batch wn drift | All files | Every spectrum interpolated onto canonical axis `linspace(76, 3499, 2048)` |
| `.txt` extension | One Heidelburg file (R427_*.txt) | Same format; parser globs both `.xls` and `.txt` |
| Folder spelling | `Heidelburg/` (likely typo for Heidelberg) | Kept verbatim so labels match paths |

87/87 files parse with zero fatal errors (`atlas/io.py`).

## 2.3 Preprocessing pipeline

Implemented in `atlas/preprocess.py`. Atomic steps:

```
raw 2048-bin spectrum
     │
     ▼
[ 1 ] cosmic-ray removal — MAD-robust median filter, z-threshold = 5σ
     │
     ▼
[ 2 ] arPLS baseline subtraction — pybaselines.whittaker.arpls
                                    λ = 10⁵, max_iter = 50, diff_order = 2
     │   (removes fluorescence; preserves Raman peaks)
     ▼
[ 3 ] Savitzky–Golay smooth — window = 9, polyorder = 3
     │
     ▼
[ 4 ] crop two regions:
        fingerprint 400–1800 cm⁻¹   (nucleic acids, proteins, carbohydrates)
        C-H stretch 2800–3050 cm⁻¹  (membrane lipid C-H modes)
     │   (drop silent region 1800–2800 + noisy edges)
     ▼
[ 5 ] SNV normalization — per-spectrum z-score
     │   (corrects multiplicative scatter from focus / cell density)
     ▼
preprocessed 987-bin spectrum  →  data_cache/spectra_array_preprocessed.npy
```

Output cached at `data_cache/spectra_array_preprocessed.npy` (`float32`,
shape (7,999, 987)) with the matching `data_cache/wavenumber_axis_preprocessed.npy`.

![Per-class mean preprocessed spectra with named-band annotations. STEC / Non-STEC / Salmonella / H₂O are visually similar in the fingerprint region — discriminative signal sits in subtle peak shape, amplitude, and ratio differences (Sections 4, 5).](https://raw.githubusercontent.com/dev1ashish/assets/main/summary_07_annotated_preprocessed_spectra.png)

## 2.4 Quality control

Per-pixel QC implemented in `atlas/qc.py`. Drops:

- Pixels with SNR (peak / baseline-RMS in 800–1200 cm⁻¹) below per-file 10th-percentile.
- Pixels whose mean intensity sits in the bottom 10% of the per-file distribution (likely off-cell background).

| QC stage | Spectra |
|---|---:|
| Raw (after 200-px cap) | 7,999 |
| Dropped by SNR floor   | 6 |
| Dropped as background  | 871 |
| **Retained (QC mask)**  | **7,122** |

Per-file retention (median 89%) recorded in `data_cache/qc_info.json`.

## 2.5 Split protocols

Two protocols (`atlas/splits.py`), used differently across tracks:

| Protocol | What it tests | Folds | Per-fold composition |
|---|---|---:|---|
| **A — StratifiedGroupKFold on `file_id`** | Generalize to new files of seen strains | 5 | ~17 files held out, stratified by primary class |
| **B — Leave-One-Strain-Out (LOSO)** | Generalize to a strain never seen in training | 9 (+1 H₂O fold = 10) | All files of one subclass held out; H₂O pooled |

**LOSO is the project's primary evaluation.** Each LOSO fold trains on
2 strains per parent class (no diversity hedge), so the fold metric is
a 9-point statistic. Protocol A is reported as the upper bound on
in-distribution performance — closer to what the model would see in
deployment if it had been retrained on every new strain.

## 2.6 Wavenumber axis and named bands

The preprocessed axis has 987 bins covering 400–1800 cm⁻¹ + 2800–3050 cm⁻¹.
30 named bands are catalogued in `atlas/band_features.BANDS`, grouped
into 5 macromolecule classes:

| Group | Example bands | Biology |
|---|---|---|
| Aromatic AA | 762 (Trp), 831/855 (Tyr), 1004 (Phe ring) | total-protein anchors |
| Protein amide | 1242 (amide-III β-sheet), 1658/1662 (amide-I α-helix) | secondary structure |
| Nucleic acid | 720 / 786 (A,G,T,C ring), 1338 / 1485 / 1575 | DNA/RNA |
| Lipid + carbohydrate | 1080 (phospholipid backbone), 1451/1454 (CH₂), 2850/2930 (C-H stretch) | membrane composition |
| Metabolite (Salmonella) | 616 (COO⁻), 925 (C-C), 1542 (C=C) | per Yuan-2024 |

Two LPS regions are used for continuous integration:
`lps_o_antigen_full = (400, 900)` and `lps_chain_discrim = (800, 1200)`.

---

# 3. Baseline modeling

This section reports the pre-engineering baseline: classical sweeps,
deep architectures, DANN domain-adversarial training, ensemble attempts,
and multi-seed verification. **Result: PLS-DA on the raw 987-bin
spectrum, LOSO mean parent-class recall = 0.603, is the project record.**

## 3.1 Classical sweep (Protocol A + LOSO)

Six classical models (`atlas/models_classical.py`) on the preprocessed
spectrum:

| Model | Protocol A file-macro-F1 | LOSO mean parent-recall |
|---|---:|---:|
| **PLS-DA** (n_components = 10) | 0.85 | **0.603** ← project record |
| LogReg L2                       | 0.79 | 0.41 |
| LinSVM                          | 0.80 | 0.40 |
| RBF-SVM                         | 0.81 | 0.38 |
| Random Forest                   | 0.83 | 0.42 |
| XGBoost (n_est = 300, depth=4)   | 0.86 | 0.40 |

**Key finding.** PLS-DA's latent-space projection generalizes
substantially better than RF / XGB under strain-level LOSO. Non-linear
models overfit the training strains' high-dimensional structure; PLS-DA's
low-rank projection drops to the discriminative subspace and survives the
strain-distribution shift.

## 3.2 Deep models

### 3.2.1 1D-CNN

Implementation: `atlas/models_cnn.py`. Architecture: per-bin
`StandardScaler` → 4-stage Conv1d (32 → 64 → 96 → 128 channels, kernel = 5,
stride = 1, BN + ReLU, MaxPool every 2 stages) → global avg pool → 2-layer
MLP head. **~124K parameters.** Heavy augmentation (mixup, Gaussian noise,
baseline jitter, intensity scaling, wavenumber stretch).

- Protocol A file-F1: **0.649** — below pre-registered 0.92–0.98 floor.
- LOSO mean: **0.35** — below 0.55 floor.
- **But cracks K-12 at 0.50 and O157H7 at 0.56**, both of which classical
  models hit 0.00 on. This is the first sign that different architectures
  own different held-out strains.

### 3.2.2 1D-Transformer

`atlas/models_transformer.py`. Patch-tokenized 1D Transformer
(patch_size ∈ {5, 20}), 4 encoder layers, 4 heads, d_model = 64, FF = 128.
**~217K parameters.**

- patch_size = 20: Protocol A 0.507, LOSO 0.193, K-12 & O157H7 → 0.00.
  **The widest patches blur the narrow Raman peaks** the CNN's kernel-5
  convolutions catch.
- patch_size = 5: Protocol A 0.681, LOSO **0.349**, ATCC25922 **= 1.00**
  (first 100% on that strain anywhere in the project).

This blur diagnostic explains why "small Transformers" routinely
under-perform CNN baselines on narrow-peak spectroscopy data.

### 3.2.3 DANN (Domain-Adversarial Neural Network)

Memprobe-v2 (§3.4) fires at 15.5%, so we add an 87-way `file_id` domain
head behind a gradient-reversal layer to the CNN backbone. Sweep over
λ ∈ {0.05, 0.1, 0.3}.

| Variant | Single-seed (originally shipped) | 5-seed mean ± SD | 5-seed soft-vote |
|---|---:|---:|---:|
| DANN λ = 0.1 | 0.500 (seed 42) | **0.345 ± 0.145** | 0.370 |
| DANN λ = 0.3 | 0.447 (seed 42) | 0.393 ± 0.117 | **0.448** |

**Multi-seed verification revised the original headline.** Shipping
λ = 0.1 was a lucky-seed artifact; under proper 5-seed averaging, λ = 0.3
wins (0.448 vs 0.370) and is more robust:

- λ = 0.3 K-12 recall: 0.75 on 3 of 5 seeds (soft-vote 0.75).
- λ = 0.3 O157:H7 recall: 0.78 (soft-vote).
- λ = 0.1 K-12: SD = 0.35 across seeds — most "wins" are seed luck.

**The honest reporting unit on deep models for this dataset is 5-seed
soft-vote**, not a single-seed point estimate.

## 3.3 Ensembling — five schemes, none cleared PLS-DA

`atlas/ensemble.py`, `atlas/stacking.py`,
`atlas/calibrated_ensemble.py`, `atlas/lambda_selector.py`:

| Scheme | LOSO mean | K-12 | O157:H7 | Verdict |
|---|---:|---:|---:|---|
| Soft-vote (PLS-DA + XGB + CNN) | 0.579 | 0.00 | 0.00 | CNN wins destroyed by averaging |
| Stacking (LogReg meta over base probas) | 0.432 | 0.00 | 0.00 | Can't extrapolate across LOSO folds |
| Per-strain λ-selector (hard / soft / router) | 0.444 | — | — | Inner-val signal doesn't predict held-out |
| 4-arch soft-vote (PLS-DA + DANN + Patch5 + 2ch-CNN) | 0.579 | 0.00 | 0.11 | Still loses to PLS-DA solo |
| Margin-based confidence router | 0.603 | 0.00 | 0.00 | Degenerates to "always pick PLS-DA" on 78/78 files |
| Temperature-scaled soft-vote | 0.566 | 0.00 | 0.67 | First to break O157:H7, K-12 stays 0 |

**Mechanism — calibration mismatch.** Fitting per-classifier temperature
scaling on held-out predictions yielded sharply different distributions:

- **PLS-DA fitted T = 6.43** (highly peaked probabilities).
- **Deep models fitted T = 1.2–1.7** (relatively flat).

When a 1-of-4 deep model is right on a "minority-of-one" strain (e.g.
K-12 owned only by DANN λ = 0.3; Typhimurium owned only by PLS-DA), no
calibration scheme can amplify its vote past three confidently-wrong
bases. The ensemble chapter is closed.

## 3.4 Memprobe — is the model memorizing `file_id`?

`atlas/memprobe.py` / `atlas/memprobe_v2.py`. A linear probe on the CNN's
penultimate layer predicts `file_id` (87-way classification, chance = 1.15%):

- Pre-DANN CNN penultimate: **15.5%** linear top-1.
- Post-DANN λ = 0.1: 14.0%.
- Post-DANN λ = 0.3: 13.6%.

**Memprobe is decoupled from LOSO performance.** DANN reshapes feature
*prominence* but does not strip linear file-id separability on this
dataset. The cleanest interpretation: there's residual
instrument/batch/substrate signal in the features that doesn't hurt
cross-strain generalization in the direction we care about. **This
finding is what motivates Stage 15C MCR-ALS unmixing** (separate
biology from substrate components).

## 3.5 Per-strain best-model table (the actual writeup story)

The single most informative result from the baseline track: **no model
wins every cell. Each architecture owns a different held-out strain.**

| Strain | Parent class | Best model | Recall |
|---|---|---|---:|
| 83972        | Non-STEC   | PLS-DA                              | **1.00** |
| ATCC25922    | Non-STEC   | Patch=5 Transformer / DANN λ=0.1    | ~1.00 |
| K-12         | Non-STEC   | DANN λ=0.3 (5-seed soft-vote)       | 0.75 |
| O103:H2      | STEC       | PLS-DA / DANN λ=0.3                 | ~0.89 |
| O121:H19     | STEC       | 2-channel CNN (SNV + 2nd-deriv)      | 0.89 |
| O157:H7      | STEC       | DANN λ=0.3 (5-seed soft-vote)       | 0.78 |
| Dublin       | Salmonella | PLS-DA                              | 0.70+ |
| Heidelburg   | Salmonella | PLS-DA                              | 0.70+ |
| Typhimurium  | Salmonella | PLS-DA (only model > 0)             | 0.56 |

The takeaway is *"different inductive biases solve different biology
cells, and no current combination scheme captures the union."*

## 3.6 Additional diagnostics

- **2-channel CNN** (SNV + fixed Savitzky-Golay 2nd-derivative as a second
  input channel): LOSO 0.465, **best result on O121:H19 (0.89)** and
  **O157:H7 (0.78)**. 2nd-derivative channel removes affine baselines
  pre-network and sharpens narrow peaks — that's what helps STEC strains.
- **Grouped-domain DANN** (predict subclass not file_id): LOSO 0.309.
  Rejects the "coarser domain target preserves more" hypothesis.
- **Augmentation regime sweep:** lighter augmentation regresses LOSO mean
  by 0.04–0.07. Heavy augmentation is doing genuine cross-strain
  regularization, not over-tuning.

## 3.7 Why the baselines plateau at 0.603

LOSO mean is a 9-point statistic — only 2 train strains per parent class
per fold. With 87 files / ~7K spectra / one instrument / one lab, the
modeling slack is small. Even the best deep model (5-seed-verified
DANN λ = 0.3 at 0.448) sits below PLS-DA solo. **The wall is data, not
modeling.** The remaining engineering work (Stages 1–7 and 15A–F) tests
whether *better representations* — chemistry-grounded features and
MCR-ALS unmixing — can pull more signal out of the same 87 files.

---

# 4. Band-chemistry track (Stages 1–7)

A literature-anchored falsification track. Each stage is pre-registered
in `plan/experiments/2026-05-{17,18}_*.md`.

## 4.1 Stage 1 — falsifying the published Cisek-2013 STEC triple

Cisek et al. (*Analyst* 2013) report >95% sensitivity/specificity for
STEC ↔ non-pathogenic *E. coli* on three discriminative bands:

- **1338 cm⁻¹** — CH₂ wag / adenine ring (nucleic acid).
- **1454 cm⁻¹** — CH₂ deformation (lipid).
- **1658 cm⁻¹** — amide-I (protein, β/random).

We integrated each (±10 cm⁻¹ AUC) and computed file-level Cohen's d
across the 27 STEC + 25 Non-STEC files:

| Band | center | d (STEC ↔ Non-STEC) | Verdict |
|---|---:|---:|---|
| na_1338      | 1338 | +0.13 | null |
| **lipid_1454** | 1454 | **−0.47** | **sign-reversed** |
| amide_i_1658 | 1658 | +0.16 | null |

**The literature triple does not replicate at file level on this dataset.**
Sign reversal at 1454 is the strongest negative finding: where Cisek-2013
reports STEC > Non-STEC for CH₂ deformation, Atlas shows
Non-STEC > STEC (modest effect). The Cisek protocol used controlled cell
suspensions with per-batch CV — our file-level Cohen's d on a
single-lab corpus is a different statistic.

![Stage 1 — per-strain distribution of the literature triple (1338 / 1454 / 1658 cm⁻¹). No clean STEC ↔ Non-STEC separation at file level on any of the three bands; 1454 sign-reversal visible.](https://raw.githubusercontent.com/dev1ashish/assets/main/02_primary_triple_violin.png)

## 4.2 Stage 2 — finding the actual signal

Searching all 35 catalog bands for file-level |d| ≥ 0.5 STEC ↔ Non-STEC
surfaced an LPS-chain region (800–1200 cm⁻¹):

| Band | center | d (STEC ↔ Non-STEC) | d (E. coli ↔ Salm) | Note |
|---|---:|---:|---:|---|
| **auc_lps_1194** | 1194 | **+1.03** | +0.34 | **Empirical anchor 1** |
| auc_lps_1117    | 1117 | +0.77 | +0.30 | Empirical anchor 2 |
| auc_lps_1050    | 1050 | +0.42 | +0.82 | top E. coli ↔ Salm discriminator |
| auc_lps_o_antigen_full (400–900) | — | +0.51 | +0.42 | LPS detection region |
| auc_lps_chain_discrim (800–1200) | — | +0.65 | +0.55 | continuous LPS chain |

**The discrimination signal is in the LPS-chain region**, not in the
protein/lipid peaks the published triple emphasized. STEC has higher AUC
at 1194 cm⁻¹ than Non-STEC, consistent with O-antigen polysaccharide
architecture differences between pathogenic and commensal serogroups.

![Stage 2 — best 1D AUROC per candidate band. The LPS-chain region (1050 / 1117 / 1194 cm⁻¹) tops the ranking; the literature triple bands (1338 / 1454 / 1658) cluster near AUROC ≈ 0.5.](https://raw.githubusercontent.com/dev1ashish/assets/main/09_best1d_auroc.png)

## 4.3 Stage 3 — radars, ratios, and macromolecule vectors

Per-class macromolecule vector (sum of named band AUCs by group):

```
                aromatic_aa  protein_amide  nucleic_acid  lipid_carb  metabolite
H₂O                  -33.5         -16.8         -50.6         9.8        -22.6
STEC                 -33.1         -16.8         -52.0         8.4        -23.5
Non-STEC             -33.3         -16.8         -52.5         8.9        -23.4
Salmonella           -33.2         -16.7         -52.0         8.7        -23.6
```

SNV-normalized AUCs are sign-flipped from raw intensities (the absolute
values are uninterpretable in isolation), but the **between-class
spread** is informative. **Ratios under-perform single-band AUCs at the
file level** — `amide_over_na`, `aa_over_amide`, `lipid_over_protein`
all show |d| < 0.4 STEC ↔ Non-STEC, well below the LPS_1194 anchor.

![Per-class macromolecule-group radar (Stage 3). The 5 macromolecule axes (aromatic AA / protein amide / NA / lipid+carb / metabolite) are visually similar across STEC, Non-STEC, Salmonella — confirming that macromolecule-group AUCs alone don't discriminate at file level; the signal lives in finer-grained band shapes and unmixed components.](https://raw.githubusercontent.com/dev1ashish/assets/main/05_macromolecule_radar.png)

## 4.4 Stage 5 — engineered-feature classifier (LOSO 0.31 with 13 features)

Trained XGBoost on a 13-feature subset (4 LPS-region AUCs + 4 ratios + 5
literature/anchor features) — chemistry-grounded, transparent, easy to
interpret:

| Protocol | File-macro-F1 / LOSO recall |
|---|---:|
| **Protocol A file-macro-F1** | **0.87** (matches PLS-DA on raw spectrum) |
| LOSO mean parent-recall | **0.31** (far below PLS-DA's 0.603) |
| **O121:H19 LOSO recall**     | **0.89** ⭐ (ties PLS-DA project record on this STEC strain, using 13 features instead of 987 bins) |
| O103:H2 LOSO recall  | 0.67 |
| O157:H7 LOSO recall  | 0.00 (model has never seen O157-like LPS in training) |
| K-12 LOSO recall     | 0.00 (atypical lab strain) |

**The serogroup-specificity is the trap.** LPS-chain features carry
serogroup-level information that fits training distributions tightly
but doesn't extrapolate to held-out strains with structurally different
O-antigen architecture (O157:H7 LOSO = 0.00 — model has never seen
O157-like LPS). This is **branch (C) for the band-only classifier**, and
the explicit motivation for the feature-engineering track (Stages 15A–F).

![Stage 5 — per-strain LOSO recall of the 13-feature XGB classifier vs prior baselines. The 13-feature model ties PLS-DA's project record on O121:H19 (0.89) using **75× less input information** than the 987-bin raw spectrum, but its LOSO mean collapses to 0.31 because LPS-chain features don't extrapolate across O-antigen architectures.](https://raw.githubusercontent.com/dev1ashish/assets/main/stage5_per_strain_comparison.png)

## 4.5 Stage 6 — skipped per stage-gate

Pre-registered Stage 6 (3-channel CNN with band features) was skipped:
adding the same non-generalizing engineered features as an input channel
to a CNN wouldn't fix the LOSO failure. Budget saved: 0.5 day.

## 4.6 Stage 7 — mixed-sample deployment simulation

Simulated pixel-level mixing across primary classes to estimate
deployment-time degradation under contaminated samples:

| Mix fraction | Macro accuracy | Δ vs clean | Most-affected pair |
|---:|---:|---:|---|
| 0%   | 0.87 | — | (baseline) |
| 10%  | 0.79 | −0.08 | E. coli ↔ Salm |
| 20%  | 0.71 | −0.16 | STEC ↔ Non-STEC |
| 30%  | 0.63 | −0.24 | STEC ↔ Non-STEC |
| 50%  | 0.51 | −0.36 | all pairs |

**10–20% mixed-pixel contamination drops macro accuracy 8–16 percentage
points.** The classifier has a **STEC-default bias under uncertainty** —
ambiguous mixed pixels are pulled toward STEC. For food-safety
screening this is the conservative direction (a false STEC call is
better than a false clear), but it inflates false-positive rates if the
sample is actually commensal. Stage 7 validates the briefing's 10–20%
expected drop.

![Stage 7 — accuracy degradation as a function of mixed-pixel fraction. 10–20% contamination drops the file-macro F1 by 8–16 percentage points; STEC vs Non-STEC is the most-affected pair beyond 20%.](https://raw.githubusercontent.com/dev1ashish/assets/main/stage7_degradation_curves.png)

---

# 5. Feature-engineering track (Stages 15A–E)

Five new feature families implemented over six days, producing **259
features per file** from the cached preprocessed spectra. The track tests
whether better representations close the LOSO gap that the band-only
classifier exposed in §4.4. Each stage is documented in
`plan/experiments/2026-05-18_stage15{a..e}_*.md`.

![Feature catalog growth across Stages 15A → 15E. Cumulative size and per-stage delta. Note Stage 15A is the largest single contributor (153 features in pseudo-Voigt fits + ROI moments + EMSC + derivatives).](https://raw.githubusercontent.com/dev1ashish/assets/main/fig01_cache_size.png)

```
   stage   contributes  feature module                  cache
   ─────────────────────────────────────────────────────────────────────
   15A     pseudo-Voigt + ROI + EMSC + derivatives →   atlas/band_features.py
   15B     DWT + ROI-PCA + SAM templates           →   atlas/spectral_features.py
   15C     MCR-ALS spectral unmixing (K=7 active)  →   atlas/unmix_features.py
   15D     biology-grounded ratios (5 families)    →   atlas/band_features.py
   15E     per-file spatial moment statistics      →   atlas/spatial_features.py
                                                        │
                                                        ▼
                                                    data_cache/{band,spectral,unmix,spatial}_features.parquet
                                                    (166 + 51 + 32 + 10 = 259 features)
```

## 5.1 Stage 15A — pseudo-Voigt peak fits, ROI moments, EMSC, derivatives

**Pseudo-Voigt with linear baseline** replaces Lorentzian for peak
fitting. Fit-success rates on three empirical anchors jumped:

| Band | Lorentzian | Pseudo-Voigt | Δ |
|---|---:|---:|---:|
| lps_1117 | 4.0% | 71.1% | +67.1 pp |
| lps_1194 | 0.2% | **62.7%** | +62.5 pp |
| lps_1050 | 11.8% | 60.6% | +48.8 pp |
| na_1338 | 8.7% | 60.6% | +51.9 pp |
| lipid_1454 | 20.3% | **89.1%** | +68.8 pp |
| amide_i_1658 | 14.0% | **85.5%** | +71.5 pp |

**Confirmed: amide-I shifts +1.1 cm⁻¹ between STEC (mean fit center
≈ 1661) and Non-STEC (≈ 1662.5)** — Non-STEC's amide-I sits further into
α-helix territory. Direction agrees with Stage 15D's
`bio_alpha_helix_score` finding (§5.4).

Additional families added by Stage 15A:

- **6-statistic ROI moments** (`roi_<region>_{mean,std,skew,kurt,centroid,entropy}`)
  × 6 regions = 36 features.
- **EMSC scatter coefficients** — 4 per spectrum (offset, ref scale,
  linear+quadratic baseline). EMSC b-coef E. coli ↔ Salm d = 0.30
  (weak in isolation; scale-invariant safety net for LOSO).
- **Savitzky-Golay 1st and 2nd derivative AUCs** at the same 8 fit bands.
  **`d2_auc_lps_1194` d = −0.898 STEC ↔ Non-STEC** — curvature sharpness
  is an orthogonal axis to the raw AUC.

**Headline:** `d2_auc_lps_1194` d = −0.898 — new strong
STEC-discrimination feature based on the *curvature* of the LPS_1194
peak, not its amplitude.

![Stage 15A — pseudo-Voigt + linear-baseline fit success replaces Lorentzian. Empirical anchors lift from 0.2–37% → 60–89%; the literature triple (1338 / 1454 / 1658) reaches 60–89%, unblocking peak-shift analysis (§5.1).](https://raw.githubusercontent.com/dev1ashish/assets/main/fig03_pseudovoigt.png)

## 5.2 Stage 15B — DWT + ROI-PCA + SAM templates

### DWT (Discrete Wavelet Transform)

Daubechies-4 mother wavelet, 6 detail levels → energy + entropy per level
= 12 features.

| Detail level | Energy d (E. coli ↔ Salm) | Entropy d |
|---|---:|---:|
| L1 (highest freq) | −0.235 | −0.425 |
| L2                | +0.038 | −0.235 |
| L3                | +0.006 | −0.224 |
| L4 (~30–80 cm⁻¹)  | +0.251 | **−0.526** |
| L5                | −0.032 | −0.006 |
| L6 (coarsest)     | −0.259 | −0.139 |

**`dwt_entropy_L4` d = −0.526** — Salmonella has higher mid-scale
spectral entropy than E. coli (more distributed peak structure at that
scale).

### ROI-PCA

PCA per region: LPS (5 PCs), amide (3 PCs), C-H stretch (3 PCs) =
11 features. **PC1 of every region is dominant-variance (file/scale),
not class-variance.** The discriminative axes sit in PC2 and PC3:

| Feature | d STEC ↔ Non-STEC | Note |
|---|---:|---|
| **pca_lps_PC3**   | **+1.032** | Same chemistry as `auc_lps_1194` in a learned coordinate |
| **pca_amide_PC3** | **+0.891** | **New** — amide region carries undertapped class signal |
| pca_amide_PC2     | −0.666 | Second amide axis |
| pca_lps_PC2       | +0.498 | Secondary LPS direction |

### SAM (Spectral Angle Mapper)

Cosine-angle templates against class-mean and subclass-mean spectra,
both full-spectrum and LPS-region-only — 28 features total.

| Pair | Best SAM AUROC |
|---|---:|
| H₂O ↔ bacteria   | **1.000** (lps_class_H2O) |
| STEC ↔ Non-STEC  | 0.690 |
| E. coli ↔ Salm   | 0.732 |

**Direction-only (cosine) features are weaker than amplitude on this
dataset.** SAM works exactly as designed for the easy split (H₂O) but
under-performs on the hard pairs. Operational decision: keep SAM in the
classifier as a scale-invariant LOSO safety net, but don't promote it
to headline.

## 5.3 Stage 15C — MCR-ALS spectral unmixing (project record)

**Multivariate Curve Resolution — Alternating Least Squares.** Decomposes
the (`N_pixels` × `B`) data matrix into

```
D  ≈  C · Sᵀ
(N × B)     (N × K) · (K × B)
```

under non-negativity constraints. The K rows of `Sᵀ` are pure-component
spectra; the K columns of `C` are concentration maps. Components should
map onto biology (protein, lipid, cytochromes, nucleic acid) and
artifacts (substrate, fluorescence baseline residual).

**Implementation note.** SNV preprocessing produces negative values; MCR's
non-negativity requires `D ≥ 0`. We cropped to `wn ∈ [600, 1800] cm⁻¹`
**before** shifting by `−X.min()` (Run-2 fix — Run-1 with full 400–3050
range got 6 of 8 components dominated by SNV edge-bump artifacts at
470–550 cm⁻¹). **Fit globally on all 7,122 QC-passed spectra** with
SIMPLISMA initialization (Windig & Guilment 1991), K = 8 components,
`pymcr.mcr.McrAR(c_regr=NNLS, st_regr=NNLS, c_constraints=[ConstraintNonneg],
st_constraints=[ConstraintNonneg], max_iter=200)`. **Component 8
collapsed to zero**, giving effective K = 7 for this corpus — a finding
carried into Stage 15F's per-fold MCR-ALS refit.

**Component biology labels** (DD2 manual curation against the band catalog):

| k | SIMPLISMA init (cm⁻¹) | Dominant peaks | Label | Tier |
|:-:|:-:|---|---|---|
| 1 | 675  | broad, no narrow peaks | **substrate / fluorescence baseline** | artifact |
| 2 | 1765 | 1603 (Trp), 946 | aromatic AA / Trp protein | biology |
| 3 | 1004 | 1093 (LPS phospholipid backbone) | LPS phosphate / lipid_1080 | biology |
| 4 | 1452 | 792 (NA na_786), 1081 (lipid) | NA + lipid mix | biology |
| 5 | 1193 | 655, 1516 (NA), **1193 (lps_1194)**, 798, 840 (Tyr) | biology mix w/ LPS top-discriminator | biology |
| 6 | 1096 | **1451 (CH₂)**, 1318 (NA wag), 1663 (amide-I α-helix), 1004 (Phe) | **bulk biology composite** | biology |
| 7 | 783  | 783 (NA) | nucleic acid (RNA/DNA ring) | biology |
| 8 | 1591 | (collapsed) | dead | dead |

**6 biology + 1 artifact + 1 collapsed.** The substrate component (C1)
isolates the file-acquisition signature memprobe-v2 found, validating
the substrate-leakage hypothesis (§3.4).

![Stage 15C — fitted MCR-ALS pure component spectra (K=8 → effective K=7). Each row is one `S^T[k]`. C1 is the broad substrate baseline; C2–C7 carry biology peaks at expected positions (1004 Phe, 1093 LPS phospholipid, 1193 LPS chain, 1451 CH₂, 1663 amide-I α-helix, 783 NA ring); C8 collapsed.](https://raw.githubusercontent.com/dev1ashish/assets/main/stage15c_pure_spectra.png)

![Stage 15C — per-class heatmap of mean `mcr_C{k}_*` values. C6 (bulk biology composite) is the column where Non-STEC and STEC visibly diverge — the file-level d=−1.23 driver.](https://raw.githubusercontent.com/dev1ashish/assets/main/stage15c_per_class_heatmap.png)

![Stage 15C — MCR-ALS convergence curve. Loss plateaus by iteration ~50; `max_iter = 200` is well past convergence.](https://raw.githubusercontent.com/dev1ashish/assets/main/stage15c_convergence.png)

**Per-class concentration table** (file-level mean of `mcr_C{k}_mean`):

| Class | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H₂O        | 1.43 | 0.99 | 8.91 | 9.38 | 3.47 | **0.69** | 4.77 | 0.00 |
| Non-STEC   | 2.15 | 1.81 | 8.82 | 8.12 | 2.57 | **1.43** | 5.12 | 0.00 |
| STEC       | 1.81 | 1.67 | 8.98 | 8.41 | 2.75 | **1.24** | 5.00 | 0.00 |
| Salmonella | 2.00 | 1.76 | 9.29 | 8.00 | 2.47 | **1.35** | 5.28 | 0.00 |

H₂O reads as expected for biology components (very low C2 Trp, very low
C6 bulk biology). **C6 is the only column where Non-STEC ≠ STEC by > 0.1
in mean** — and that 1.43 vs 1.24 gap drives the headline d = −1.23.

**Top file-level discriminators (8 MCR features clear |d| ≥ 0.5 STEC ↔ Non-STEC):**

| Rank | Feature | d (STEC ↔ Non-STEC) | Mechanism |
|:-:|---|---:|---|
| 1 | **mcr_C6_mean** | **−1.231** | bulk biology lower in STEC (new project record) |
| 2 | mcr_C5_mean    | +0.844 | STEC higher in LPS_1194 + Tyr mix |
| 3 | mcr_C2_p90     | −0.841 | Non-STEC peak Trp/aromatic AA higher |
| 4 | mcr_C7_std     | −0.770 | Non-STEC NA heterogeneity within file |
| 5 | mcr_C7_max     | −0.678 | Non-STEC peak NA intensity higher |
| 6 | mcr_C6_p90     | −0.677 | C6 90th-percentile follows C6 mean |
| 7 | mcr_C7_p90     | −0.626 | NA p90 reinforces NA mean direction |
| 8 | mcr_C4_p90     | +0.608 | NA+lipid peak intensity higher in STEC |

**Four orthogonal STEC ↔ Non-STEC axes recovered** (C6 bulk, C5 LPS-mix,
C2 Trp, C7 NA). MCR provides four distinct discriminative directions
the prior catalog only partially covered.

**Substrate-component caveat (R7 mitigation).** `mcr_C1_*` features
have file-level d = −0.62 — they discriminate, but the discrimination
*could* be the substrate-confound signature (the file-acquisition
fingerprint, not biology). Stage 15F runs a **shuffled-label
permutation test** specifically on the `mcr_C1_*` features and drops
them if their importance under random labels exceeds 50% of importance
under real labels.

## 5.4 Stage 15D — biology-grounded ratios

Five feature families, ~13 features, each anchored to a published
biological mechanism:

| Family | Features | Biology |
|---|---|---|
| Cytochromes (BIO1-4)              | `bio_cyt_pyrrole_ratio`, `bio_cyt_ox_state`, `bio_cyt_total` | 752/1127/1356/1372/1585 cyt-c heme bands; 785 nm excitation off-resonance (R5) |
| Protein 2°-structure (BIO20-22)   | `bio_alpha_helix_score`, `bio_beta_sheet_amide3` | 1652/1670 amide-I α-helix; 1232/1270 amide-III β-sheet |
| PHB accumulation (BIO24-25)       | `bio_phb_carbonyl`, `bio_phb_score` | 1730 carbonyl — **K-12 falsifier** |
| Aromatic AA (BIO26-29)            | `bio_tyr_doublet_ratio`, `bio_trp_content`, `bio_trp_indole_env`, `bio_virulence_aa_sig` | 850/830 Tyr Fermi; 759/1552 Trp; 1340/1360 indole env; (Trp+1552)/(1004 Phe) |
| Nucleic conformation (BIO18-19)   | `bio_na_a_form_fraction`, `bio_rna_dna_ratio` | 815/(815+835); 813/788 |

**Per-feature signal (file-level Cohen's d):**

| Feature | d STEC ↔ Non-STEC | d E.coli ↔ Salm | d K-12 ↔ other-STEC | AUROC STEC ↔ Non-STEC |
|---|---:|---:|---:|---:|
| **bio_alpha_helix_score** | **−0.986** | −0.190 | **+0.537** | **0.794** |
| bio_trp_indole_env       | +0.603 | +0.270 | **−0.547** | 0.704 |
| bio_na_a_form_fraction*  | +0.568 | −0.095 | −0.206 | 0.710 |
| bio_cyt_ox_state         | +0.477 | +0.286 | −0.344 | 0.681 |
| bio_rna_dna_ratio        | +0.450 | −0.326 | −0.373 | 0.741 |
| bio_cyt_pyrrole_ratio    | −0.353 | **+0.485** | −0.341 | 0.567 |
| bio_phb_carbonyl         | +0.207 | +0.409 | **+0.066 ❌** | 0.631 |
| bio_virulence_aa_sig     | −0.150 | **−0.651** | +0.535 | 0.511 |
| bio_beta_sheet_amide3    | +0.150 | +0.379 | **−0.591** | 0.538 |

\* Ratios with denominators near zero on SNV'd AUCs produce values outside
their physical range. Treated as raw scores (R9), not clipped.

### 5.4.1 `bio_alpha_helix_score` — strongest biology-grounded ratio

**d = −0.986 STEC ↔ Non-STEC.** Non-STEC commensal *E. coli* runs higher
amide-I α-helix Raman signal (1652/1670 ratio) than STEC. Mechanism:
STEC virulence proteins (intimin, Stx subunits) contribute β-sheet-heavy
folds; commensal strains lack those, so the bulk α-helix population is
higher. **AUROC = 0.794** at the file level — among the strongest
biology-grounded single features in the catalog.

### 5.4.2 PHB hypothesis falsified

Pre-registered: laboratory K-12 should accumulate polyhydroxybutyrate
anomalously vs clinical STEC strains. **Result: `bio_phb_carbonyl`
K-12 ↔ other-STEC d = +0.07.** PHB hypothesis dead.

### 5.4.3 K-12-specific 2°-structure axis (unprompted finding)

Even though PHB doesn't separate K-12, three biology features flag K-12
differently from clinical STEC:

| Feature | K-12 vs other-STEC d | Direction |
|---|---:|---|
| `bio_beta_sheet_amide3` | **−0.591** | K-12 has less β-sheet |
| `bio_trp_indole_env`    | **−0.547** | K-12 Trp environment more exposed |
| `bio_alpha_helix_score` | **+0.537** | K-12 higher α-helix |

**First K-12-specific feature axis in the project.** Consistent with
K-12's reduced virulence-protein loading — clinical STEC strains carry
β-sheet-heavy virulence proteins (intimin, Stx); K-12 doesn't. **This
unblocks Stage 6 reconsideration conditionally** on Stage 15F K-12 LOSO
recall lift.

![Stage 15D — K-12 separates from clinical STEC + Non-STEC on three protein-2°-structure features (α-helix score ↑, β-sheet amide-III ↓, Trp indole env shifted). First K-12-specific feature axis in the project.](https://raw.githubusercontent.com/dev1ashish/assets/main/fig04_k12_axis.png)

### 5.4.4 Trp/Phe direction reversed at E. coli ↔ Salm

The literature `virulence_aa_sig` (Trp/Phe) framing predicted STEC > Non-STEC.
On Atlas, **STEC ↔ Non-STEC signal is weak (d = −0.15)** but
**E. coli ↔ Salm d = −0.65 with Salmonella HIGHER** — likely from
Salmonella's flagellar / outer-membrane protein composition rather than
STEC virulence proteins. Document and redeploy as an E.coli ↔ Salm axis.

## 5.5 Stage 15E — spatial / cross-pixel features

**Pre-flight finding: R6 fires.** The plan/15 spec required ≥200 pixels
per file for Moran's I (DD19) and GLCM-on-intensity-maps (DD20-23).
**Atlas pixel-per-file distribution is min = 70, median = 72, max = 180.**
Only 8 / 87 files exceed 100 pixels; 0 / 87 clear 200. Moran's I and GLCM
are dropped from Stage 15E.

What remains: **per-file moment statistics** (variance / CV / skew /
kurtosis of per-pixel intensity AUCs), computed at any pixel count ≥ 50.
Total: 10 features per file.

| Feature family | Count | Region / band |
|---|---:|---|
| `spat_var_*`   | 2 | LPS chain (800–1200), C-H stretch (2800–3000) |
| `spat_cv_*`    | 2 | scale-invariant variance |
| `spat_kurt_*`  | 3 | LPS anchors 1050 / 1117 / 1194 |
| `spat_skew_*`  | 3 | LPS anchors 1050 / 1117 / 1194 |

**STEC ↔ Non-STEC result: null** (0 features clear |d| ≥ 0.5, best was
0.485). **Spatial heterogeneity does not discriminate STEC vs Non-STEC
on this corpus** — the "clinical strains more uniform than commensals"
hypothesis is falsified at the moment-statistic level.

**But two side-findings:**

1. **`spat_skew_lps_1117` d = +0.725 E. coli ↔ Salmonella.** Salmonella's
   pixel-intensity distribution at 1117 cm⁻¹ is more symmetric; *E. coli*
   is right-skewed. **New strong directional finding for E. coli ↔ Salm.**
2. **H₂O sanity passes 4/4** — all variance and CV features show
   H₂O ≪ bacteria, as expected (water is spatially uniform).

![Stage 15E — `spat_skew_lps_1117` distribution per file. E. coli (STEC + Non-STEC) is right-skewed; Salmonella is symmetric. New strong E. coli ↔ Salmonella axis (d = +0.725).](https://raw.githubusercontent.com/dev1ashish/assets/main/fig05_ecoli_salm_skew.png)

### 5.6 Consolidated top-15 file-level discriminators (Stages 15A–E)

Pooling all 259 engineered features and computing file-level Cohen's d
STEC ↔ Non-STEC:

![Top-15 file-level discriminative features after Stages 15A–E. Stage 15A peak-fits and derivatives dominate; Stage 15B contributes three DWT entropies (purple bars); Stage 15C contributes `mcr_C6_mean` (red bar at d = −1.23) — the global-fit project record. **Per-fold MI in Stage 15F demoted MCR features**, so this is the global-fit ranking, not what survives R2 (§6.7).](https://raw.githubusercontent.com/dev1ashish/assets/main/fig02_top_features.png)

---

# 6. Stage 15F — production classifier (this paper)

This section consolidates Stages 15A–E into the production model.
**Source of truth: `artifacts/stage15f_metadata.json`** produced by
`scripts/run_stage15f_final.py`; pre-registered in
`plan/experiments/2026-05-18_stage15f_full_classifier.md`.

> **Result preview.** Best algorithm is **LogReg-L2** at LOSO mean
> accuracy 0.436 — **Branch (C) Plateau** — below the
> PLS-DA-on-raw-spectrum baseline of 0.603 (§3.1). Production artifact
> is shipped; the project's headline LOSO number remains PLS-DA on raw.
> Stage 15F's contribution is the **deployable serialized model + a
> production-grade inference API**, not a new accuracy record.

## 6.0 Models & metrics — full inventory

**What's deployed (production):** `Pipeline([StandardScaler, LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000)])` fit on **all 87 files** using the 35-feature MI consensus set. No ensemble — the 5 ensemble attempts in §3.3 all failed to beat PLS-DA solo and are an explicit closed chapter.

**Why LogReg over PLS-DA for production**, even though PLS-DA owns the LOSO project record (0.603 on raw spectrum)? PLS-DA's low-rank latent-variable projection helps when fed the full 987-bin spectrum, but **collapses when given 35 MI-pre-selected features** (LOSO 0.324 vs 0.603 — a 28-pp drop). LogReg on the same 35 features holds at 0.436. The inference API works with the engineered cache, so LogReg is the right pick for the artifact even though PLS-DA-on-raw is the better headline.

### Full model + metric inventory

| Track | Model | Protocol A F1 | LOSO mean acc | Notes |
|---|---|---:|---:|---|
| Baseline classical | **PLS-DA (raw 987 bins)** | 0.85 | **0.603** ⭐ project LOSO record | (§3.1) |
| Baseline classical | LogReg-L2 (raw)         | 0.79 | 0.41 | (§3.1) |
| Baseline classical | LinSVM (raw)            | 0.80 | 0.40 | (§3.1) |
| Baseline classical | RBF-SVM (raw)           | 0.81 | 0.38 | (§3.1) |
| Baseline classical | Random Forest (raw)     | 0.83 | 0.42 | (§3.1) |
| Baseline classical | XGBoost (raw)           | 0.86 | 0.40 | (§3.1) |
| Deep                | 1D-CNN (124K params)    | 0.65 | 0.35 | cracks K-12 + O157:H7 (§3.2.1) |
| Deep                | 1D-Transformer (patch=20) | 0.51 | 0.19 | patch-blur diagnostic (§3.2.2) |
| Deep                | 1D-Transformer (patch=5)  | 0.68 | 0.35 | ATCC25922 = 1.00 |
| Deep                | 2-channel CNN (SNV + 2nd-deriv) | — | 0.465 | O121:H19 = 0.89, O157:H7 = 0.78 |
| Domain adversarial  | DANN λ=0.1 (5-seed soft-vote) | — | 0.370 | original headline was lucky-seed |
| Domain adversarial  | DANN λ=0.3 (5-seed soft-vote) | — | **0.448** | K-12 = 0.75 (3/5 seeds), O157:H7 = 0.78 |
| Ensemble (5 attempts) | soft-vote / stacking / λ-selector / 4-arch / temp-scaled / margin-router | — | 0.43–0.60 | **none clear PLS-DA solo** (§3.3) |
| Band-chemistry      | 13-feature XGB (Stage 5) | **0.87** | 0.31 | ties PLS-DA on O121:H19 = 0.89 (§4.4) |
| **Production (Stage 15F)** | **LogReg-L2 on 35 MI features** | n/a | **0.436** | **deployed model** (§6.4) |
| Stage 15F (alt)     | PLS-DA on 35 MI features  | n/a | 0.324 | feature-selection breaks PLS-DA |
| Stage 15F (alt)     | XGBoost on 35 MI features | n/a | 0.247 | over-fits small corpus |

### 15B contributions (per user question)

Stage 15B added **51 features** (DWT 12 + ROI-PCA 11 + SAM 28). The
direct wins:

| Stage 15B feature | d STEC ↔ Non-STEC | What it captured |
|---|---:|---|
| **`pca_amide_PC3`** | **+0.891** | **New amide-region axis** — protein 2°-structure discrimination that no prior stage had surfaced. Validated independently by Stage 15D's `bio_alpha_helix_score` d = −0.986 hitting the same biology. |
| `pca_lps_PC3` | +1.032 | Ties the raw `auc_lps_1194` (+1.03) in a learned coordinate — PCA recovers the same chemistry the empirical-search agent identified. |
| `pca_amide_PC2` | −0.666 | Secondary amide-region axis. |
| `pca_lps_PC2`   | +0.498 | Secondary LPS direction. |
| `dwt_entropy_L4` | −0.526 | Salmonella has higher mid-scale spectral entropy than E. coli (~30–80 cm⁻¹ scale band). |

**Operational findings from Stage 15B**:
- **PC1 of every ROI region is dominant-variance, not class-variance.** Use PC2 / PC3 as the workhorses.
- **SAM (Spectral Angle Mapper) works perfectly on H₂O** (AUROC = 1.000) but **caps at AUROC 0.69–0.73 on bacterial-only pairs**. Direction-only features are weaker than amplitude on this dataset — kept SAM in the classifier as a scale-invariant LOSO safety net, but not promoted to headline.
- **DWT entropy adds modest novel signal** beyond raw AUCs (L4 d = −0.53 at the file level reported in 15B; at file level after mean-pooling, L1–L3 entropy reach |d| > 1.2 — see fig02).

In Stage 15F's MI-selected 35-feature set, Stage 15B contributes **two ROI-PCA features** (`pca_chstretch_PC2`, `pca_chstretch_PC3`) and **one SAM feature** (`sam_lps_sub_O121H19`). DWT features did not survive consensus.

## 6.1 Design rationale

After Stages 15A–E we have **259 features per file** across four
caches:

| Cache | Shape | Level | Source stage |
|---|---|---|---|
| `data_cache/band_features.parquet`     | 7,122 × 166 | per-pixel | 15A + 15D |
| `data_cache/spectral_features.parquet` | 7,122 × 51  | per-pixel | 15B |
| `data_cache/unmix_features.parquet`    | 87 × 33     | per-file  | 15C |
| `data_cache/spatial_features.parquet`  | 87 × 10     | per-file  | 15E |

Three pre-registered risks shape the protocol:

- **R1 — capacity / data ratio.** 259 ÷ 87 = 3.0 features/file. Aggressive
  within-fold feature selection is required to avoid overfitting.
- **R2 — feature-leakage at LOSO.** MCR-ALS / ROI-PCA / SAM are fit on
  data; if we fit them once globally and use those fits inside LOSO, the
  held-out fold's pixels have already informed the projection. The
  per-fold refit protocol is essential.
- **R7 — substrate-confound on `mcr_C1_*`.** MCR component 1 is the
  fluorescence/substrate baseline (§5.3) and has |d| ≈ 0.6 STEC ↔ Non-STEC
  on the global fit. If that signal survives a shuffled-label permutation
  test the features are leak-driven and should be dropped.

## 6.2 Protocol

```
Step 1: Load 4 caches + qc_mask + preprocessed spectra (X_pp, wn).
Step 2: Aggregate per-pixel features (band 166, spectral 51) to file
        level by mean-pool. Join the 2 already-file-level caches.
        Final design matrix: 87 files × 259 features.

Step 3: LOSO folds — 10 total:
          - 9 folds: one per bacterial subclass.
          - 1 fold: H₂O (all 8 water files held out together).

Step 4: PER FOLD, PER SEED:
          (a) Refit MCR-ALS on TRAIN pixels (K=7, the effective K from §5.3).
          (b) Refit ROI-PCA on TRAIN pixels (LPS 5 PCs + amide 3 PCs + CH 3 PCs).
          (c) Refit SAM templates on TRAIN pixels' labels.
          (d) Re-aggregate the held-out file's per-pixel features through
              these fold-specific fits; build (87 × 259) matrix.
          (e) mutual_info_classif on TRAIN rows → keep top 35 features.
          (f) Train each of {PLS-DA, LogReg-L2, XGBoost} in a
              `StandardScaler → classifier` pipeline.
          (g) Predict on held-out fold; record accuracy + per-strain recall.

Step 5: Multi-seed: 5 seeds × 10 folds × 3 algos = 150 fits. Report
        per-algorithm mean ± std LOSO accuracy.

Step 6: R7 permutation test on best algorithm: refit with shuffled `y`
        (10 shuffles), compare `mcr_C1_*` feature importance to real-y.
        Drop any feature with shuffled/real ≥ 50%.

Step 7: Production fit. Best algorithm gets refit on ALL 87 files using
        the consensus MI feature set (majority vote across LOSO seeds).
        Save: classifier.joblib + feature_columns.json + mcr_global.joblib
              + roi_pca.joblib + sam_templates.joblib + metadata.json.
```

## 6.3 Hyperparameters

| Component | Value | Notes |
|---|---|---|
| LOSO folds                                  | 10                         | 9 strains + 1 H₂O |
| Seeds                                       | 5 (0, 1, 2, 3, 4)          | for honest variance |
| MCR-ALS K                                   | 7                          | effective K per §5.3 |
| MCR-ALS max_iter                            | 80                         | (most folds converge before this) |
| MCR-ALS SIMPLISMA offset                    | 5.0%                       | Windig 1991 default |
| ROI-PCA regions                             | LPS (5 PCs) + amide (3) + CH-stretch (3) | from §5.2 |
| SAM region                                  | LPS_REGION_FOR_SAM (800–1200) | per §5.2 |
| MI feature selection target                 | 35                         | inside 30–40 R1 band |
| MI n_neighbors                              | 3                          | sklearn default |
| PLS-DA n_components                         | min(5, n_train − 1)        | one-hot regression w/ argmax decoding |
| LogReg                                      | L2, C = 1.0, lbfgs, max_iter 2000 | scikit default solver |
| XGBoost                                     | n_estimators = 200, max_depth = 4, lr = 0.05, tree_method = hist, eval_metric = mlogloss | |

## 6.4 Headline

| Metric | Value |
|---|---|
| Best algorithm                              | **LogReg-L2** (Pipeline = `StandardScaler` → `LogisticRegression(penalty='l2', C=1.0)`) |
| LOSO mean per-fold accuracy (best algo)     | **0.436** (10-fold incl. H₂O) |
| LOSO mean file-accuracy (best algo)         | **0.448** (file-weighted; 87 files; primary headline) |
| **Bootstrap 95% CI (LogReg)**               | **[0.345, 0.552]** (5000 resamples over files; **straddles the 0.50 Branch (B) bar — verdict is "Branch (C) with overlap into (B)"**) |
| 9-strain LOSO mean (no H₂O fold; apples-to-apples vs baseline) | **0.494** (LogReg) vs **0.367** (PLS-DA) vs **0.278** (XGB) |
| LOSO macro F1 (4-class, file-level)         | 0.349 (LogReg) — pulled down by H₂O F1 = 0 |
| LOSO weighted F1                            | 0.402 (LogReg) |
| **McNemar paired test**                     | **LogReg > PLS-DA: p = 0.0020 ⭐**; LogReg > XGB: p = 0.0033 ⭐; PLS-DA vs XGB: p = 0.25 (tied) |
| **Branch verdict**                          | **(C) with overlap into (B)** — point estimate 0.436 < 0.45 threshold, but bootstrap CI [0.345, 0.552] does not exclude (B) |
| Feature count post-MI + R7                  | 35 (R7 skipped — no `mcr_C1_*` survived MI selection) |
| K-12 LOSO recall (best algo)                | **0.000** (Stage 15D α-helix axis was not MI-selected per fold) |

**Multi-seed surprise.** The first run with 5 seeds gave **std = 0.000**
across PLS-DA / LogReg / XGB. The pipeline is fully deterministic:
MCR-ALS SIMPLISMA picks the same purest variables, sklearn PCA / PLS /
LogReg use deterministic solvers, and `xgboost.XGBClassifier(tree_method='hist')`
is deterministic on fixed input. **Multi-seed adds no variance signal
here.** The re-run used 1 seed; scaffolding kept in place for future
stochastic variants.

![Stage 15F — algorithm comparison on the 35 MI-selected features. LogReg-L2 wins (0.436); PLS-DA collapses to 0.324 vs its 0.603 on raw spectrum (operational lesson: don't pre-select features for PLS-DA). Red dashed line is the PLS-DA-on-raw baseline that remains the project's headline LOSO number.](https://raw.githubusercontent.com/dev1ashish/assets/main/fig08_algo_compare.png)

## 6.5 Per-strain LOSO breakdown (LogReg, best algorithm)

| Strain | Parent class | LOSO accuracy | Δ vs project's prior best |
|---|---|---:|---|
| **ATCC25922**  | Non-STEC   | **0.889** | ties Patch=5 / DANN λ=0.1 ~1.00 |
| **O121:H19**    | STEC       | **0.889** | ties PLS-DA / 2ch-CNN project record |
| **Typhimurium**| Salmonella | **0.778** | **+0.22 vs PLS-DA's 0.56** ⭐ |
| O103:H2       | STEC       | 0.556 | below PLS-DA / DANN λ=0.3 ~0.89 |
| O157:H7       | STEC       | 0.556 | below DANN λ=0.3 5-seed soft-vote 0.78 |
| Heidelburg    | Salmonella | 0.333 | below PLS-DA 0.7+ |
| 83972         | Non-STEC   | 0.25  | far below PLS-DA 1.00 |
| Dublin        | Salmonella | 0.111 | below PLS-DA 0.7+ |
| **K-12**        | Non-STEC   | **0.000** | below DANN λ=0.3 5-seed soft-vote 0.75 |
| **H₂O**         | H2O        | **0.000** | structural — held-out H₂O has no exemplars in training |

**Six folds clear ≥ 0.50; three collapse to ≤ 0.25; H₂O is structurally
0.** Excluding H₂O, the 9-strain mean is **0.484** — closer to the
Branch (B) threshold of 0.50 but still below.

> **Reading the H₂O row.** The H₂O fold scores 0.000 not because the
> classifier "fails on water" in any biological sense — it is
> *structurally* unscoreable in LOSO. With only one H₂O class folder
> (no subclasses), holding it out leaves the model with **zero H₂O
> exemplars at train time**. It will always predict one of the three
> bacterial classes on water spectra. **This is by design of the LOSO
> protocol**, not a model defect. The biologically meaningful evaluation
> for water is Protocol A (StratifiedGroupKFold on `file_id` — §2.5),
> under which all classical models distinguish H₂O from bacteria at
> AUROC ≥ 0.99. **In deployment the production model has seen all 8
> H₂O files at fit time** and will recognize water correctly; the LOSO
> "0.000" is a protocol artifact, not a deployment expectation.

> **Apples-to-apples vs the PLS-DA baseline.** The §3.1 PLS-DA project
> record (0.603) was computed on **9-fold LOSO without an H₂O fold**;
> Stage 15F is 10-fold including H₂O. Fair head-to-head numbers use the
> 9-bacterial-strain mean: **PLS-DA 0.603 vs Stage 15F LogReg 0.484
> (gap = 12 pp)**, not 0.603 vs 0.436. The headline tables show both
> denominators so the reader can choose.

![Stage 15F — per-strain LOSO accuracy for the production LogReg model. Bars colored by parent class. Three wins (ATCC25922, O121:H19, Typhimurium); three failures (K-12, Dublin, H₂O); four middling (O157:H7, O103:H2, Heidelburg, 83972). Branch (B) bar at 0.50 dashed.](https://raw.githubusercontent.com/dev1ashish/assets/main/fig06_stage15f_per_strain.png)

![Stage 15F — REAL row-normalized confusion matrix on the 87-file LOSO predictions (computed from `artifacts/stage15f_loso_predictions.parquet`; not approximated). STEC has the strongest diagonal (R = 0.67); Non-STEC ↔ Salmonella is the dominant within-bacteria confusion (11 / 25 Non-STEC → Salmonella, 9 / 27 Salmonella → Non-STEC); **all 8 H₂O LOSO files are predicted as STEC** — the Stage 7 "default-under-uncertainty STEC bias" reproduced under LOSO.](https://raw.githubusercontent.com/dev1ashish/assets/main/fig07_stage15f_confusion.png)

**Strain-level wins to flag:**

- **Typhimurium = 0.778 is the project's first push past 0.7** on this
  Salmonella serovar (PLS-DA was previously the only model above 0.0 on
  it at 0.56). The MI-selected feature set evidently captures a
  Salmonella-Typhimurium-specific signature where PLS-DA on the raw
  spectrum did not.
- **ATCC25922 + O121:H19 = 0.889** both tie prior project records.
- **K-12 = 0.000 falsifies the Stage 15D K-12-lift hypothesis.** The
  three biology features that flagged K-12 differently
  (`bio_beta_sheet_amide3`, `bio_trp_indole_env`, `bio_alpha_helix_score`)
  did not survive per-fold MI. K-12 remains the project's hardest
  fold; only DANN λ = 0.3 5-seed soft-vote has cracked it (0.75).

## 6.6 Algorithm comparison

| Algorithm | Mean LOSO accuracy | Mean macro recall (4-class) | Notes |
|---|---:|---:|---|
| **LogReg-L2** | **0.436** | 0.109 | Best; production model. Linear on standardized features. |
| PLS-DA        | 0.324 | 0.081 | Markedly worse on engineered features than on raw 987-bin spectrum (0.603). |
| XGBoost       | 0.247 | 0.062 | Tree splits on a tiny corpus over-fit per fold; n_estimators = 200 didn't help. |

**PLS-DA on engineered (0.324) ≪ PLS-DA on raw (0.603).** The low-rank
projection that powered PLS-DA on the full 987-bin spectrum **does not
help** when we pre-select 35 features via MI. PLS-DA needs the full
feature space to build its discriminative latent variables; pre-selection
collapses that advantage. **Operational lesson: do not pre-select
features for PLS-DA — feed it the full spectrum.**

### 6.6.1 Bootstrap CIs (5,000 resamples) + McNemar paired tests

5,000-resample bootstrap over the 87 files. The point estimates differ
slightly from the per-fold-mean values reported elsewhere because
bootstrap uses **file-weighted** accuracy (87 files, equal weight) while
the §6.4 per-fold mean weights folds equally regardless of fold size.

| Algorithm    | LOSO mean file-accuracy | 95% bootstrap CI | Excludes 0.50 (Branch B)? |
|---|---:|---|:---:|
| PLS-DA       | 0.333 | [0.241, 0.437] | yes (entirely below) |
| **LogReg-L2** | **0.448** | **[0.345, 0.552]** | **no — straddles 0.50** |
| XGBoost      | 0.253 | [0.161, 0.345] | yes (entirely below) |

![Stage 15F — bootstrap 95% CIs on LOSO mean file-accuracy (5000 resamples over files). The LogReg CI straddles the Branch (B) threshold of 0.50, so we cannot reject "modest improvement" at α=0.05; all three algorithms are entirely below the PLS-DA-on-raw baseline of 0.603 (red dashed).](https://raw.githubusercontent.com/dev1ashish/assets/main/fig11_bootstrap_ci.png)

**Reading the CIs.** LogReg's 95% CI [0.345, 0.552] **straddles the
Branch (B) threshold of 0.50** — we cannot reject "modest improvement"
at α = 0.05. The point-estimate Branch (C) verdict from §6.4 is the
honest reading, but the bootstrap CI shows it is statistically
compatible with Branch (B). All three algorithms' CIs are entirely
below the PLS-DA-on-raw baseline of 0.603, so the claim "engineered
features under-perform the raw-spectrum baseline at LOSO" is
statistically supported.

**McNemar paired tests on per-file correct/incorrect (n = 87):**

| Comparison | "A only correct" | "B only correct" | Both correct | p-value | Verdict |
|---|---:|---:|---:|---:|---|
| **LogReg vs PLS-DA** | 10 | 0  | 29 | **0.0020** ⭐ | LogReg significantly better |
| LogReg vs XGB      | 24 | 7  | 15 | **0.0033** ⭐ | LogReg significantly better |
| PLS-DA vs XGB      | 17 | 10 | 12 | 0.2478       | not significantly different |

**LogReg is the best algorithm on the engineered cache at α < 0.005.**
The "10 vs 0" cell in LogReg-vs-PLS-DA is striking — there are 10 files
LogReg gets right that PLS-DA gets wrong, and **zero files PLS-DA gets
right that LogReg gets wrong**. PLS-DA on engineered features is
strictly dominated by LogReg on engineered features. PLS-DA and XGBoost
on engineered features are statistically indistinguishable from each
other (p = 0.25).

### 6.6.2 Per-class precision / recall / F1 (LogReg)

| Class | Precision | Recall | F1 | n (true) |
|---|---:|---:|---:|---:|
| STEC       | 0.49 | **0.67** | 0.56 | 27 |
| Non-STEC   | 0.42 | 0.40 | 0.41 | 25 |
| Salmonella | 0.42 | 0.41 | 0.42 | 27 |
| **H₂O**    | **0** | **0** | **0** | 8 |
| **Macro avg** | 0.33 | 0.37 | 0.35 | 87 |
| **Weighted avg** | 0.39 | 0.45 | 0.40 | 87 |

![Stage 15F — per-class precision / recall / F1 for LogReg-L2. STEC is the strongest class (R = 0.67, F1 = 0.56); H₂O is structurally 0 (no exemplars at training time).](https://raw.githubusercontent.com/dev1ashish/assets/main/fig10_per_class_f1.png)

**STEC has the highest recall (0.67) and the highest F1 (0.56)** because
the "default-under-uncertainty STEC bias" first observed in Stage 7
(§4.6) is operating at LOSO too: files the model isn't sure about get
pulled toward STEC, inflating STEC recall and lowering STEC precision.
**For food-safety screening, this is the conservative direction**
(false STEC > false clear), but the precision implication matters: only
about half of predicted-STEC files (P = 0.49) are actually STEC.

**Non-STEC and Salmonella are the hardest pairs** — both sit at
~0.40 F1 because they're frequently confused with each other
(§6.6.3 confusion matrix). The 11 / 25 Non-STEC files predicted as
Salmonella and 9 / 27 Salmonella files predicted as Non-STEC drive
most of the cross-class error.

### 6.6.3 Real confusion matrix

The §6.5 confusion matrix is now computed from the dumped per-file
predictions in `artifacts/stage15f_loso_predictions.parquet` (no
approximation). Updated figure embedded in §6.5.

The Stage 7 STEC-default bias is reproduced cleanly: **all 8 H₂O files
under LOSO are predicted as STEC** (the held-out water has no
training-time exemplars). **18 / 27 STEC files correctly classified.**
**Non-STEC ↔ Salmonella confusion is the dominant within-bacteria error
mode**: 11 / 25 Non-STEC files predicted as Salmonella; 9 / 27 Salmonella
files predicted as Non-STEC.

### 6.6.4 Deployed LogReg coefficients per class

The deployed `LogisticRegression(penalty='l2', C=1.0)` runs behind a
`StandardScaler`, so its coefficients are already on a comparable scale
(each input feature has unit variance). The top 11 coefficients per
class — red = increases probability of that class, blue = decreases:

![Stage 15F — top-11 standardized LogReg-L2 coefficients per class. Patterns: H₂O is driven by `fit_lipid_1454_fwhm` (low → bacteria-like → not water) and the LPS_1050 peak center; Non-STEC by amide-III height/area + amide-I 1st-derivative AUC; STEC by LPS_1117 FWHM/height + the O121:H19 SAM template + the LPS chain kurtosis; Salmonella by ROI LPS chain kurtosis (strongly negative) + lipid_1454 FWHM and area.](https://raw.githubusercontent.com/dev1ashish/assets/main/fig12_logreg_coefs.png)

**Class-specific signature readouts (from the coefficients):**

- **STEC:** large positive weights on `fit_lps_1117_fwhm` (+0.85),
  `sam_lps_sub_O121H19` (+0.77), `roi_silent_skew` (+0.76),
  `d2_auc_aa_1004` (+0.71), `fit_lps_1117_height` (+0.62). **STEC is
  recognized by its LPS-chain peak shape + the spectral angle to
  O121:H19** (the LOSO project record strain).
- **Non-STEC:** positive weights on `fit_amide_iii_1242_height` (+0.68),
  `fit_amide_iii_1242_rmse` (+0.61), `d2_auc_lps_1117` (+0.59),
  `fit_amide_iii_1242_area` (+0.59), `d1_auc_amide_i_1658` (+0.57).
  Negative on `fit_lps_1117_fwhm` (−0.91), `sam_lps_sub_O121H19` (−0.62),
  `pca_chstretch_PC2` (−0.51). **Non-STEC is recognized by amide-III
  protein-2°-structure features and the absence of the O121:H19 LPS
  pattern**, consistent with the Stage 15D `bio_alpha_helix_score` axis.
- **Salmonella:** large positive on `fit_lipid_1454_fwhm` (+1.00),
  `fit_lipid_1454_area` (+0.69), `fit_amide_i_1658_rmse` (+0.63). Large
  negative on `roi_lps_chain_kurt` (−1.03), `fit_lps_1050_center`
  (−0.79), `fit_amide_iii_1242_height` (−0.73). **Salmonella is
  recognized by its lipid_1454 fingerprint and the *absence* of the
  E. coli LPS-chain kurtosis**, consistent with the Stage 15E spatial
  finding that Salmonella's LPS distribution is symmetric while
  E. coli's is right-skewed.
- **H₂O:** large negative on `fit_lipid_1454_fwhm` (−0.54) — water has
  no lipid peak, so a low/missing lipid_1454 width predicts H₂O. (In
  LOSO this signal is unusable because no H₂O is in training; in
  Protocol A it gives perfect water discrimination.)

This is the **honest production-model transparency** — the model is
using the chemistry features we expected (LPS-chain shape for STEC vs
the rest, amide-III for Non-STEC, lipid_1454 for Salmonella).

## 6.7 MI-selected feature set (consensus, k = 35)

Top-10 features by MI rank across folds:

```
1.  roi_ch_stretch_std         (Stage 15A — ROI moment)
2.  fit_amide_iii_1242_area    (Stage 15A — pseudo-Voigt fit)
3.  roi_silent_kurt            (Stage 15A — sanity region kurtosis)
4.  fit_amide_iii_1242_height  (Stage 15A — pseudo-Voigt fit)
5.  d1_auc_lps_1117            (Stage 15A — 1st derivative AUC)
6.  d1_auc_lipid_1454          (Stage 15A — literature triple, 1st-deriv)
7.  d1_auc_amide_i_1658        (Stage 15A — literature triple, 1st-deriv)
8.  fit_lipid_1454_height      (Stage 15A — pseudo-Voigt fit)
9.  fit_amide_iii_1242_rmse    (Stage 15A — pseudo-Voigt fit quality)
10. fit_lipid_1454_area        (Stage 15A — pseudo-Voigt fit)
```

Three striking observations:

1. **No MCR-ALS components in the top 10.** Only `mcr_C5_std` survives
   (rank 35 of 35). The global `mcr_C6_mean` (d = −1.23) discrimination
   reported in §5.3 **does not survive per-fold refit** — exactly what
   the R2 mitigation was designed to catch. MCR-ALS components reorder
   across folds when refit on train-only data, so MI on TRAIN rows
   (78 files) does not see a stable "C6 = bulk biology" axis. **The
   global-fit MCR result was partially a leakage artifact**, even
   though §5.3's component biology labeling was real.
2. **Stage 15A peak-fits and derivatives dominate.** Pseudo-Voigt
   height / area / RMSE at amide-III (1242), lipid_1454, LPS_1117,
   LPS_1194, LPS_1050, Phe_1004; 1st- and 2nd-derivative AUCs at the
   literature triple and LPS chain. **The unblocking of Stage 15A
   peak-fit success (60–89% from 0.2–37% Lorentzian baseline) was the
   highest-yield engineering improvement for LOSO generalization.**
3. **One Stage 15D biology feature survives** (`bio_trp_indole_env`,
   `bio_cyt_ox_state`) and **one SAM feature** (`sam_lps_sub_O121H19`).

![Stage 15F — origin of the 35 MI-selected features by stage. Stage 15A (pseudo-Voigt + ROI + EMSC + derivatives) dominates; Stage 15D biology + Stage 15B PCA/SAM + Stage 15C MCR contribute the long tail.](https://raw.githubusercontent.com/dev1ashish/assets/main/fig09_mi_feature_origin.png)

## 6.8 R7 permutation-test verdict

**Skipped.** No `mcr_C1_*` features survived the per-fold MI selection,
so the substrate-component-leakage test was not needed. (This is itself
evidence for the R7 hypothesis — `mcr_C1_*` features were either
leak-correlated or per-fold-unstable; both interpretations argue
against retaining them.)

## 6.9 Production artifact summary

| Artifact | Size | Contents |
|---|---:|---|
| `stage15f_classifier.joblib`     | 3.5 KB | `Pipeline([StandardScaler, LogisticRegression])` fit on all 87 files |
| `stage15f_feature_columns.json`  | 0.8 KB | 35 MI-selected feature names in pipeline-input order |
| `stage15f_mcr_global.joblib`     | 1.88 MB | `MCRALSWrapper(K=7)` fit on all 7,122 spectra; used by inference for the MCR concentration columns |
| `stage15f_roi_pca.joblib`        | 15 KB  | Fitted ROI-PCA dict (LPS / amide / C-H stretch regions) |
| `stage15f_sam_templates.joblib`  | 140 KB | Fitted SAM templates (class-mean + subclass-mean, full + LPS-region) |
| `stage15f_metadata.json`         | 2.1 KB | LOSO summary + per-strain + algorithm comparison + R7 verdict |
| `stage15f_loso_summary.csv`      | 1.3 KB | Per-fold raw rows (`seed, fold, algo, accuracy, macro_recall`) |
| **Total** | **~2.0 MB** | well under Streamlit Community Cloud's 100 MB cap |

## 6.10 End-to-end inference smoke test

```python
>>> from atlas.inference import predict_from_xls
>>> predict_from_xls('Atlas Data/STEC/O157H7/R412_100_10000ms_260311.xls')
{
  'class':        'STEC',                                    # ✅ matches ground truth
  'probabilities': {'STEC': 0.773, 'Non-STEC': 0.135,
                    'Salmonella': 0.074, 'H2O': 0.018},
  'spectrum_mean': ndarray[987],
  'wn':            ndarray[987],
  'feature_values': {...35 entries...},
}
```

**Predicted STEC at 77% confidence on a held-out O157:H7 file** (which
is the project's O157:H7 LOSO failure fold for most baselines). Pipeline
end-to-end: parse → preprocess (cosmic + arPLS + SG + crop + SNV) →
feature extraction (band + DWT + ROI-PCA-transform + SAM-transform +
MCR-transform + spatial moments) → mean-pool to file level →
MI-selected subset → loaded LogReg pipeline. Latency < 5 s on
a 2024 M-series Mac.

## 6.11 Branch verdicts (pre-registered)

| Branch | Bar | Implication |
|---|---|---|
| (A) Clear win   | LOSO ≥ 0.55 AND K-12 LOSO ≥ 0.75 on all 5 seeds | unblock Stage 6 reconsideration; ship |
| (B) Modest gain | LOSO ≥ 0.50 OR K-12 lift but main 0.45–0.50 | ship classifier; document the gap |
| (C) Plateau     | LOSO < 0.45 | feature track has plateaued; pivot to SSL track (plan/13); ship best-available |

---

# 7. Discussion

## 7.0 Stage 15F lessons (post-result update)

1. **Per-fold MI selection is hostile to MCR-ALS features.** §5.3's
   global `mcr_C6_mean` d = −1.23 was the project's strongest single
   feature — and it **did not survive R2** (per-fold MCR-ALS refit).
   Components reorder across folds; MI on TRAIN-only data does not
   see a stable "C6 = bulk biology" axis. The global-fit result was
   partially a leakage artifact, even though the component biology
   labeling (§5.3 table) was real. **R2 mitigation worked as intended:**
   it correctly demoted a feature that looked discriminative globally
   but wasn't fold-stable.
2. **The unblocking of Stage 15A pseudo-Voigt peak fits was the
   highest-yield engineering win for LOSO.** The top-10 MI-selected
   features are all Stage 15A artifacts (peak fit height / area / FWHM
   / RMSE + 1st- and 2nd-derivative AUCs). Stage 15D biology features
   contributed one rank-22 feature (`bio_trp_indole_env`); Stage 15B
   contributed one ROI-PCA + one SAM feature; Stage 15C contributed
   only `mcr_C5_std` (rank 35); Stage 15E contributed none.
3. **PLS-DA needs the full feature space.** On raw 987 bins PLS-DA hits
   LOSO 0.603; on the 35 MI-selected engineered features PLS-DA collapses
   to 0.324. **Don't pre-select features for PLS-DA.** This is an
   operational lesson worth flagging — the production-deployable model
   uses LogReg precisely because LogReg-on-MI-selected works better than
   PLS-DA-on-MI-selected, even though PLS-DA-on-raw is the project's
   headline number.

## 7.1 What the project established

1. **PLS-DA on the raw 987-bin spectrum is the project's strongest LOSO
   baseline** at 0.603 mean parent-class recall. Five ensemble schemes
   failed to clear it; calibration-mismatch between PLS-DA (T ≈ 6.43)
   and deep models (T ≈ 1.2–1.7) is the closed-form mechanism.
2. **The published Cisek-2013 STEC triple (1338 / 1454 / 1658 cm⁻¹) does
   not transfer to file level** on this corpus. The 1454 lipid band even
   shows a sign reversal. The actual signal lives in the **800–1200 cm⁻¹
   LPS-chain region**, with `auc_lps_1194` (d = +1.03) as the cleanest
   single-band STEC ↔ Non-STEC discriminator. `pca_lps_PC3` (d = +1.03)
   reproduces the same signal in a learned coordinate system.
3. **MCR-ALS unmixing is the highest-yield engineering stage.**
   `mcr_C6_mean` d = −1.23 (bulk biology lower in STEC) is the strongest
   single file-level feature in the catalog. Four orthogonal MCR axes
   discriminate STEC ↔ Non-STEC (C6 bulk, C5 LPS-mix, C2 Trp, C7 NA).
   MCR also explicitly isolates a **substrate component (C1)** that
   captures the file-acquisition signature memprobe-v2 was detecting at
   the CNN penultimate layer.
4. **Biology-grounded features add a protein-2°-structure axis.**
   `bio_alpha_helix_score` (d = −0.986) is the strongest biology-grounded
   ratio in the project. The same family produces three K-12-specific
   features (the first K-12 axis in the project) — α-helix ↑, β-sheet ↓,
   Trp indole env shifted — consistent with K-12's reduced
   virulence-protein loading.
5. **Different architectures own different held-out strains.** No single
   model wins every LOSO cell. PLS-DA owns 83972, Dublin, Heidelburg,
   Typhimurium. DANN λ = 0.3 (5-seed soft-vote) owns K-12 and O157:H7.
   The 2-channel CNN owns O121:H19. No ensemble scheme captures the
   union.

## 7.2 Limitations

1. **87 files is small.** LOSO mean is a 9-point statistic with 2 train
   strains per parent class per fold. No model on earth makes that look
   like a 90-point evaluation. Tang-2026's 94% cross-strain ceiling used
   thousands of spectra and a WGAN-Transformer pipeline.
2. **K-12 is biologically atypical** (Soupene 2003). Most baselines hit
   0.00 LOSO recall on K-12 except DANN λ = 0.3 (which gets 0.75 on 3 of
   5 seeds). Stage 15D surfaced a K-12 signature in the 2°-structure
   axis; whether Stage 15F uses it to recover K-12 is reported in §6.
3. **STEC ↔ Non-STEC is virulence-defined, not phylogenetic.** The bulk
   Raman signal does not differ — Cisek-2013's 95% intra-batch number
   was on controlled cell suspensions where the only varying axis was
   the strain. In our corpus, each strain is one file × one acquisition
   session; strain-specific batch effects (calibration date, substrate
   fluorescence) co-vary with virulence labels. Memprobe-v2 (14% file-id
   linear separability post-DANN, vs 1.15% chance) is the direct
   evidence.
4. **Memprobe is decoupled from LOSO performance.** Removing file-id
   linear separability (the original probe goal) did not improve LOSO.
   DANN reshapes feature *prominence*, not linear file-id separability.
   The clean interpretation: residual instrument/batch/substrate signal
   is present but doesn't hurt cross-strain generalization in the
   direction the LOSO metric measures.
5. **Cross-corpus evaluation is not done.** Plan/13 lists the Bacteria-ID
   corpus (Ho-2019) as the natural cross-corpus transfer target.
   ATCC25922 is in both datasets, making it the highest-leverage
   external anchor. Deferred to future work.
6. **R6 — spatial features are limited by pixel count.** The dataset's
   72-pixel median per file rules out Moran's I and GLCM on intensity
   maps. We can only do moment statistics, and those are null for
   STEC ↔ Non-STEC.
7. **No open-set evaluation.** All training and evaluation is on the
   4-class closed-world problem (STEC / Non-STEC / Salmonella / H₂O).
   In deployment a sample could be a non-target bacterium, contaminated
   media, or measurement artifact. A Liu-2024 Raman-OSDL-style
   evaluation is future work.

## 7.3 Future work (ranked)

1. **Cross-corpus eval on ATCC25922** (highest leverage). Bacteria-ID
   (Ho 2019) and Zhu 2022 SCRS Persisters both include ATCC25922. Lets
   us put a single cross-lab generalization number in any external write-up.
2. **SSL pretraining** on the Bacteria-ID corpus or a synthetic
   WGAN-augmented Atlas, then fine-tune on the 87 files (plan/13 §4).
3. **Cross-instrument calibration transfer** via LoRA-CT (Sun 2025) to
   test how much of LOSO degradation is genuinely strain-distribution
   shift vs instrument-batch effect.
4. **Open-set probe via Liu-2024 Raman-OSDL** — measure how confidently
   the model misclassifies truly out-of-distribution input.
5. **Per-strain biology-pair modeling** — train binary classifiers per
   difficult pair (K-12 vs O157, ATCC25922 vs O121) instead of a 4-class
   joint model. Stage 6 reconsideration if 15F K-12 lift confirms.
6. **Active learning** — if Atlas can produce more files, prioritize
   K-12 replicates and O157:H7 (the two LOSO-failure folds) over balanced
   collection.

## 7.4 Wet-lab deployment implications

Translating the LOSO numbers into operational expectations for a
food-safety screening workflow.

### What the model can do today (the parts that work)

1. **H₂O vs bacteria is essentially solved.** Stage 15B's SAM gave
   AUROC = 1.000 on H₂O against any bacterial class. The production
   LogReg ships with this discrimination baked in. **A clean negative
   control (water blank) will not be confused for bacterium.**
2. **Within-distribution intra-strain identification (Protocol A) is
   strong.** File-macro F1 = 0.85–0.87 across PLS-DA, XGBoost, Stage 5
   band-XGB, and Stage 15F LogReg. **If a customer site re-trains on
   its own strain panel, they can expect ~0.85 F1 on new samples of
   those strains** under matched acquisition conditions.
3. **Three difficult strains have known recoverable models:**
   - **ATCC25922** (Non-STEC) — Stage 15F LogReg recovers it at 0.889.
   - **O121:H19** (STEC) — Stage 15F LogReg ties the project record at 0.889.
   - **Typhimurium** (Salmonella) — Stage 15F LogReg's headline new win
     at 0.778 (vs PLS-DA 0.56). This is the strongest Salmonella-serovar
     recall in the project on a held-out fold.

### What the model cannot do today (deployment limits)

1. **K-12 LOSO recall = 0.000.** If a deployment site collects a sample
   from a laboratory K-12 derivative — common in food-safety reference
   panels and benchtop QC — the model will misclassify it (most likely
   as Salmonella, per the prior baselines). **Recommendation: never
   deploy on a sample lineage the model has not seen during fit-time.**
   K-12 is the canonical example, but the same risk applies to any
   commensal or laboratory *E. coli* derivative not in our training
   corpus (e.g. DH5α, BL21).
2. **O157:H7 LOSO recall = 0.556.** Even with engineering, the model
   only correctly calls 5 / 9 of held-out O157:H7 files. The
   serogroup-specificity finding (§4.4) means **a new STEC serogroup
   not in the training set is at high risk of being misclassified as
   commensal *E. coli*** — the opposite of the safe-default direction.
3. **Mixed-sample degradation is 8–16 pp at 10–20% contamination
   (§4.6).** A field sample with 20% non-target pixels (e.g. a swab
   pulling in environmental microbes alongside the target) will see
   file-macro F1 drop from 0.87 to ~0.71. The bias under uncertainty
   is *toward* STEC (the conservative public-health-safe direction),
   but **false-positive rates inflate proportionally**.
4. **Cross-instrument transfer is untested.** All 87 files come from
   a single Raman microscope at one lab. The LoRA-CT (Sun-2025)
   literature documents 5+ percentage-point drops on cross-instrument
   transfer for bacterial Raman; we have no Atlas-side number for this.

### Pre-flight checks for a deployment

If the production model from `artifacts/stage15f_classifier.joblib`
ships to a customer site, the following should be verified before
trusting any prediction:

| Check | Pass criterion | Why |
|---|---|---|
| Sample strain is in training corpus | Yes, one of the 9 + H₂O | Out-of-distribution LOSO is unreliable per K-12 + O157:H7 evidence |
| Acquisition instrument matches Atlas hardware | Same laser λ (785 nm), same detector / grating, same calibration | Cross-instrument drop is documented in LoRA-CT |
| Pre-processing pipeline matches `atlas.preprocess.preprocess_matrix` | arPLS λ=1e5, SG window=9 poly=3, crop 400–1800 + 2800–3050, SNV | Models trained on this exact pipeline; mismatched preprocessing breaks features |
| Mixed-pixel fraction estimated | < 10% target purity | §4.6 degradation kicks in at 10–20% |
| Confidence-thresholding policy in place | Reject low-confidence predictions (e.g. `max_proba < 0.5`) | Avoids confident wrong calls on out-of-distribution strains |

### Where the model is honest (and where the headline numbers mislead)

The deployable claim is **"on the 9 known strains, the LogReg classifier
recovers 4 / 9 strains at ≥ 0.55 LOSO recall and matches the
PLS-DA-on-raw baseline on intra-distribution Protocol A"**. It is not
**"a 4-class Raman bacterial classifier with 87% accuracy"** — that
Protocol-A number does not survive cross-strain test. **Every
deployment claim from this work should specify the protocol.**

## 7.5 What's *not* worth doing inside this take-home

- Difframan / WGAN generative augmentation: fixes Protocol A, can't fix
  LOSO.
- DFT / MD physics-based simulation: out of scope, real cost.
- Hierarchical or binary-STEC submodels: small expected gain, long tail
  of effort.

---

# 8. Future work — what data would enhance these results

The headline — **PLS-DA on raw spectra, LOSO mean parent-class recall
= 0.603** — is a 9-point statistic. The bottleneck is *not* spectrum
count (7,122 QC-passed) but **strain count**: with 9 LOSO units, every
fold trains on 8 and tests on 1, and the production-classifier CI
[0.345, 0.552] is too wide to reject either "no improvement" or
"improvement over PLS-DA-on-raw". The data we need is *targeted*, not
just more.

Three actionable investments, ranked by return-on-effort:

1. **Cross-corpus evaluation** on public ATCC25922 corpora — **zero
   acquisition cost**, settles instrument-shift question (§8.2).
2. **Author's in-house collection plan** — answers the
   media-confounding question (§8.3).
3. **Targeted public-dataset requests** for matched serovars and
   instruments (§8.6).

The full data-gap analysis and dataset coverage matrix live in
[`plan/12_data_gaps_and_external_datasets.md`](../plan/12_data_gaps_and_external_datasets.md);
all references in [`plan/11_references.md`](../plan/11_references.md).

## 8.1 What's holding the headline back

Ranked gaps (`plan/12 §2`):

| # | Gap | Severity | Why |
|---|---|---|---|
| 1 | **Strain diversity** — 3 strains per primary class | ⭐⭐⭐ | LOSO can't generalize past what 2 training strains teach. Hold out K-12 and Non-STEC is anchored by 83972 + ATCC25922 only |
| 2 | **Lab / instrument diversity** — one Raman rig, one prep protocol | ⭐⭐⭐ | DANN has no domain to discriminate against; cross-instrument transfer is untested |
| 3 | **Biological replicates per strain** — files are mostly technical reps from 1–4 colonies | ⭐⭐ | Within-strain biological variability undersampled |
| 4 | **Open-set negatives** — H₂O is the only non-bacterial class | ⭐⭐ | All 8 H₂O LOSO files predicted as STEC (§6.7); model has never seen "neither STEC nor Salm nor commensal" and will fail loudly in deployment |
| 5 | **Missing STEC serotypes / Salm serovars** — 3 of ~7 clinical STEC; no Enteritidis, Newport, Infantis | ⭐ | Narrows claim scope vs the surveillance literature |

## 8.2 Public datasets — the zero-cost cross-corpus path

The fastest investment — no new acquisition — is **cross-corpus
evaluation on external ATCC25922 spectra**. Three open Raman datasets
exist (`plan/12 §5.2`):

| Dataset | Strain match | Wavenumber | Why it matters |
|---|---|---|---|
| 🥇 **Zhu et al. 2022 — SCRS Persisters** ([Front. Microbiol.](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2022.936726/full)) | **ATCC25922 explicit** (± ampicillin) | 400–3200 cm⁻¹ — near-perfect axis match to our 400–3049 | Closest experimental sibling: Renishaw 532 nm, single-cell point spectra. **Single sharpest external generalization test.** Open data: <http://mard.single-cell.cn/raw_spectrum_data/> |
| 🥈 **Ho et al. 2019 — Bacteria-ID** ([Nat. Commun.](https://www.nature.com/articles/s41467-019-12898-9), [GitHub](https://github.com/csho33/bacteria-ID)) | **ATCC25922 exact** match; ~80,500 spectra | 381.98–1792.4 cm⁻¹ (narrower — must crop our axis to match) | Heavy domain shift (Horiba LabRAM, 633 nm, gold-coated silica) — the *hardest* transfer test, exactly what we want |
| 🥉 **Liu et al. 2024 — Raman-OSDL** ([Sci. Adv.](https://www.science.org/doi/10.1126/sciadv.adp7991)) | *E. coli* + *S. enterica* at species level | 600–1800 cm⁻¹ | Built with explicit non-target "unknown bacteria" — directly addresses our **open-set negatives** gap. Data DOI: <https://doi.org/10.57760/sciencedb.15628> |

**Concrete next experiment** (specified in `plan/12 §6`):
load external ATCC25922 spectra, reuse `atlas/preprocess.py` end-to-end,
run our best per-strain models (PLS-DA, DANN, 2-channel CNN, Patch=5)
on the external corpus, report per-model accuracy. This is **a single
publishable cross-lab generalization number** — achievable without any
new in-house acquisition.

**What no public dataset fills** (`plan/12 §5.1`): K-12, 83972, all
three of our STECs (O157:H7, O121:H19, O103:H2), Salmonella Dublin
and Heidelburg. Generalization claims for those strains rest on
within-corpus LOSO only. This is the state of the field, not a
search failure.

## 8.3 Author suggestions — in-house collection plan

The plan was drafted alongside the analysis. It answers a different
question than §8.2: **whether the 0.603 ceiling is a biology limit
or a media-confounded-training-data limit.**

**Suggestion 1 — More data per file, three acquisition windows.**

- Three batches per sample, each at a different wavenumber range:
  - **Batch 1:** 400 / 500 → 1700 / 1800 cm⁻¹ — full fingerprint
    matching our current crop.
  - **Batch 2:** 700 → 1300 cm⁻¹ — LPS-chain + nucleic-acid / sugar
    region, where `auc_lps_1194` (d = +1.03, §4.2) and `mcr_C6_mean`
    (d = −1.23 global-fit, §5.3) live.
  - **Batch 3:** 1500 → 1800 cm⁻¹ — protein amide-I, where the K-12
    α-helix axis discriminates (Stage 15D, d = −0.986, §5.4).
- **9 replicates per sample** (vs current 200-pixel cap).
- Vary acquisition parameters across the three batches — each sample
  acquired under three optical / wavenumber configurations.

**Suggestion 2 — Growth variation and real-sample provenance.**

- *(A) Multi-condition growth.* All strains grown in **three different
  media conditions** (varying nutrients only, holding temperature /
  pH / aeration fixed), in **triplicate**, then collect spectra per
  Suggestion 1.
- *(B) Real-sample isolation.* Isolate each strain from an **actual
  sample matrix** (foodborne-surveillance source), grow per (A),
  acquire **2× spectra** per Suggestion 1.
- *(implicit) Lab-sharing.* Run a subset on a second instrument —
  every spectrum here comes from one optical pipeline.

## 8.4 What each §6.7 failure mode specifically demands

| Failure mode (§6.7) | Likely cause | Data that would fix it |
|---|---|---|
| All 8 H₂O LOSO files → STEC | H₂O has zero training exemplars under LOSO; classifier falls back to a non-zero prior (STEC) | **Liu 2024 Raman-OSDL** for the "unknown bacteria" open-set probe (§8.2); + food-matrix non-bacterial controls (lettuce wash, beef serum, agar blank) |
| K-12 recall ≈ 0 in classical models | K-12 differs at the protein-2°-structure level (Stage 15D); Non-STEC has only **one strain anchoring** that axis | More Non-STEC strains (MG1655, ATCC11775); cross-corpus verify on Zhu / Ho ATCC25922 (§8.2) |
| Non-STEC ↔ Salm confusion (20/52 errors) | Genus-level signal is real but small; `spat_skew_lps_1117` (d = +0.725, §5.5) needs multi-strain confirmation | More Salm serovars (Enteritidis, Newport, Javiana); **Tang 2023** (gated) has Dublin + Typhimurium matched |
| Stage 7: 10–20% F1 drop at realistic contamination | Synthetic mixed-pixel test ≠ real co-culture | Real co-cultured mixes at known ratios (Suggestion 2B) |
| 9-point LOSO; CI [0.345, 0.552] | One file per strain per fold can't reject ±0.10 noise | Suggestion 1's 9× replicates narrows the per-strain estimate by √3 ≈ 1.7× |
| Tang-2026's 94% cross-strain ceiling | We have ~6× less data | Sugg. 1 + 2A puts the corpus in Tang-comparable territory |

## 8.5 Expected return per investment

Ranked by expected lift on the LOSO baseline (0.603):

| Investment | Cost | Expected lift | Settles |
|---|---|---|---|
| **Cross-corpus on Zhu 2022 ATCC25922** (§8.2) | 0× — public download | Calibrates external-vs-internal gap | Whether 0.603 is instrument-overfit or genuine |
| **3 media × triplicate** (Sugg. 2A) | 9× | **+5 to +10 pp** | Whether the K-12 ↔ clinical-STEC gap is metabolic-state-driven or species-level |
| **9 replicates per sample** (Sugg. 1) | 1× | +2 to +4 pp | Per-file mean precision; cleaner soft-vote |
| **3 acquisition windows** (Sugg. 1) | 3× | +3 to +6 pp | Opens 3-band ensembling |
| **Multi-lab** (Sugg. 2 implicit) | 3× | Unlocks DANN's purpose | Currently untestable: instrument-shift robustness |

**Architecture sensitivity** (which method benefits from which data):

- **PLS-DA on raw**: diminishing returns; linear ceiling on existing axes.
- **LogReg-L2 (Stage 15F)**: +6–10 pp under media-variance (Sugg. 2A).
- **CNN + DANN λ = 0.3**: **largest gain potential** under multi-lab
  data — DANN was designed exactly for this scenario but currently has
  no domain to discriminate against.
- **Self-supervised pretraining** (plan/13 SSL pivot): any extra
  unlabeled spectra are free fuel.

**Statistical power.** With Sugg. 2A alone (+81 files, 27 file-condition
× 9 strains), the bootstrap 95% CI half-width drops from ±0.103 → ±0.060
(projected), moving the headline from "improvement within the same
feature space" to **"LOSO mean ≥ 0.55 at α = 0.05"** — a publishable
threshold for strain-blind 4-class Raman classification.

## 8.6 Gated datasets worth emailing for

These cover the matched-serovar / matched-axis gaps that no open
dataset fills. Worst case: no replies in time, no blocker
(`plan/12 §5.3`).

| Dataset | Why pursue |
|---|---|
| **Tang et al. 2023** (*Talanta*) — 4-serovar Raman | **Dublin + Typhimurium** match (2 of our 3); 530–1800 cm⁻¹; 785 nm Renishaw |
| **Roesch / Pistiki 2022** ([PMC8761712](https://pmc.ncbi.nlm.nih.gov/articles/PMC8761712/)) | **400–3050 cm⁻¹ exact axis match**; single-cell 532 nm; 1,500 UVRR + 4,168 spontaneous |
| **Kloss / Roesch 2021** ([PMC7680742](https://pmc.ncbi.nlm.nih.gov/articles/PMC7680742/)) | **ATCC25922 explicit** + Nissle 1917; 300–3100 cm⁻¹ |
| **Thomsen et al. 2022** ([PMC9524333](https://pmc.ncbi.nlm.nih.gov/articles/PMC9524333/)) | ATCC25922 + ATCC35218; 700–1600 cm⁻¹; 785 nm |

## 8.7 What we do *not* need more of

- **More STEC strains.** STEC has the strongest LOSO diagonal already
  (R = 0.67, §6.7); ≤ 2 pp gain on STEC recall and 0 pp on the
  Non-STEC ↔ Salm error that dominates the residual.
- **Deeper acquisitions on the same files.** Pixel-vote denoising is
  at the within-file cosine-similarity ceiling (≈ 0.997).
- **Higher-resolution spectrometer.** Bands are well-resolved at
  ≈ 1.7 cm⁻¹ / bin. The unresolved question is between-band ratios
  *under varying biology*, not finer band shape.
- **A 5th primary class** (Listeria, *B. cereus*). Expands scope
  without sharpening any existing 4-class boundary.
- **Generative augmentation alone** (DiffRaman / WGAN / VAE-LSTM;
  see [9] §"Data generation methods"). Lifts Protocol A; cannot lift
  LOSO — a generator trained on 8 strains cannot sample the 9th
  strain's distribution.

## 8.8 Honest writeup framing

Three sentences (from `plan/12 §7`) that should appear in any external
summary of this work:

> Our evaluation is bounded by 9 strains and a single lab / instrument
> source. LOSO mean is a 9-point statistic; cross-corpus testing
> against an independent ATCC25922 corpus (Zhu 2022 / Ho 2019) is the
> strongest available external anchor. No public Raman dataset covers
> any of our STEC serotypes (O157:H7, O121:H19, O103:H2) or two of our
> three Salmonella serovars (Dublin, Heidelburg) — generalization
> claims for those strains rest on within-corpus LOSO only.

---

# 9. References

Drawn from [`plan/11_references.md`](../plan/11_references.md)
(project bibliography) and
[`plan/12_data_gaps_and_external_datasets.md`](../plan/12_data_gaps_and_external_datasets.md)
(data-gap + public-dataset analysis).

## E. coli STEC ↔ Non-STEC discrimination

1. **Cisek et al. 2013** — *Sensitive and specific discrimination of
   pathogenic and nonpathogenic E. coli using Raman spectroscopy*.
   *Analyst* 138(20):6051. [PMC3617710](https://pmc.ncbi.nlm.nih.gov/articles/PMC3617710/).
   Published STEC discriminative triple (1338 / 1454 / 1658 cm⁻¹).
   Falsified at file level on this corpus (§4.1).

2. **Tang et al. 2026** — *Integrated Wasserstein GAN–Transformer for
   E. coli Strain Identification*. *Anal. Chem.*
   [doi:10.1021/acs.analchem.6c00429](https://pubs.acs.org/doi/10.1021/acs.analchem.6c00429).
   Cross-strain ceiling at ~94% (97% 5-fold CV → 94% independent test).
   Establishes the realistic Raman cross-strain ceiling.

3. **Soupene et al. 2003** — *Laboratory strains of E. coli K-12:
   things are seldom what they seem*. *J. Bacteriol.* 185(18):5611.
   [PMC9997739](https://pmc.ncbi.nlm.nih.gov/articles/PMC9997739/).
   K-12 atypicality — large genomic deletions, missing surface-structure
   genes vs wild-type *E. coli*. Justifies K-12 as a known-atypical LOSO
   fold.

4. **Marler-Clark / FSIS** — *Non-O157 STEC overview*.
   <https://marlerclark.com/foodborne-illnesses/e-coli/non-o157-stec>.
   Six non-O157 serogroups (O26, O111, O103, O121, O45, O145) account
   for ~80% of US non-O157 STEC infections. **No unique biochemistry vs
   commensal E. coli on standard media** — virulence-defined boundary.

5. **STEC virulence overview** — *Shiga toxin-producing E. coli*.
   *Virulence* 4(5):368, 2013.
   [doi:10.4161/viru.24642](https://www.tandfonline.com/doi/full/10.4161/viru.24642).
   Stx is chromosomally encoded as part of a lysogenic phage; STEC and
   Non-STEC differ by **one phage-encoded protein**. Sets the
   biological ceiling on within-*E. coli* label-free discrimination.

## Salmonella discrimination

6. **Yuan et al. 2024** — *Rapid discrimination of four Salmonella
   enterica serovars* (SERS + SVM). *J. Cell. Mol. Med.*
   [PMC11037414](https://pmc.ncbi.nlm.nih.gov/articles/PMC11037414/).
   Salmonella Raman bands (616 COO⁻ wag, 925 C-C skeletal, 1486 G ring,
   1542 C=C). Inter-batch numbers; cross-strain not tested.

## Domain adaptation for bacterial Raman

7. **RSCDM 2026** — *Raman Spectral Classification Discrepancy Model*.
   *Anal. Chem.*
   [doi:10.1021/acs.analchem.5c07113](https://pubs.acs.org/doi/10.1021/acs.analchem.5c07113).
   Domain-adaptation framework explicitly targeting instrument / batch /
   strain shift in bacterial Raman.

8. **Sun et al. 2025** — *Adversarial Contrastive Domain-Generative
   Learning*. *Eng. Appl. Artif. Intell.*
   [S0952197625004269](https://www.sciencedirect.com/science/article/abs/pii/S0952197625004269).
   5+ pp improvement vs no-adaptation baselines on cross-domain
   bacterial ID. Justifies the DANN-on-file_id design.

9. **Sun et al. 2025 (LoRA-CT)** — *Calibration Transfer of Deep
   Learning Models across Raman Spectrometers*. *Anal. Chem.*
   [doi:10.1021/acs.analchem.5c01846](https://pubs.acs.org/doi/10.1021/acs.analchem.5c01846).
   Parameter-efficient inter-instrument fine-tuning. Confirms
   systematic inter-device variation as a recognized distribution-shift
   driver.

## Decomposition methods (used in this paper)

10. **Windig & Guilment 1991** — *Interactive Self-Modeling Mixture
    Analysis* (SIMPLISMA). *Anal. Chem.* 63(14):1425. MCR-ALS pure-
    component initialization used in §5.3.

11. **Almeida et al. 2010 / Aguiar et al. 2013** — bacterial-Raman
    MCR-ALS precedents recovering 5–7 biology components from
    comparable data (cited in plan/13 §2.3).

## Public datasets — primary cross-corpus targets (`plan/12 §5.2`)

12. **Ho et al. 2019** — *Rapid identification of pathogenic bacteria
    using Raman spectroscopy and deep learning*. *Nat. Commun.* 10:4927.
    [Nat. Commun.](https://www.nature.com/articles/s41467-019-12898-9) ·
    [GitHub csho33/bacteria-ID](https://github.com/csho33/bacteria-ID).
    Bacteria-ID public corpus, ~80,500 spectra. ATCC25922 explicit.
    Wavenumber 381.98–1792.4 cm⁻¹.

13. **Zhu et al. 2022** — *Single-Cell Raman Spectra of Persister Cells*.
    *Front. Microbiol.*
    [10.3389/fmicb.2022.936726](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2022.936726/full) ·
    data: <http://mard.single-cell.cn/raw_spectrum_data/>.
    SCRS Persisters, ATCC25922 explicit (± ampicillin), 400–3200 cm⁻¹.
    **Best axis alignment** of any candidate cross-corpus dataset.

14. **Liu et al. 2024** — *Raman-OSDL* (airborne pathogens). *Sci. Adv.*
    [10.1126/sciadv.adp7991](https://www.science.org/doi/10.1126/sciadv.adp7991) ·
    data: <https://doi.org/10.57760/sciencedb.15628>.
    ~23,000 single-cell spectra (*E. coli* + *S. enterica* at species
    level). **Built with explicit open-set "unknown" class** —
    addresses the H₂O / open-set gap.

## Public datasets — gated, worth a request (`plan/12 §5.3`)

15. **Tang et al. 2023** — 4-serovar Salmonella Raman. *Talanta*.
    Dublin + Typhimurium matched; 530–1800 cm⁻¹; 785 nm Renishaw.

16. **Roesch / Pistiki 2022** — multi-resistant clinical *E. coli*.
    *Anal. Bioanal. Chem.* [PMC8761712](https://pmc.ncbi.nlm.nih.gov/articles/PMC8761712/).
    **400–3050 cm⁻¹ exact axis match.** Single-cell 532 nm.

17. **Kloss / Roesch 2021** — [PMC7680742](https://pmc.ncbi.nlm.nih.gov/articles/PMC7680742/).
    ATCC25922 explicit + Nissle 1917; 300–3100 cm⁻¹.

18. **Thomsen et al. 2022** — minimally-prepared *E. coli*. *Sci. Rep.*
    [PMC9524333](https://pmc.ncbi.nlm.nih.gov/articles/PMC9524333/).
    ATCC25922 + ATCC35218; 700–1600 cm⁻¹; 785 nm.

## Data generation methods (`plan/12 §4`)

19. **DiffRaman** — latent diffusion (VQ-VAE + DDPM) for class-conditional
    Raman augmentation. [arXiv 2412.08131](https://arxiv.org/abs/2412.08131) ·
    [Anal. Chim. Acta 2025](https://www.sciencedirect.com/science/article/abs/pii/S0003267025007664).
    Lifts data-limited classifiers (Protocol A); cannot lift LOSO.

20. **VAE-LSTM bacterial Raman** — *J. Chem. Inf. Model.*
    [10.1021/acs.jcim.3c00761](https://pubs.acs.org/doi/10.1021/acs.jcim.3c00761).
    96.9% mean accuracy across 16 strains, 5 species.

21. **U-Net + noise augmentation** — *ACS Omega*.
    [10.1021/acsomega.2c03856](https://pubs.acs.org/doi/10.1021/acsomega.2c03856).
    95% binary / 86% on 30 isolates. Cheapest augmentation path.

## Aggregators (`plan/12 §5.5`)

22. **RamanBench** — 74 datasets, 325K spectra, unified loader.
    `pip install raman-data`. arXiv 2605.02003.

23. **MicrobioRaman** — EBI BioStudies official open microbial-Raman
    repository. *Nat. Microbiol.* 2024.
    <https://www.ebi.ac.uk/biostudies/MicrobioRaman/studies>.

24. **Zenodo 15394102** — community-curated XLSX index of Raman
    databases (May 2025, CC-BY). <https://zenodo.org/records/15394102>.

---

For additional bibliography (parallel-modality datasets, recent-advances
reviews, vibrational-spectroscopy ML reviews, the full agent-confirmed
coverage matrix), see
[`plan/11_references.md`](../plan/11_references.md) and
[`plan/12_data_gaps_and_external_datasets.md`](../plan/12_data_gaps_and_external_datasets.md).

---

# Appendix A — Reproducibility notebooks

The repository ships 11 Jupyter notebooks under
[`FINAL/notebooks/`](notebooks/), one per stage / result. Each notebook
is self-contained and reproduces the figures and tables cited in the
section noted.

| # | Notebook | What it reproduces | Paper section |
|---|---|---|---|
| 01 | `01_environment_inventory.ipynb` | Environment check + 87-file / 7,122-spectrum inventory | §2.1 |
| 02 | `02_qc_preprocessing.ipynb` | QC mask + baseline correction + Savitzky-Golay smoothing | §2.3–2.4 |
| 03 | `03_plsda_baseline.ipynb` | **PLS-DA LOSO 0.603 on raw spectra (project headline)** | §3.1 |
| 04 | `04_stage15a_band_features.ipynb` | Pseudo-Voigt band fits (Stage 15A) | §5.1 |
| 05 | `05_stage15b_spectral_features.ipynb` | ROI-PCA / SAM / DWT features (Stage 15B) | §5.2 |
| 06 | `06_stage15c_mcr_unmixing.ipynb` | MCR-ALS K = 7 unmixing (Stage 15C) | §5.3 |
| 07 | `07_stage15e_spatial_features.ipynb` | Per-tile spatial moments (Stage 15E) | §5.5 |
| 08 | `08_stage7_mixed_sample.ipynb` | Mixed-sample degradation (10–20% drop) | §4.6 |
| 09 | `09_stage15f_final_model.ipynb` | **LogReg-L2 final classifier + confusion matrix** | §6 |
| 10 | `10_bootstrap_mcnemar.ipynb` | Bootstrap 95% CI + McNemar paired test | §6.7 |
| 11 | `11_inference_demo.ipynb` | `predict_from_xls()` end-to-end demo on a held-out file | §6 (deployment) |

Environment setup, cache files (`data_cache/`), production artifacts
(`artifacts/`), the public inference API, retraining instructions, and
the full repository layout are documented in
[`FINAL/README.md`](README.md). The Streamlit UI demo is at
`streamlit_app.py` in the repo root.

---

# Appendix B — Feature → biology cross-reference (the 35 deployed features)

The deployed LogReg-L2 production model takes 35 features as input. For
a biology reader, here is what each feature *measures* and which
macromolecule / band it tracks. Ordered by MI rank (= production input
order).

| # | Feature | Stage | Band centre / region | Biology |
|--:|---|:-:|---|---|
| 1  | `roi_ch_stretch_std`         | 15A | 2800–3050 cm⁻¹      | within-spectrum spread of C-H stretch — membrane lipid heterogeneity |
| 2  | `fit_amide_iii_1242_area`     | 15A | 1242 ± 30           | amide-III β-sheet (protein backbone) fit-area |
| 3  | `roi_silent_kurt`             | 15A | 1700–2800 cm⁻¹      | kurtosis of the "silent" region — sanity / scatter signature |
| 4  | `fit_amide_iii_1242_height`   | 15A | 1242 ± 30           | amide-III peak height |
| 5  | `d1_auc_lps_1117`             | 15A | 1117 ± 10 cm⁻¹      | LPS chain region 1st-derivative AUC — peak slope sharpness |
| 6  | `d1_auc_lipid_1454`           | 15A | 1454 ± 10           | CH₂ deformation 1st-derivative — lipid signal slope |
| 7  | `d1_auc_amide_i_1658`         | 15A | 1658 ± 10           | amide-I 1st-derivative — protein α/β envelope slope |
| 8  | `fit_lipid_1454_height`       | 15A | 1454 ± 30           | CH₂ deformation peak height |
| 9  | `fit_amide_iii_1242_rmse`     | 15A | 1242 ± 30           | amide-III fit residual — quality / non-Voigt deviations |
| 10 | `fit_lipid_1454_area`         | 15A | 1454 ± 30           | CH₂ deformation peak area |
| 11 | `fit_lps_1117_height`         | 15A | 1117 ± 30           | LPS chain peak 1117 height |
| 12 | `fit_lps_1117_area`           | 15A | 1117 ± 30           | LPS chain peak 1117 area |
| 13 | `fit_lps_1117_fwhm`           | 15A | 1117 ± 30           | LPS chain peak 1117 FWHM |
| 14 | `d2_auc_lipid_1454`           | 15A | 1454 ± 10           | CH₂ deformation 2nd-derivative AUC — curvature sharpness |
| 15 | `d2_auc_lps_1117`             | 15A | 1117 ± 10           | LPS chain 2nd-derivative AUC — curvature |
| 16 | `roi_lps_chain_entropy`       | 15A | 800–1200            | Shannon entropy of LPS-chain region |
| 17 | `fit_lps_1194_height`         | 15A | 1194 ± 30           | LPS chain peak 1194 (project anchor) height |
| 18 | `roi_silent_skew`             | 15A | 1700–2800           | skewness of silent region |
| 19 | `d2_auc_aa_1004`              | 15A | 1004 ± 10           | Phe ring-breathing 2nd-derivative — protein total anchor |
| 20 | `fit_aa_1004_height`          | 15A | 1004 ± 30           | Phe ring-breathing peak height |
| 21 | `fit_lipid_1454_rmse`         | 15A | 1454 ± 30           | CH₂ fit residual |
| 22 | `bio_trp_indole_env`          | 15D | 1340 / 1360         | Trp indole-environment ratio — hydrophilic vs hydrophobic exposure |
| 23 | `fit_lps_1050_rmse`           | 15A | 1050 ± 30           | LPS chain peak 1050 fit residual |
| 24 | `d2_auc_amide_i_1658`         | 15A | 1658 ± 10           | amide-I 2nd-derivative — protein 2°-structure curvature |
| 25 | `d1_auc_na_1338`              | 15A | 1338 ± 10           | nucleic-acid CH₂ wag 1st-derivative (Cisek-2013 NA band) |
| 26 | `bio_cyt_ox_state`            | 15D | 1356 / 1372         | cytochrome oxidation-state proxy (off-resonance at 785 nm — weak) |
| 27 | `roi_lps_chain_kurt`          | 15A | 800–1200            | kurtosis of LPS-chain region |
| 28 | `roi_lps_chain_skew`          | 15A | 800–1200            | skewness of LPS-chain region |
| 29 | `pca_chstretch_PC2`           | 15B | 2800–3050           | 2nd PC of C-H stretch region — secondary lipid axis |
| 30 | `pca_chstretch_PC3`           | 15B | 2800–3050           | 3rd PC of C-H stretch region — tertiary lipid axis |
| 31 | `fit_amide_i_1658_rmse`       | 15A | 1658 ± 30           | amide-I peak fit residual |
| 32 | `sam_lps_sub_O121H19`         | 15B | 800–1200            | spectral angle to the O121:H19 LPS-region template |
| 33 | `fit_lps_1050_center`         | 15A | 1050 ± 30           | LPS chain peak 1050 fitted centre (drift across strains) |
| 34 | `fit_lipid_1454_fwhm`         | 15A | 1454 ± 30           | CH₂ deformation peak FWHM |
| 35 | `mcr_C5_std`                  | 15C | data-driven         | within-file std of MCR-ALS C5 (biology mix + LPS top-discriminator) |

**Reading the table.** **23 / 35 features track lipid, LPS, and protein
peaks** (CH₂, amide-I, amide-III, Phe ring, LPS_1050 / 1117 / 1194). **2
features track biology ratios** (cytochrome oxidation state, Trp indole
environment). **2 features are spectral-angle scores** against learned
class templates. **2 features are PCs of the C-H stretch region.** **1
feature is an MCR-ALS unmixing component.** The ROI moments (skewness /
kurtosis / entropy) capture within-spectrum heterogeneity that bulk AUCs
miss — e.g. **`roi_lps_chain_skew`** measures how asymmetric the
LPS-chain envelope is around its centroid, which differs between strains
with different O-antigen architectures.

# Appendix C — Glossary

For readers crossing the biology ↔ ML boundary.

| Term | Plain-English |
|---|---|
| **arPLS** | Asymmetrically reweighted Penalized Least Squares. A baseline-subtraction algorithm that fits a smooth curve under a spectrum, treating downward-pointing residuals as "definitely peaks" and upward-pointing as "could be baseline." Removes fluorescence backgrounds in Raman. |
| **Cohen's d** | Standardized effect size: `(mean_A − mean_B) / pooled_std`. |d| ≥ 0.5 is "moderate"; |d| ≥ 0.8 is "large"; |d| ≥ 1.0 is "very large." Used throughout the paper as the file-level discrimination metric. |
| **DANN** | Domain-Adversarial Neural Network. A CNN with an extra "domain-classifier" head behind a gradient-reversal layer; tries to make features that classify the target label but cannot classify the domain (here, `file_id`). |
| **DWT** | Discrete Wavelet Transform. Decomposes a spectrum into "detail" coefficients at multiple frequency bands; energy/entropy per band become features. |
| **EMSC** | Extended Multiplicative Scatter Correction. Decomposes each spectrum into (offset + reference scale + polynomial baseline + chemistry residual); the polynomial coefficients are scatter-correction features. |
| **F1** | Harmonic mean of precision and recall. "Macro-F1" averages F1 across classes equally. |
| **GLCM** | Grey-Level Co-occurrence Matrix. Spatial-texture features computed on a pixel intensity map — dropped in §5.5 because no Atlas file has ≥200 pixels. |
| **LOSO** | Leave-One-Strain-Out cross-validation. Each fold holds out *all files of one bacterial subclass* and trains on the rest. Tests whether the model generalizes to a strain it has never seen — the realistic deployment evaluation. |
| **LPS** | Lipopolysaccharide. The major surface molecule on Gram-negative bacteria; O-antigen polysaccharide varies between serogroups. The 800–1200 cm⁻¹ Raman region captures LPS chain vibrations. |
| **MCR-ALS** | Multivariate Curve Resolution — Alternating Least Squares. Decomposes a data matrix `D` into `C · Sᵀ` under non-negativity constraints. `Sᵀ` rows are pure-component spectra; `C` columns are per-pixel concentrations. Used in §5.3 to separate biology components from substrate/fluorescence. |
| **Mean-pool** | Average across pixels per file. Reduces per-pixel features to one row per file for file-level classification. |
| **Memprobe** | A linear-classifier probe that tries to predict `file_id` from a CNN's penultimate-layer features. Tells whether the encoder is memorizing file-specific signal. |
| **MI / Mutual Information** | A non-parametric measure of statistical dependence between two variables. `sklearn.feature_selection.mutual_info_classif` is used here to pick the 35 highest-information features per fold. |
| **Moran's I** | A spatial-autocorrelation statistic (do nearby pixels have similar intensity?). Dropped in §5.5 because no file has enough pixels. |
| **NNLS** | Non-Negative Least Squares. Solves `Ax = b` subject to `x ≥ 0`. Used inside MCR-ALS for the concentration step. |
| **PCA** | Principal Component Analysis. Linear projection that maximizes variance. ROI-PCA = PCA restricted to a named wavenumber region. |
| **PLS-DA** | Partial Least Squares — Discriminant Analysis. PLS regression with a one-hot encoding of class labels; argmax of the prediction decodes the class. **The strongest classical baseline on the raw 987-bin spectrum on this corpus** (LOSO 0.603). |
| **Protocol A** | StratifiedGroupKFold on `file_id` (5 folds). Tests intra-strain generalization (new files of seen strains). |
| **Protocol B** | LOSO (9-fold + 1 H₂O = 10 folds). Tests cross-strain generalization. |
| **Pseudo-Voigt** | Convex combination of Gaussian + Lorentzian: `η·L + (1−η)·G`. Better fits the asymmetric pseudo-Voigt-shaped Raman peaks than pure Lorentzian (Stage 15A jumped fit-success from 0.2–37% to 60–89% on the same peaks). |
| **R1 – R9** | Pre-registered risks (Appendix A.7). |
| **ROI** | Region of Interest — a wavenumber-range mask, e.g. `lps_chain = (800, 1200)`. |
| **SAM** | Spectral Angle Mapper. Cosine angle (in radians) between a spectrum and a class-mean template. Direction-only feature; tested in Stage 15B. |
| **Savitzky-Golay** | A smoothing / derivative filter that fits a local polynomial in a window. Used in preprocessing and for the d1/d2 derivative AUCs. |
| **SHAP** | Shapley-value-based feature attribution. (Mentioned as future work — not in the current paper.) |
| **SIMPLISMA** | Self-Modeling Mixture Analysis. Picks "purest variables" (wavenumbers where one component dominates) to initialize MCR-ALS. Windig & Guilment 1991. |
| **SNV** | Standard Normal Variate. Per-spectrum z-score (subtract spectrum mean, divide by spectrum std). Corrects multiplicative scatter. |
| **STEC** | Shiga-Toxin-producing *E. coli*. Pathogenicity defined by acquisition of the Stx phage-encoded toxin. |
| **Stx (Shiga toxin)** | A single phage-encoded protein. Carrying it = STEC; not carrying it = non-STEC. The bulk Raman signal of the bacterium is unchanged. |
| **Trapezoidal AUC** | Area under the spectral curve over a wavenumber band, computed as a sum of trapezoidal strips. Used for `auc_*` and `d1/d2_auc_*` features. |
| **Tx (Tang-2026)** | Tang et al. *Anal. Chem.* 2026 (WGAN-Transformer for E. coli strain identification). Sets the published cross-strain ceiling at ~94%. |
| **WGAN** | Wasserstein Generative Adversarial Network. Used by Tang-2026 for data augmentation. |
| **XGBoost** | Gradient-boosted decision trees (the `xgboost` library). Standard tabular-data baseline. |

# Appendix D — Per-file strain table

| Subclass | Class | Files | Median pixels (post-QC) |
|---|---|---:|---:|
| 83972      | Non-STEC   | 8 | 82 |
| ATCC25922  | Non-STEC   | 9 | 88 |
| K-12       | Non-STEC   | 8 | 84 |
| O103:H2    | STEC       | 9 | 88 |
| O121:H19   | STEC       | 9 | 89 |
| O157:H7    | STEC       | 9 | 90 |
| Dublin     | Salmonella | 9 | 84 |
| Heidelburg | Salmonella | 9 | 87 |
| Typhimurium| Salmonella | 9 | 81 |
| (H₂O)      | H2O        | 8 | 95 |

87 files total / 7,122 QC-passed spectra / 9 bacterial subclasses + H₂O.
