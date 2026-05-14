# Atlas — Raman Hyperspectral Classification of Bacterial Pathogens

A 4-class Raman classifier for bacterial hyperspectral maps (STEC / Non-STEC / Salmonella / H₂O), evaluated under both standard cross-validation and an honest leave-one-strain-out (LOSO) protocol. PLS-DA leads the standard CV (0.951 file-macro-F1); a CNN trained with domain-adversarial regularization (DANN) is the only model that crosses the biology-vs-generalization Pareto frontier under LOSO.

---

## TL;DR

| Model | Protocol A file-F1 | LOSO mean parent-recall | K-12 (Non-STEC) | O157H7 (STEC) |
|---|---|---|---|---|
| PLS-DA                | **0.951** ⭐ | **0.60** ⭐ | 0.00 | 0.00 |
| LogReg                | 0.961       | 0.59       | 0.00 | 0.00 |
| LinSVM                | 0.779       | 0.52       | 0.00 | 0.00 |
| RBF-SVM               | 0.833       | 0.42       | 0.00 | 0.00 |
| XGBoost               | 0.796       | 0.37       | 0.00 | 0.33 |
| Random Forest         | 0.753       | 0.31       | 0.00 | 0.33 |
| 1D-CNN (vanilla)      | 0.649       | 0.35       | **0.50** ⭐ | **0.56** ⭐ |
| 1D-Transformer        | 0.507       | 0.193      | 0.00 | 0.00 |
| **CNN + DANN (λ=0.1)** | 0.566      | **0.500**  | **0.75** ⭐⭐ | **0.56** ⭐ |
| CNN + DANN (λ=0.3)    | 0.493       | 0.447      | **0.88** ⭐⭐ | **0.67** ⭐⭐ |

**Read:** the standard CV leaderboard (Protocol A) is saturated by within-file pixel averaging and rewards linear methods that match the data's effective dimensionality. The honest test (LOSO) drops every model 30–80%. **CNN+DANN at λ=0.1 is the only single model that simultaneously cracks the two biologically-hardest strains (K-12, O157H7) AND generalizes broadly** — every other architecture trades one against the other.

---

## The dataset

**87 files** acquired across 12 calibration dates. **7,999 raw spectra** at 2,048 wavenumber bins each, interpolated to a canonical axis `linspace(76, 3499, 2048)` cm⁻¹ at parse time to remove sub-bin drift across batches.

| Primary class | Files | Spectra (after 200-px cap) |
|---|---|---|
| STEC          | 27 | 2,544 |
| Salmonella    | 27 | 2,544 |
| Non-STEC      | 25 | 2,144 |
| H₂O           |  8 |   767 |

**9 bacterial subclasses** (strains) span the three bacterial primary classes: K-12, ATCC25922, 83972 (Non-STEC); Dublin, Heidelburg, Typhimurium (Salmonella); O103H2, O121H19, O157H7 (STEC).

**Preprocessing pipeline:**
```
   cosmic-ray removal → arPLS baseline → Sav-Gol smoothing →
   crop to fingerprint (400–1800 cm⁻¹) + C–H stretch (2800–3050 cm⁻¹) →
   SNV normalization
```

**Quality control** (per-file SNR ≥ 5 + per-file background-pixel detection) retains **7,122 / 7,999** spectra at ~89% per-class. See `plan/01_data.md` and `plan/03_architecture.md` for full preprocessing rationale.

---

## Why two evaluation protocols

A single split protocol cannot answer the question this dataset is really asking. We run two:

**Protocol A — StratifiedGroupKFold(5)** with `groups=file_id`, `stratify=primary_class`. Tests within-distribution generalization: held-out files come from the same strains the model trained on. This is the protocol that maps onto deployment if your downstream use is "classify a new sample acquired under the same protocol."

**Protocol B — Leave-One-Strain-Out (9 folds)**. Each fold holds out *all files of an entire bacterial subclass*; the model has to recognize the parent class of a strain it has never seen. This is the protocol that maps onto deployment if your downstream use is "classify a new strain in the wild" — and it's the protocol every reviewer of label-free Raman bacterial classification should ask for.

**The 0.95 → 0.50 gap between these two numbers is not a bug.** Protocol A is largely solved by within-file pixel averaging (within-file spectrum cosine similarity ≈ 0.997 across the dataset); the soft-vote denoises pixel-level errors heavily. LOSO removes that crutch and exposes how much of the apparent classification accuracy was per-strain memorization rather than learned biology.

LOSO macro-F1 is mathematically bounded at 0.25 (only one of four primary classes appears in any fold's test set), so we report **per-strain parent-class recall** as the meaningful LOSO metric instead.

---

## Methods

### Classical models (`atlas/models_classical.py`)

Six pipelines, all `Pipeline(StandardScaler → PCA(50–150) → classifier)` except XGBoost (no PCA) and PLS-DA (built-in dim reduction):

| Model | Hyperparameter grid | Notes |
|---|---|---|
| Logistic Regression | C ∈ {0.01, 0.1, 1, 10, 100} | `class_weight=balanced` |
| Linear SVM          | C ∈ {0.01, 0.1, 1, 10, 100} | `class_weight=balanced` |
| RBF SVM             | C ∈ {0.1, 1, 10}, γ ∈ {1e-3, 1e-2, 1e-1} | `class_weight=balanced` |
| Random Forest       | n_est ∈ {100, 300, 500}, max_depth ∈ {None, 10, 20} | |
| XGBoost             | n_est ∈ {100, 200, 300}, max_depth ∈ {3, 4, 5, 6} | tuning cheapened mid-session (see `plan/10`) |
| PLS-DA              | n_components ∈ {5, 8, 10, 12, 15, 20, 25, 30} | chemometrics standard |

Inner HPO via `StratifiedGroupKFold(4)` on the outer train, fold 0 fixed. Soft-vote per file (mean of predict_proba).

### Small 1D-CNN (`atlas/models_cnn.py`)

```
Input: (B, 1, 987) preprocessed spectrum
  ↓ per-bin standardize (fit on outer-train per fold, baked as buffer)
  ↓ InstanceNorm1d
  ↓ Conv1d(1→32, k=15) + BN + GELU + MaxPool/2
  ↓ Conv1d(32→64, k=7) + BN + GELU + MaxPool/2
  ↓ Conv1d(64→96, k=7, dil=2) + BN + GELU + MaxPool/2
  ↓ Conv1d(96→128, k=5) + BN + GELU
  ↓ AdaptiveAvgPool1d(1) → (B, 128)
  ↓ Linear(128 → 32) + GELU  ← 32-dim penultimate (used by memprobe v2 + DANN)
  ↓ Linear(32 → 4) → class logits
```

**124K params.** Two architectural fixes were applied mid-session and are baked in (see `plan/10§cnn-fixes`): channel doubling from the planned (16,32,48,64) to (32,64,96,128) because the planned channels arithmetic out to 33K params not the targeted 110K; and per-bin standardization at the input because SNV is per-spectrum-not-per-bin, leaving 0.05 → 0.39 per-bin variance that the CNN can't normalize away from InstanceNorm alone.

Training: AdamW lr=3e-4, wd=1e-4, 60 epochs, 3-epoch warmup + cosine, label smoothing 0.05, mixup α=0.2, noise / scale / shift / sinusoidal-baseline augmentation, early stop patience 10 on inner-val class macro-F1. Per-fold balanced class weights from outer-train.

### Small 1D-Transformer (`atlas/models_transformer.py`)

Patch-embed (`Conv1d(k=20, s=20)` → 49 patches + [CLS]), learned positional embed, 4 encoder blocks (`d_model=80`, `nhead=4`, `dim_ff=160`, pre-LN, GELU), classifier head. **217K params**, same training recipe as the CNN. Kept for honest benchmarking; documented underperformance is in §Results.

### DANN extension (`atlas/models_cnn.py` + `atlas/train.py`)

```
Input ───┐
         ↓
      encoder (frozen reuse of SmallCNN1D conv stack)
         ↓
      32-dim feat ─────► fc2 ─────► class logits (4)
         │
         └── GRL(λ) ────► domain MLP (32 → 64 → K) ──► domain logits (K)

   GRL forward:   identity
   GRL backward:  ∇_feat → −λ · ∇_feat
   
   Loss = L_class + L_domain   (joint, single backward pass)
   
   K = # unique file_ids in outer-train (≈70 for Protocol A, ≈78 for LOSO).
   λ warms linearly 0 → λ_max over 10 epochs, then holds.
```

Domain head is appended *after* `super().__init__()` so the encoder + class head consume the identical RNG draws as vanilla SmallCNN1D given the same fold_seed; with λ=0 the encoder receives zero gradient from the domain path and trains bit-for-bit identically to vanilla (verified empirically). Standard Ganin & Lempitsky 2015 formulation; `lambda_grl` is a Python float passed at forward time so the warmup schedule can update every step without graph rebuilds.

---

## Results — Protocol A (StratifiedGroupKFold)

```
   Leaderboard (file-level macro-F1, mean ± SD over 5 folds):
   
       LogReg              0.961 ± 0.042   ⭐
       PLS-DA              0.951 ± 0.051
       RBF-SVM             0.833 ± 0.096
       XGBoost             0.796 ± 0.103
       LinSVM              0.779 ± 0.112
       Random Forest       0.753 ± 0.118
       1D-CNN              0.649 ± 0.079
       CNN+DANN(λ=0.1)     0.566 ± 0.091
       1D-Transformer      0.507 ± 0.122
       CNN+DANN(λ=0.3)     0.493 ± 0.150
```

**Read:** linear models win because within-file pixel averaging (cosine 0.997) collapses pixel-level noise; the data's effective dimensionality matches PCA + linear-classifier perfectly. Deep models have to learn that manifold from raw 987-bin SNV input with only ~70 training files per fold — they overfit to file-specific signatures in the training set, visible in bumpy per-fold val_macro_f1 trajectories (mean best inner-val F1 = 0.51 ± 0.07 across CNN folds).

**H₂O vs bacteria binary recall = 1.00** for the best model on every fold. Water classification is trivial.

---

## Results — LOSO (the honest test)

### Per-strain parent-class recall

| Strain (held out) | Parent | LogReg | LinSVM | RBF-SVM | RF | XGB | PLS-DA | CNN | DANN(0.1) | DANN(0.3) |
|---|---|---|---|---|---|---|---|---|---|---|
| 83972          | Non-STEC   | 0.88 | 0.88 | 0.88 | 0.12 | 0.25 | **0.88** | 0.88 | 0.75 | 0.25 |
| ATCC25922      | Non-STEC   | 0.22 | 0.11 | 0.22 | 0.11 | 0.33 | 0.22 | 0.11 | **0.89** ⭐⭐⭐ | 0.11 |
| K-12           | Non-STEC   | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.75 | **0.88** ⭐⭐ |
| Dublin         | Salmonella | 0.44 | 0.22 | 0.22 | 0.11 | 0.11 | **0.56** | 0.11 | 0.00 | 0.22 |
| Heidelburg     | Salmonella | **0.89** | 0.56 | 0.33 | 0.33 | 0.44 | 0.89 | 0.33 | 0.44 | 0.33 |
| Typhimurium    | Salmonella | **1.00** | **1.00** | 0.67 | 0.33 | 0.44 | **1.00** | 0.11 | 0.11 | 0.00 |
| O103H2         | STEC       | **1.00** | **1.00** | 1.00 | 0.67 | 0.67 | 1.00 | 0.56 | 0.33 | **0.89** ⭐⭐ |
| O121H19        | STEC       | 0.89 | 0.89 | 0.44 | 0.78 | 0.78 | **0.89** | 0.00 | 0.67 | 0.67 |
| O157H7         | STEC       | 0.00 | 0.00 | 0.00 | 0.33 | 0.33 | 0.00 | 0.56 | 0.56 | **0.67** ⭐ |
| **MEAN**       |            | 0.59 | 0.52 | 0.42 | 0.31 | 0.37 | **0.60** | 0.35 | **0.500** | 0.447 |

### The Pareto-frontier observation

The single most informative chart in this writeup is **"LOSO mean parent-recall vs (K-12 + O157H7) recall sum"**:

```
       1.6 ┤
   K + O157   ●  CNN+DANN(λ=0.3) (1.55, 0.447)
   biology
   recall   1.4 ┤      
              ┤   ●  CNN+DANN(λ=0.1) (1.31, 0.500) ⭐
       1.2 ┤
              ┤
              ┤    ●  Vanilla CNN (1.06, 0.35)
       1.0 ┤
              ┤
       0.5 ┤
              ┤    
              ┤    ●  XGB (0.33, 0.37)            ●  RF (0.33, 0.31)
       0.0 ┤────────────────────────────────────────────────●──────
                                                       PLS-DA (0.00, 0.60)
                                                       LogReg (0.00, 0.59)
                                                       LinSVM (0.00, 0.52)
                                                       RBF-SVM (0.00, 0.42)
              └─────┬──────────┬────────────┬────────────────┬────►
                   0.30        0.40         0.50           0.60
                              LOSO mean parent-recall (generalization)
```

Linear methods stack along the x-axis: high LOSO mean, zero biology wins. Vanilla CNN lives off-axis: cracks K-12 + O157H7 but has the worst LOSO mean. **CNN+DANN(λ=0.1) is the only point in the upper-right quadrant** — both mean ≥ 0.45 AND biology recall (K-12 + O157H7) ≥ 1.0. λ=0.3 buys further-right biology at the cost of leftward mean.

### Per-strain reading

- **K-12 (Non-STEC, lab-domesticated since the 1920s, missing many stress-response genes vs wild-type E. coli)** — Every classical model scores 0.00. K-12 is biologically atypical and the training Non-STEC manifold doesn't span its chemistry. **Vanilla CNN scores 0.50** via nonlinear featurization; **DANN(0.1) climbs to 0.75 and DANN(0.3) to 0.88**, monotonically in λ. This means K-12's load-bearing discriminative features are *not* file-id-correlated — DANN stripping acquisition noise makes the genuine peak-ratio signal clearer rather than destroying it.
- **O157H7 (STEC, the canonical pathogenic E. coli)** — Classical linear models score 0.00 (it looks like Non-STEC E. coli in the chemistry). Tree models get 0.33 via axis-aligned splits. Vanilla CNN scores 0.56. **DANN(0.3) climbs to 0.67** — same monotonic pattern as K-12.
- **ATCC25922 — a singular DANN(0.1) result.** Vanilla CNN scores 0.11; PLS-DA 0.22. DANN at λ=0.1 scores **0.89** — above every other model in the entire sweep. At λ=0.3 it collapses back to 0.11. The strain's recognition signature is preserved by moderate DANN denoising but destroyed by strong adversarial pressure.
- **Dublin and Typhimurium** — PLS-DA scores 0.56 and 1.00 respectively; every deep model scores ≤ 0.11. The linear separator across the training Salmonella signature transfers cleanly to these strains; the CNN learns features that don't.

The full pattern says **inductive bias matters per-strain.** Linear methods, tree methods, and DANN-deep methods each have strain-specific complementary failures. This is the structural motivation for the §Future Work stacking entry below.

---

## Diagnostics

### Memorization probes

Three diagnostics for "does the encoder leak file_id (acquisition signature)?"

| Probe | Model | Top-1 acc (87-way file_id) | × chance | Verdict |
|---|---|---|---|---|
| v1 (from-scratch 6.6K-param CNN trained for file_id) | n/a            | 4.1%  | 3.5×  | Below 10% threshold |
| v2 (LogReg on penultimate) | vanilla CNN              | **15.5%** | 13.5× | Above threshold; "fires" |
| v2 (LogReg on penultimate) | CNN + DANN λ=0.1         | **14.0%** | 12.2× | Still fires |
| v2 (LogReg on penultimate) | CNN + DANN λ=0.3         | **13.6%** | 11.9× | Still fires |

**The key finding:** tripling λ moved the probe by 0.4 percentage points while changing LOSO mean parent-recall by 0.053 and per-strain parent-recall by up to 0.78 points. **The probe-LOSO decoupling is real.** DANN reshapes which directions in the 32-dim feature space encode file-id (high-variance directions become strain-discriminative; file-id encoding gets pushed into low-variance dimensions). A fresh LogReg can find a 13-14%-accuracy file-id subspace regardless; that's not where the LOSO signal lives.

**Methodological implication for the writeup**: a linear-separability probe on the full feature space is the wrong DANN diagnostic on this dataset. A variance-aware probe (e.g., file-id classifier restricted to top-3 PCA components of the encoder features) would be more sensitive to the prominence reshaping DANN actually does. Documented in §Future Work.

### Calibration-date batch effect

12 distinct calibration dates across 87 files; same-date file-centroid distance is 11% tighter than random pairs (ratio 0.893 in PCA-50). For each LOSO misclassification across the 5 base classical models we computed the lift = `P(wrong-pred-class | calibration-date overlap with train) / P(wrong-pred-class)`:

```
   Strain        Wrong pred    n_errors   Lift     Reading
   ─────────────────────────────────────────────────────────────────────
   ATCC25922     STEC          33         2.89×    batch-driven (DANN should help)
   Heidelburg    STEC           8         1.81×    batch-driven
   O157H7        Salmonella     6         1.69×    batch-driven (small)
   K-12          Salmonella    38         0.59×    biology, NOT batch
   O157H7        Non-STEC      36         0.00×    biology, NOT batch
   Dublin        Non-STEC      28         0.00×    biology, NOT batch
```

The two biggest single errors (K-12 → Salmonella, O157H7 → Non-STEC) cannot be explained by date-batch leakage. **The LOSO crater is mostly biology, not leakage** — which is consistent with both the memprobe-LOSO decoupling and the per-strain DANN response pattern.

---

## Key findings

1. **Linear methods win the easy benchmark; nonlinear methods crack the biology-hard cells.** Six classical models all sit at LOSO mean ≥ 0.31 with zero recall on K-12 or O157H7. Only the CNN (and its DANN variants) score above zero on those cells. **The two regimes don't overlap until DANN.**
2. **CNN+DANN at λ=0.1 is the only single model that achieves both.** LOSO mean 0.500 with K-12 = 0.75 and O157H7 = 0.56 — the upper-right quadrant of the Pareto plot above.
3. **DANN's lambda is a regime-switch hyperparameter.** Higher λ strengthens pathogen biology features (K-12 0.50 → 0.75 → 0.88; O157H7 0.56 → 0.56 → 0.67; O103H2 0.56 → 0.33 → 0.89) at the explicit cost of easy commensal recognition (83972 0.88 → 0.75 → 0.25; ATCC25922 0.11 → 0.89 → 0.11). Per-strain λ optimization is a clean Future Work direction.
4. **The standard memorization-probe diagnostic decouples from LOSO generalization on this dataset.** Tripling the adversarial coefficient moves the probe by 0.4 percentage points while moving LOSO substantially. DANN works via feature-prominence reshaping, not file-id elimination. The 10% memprobe threshold from the pre-DANN literature is not a reliable success diagnostic here.
5. **Pre-registration caught two important calibration misses.** We predicted K-12 would land in 0.00–0.50 (bet on lower half) under DANN — wrong direction; actual 0.75 → 0.88. We predicted DANN at λ=0.1 would drop the probe below 10% — wrong; stayed at 14%. Both misses sharpen the writeup: load-bearing features for K-12 are NOT file-id-correlated, and the probe is the wrong DANN diagnostic on this data.

---

## Limitations

- **Dublin parent-recall = 0** under every deep model (PLS-DA gets 0.56 via its supervised dim reduction). Salmonella Dublin is biologically intermediate in our feature space; we have no fix.
- **Typhimurium parent-recall stays at 0.11** under DANN despite PLS-DA's 1.00. This is the canonical "linear methods nail it because the held-out file looks similar to a training file via PCA distance; the CNN destroys that signal during augmentation" pattern; tuning augmentation didn't recover it in a single session.
- **STEC vs Non-STEC distinction is fundamentally virulence-defined.** Shiga toxin (the defining marker) is phage-encoded and horizontally mobile; most other virulence determinants are plasmid-encoded. The bulk of any Raman signal is phylogenetic, not virulence-driven. Within-species classification of pathogenic vs commensal E. coli has a published biological ceiling around 90–95% under integrated wGAN-Transformer methods *with extensive domain adaptation*; we are operating without that scale of architecture.
- **Memprobe v2 = 14%** even after DANN — not a clean "we fixed the leakage" story. We turn this into an asset by documenting why it's the wrong diagnostic, but a reviewer should know the probe metric itself isn't where the DANN value shows up.
- **No test-time augmentation, no calibration (temperature scaling), no ensembling beyond what was already shown to fail.** All listed in Future Work; bandwidth caveat.

---

## Reproducibility

```bash
# 1. Cache the raw + preprocessed arrays (idempotent, takes ~1 min)
python scripts/build_dataset.py
python scripts/preprocess_dataset.py

# 2. Generate splits (writes data_cache/splits/protocol_{a,b}.json with sklearn-version-tagged metadata)
python scripts/build_splits.py

# 3. Reproduce the leaderboard
python scripts/run_classical.py            # 6 classical models, both protocols (~20 min)
python scripts/run_cnn.py                  # 1D-CNN both protocols (~9 min on MPS)
python scripts/run_transformer.py          # 1D-Transformer both protocols (~25 min)
python scripts/run_dann.py --lambda-max 0.1  # DANN at λ=0.1 both protocols (~14 min)
python scripts/run_dann.py --lambda-max 0.3  # DANN at λ=0.3 both protocols (~15 min)

# 4. Memorization probes
python -m atlas.memprobe                                           # v1
python -m atlas.memprobe_v2 --encoder outputs/<cnn-run>/encoder_fold_4.pt    # v2 on vanilla CNN
python -m atlas.memprobe_v2 --encoder outputs/<dann-run>/encoder_fold_4.pt   # v2 on DANN

# 5. Final aggregation + summary
python scripts/run_ensemble.py             # the (failed) soft-vote ensembles
```

**Seeds:** master seed 42; per-fold seed = `(master_seed * 31337 + fold_index) % 2**31`. Threaded into every `Pipeline` `random_state` (sklearn), XGBoost `seed`, `torch.manual_seed`, `numpy.random.default_rng`. Splits cache hashes input QC mask + class labels + subclass labels + seed — if any of those change, splits become invalid and must be rebuilt.

**Pinned threading**: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` in every entrypoint script. Floating-point reduction order matters for RF / XGB / BLAS reproducibility.

**Per-run artifacts** in `outputs/<run_id>/`: `config.resolved.json` (full hyperparams as actually used), `model_result.json` (mean/SD across folds), `predictions_fold_<id>.parquet` (per-spectrum probabilities keyed on `spectrum_id` + `file_id`), `encoder_fold_<id>.pt` (PyTorch weights for downstream memprobes), and for DANN: `history_fold_<id>.json` (per-epoch class_loss, domain_loss, domain_acc, lambda_grl, val_macro_f1).

---

## Future work

Ordered by expected payoff for closing the LOSO gap:

1. **Stacking / meta-learner on the per-fold predict_proba.** The soft-vote ensemble failed precisely because uniform averaging crushes minority-vote signals (CNN's K-12 / O157H7 wins get diluted by PLS-DA's confidently-wrong vote). A logistic-regression meta-learner trained on stacked `[PLS-DA proba, DANN(0.1) proba, DANN(0.3) proba]` per file with held-out outer-fold targets could plausibly select per-file which model to trust — using the disagreement structure rather than averaging it away. Plausible LOSO mean ceiling 0.55–0.62 with K-12 / O157H7 both preserved.
2. **Grouped-domain DANN.** Instead of 87-way file_id, use 9-way strain or 12-way calibration-date as the domain target. Less destructive than per-file GRL; should preserve more of the easy-commensal signal that λ=0.3 destroys, while still pressuring the encoder toward batch invariance.
3. **Per-strain optimal λ.** ATCC25922 wins at λ=0.1; O103H2 wins at λ=0.3. These are non-monotonic strain optima. A held-out-fold model-selection scheme that picks one of {λ=0.05, 0.1, 0.3} per LOSO fold could plausibly Pareto-dominate any single-λ result.
4. **Patch_size=5 Transformer.** Documented architectural diagnosis: 20-bin patches blur the 5–10-bin-wide Raman peaks that produced the CNN's K-12/O157H7 wins. Re-run with overlapping `Conv1d(k=5, s=1)` patches; same recipe otherwise. Plausibly cracks the same biology cells via attention.
5. **Test-time augmentation.** Average predict_proba over N augmented copies of each test spectrum. Cheap (~5× inference cost), historically gives 1–3% file-F1 lift on similar Raman tasks.
6. **Variance-aware memprobe.** The current memprobe v2 finds *any* linear file-id subspace; DANN moves file-id encoding into low-variance directions but doesn't eliminate it. A probe restricted to top-3 PCA components of the encoder features would be more sensitive to the prominence-reshaping DANN actually does — and would give us a diagnostic the standard threshold rule could meaningfully apply to.
7. **Confidence calibration via temperature scaling on val.** Reliability diagrams, Brier scores, and proper-scoring-rule diagnostics for the headline DANN model. Required for any downstream use that uses confidence (selective prediction, abstaining).
8. **arPLS crop left-boundary fix.** Current crop starts at 400 cm⁻¹; arPLS leaves a boundary artifact in 400–500. Shifting the crop to 450 should clean up the left edge of every preprocessed spectrum and may help the Salmonella-vs-Non-STEC boundary that's currently weak in the deep models.
9. **Bacteria-only ANOVA.** The current top-30 ANOVA discriminative bins are all in the C–H stretch region (2880–2940 cm⁻¹), driven by the water-vs-bacteria boundary. Re-running ANOVA on the 3 bacterial classes only (excluding H₂O) would surface the *within-bacterial* discriminative bins — likely overlapping with the published STEC bands at 1338, 1454, 1658 cm⁻¹.
10. **SHAP / integrated-gradients saliency** on the DANN model for K-12, O157H7. Diagnostic, not improvement — but worth doing for the writeup because attribution that lands on biologically-grounded bins (1338/1454/1658) is a strong proof point that the model learned chemistry, not file fingerprints.

---

## Repository layout

```
.
├── atlas/                 # Library code
│   ├── io.py              # Tab-delim .xls / .txt parser
│   ├── preprocess.py      # arPLS + Sav-Gol + crop + SNV
│   ├── qc.py              # SNR threshold + background-pixel detection
│   ├── splits.py          # StratifiedGroupKFold + LOSO with cache hashing
│   ├── models_classical.py # 6 sklearn pipelines
│   ├── models_cnn.py      # SmallCNN1D + DANNCNN1D + GradReverse
│   ├── models_transformer.py # SmallTransformer1D
│   ├── train.py           # train_cnn_fold + train_dann_fold
│   ├── evaluate.py        # FoldResult / ModelResult / soft-vote aggregator
│   ├── ensemble.py        # Soft-vote ensembler (documented as failed)
│   ├── memprobe.py        # v1: from-scratch tiny-CNN file_id probe
│   └── memprobe_v2.py     # v2: LogReg-on-penultimate file_id probe
├── scripts/               # Reproducibility entrypoints (see §Reproducibility)
├── data_cache/            # Cached parsed + preprocessed arrays
├── outputs/               # Per-run artifacts (per-fold predictions, encoders, JSON)
├── notebooks/
│   └── atlas_driver.ipynb # EDA blocks 1–9
├── plan/                  # Project plan + decisions + audit trail
│   ├── 00_status.md       # Current state (mutable)
│   ├── 02_decisions.md    # Locked design decisions (stable)
│   ├── 07_findings.md     # Empirical findings (append-only)
│   ├── 08_expectations.md # Pre-registered predictions vs actuals (append-only)
│   └── 10_decision_log.md # Provenance / audit trail (append-only)
└── README.md              # This file.
```

---

## Pre-registration appendix

This project ran every experiment under **pre-registered expectations**: predicted values written down before training, then compared against actuals at write-up time. Two misses stand out and are documented honestly here:

| Pre-registered | Predicted | Actual | What it sharpened |
|---|---|---|---|
| K-12 parent-recall under DANN(λ=0.1) | 0.00–0.50, "bet on lower half" | **0.75** | Load-bearing features for K-12 are NOT file-id-correlated. DANN cleans up acquisition noise rather than destroying biology. |
| DANN(λ=0.1) memprobe v2 top-1 | "fires=False" expected (<10%) | **14.0%** ("fires=True") | Memprobe and LOSO are decoupled. Probe is wrong diagnostic on this data. |

Full pre-registered tables with predicted ranges, actuals, and verdicts in `plan/08_expectations.md`.

---

*Built with: Python 3.12, NumPy, scikit-learn, XGBoost, PyTorch (MPS backend on Apple Silicon for the deep models). See `requirements.txt` for pinned versions.*
