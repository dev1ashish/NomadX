# 2026-05-18 — Stage 15F: full-feature LOSO classifier {#2026-05-18--stage15f-full-classifier}

> **Status:** complete
> **Stage / track:** [plan/15 §6 Stage 15F](../15_feature_engineering_research.md), final
> stage of the feature-engineering implementation track. Gates Phase 2 (paper)
> and Phase 3 (Streamlit deploy) of [plan/16](../16_deploy_plan.md).
> **Branch hit:** **(C) Plateau.** Best algorithm LogReg-L2; LOSO mean parent-class accuracy = **0.436** (below 0.45 (C)/B boundary); per-fold held-out-class macro recall = 0.109 (consistent with single-class-per-fold structure). PLS-DA on raw 987-bin spectrum baseline (0.603 LOSO mean) **remains unbeaten** by the 259-feature engineered cache under per-fold MCR-ALS/ROI-PCA/SAM refit + MI selection.
> **One-line headline:** **Engineered features plateau below raw-spectrum PLS-DA at LOSO** — LogReg-L2 wins the 3-way comparison (LOSO 0.436 vs PLS-DA 0.324 vs XGB 0.247), surfaces ATCC25922 + O121H19 at **0.889** each + O157H7 at 0.556 + Typhimurium at 0.778, but **K-12 collapses to 0.000 across all 3 algos** (Stage 15D's α-helix axis was not selected per fold). Production artifact shipped; Branch (C) sustains plan/13 SSL pivot for future work.
> **Cross-refs:** [Stage 15A](2026-05-18_stage15a_pseudovoigt_roi_emsc_derivatives.md) · [Stage 15B](2026-05-18_stage15b_dwt_pca_sam.md) · [Stage 15C](2026-05-18_stage15c_mcr_als_unmixing.md) · [Stage 15D](2026-05-18_stage15d_biology_features.md) · [Stage 15E](2026-05-18_stage15e_spatial_features.md) · [plan/16 deploy plan](../16_deploy_plan.md)

---

## Pre-registration

### Method

**Inputs (4 feature caches, joined on `file_id`):**

| Cache | Shape | Level | Notes |
|---|---|---|---|
| `data_cache/band_features.parquet` | 7,122 × 166 | per-pixel | QC-filtered. Mean-pool over pixels per `file_id` → (87, 166). |
| `data_cache/spectral_features.parquet` | 7,122 × 51 | per-pixel | QC-filtered. Mean-pool over pixels per `file_id` → (87, 51). |
| `data_cache/unmix_features.parquet` | 87 × 33 | per-file | Already file-level (mean / std / max / p90 + residual_norm). |
| `data_cache/spatial_features.parquet` | 87 × 10 | per-file | Already file-level (moment statistics). |

Final design matrix: **87 files × ~259 features** (after dropping the
`mcr_C8_*` slot if Stage 15C effective K=7 takes effect; ≤260 either way).

**Evaluation protocol — strain-level LOSO with per-fold leak-fitted features.**

- **10 folds.** One per held-out strain: 9 bacterial subclasses
  (`O157H7`, `O121H19`, `O103H2`, `ATCC25922`, `83972`, `K-12`, `Dublin`,
  `Heidelburg`, `Typhimurium`) + 1 H₂O fold (8 water files held out together).
- **Per-fold refit (R2 mitigation):**
    - MCR-ALS (K=7, the effective K per [Stage 15C](2026-05-18_stage15c_mcr_als_unmixing.md))
      via `MCRALSWrapper.fit/transform` on train pixels only.
    - ROI-PCA via `fit_roi_pca/transform_roi_pca` (LPS/amide/CH-stretch
      PCs, 11 features) on train pixels only.
    - SAM templates via `fit_sam_templates/transform_sam` on train pixels only.
  These overwrite the cached file-level mean-pool of MCR/PCA/SAM columns for
  the held-out fold's rows.
- **Per-fold MI selection (R1 mitigation):** `sklearn.feature_selection.mutual_info_classif`
  with `n_jobs=1, random_state=seed`, target **30–40 features** out of 259.
  Selection runs on the train fold's (file, features) matrix — held-out file's
  features are masked through the same selector at predict time.
- **Algorithms (3):**
    - PLS-DA (`PLSRegression` w/ one-hot → argmax decoder)
    - LogReg L2 (`LogisticRegression(penalty='l2', C=1.0)`)
    - XGBoost (`XGBClassifier(n_estimators=200, max_depth=4, eval_metric='mlogloss')`)
  Each runs in a `Pipeline([StandardScaler, classifier])`.
- **Multi-seed:** 5 seeds × 10 folds × 3 algos = 150 fits. Per-seed LOSO
  mean parent-class recall + variance across seeds reported per algo.

**R7 — shuffled-label permutation test on `mcr_C1_*` features.**
After picking the best algorithm: refit with `y` shuffled within fold (10
shuffles), aggregate feature importance, compare `mcr_C1_*` real-vs-shuffled
importance. If shuffled ≥ 50% of real for any `mcr_C1_*` slot, drop those
4 features and re-run all 5 seeds. Log the verdict in Results.

**R9 — score-vs-fraction handling.** `bio_tyr_doublet_ratio`,
`bio_na_a_form_fraction`, `bio_phb_score`, and other ratio features on
SNV'd AUCs can fall outside [0, 1]. Treated as raw scores, not clipped or
log-transformed.

**Production classifier.** Best algorithm gets refit on **all 87 files**
(no holdout, using one consensus MI-feature set chosen by majority vote
across the 5 LOSO seeds' feature lists). This single fit is the
deployable artifact for the Streamlit UI.

### Predictions

| Quantity | Predicted | Rationale |
|---|---|---|
| MI-selected feature count per fold | 30–40 (target band) | R1 capacity bound: 259 ÷ 87 ≈ 3.0 features/file. |
| Best-algorithm LOSO mean parent-class recall | **0.50 – 0.62** | PLS-DA project record is 0.603 on band features alone; +93 new features (MCR/PCA/SAM/spatial/biology) should match or modestly exceed it. |
| Best-algorithm LOSO recall on held-out K-12 fold | **≥ 0.70** | Stage 15D found 3 K-12-specific features (the `bio_alpha_helix_score` axis). Lift expected. |
| `mcr_C6_mean` survives MI selection in ≥ 4/5 seeds | yes | Stage 15C made it the project's strongest single feature (d=−1.23). |
| R7 shuffled-vs-real `mcr_C1_*` importance ratio | < 50% (real wins) | Per Stage 15C, mcr_C1 had reasonable physical interpretability and weak file-level effect — if it's leak-driven we drop it. |
| Algorithm ranking | XGBoost ≈ LogReg > PLS-DA (small margin) | XGB handles non-linear MCR×biology interactions; PLS-DA still strong on linear band+ROI signal. |

### Branching verdicts

- **(A) Clear win.** Best LOSO mean parent-recall ≥ 0.55 AND K-12 LOSO
  recall stable ≥ 0.75 across all 5 seeds → unblock Stage 6
  (per-strain biology-pair modeling) reconsideration; ship.
- **(B) Modest improvement.** Best LOSO ≥ 0.50 (plan/15 §6 success bar)
  OR K-12 lift but main LOSO 0.45–0.50 → ship classifier with the gap
  documented in the paper; plan/15 §9 Q1 remains open.
- **(C) Plateau.** Best LOSO < 0.45 → feature track has plateaued. Plan/15 §9
  Q1 flips to "labeled-data problem"; future work pivots to plan/13 SSL track.
  Ship best-available model anyway for the demo.

### Stage-gate

If the experiment lands at (B) or better → Phase 2 (paper) and Phase 3
(Streamlit UI) of plan/16 proceed using the saved classifier. If (C), the
paper writeup leads with the negative finding and the UI ships best-available
to demonstrate the pipeline end-to-end even if accuracy is limited.

---

## Results

### Headline

**Branch (C) Plateau.** Stage 15F's consolidated 259-feature classifier
under per-fold MCR-ALS / ROI-PCA / SAM refit + MI selection (target k=35)
peaks at **LogReg-L2 LOSO mean accuracy = 0.436** — below both the
project's PLS-DA-on-raw-spectrum baseline (0.603) and the Branch (B)
threshold (0.50). PLS-DA on the same engineered features is markedly
worse (0.324) than PLS-DA on the raw 987-bin spectrum (0.603), indicating
**MI-selection on a small (87-file) corpus collapses PLS-DA's
low-rank-projection advantage**. XGBoost (0.247) is the weakest of the
three. Two strong per-strain wins: **ATCC25922 = 0.889 and
O121:H19 = 0.889** (LogReg ties the prior PLS-DA project record on
O121:H19); **O157:H7 = 0.556** and **Typhimurium = 0.778** are also
encouraging on otherwise-difficult strains. **K-12 collapses to 0.000
for all three algorithms** — Stage 15D's `bio_alpha_helix_score` axis
(K-12 ↔ other-STEC d=+0.537) did not survive the per-fold MI cut. **The
6.43-temperature PLS-DA on raw spectrum remains the project's headline
LOSO number**; Stage 15F's contribution is the deployable serialized
model + production-grade inference API, not a new accuracy record.

A surprise on multi-seed: **5 seeds gave identical results (std = 0.000
across PLS-DA / LogReg / XGB)**. The pipeline is deterministic given a
fixed train/test split — MCR-ALS SIMPLISMA picks the same purest
variables, sklearn PCA/PLS/LogReg use deterministic solvers, and XGBoost
`tree_method=hist` is deterministic. Re-ran with 1 seed at the same
result; **multi-seed scaffolding kept in place for future stochastic
variants but downgraded for re-runs**.

### Detailed results

#### 1. Algorithm comparison (best per-fold accuracy)

| Algorithm | Mean LOSO accuracy | Mean macro recall (4-class) | Best fold | Worst fold |
|---|---:|---:|---|---|
| **LogReg-L2 (best)** | **0.436** | 0.109 | ATCC25922, O121H19 = 0.889 | H2O, K-12 = 0.000 |
| PLS-DA              | 0.324 | 0.081 | O121H19 = 0.889 | Dublin, H2O, K-12 = 0.000 |
| XGBoost             | 0.247 | 0.062 | ATCC25922 = 0.667 | 83972, Dublin, H2O, O157H7 = 0.000 |

Macro recall ≈ accuracy / 4 because each LOSO fold has exactly one
parent class present in `y_true` (held-out strain → one parent class).
Accuracy = held-out-class recall is the correct headline metric.

#### 2. Per-strain accuracy (LogReg best algorithm)

| Strain | Parent class | LOSO accuracy | Prior best |
|---|---|---:|---|
| **ATCC25922** | Non-STEC   | **0.889** | Patch=5 / DANN λ=0.1 ~1.00 |
| **O121:H19**   | STEC       | **0.889** | PLS-DA / 2ch-CNN 0.89 (ties record) |
| Typhimurium  | Salmonella | 0.778 | PLS-DA 0.56 (Stage 15F lifts this) |
| O103:H2      | STEC       | 0.556 | PLS-DA / DANN λ=0.3 ~0.89 |
| O157:H7      | STEC       | 0.556 | DANN λ=0.3 5-seed soft-vote 0.78 |
| Heidelburg   | Salmonella | 0.333 | PLS-DA 0.7+ |
| 83972        | Non-STEC   | 0.25 | PLS-DA 1.00 |
| Dublin       | Salmonella | 0.111 | PLS-DA 0.7+ |
| **K-12**       | Non-STEC   | **0.000** | DANN λ=0.3 5-seed soft-vote 0.75 |
| **H2O**        | H2O        | **0.000** | structurally — model never sees H2O during this fold's training |

**Six folds clear ≥ 0.50; three collapse to ≤ 0.25; H2O is structurally
0.** Excluding H2O, the 9-strain mean is 0.484 (closer to the 0.50
Branch (B) threshold but still below it).

#### 3. MI-selected features (consensus, k = 35)

The MI selector's top-10 across LOSO folds:

```
roi_ch_stretch_std       fit_amide_iii_1242_area  roi_silent_kurt
fit_amide_iii_1242_height  d1_auc_lps_1117        d1_auc_lipid_1454
d1_auc_amide_i_1658      fit_lipid_1454_height    fit_amide_iii_1242_rmse
fit_lipid_1454_area
```

Striking: **0 MCR-ALS components in the top-10**. Only `mcr_C5_std`
(rank 35) survives consensus across folds. The headline `mcr_C6_mean`
(d = −1.23 globally) was **not selected per fold** — its global
discrimination is driven by the file-acquisition signature (substrate
component is C1; C6 is the bulk-biology composite, which correlates with
file-id metadata too). Per-fold MI on a TRAIN-only subset (78 files)
penalizes MCR features because per-fold MCR-ALS refit makes the
components reorder (no consistent C6 across folds).

The selected features are dominated by:
- **Pseudo-Voigt fit parameters** (Stage 15A) — height/area/RMSE at the
  amide-III, lipid_1454, and LPS anchor bands.
- **Derivatives** (Stage 15A) — d1/d2 AUCs at the LPS-chain and literature triple.
- **ROI moments** (Stage 15A) — std/kurt/skew of the C-H stretch +
  silent region (sanity) + LPS chain.

**One biology feature survives:** `bio_trp_indole_env` (Stage 15D),
`bio_cyt_ox_state`. **One SAM feature:** `sam_lps_sub_O121H19`. **One
ROI-PCA feature pair:** `pca_chstretch_PC2`, `pca_chstretch_PC3`.

#### 4. R7 permutation test — verdict: skipped

No `mcr_C1_*` features survived MI selection. R7 mitigation moot for
this run; reported as `skipped — no mcr_C1_* in selected feature set`.
Bug in the original run: the skipped path was missing `drop: []` key,
crashed `main()` after LOSO completed. Fixed (`r7_permutation_test`
skipped path now returns `{"verdict": ..., "ratios": {}, "drop": []}`)
and re-run at 1 seed completed cleanly.

#### 5. R9 — score-vs-fraction handling

Implemented per pre-reg: `bio_tyr_doublet_ratio`,
`bio_na_a_form_fraction`, and other SNV-AUC ratios treated as raw
scores (no clipping). `bio_trp_indole_env` survived MI selection
(rank 22); other ratios were not selected.

#### 6. Production model

LogReg-L2 refit on **all 87 files** using the consensus 35-feature
set, no holdout. Serialized to:

- `artifacts/stage15f_classifier.joblib` — `Pipeline([StandardScaler, LogisticRegression])`, 3.5 KB.
- `artifacts/stage15f_feature_columns.json` — 35-feature ordered list, 0.8 KB.
- `artifacts/stage15f_mcr_global.joblib` — fitted `MCRALSWrapper` (K=7) on all 7,122 spectra, 1.88 MB.
- `artifacts/stage15f_roi_pca.joblib` — fitted ROI-PCA dict, 15 KB.
- `artifacts/stage15f_sam_templates.joblib` — fitted SAM templates, 140 KB.
- `artifacts/stage15f_metadata.json` — LOSO summary + per-strain + R7, 2.1 KB.
- `artifacts/stage15f_loso_summary.csv` — per-fold raw rows.

Total disk footprint: **~2.0 MB** — well under the 100 MB cap for
Streamlit Community Cloud.

#### 7. Inference smoke test

End-to-end on `Atlas Data/STEC/O157H7/R412_100_10000ms_260311.xls`:

```
class: STEC
probabilities:
  STEC:        0.773
  Non-STEC:    0.135
  Salmonella:  0.074
  H2O:         0.018
spectrum_mean: ndarray[987]
wn:            ndarray[987]
```

**Predicted class = STEC at 77% confidence** — matches ground truth.
Pipeline runs end-to-end: parse → preprocess → feature_frame →
mcr.transform → roi_pca.transform → sam.transform → mean-pool to
file-level → MI-selected subset → loaded LogReg pipeline. Latency
< 5 s.

### Pre-registration verdicts

| Pre-reg | Predicted | Actual | Verdict |
|---|---|---|:-:|
| MI-selected feature count | 30–40 | 35 (target) | ✅ in range |
| Best LOSO mean recall | 0.50–0.62 | **0.436** (LogReg accuracy; macro recall 0.109) | ❌ below range — Branch (C) |
| K-12 LOSO recall | ≥ 0.70 | **0.000** | ❌ collapse — α-helix axis not selected per fold |
| `mcr_C6_mean` survives MI in ≥ 4/5 seeds | yes | only `mcr_C5_std` survives (rank 35); C6 dropped | ❌ |
| R7 `mcr_C1_*` shuffled/real importance | < 50% | skipped (no mcr_C1 in selected set) | ⚠️ moot |
| Algorithm ranking | XGB ≈ LogReg > PLS-DA | LogReg > PLS-DA > XGB | ⚠️ XGB last not first |
| Multi-seed LOSO variance | nontrivial | std = 0.000 (deterministic pipeline) | ⚠️ informative — multi-seed adds no signal |

### Implications

1. **Branch (C) verdict triggers.** Engineered features have plateaued
   below the PLS-DA-on-raw-spectrum baseline. Plan/15 §9 Q1 ("can
   engineering close the LOSO gap?") **flips to "labeled-data
   problem"**; future work pivots to plan/13 SSL pretraining + cross-
   corpus transfer, not more feature engineering.

2. **Per-fold MI selection is hostile to MCR features.** MCR-ALS is
   refit per fold (R2 mitigation), so components reorder across folds
   — there is no stable "C6 = bulk biology" across the 10 folds. MI
   on TRAIN-only (78 files) picks per-fold-stable features (band AUCs,
   peak-fit parameters, derivatives), not per-fold-reordering ones.
   **The global MCR finding (`mcr_C6_mean` d=−1.23) was a global-fit
   artifact** that didn't survive R2 — exactly what the per-fold refit
   was designed to catch. Counter-intuitive: MCR features rank highly
   on a single global fit but are unreliable LOSO discriminators.

3. **K-12 = 0.000 falsifies the Stage 15D K-12 lift hypothesis.** The
   three biology features that flagged K-12 differently from clinical
   STEC (`bio_beta_sheet_amide3`, `bio_trp_indole_env`,
   `bio_alpha_helix_score`) did not survive per-fold MI. K-12 remains
   the "minority-of-one" strain that only DANN λ=0.3 5-seed soft-vote
   has cracked (0.75). **Stage 6 (3-channel CNN) reconsideration is
   off the table** per Stage 15D's conditional gate.

4. **Two LOSO wins worth flagging:**
   - **Typhimurium = 0.778** (LogReg) lifts from PLS-DA's 0.56 — Stage
     15F is the first model in the project to push Typhimurium past
     0.7 since PLS-DA was the only model > 0 on Typhimurium.
   - **ATCC25922 = 0.889** ties prior Patch=5 / DANN best.

5. **PLS-DA-on-engineered (0.324) ≪ PLS-DA-on-raw (0.603).** The
   low-rank projection that helps PLS-DA on the full 987-bin spectrum
   does **not** help on 35 MI-selected features. PLS-DA needs the full
   feature space to build its discriminative latent variables;
   pre-selection collapses that. **Don't pre-select features for PLS-DA.**

6. **The production model is LogReg, not PLS-DA.** Project headline
   should still cite PLS-DA-on-raw = 0.603 as the strongest LOSO
   number, but the **deployable artifact** is LogReg-L2 on 35
   MI-selected features — chosen because it generalizes per-fold on the
   engineered cache that the inference API also uses. The UI ships
   LogReg.

7. **Multi-seed scaffolding stays for future stochastic variants.**
   The Stage 15F pipeline is fully deterministic (MCR-ALS SIMPLISMA,
   sklearn PCA/PLS/LogReg, XGB hist) so multi-seed adds no signal.
   For any future variant adding dropout, bootstrap aggregation, or
   randomized projection, restore SEEDS = [0, 1, 2, 3, 4].

8. **Plan/16 Phases 2 + 3 unblocked.** Paper §6 backfilled from
   `artifacts/stage15f_metadata.json`; Streamlit UI loads the
   `artifacts/` directory; production deploy proceeds with the
   serialized LogReg pipeline.

---

## Artifacts

- `scripts/run_stage15f_final.py` — training script.
- `atlas/inference.py` — production prediction API.
- `artifacts/stage15f_classifier.joblib` — fitted sklearn pipeline (best algo).
- `artifacts/stage15f_feature_columns.json` — MI-selected feature names (ordered).
- `artifacts/stage15f_mcr_global.joblib` — `MCRALSWrapper` fitted on all pixels.
- `artifacts/stage15f_roi_pca.joblib` — ROI-PCA fitted on all data.
- `artifacts/stage15f_sam_templates.joblib` — SAM templates fitted on all data.
- `artifacts/stage15f_metadata.json` — LOSO summary, per-strain recall, training date, model type.
