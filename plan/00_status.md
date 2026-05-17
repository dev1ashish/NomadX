# 00 — Status

> **Mutability:** mutable. Rewrite freely.
> **Last updated:** 2026-05-17.

## Phase

| Phase | State | Notes |
|---|---|---|
| 0. Discovery | done | Raman hyperspectral maps, tab-text `.xls` files. |
| 1. Subagent research (5 in parallel) | done | data-engineer, data-scientist, ml-engineer, ai-engineer, mlops-engineer all returned. |
| 2. Synthesis into master plan | done | Now split across `plan/`. |
| 3. User approval of plan | done | Open decisions resolved — see `02_decisions.md`. |
| 4. Implementation | in progress | Steps 1–3 done (parser, EDA, preprocess+QC, classical models, CNN, ensemble, Transformer, **DANN ablation**). README left. |

## What's done

- `atlas/io.py` — parses tab-delimited .xls/.txt, strips comma thousands, derives grid from coord uniqueness, interpolates to canonical wn axis.
- `scripts/build_dataset.py` — discovers files, runs the parser, caches to `data_cache/`. Idempotent via (mtime, sha256).
- Raw cache: 87/87 files parse with 0 fatal errors. 7,999 spectra @ 2048 bins.
- `notebooks/atlas_driver.ipynb` — EDA blocks 1–9 (1–8 on raw+SNV, 9 on full preprocessing).
- `outputs/eda/` and `outputs/eda_v2/` — 13 + 7 plot/csv artifacts.
- `atlas/preprocess.py` — cosmic-ray removal → arPLS baseline → Sav-Gol smoothing → crop → SNV → optional 2nd derivative.
- `atlas/qc.py` — SNR ≥ 5 + per-file background pixel detection.
- `scripts/preprocess_dataset.py` — caches preprocessed array. 7,122/7,999 spectra retained after QC.
- `atlas/splits.py` — StratifiedGroupKFold(5) Protocol A + LOSO Protocol B. Cached at `data_cache/splits/protocol_{a,b}.json`.
- `atlas/models_classical.py` + `atlas/evaluate.py` — 6 models (LogReg, LinSVM, RBF-SVM, RF, XGB, PLS-DA) under both protocols.
- `atlas/memprobe.py` — v1 (from-scratch tiny CNN → file_id, 4.1% top-1).
- `atlas/models_cnn.py` + `atlas/train.py` + `scripts/run_cnn.py` — small 1D-CNN under both protocols. 124K params after the channel-doubling fix (`16-32-48-64 → 32-64-96-128`) and per-bin StandardScaler at input — see [10_decision_log.md §cnn-fixes](10_decision_log.md#2026-05-14--cnn-small-variant-architectural-fixes-mid-session).
- `atlas/memprobe_v2.py` — penultimate-features → 87-way LogReg. Fires at 15.5% top-1 / 37.0% top-5 (vs 1.15% chance) — above the 10% DANN threshold.
- `atlas/ensemble.py` + `scripts/run_ensemble.py` — soft-vote ensembles over pre-computed per-fold parquets. Three ensembles run (PLS-DA+XGB, PLS-DA+CNN, PLS-DA+XGB+CNN) under both protocols. **None beats PLS-DA solo on LOSO; CNN's K-12 and O157H7 wins are destroyed by averaging** — see [07§ensemble-fails-to-clear-plsda](07_findings.md#2026-05-14--ensemble-fails-to-clear-plsda).
- `atlas/models_transformer.py` + `scripts/run_transformer.py` — small 1D-Transformer (~217K params, patch_size=20). tqdm-instrumented training loop (`atlas/train.py`). **Weakest single-model arm: Protocol A file-F1 0.507, LOSO mean parent-recall 0.193, K-12 / O157H7 both collapse to 0.00.** Per-strain finding: 20-bin patches blur the narrow-peak signal the CNN's k=5-15 kernels caught — see [07§transformer-underperforms-cnn](07_findings.md#2026-05-14--transformer-underperforms-cnn).
- `atlas/models_cnn.py` (DANNCNN1D + GradReverse) + `atlas/train.py` (train_dann_fold + DANNConfig) + `scripts/run_dann.py` — CNN with Gradient Reversal Layer feeding an 87-way (per-fold K-way) file_id domain head. Lambda warms 0→λ_max over 10 epochs then holds. Two settings swept: **λ=0.1 ships as the headline; λ=0.3 documented as the high-pressure regime.** See [07§dann-ablation-clears-verdict-a](07_findings.md#2026-05-14--dann-ablation-clears-verdict-a) and [07§dann-lambda-frontier](07_findings.md#2026-05-14--dann-lambda-frontier).
  - λ=0.1: K-12 0.50→0.75, O157H7 0.56→0.56, ATCC25922 0.11→0.89, O121H19 0.00→0.67. **LOSO mean 0.35→0.500 (+0.15)**. Protocol A 0.649→0.566. Verdict (A) hit.
  - λ=0.3: K-12 0.75→0.88, O157H7 0.56→0.67, O103H2 0.33→0.89 (all stronger biology), but 83972 0.75→0.25 and ATCC25922 0.89→0.11 crater. LOSO mean 0.500→0.447. Protocol A 0.566→0.493.
  - **Memprobe v2 on DANN encoder essentially unchanged at both λ:** vanilla 15.5% → λ=0.1 14.0% → λ=0.3 13.6%. **DANN reshapes feature prominence, not linear file-id separability** — the 10% memprobe threshold from [02§decisions](02_decisions.md) is rejected for this dataset.

## What's next (in order)

1. ✅ **Re-ensemble with the architecturally-diverse 4-model base — DONE 2026-05-15.** All three combination schemes (soft-vote, stacking meta-learner, confidence-router) fail to Pareto-dominate PLS-DA solo on LOSO. **4th negative result on ensembling — verdict (Z) by intent.** See [07§2026-05-15--4-architecture-re-ensemble-fails-to-beat-plsda](07_findings.md#2026-05-15--4-architecture-re-ensemble-fails-to-beat-plsda).
2. ✅ **Temperature-scaled soft-vote — DONE 2026-05-15.** Calibration mismatch confirmed (PLS-DA T=6.43 vs deep models T=1.2–1.7) and partial mechanism fix (ATCC25922 0.33→0.67, O157H7 0.11→0.67) but mean 0.566 still trails PLS-DA solo 0.603. **Branch (B) — partial confirmation.** Sharp finding: K-12 + Typhimurium are *minority-of-one* cells that no calibration scheme can fix. Margin-based router also degenerates to PLS-DA. **Ensemble chapter closed.** See [07§2026-05-15--temperature-scaled-softvote](07_findings.md#2026-05-15--temperature-scaled-softvote).
3. ⏳ Remaining diagnostic + polish experiments (no expected LOSO lift, but valuable for the writeup): variance-aware memprobe (PCA-3 probe), TTA on best DANN, arPLS crop fix, SHAP/saliency on the best DANN and 2-channel CNN.
4. ⏳ **Band-chemistry track ([plan/14](14_band_chemistry_research.md)) — opened 2026-05-17.** Annotated preprocessed-spectra figure landing this session; `atlas/band_features.py` + `notebooks/band_chemistry.ipynb` (bacteria-only ANOVA → primary-triple 1338/1454/1658 → macromolecule radar → band ratios → Lorentzian fits → engineered-feature classifier → 3-channel CNN) over the next 3–4 working days. Pre-committed success bar is LOSO ≥ 0.55 (interpretability parity), not absolute lift over PLS-DA solo.
5. ⏳ Final plots, README narrative update with all the new findings, `make verify`, CI green.

## Open items / TODOs

- Tweak arPLS crop start to ~450 cm⁻¹ (currently 400) — boundary artifact at left edge of preprocessed spectra. See [findings.md §arpls-boundary](07_findings.md#arpls-boundary-artifact).
- Re-run Block 9 after crop tweak.
- ~~Run a bacteria-only ANOVA (3 classes, excluding H₂O)~~ — **moved to [plan/14 §6.2](14_band_chemistry_research.md) as the bacteria-only-ANOVA step of the new band-chemistry track.** The wider track also covers macromolecule biochemistry vectors, band ratios, Lorentzian peak fits, and engineered-feature classifiers.
- ~~**Tune CNN augmentation regime.**~~ **CLOSED 2026-05-15.** Swept DANN λ=0.1 across `light` (all p halved) and `no_mixup` presets. Both regress vs default: light 0.347 (-0.153), no_mixup 0.423 (-0.077). **Heavy aug is load-bearing for LOSO cross-strain regularization, not over-tuning** — the original "train_acc 0.4-0.5 = undertraining" diagnosis was using a Protocol-A proxy for a LOSO problem. Surprise finding kept: no_mixup gives K-12 = 0.875 (joint-best K-12 across all DANN variants, ties DANN λ=0.3). See [07§2026-05-15--dann-aug-regime-sweep](07_findings.md#2026-05-15--dann-aug-regime-sweep).

## CNN session summary (2026-05-14)

| Headline metric | Pre-registered range | Actual | Verdict |
|---|---|---|---|
| Protocol A file-macro-F1 (mean over 5 folds) | 0.92 – 0.98 | **0.649 ± 0.079** | ❌ below floor by 0.27 |
| LOSO mean parent-class recall (9 strains) | 0.55 – 0.72 | **0.35** | ❌ below floor by 0.20 |
| K-12 parent-class recall (Non-STEC) | 0.00 – 0.15 (biological ceiling) | **0.50** | ⭐ above ceiling by 0.35 |
| O157H7 parent-class recall (STEC) | 0.15 – 0.50 | **0.56** | ⭐ above ceiling by 0.06 |
| memprobe v2 top-1 (87-way file_id from penultimate) | 2 – 12% | **15.5%** | ❌ probe fires |

The CNN doesn't beat classical models on the headline metric on either protocol, but it cracks the two biologically-hardest folds (K-12, O157H7) that no classical model could touch. The new headline is **"different inductive biases fail on different strains"**, generalizing the linear-vs-tree complementary-failure observation from the classical session. The right next step is an **ensemble** (PLS-DA + XGB + CNN), not a bigger CNN.

## Transformer session summary (2026-05-14)

| Headline metric | Pre-registered range | Actual | Verdict |
|---|---|---|---|
| Protocol A file-macro-F1 (mean over 5 folds) | 0.60 – 0.75 | **0.507 ± 0.122** | ❌ below floor by 0.09 |
| LOSO mean parent-class recall (9 strains) | 0.30 – 0.50 | **0.193** | ❌ below floor by 0.11 — **weakest arm overall** |
| K-12 parent-class recall (Non-STEC) | 0.00 – 0.30 | **0.00** | ✅ in range but at floor — patches blur the signal CNN found |
| O157H7 parent-class recall (STEC) | 0.00 – 0.40 | **0.00** | ✅ in range but at floor — same mechanism |
| One bright spot | n/a | O121H19 = 0.22 (vs CNN 0.00) | small partial recovery on one easy STEC strain |

**Diagnostic finding: 20-bin patches blur the narrow-peak signal.** Sanity-check no-aug fold 0 already showed val_f1 ceiling 0.50 — at the InstanceNorm-only CNN level despite the Transformer having both per-bin standardize and more raw params (217K vs 124K CNN). The bottleneck isn't optimization; it's that strided Conv1d(k=20, s=20) averages each 5-10 bin wide Raman peak with its neighborhood before any attention pass sees it. Decision: Transformer documented as benchmark completeness arm only; future work entry added for patch_size=5 or overlapping patches. Detail at [07§transformer-underperforms-cnn](07_findings.md#2026-05-14--transformer-underperforms-cnn).

## DANN session summary (2026-05-14)

| Headline metric | Pre-registered range | Actual | Verdict |
|---|---|---|---|
| Protocol A file-macro-F1 (mean over 5 folds) | 0.55 – 0.70 | **0.566 ± 0.091** | ✅ in range, lower half |
| LOSO mean parent-recall (9 strains) | 0.30 – 0.55 | **0.500** | ✅ upper half; +0.15 over vanilla CNN; 0.10 below PLS-DA |
| K-12 parent-class recall (Non-STEC) | 0.00 – 0.50 (bet lower half) | **0.75** | ⭐⭐ above ceiling by 0.25 — biology win improved on |
| O157H7 parent-class recall (STEC) | 0.00 – 0.50 (bet lower half) | **0.56** | ⭐ above ceiling — biology win preserved exactly |
| ATCC25922 parent-class recall (Non-STEC) | 0.00 – 0.40 | **0.89** | ⭐⭐⭐ above ceiling by 0.49 — new best across ALL models |
| O121H19 parent-class recall (STEC) | 0.00 – 0.50 | **0.67** | ⭐⭐ above ceiling by 0.17 — major recovery |
| memprobe v2 top-1 (87-way file_id from penultimate) | 1.5 – 10% | **14.0%** | ❌ above ceiling by 4pp — probe still fires |

**Verdict branch (A) hit cleanly.** DANN preserves both biology-hard cells (K-12 0.75, O157H7 0.56) AND lifts mean parent-recall by 0.15. The pre-registered K-12 bet ("lower half") was wrong: load-bearing features for K-12 were not the file-id-correlated ones — DANN stripped acquisition noise and the genuine peak-ratio signal got *clearer*, lifting K-12 from 0.50 to 0.75 instead of destroying it. **The memprobe puzzle was resolved by a follow-up λ=0.3 sweep:** tripling λ moved the probe by 0.4 pp (14.0% → 13.6%) while changing the LOSO per-strain pattern significantly. **DANN reshapes feature prominence, not linear file-id separability — the 10% memprobe threshold is not a reliable DANN diagnostic on this dataset.** Detail at [07§dann-ablation-clears-verdict-a](07_findings.md#2026-05-14--dann-ablation-clears-verdict-a) and [07§dann-lambda-frontier](07_findings.md#2026-05-14--dann-lambda-frontier).

## DANN λ frontier follow-up (2026-05-14)

| Headline metric | λ=0.1 (ships) | λ=0.3 | Notes |
|---|---|---|---|
| LOSO mean parent-recall | **0.500** | 0.447 | λ=0.1 wins on mean |
| K-12 (Non-STEC) | 0.75 | **0.88** | monotonic in λ |
| O157H7 (STEC) | 0.56 | **0.67** | monotonic in λ |
| ATCC25922 (Non-STEC) | **0.89** | 0.11 | λ=0.1 owns; λ=0.3 craters |
| 83972 (Non-STEC) | 0.75 | 0.25 | λ=0.3 craters easy commensal |
| O103H2 (STEC) | 0.33 | **0.89** | λ=0.3 unlocks this strain |
| Protocol A file-F1 | **0.566** | 0.493 | strict cost in λ |
| Memprobe v2 top-1 | 14.0% | 13.6% | probe is **decoupled from LOSO** |

**Operational decision: ship λ=0.1.** λ=0.3 is a useful regime finding — strengthens pathogen biology features (K-12, O157H7, O103H2) at the explicit cost of easy commensal recognition (83972, ATCC25922). λ=0.1 has higher mean and no crater cells.

**λ=0.05 added 2026-05-14** to complete the lambda curve: mean parent-recall 0.321 (BELOW vanilla CNN 0.35) with K-12 collapse 0.50→0.12. Useful negative finding — there's a minimum effective DANN pressure around λ≈0.07-0.10 on this dataset; below that the GRL just adds noise. See [07§dann-lambda-curve-completed](07_findings.md#2026-05-14--dann-lambda-curve-completed).

**Grouped-domain DANN (subclass, λ=0.1) added 2026-05-14:** LOSO mean 0.309 (rejected the hypothesis that coarser domain target preserves more), Protocol A 0.654 (BEST Protocol A across all DANN variants — Pareto-optimal for within-distribution use). See [07§grouped-domain-dann-rejects-hypothesis](07_findings.md#2026-05-14--grouped-domain-dann-rejects-hypothesis).

**Stacking meta-learner tried + failed 2026-05-14:** 3 variants of LogReg over {PLS-DA, DANN(0.05), DANN(0.1), DANN(0.3)} all underperform the best base model. LOSO stacking is fundamentally limited because DANN's per-strain behavior is heterogeneous — meta-learner can't extrapolate from training strains where DANN is bad to held-out strains where DANN is good. Useful negative finding. See [07§stacking-meta-learner-fails](07_findings.md#2026-05-14--stacking-meta-learner-fails).

**Per-strain λ selector tried + failed 2026-05-14:** 3 leakage-bounded variants (hard / soft / router) over {DANN(0.05), DANN(0.1), DANN(0.3)} all underperform the best single base model. Mean 0.435-0.444 vs DANN λ=0.1 solo's 0.500. **Inner-val F1 and test-time confidence both fail to predict which λ is best on a held-out strain** — the information needed is by definition held out. Third negative result on multi-λ combination. See [07§per-strain-lambda-selection-fails](07_findings.md#2026-05-14--per-strain-lambda-selection-fails).

**Patch=5 Transformer added 2026-05-14:** Re-ran the failed patch=20 Transformer with patch_size=5 to test the patch-blur hypothesis. **Partial confirmation. Two new per-strain SOTA records:** ATCC25922 = 1.00 (9/9 correct; first 100% on this strain in the entire sweep), O157H7 = 0.78 (the canonical pathogenic STEC; highest in sweep). LOSO mean 0.349 (much better than patch=20's 0.193 but below DANN λ=0.1's 0.500). K-12 stays at 0.00 — patch=5 doesn't fix K-12 because K-12 uses broader-scale chemistry not narrow-peak structure. **Different architectures crack different biology cells; the writeup story foregrounds this per-strain mechanism split.** See [07§patch5-transformer-partially-confirms-blur-hypothesis](07_findings.md#2026-05-14--patch5-transformer-partially-confirms-blur-hypothesis).

**2-channel CNN (SNV + 2nd-derivative) added 2026-05-14:** Fixed (1, -2, 1) Laplacian as a second input channel. **Second-best single-model LOSO mean of the entire sweep at 0.465** (after DANN λ=0.1's 0.500; ahead of vanilla CNN 0.35, patch=5 0.349, DANN λ=0.3 0.447). **Two new per-strain records:** O121H19 = 0.89 (first deep model to tie PLS-DA on this STEC strain), O157H7 = 0.78 (ties patch=5 Transformer). K-12 collapses to 0.00 — confirms K-12 uses broad-scale not narrow-peak chemistry. Protocol A regresses to 0.560 ± 0.150 (folds 3+4 early-stopped at ep13/18) — documented as the cost of the channel addition. **The per-strain best-model table now has 3 distinct deep architectures owning different biology cells.** See [07§2nd-derivative-channel-second-best-loso](07_findings.md#2026-05-14--2nd-derivative-channel-second-best-loso).

## Ensemble session summary (2026-05-14)

| Headline metric | Pre-registered range | Actual | Verdict |
|---|---|---|---|
| Best ensemble LOSO mean parent-recall (b: PLS-DA + CNN) | 0.55 – 0.72 | **0.579** | ✅ in range; **fails to beat PLS-DA solo 0.60** |
| Best ensemble Protocol A file-F1 (b: PLS-DA + CNN) | 0.85 – 0.94 | **0.919 ± 0.030** | ✅ in range; below PLS-DA solo 0.951 |
| K-12 recall in best ensemble | 0.13 – 0.50 | **0.00** | ❌ below floor — CNN's win destroyed |
| O157H7 recall in best ensemble | 0.14 – 0.56 | **0.00** | ❌ below floor — CNN's win destroyed |

**The hypothesized ensemble win does not materialize.** Soft-vote averaging crushes minority-vote signal: even when CNN is correct on K-12 and O157H7 at the per-spectrum level, its proba mass on the right class is the file-level runner-up, not the file-level argmax. Averaging with two confident classical models that are confidently wrong on those folds pulls the file-level decision back to the wrong class. The verdict structure pre-locked branch "all three ensembles ≤ 0.60 mean → ship PLS-DA solo" is the one we hit. CNN's K-12 / O157H7 wins remain real but only retrievable via per-strain model selection, not soft-vote ensemble. Detail at [07§ensemble-fails-to-clear-plsda](07_findings.md#2026-05-14--ensemble-fails-to-clear-plsda).

## Multi-seed DANN λ=0.3 verification session summary (2026-05-15) — K-12 win survives

5 seeds × DANN λ=0.3. Tests whether the K-12 = 0.875 claim from [§dann-lambda-frontier](07_findings.md#2026-05-14--dann-lambda-frontier) is robust or another lucky-seed artifact like λ=0.1's 0.500.

| Stat | DANN λ=0.3 |
|---|---|
| Single-seed shipped (seed=42) | 0.447 |
| 5-seed mean ± SD | 0.393 ± 0.117 |
| **5-seed soft-vote** | **0.448** ⭐ |

**The single-seed 0.447 headline is robust** — 5-seed soft-vote essentially reproduces it (0.448). Unlike DANN λ=0.1 where 0.500 → 0.370 under multi-seed averaging.

**K-12 verification: survives.** 5-seed soft-vote K-12 = **0.75**; per-seed K-12 mean 0.55 ± 0.41 with 3 of 5 seeds at ≥0.75. The narrative drops from "0.875 always" to "0.75 typically (3 of 5 random inits)" but the cell is genuinely solved by DANN λ=0.3.

**O157H7 also verified robust** for DANN λ=0.3: 5-seed soft-vote 0.78, mean 0.60 ± 0.24.

**Surprise: λ=0.3 dominates λ=0.1 once properly characterized.**

| | DANN λ=0.1 5-seed soft-vote | DANN λ=0.3 5-seed soft-vote |
|---|---|---|
| LOSO mean | 0.370 | **0.448** ⭐ |
| K-12 | 0.38 | **0.75** ⭐ |
| O157H7 | 0.44 | **0.78** ⭐ |
| ATCC25922 | **1.00** ⭐ | 0.89 |

The original "λ=0.1 ships as headline because higher mean" decision was based on single-seed numbers — it was wrong. **λ=0.3 is the actually-better DANN config under multi-seed characterization.**

**Methodological finding kept for writeup:** 5-seed soft-vote (averaging per-spectrum probas across seeds, then aggregating file-level) is the right way to report deep-model LOSO. It captures consensus at the probability level and preserves minority-correct cells (e.g. K-12 = 0.75 survives even when 2 of 5 seeds fail).

**Operational decisions:**
- **DANN headline → λ=0.3 5-seed soft-vote: 0.448 (K-12 = 0.75, O157H7 = 0.78, ATCC25922 = 0.89).**
- DANN λ=0.1 5-seed soft-vote 0.370 documented as the λ-frontier low-pressure endpoint (ATCC25922 = 1.00 as its robust biology win).
- PLS-DA solo 0.603 remains the headline. Gap to DANN now 0.16.
- Per-strain table updated with verified-robust 5-seed numbers.

Detail at [07§2026-05-15--dann-lam03-5seed-verification](07_findings.md#2026-05-15--dann-lam03-5seed-verification).

## Multi-seed DANN λ=0.1 robustness session summary (2026-05-15) — **major revision to prior headlines**

5 seeds of DANN λ=0.1 + default aug + LOSO. Bug fix: `--seed` was previously cosmetic (fold_seed read from splits JSON directly); fixed with `fold_seed = orig + (seed - 42)` so each seed runs different training trajectories.

**Headline revision: DANN λ=0.1 LOSO is 0.345 ± 0.145, NOT 0.500.**

| Seed | LOSO mean | Notes |
|---|---|---|
| 42 (originally shipped) | 0.500 | 2nd-best of 5 — lucky-middle seed |
| 1 | 0.153 | Worst of 5 — bad init |
| 2 | **0.528** ⭐ | Best of 5 — also gave 83972 = 1.00, ATCC25922 = 1.00, Typhimurium = 0.56 (first deep Salmonella recovery) |
| 3 | 0.273 | |
| 4 | 0.270 | |
| **5-seed mean ± SD** | **0.345 ± 0.145** | **The honest DANN λ=0.1 LOSO value** |
| 5-seed soft-vote | 0.370 | The robust single-number ship |
| Oracle (max per strain) | 0.676 | Test-leakage; informational ceiling |

**Per-strain seed-stability map (5-seed analysis):**

| Strain | Mean | SD | Verdict |
|---|---|---|---|
| **ATCC25922** | **0.844** | 0.206 | ⭐ ROBUST DANN win (soft-vote = 1.00) |
| Dublin | 0.000 | 0.000 | ROBUST failure |
| 83972 | 0.600 | 0.348 | fragile — only seed=2/4 hit ≥0.875 |
| O103H2 | 0.444 | 0.330 | fragile |
| **K-12** | 0.325 | **0.350** | **HIGHLY fragile — claimed 0.75 only on 2 of 5 seeds** |
| O157H7 | 0.311 | 0.257 | fragile |
| O121H19 | 0.244 | 0.301 | fragile |
| Typhimurium | 0.156 | 0.206 | fragile |
| Heidelburg | 0.178 | 0.151 | mostly low |

**The K-12 narrative requires major revision.** Plan/07's "DANN cracks K-12 at 0.75 via broad-scale adversarial denoising" finding ([§dann-ablation-clears-verdict-a](07_findings.md#2026-05-14--dann-ablation-clears-verdict-a)) is now characterized as a **lucky-seed artifact** — only 2 of 5 random inits land in that basin. Mean K-12 across seeds is 0.325, soft-vote is 0.380.

**Implications for the per-strain best-model table:**

All previously-claimed deep-model per-strain wins (DANN λ=0.3 K-12 = 0.875; Patch=5 ATCC25922 = 1.00; 2-channel CNN O121H19 = 0.89, O157H7 = 0.78) are **suspect single-seed estimates** and should be verified with multi-seed runs. **Only ATCC25922 = 1.00 from DANN is verified robust across seeds.**

**Operational decision:**
- **PLS-DA solo 0.603 remains LOSO headline.** Gap to DANN is now 0.23+, not 0.10.
- **DANN λ=0.1 reported as 0.345 ± 0.145 (5 seeds), or 0.370 (5-seed soft-vote).**
- **Verify other deep-model headlines with multi-seed** before shipping. Most-suspect: DANN λ=0.3 K-12 = 0.875, Patch=5 ATCC25922 = 1.00. ~3 hours total to multi-seed all deep variants.

Detail at [07§2026-05-15--dann-5seed-robustness](07_findings.md#2026-05-15--dann-5seed-robustness).

## Aug-sweep session summary (2026-05-15)

Tested plan/00's "aug is over-tuned" TODO by sweeping DANN λ=0.1 LOSO across 3 lighter aug presets. **Result: aug TODO closed as incorrect.** Heavy aug is doing important cross-strain regularization for LOSO; lighter aug overfits training strains. Detail at [07§2026-05-15--dann-aug-regime-sweep](07_findings.md#2026-05-15--dann-aug-regime-sweep).

| Variant | LOSO mean | Δ vs default | K-12 | Notes |
|---|---|---|---|---|
| default (baseline) | **0.500** | — | 0.750 | Current shipped DANN λ=0.1 |
| light (all p halved) | 0.347 | −0.153 | 0.375 | Direct regression |
| no_mixup | 0.423 | −0.077 | **0.875** ⭐ | Mean drops but **K-12 = joint-best across all DANN variants** (ties DANN λ=0.3 0.875). Mixup specifically suppresses K-12's broad-scale signal. |
| minimal | skipped | — | — | Predicted strictly worse than light. |

**Operational decision:** keep default aug. Pivot to multi-seed averaging as next swing — different mechanism (per-strain variance reduction, not regularization tuning). Plausible lift +0.03-0.07 LOSO if K-12 / Typhimurium / Dublin per-seed variance is what's holding DANN λ=0.1 back at 0.500 single-seed.

## Temperature-scaled soft-vote session summary (2026-05-15)

Tested the calibration-mismatch hypothesis from the 4-architecture re-ensemble post-mortem by per-base temperature scaling before averaging. Implementation: `atlas/calibrated_ensemble.py` + `scripts/run_calibrated_ensemble.py`. Pre-registration and post-run resolution: [plan/08 §2026-05-15-temperature-scaled-soft-vote](08_expectations.md).

| Variant | Pre-registered range | Actual mean | K-12 | O157H7 | Verdict |
|---|---|---|---|---|---|
| T-scaled 4-base (all included) | 0.50 – 0.68 (central 0.58) | **0.566** | 0.000 | 0.667 | ✅ in range; **branch (B)** — K-12 gate fails |
| T-scaled 3-deep (PLS-DA excluded sanity) | 0.40 – 0.60 (central 0.50) | **0.492** | **0.375** ⭐ | 0.667 | ✅ in range; **first ensemble to break K-12 above 0.00** |
| Margin-based router (4-base, calibration-invariant) | not pre-registered | **0.603** | 0.000 | 0.000 | also degenerates to PLS-DA solo (78/78 files) |

**Fitted temperatures (LOO over 9 folds):** PLS-DA = 6.43, DANN = 1.23, Patch5 = 1.70, 2-ch CNN = 1.63. PLS-DA needed ~5× more softening — quantitatively confirms the calibration mismatch.

**Sharp mechanism finding.** Temperature scaling fixes the cells where 3-of-4 bases vote correctly and PLS-DA confidently-wrong outvotes them (ATCC25922 0.33 → 0.67; O157H7 0.11 → 0.67) but cannot fix **minority-of-one cells** — K-12 (only DANN right) and Typhimurium (only PLS-DA right) — because no calibration scheme can amplify a 1-of-4 vote past a 3-of-4 wrong consensus. Margin router still degenerates to PLS-DA because PLS-DA's classifier produces systematically more peaked distributions, not just higher max-proba.

**Operational decision: ship PLS-DA solo (0.603) as the LOSO headline.** Temperature-scaled soft-vote is the second-best ensemble result and the cleanest mechanism demonstration — include in writeup as the "what would have to be true for ensembles to work" narrative. The ensemble chapter is now fully closed: **5 distinct combination schemes × 3 base sets, no Pareto improvement over PLS-DA solo.** Detail at [07§2026-05-15--temperature-scaled-softvote](07_findings.md#2026-05-15--temperature-scaled-softvote).

## 4-architecture re-ensemble session summary (2026-05-15)

Re-ran soft-vote + stacking + confidence-router over the architecturally-diverse 4-model base {PLS-DA, DANN λ=0.1, Patch=5 Transformer, 2-channel CNN}. Pre-registration: [plan/08 §2026-05-15-re-ensemble-with-4-architecturally-diverse-bases](08_expectations.md).

| Variant | Pre-registered range | Actual mean | K-12 | O157H7 | Verdict |
|---|---|---|---|---|---|
| Soft-vote uniform | 0.45 – 0.60 (central 0.52) | **0.579** | 0.000 | 0.111 | ✅ in range upper half; **fails X's biology gates (K-12=0.00, O157H7=0.11)** |
| Stacking (LogReg meta over 16-D file-level features) | 0.40 – 0.55 (central 0.46) | **0.432** | 0.000 | 0.000 | ✅ in range lower half; cleanly in (Z) |
| Confidence-router (file-level argmax mean max-proba) | 0.42 – 0.58 (central 0.48) | **0.603** | 0.000 | 0.000 | ⚠️ above ceiling; **degenerates to "always pick PLS-DA" on 78/78 files** |

**Verdict (Z) hit by intent — 4th negative result on ensembling.** Soft-vote and router each have mean ≥ 0.55 but fail both biology gates because PLS-DA's max-proba is calibrated systematically higher than any deep model's. Soft-vote inherits PLS-DA's predictions on the cells PLS-DA owns (Heidelburg, O103H2, O121H19, Typhimurium — all *exceed* predicted upper bounds) and lets PLS-DA's confidently-wrong vote drown 3-of-4 deep majorities on ATCC25922, K-12, O157H7. Router degenerates to PLS-DA solo by tautology. Stacking can't extrapolate base-pattern heterogeneity across LOSO folds (same problem as the prior 3-DANN stack).

**Operational decision: ship PLS-DA solo at 0.603 as the LOSO headline.** Per-strain best-model table remains the central writeup story: DANN owns K-12, Patch=5 owns ATCC25922, 2-channel CNN ties for O121H19 + O157H7, PLS-DA owns the Salmonella triplet + 83972 + Heidelburg. No single ensemble captures all of them — the writeup story stays *"complementary per-strain wins across different inductive biases, no soft-vote / meta-learner / router on this dataset captures all of them."* Detail at [07§2026-05-15--4-architecture-re-ensemble-fails-to-beat-plsda](07_findings.md#2026-05-15--4-architecture-re-ensemble-fails-to-beat-plsda).
