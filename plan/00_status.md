# 00 — Status

> **Mutability:** mutable. Rewrite freely.
> **Last updated:** 2026-05-14.

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

1. ⏳ Final plots, README narrative, `make verify`, CI green. **New headline: CNN+DANN λ=0.1 is the only model that cracks the biology-hard strains AND generalizes; PLS-DA still has the best raw LOSO mean (0.60 vs DANN 0.500) but zero biology wins.** README should present DANN as the headline LOSO model with the "Pareto frontier crossing" framing; keep PLS-DA + vanilla CNN as comparison rows.
2. ⏳ (optional, deferred from this session per user-staging) lambda_max=0.05 and lambda_max=0.3 sweeps. 0.3 is the higher-value diagnostic because it would disambiguate the memprobe puzzle: does more DANN drop the probe below 10% while preserving biology wins, or does the probe stay decoupled from LOSO regardless of lambda?

## Open items / TODOs

- Tweak arPLS crop start to ~450 cm⁻¹ (currently 400) — boundary artifact at left edge of preprocessed spectra. See [findings.md §arpls-boundary](07_findings.md#arpls-boundary-artifact).
- Re-run Block 9 after crop tweak.
- Run a bacteria-only ANOVA (3 classes, excluding H₂O) to surface within-bacterial discriminative bins. Current 4-class ANOVA bins are about water-vs-bacteria, per [findings.md §anova-bins-vs-stec-discriminative-bands](07_findings.md#anova-bins-vs-stec-discriminative-bands).
- **Tune CNN augmentation regime.** Current default (per §E) slows training so much that every fold early-stops 5-30 epochs short of the 60-epoch budget while train_acc is still 0.4-0.5. A no-aug run on fold 0 reaches train_acc 0.88 in 60 epochs; the spec'd aug looks over-tuned. Worth a Beta(α) ablation on mixup α and a sweep on the per-sample probabilities. Defer to after the Transformer + ensemble runs land.

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
