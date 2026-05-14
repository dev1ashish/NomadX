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
| 4. Implementation | in progress | Steps 1–3 done (parser, EDA, preprocess+QC, classical models, CNN, ensemble, Transformer). DANN ablation + README left. |

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

## What's next (in order)

1. ⏳ **DANN ablation on CNN** ([07§memprobe-v2-fires](07_findings.md#2026-05-14--memprobe-v2-fires)) — verify whether DANN keeps the K-12 + O157H7 wins or destroys them. Ensemble ([07§ensemble-fails-to-clear-plsda](07_findings.md#2026-05-14--ensemble-fails-to-clear-plsda)) and Transformer ([07§transformer-underperforms-cnn](07_findings.md#2026-05-14--transformer-underperforms-cnn)) are both ruled out as alternative paths to recover those wins. DANN is the only remaining lever.
2. ⏳ Final plots, README narrative, `make verify`, CI green. **Ship PLS-DA solo as headline LOSO model; flag CNN as single-model-best on K-12 (0.50) and O157H7 (0.56) per-strain.**

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

## Ensemble session summary (2026-05-14)

| Headline metric | Pre-registered range | Actual | Verdict |
|---|---|---|---|
| Best ensemble LOSO mean parent-recall (b: PLS-DA + CNN) | 0.55 – 0.72 | **0.579** | ✅ in range; **fails to beat PLS-DA solo 0.60** |
| Best ensemble Protocol A file-F1 (b: PLS-DA + CNN) | 0.85 – 0.94 | **0.919 ± 0.030** | ✅ in range; below PLS-DA solo 0.951 |
| K-12 recall in best ensemble | 0.13 – 0.50 | **0.00** | ❌ below floor — CNN's win destroyed |
| O157H7 recall in best ensemble | 0.14 – 0.56 | **0.00** | ❌ below floor — CNN's win destroyed |

**The hypothesized ensemble win does not materialize.** Soft-vote averaging crushes minority-vote signal: even when CNN is correct on K-12 and O157H7 at the per-spectrum level, its proba mass on the right class is the file-level runner-up, not the file-level argmax. Averaging with two confident classical models that are confidently wrong on those folds pulls the file-level decision back to the wrong class. The verdict structure pre-locked branch "all three ensembles ≤ 0.60 mean → ship PLS-DA solo" is the one we hit. CNN's K-12 / O157H7 wins remain real but only retrievable via per-strain model selection, not soft-vote ensemble. Detail at [07§ensemble-fails-to-clear-plsda](07_findings.md#2026-05-14--ensemble-fails-to-clear-plsda).
