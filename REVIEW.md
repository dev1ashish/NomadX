# NomadX / Atlas — Project Review

> One-stop summary of: what the dataset is, what we're trying to do, what we've done so far, what's actually working, and what's blocking us.
> Pointers to the deep docs are inline — this file is the index, not the source of truth.
> Last updated: 2026-05-17.

---

## 1. The 30-second version

We're given 87 Raman-spectroscopy files of bacteria (plus water), totalling ~10K spectra. The task is to classify each **file** into one of 4 classes (STEC E. coli / Non-STEC E. coli / Salmonella / H₂O) and ideally to identify the **strain** under realistic generalization (held-out strain, held-out file).

We built the full pipeline (parse → preprocess → QC → split → classical + CNN + Transformer + DANN + ensembles) and ran ~70 experiments. The honest headline:

- **Easy protocol (random file split, 5-fold):** PLS-DA hits ~0.95 file-F1. Solved.
- **Hard protocol (Leave-One-Strain-Out):** PLS-DA solo at **0.603** mean parent-class recall is still the best single number. Every ensemble we tried (5 schemes, 3 base sets) failed to beat it.
- **The genuinely interesting finding:** different model architectures crack *different* held-out strains. No single model wins everywhere; no soft-vote / stack / router captures the union.
- **The real ceiling isn't the model — it's the data.** LOSO mean is a 9-datapoint statistic (one per strain), with only 2 train strains per class. That's a structural cap no architecture fixes.

---

## 2. The dataset

### What's in the box

```
87 files / ~10K spectra / single Raman instrument / single lab
│
├── H2O/                        8 files   water blanks
├── STEC E. coli/              27 files   O103H2 (9) + O121H19 (9) + O157H7 (9)
├── Non STEC E. coli/          25 files   83972 (8) + ATCC25922 (9) + K-12 (8)
└── Salmonella/                27 files   Dublin (9) + Heidelburg (9) + Typhimurium (9)
```

Total: **4 primary classes**, **9 bacterial strains** (LOSO units), **~9.7 files per strain**.

Per file: ~100–700 single-pixel spectra collected on a Raman map of a bacterial smear. Each spectrum has 2048 wavenumber bins from ~76 to ~3499 cm⁻¹.

### Per-file format (the ugly truth)

- Tab-delimited ASCII despite `.xls` extension.
- ~44 metadata header lines (`#KEY=\tVALUE`).
- One wavenumber row, then N pixel rows of `x \t y \t intensity[0..2047]`.
- **Intensity values use comma thousands-separators** (`1,034.00`) — must strip before float-cast.
- **`#NUMX` / `#NUMY` headers are wrong** for early-batch files (R357–R371). Always derive grid from `unique(x) × unique(y)`.
- Two files are **mosaics** (R364 with 9 stitched maps, R370 with 9 stitched maps). One file (R371 Typhimurium) is a **partial scan**.
- Wavenumber axis drifts ~0.05 cm⁻¹ across calibration batches → we interpolate every spectrum onto a single canonical axis `linspace(76, 3499, 2048)`.
- Heidelburg vs Heidelberg — folder name is the misspelling. Kept as-is.

Deep dive: [`plan/01_data.md`](plan/01_data.md).

### What we get after preprocessing

```
7,999 spectra raw (after a 200-px-per-file cap to stop the 2 mosaics from dominating)
  ↓ cosmic-ray removal → arPLS baseline → Sav-Gol smoothing
  ↓ crop to 400–1800 + 2800–3050 cm⁻¹ (drop the noisy edges)
  ↓ SNV (per-spectrum standardize) → optional 2nd derivative
  ↓ QC: SNR ≥ 5 + per-file background-pixel detection
7,122 spectra retained, 987 wavenumber bins each
```

Cached artifacts live in `data_cache/`. Pipeline code: `atlas/io.py`, `atlas/preprocess.py`, `atlas/qc.py`.

### The two split protocols (this is where the action is)

| Protocol | What it is | What it tests | Folds |
|---|---|---|---|
| **A** — StratifiedGroupKFold | Random 5-fold split by `file_id`, stratified by primary class | "Can the model classify a new file from a *seen* strain?" | 5 |
| **B** — Leave-One-Strain-Out (LOSO) | Hold out all files of one strain at a time | "Can the model recognize STEC-ness *as such*, vs strain identity?" | 9 |

Protocol A is the easy one. Protocol B is the real test — and it's where everything interesting happens. Code: `atlas/splits.py`.

---

## 3. What we're trying to do

Four nested questions, hardest last:

1. **Classify the file** into STEC / Non-STEC / Salmonella / H₂O (4-way, headline = macro-F1).
2. **Identify the strain** (9-way, per-strain recall reported as a table).
3. **Do (1) and (2) on a held-out strain** (LOSO) — i.e. generalize across strains, not just files.
4. **Don't accidentally learn instrument / batch / file-id artefacts** — i.e. the model should be doing chemistry, not memorization.

The hardest of these is #3. The whole second half of the project is about closing the LOSO gap.

---

## 4. What we've built (the pipeline)

```
.xls files         atlas/io.py           parse + canonical axis
   │                                            │
   ▼                                            ▼
data_cache/        atlas/preprocess.py   cosmic → arPLS → SG → crop → SNV
   │                                            │
   ▼                                            ▼
splits.py          atlas/splits.py       Protocol A (5-fold) + B (LOSO 9-fold)
   │
   ├─► atlas/models_classical.py         LogReg / LinSVM / RBF-SVM / RF / XGB / PLS-DA
   ├─► atlas/models_cnn.py               small 1D-CNN (124K params) + DANN variant
   ├─► atlas/models_transformer.py       small 1D-Transformer (217K params)
   │
   ├─► atlas/ensemble.py                 soft-vote ensembles
   ├─► atlas/stacking.py                 meta-learner over base probas
   ├─► atlas/lambda_selector.py          per-strain DANN-λ routing
   ├─► atlas/calibrated_ensemble.py      temperature-scaled soft-vote
   │
   ├─► atlas/memprobe.py / memprobe_v2.py   "is the encoder memorizing file_id?" probe
   └─► atlas/evaluate.py                 per-spectrum → per-file aggregation + metrics
```

Each `scripts/run_*.py` is the CLI wrapper. Every run drops to `outputs/<DATE>_<NAME>_<HASH>/` with parquets of per-spectrum probas, a config snapshot, and metrics JSON.

---

## 5. What we've actually run (chronological highlights)

### Phase 1 — Classical baselines (LogReg, LinSVM, RBF-SVM, RF, XGB, PLS-DA)

Both protocols, all 6 models. PLS-DA emerges as the LOSO winner; everything else is competitive on Protocol A but falls apart on LOSO. **PLS-DA solo: 0.603 LOSO mean. This is the number that hasn't been beaten.**

### Phase 2 — 1D-CNN (small, 124K params)

Built around a per-bin StandardScaler + 4-stage Conv1d with channel widths `32-64-96-128`. Trained with heavy augmentation (mixup, noise, baseline jitter, intensity scaling, wavenumber stretch).

- Protocol A file-F1: **0.649** (below the 0.92–0.98 pre-registered floor).
- LOSO mean: **0.35** (below the 0.55 floor).
- **But — cracks K-12 (Non-STEC) at 0.50 and O157H7 (STEC) at 0.56**, both of which classical models hit 0.00 on.
- **Memprobe v2 fires at 15.5%**: the encoder has linearly-decodable file_id signal (chance is 1.15%). Triggers the DANN branch.

### Phase 3 — Small 1D-Transformer (~217K params, patch_size=20)

The weakest single arm. Protocol A 0.507, LOSO 0.193, K-12 and O157H7 both collapse to 0.00.

**Diagnostic finding:** 20-bin patches blur the narrow-peak Raman signal that the CNN's k=5–15 kernels catch. Confirmed by re-running with patch_size=5: LOSO jumped to 0.349 and **ATCC25922 hit 1.00** (first 100% on that strain anywhere).

### Phase 4 — DANN (CNN + Gradient Reversal Layer)

Added an 87-way file_id domain head to the CNN, trained adversarially. Two settings shipped:

- **λ=0.1** (single-seed 42): LOSO **0.500**, K-12 0.75, O157H7 0.56, ATCC25922 0.89, O121H19 0.67. The original "ship this" number.
- **λ=0.3** (single-seed 42): LOSO 0.447, K-12 **0.88**, O157H7 **0.67**, but craters on ATCC25922 (0.11) and 83972 (0.25).

**Both numbers got partially walked back by multi-seed verification** — see Phase 6 below.

### Phase 5 — Ensembling (5 schemes, none worked)

Ran every reasonable combination strategy:

| Scheme | Best LOSO mean | K-12 | O157H7 | Verdict |
|---|---|---|---|---|
| Soft-vote (PLS-DA + XGB + CNN) | 0.579 | 0.00 | 0.00 | CNN's wins destroyed by averaging |
| Stacking (LogReg meta over base probas) | 0.432 | 0.00 | 0.00 | Can't extrapolate across LOSO folds |
| Per-strain λ-selector (hard / soft / router) | 0.444 | — | — | Inner-val signal doesn't predict held-out |
| 4-arch soft-vote (PLS-DA, DANN, Patch5, 2ch-CNN) | 0.579 | 0.00 | 0.11 | Still loses to PLS-DA solo |
| Margin-based confidence router | 0.603 | 0.00 | 0.00 | Degenerates to "always pick PLS-DA" 78/78 files |
| Temperature-scaled soft-vote | 0.566 | 0.00 | 0.67 | First to break O157H7, but K-12 stays 0 |

**Mechanism finding (sharp):** PLS-DA produces systematically more peaked probability distributions (fitted T = 6.43 vs deep models 1.2–1.7). When a 1-of-4 deep model is right on a "minority-of-one" strain (K-12 owned only by DANN; Typhimurium owned only by PLS-DA), no calibration scheme can amplify its vote past three confidently-wrong bases.

The ensemble chapter is closed: **5 schemes × 3 base sets, no Pareto improvement.**

### Phase 6 — Multi-seed verification (the painful one)

Re-ran the headline DANN runs across 5 random seeds each. Reality check:

| Variant | Single-seed (originally shipped) | 5-seed mean ± SD | 5-seed soft-vote |
|---|---|---|---|
| DANN λ=0.1 | 0.500 | **0.345 ± 0.145** | 0.370 |
| DANN λ=0.3 | 0.447 | 0.393 ± 0.117 | **0.448** |

**The original "ship λ=0.1" decision was based on a lucky seed.** Under multi-seed averaging, **λ=0.3 actually wins** (0.448 vs 0.370) and is robust on K-12 (0.75 soft-vote) and O157H7 (0.78 soft-vote).

**The K-12 narrative needs a revision:** the "DANN cracks K-12 at 0.75" claim is real for λ=0.3 (3 of 5 seeds hit ≥0.75) but at λ=0.1 it's mostly a single-seed artefact (K-12 SD = 0.35 across seeds).

### Phase 7 — Other diagnostics & dead-ends (kept for the writeup)

- **2-channel CNN** (SNV + fixed 2nd-derivative Laplacian): LOSO 0.465, second-best single arm. New highs on O121H19 (0.89) and O157H7 (0.78).
- **Grouped-domain DANN** (predict subclass not file_id): LOSO 0.309. Rejects the "coarser domain target preserves more" hypothesis.
- **Aug regime sweep:** lighter augmentation regresses on LOSO. Heavy aug is doing genuine cross-strain regularization, not over-tuning. (Closed the open TODO from `plan/00`.)
- **Memprobe revisits:** the 14% file_id probe survives at all DANN λ values (14.0 → 13.6%). **The probe is decoupled from LOSO performance.** The 10% probe-threshold rule we pre-registered was wrong for this dataset; DANN reshapes feature *prominence*, not linear file-id separability.

---

## 6. The per-strain best-model table (the actual writeup story)

This is the single most informative table in the project. No model wins every cell. Each architecture owns a different held-out strain.

```
Strain              Class      Best model                       Recall
─────────────────────────────────────────────────────────────────────
83972               Non-STEC   PLS-DA                            1.00
ATCC25922           Non-STEC   Patch=5 Transformer / DANN λ=0.1  ~1.00
K-12                Non-STEC   DANN λ=0.3 (5-seed soft-vote)     0.75
O103H2              STEC       PLS-DA / DANN λ=0.3               ~0.89
O121H19             STEC       2-channel CNN                     0.89
O157H7              STEC       DANN λ=0.3 (5-seed soft-vote)     0.78
Dublin              Salmonella PLS-DA                             0.7+
Heidelburg          Salmonella PLS-DA                             0.7+
Typhimurium         Salmonella PLS-DA (only model > 0)            ~0.56
```

Read the row, read the column — the takeaway is *"different inductive biases solve different biology cells, and no current combination scheme captures the union."*

Full per-strain breakdown with confidence intervals: [`plan/07_findings.md`](plan/07_findings.md).

---

## 7. The problems we're facing (honest list)

### 7.1 The big one — data ceiling

LOSO mean is a **9-point statistic**. With only 3 strains per class, every LOSO fold trains on 2 strains and tests on 1. **No model on earth makes that look like a 90-point eval.** This is structural, not algorithmic.

Concrete: only ~7K labelled spectra, one Raman instrument, one prep, probably 1–4 biological replicates per strain (rest are technical replicates). Open-set negatives are basically nonexistent (only H₂O). See [`plan/12_data_gaps_and_external_datasets.md`](plan/12_data_gaps_and_external_datasets.md) for the full accounting.

**What this means:** any model that hits LOSO 0.9 on this data is overfitting. The 0.60 wall is probably close to the true achievable number with these splits.

### 7.2 Ensembles don't work here

5 different schemes × 3 base sets, zero Pareto improvement over PLS-DA solo. Mechanism is well-understood now (calibration mismatch + minority-of-one strains), but it means the obvious "throw a stacker on top" win is closed.

### 7.3 Seed variance is large for deep models

5-seed SD ≈ 0.12–0.15 LOSO mean for DANN, with per-strain SD up to 0.35 (K-12 at λ=0.1). **Anything reported single-seed is suspect.** The honest reporting unit is 5-seed soft-vote, not single-seed point estimates.

This bit us once: shipped λ=0.1 as the headline based on a lucky seed; under proper multi-seed characterization λ=0.3 dominates.

### 7.4 Memprobe still fires

The 87-way file_id probe sits at ~14% top-1 (chance 1.15%) even after DANN. The probe was supposed to be the "is the model memorizing?" guard, but it turns out to be **decoupled from LOSO performance** on this dataset — DANN reshapes feature *prominence* but doesn't strip linear file-id separability. The cleanest interpretation: there's residual instrument/batch signal in the features that doesn't hurt cross-strain generalization in the direction we care about.

### 7.5 arPLS boundary artefact

Left edge of the preprocessed spectra (~400 cm⁻¹) has a baseline-correction spike that's not real chemistry. Open TODO: re-crop to 450 cm⁻¹ and re-run.

### 7.6 The "minority-of-one" strains

- **K-12** is only solved by DANN (broad-scale adversarial denoising).
- **Typhimurium** is only solved by PLS-DA.
- No ensemble captures both because the rest of the bases vote confidently-wrong on each.

The only known fix is per-strain model selection, which leaks the held-out label. Tried 3 leakage-bounded selectors, all failed.

### 7.7 What's NOT a problem

- Parser is stable, 87/87 files parse, comma/grid/mosaic edge cases all handled.
- Preprocessing pipeline is solid (modulo the 7.5 boundary tweak).
- Reproducibility is good (everything seeded, every run dumps its config snapshot, every output is hashable).
- Both protocols implemented correctly, no file leakage.

---

## 8. Where we are vs the budget

| Phase | State |
|---|---|
| 0. Discovery | done |
| 1. Subagent research (5 in parallel) | done |
| 2. Synthesis into master plan | done |
| 3. User approval of plan | done |
| 4. Implementation — parser, EDA, preprocess, QC, classical, CNN, Transformer, DANN, ensembles | done |
| 4b. Multi-seed verification & ensemble post-mortem | done |
| 5. Remaining diagnostics (arPLS boundary, bacteria-only ANOVA, SHAP/saliency, TTA) | open |
| 6. README narrative update + plots | open |
| 7. `make verify` + CI green | open |

Open TODOs are listed in [`plan/00_status.md`](plan/00_status.md) §"Open items / TODOs".

---

## 9. What's worth doing next (ranked)

The user-facing question is *"is there a research swing left, or is it writeup time?"* Honest take: there are 1–2 real swings, then it's writeup.

1. **Cross-corpus eval on ATCC25922** (highest leverage). Two public Raman datasets explicitly contain ATCC25922 (Zhu 2022 SCRS Persisters; Ho 2019 Bacteria-ID). Wavenumber axes are compatible. Lets us put a **single cross-lab generalization number** in the writeup — the strongest external anchor available. See [`plan/12 §5.2`](plan/12_data_gaps_and_external_datasets.md).
2. **Open-set probe via Liu 2024 Raman-OSDL.** Their non-target bacteria spectra → measure how confidently our models mispredict on truly out-of-distribution input. Directly addresses gap #4 (we've never trained on open-set negatives).
3. **arPLS crop fix + bacteria-only ANOVA.** Cheap polish; re-run Block 9 EDA.
4. **SHAP / saliency on the best DANN and 2-channel CNN.** No expected LOSO lift, but it gives the writeup a "the model is looking at chemistry, not artefacts" figure.
5. **README narrative refresh** with all the post-2026-05-14 findings (per-strain table is the new headline, not a single number).

What's *not* worth doing inside this take-home (moved to [`plan/09_future_work.md`](plan/09_future_work.md)):

- Self-supervised pretraining on external Raman corpora (10–15 days + GPU).
- DiffRaman / WGAN generative augmentation (fixes Protocol A, can't fix LOSO).
- DFT / MD physics-based simulation (out of scope; real cost).
- Hierarchical or binary-STEC submodels (small expected gain; long tail of effort).

---

## 10. Where everything lives

```
plan/
├── 00_status.md                          ← living status, single most useful file
├── 01_data.md                            ← stable dataset spec
├── 02_decisions.md                       ← locked design choices + rationale
├── 03_architecture.md                    ← module/code layout
├── 04_eda_plan.md
├── 05_implementation_order.md
├── 06_risks.md                           ← risk register, what's realized
├── 07_findings.md                        ← 99K of dated experiment results
├── 08_expectations.md                    ← pre-registrations for every run
├── 09_future_work.md                     ← out-of-scope follow-ups
├── 10_decision_log.md                    ← every decision change, dated
├── 11_references.md                      ← lit review
├── 12_data_gaps_and_external_datasets.md ← public data accounting
└── 13_methods_research_synthesis.md      ← methods lit synthesis

atlas/                  pipeline + model code
scripts/                CLI runners (one per arm)
outputs/                70+ dated experiment dirs, each with config + per-fold parquets
data_cache/             parsed + preprocessed numpy arrays + split JSONs
README.md               the user-facing writeup (still pre-Phase-7 in places)
```

If you read only two files: this one for the map, [`plan/00_status.md`](plan/00_status.md) for the live state.

---

## 11. TL;DR for someone walking in cold

- **Pipeline works.** Parse + preprocess + QC + split + 4 model families + ensembling are all in `atlas/` and tested under both protocols.
- **Best single number is PLS-DA at LOSO 0.603.** That's the headline.
- **Best per-strain story is "different models own different strains."** That's the actual writeup.
- **Ensembles failed to combine them.** 5 schemes, 3 base sets, none clear PLS-DA solo. Mechanism understood.
- **Multi-seed verification revised some headlines.** DANN λ=0.3 5-seed soft-vote (0.448) > DANN λ=0.1 single-seed (originally shipped at 0.500). Always trust the 5-seed number.
- **The wall is data, not modeling.** 9 strains, 1 instrument, 1 lab. A cross-corpus result on ATCC25922 (publicly available) is the best remaining swing.
