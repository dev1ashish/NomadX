# 13 — Methods Research Synthesis

> **Mutability:** mutable. Living document.
> **Last updated:** 2026-05-16.
> **Source:** 4 parallel research subagents (tensor/multi-way, foundation models, OOD/uncertainty/calibration, production deployment) + WebSearch.
> **Purpose:** consolidate methods research across 4 directions, rank concrete next experiments, scope the production upload→classify system.
> **Sibling docs:** `12_data_gaps_and_external_datasets.md` (the data side); `09_future_work.md` (the long tail).

---

## 1. TL;DR

```
WHAT                              EFFORT   ATTACKS                  WHEN
─────────────────────────────────────────────────────────────────────────
1. MCR-ALS unmixing               2 days   memprobe leakage         NOW
2. VICReg SSL on our 7K           2 days   small-data ceiling       NEXT
3. Cross-corpus eval on Zhu       1 day    credibility anchor       NEXT
   (ATCC25922 cross-lab test)
4. Objectosphere + Mahalanobis++  3 days   open-set / OOD safety    THEN
5. Streamlit demo on HF Spaces    2 days   the upload UI            LAST
```

**Headline finding:** MCR-ALS unmixing is the single highest-EV experiment in the whole research scan. It directly attacks the memprobe v2 leakage that's been the unresolved problem across the entire project. ~2 days, drop-in library (`pyMCR`).

---

## 2. Tensor / multi-way verdict

### 2.1 Framing fix

"Form tensors per class and decompose" is fuzzy as stated. Per-class isn't a tensor mode — it's a partition. The natural 3-mode tensor is **per file** = `(x, y, wavenumber)`. The current pipeline flattens this to `(pixel, wavenumber)` and discards spatial structure.

Sharper question: *"Can we exploit the spatial coherence we're currently discarding?"*

### 2.2 Method-by-method

| Method | Verdict | Why |
|---|---|---|
| **PARAFAC** | ❌ skip | Raman maps aren't trilinear. Zero bacterial Raman papers use it. Originally a fluorescence-EEM tool. |
| **Tucker / HOSVD** | ❌ skip | More flexible than PARAFAC but our 5–15 px maps are too small for spatial factors to be meaningful. Useful only as a denoiser. |
| **3D-CNN on (x, y, wn) cubes** | ⚠️ defer | The Anal. Chem. 2025 paper got +4 macro-F1 from 1D→3D-CNN, but they used ~1M spectra / 156 mice. Our 87 maps with 5–15 px sides will overfit. |
| **Spatial-spectral transformers / HybridSN** | ❌ skip | Way too data-hungry for 87 maps. Mature in remote-sensing HSI; near-zero published Raman bacterial use. |
| **N-PLS (multilinear PLS-DA)** | ✅ sanity check | TensorLy's `CP_PLSR`. 1 day. Direct apples-to-apples vs current PLS-DA — tests whether spatial structure carries any class info. If N-PLS materially beats PLS-DA, spatial-DL track is worth pursuing. |
| **MCR-ALS** | ✅✅ **DO THIS** | See §2.3. The standout. |
| **NMF class templates (GBR-NMF)** | ✅ pair with MCR | Forces the classifier to commit to chemistry, not instrument signatures. 1 day on top of pyMCR or sklearn NMF. |

### 2.3 Why MCR-ALS is the standout

MCR-ALS unmixes each map into ~8 components × pure spectra under non-negativity constraints. Used in Anal. Chem. 2024 ([PMC11411491](https://pmc.ncbi.nlm.nih.gov/articles/PMC11411491/)) to separate proteins, lipids, cytochromes, carotenoids, secondary metabolites from substrate / fluorescence drift / acquisition artifacts in microbial-colony Raman maps.

```
CURRENT PIPELINE                       MCR-ALS PIPELINE
─────────────────                      ─────────────────
raw map                                raw map
   ↓                                      ↓
preprocess (SNV etc.)                  MCR-ALS decomposes into
   ↓                                   ~8 components:
feed everything to model                  • protein, lipid, nucleic acid
                                          • fluorescence drift
                                          • substrate signal
                                          • cosmic / fiber background
                                          • per-slide acquisition artifacts
                                       ↓
                                       KEEP ONLY bacterial components
                                       DROP slide/instrument components
                                          ↓
                                       feed cleaned signal to PLS-DA
```

**This is the most direct classical lever against memprobe v2 firing at 14%.** If file-id leakage is genuinely about substrate / fluorescence variation across the 87 slides (plausible), MCR-ALS separates it out before any classifier touches the data.

**Library:** `pyMCR` (NIST). Drop-in. ~2 days to integrate.

**Honest caveats:**
- MCR-ALS only separates substrate if substrate spectrum is *consistent enough* to be a stable component. If each slide has a different fluorescence baseline shape, the decomp can absorb it into multiple components and the leakage stays.
- If file-id is genuinely confounded with strain-id (one slide = one strain instance), no decomp untangles them. Worth checking by inspecting the extracted bacterial components per strain.

### 2.4 What this CANNOT fix

- 9 strains, LOSO denominator. None of these methods generate new strains.
- The fundamental data-size ceiling. 87 maps is small.
- Genuine biological confounding between file-id and strain-id.

---

## 3. Foundation models / pretrained / SSL

### 3.1 The realistic landscape

| Question | Answer |
|---|---|
| Is there a "Raman GPT" we can fine-tune? | **No.** As of May 2026. |
| Published recipes with public weights? | SMAE (no weights), DSCF (gated), SemiRaman (broken repo link), SCDC (no weights). |
| Transfer from time-series foundation models (Chronos, Moirai, TimeGPT)? | **No.** They quantize signals — wrong inductive bias for smooth chemical spectra with localized peaks. |
| Cross-modal NIR→Raman transfer? | **No.** Different physics (overtones vs fundamentals). Don't expect transfer. |
| Pretrain on Bacteria-ID ourselves? | **Yes** — this is the move. |
| SSL on our own 7K spectra? | **Yes** — cheapest, no external data needed. |

### 3.2 Top SSL plays

**(A) VICReg on our 7,122 spectra (CHEAPEST, RECOMMENDED FIRST)**
- Augmentations: wn shift (±5 bins), additive Gaussian noise, polynomial baseline drift, intensity scaling, random masking
- VICReg preferred over SimCLR (no negative-pair batch-size pressure at small scale)
- 200–500 epochs SSL on existing 1D-CNN backbone (124K params) → freeze encoder → linear probe + full fine-tune
- ~2 days on MPS. Pure additive to existing pipeline.

**(B) SMAE-style masked-spectrum pretraining on Bacteria-ID**
- Download Bacteria-ID 60K reference spectra (already on the get-list per `plan/12`)
- Restrict to 400–1792 cm⁻¹ overlap region
- Small Transformer (8 heads, ~200–500K params), mask 50% of wn patches, MSE reconstruction
- Reference: SMAE paper arXiv 2504.16130 (recipe, no released weights)
- ~3 days. Higher expected lift than (A) but external data dependency.

**(C) Prototypical networks on top of (A) or (B)**
- With SSL encoder frozen, build class prototypes per strain from few support spectra; classify by L2 distance in embedding space
- Directly fits few-strain-per-class problem
- Reference: Snell et al. NeurIPS 2017
- ~1 day on top of (A) or (B).

### 3.3 Skip list

- Chronos / Moirai / TimeGPT — wrong inductive bias for spectra
- DreaMS / Casanovo / FACT — mass-spec foundation models, peak-list inductive bias doesn't transfer
- NIR pretrained models — wrong physics
- DSCF — gated, no weights
- SemiRaman — repo link broken
- MAML — only 9 strains = too few episodic tasks

---

## 4. OOD / uncertainty / calibration — the production-system foundation

**This is what makes the upload→classify system trustworthy.** Without it, the model returns confident-wrong answers on every novel bacterium, contamination, instrument shift, and the users trust them. Critical for our planned demo.

### 4.1 The 5-step recipe (from the OOD agent, lightly edited)

```
1. INPUT GATE: 1D conv autoencoder on training spectra.
   Reject if reconstruction MSE > 99th percentile of training error.
   Catches: contamination, water spectra, instrument shifts,
            unknown bacteria, garbage uploads.

2. DEEP ENSEMBLE: 5× CNNs with different seeds, average softmax.
   Gold standard for predictive uncertainty.
   Cost: 5× train, 5× infer. Worth it.

3. MAHALANOBIS++ ON PENULTIMATE: per-class mean/cov on training,
   L2-normalize features first (ICML 2025 fix).
   Threshold at 95th percentile of in-dist validation scores.
   Combine with energy-score OR-rule on logits.

4. CALIBRATION:
   - Temperature scaling per CNN ensemble member (Guo 2017)
   - Isotonic calibration on PLS-DA max-proba (it's systematically
     over-confident — see plan/00_status.md L138)
   - Wrap final ensemble in split conformal prediction at 90% coverage
     → output is a SET of plausible classes, not a single label

5. DRIFT MONITORING (post-deploy):
   - alibi-detect MMD on penultimate features, weekly job
   - Rolling histograms of step-1 reconstruction MSE
   - Auto-alert on KS test p<0.01
```

### 4.2 The most relevant Raman-specific paper

**Balytskyi et al. 2024** — "Enhancing Open-World Bacterial Raman Spectra Identification by Feature Regularization" — directly benchmarked OpenMax vs Mahalanobis vs ODIN vs Energy on Bacteria-ID. Their findings:
- ODIN > Mahalanobis > OpenMax as post-hoc detector
- Training-time **Objectosphere loss** (penalizes feature norm of known classes outward, unknown/background inward) was the big win → ODIN 0.96 (known) vs 0.18 (unknown)
- Repo: [`BalytskyiJaroslaw/PathogensRamanOpenSet`](https://github.com/BalytskyiJaroslaw/PathogensRamanOpenSet) — fork and adapt

### 4.3 The Liu et al. 2024 open-set method (referenced in `plan/12`)

The "open-set deep learning" airborne-pathogens paper uses **OpenMax** (Bendale & Boult CVPR 2016) — fits Weibull on penultimate-activation distances, recalibrates softmax, emits a "none of the above" probability. 93% on 5 known, 84% rejection on >4600 unseen species, threshold 0.98 on val. Implementable in a weekend for 9 classes.

### 4.4 What WON'T help us

- MC Dropout alone — consistently weaker than ensembles, under-estimates epistemic uncertainty on OOD
- Bayesian neural nets — training cost vs benefit terrible at our scale
- Evidential Deep Learning as primary OOD detector — Bengs et al. ICML 2023 showed epistemic estimate inconsistent under data scaling. Use as supplemental, not primary.
- ViM / GradNorm — designed for ImageNet-scale, marginal benefit on 4 classes

### 4.5 The honest 9-strain ceiling

Even the best UQ methods produce well-calibrated uncertainty *over our 4 classes*. They cannot tell the model what it doesn't know about bacteria 10 through 10,000. **OOD detection (autoencoder + Mahalanobis++) is doing the heavy lifting; UQ methods calibrate within-class confidence.** Both layers are needed.

---

## 5. The upload→classify production system

### 5.1 What the user asked for

```
   user upload   →   model   →   "this is STEC"
```

### 5.2 What it must actually be

```
user upload
    ↓
FORMAT LOADER (CSV / JCAMP / Renishaw .wdf — pick 3)
    ↓
PREPROCESS (interpolate to canonical axis 400–1800 cm⁻¹,
            arPLS baseline, Sav-Gol, SNV)
    ↓
┌─── OOD GATE 1: AUTOENCODER RECONSTRUCTION ─────────────────────┐
│  If reconstruction error > 99th percentile of training,         │
│  return "unfamiliar spectrum, refusing prediction"              │
└─────────────────────────────────────────────────────────────────┘
    ↓ (if passed)
CLASSIFIER (calibrated ensemble: PLS-DA + DANN λ=0.1 + 2-channel CNN)
    ↓
┌─── OOD GATE 2: MAHALANOBIS++ ON PENULTIMATE FEATURES ──────────┐
│  If distance > 95th percentile, return "low confidence"         │
└─────────────────────────────────────────────────────────────────┘
    ↓ (if passed)
CONFORMAL PREDICTION at 90% coverage
    ↓
RESPONSE: e.g. "{STEC, Salmonella} — 90% confidence one of these,
          cannot disambiguate. Recommend PCR confirmation."
```

The thing on the right intentionally doesn't say "this is STEC" because with 9 strains in 1 lab, that answer is almost never appropriately confident.

### 5.3 Concrete tech stack

| Layer | Choice | Why |
|---|---|---|
| File formats | CSV/TSV, JCAMP-DX (.dx/.jdx), Renishaw .wdf | Covers ~95% of microbiology uploads; OPUS/SPC adds edge cases without much real coverage |
| Format loaders | `ramanchada2` (JCAMP), `RamanSPy.load.renishaw` (.wdf), pandas (CSV) | All open-source, audited |
| Canonical axis | 400–1800 cm⁻¹, 1 cm⁻¹ spacing (1401 points) | Fingerprint region; matches Ho/Bacteria-ID for cross-corpus compatibility |
| Preprocessing | Whitaker–Hayes cosmic ray → arPLS baseline → Sav-Gol (11,3) → SNV | Standard published bacterial-Raman pipeline |
| Demo framework | Streamlit on HF Spaces | Mirror [polymer-aging-ml Space](https://huggingface.co/spaces/dev-jas/polymer-aging-ml) pattern |
| Production endpoint | FastAPI + sklearn pickle for now | ONNX export later if needed |
| Inference latency | <10ms per 1000-point spectrum on CPU | Batching not needed at portfolio scale |

### 5.4 The 6-step deployment recipe

1. **Freeze canonical axis** at 400–1800 cm⁻¹, 1 cm⁻¹ spacing. Retrain PLS-DA + DANN + CNN on this axis (this also enables Bacteria-ID cross-corpus).
2. **Single `preprocess(wn, intensity) → np.ndarray[1401]` function** using RamanSPy primitives.
3. **Three loaders**: `load_csv`, `load_jcamp` (ramanchada2), `load_wdf` (RamanSPy). Reject anything else with clear error.
4. **OOD gate**: Mahalanobis on PCA(10) of training, threshold at 99th percentile. Return "out-of-distribution" flag instead of suppressing.
5. **Streamlit app on HF Spaces**: file upload → preprocessing trace plot → top-3 predictions + probabilities → OOD flag.
6. **Stub `/predict` FastAPI endpoint** wrapping same `preprocess` + model. ONNX export stays in README as productionization note.

### 5.5 Real-world Raman classifier references

- **Renishaw inVia + WiRE Identify** — spectral search against curated library (not a trained classifier); .wdf upload, ranked hit list
- **Wasatch Photonics RamanID** — embedded classifier on portable spectrometer; closed system
- **Nature Commun 2025 microfluidic-Raman urine diagnostic** — 20-min sample-to-report, 95.4% culture agreement, 305-patient validation; integrated capture → spectrum → CNN
- **Bacteria-ID (Ho 2019)** — the canonical academic-to-OSS pipeline; .npy arrays + pretrained CNN checkpoints

---

## 6. Recommended execution order

```
WEEK 1
  Day 1-2: MCR-ALS unmixing (pyMCR) on a few maps; visual sanity check
           that bacterial vs substrate components separate cleanly
  Day 3:   wire MCR-ALS into preprocess pipeline; rerun PLS-DA, check
           memprobe v2 score (should drop) and LOSO mean
  Day 4-5: VICReg SSL on 7K spectra; linear probe + fine-tune; LOSO

WEEK 2
  Day 1:   download Zhu SCRS Persisters ATCC25922 cross-corpus data
           (per plan/12 §5.2 priority #1); set up matched eval
  Day 2-3: cross-corpus eval — train on our ATCC25922 → test on theirs
           (and reverse); single credibility number for writeup
  Day 4-5: Mahalanobis++ + autoencoder OOD gate on best classifier;
           Liu et al. ScienceDB open-set probe (per plan/12 §5.2 #3)

WEEK 3
  Day 1-2: Objectosphere loss retraining (fork BalytskyiJaroslaw repo);
           re-eval OOD scores
  Day 3-4: conformal prediction wrapper (crepes) + isotonic calibration
           on PLS-DA; calibrated ensemble v2
  Day 5:   Streamlit demo on HF Spaces; one-file pipeline; OOD flag
           visible in UI
```

**Stretch (only if all above lands):** SMAE-style pretraining on Bacteria-ID; N-PLS as comparator vs PLS-DA; NMF class templates.

**Out of scope:** PARAFAC, Tucker, 3D-CNN, spatial transformers, MAML, time-series foundation models, NIR transfer. All ruled out by the agents above.

---

## 7. The brutally-honest framing for the writeup

Five sentences for the README:

> Our evaluation is bounded by 9 strains and a single lab/instrument source.
> Cross-corpus testing against an independent ATCC25922 corpus (Zhu 2022) is the strongest available external anchor.
> The production demo wraps the classifier in two OOD gates (autoencoder reconstruction + Mahalanobis distance on penultimate features) and returns calibrated conformal prediction sets, not point predictions, to fail safely on inputs outside the training distribution.
> No public Raman dataset covers any of our STEC serotypes or two of our three Salmonella serovars — generalization claims for those strains rest on within-corpus LOSO only.
> The strain-count ceiling is real; the cross-corpus anchor and the OOD gates are how the model becomes trustworthy in spite of it.

---

## 8. References

### Tensor / multi-way
- [In Situ Raman Hyperspectral Analysis of Microbial Colonies via MCR-ALS (Anal. Chem. 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11411491/)
- [3D Hyperspectral Data Analysis with Spatially Aware Deep Learning (Anal. Chem. 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12004353/)
- [Hyperspectral unmixing via physics-constrained autoencoders (PNAS 2024)](https://www.pnas.org/doi/10.1073/pnas.2407439121)
- [GBR-NMF for Raman molecular histotyping (Appl. Spectrosc. 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9003771/)
- [Shared subspace via partial Tucker for HSI classification (Spectrochim Acta 2025)](https://www.sciencedirect.com/science/article/abs/pii/S1386142525008911)
- [TensorLy CP_PLSR (multiway PLS regression)](https://tensorly.org/dev/modules/generated/tensorly.regression.CP_PLSR.html)
- [pyMCR (NIST)](https://github.com/usnistgov/pyMCR)
- [TensorLy](https://github.com/tensorly/tensorly)

### Foundation models / SSL
- [SMAE — arXiv 2504.16130](https://arxiv.org/abs/2504.16130)
- [DSCF — Nature Machine Intelligence 2025](https://www.nature.com/articles/s42256-025-01027-5)
- [RamanFormer — ACS Omega 2024](https://pubs.acs.org/doi/10.1021/acsomega.3c09247)
- [SemiRaman — Spectrochim. Acta A 2025](https://www.sciencedirect.com/science/article/abs/pii/S1386142525016646)
- [SCDC — arXiv 2412.20060](https://arxiv.org/abs/2412.20060)
- [VICReg — arXiv 2105.04906](https://arxiv.org/abs/2105.04906)
- [Prototypical Networks (Snell et al. 2017)](https://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning)
- [Bacteria-ID dataset](https://github.com/csho33/bacteria-ID)

### OOD / uncertainty / calibration
- [Liu et al. Sci. Adv. 2025 — Open-set Raman](https://www.science.org/doi/10.1126/sciadv.adp7991)
- [Balytskyi 2024 — Objectosphere + ODIN on Bacteria-ID](https://pubs.acs.org/doi/10.1021/cbmi.4c00007)
- [Balytskyi repo](https://github.com/BalytskyiJaroslaw/PathogensRamanOpenSet)
- [Energy-based OOD — NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf)
- [Mahalanobis++ — ICML 2025](https://arxiv.org/abs/2505.18032)
- [pytorch-ood library](https://github.com/kkirchheim/pytorch-ood)
- [Evidential Deep Learning — NeurIPS 2018](https://arxiv.org/abs/1806.01768)
- [Temperature scaling — ICML 2017](https://arxiv.org/abs/1706.04599)
- [Conformal classification (Angelopoulos)](https://github.com/aangelopoulos/conformal_classification)
- [crepes conformal prediction](https://github.com/henrikbostrom/crepes)
- [alibi-detect](https://docs.seldon.io/projects/alibi-detect/en/latest/)
- [OpenMax — Bendale & Boult CVPR 2016](https://arxiv.org/abs/1511.06233)

### Deployment
- [RamanSPy GitHub](https://github.com/barahona-research-group/RamanSPy)
- [RamanSPy — Anal. Chem. 2024](https://pubs.acs.org/doi/10.1021/acs.analchem.4c00383)
- [ramanchada2 — J. Raman Spectrosc. 2025](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/jrs.6789)
- [SpectroChemPy read_jcamp](https://www.spectrochempy.fr/reference/generated/spectrochempy.read_jcamp.html)
- [polymer-aging-ml Streamlit Space (template)](https://huggingface.co/spaces/dev-jas/polymer-aging-ml)
- [skl2onnx supported models](https://onnx.ai/sklearn-onnx/supported.html)
- [Microfluidic-Raman clinical pipeline (Nature Commun 2025)](https://www.nature.com/articles/s41467-025-66996-y)
- [Liu & Hennelly 2024 wavenumber calibration](https://pmc.ncbi.nlm.nih.gov/articles/PMC11340246/)
- [Cross-device SERS standardization 2025](https://www.sciencedirect.com/science/article/pii/S1386142525012387)
