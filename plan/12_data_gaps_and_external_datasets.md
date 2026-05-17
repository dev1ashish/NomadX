# 12 — Data Gaps and External Datasets

> **Mutability:** mutable. Living document.
> **Last updated:** 2026-05-15.
> **Source:** 4 parallel research subagents (Ho-canon, E.-coli-focused, Salmonella-focused, repository-crawl) + WebSearch on data generation methods.
> **Purpose:** map what we lack, what's publicly available, and what to do about it.

---

## 1. Dataset accounting

### What we have

```
LEVEL          COUNT     WHY IT MATTERS
─────────────────────────────────────────────────────────
strains          9       <-- the LOSO unit
files           87       avg 9.7 per strain
spectra      7,999       7,122 after QC
bins         2,048       per spectrum (raw)
working bins   987       400.4–3049.2 cm⁻¹ post-crop
step          1.67       cm⁻¹
classes          4       STEC, Salmonella, Non-STEC, H2O
```

### Per-strain (LOSO units)

| Class | Strain | Files | Spectra |
|---|---|---|---|
| Non-STEC E. coli | 83972 | 8 | 648 |
| Non-STEC E. coli | ATCC25922 | 9 | 848 |
| Non-STEC E. coli | K-12 | 8 | 648 |
| STEC E. coli | O103H2 | 9 | 848 |
| STEC E. coli | O121H19 | 9 | 848 |
| STEC E. coli | O157H7 | 9 | 848 |
| Salmonella | Dublin | 9 | 848 |
| Salmonella | Heidelburg | 9 | 848 |
| Salmonella | Typhimurium | 9 | 848 |
| H2O | (n/a) | 8 | 767 |

---

## 2. The five gaps, ranked

The bottleneck is NOT spectrum count — it is **strain count and source diversity**. We have 7K spectra but only 9 LOSO units; LOSO mean is a 9-datapoint evaluation no matter how many pixels feed into each.

| # | Gap | Severity | Why |
|---|---|---|---|
| 1 | **Strain diversity** (9 strains total; only 3 per class) | ⭐⭐⭐ blocking | LOSO can't generalize past what 2–3 train strains teach. Hold out K-12 and the model has only 83972 + ATCC25922 to learn "Non-STEC E. coli." |
| 2 | **Lab/instrument diversity** (all spectra from one Raman rig, one prep) | ⭐⭐⭐ blocking | memprobe v2 still fires at 14% → file-id leakable → claims won't transfer to a new instrument. |
| 3 | **Biological replicates per strain** (probably 1–4 colonies, rest are tech reps) | ⭐⭐ heavy | within-strain variability undersampled vs reality. |
| 4 | **Open-set negatives** (only H₂O — no unrelated genera, contaminants, media-only) | ⭐⭐ heavy | model has never seen "neither STEC nor Salmonella nor a known commensal" and will probably fail loudly on it. |
| 5 | **Missing STEC serotypes / Salmonella serovars** (only 3 of ~7 clinical STEC; no Enteritidis, Newport, Infantis) | ⭐ moderate | claim scope is narrower than "STEC vs Salmonella vs Non-STEC." |

---

## 3. The "ecoli thing" sharpened

The brutal cap: with only **3 strains per class**, every LOSO fold trains on 2 strains and tests on 1. No model on earth makes that look like a 90-point eval. The most credible writeup move is to **name the strain-count ceiling explicitly** and, if possible, **add a cross-corpus result** to anchor generalization claims to an outside reference.

---

## 4. Data-generation literature scan (web search)

### (A) Generative models trained on our own spectra — fix sample count, NOT strain count
- **DiffRaman** (latent diffusion, class-conditional VQ-VAE + DDPM). Lifts data-limited classifiers. [arXiv 2412.08131](https://arxiv.org/abs/2412.08131) · [Anal. Chim. Acta](https://www.sciencedirect.com/science/article/abs/pii/S0003267025007664)
- **Wasserstein GAN + Transformer** for E. coli strain ID. [Anal. Chem.](https://pubs.acs.org/doi/10.1021/acs.analchem.6c00429)
- **VAE-LSTM** at 96.9% mean accuracy across 16 strains, 5 species. [J. Chem. Inf. Model.](https://pubs.acs.org/doi/10.1021/acs.jcim.3c00761)

**Verdict for us:** would lift Protocol A. Will not lift LOSO. A generator trained on 8 strains cannot sample the 9th strain's distribution.

### (B) Physics-based simulation (DFT / MD) — real "novel strain" priors, brutal cost
- DFT of cell-wall components (mycolic acid, LPS, peptidoglycan, nucleic acids). [DFT for M. tuberculosis cell wall](https://www.sciencedirect.com/science/article/abs/pii/S0022286024035579)
- ML-accelerated Raman from MD. [J. Chem. Phys.](https://pubs.aip.org/aip/jcp/article/163/12/120901/3365046/Machine-learning-accelerates-Raman-computations)
- MD + SERS reached >98% on 6 species.

**Verdict for us:** powerful in principle. Out of scope for this take-home. Move to `09_future_work.md`.

### (C) Public datasets — free strain diversity if matched. **Highest leverage.** See §5.

### (D) Classical augmentation (you already do some)
- Noise injection, baseline jitter, intensity scaling, mixup, wavenumber stretch. U-Net + noise aug hit 95% binary / 86% on 30 isolates. [ACS Omega](https://pubs.acs.org/doi/10.1021/acsomega.2c03856)

**Verdict for us:** cheapest win. Finish the mixup-α tuning already on the TODO in `00_status.md` line 50 before reaching for generators.

---

## 5. Public dataset hunt — agent-confirmed findings

### 5.1 Coverage matrix

```
Strain we need        Public Raman data?   Notes
─────────────────────────────────────────────────────────────────────────────
E. coli ATCC25922     ✅ YES               2 confirmed sources (Ho, Zhu)
E. coli K-12          ❌ NO                Universal lab strain, never deposited
E. coli 83972         ❌ NO                Probably first ever Raman of this strain
STEC O157:H7          ❌ NO public Raman   Plenty of papers, no open data
STEC O121:H19         ❌ NO public Raman
STEC O103:H2          ❌ NO public Raman
Salmonella Dublin     ❌ NO open license   Tang 2023 has it, request-only
Salmonella Heidelburg ❌ NO public Raman   USDA has HMI not Raman, unreleased
Salmonella Typhimurium ⚠️ species-only     Generic "S. enterica" in open data
```

**The empty rows are the headline.** Worth explicit acknowledgment in the writeup — this is the state of the field, not a search failure.

### 5.2 Actionable datasets (download these)

#### 🥇 Zhu et al. — SCRS Persisters (Front. Microbiol. 2022)
- **URL:** http://mard.single-cell.cn/raw_spectrum_data/
- **License:** Frontiers CC-BY
- **Strain match:** **E. coli ATCC25922** explicit (ampicillin ±)
- **Wavenumber:** 400–3200 cm⁻¹ — near-perfect match to our 400–3049
- **Instrument:** Renishaw WiRE 5.3, 532 nm, 5 s, single-cell, 100× NA 0.9
- **Why #1:** closest experimental sibling we can get. Best axis alignment of any candidate. Single-cell point spectra = same granularity as ours.
- **Use:** train on our ATCC25922 → test on theirs (and vice versa). Single cross-lab generalization number for the writeup.
- [Paper](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2022.936726/full)

#### 🥈 Ho et al. 2019 — "Bacteria-ID" (Nat. Comm.)
- **URL:** https://github.com/csho33/bacteria-ID → Dropbox bundle linked in README
- **License:** MIT code; data effectively "research use, cite"
- **N spectra:** ~80,500 across reference / finetune / test / clinical-2018 / clinical-2019
- **Strain match:** **E. coli ATCC25922 — exact match.** K-12 NOT present (config.py `"E. coli 1"`/`"E. coli 2"` are anonymous IDs, not K-12; Supp. Table 1 confirms). Salmonella entry is *S. enterica* subsp. *arizonae* (ATCC 13314), NOT our serovars.
- **Wavenumber:** 381.98–1792.4 cm⁻¹, ~1.2 cm⁻¹ step — narrower than ours (must crop to 400–1792); CH-stretch region 1800–3049 cannot be cross-tested.
- **Instrument:** Horiba LabRAM HR Evolution, 633 nm @ 13.17 mW, gold-coated silica, 1–2 s exposure. **Heavy domain shift** vs ours.
- **Use:** second independent cross-corpus point on ATCC25922. Hardest transfer test of the shortlist — exactly what we want.
- [Paper](https://www.nature.com/articles/s41467-019-12898-9) · [GitHub](https://github.com/csho33/bacteria-ID)

#### 🥉 Liu et al. 2024 — Raman-OSDL airborne pathogens (Sci. Adv.)
- **URL:** https://doi.org/10.57760/sciencedb.15628 (data) + https://doi.org/10.57760/sciencedb.12074 (code)
- **License:** CC BY-NC 4.0
- **N spectra:** ~23,000 (7,552 train, 3,382 real-world test)
- **Strain match:** *E. coli* + *S. enterica* at species level only. No strain/serovar IDs.
- **Wavenumber:** 600–1800 cm⁻¹ — narrower than ours; fingerprint region only.
- **Instrument:** Horiba LabRAM Aramis, 532 nm, 3 s × 3, single-cell aerosolized.
- **Use:** **open-set / OOD evaluation.** They built this with explicit "non-target" unknown bacteria — directly maps to our open-set gap (#4 in §2).
- [Paper PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11708874/)

### 5.3 Gated but worth one email each

| Dataset | Why pursue | Contact |
|---|---|---|
| **Tang et al. 2023** Talanta — 4-serovar Raman | **Dublin + Typhimurium** match (2 of our 3); 530–1800 cm⁻¹; 785 nm Renishaw — best Salmonella biology match in any modality | corresponding author |
| **Roesch/Pistiki 2022** Anal. Bioanal. Chem. — multi-resistant clinical E. coli | **400–3050 cm⁻¹ axis exact match** to ours; single-cell 532 nm; 1,500 UVRR + 4,168 spontaneous | Petra Roesch (Jena) |
| **Kloss/Roesch 2021** | **ATCC25922 explicit** + Nissle 1917, 300–3100 cm⁻¹ | Petra Roesch (Jena) |
| **Thomsen 2022** Sci. Rep. minimally-prepared | ATCC25922 + ATCC35218; 700–1600 cm⁻¹; 785 nm | corresponding author |

Worst case: no replies in time, no blocker. Send the emails in parallel with the cross-corpus work.

### 5.4 Skip

| Dataset | Why |
|---|---|
| Papa et al. 2025 Zenodo 16740800 (5 Salmonella serovars) | **VNIR HMI, not Raman.** Can't merge spectra. Useful only as a parallel-task reference benchmark in prose. No Dublin or Heidelberg anyway. |
| USDA-ARS HMI Salmonella (Park/Eady) | Unreleased + wrong modality (VNIR not Raman) |
| Zenodo 7109184 (E. coli + heavy metals) | OOD/contamination spectra, not phenotype classification |
| Christensen 2022 figshare | No Salmonella/STEC confirmed; methicillin-resistance focus |
| Dryad UTI SERS chip (mr677g8) | UPEC strain (CFT073), not our strains; SERS not spontaneous |
| Zhang 2021 SERS Salmonella | SERS substrate enhancement makes intensities/baselines non-comparable |

### 5.5 Aggregators / meta-resources

- **RamanBench** (`pip install raman-data`) — 74 datasets, 325K spectra, unified loader. arXiv 2605.02003.
- **MicrobioRaman** — EBI BioStudies, https://www.ebi.ac.uk/biostudies/MicrobioRaman/studies — official open microbial-Raman repo (Nat. Microbiol. 2024). Worth a manual browse.
- **Zenodo 15394102** — community-curated XLSX index of Raman databases (May 2025), CC-BY. Download once, grep later.

---

## 6. Recommended next steps (concrete, ranked)

```
1. Download SCRS Persisters from mard.single-cell.cn.
   Confirm link works; isolate the ATCC25922 spectra.

2. Clone csho33/bacteria-ID; grab Dropbox bundle; isolate ATCC25922 subset.
   Crop our spectra to 400–1792 cm⁻¹ for matched eval.

3. Build cross-corpus eval script:
   - load external ATCC25922 spectra
   - reuse atlas/preprocess.py pipeline
   - run our best per-strain models (PLS-DA, DANN λ=0.1, 2-channel CNN, Patch=5)
   - report per-model accuracy on the external corpus

4. Pull ScienceDB 15628 (Liu et al.) for an open-set probe:
   - run inference on their non-target air bacteria
   - inspect max-proba distribution, confusion patterns
   - addresses gap #4 (open-set negatives)

5. Send 4 parallel emails: Tang 2023, Roesch (2x), Thomsen 2022.
   No blocker if no reply.

6. Finish mixup-α tuning on the CNN (already on TODO in 00_status.md L50)
   before any generative-model work.
```

**Stretch (only if 1–6 are done):** DiffRaman on our 7K spectra to balance classes for Protocol A. Document as a methods addition, not a LOSO claim. Generative augmentation cannot fix the strain-count ceiling.

**Out of scope:** DFT/MD physics-based simulation. Move to `09_future_work.md`.

---

## 7. Writeup framing (the honest version)

Three sentences that should appear somewhere in the README:

> Our evaluation is bounded by 9 strains and a single lab/instrument source. LOSO mean is a 9-point statistic; cross-corpus testing against an independent ATCC25922 corpus (Zhu 2022 / Ho 2019) is the strongest available external anchor. No public Raman dataset covers any of our STEC serotypes (O157:H7, O121:H19, O103:H2) or two of our three Salmonella serovars (Dublin, Heidelburg) — generalization claims for those strains rest on within-corpus LOSO only.

---

## 8. References (full list)

### Methods (data generation)
- DiffRaman — [arXiv 2412.08131](https://arxiv.org/abs/2412.08131) · [Anal. Chim. Acta 2025](https://www.sciencedirect.com/science/article/abs/pii/S0003267025007664)
- WGAN-Transformer — [Anal. Chem.](https://pubs.acs.org/doi/10.1021/acs.analchem.6c00429)
- VAE/GAN comparison — [J. Chem. Inf. Model.](https://pubs.acs.org/doi/10.1021/acs.jcim.3c00761)
- U-Net + noise augmentation — [ACS Omega](https://pubs.acs.org/doi/10.1021/acsomega.2c03856)
- DFT cell-wall (M. tuberculosis) — [J. Mol. Struct.](https://www.sciencedirect.com/science/article/abs/pii/S0022286024035579)
- ML-accelerated Raman from MD — [J. Chem. Phys.](https://pubs.aip.org/aip/jcp/article/163/12/120901/3365046/Machine-learning-accelerates-Raman-computations)
- Recent advances review — [PMC10989577](https://pmc.ncbi.nlm.nih.gov/articles/PMC10989577/)
- Vibrational spectroscopy ML review — [PMC12788301](https://pmc.ncbi.nlm.nih.gov/articles/PMC12788301/)

### Public datasets (top tier)
- Ho et al. 2019 — [Nat. Comm.](https://www.nature.com/articles/s41467-019-12898-9) · [GitHub](https://github.com/csho33/bacteria-ID)
- Zhu et al. 2022 SCRS Persisters — [Front. Microbiol.](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2022.936726/full) · http://mard.single-cell.cn/raw_spectrum_data/
- Liu et al. 2024 Raman-OSDL — [Sci. Adv.](https://www.science.org/doi/10.1126/sciadv.adp7991) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11708874/) · https://doi.org/10.57760/sciencedb.15628

### Public datasets (gated / partial)
- Tang et al. 2023 — [Talanta PMC11037414](https://pmc.ncbi.nlm.nih.gov/articles/PMC11037414/)
- Pistiki/Roesch 2022 — [PMC8761712](https://pmc.ncbi.nlm.nih.gov/articles/PMC8761712/)
- Kloss/Roesch 2021 — [PMC7680742](https://pmc.ncbi.nlm.nih.gov/articles/PMC7680742/)
- Thomsen 2022 — [PMC9524333](https://pmc.ncbi.nlm.nih.gov/articles/PMC9524333/)

### Public datasets (skip / reference only)
- Papa et al. 2025 (VNIR HMI Salmonella serovars) — [Zenodo 16740800](https://zenodo.org/records/16740800) · [HF mirror](https://huggingface.co/datasets/food-ai-nexus/salmonella-serovar-hyperspectral)
- Kang/Park 2019 non-O157 STEC (hyperspectral, not Raman) — [Spectrochim. Acta A](https://www.sciencedirect.com/science/article/abs/pii/S1386142519307760)
- Eady/Park 2016 — [J. Microscopy](https://onlinelibrary.wiley.com/doi/10.1111/jmi.12368)

### Aggregators
- RamanBench — arXiv 2605.02003 · [PyPI raman-data](https://pypi.org/project/raman-data/)
- MicrobioRaman — https://www.ebi.ac.uk/biostudies/MicrobioRaman/studies
- Zenodo 15394102 (DB index XLSX) — https://zenodo.org/records/15394102
- RamanSPy datasets — https://ramanspy.readthedocs.io/en/latest/datasets.html
