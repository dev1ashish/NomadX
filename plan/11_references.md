# 11 — References

> **Mutability:** append-only. Add new entries with date + tag. Cite as [(tag)](#anchor) inline from other plan files.
> **Purpose:** literature pointers we don't want to re-discover. Skim before starting any new modeling experiment.

---

## E. coli STEC vs Non-STEC discrimination

### `cisek-2013` — Cisek et al., *Analyst* 2013, "Sensitive and specific discrimination of pathogenic and nonpathogenic Escherichia coli using Raman spectroscopy"
- **URL:** [PMC3617710](https://pmc.ncbi.nlm.nih.gov/articles/PMC3617710/)
- **What:** Visible Raman, 600–2000 cm⁻¹, on E. coli O157:H7 vs non-pathogenic E. coli C, K-12 (Hfr), HF4714. Two chemometric techniques.
- **Key result:** >95% sensitivity and specificity. **Discriminative bands: 1658 cm⁻¹ (amide I, protein), 1454 cm⁻¹ (CH₂/CH₃ deformation, lipid), 1338 cm⁻¹ (CH₂ wagging / adenine ring breathing).**
- **Why it matters for us:** these 3 bands are all in our fingerprint crop (400–1800). The 2880–2940 C-H stretch peaks we found in our 4-class ANOVA ([07_findings.md §anova-c-h-stretch](07_findings.md#anova-c-h-stretch)) are about the **water-vs-bacteria boundary**, not E. coli subtype discrimination. For the hard STEC↔Non-STEC cell we need to look at fingerprint-region differences, not C-H. **Implication:** when interpreting model errors and feature importance, check whether RF/XGBoost weight bins are 1338/1454/1658 — that's the published STEC-relevant signal.
- **Caveat:** they used controlled cell suspensions; per-batch CV, not LOSO. Their 95% number is an upper bound on what we should expect, not a floor.

### `tang-2026-wgan` — Tang et al., *Anal. Chem.* 2026, "Integrated Wasserstein GAN–Transformer for E. coli Strain Identification"
- **URL:** [acs.analchem.6c00429](https://pubs.acs.org/doi/10.1021/acs.analchem.6c00429)
- **What:** WGAN augmentation + Transformer for E. coli strain ID.
- **Key result:** 97% on 5-fold CV → 94% on independent test set. 3pp drop with state-of-the-art deep learning + augmentation.
- **Why it matters for us:** the field knows GroupKFold inflates over independent strains. Our 80pp LOSO crater is huge by comparison because (a) we have only 87 files vs their thousands and (b) classical models can't compensate without domain adaptation. **Realistic deep-learning ceiling on transfer-to-new-strain is ~94%, not 99%.**

### `soupene-2003-k12` — Soupene et al., *J. Bacteriol.* 2003, "Laboratory strains of Escherichia coli K-12: things are seldom what they seem"
- **URL:** [PMC9997739](https://pmc.ncbi.nlm.nih.gov/articles/PMC9997739/)
- **What:** Genomic + phenotypic characterization of common K-12 derivatives.
- **Key result:** K-12 has been laboratory-domesticated since the 1920s; accumulates large genomic deletions vs wild-type E. coli; missing common stress-response and surface-structure genes. K-12 is **genuinely atypical** vs clinical or environmental E. coli.
- **Why it matters for us:** K-12 in our LOSO fold gets misclassified as Salmonella 8/8 times. **This is not a bug** — K-12 has diverged enough from typical Non-STEC E. coli that the model can't find the right manifold. Honest reporting should note "K-12 is a laboratory strain known to be atypical" rather than treating it as a normal Non-STEC test case.

### `non-o157-stec-overview` — Marler Clark / FSIS Non-O157 STEC overview
- **URL:** [marlerclark/non-o157-stec](https://marlerclark.com/foodborne-illnesses/e-coli/non-o157-stec)
- **What:** Background on why non-O157 STEC is medically and analytically harder than O157:H7.
- **Key fact:** ~80% of US non-O157 STEC infections are caused by 6 serogroups: O26, O111, O103, O121, O45, O145. **Non-O157 STEC has no unique biochemical characteristics** to distinguish them from non-pathogenic E. coli on standard media. Our STEC subclasses include O103:H2, O121:H19 — both in this "hard" non-O157 group.
- **Why it matters for us:** the STEC-vs-Non-STEC boundary is virulence-defined (Shiga toxin presence), not phylogenetic. The bacterium itself is mostly identical; the difference is one phage-encoded protein. Hard problem by construction.

## Salmonella discrimination

### `yuan-2024-salmonella` — Yuan et al., *J. Cell. Mol. Med.* 2024, "Rapid discrimination of four Salmonella enterica serovars"
- **URL:** [PMC11037414](https://pmc.ncbi.nlm.nih.gov/articles/PMC11037414/)
- **What:** SERS + SVM on Dublin / Enteritidis / Typhi / Typhimurium, 920 spectra (230 per serovar, 4 strains each).
- **Key result:** SVM 99.97% (handheld) / 99.38% (benchtop). Discriminative bands: 616 (COO⁻ wag), 925 (C-C skeletal), **1486 (guanine ring)**, 1542 (C=C).
- **Why it matters for us:** Salmonella subclass discrimination IS achievable. Their 99% number is intra-batch though; cross-strain we're seeing 0.3 parent-class recall in LOSO. **The realistic Salmonella-serovar ceiling on cross-strain transfer is unknown — published results don't test it.**

## Domain adaptation for bacterial Raman

### `rscdm-2026` — Anal. Chem. 2026, "Raman Spectral Classification Discrepancy Model"
- **URL:** [acs.analchem.5c07113](https://pubs.acs.org/doi/10.1021/acs.analchem.5c07113)
- **What:** Domain-adaptation framework targeting "instrument heterogeneity, batch variability, **strain diversity**".
- **Why it matters for us:** the three domain-shift axes we have (calibration-date batch effect, file-level batch, cross-strain LOSO). Direct method to compare against if we enable DANN.

### `sun-2025-contrastive` — Sun et al. 2025, "Adversarial Contrastive Domain-Generative Learning"
- **URL:** [S0952197625004269](https://www.sciencedirect.com/science/article/abs/pii/S0952197625004269)
- **What:** Adversarial contrastive learning for cross-domain bacterial Raman ID.
- **Key result:** 5+ percentage point improvement vs no-adaptation baselines on cross-domain bacterial ID.
- **Why it matters for us:** justifies the DANN-on-file_id plan in `03_architecture.md` §E. If memorization probe fires, this is the family of methods to apply.

### `lora-ct-2025` — Sun et al. 2025, "LoRA-CT: Calibration Transfer of Deep Learning Models"
- **URL:** [acs.analchem.5c01846](https://pubs.acs.org/doi/10.1021/acs.analchem.5c01846)
- **What:** Parameter-efficient fine-tuning across Raman spectrometers.
- **Why it matters for us:** confirms "systematic interdevice variations" (≈ our calibration-date drift) is a recognized primary distribution-shift driver. Validates pre-build memo concern.

## STEC virulence factor biology (for interpretation)

### `stec-virulence-overview` — *Virulence* 2013, "Shiga toxin-producing Escherichia coli"
- **URL:** [tandfonline doi/10.4161/viru.24642](https://www.tandfonline.com/doi/full/10.4161/viru.24642)
- **Key fact:** STEC virulence is defined by Shiga toxin production. Stx is **chromosomally encoded as part of a lysogenic phage** — i.e., the bacterium acquires/loses Stx via horizontal gene transfer. stx1 has 4 subtypes, stx2 has 12 subtypes. Other virulence markers (ehxA, Saa, katP, espP, stcE, subAB) are mostly plasmid-encoded.
- **Why it matters for us:** the STEC vs Non-STEC distinction is at the level of one phage-encoded protein against an otherwise-identical bacterium. **The cell wall, ribosomes, core cytoplasm — i.e. the bulk Raman signal — are NOT different between STEC and Non-STEC.** Expecting a CNN to find a virulence-protein signature in a label-free Raman map is asking a lot. This sets a hard biological ceiling on the within-E. coli problem.

## Cross-reference back to plan files

When you read these papers, consider updating:
- **plan/07_findings.md** — append observations that confirm or contradict the published bands (e.g. if RF feature importance peaks at 1454 or 1658, cite `cisek-2013`).
- **plan/09_future_work.md** — if DANN is enabled, mention `rscdm-2026` and `sun-2025-contrastive` as method comparators.
- **plan/06_risks.md** — `soupene-2003-k12` justifies adding a K-12-specific risk note: "K-12 may be an unrecoverable LOSO fold for biological reasons, not data-quality reasons."
