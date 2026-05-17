# NomadX / Atlas — Project Review (Biology-Side Version)

> Written for a wet-lab / microbiology / Raman-spectroscopy reader who wants to know:
> (a) what's actually in the dataset,
> (b) what we're asking a model to do with it,
> (c) why this is harder than it looks.
> No machine-learning jargon assumed. The ML-side version of this same review is in [`REVIEW.md`](REVIEW.md).

Last updated: 2026-05-17.

---

## 1. The one-paragraph version

Someone gave us 87 Raman spectroscopy files — one per bacterial culture (or water blank) — and asked: *can a model look at the Raman spectrum and tell us what the organism is?* Specifically: STEC E. coli vs Non-STEC E. coli vs Salmonella vs water. We built the full pipeline, ran a stack of models, and the punchline is: **the easy version of the question (recognize a new file from a strain we've seen before) is essentially solved at ~95% file-level accuracy. The hard version (recognize a strain we've never seen before, from spectra alone) tops out around 60% mean per-strain recall, and that ceiling is mostly about the dataset, not the model.**

---

## 2. What's in the dataset (the biology)

87 files, all collected on a single confocal Raman microscope at one lab. Each file is one bacterial map (a smear or pellet imaged pixel-by-pixel, typically a few hundred single-cell-or-near-cell spectra per file). Four primary classes, nine bacterial strains:

```
CLASS              STRAIN          FILES   ~SPECTRA   NOTES
─────────────────────────────────────────────────────────────────────────────────
H₂O                —                  8       767     water blanks, instrument floor
STEC E. coli       O157:H7            9       848     the canonical food-poisoning STEC
                   O121:H19           9       848     non-O157 STEC, clinically real
                   O103:H2            9       848     non-O157 STEC
Non-STEC E. coli   K-12               8       648     universal lab workhorse strain
                   ATCC25922          9       848     QC reference strain (clinical isolate origin)
                   83972              8       648     asymptomatic-bacteriuria isolate
Salmonella         Typhimurium        9       848     S. enterica serovar Typhimurium
                   Heidelburg         9       848     S. enterica serovar Heidelberg (note the misspelled folder name)
                   Dublin             9       848     S. enterica serovar Dublin
                                                ─────
                                                ~7,999 raw spectra; 7,122 after quality control
```

### Why these four classes are the right question

- **STEC = Shiga-toxin-producing E. coli.** Outbreaks, HUS, public-health-reportable. Distinguishing STEC from a benign commensal *E. coli* in a food matrix is a real-world food-safety problem.
- **Non-STEC E. coli** here is three strains that share a species with STEC but lack the Shiga-toxin phenotype. They're the hardest discrimination — *same species, different pathogenicity*. K-12 and ATCC25922 are lab strains; 83972 is a clinical isolate from asymptomatic bacteriuria.
- **Salmonella** is the other major foodborne genus we'd want to call out separately. Three serovars represented.
- **H₂O** is the negative control — water on the same instrument under the same conditions. It also pins down the "what does the *instrument* look like with nothing biological in it" baseline.

### What each individual file is

A single Raman map of one culture. The microscope rasters over a ~22×17 (or larger) grid of micrometre-scale pixels and records a full Raman spectrum at each pixel. We get out:

- ~44 lines of header metadata (laser power, integration time, calibration date, etc.).
- The wavenumber axis (2048 points, ~76–3499 cm⁻¹).
- One spectrum per pixel (each spectrum = 2048 intensity values).

A "file" therefore contains anywhere from ~100 to ~700 single-point spectra, all from the same culture. After QC and a per-file cap of 200 pixels (to stop two unusually large mosaics from dominating their classes), we have **7,122 quality-controlled spectra across the 87 files.**

### The spectral region we keep

Raw axis spans 76 to 3499 cm⁻¹, but the useful region for bacteria is:

```
400 ── 1800 cm⁻¹  "fingerprint" region — nucleic acids, proteins, carbohydrates,
                  phenylalanine ring breathing (~1003), amide I (~1660), etc.
2800 ── 3050 cm⁻¹  C–H stretch region — lipids, membrane composition signal
```

We crop to those two windows (987 wavenumber bins total) and drop the noisy edges and the silent region in between. Pipeline order: cosmic-ray spike removal → baseline correction (arPLS) → Savitzky–Golay smoothing → crop → SNV normalization. Optional 2nd-derivative channel for the linear models.

### Edge cases worth knowing about

- **Wavenumber axis drifts ~0.05 cm⁻¹ across calibration batches.** We interpolate every spectrum onto a single canonical axis to fix this.
- **Two files are mosaics** (multiple stitched maps in one file): R364 has 9 tiled STEC maps (324 pixels); R370 has 9 tiled Salmonella Dublin maps (720 pixels). Without the 200-pixel-per-file cap, these would dominate their classes.
- **R371 (Typhimurium) is a partial scan** — 351 of an expected 360 pixels. We keep it with a `is_complete_scan=False` flag.
- **Early-batch files (Feb–early March) have wrong `#NUMX` / `#NUMY` headers.** We derive the grid from the actual x/y coordinates instead of trusting the header.
- **Folder name is "Heidelburg"** — likely a typo for the German city Heidelberg. We keep the original spelling so the data path matches the labels.

For the fully detailed dataset spec see [`plan/01_data.md`](plan/01_data.md).

---

## 3. Raman spectroscopy — the 60-second refresher

You probably know this, but to fix terms used later:

- Each spectrum is a vector of intensities at 2048 wavenumber bins.
- A peak at, say, 1003 cm⁻¹ is the phenylalanine ring-breathing mode — present in every bacterial spectrum because every bacterium has phenylalanine in its proteins.
- A peak at ~785 cm⁻¹ comes from nucleic acid ring breathing. ~1450 from CH₂ scissoring. The 2800–3050 region is dominated by membrane-lipid C–H stretches. And so on.
- *Within a species*, the spectra are very similar — the chemistry is mostly the same. The question is whether the **small** differences (relative peak heights, subtle shifts, the precise shape of broad envelopes) are enough to tell, say, K-12 apart from ATCC25922.
- *Across species*, the differences are larger but still not dramatic — and they sit on top of instrument-level variability (laser drift, focus, sample thickness, baseline shape) that has nothing to do with the biology.

This is the central problem the project lives with: **the chemistry differences we want are small, and they're embedded in measurement noise that's not small.**

---

## 4. What we're asking the model to do

Two tests. The first is the easy one; the second is what you actually care about.

### Test A — "Recognize a new file from a strain we've seen"

We randomly split the 87 files into 5 folds (4 folds train, 1 fold test, rotate). The test files come from strains that are also represented in the training set. We just haven't seen *these particular cultures* before.

Biologically: this is the "I have a new sample from K-12 — can the model recognize it as Non-STEC E. coli?" question.

**Result: ~95% file-level accuracy with classical models (PLS-DA leads).** Essentially solved.

### Test B — Leave-One-Strain-Out (LOSO). "Recognize a strain we've never seen."

We hold out all files from one entire strain at a time. So when we test on K-12 files, the model has never been trained on a single K-12 spectrum — it only saw 83972 and ATCC25922 as examples of "Non-STEC E. coli." Then we rotate through all 9 strains.

Biologically: this is the much more honest question. *Has the model actually learned what "STEC-ness" looks like as a chemical signature, or has it just memorized each strain's individual fingerprint?* A real deployment would constantly see new isolates the lab has never characterized before — LOSO is the closest in-corpus proxy for that.

**Result: ~60% mean per-strain recall, best case. This is the wall.**

---

## 5. What we actually learned about these bacteria from the modeling

Even though no model fully solves LOSO, the *pattern of failure* is biologically interesting. Different model families crack different strains. Here's the per-strain best-model table — read it as "this strain has a chemical signature shaped like the kind of pattern that *this* family of models is good at":

```
STRAIN                CLASS         BEST MODEL                        RECALL
───────────────────────────────────────────────────────────────────────────────
83972                 Non-STEC      PLS-DA (linear/chemometrics)        1.00
ATCC25922             Non-STEC      Transformer (small patches)        ~1.00
K-12                  Non-STEC      DANN-trained CNN                    0.75
O103:H2               STEC          PLS-DA / DANN                      ~0.89
O121:H19              STEC          2-channel CNN (SNV + derivative)    0.89
O157:H7               STEC          DANN-trained CNN                    0.78
Dublin                Salmonella    PLS-DA                              0.7+
Heidelburg            Salmonella    PLS-DA                              0.7+
Typhimurium           Salmonella    PLS-DA (only model > 0)            ~0.56
```

A loose biological reading of that table:

- **The Salmonella triplet and 83972 are recognized best by a *linear* model (PLS-DA).** That implies their LOSO-relevant signal lives in straightforward peak-ratio differences — exactly what PLS-DA is built to exploit. Encouraging chemistry-wise: there appears to be a low-rank "this is Salmonella" subspace in the fingerprint region.
- **K-12 only works when we adversarially strip out file-level / instrument-level artefacts (DANN).** The interpretation we landed on: K-12's distinguishing signal lives in broad-scale chemistry that's the same band the instrument's batch effects live in. Naive models lock onto the batch effect; the adversarial training forces the model to find the underlying chemistry.
- **O121:H19 is best recognized when we explicitly hand the model the 2nd-derivative spectrum** (which sharpens peaks and removes broad baselines). That's consistent with O121:H19 having distinctive *narrow* features (sharp peaks) rather than broad envelope differences.
- **Typhimurium is the hardest single strain.** Only PLS-DA gets meaningfully above chance, and even then only to ~0.56. The chemistry overlap with Dublin and Heidelberg is apparently very tight in this corpus.
- **The "STEC vs Non-STEC E. coli" boundary is genuinely hard.** Same species, different toxin phenotype. The model that's best at K-12 (Non-STEC) is also the model that's best at O157:H7 (STEC) — suggesting a common "remove instrument noise, look at the real chemistry" recipe is what's needed for the *E. coli*-vs-*E. coli* call, regardless of toxin status.

**The headline finding for a biology audience is the per-strain table itself.** It says: *the chemical signatures of these strains are real and distinguishable, but each one lives in a different feature subspace.* No one model captures all of them.

---

## 6. Why this is hard for ML models — five reasons

### 6.1 Nine strains is not very many

Every LOSO fold is "given two strains as examples of this class, classify a third." Imagine you're being asked to recognize a Pekingese as a dog after only ever seeing a Great Dane and a Greyhound. The within-class diversity is large; the training-side coverage is tiny. There's a real ceiling here that no architecture can sidestep.

Concretely: 3 strains per bacterial class. Hold one out → train on 2. Mean is a 9-point statistic regardless of how many pixels feed into each point.

### 6.2 One instrument, one lab, one prep

Every spectrum came from the same Raman microscope. So **whatever signal the model learns, some of it is "this is what bacteria look like *on this rig, with this preparation protocol*"** — not "this is what bacteria look like in general." We have no way to test cross-lab transfer from this corpus alone. A diagnostic probe we built ("can a simple classifier guess which *file* a spectrum came from, just from the deep model's internal features?") still fires at ~14% (vs ~1% by chance) even after our most aggressive adversarial training. The model can still tell files apart by their instrument fingerprint, even when we're actively training it not to.

### 6.3 The "minority-of-one" problem

For some strains, exactly one model family in our stack recognizes them. K-12 is only solved by DANN. Typhimurium is only solved by PLS-DA. We tried 5 different schemes for combining models into a single ensemble — soft-voting, stacking, confidence-routing, temperature-scaled averaging, per-strain selectors — and **none of them outperform PLS-DA alone**. The mechanism is: when 3 of 4 models are confidently wrong on a strain, averaging cannot rescue the 1 model that's right.

Practical implication: there is *probably no single model* you can deploy that recognizes all 9 strains well at LOSO. There's a per-strain mosaic of best models, and the obvious "just ensemble them" trick doesn't compose.

### 6.4 Within-species genomic similarity translates to within-species spectral similarity

This is the real biology bottleneck. STEC and Non-STEC E. coli are the same species. Their Raman spectra are nearly identical. The *only* signal distinguishing them in spectroscopic data is whatever downstream consequences the Shiga-toxin and Locus-of-Enterocyte-Effacement carriage have on the cell's chemistry — cell-wall composition shifts, protein expression differences, possibly membrane lipid differences. Those signals exist (the model finds them) but they are small relative to:

- Strain-individual variation within the same toxin phenotype.
- Growth-phase variation within the same strain.
- Instrument variation within the same file.

The hierarchy of variance is *roughly* "instrument > strain identity > toxin phenotype." We're asking the model to recognize the smallest term while being robust to the larger two.

### 6.5 Random initialisation matters more than you'd expect

For our deep models, training the same architecture from 5 different random starting points gives 5 different LOSO scores spread over a ±0.15 range (on a 0–1 scale). That means **a single run of a deep model is not a reliable estimate of its performance.** We learned this the hard way — our original "ship this" deep-model headline at LOSO 0.50 turned out to be a lucky seed; the honest 5-seed average was 0.35. A different λ setting that originally looked worse (0.45) turned out, under multi-seed averaging, to actually be the better one (0.45). The lesson is operational: report 5-seed averages, not single-run numbers.

---

## 7. The realistic ceiling

Putting all of section 6 together, a fair statement is:

> Our **best single-number LOSO performance is ~0.60 mean per-strain recall**, achieved by PLS-DA, a classical chemometric model that's been the gold standard in the Raman / FTIR field for 30+ years. Modern deep models match this on some strains and beat the classical model on individually-hard strains (K-12, O157:H7, ATCC25922) — but none of them, individually or in combination, beats 0.60 on the mean.

Whether 0.60 is "close to the true ceiling" or "a lot of headroom left" is the central open question. Three things suggest it's close-ish:

1. The per-strain table covers each strain's best-case with a different architecture — i.e., we've probed the data from multiple angles, not just optimized one.
2. The minority-of-one strains are a fundamental information-theoretic constraint: if only one base model contains the right signal for a strain, you cannot combine it with three wrong models without losing the signal.
3. Ensemble methods, which empirically lift performance on most ML benchmarks, *failed across the board* on this dataset.

And three things suggest there's still headroom:

1. We've not yet tested any model against external Raman data from a *different* lab on *the same strains* (specifically ATCC25922, which is publicly available in two other Raman corpora). That's the highest-leverage remaining experiment.
2. Multi-seed soft-vote of deep models is a strict win over single-seed reporting — there's probably more to extract here.
3. Self-supervised pretraining on a large unlabelled Raman corpus is a known recipe in the field for small-dataset classification problems. We haven't tried it (it's out of scope for the take-home).

---

## 8. What more data would buy us

If we could redo the data collection from scratch, in priority order:

1. **More strains per class.** 3 → 6 or more. This is the single biggest lever. Every additional strain per class makes LOSO a meaningfully better estimator and exposes the model to more of the within-class chemical diversity.
2. **A second instrument / second lab.** Even one external batch would let us measure cross-instrument transfer directly. Right now we have no way to separate "the model learned bacteria" from "the model learned this particular Raman microscope's quirks on bacteria."
3. **More biological replicates per strain.** We have ~9 files per strain, but those are probably 1–4 distinct biological replicates with the rest being technical (re-imaged) replicates. Honest within-strain variability is undersampled.
4. **Open-set negatives.** Right now the model has seen STEC, Non-STEC E. coli, Salmonella, and water — nothing else. A real food-safety deployment will encounter *Listeria*, *Campylobacter*, *Pseudomonas*, environmental contaminants, residual food matrix, etc. The model has no idea what to do with any of those.
5. **Missing STEC serotypes / Salmonella serovars.** Only 3 of the ~7 clinically-important STEC serotypes are represented (no O26, O45, O111, O145). Only 3 Salmonella serovars (no Enteritidis, Newport, Infantis). Generalization to unrepresented serotypes/serovars is unmeasured.

Detailed accounting of public data we could pull, with the strain-match-or-not for each candidate dataset, lives in [`plan/12_data_gaps_and_external_datasets.md`](plan/12_data_gaps_and_external_datasets.md).

---

## 9. What this project does *not* claim

To stay honest:

- **We do not claim cross-instrument generalization.** Untested.
- **We do not claim cross-serotype generalization within STEC.** The three serotypes we have are not necessarily representative of all clinical STEC.
- **We do not claim open-set robustness.** The model has never seen a non-target organism and will probably misclassify one confidently.
- **We do not claim biological-replicate-level robustness beyond what 9 files per strain can tell us.**
- **The 95% on Test A is mostly a function of file-level structure**, not strain-level chemistry — it reflects that two random pixels from the same file are easy to associate, which is a much weaker claim than "the model knows what STEC is."

---

## 10. If you read nothing else

- **Dataset: 87 files, 9 bacterial strains across 4 primary classes (STEC E. coli, Non-STEC E. coli, Salmonella, H₂O), ~7K Raman spectra after QC, one lab, one instrument.**
- **Easy test (random file holdout): ~95% accuracy. Solved.**
- **Hard test (leave-one-strain-out): ~60% best mean per-strain recall. This is probably close to what's achievable with 3 strains per class on a single instrument.**
- **Different model families recognise different strains.** That's both the most interesting finding and the explanation for why ensembling failed: no single combination scheme captures the per-strain mosaic.
- **The bottleneck is not the model; it is strain count, lab diversity, and biological replicates.** More architectural work has diminishing returns. More *data*, particularly external-lab Raman of the same or related strains, is the highest-leverage next step.

For the ML-side review of the same material (architectures, training details, experiment log): [`REVIEW.md`](REVIEW.md).
For the deep status / experiment-by-experiment record: [`plan/00_status.md`](plan/00_status.md) and [`plan/07_findings.md`](plan/07_findings.md).
