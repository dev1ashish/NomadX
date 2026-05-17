# 14 — Band-chemistry research plan

> **Mutability:** stable. The catalog of bands, macromolecule groupings, and
> targeted analyses lives here. New experimental findings go into
> `07_findings.md`; new method ideas accumulate at the bottom of this file as a
> small "Future" subsection.
>
> **Last updated:** 2026-05-17.
>
> **Companion docs:** [`11_references.md`](11_references.md) (published bands +
> performance ceilings), [`04_eda_plan.md`](04_eda_plan.md) (notebook block
> ordering), [`13_methods_research_synthesis.md`](13_methods_research_synthesis.md)
> (the MCR-ALS / SSL / cross-corpus track this complements).

---

## 1. Why this doc exists

The existing pipeline reads intensity at fixed wavenumber bins (PCA on full
spectrum, boxplots at 1004/1450/1660/2900, ANOVA-F per bin). It does **not**:

- aggregate intensity into per-macromolecule scores (protein / NA / lipid / LPS),
- fit peak centers — so sub-bin O-antigen shifts between O157/O121/O103 are invisible,
- extract peak shape (FWHM, asymmetry, 2nd-derivative sharpness),
- compute band ratios (batch-effect-robust features),
- run analyses restricted to chemistry-relevant sub-regions (e.g. LPS 800–1200 cm⁻¹).

Plan/13 closed the methodological loop on **representation** (MCR-ALS unmixing,
VICReg SSL, N-PLS); this doc closes the loop on **chemistry-grounded features
and analyses** — i.e. squeezing the published Raman literature into the model
inputs and the writeup vocabulary, not just the references section. The
specific bands and macromolecule groupings come from a user-supplied briefing
(2026-05-17), cross-checked against
[`cisek-2013`](11_references.md#cisek-2013--cisek-et-al-analyst-2013-sensitive-and-specific-discrimination-of-pathogenic-and-nonpathogenic-escherichia-coli-using-raman-spectroscopy)
and
[`yuan-2024-salmonella`](11_references.md#yuan-2024-salmonella--yuan-et-al-j-cell-mol-med-2024-rapid-discrimination-of-four-salmonella-enterica-serovars).

---

## 2. Canonical band catalog

Wavenumbers are in cm⁻¹. "Source" column: `briefing` = user 2026-05-17, `cisek`
= Cisek 2013, `yuan` = Yuan 2024. All bands lie inside the current preprocessed
range (400–3049 cm⁻¹) and most inside the fingerprint subset (400–1900 cm⁻¹).

### 2.1 Master table

| cm⁻¹ | Assignment | Macromolecule group | Source | Discrimination role |
|---:|---|---|---|---|
| 487  | (data-driven peak from existing class means)            | unassigned          | EDA  | candidate; reassign on inspection |
| 599  | (data-driven peak from existing class means)            | unassigned          | EDA  | candidate |
| 616  | COO⁻ wag                                                | metabolite          | yuan | Salmonella serovar split |
| 720–780 | ring breathing (A, G, T)                             | nucleic_acid        | cisek/briefing | bacterial NA content |
| 762  | aromatic AA (Trp ring)                                  | aromatic_aa         | briefing | protein signal |
| 786  | ring breathing (DNA/RNA U,C,T)                          | nucleic_acid        | briefing | NA; also assigned aromatic_aa in some sources — keep flagged as **dual** |
| 790  | (data-driven; near 786)                                 | nucleic_acid        | EDA  | likely same as 786 line at our resolution |
| 831  | aromatic AA (Tyr Fermi)                                 | aromatic_aa         | briefing | Tyr signature |
| 855  | aromatic AA (Tyr)                                       | aromatic_aa         | briefing | Tyr/Phe |
| 925  | C-C skeletal                                            | metabolite          | yuan | Salmonella serovar split |
| 1004 | Phe ring breathing                                      | aromatic_aa         | cisek/briefing | classic protein anchor; intensity is total-protein proxy |
| 1014 | aromatic AA (Phe / Trp)                                 | aromatic_aa         | briefing | protein |
| 1050 | (data-driven peak from existing class means)            | unassigned          | EDA  | candidate; possibly C-O / C-C of carbohydrate |
| 1080 | C–C / C-O / PO₂⁻ symmetric stretch                      | lipid_carbohydrate  | briefing | phospholipid backbone |
| 1176 | aromatic AA (Tyr / Phe)                                 | aromatic_aa         | briefing | protein |
| 1212 | aromatic AA (Tyr / Phe C-C ring)                        | aromatic_aa         | briefing | protein |
| 1242 | amide III                                               | protein_amide       | briefing | β-sheet/random-coil protein backbone |
| 1335 | CH₂/CH₃ wag                                             | nucleic_acid        | briefing | adenine-related |
| **1338** | CH₂ wagging / adenine ring                          | nucleic_acid        | cisek    | **STEC↔non-STEC primary discriminant — stx1/stx2 NA signal** |
| 1362 | nucleic acid (G ring)                                   | nucleic_acid        | briefing | NA |
| 1451 | CH₂ deformation                                         | lipid_carbohydrate  | briefing | lipid + carb |
| **1454** | CH₂/CH₃ deformation                                | lipid_carbohydrate  | cisek    | **STEC↔non-STEC primary discriminant — lipid/carb content** |
| 1485 | guanine ring                                            | nucleic_acid        | briefing | NA |
| 1486 | guanine ring                                            | nucleic_acid        | yuan     | Salmonella serovar split |
| 1530 | nucleic acid                                            | nucleic_acid        | briefing | NA |
| 1542 | C=C                                                     | metabolite          | yuan     | Salmonella serovar split |
| 1555 | (data-driven peak from existing class means)            | nucleic_acid (likely) | EDA  | inspect |
| 1575 | NA bases (purines)                                      | nucleic_acid        | briefing | NA |
| 1585 | lipid / carb / NA (overlap)                             | lipid_carbohydrate  | briefing | mixed |
| 1617 | aromatic AA (Trp)                                       | aromatic_aa         | briefing | Trp side chain |
| **1658** | amide I (β-sheet/random)                           | protein_amide       | cisek    | **STEC↔non-STEC primary discriminant — virulence-related protein bulk** |
| 1662 | amide I (α-helix)                                       | protein_amide       | briefing | protein 2° structure |
| 2850 | sym CH₂ stretch                                         | lipid_carbohydrate  | atlas EDA | lipid acyl chains |
| 2900 | C-H stretch (sym + asym)                                | lipid_carbohydrate  | atlas EDA | bulk lipid+protein; dominated by water-vs-bacteria boundary in our 4-class ANOVA |
| 2930 | asym CH₂ stretch                                        | lipid_carbohydrate  | atlas EDA | bulk lipid+protein |

### 2.2 Macromolecule grouping (5 groups + LPS-region anchor)

For per-spectrum biochemistry vectors, integrate intensity across these
band sets (sum or trapezoidal over ±10 cm⁻¹ around each named center):

| Group key            | Bands (cm⁻¹) | Interpretation |
|---|---|---|
| `aromatic_aa`        | 762, 831, 855, 1004, 1014, 1176, 1212, 1617 | protein side-chain content (Phe/Tyr/Trp) |
| `protein_amide`      | 1242, 1658, 1662 | protein backbone (amide III + amide I) |
| `nucleic_acid`       | 720, 760, 786, 1335, **1338**, 1362, 1485, 1486, 1530, 1575 | nucleic-acid bulk |
| `lipid_carbohydrate` | 1080, 1451, **1454**, 1585, 2850, 2900, 2930 | lipid acyl chains + carbohydrate |
| `lps_o_antigen`      | full 400–900 region treated as one band group (trapezoidal AUC) | outer-membrane polysaccharide fingerprint |

`lps_o_antigen` is the only group that uses a continuous integration region
(per briefing: 400–900 cm⁻¹ is treated as a single LPS fingerprint zone) rather
than discrete ±10 windows. Justification: published O-antigen polysaccharide
spectra are dense, overlapping peaks not assignable to single bands at our
resolution; integrated AUC over the region is the textbook reduction.

### 2.3 Targeted-discrimination "primary triple"

Per [`cisek-2013`](11_references.md#cisek-2013--cisek-et-al-analyst-2013-sensitive-and-specific-discrimination-of-pathogenic-and-nonpathogenic-escherichia-coli-using-raman-spectroscopy):

1. **1338 cm⁻¹** — NA (stx1/stx2 virulence gene signal)
2. **1454 cm⁻¹** — lipid/carb
3. **1658 cm⁻¹** — protein amide-I (virulence proteins: intimin, Shiga toxin)

These three are the only bands the literature pre-commits to as **STEC vs
non-STEC** discriminators. Every band-aware analysis below treats them as the
headline test bands; everything else is supporting biochemistry.

### 2.4 Data-driven additions

In addition to literature bands, the pipeline already surfaces peaks at 487,
599, 790, 1050, 1055, 1555 cm⁻¹ from `scipy.signal.find_peaks` on class
means ([07§block-5-top-peaks](07_findings.md)). These are kept in the catalog
as `unassigned`/likely-assignments pending data-driven re-inspection (see §6.2).
**The data-driven workflow is to run peak-finding on bacteria-only class means
(H₂O excluded) and any peak not already in the table that beats a prominence
threshold gets appended with `EDA` source.**

---

## 3. State of the pipeline today

### 3.1 What's captured

| Capability | Where | Status |
|---|---|---|
| Class means + p5/p95 bands (raw + processed) | `notebooks/atlas_driver.ipynb` block 2, 9.1; `notebooks/dataset_visualization.ipynb` per-file 04 | ✅ |
| Subclass means | atlas_driver block 8.1, 9; dataset_viz summary 04 | ✅ |
| Per-bin ANOVA-F + mutual info | atlas_driver block 6 | ✅ (4-class) |
| Per-bin boxplots at 1004 / 1450 / 1660 / 2900 | atlas_driver block 9.6 | ✅ (only 4 of ~25 bands) |
| PCA / UMAP latent space | atlas_driver 3, 4, 9.2-3 | ✅ but **uninterpreted in chemistry vocabulary** |
| Per-file mean-spectrum cosine similarity | atlas_driver 7, 9.4 | ✅ |
| Per-file overlay / heatmap / spatial / mean-band PNGs | `images/<class>/<sub>/<file_id>/0[1-4].png` | ✅ (this session) |

### 3.2 Gaps to close

| Gap | Why it matters | Where to fix |
|---|---|---|
| Sub-bin peak shifts | O157 vs O121 vs O103 differ in O-antigen polysaccharide structure (briefing). Reading intensity at a fixed bin is blind to a 1–3 cm⁻¹ peak-center drift. | §5 `atlas/band_features.py` Lorentzian fit |
| Macromolecule scores | Per-spectrum 5-dim biochemistry vector lets us say "STEC has higher NA, lower lipid" — interpretable in the same words the briefing uses. PCA can't do this. | §5 `band_features.aucs_per_group` |
| Band ratios | Ratios cancel multiplicative file-level offsets — far more batch-effect-robust than raw intensities. Calibration-date PCA in 9.7 showed file batch is real on this dataset. | §5 `band_features.ratios` |
| LPS-region focus | E. coli vs Salmonella diff is concentrated in 800–1200 (LPS chain). Whole-spectrum PCA dilutes this. | §6.4 LPS-region PCA & LDA |
| Bacteria-only ANOVA | Current 4-class ANOVA top bins are all in C-H stretch — the water-vs-bacteria boundary, not bacterial subtype. [§anova-bins-vs-stec-discriminative-bands](07_findings.md#2026-05-14--anova-bins-vs-stec-discriminative-bands) flagged this; still open in plan/00 "Open items". | §6.2 |
| Peak-shape descriptors (FWHM, asymmetry) | β-sheet vs α-helix amide-I differ in width, not center. | §5 Lorentzian fit returns FWHM |
| Explicit "primary triple" plots | The 1338/1454/1658 trio gets one mention in findings — no dedicated plot or feature. | §6.3 |

---

## 4. Research questions (pre-registered)

These are the questions the band-chemistry track answers. Each one gets a
pre-registered prediction in `08_expectations.md` **before** the corresponding
notebook block runs, per repo norm. Predictions are not in this file — this file
just locks the question.

| RQ | Hypothesis to test | Test |
|---|---|---|
| RQ1 | At 1338/1454/1658, STEC and Non-STEC distributions differ significantly (≥10% mean difference, Welch t and Mann-Whitney both p<0.01 after FDR). | Per-class violin/strip plot + statistical tests on integrated intensities |
| RQ2 | The protein-amide:NA ratio (1658/1338) and lipid:protein ratio (1454/1658) separate STEC from Non-STEC more cleanly than any single band. | Scatter plot of these two ratios, colored by class; report best-1D AUC |
| RQ3 | At the O-antigen region (400–900 LPS AUC), Salmonella separates from E. coli cleanly (AUC>0.85), but the 3 STEC serotypes (O157/O121/O103) split internally only via sub-bin Lorentzian-fitted peak centers. | LPS-region AUC + Lorentzian center scatter |
| RQ4 | A bacteria-only (H₂O-excluded) ANOVA surfaces the published 1338/1454/1658 trio inside top-30 discriminative bins (it does *not* currently — top bins are all in 2880–2940). | Re-run ANOVA on bacteria-only subset; tabulate top 30 |
| RQ5 | A classical model fit on the 5-dim macromolecule vector + 6 ratios + 3 fitted peak centers (~14 features total) hits LOSO mean parent-recall ≥ 0.55 — closer to PLS-DA solo (0.603) than to vanilla CNN (0.35). The promise is **interpretable parity**, not absolute lift. | XGBoost / LogReg on `band_features` only, both protocols |
| RQ6 | Adding `band_features` as additional input channels to the 2-channel CNN does **not** improve LOSO (information already in spectrum) but **does** improve calibration / shrink variance across seeds. | 3-channel CNN run, multi-seed |

RQ1, RQ4 are the cheap wins (statistical, no new training). RQ5 is the
interpretability swing. RQ6 is the "are we adding new info or just rephrasing"
falsification check.

---

## 5. Module spec — `atlas/band_features.py`

New module. Signature, not implementation. The module exposes pure functions
over a `(spectra: ndarray[N, B], wn: ndarray[B])` interface so it composes with
both raw and preprocessed cache loads.

### 5.1 Public surface

```python
# atlas/band_features.py

BANDS: dict[str, BandSpec]       # canonical catalog from §2.1 + 2.2
MACROMOLECULE_GROUPS: dict[str, list[str]]   # §2.2 grouping
PRIMARY_TRIPLE: tuple[str, str, str] = ("nucleic_1338", "lipid_1454", "amide_1658")

def integrate_band(X, wn, center, half_width=10) -> ndarray[N]:
    """Trapezoidal AUC over [center-hw, center+hw]."""

def integrate_region(X, wn, lo, hi) -> ndarray[N]:
    """Trapezoidal AUC over [lo, hi]. For LPS 400-900."""

def macromolecule_vector(X, wn) -> dict[str, ndarray[N]]:
    """Return dict of per-spectrum AUC per group (aromatic_aa, protein_amide,
    nucleic_acid, lipid_carbohydrate, lps_o_antigen). Plus 'total' for
    normalization."""

def band_ratios(X, wn, pairs=None) -> dict[str, ndarray[N]]:
    """Compute named band-ratio features. Default pairs include:
      protein_amide/nucleic_acid   (1658/1338)
      lipid_carbohydrate/protein_amide (1454/1658)
      aromatic_aa/protein_amide   (1004/1658)  -- proxy for AA composition
      nucleic_acid/lipid_carbohydrate (1338/1454)
      lps_o_antigen/protein_amide  -- E. coli vs Salmonella
      786/1004                    -- NA/Phe classical ratio
    Caller can supply additional pairs."""

def fit_peak(X_row, wn, center, window=30, model="lorentz") -> PeakFit:
    """Fit a single Lorentzian/Gaussian/Voigt over wn ∈ [center-w, center+w].
    Returns PeakFit(center_fitted, height, fwhm, area, baseline_offset, rmse).
    NaN for any of these means fit failed (caller should fall back to integrate_band)."""

def fit_peaks_batch(X, wn, centers, **kwargs) -> ndarray[N, P, 4]:
    """Vectorized over spectra. centers: list of band centers in cm-1.
    Returns (N, P, 4) where last axis = (center, height, fwhm, area)."""

def feature_frame(X, wn, ratios=True, fits=True) -> pd.DataFrame:
    """Convenience: one-shot DataFrame of all band-aware features for each
    spectrum. Columns: auc_<group>, ratio_<num>_<den>, fit_<band>_<center|height|fwhm>.
    ~25-40 columns total. Stable column ordering for downstream caching."""
```

### 5.2 Caching

- `feature_frame` output cached to `data_cache/band_features.parquet` (one row
  per spectrum, joined on `spec_df` index).
- Lorentzian fit is the expensive call (~scipy.optimize per peak × ~6 peaks × 7K
  spectra = a few minutes). Cache invalidates on `wn_proc.npy` hash change.

### 5.3 Tests

- Synthetic Lorentzian (known center/FWHM/height) → fit recovers within 0.1%.
- Known band integration on flat unit signal → AUC ≈ width × 1 (sanity).
- Same band-ratio on a pure (1,1,...) signal → 1.0 (sanity).
- Class-mean smoke test: protein-amide AUC of H₂O class < any bacterial class.

### 5.4 Failure modes the module must handle

- Peak fit fails (curve_fit `RuntimeError`) → return NaN; downstream uses fallback `integrate_band`.
- Region includes a NaN bin (shouldn't happen after preprocessing/QC, but defensive) → mask and integrate over valid only.
- Band center outside `wn` axis → raise at validation, not silently 0.

---

## 6. Notebook spec — `notebooks/band_chemistry.ipynb`

New notebook. Builder via `nbformat` like the other notebooks. Saves figures to
`outputs/band_chemistry/` (new) and `images/_summary/`. Each section maps to a
research question from §4.

### 6.1 Section A — Annotated preprocessed spectra (already-buildable)

The visual deliverable for the current session. Stacked-panel figure of
preprocessed class & subclass means, common X-axis cropped to fingerprint
region (400–1900 cm⁻¹), with vertical lines at every band in §2.1 color-coded
by macromolecule group. Y-axis: "SNV intensity (post baseline + SG)" labeled
per panel. Saved to `images/_summary/07_annotated_preprocessed_spectra.png`
(headline) plus a second figure at `08_annotated_preprocessed_subclass.png`
(one panel per subclass, common X).

Two layouts produced from the same data:

- **Per-class panel** (4 rows): primary classes with subclasses overlaid in
  each panel, vertical band annotations on top axis with rotated wavenumber
  labels. Used as the "see the chemistry on the spectra" reference figure.
- **Per-subclass panel** (9 rows + H₂O at the bottom): one strain per axis,
  mean ± p5/p95, same band overlay. Used for visual side-by-side strain
  comparison in §6.7.

### 6.2 Section B — Bacteria-only ANOVA + MI

Closes plan/00 open TODO. Re-runs the per-bin ANOVA and mutual info on the
3 bacterial classes (STEC vs Non-STEC vs Salmonella). H₂O excluded so the
top bins are driven by bacterial-subtype variance, not water-vs-bacteria.

Outputs:
- `outputs/band_chemistry/02_anova_bacteria_only.png` (replaces the C-H-only top-30)
- Top-30 table (CSV) with each surviving band cross-referenced to §2.1 catalog
- Numeric test: do 1338, 1454, 1658 appear in top 30?

Pre-registered prediction goes in `08_expectations.md` before the run.

### 6.3 Section C — The primary triple (1338/1454/1658)

The deliverable for RQ1. For each of the three bands:

- Per-class violin + strip plot of integrated AUC (±10 cm⁻¹)
- Welch t (STEC vs Non-STEC) and Mann-Whitney U with Benjamini-Hochberg FDR
- Cohen's d effect size
- Per-strain breakdown (so we see whether the difference is driven by O157 alone or by all 3 STEC strains)

Output: `outputs/band_chemistry/03_primary_triple.png` (3×4 grid: 3 bands ×
{all-class violin, STEC-vs-Non-STEC violin, per-strain strip, ROC of best
single band)).

### 6.4 Section D — Macromolecule biochemistry vectors

Per-spectrum 5-dim vector (aromatic_aa, protein_amide, nucleic_acid,
lipid_carbohydrate, lps_o_antigen).

Plots:
- Per-class radar (5 axes, one polygon per primary class, plus dashed H₂O).
- Per-subclass radar grid (9 subclasses, hopes to show STEC vs Non-STEC differ
  on the protein/NA balance while Salmonella differs on LPS-region AUC).
- LPS-region AUC violin per class, with explicit E. coli (3 STEC + 3 Non-STEC,
  pooled) vs Salmonella t-test. Tests the "E. coli vs Salmonella is easy"
  briefing claim.

### 6.5 Section E — Band ratios

Six default ratios per §5.1. For each:
- Per-class boxplot
- Pairwise scatter of two most-informative ratios (color by class) — should
  produce a cleaner separation than any raw PC1×PC2.
- Best-1D AUROC per ratio for STEC vs Non-STEC.

### 6.6 Section F — Lorentzian fits (peak position + shape)

The sub-bin shift probe. Fit Lorentzian at every catalog band per spectrum;
extract (center, height, FWHM).

Plots:
- Dotplot: fitted peak center distribution per subclass, for each of the 3
  STEC serotypes' expected diagnostic LPS-region peaks. Looking for sub-bin
  drift in O157 vs O121 vs O103.
- FWHM distribution at 1658 (amide-I): β-sheet vs α-helix broadens the peak.
  Tests whether STEC virulence proteins shift the secondary-structure balance
  detectably.
- "Peak shift × strain" heatmap (rows = bands, cols = subclasses, color = mean
  fitted center − catalog center).

### 6.7 Section G — Targeted modeling (RQ5)

XGBoost and LogReg fit on `feature_frame` output only (the engineered features,
no raw spectrum). Run under Protocol A (StratifiedGroupKFold) and Protocol B
(LOSO). Headline numbers join the existing per-protocol tables in `07_findings.md`.

The pre-commit prediction: ≥ 0.55 LOSO mean. Anything ≥ 0.55 is a publishable
interpretability win; ≥ 0.60 is parity with PLS-DA on raw bins. Below 0.50
means the engineered features are losing information vs PLS-DA on full
spectrum — still useful as the *interpretable* version of "what the model
sees" but not a candidate ensemble member.

### 6.8 Section H — Hybrid: spectrum + band features

3-channel CNN: (SNV, 2nd-derivative, band-feature-broadcast). Per RQ6 we don't
expect a LOSO lift, only a calibration / multi-seed-variance improvement. Run
5 seeds for honest comparison with the existing DANN λ=0.1 (5-seed 0.370) and
DANN λ=0.3 (5-seed 0.448) baselines.

---

## 7. Implementation order

1. **(this session)** Section A figures — annotated preprocessed spectra, both
   layouts. No new module needed; uses cached preprocessed array directly.
   Saved to `images/_summary/`. Update `07_findings.md` with one observation
   per primary triple if visible in the figure.
2. `atlas/band_features.py` skeleton + tests. ~1 day. Cache to `data_cache/band_features.parquet`.
3. `notebooks/band_chemistry.ipynb` Sections B–E. ~1 day.
4. Section F (Lorentzian fits). ~0.5 day (slow vectorized scipy.optimize).
5. Pre-register predictions in `08_expectations.md` for RQ5, RQ6.
6. Section G targeted classical modeling. ~0.5 day. Append result to `07_findings.md`.
7. Section H hybrid CNN. ~0.5 day. Append result. **Only run if Section G is
   ≥ 0.50 LOSO** — otherwise the features aren't carrying enough new info to
   warrant the deep-model run.

Total budget: ~3–4 working days. RQ1, RQ4, the annotated figure, and the
bacteria-only ANOVA are the cheap deliverables; everything else compounds onto
that base.

---

## 8. Deliverables / artifacts

Files this plan will eventually produce:

| Artifact | Path | Section |
|---|---|---|
| Annotated preprocessed-spectra (per-class) | `images/_summary/07_annotated_preprocessed_spectra.png` | 6.1 |
| Annotated preprocessed-spectra (per-subclass) | `images/_summary/08_annotated_preprocessed_subclass.png` | 6.1 |
| Band catalog (machine-readable) | exposed in `atlas/band_features.BANDS` | 5.1 |
| Bacteria-only ANOVA top bins | `outputs/band_chemistry/02_anova_bacteria_only.{png,csv}` | 6.2 |
| Primary-triple panel | `outputs/band_chemistry/03_primary_triple.png` | 6.3 |
| Macromolecule radar per class | `outputs/band_chemistry/04_macromolecule_radar.png` | 6.4 |
| LPS-region E. coli vs Salmonella violin | `outputs/band_chemistry/04b_lps_region.png` | 6.4 |
| Band-ratio scatter | `outputs/band_chemistry/05_band_ratios.png` | 6.5 |
| Peak-center sub-bin dotplot | `outputs/band_chemistry/06_peak_shift.png` | 6.6 |
| FWHM at 1658 (β/α probe) | `outputs/band_chemistry/06b_fwhm_amide.png` | 6.6 |
| Engineered-feature classifier results | appended to `07_findings.md` | 6.7 |
| 3-channel CNN (multi-seed) | appended to `07_findings.md`; pred parquets to `outputs/preds/` | 6.8 |
| `band_features.parquet` cache | `data_cache/band_features.parquet` | 5.2 |

---

## 9. Risks and known limitations

1. **Band assignments are non-unique.** 786 cm⁻¹ shows up under both aromatic
   AA (Trp ring) and NA (DNA/RNA base ring) in different references. The catalog
   marks these as `dual`; the macromolecule grouping in §2.2 makes a single
   defensible call (NA), but the feature-frame caller can override by passing
   a custom group mapping.
2. **The literature ceilings cited in the briefing (>95%) are pure-culture
   intra-batch numbers.** Our LOSO problem is across-strain, not across-batch.
   The realistic ceiling per [`tang-2026-wgan`](11_references.md#tang-2026-wgan)
   is ~94% on independent test sets with thousands of training spectra and
   modern WGAN augmentation — we have 87 files. The 95% figure cannot be a
   pre-registered floor on this dataset; it's an upper bound. The band-chemistry
   track aims for **interpretability**, not absolute accuracy.
3. **K-12 will likely stay broken regardless of band features.**
   [`soupene-2003-k12`](11_references.md#soupene-2003-k12--soupene-et-al-j-bacteriol-2003-laboratory-strains-of-escherichia-coli-k-12-things-are-seldom-what-they-seem)
   documents K-12 as a laboratory-domesticated strain that has diverged from
   typical E. coli; the per-strain best-model table currently shows K-12 only
   recoverable via DANN λ=0.3 (and even there fragile across seeds —
   [00§multi-seed-dann-l01-robustness](00_status.md#multi-seed-dann-l01-robustness-session-summary-2026-05-15--major-revision-to-prior-headlines)).
   Band features should not be promised as a K-12 fix.
4. **Per-spectrum vs per-file features.** All band-aware features are
   per-spectrum (per-pixel). File-level aggregation for classifier input
   follows the existing scheme in `atlas/evaluate.py` (mean of per-pixel
   probabilities). Caller may also experiment with per-file-mean features
   directly — small effect expected but worth noting.
5. **Fluorescence baseline residuals.** arPLS-corrected spectra are flatter
   than raw but not perfectly flat at every band — small residual baselines
   can bias integrated AUCs. Mitigation: fit a 5–10 cm⁻¹ local linear baseline
   under each band before integration. Implementation hook lives in `integrate_band`.
6. **The cached preprocessed array currently crops at 400 cm⁻¹.** Plan/00
   "Open items" notes the arPLS boundary artifact at 400 cm⁻¹. The LPS-region
   AUC (400–900) is partially compromised by that artifact at its low edge.
   Either (a) crop at 450 for LPS specifically, or (b) wait until the arPLS
   re-crop lands. The annotated-spectra figure in Section A should clip its
   x-axis at 420 (not 400) until the underlying fix is in.

---

## 10. Cross-references

- Bands and references: [`11_references.md`](11_references.md) — especially
  `cisek-2013` (the 1338/1454/1658 trio), `yuan-2024-salmonella` (1486, 1542, 925, 616),
  `soupene-2003-k12` (why K-12 is hard).
- The MCR-ALS unmixing track in [`13_methods_research_synthesis.md`](13_methods_research_synthesis.md)
  is **complementary** to this one: MCR-ALS unmixes the spectrum into
  data-driven components; band-features uses literature-prescribed component
  groupings. Run both; compare the inferred pure-component spectra to the
  catalog in §2.1. If MCR-ALS components match the macromolecule groups, that's
  cross-validation; if they diverge, the divergence is the interesting finding.
- The existing open TODO in [`00_status.md`](00_status.md#open-items--todos)
  for "bacteria-only ANOVA" is subsumed by §6.2 of this plan.
- Per-strain best-model table currently lives in
  [`07_findings.md`](07_findings.md) — engineered-feature classifier and
  3-channel CNN results from §6.7/6.8 will be added as new rows.

---

## Future / not-now

Ideas that came up while writing this plan but should wait until the
band-feature classifier results land:

- **Per-band saliency** on the best DANN model. If saliency lands on catalog
  bands, the model is doing chemistry; if it lands at 2900 C-H, it's doing
  water-vs-bacteria.
- **Counterfactual band ablation**: zero out a single catalog band group's
  intensity, re-predict, measure accuracy drop. Quantifies how much each
  macromolecule group contributes to the model's decision.
- **Cross-corpus band-feature transfer.** Compute `feature_frame` on the Zhu
  cross-corpus dataset (see plan/13 §3 Zhu cross-corpus eval) and check whether
  ATCC25922 lands in the same band-feature region as our ATCC25922 files —
  cleaner test of "same biology, different instrument" than raw-spectrum
  cross-corpus eval.
- **Mixed-sample simulation.** Per briefing, mixed-sample accuracy typically
  drops 10–20% vs pure-culture. We can simulate by linearly mixing two
  per-file mean spectra at 0.1–0.9 ratios and measuring band-feature
  classifier degradation. Predictive of real-world food-matrix use case.
