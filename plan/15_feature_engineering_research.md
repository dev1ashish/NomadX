# 15 — Feature Engineering Research

> **Mutability:** stable. Catalog and prioritization live here. Implementation
> outputs (per-feature findings, classifier results) go in `07_findings.md`.
>
> **Last updated:** 2026-05-18.
>
> **Companion docs:** [`14_band_chemistry_research.md`](14_band_chemistry_research.md)
> (the band-chemistry track that produced the Stage 5 Branch (C) result and
> motivates this doc), [`13_methods_research_synthesis.md`](13_methods_research_synthesis.md)
> (MCR-ALS / SSL / cross-corpus methods track), [`11_references.md`](11_references.md)
> (published bands + performance ceilings).

---

## 1. Why this doc exists

Stage 5 of the band-chemistry track ([07§stage5-band-classifier](07_findings.md#2026-05-17--stage5-band-classifier))
hit Branch (C) at LOSO mean parent-recall = 0.312, despite Protocol-A
file-F1 = 0.870 with 13 engineered features. Two structural problems were
visible:

1. **Feature space too narrow.** The 13 anchor features all hit the same
   discriminative axis (LPS 800–1200 cm⁻¹). Stage 5's STEC-default class
   bias (Stage 7 finding [07§stage7-mixed-sample](07_findings.md#2026-05-18--stage7-mixed-sample))
   showed Non-STEC files live in a low-density region of the 13-D feature
   space — any perturbation pushes them into "more like STEC" territory.
2. **Feature engineering didn't exploit biology beyond bulk macromolecule
   AUCs.** Cytochromes, secondary structure, peak-shift markers, PHB,
   Tyr/Trp doublets — none of these biology-grounded directions were in
   the feature set. The literature triple was the only biology-anchored
   feature group, and it's null at file-level.

Plan/15 is the systematic survey of what features we *should* be extracting
but aren't. Three research subagents (2026-05-18) covered:
(a) spectral / signal-processing features,
(b) biology / chemistry-grounded markers,
(c) data-driven / template-matching / cross-pixel features.

This document is the synthesis: complete catalog, prioritization, and
implementation roadmap.

---

## 2. Taxonomy

Four orthogonal axes. Most features sit in one cell; some span two.

|   | **Spectral processing** | **Biology / chemistry** | **Data-driven** | **Spatial / cross-pixel** |
|---|---|---|---|---|
| **Per-spectrum** | derivatives, wavelets, moments, peak shape, FFT, EMSC scatter coefs | named-band AUCs, ratios, peak-fit center/FWHM, conformation markers | PCA / PLS scores, NMF coefficients, spectral angle, sparse coding, dictionary learning | — |
| **Per-file** | aggregate stats over pixel-spectra (mean, std, kurtosis) | macromolecule group scores, mean primary triple, etc. | MCR-ALS component concentrations, file-mean PLS scores | within-file variance per band, GLCM texture on intensity maps, Moran's I, outlier-pixel count |
| **Cross-corpus** | EMSC reference choice, harmonic-baseline params | published-band lookup tables | Bacteria-ID anchor angles, RRUFF substrate templates | — |

**LOSO-relevance axis** cross-cuts the table:
- **Scale-invariant features** (SAM, cosine similarity, EMSC coefficients,
  derivative-based AUCs, ratios) — robust to per-file intensity drift, expected
  to help LOSO.
- **Absolute-magnitude features** (raw band AUCs, macromolecule sums) — sensitive
  to file-level scale, more useful for Protocol-A than LOSO.
- **Topology-/shape-features** (peak shapes, wavelet energies, GLCM) — generally
  scale-invariant.

---

## 3. Master feature catalog

~100 candidate features across the four directions. The **EV column** is
expected value relative to the existing 13-feature anchor set: 🟢 high, 🟡
medium, 🔴 low. **LOSO column**: ✅ scale-invariant (good for LOSO), ⚠️
absolute-magnitude. **Small-data**: ✅ K params ≤ 87 files, ⚠️ may overfit.

### 3.1 Spectral processing (Agent 1)

| ID | Feature | Definition | Library | EV | LOSO | Cost |
|---|---|---|---|:-:|:-:|---|
| SP1  | D1_AUC_band (×8 bands) | `∫ |y'|` over band±10 cm⁻¹; `y'=savgol(y, 11, 3, deriv=1)` | scipy | 🟢 | ✅ | O(N) |
| SP2  | D2_AUC_band (×8) | same with `deriv=2`; sharpens shoulders | scipy | 🟢 | ✅ | O(N) |
| SP3  | D2_negpeak_band (×8) | `min(y'')` in band — 2nd-deriv trough = Raman peak | numpy | 🟢 | ✅ | O(N) |
| SP4  | D1_zero_crossings_LPS | count sign changes of `y'` in 800–1200 | numpy | 🟡 | ✅ | O(N) |
| SP5  | DWT_energy_L1..L6 | `sum(c_L²)` per detail level from `pywt.wavedec(y, 'db4', level=6)` | pywt | 🟢 | ✅ | O(N) |
| SP6  | DWT_entropy_L1..L6 | Shannon entropy of normalized coefs²  | pywt + scipy | 🟢 | ✅ | O(N) |
| SP7  | DWT_coef_band_L2..L4 | sum of \|coeffs\| whose support overlaps each named band | pywt | 🟡 | ✅ | O(N) |
| SP8  | CWT_ridge_count_LPS | # CWT ridges from mexh wavelet localized in 800–1200 | pywt + scipy | 🟡 | ✅ | O(N·S) |
| SP9  | CWT_energy_scale_2/4/8/16 | `sum(|CWT[s,:]|²)` per scale | pywt | 🟡 | ✅ | O(N·S) |
| SP10 | ROI_mean (×6 ROIs) | mean intensity per region (400-800, 800-1200, 1200-1500, 1500-1700, 1700-2800, 2800-3050) | numpy | 🔴 | ⚠️ | O(N) |
| SP11 | ROI_std (×6) | std per region | numpy | 🟡 | ✅ | O(N) |
| SP12 | ROI_skew (×6) | `scipy.stats.skew` per region | scipy | 🟡 | ✅ | O(N) |
| SP13 | ROI_kurt (×6) | `scipy.stats.kurtosis` per region | scipy | 🟡 | ✅ | O(N) |
| SP14 | ROI_centroid (×6) | `sum(wn·y)/sum(y)` — "center of mass" of region | numpy | 🟢 | ✅ | O(N) |
| SP15 | ROI_entropy (×6) | Shannon entropy of `y/sum(y)` per region | scipy | 🟡 | ✅ | O(N) |
| SP16 | ROI_flatness (×6) | gmean/amean of intensities (Wiener entropy) | scipy | 🟡 | ✅ | O(N) |
| SP17 | FFT_low_energy | sum of \|FFT(y)\|² in lowest 10% bins | numpy | 🔴 | ⚠️ | O(N log N) |
| SP18 | FFT_high_energy | sum in top 30% (noise estimate) | numpy | 🔴 | ⚠️ | O(N log N) |
| SP19 | FFT_dominant_period | wavelength of largest non-DC FFT peak | numpy | 🔴 | ⚠️ | O(N log N) |
| SP20 | Voigt_sigma (×8 bands) | Gaussian width from pseudo-Voigt fit | lmfit | 🟢 | ✅ | per-spectrum |
| SP21 | Voigt_gamma (×8) | Lorentzian width from pseudo-Voigt fit | lmfit | 🟢 | ✅ | per-spectrum |
| SP22 | Voigt_eta (×8) | Gauss/Lorentz mixing fraction | lmfit | 🟢 | ✅ | per-spectrum |
| SP23 | Voigt_asym_a (×8) | Korepanov asymmetry parameter | custom (~30 lines) | 🟢 | ✅ | per-spectrum |
| SP24 | Peak_shoulder_index (×8) | (peak_h − valley_h) / peak_h | scipy.signal | 🟡 | ✅ | O(N) |
| SP25 | EMSC_coef_a,b,c,d | offset + ref-coef + linear + quadratic baseline coefs | biospectools | 🟢 | ✅ | per-spectrum |
| SP26 | MSC_offset, MSC_slope | simpler scatter correction params | rampy | 🟡 | ✅ | per-spectrum |
| SP27 | SAM_class_mean_k (×4) | spectral angle to class-mean template | numpy | 🟢 | ✅ | O(N) |
| SP28 | SAM_LPS_class_mean_k (×4) | SAM restricted to 800–1200 cm⁻¹ | numpy | 🟢 | ✅ | O(N) |
| SP29 | Cosine_class_mean_k (×4) | redundant with SAM but linear-model-friendly | numpy/sklearn | 🟡 | ✅ | O(N) |
| SP30 | GLCM_features_1D (×12) | contrast, homogeneity, energy, correlation at offsets 1-3 | skimage | 🔴 | ✅ | O(N) |
| SP31 | PCA_LPS_PC1..PC5 | top-5 PCs from PCA on 800–1200 cm⁻¹ subset | sklearn | 🟢 | ✅ | per-fold fit |
| SP32 | PCA_amideI_PC1..PC3 | top-3 PCs from PCA on 1600–1700 | sklearn | 🟡 | ✅ | per-fold fit |
| SP33 | PCA_CHstretch_PC1..PC3 | top-3 PCs from PCA on 2800–3050 | sklearn | 🟡 | ✅ | per-fold fit |

### 3.2 Biology / chemistry (Agent 2)

Cross-referenced with `[[atlas-raman-bands]]` and plan/14 §2 catalog.
Recipes use `±10 cm⁻¹` window default.

| ID | Feature | Bands | Biology | EV | LOSO | Note |
|---|---|---|---|:-:|:-:|---|
| BIO1  | Cyt_pyrrole_ratio | 752 / 1004 | heme cyt-c vs Phe protein | 🟢 | ✅ | resonance enhanced at 405–633 nm; weaker at 785 nm |
| BIO2  | Cyt_oxidation_state | 1356 / 1372 | Fe²⁺/Fe³⁺ proxy | 🟢 | ✅ | |
| BIO3  | Cyt_center_1585 | Lorentzian center of 1585 | b-type heme Cα-Cβ | 🟢 | ✅ | center drift between 1582 (ox) and 1588 (red) |
| BIO4  | Cyt_total | 750 + 1127 + 1585 | bulk cyt loading | 🟢 | ⚠️ | |
| BIO5  | Carotenoid_score | 1525 + 1155 | C=C + C-C polyene | 🟡 | ✅ | likely null on Atlas (non-pigmented) |
| BIO6  | Carotenoid_norm | (1525 + 1155) / 1450 | normalized by bulk CH₂ | 🟡 | ✅ | |
| BIO7  | PG_glycosidic | 897 | NAG/NAM glycosidic linkage | 🟡 | ✅ | |
| BIO8  | PG_ratio | 1080 / 1004 | carbohydrate vs protein | 🟡 | ✅ | |
| BIO9  | PE_marker | 880 / 1080 | phosphatidylethanolamine NH₃⁺ rocking | 🟡 | ✅ | speculative; published assignments contested |
| BIO10 | PG_marker | 1063 + 1130 | phosphatidylglycerol C-O | 🟡 | ✅ | |
| BIO11 | Cardiolipin_proxy | 1080 × 1130 / 1450² | joint-occurrence × normalization | 🔴 | ✅ | speculative |
| BIO12 | Acyl_unsaturation | 1655 / 1440 | C=C vs CH₂ membrane fluidity | 🟡 | ✅ | overlaps amide-I |
| BIO13 | Lipid_A_acyl | 1735 / 1450 | LPS Lipid A ester C=O | 🟡 | ✅ | confound with PHB (BIO27) |
| BIO14 | Core_phosphate | 815 / 1080 | asym/sym PO₂⁻ in LPS core | 🟡 | ✅ | confound with DNA conformation |
| BIO15 | O_antigen_rhamnose | 975 | rhamnose ring (O157 has it) | 🟡 | ✅ | |
| BIO16 | O_antigen_sialic | 1632 | sialic acid (O104 marker) | 🟡 | ✅ | testable hypothesis |
| BIO17 | Fit_center_1090 | Lorentzian fit center | sub-bin O-antigen drift | 🟢 | ✅ | already in plan/14 §6.6 — gated on fit fix |
| BIO18 | NA_A_form_fraction | 815 / (815+835) | C3'-endo vs C2'-endo sugar pucker | 🟢 | ✅ | conformation, not composition |
| BIO19 | RNA_DNA_ratio | 813 / 788 | RNA-specific vs DNA-specific | 🟡 | ✅ | growth-phase covariate too |
| BIO20 | Alpha_helix_score | 1652 / 1670 | amide-I α/β ratio | 🟢 | ✅ | Williams 1986 method |
| BIO21 | Beta_sheet_amide3 | 1232 / 1270 | amide-III β/α | 🟢 | ✅ | |
| BIO22 | Amide_FWHM_1655 | Lorentzian fit FWHM | broader = more disorder | 🟡 | ✅ | gated on fit fix |
| BIO23 | Metabolic_load | 725 + 750 + 1280 | NADH + FAD + ATP cofactors | 🔴 | ⚠️ | growth-phase covariate; use to deconfound not as feature |
| BIO24 | PHB_carbonyl | 1730 | polyhydroxybutyrate ester C=O | 🟢 | ✅ | K-12 falsifier — should be ~0 in non-PHB cells |
| BIO25 | PHB_score | 1730 × 1058 / 1450² | joint PHB markers | 🟢 | ✅ | |
| BIO26 | Tyr_doublet_ratio | 850 / 830 | Tyr OH H-bond environment | 🟢 | ✅ | classic Raman protein descriptor (Siamwiza 1975) |
| BIO27 | Trp_content | 759 + 1552 | Trp indole bands | 🟢 | ✅ | |
| BIO28 | Trp_indole_env | 1340 / 1360 | Trp W7 hydrophobicity | 🟡 | ✅ | |
| BIO29 | Virulence_AA_sig | Trp / Phe | (759+1552) / 1004 | 🟢 | ✅ | STEC vs Non-STEC test based on virulence-protein composition |
| BIO30 | O_antigen_fingerprint | AUC(450..850 in 50 cm⁻¹ bins) | 8-D sub-region histogram | 🟡 | ✅ | feed as feature block |

### 3.3 Data-driven / template-matching (Agent 3)

All trained features must be **fit inside the CV fold** to avoid leakage.

| ID | Feature | Definition | Library | EV | LOSO | Small-data |
|---|---|---|---|:-:|:-:|:-:|
| DD1  | MCR-ALS_C_summary (×4K, K=6-8) | mean, std, max, p90 of each MCR concentration column | pyMCR | 🟢🟢 | ✅ (global S fit) | ✅ |
| DD2  | MCR_biology_sum | sum AUC of "biology" components after manual tagging | pyMCR + curation | 🟢🟢 | ✅ | ✅ |
| DD3  | NMF_global_K20_mean | mean W[k] across pixels per file, K=20 | sklearn.NMF | 🟢 | ✅ | ✅ |
| DD4  | NMF_LPSregion_K5 | NMF restricted to 800–1200 cm⁻¹, K=5 | sklearn | 🟢 | ✅ | ✅ |
| DD5  | NMF_guided_K8 | NMF with partial init from reference spectra | nimfa | 🟢 | ✅ | ✅ |
| DD6  | NMF_sparse_K30 | sparse-NMF with L1 penalty | sklearn | 🟡 | ✅ | ✅ |
| DD7  | PLS-DA_scores_K10 | PLS scores from PLSRegression(n=10) on 9-class one-hot | sklearn | 🟢🟢 | ✅ (per-fold) | ✅ |
| DD8  | PLS-DA_LPS_scores_K5 | PLS in 800–1200 only, K=5 | sklearn | 🟢 | ✅ | ✅ |
| DD9  | PCA_full_K10 | top-10 PCs on full spectrum, train-fold fit | sklearn | 🟡 | ✅ | ✅ |
| DD10 | PCA_LPS_K5 (mean+std per file) | already covered as SP31, listed for cross-ref | — | 🟢 | ✅ | ✅ |
| DD11 | PCA_residual_energy | reconstruction error after PCA-10 | sklearn | 🟡 | ✅ | ✅ |
| DD12 | SAM_class_mean_9 | SAM to each of 9 subclass means (not 4 — finer) | numpy | 🟢🟢 | ✅ (per-fold) | ✅ |
| DD13 | SID_class_mean_9 | spectral information divergence to subclass means | numpy | 🟡 | ✅ | ✅ |
| DD14 | SAM_LPS_subclass_9 | SAM in 800–1200 to subclass means | numpy | 🟢🟢 | ✅ | ✅ |
| DD15 | BacteriaID_anchor_30 | angle to ~30 species means from Bacteria-ID public corpus | numpy + Bacteria-ID download | 🟢 | ✅ (external ref) | ✅ |
| DD16 | File_pixel_variance_LPS | mean variance across pixels in 800–1200 region | numpy | 🟢 | ✅ | ✅ |
| DD17 | File_pixel_variance_CH | mean variance in 2800–3000 region | numpy | 🟡 | ✅ | ✅ |
| DD18 | File_pixel_kurt_anchor (×3) | kurtosis at lps_1050/1117/1194 | scipy | 🟡 | ✅ | ✅ |
| DD19 | Moran_I_lps_1194 | spatial autocorrelation of intensity at 1194 | pysal | 🟡 | ✅ | ✅ |
| DD20 | GLCM_lps_1194 (×5) | contrast, homogeneity, energy, correlation, entropy on 2D intensity map | skimage.feature | 🟡 | ✅ | ⚠️ small grids noisy |
| DD21 | GLCM_lps_1117 (×5) | same on 1117 map | skimage | 🟡 | ✅ | ⚠️ |
| DD22 | GLCM_lps_1050 (×5) | same on 1050 map | skimage | 🟡 | ✅ | ⚠️ |
| DD23 | GLCM_aa_1004 (×5) | same on 1004 map | skimage | 🟡 | ✅ | ⚠️ |
| DD24 | Spatial_gradient (×4 bands) | mean \|∇I\| via Sobel filter on band intensity maps | skimage | 🟡 | ✅ | ⚠️ |
| DD25 | PCA_outlier_count | count pixels with Mahalanobis > 99th percentile | sklearn | 🟡 | ✅ | ✅ |
| DD26 | IsolationForest_frac | fraction of pixels flagged outlier | sklearn | 🔴 | ✅ | ✅ |
| DD27 | ConvexHull_AUC | area between spectrum and lower convex-hull baseline | scipy.spatial | 🟡 | ✅ | ✅ |
| DD28 | ConvexHull_AUC_LPS | restricted to 800–1200 | scipy.spatial | 🟡 | ✅ | ✅ |
| DD29 | Spectrum_autocorr_peaks | first 3 secondary maxima of 1D autocorrelation | numpy | 🔴 | ✅ | ✅ |
| DD30 | Wavelet_scatter_J4Q8 | translation-stable scattering coefs (~100 → PCA-20) | kymatio | 🟡 | ✅ | ⚠️ overfit risk |
| DD31 | CWT_denoised_AUC (×8) | re-run integrate_band on CWT-denoised spectrum | scipy + numpy | 🟡 | ✅ | ✅ |
| DD32 | Dictionary_coef_K40 | sparse-coding coefs from learned dictionary | sklearn.DictionaryLearning | 🔴 | ✅ | ⚠️ slow fit |

---

## 4. Prioritization framework

Three filters applied to the ~95 candidates:

### 4.1 LOSO-relevance filter

**Pass** if the feature is scale-invariant (ratio/angle/normalized) OR
extracted from a shape descriptor (FWHM, asymmetry, position center).
**Fail** if the feature is an absolute-magnitude AUC alone.

This filter is strict. Per Stage 7's STEC-bias finding, absolute AUCs put
each class in a fixed feature-space region; LOSO shifts that region for
the held-out strain. Scale-invariant features cancel multiplicative file
drift and survive the shift.

### 4.2 Small-data safety filter

**Pass** if model parameters ≤ 87 (number of files) OR feature is
parameter-free. K=20 NMF (8,640 params for 987-bin basis × 20) is on the
edge; K=10 PLS (10×K_class) is safe.

### 4.3 Novelty filter (relative to current 13-feature anchor)

**Hit** if the feature carries discriminative axis not already in the anchor
set. Cytochromes, secondary structure, peak shape, MCR/NMF components are
all hits. Macromolecule group AUCs (already in cache) are not.

### 4.4 Recommended top-15 (passes all 3 filters + 🟢 EV)

| Rank | ID | Feature | Why |
|:-:|---|---|---|
| 1  | DD1  | **MCR-ALS concentration summaries** | Plan/13 already flagged as highest-EV; directly attacks substrate/fluor leakage that memprobe-v2 catches at 14% |
| 2  | DD7  | **PLS-DA scores K=10** | Chemometrics workhorse; supervised projection; ~20 features/file |
| 3  | DD14 | **SAM in 800–1200 to subclass means** (×9) | Scale-invariant within-LPS-region similarity |
| 4  | SP25 | **EMSC scatter-correction coefficients** | 4 features per spectrum; physics-motivated; rampy/biospectools |
| 5  | SP31 | **PCA in LPS region, top-5** | Optimal linear projection of the discriminative region |
| 6  | BIO20 | **Alpha-helix score** (amide-I 1652/1670) | Williams-1986 method; ratio; interpretable in protein 2°-structure vocabulary |
| 7  | BIO24 | **PHB carbonyl** (1730) | K-12 falsifier — currently no model explains K-12; PHB accumulation is a plausible mechanism |
| 8  | BIO1  | **Cyt pyrrole ratio** (752/1004) | Cytochrome content normalized to Phe; STEC/Salmonella differ in cyt-bd expression |
| 9  | BIO26 | **Tyr doublet ratio** (850/830) | Single most-cited Raman protein descriptor; H-bond environment |
| 10 | SP1+SP2 | **D1 + D2 AUC per band** (×8+8) | Baseline-invariant; sharpens peak structure |
| 11 | DD4  | **NMF in LPS region, K=5** | Lightweight unsupervised decomposition of the empirical anchor region |
| 12 | BIO29 | **Trp/Phe ratio** (Virulence_AA_sig) | STEC virulence proteins are Trp-rich; testable specifically for STEC↔Non-STEC |
| 13 | SP14 | **ROI spectral centroid** (×6 ROIs) | Captures sub-bin peak shifts the fixed-AUC missed (Stage 1 finding) |
| 14 | DD16 | **File pixel-variance in LPS region** | Cross-pixel within-file feature; strain-heterogeneity signature; uses existing spec_df |
| 15 | SP5+SP6 | **DWT energy + entropy** (×12) | Multi-scale features; baseline-invariant; small-data safe |

**Total new features in top-15: ~85** (with multi-feature IDs counted by their
band/scale/component count). Down from ~95 candidates after filtering.

### 4.5 Deferred (good ideas, lower priority)

- **DD15 Bacteria-ID anchor angles** — high LOSO value but needs external data download (plan/12 track)
- **DD30 Wavelet scattering** — overfit risk on 87 files unless PCA-reduced
- **DD20-23 GLCM on intensity maps** — too noisy on small (≤15×15) grids
- **SP30 1D GLCM** — niche, low literature support
- **BIO5-6 carotenoids** — likely null on this dataset (non-pigmented cultures)
- **BIO9-12 PE/PG/CL membrane lipids** — published assignments are contested
- **DD32 dictionary learning** — slow fit, marginal vs NMF

### 4.6 Stage 2 fix dependency

Plan/14 Stage 4 was gated on improving Lorentzian fit success rate (currently
0.2–37%). Plan/15 adopts the **pseudo-Voigt + linear baseline + bounded params**
fix from Agent 1 (SP20–SP23) before the BIO17, BIO22, BIO3 features can be
extracted reliably. This is a **prerequisite for Stage 4 and for any feature
that uses peak fitting** (BIO3, BIO17, BIO22).

---

## 5. Implementation roadmap

### Stage 15A — Pseudo-Voigt fix + ROI / shape / scatter features (~1 day)

Module: extend `atlas/band_features.py` with:
- `fit_peak_pseudovoigt(spec, wn, center, window=30)` — replaces `fit_peak` Lorentzian
- `roi_moments(X, wn, regions=DEFAULT_ROIS)` → 6 ROIs × {mean, std, skew, kurt, centroid, entropy} = 36 features
- `emsc_correct(X, ref_spectrum)` returning corrected X + 4 EMSC coefficients per spectrum
- `derivative_band_auc(X, wn, deriv=1|2, band_keys)` — SP1, SP2

**Deliverable:** updated band_features.parquet with ~50 new columns (current
81 + new 50 = 131). Verify pseudo-Voigt fit rate ≥ 80% on the 3 empirical
anchor bands (lps_1050, 1117, 1194) and the literature triple — same
acceptance bar as plan/08 Stage 2 minus the failure.

**Smoke check:** the per-class Trp/Phe ratio (BIO29) should be ≥0.5 SD
higher in STEC vs Non-STEC if the literature claim holds.

### Stage 15B — Wavelet, PCA-in-region, spectral angle (~1 day)

New module: `atlas/spectral_features.py` (separate from `band_features.py`
because these are non-band features).

- `dwt_features(X, wavelet='db4', max_level=6)` → SP5, SP6
- `roi_pca_scores(X, wn, regions, n_components_per_region)` → SP31, SP32, SP33
- `sam_to_templates(X, templates)` → SP27, SP28, DD12, DD14
- `bacteria_id_anchors(X, anchor_templates)` → DD15 (deferred to Stage 12 if external data download is ready)

**Deliverable:** per-fold-cached SAM templates (since they require train-only
class means). Returns ~40 new features.

### Stage 15C — MCR-ALS unmixing (~2 days, highest single-feature EV)

New module: `atlas/unmix_features.py`.

- `mcr_als_fit(X_train_per_pixel, n_components=8)` → returns (S, init metadata)
- `mcr_als_project(X_test, S)` → returns C per pixel
- `mcr_als_concentration_summary(C, file_ids)` → DD1 features per file (mean, std, max, p90 per component)

**Caveat:** MCR-ALS needs proper initialization (SIMPLISMA) and convergence
checking. Allocate 0.5 day for pyMCR setup and 1 day for component
inspection + biology tagging (DD2 manual step).

**Deliverable:** ~32 new MCR features per file. Re-run Stage 5 classifier
with the augmented feature set. **Pre-commit success bar: LOSO mean ≥ 0.45**
(half-way between current 0.31 and the original 0.55 bar — MCR is expected
to help substantially per plan/13 §2.3).

### Stage 15D — Biology-specific features (~0.5 day)

Extend `atlas/band_features.py` with:
- `protein_secondary_structure(X, wn)` → BIO20, BIO21, BIO22 (depends on Stage 15A fix)
- `cytochrome_features(X, wn)` → BIO1, BIO2, BIO3, BIO4
- `phb_features(X, wn)` → BIO24, BIO25
- `aromatic_aa_features(X, wn)` → BIO26, BIO27, BIO28, BIO29
- `nucleic_conformation_features(X, wn)` → BIO18, BIO19

**Deliverable:** ~15 new biology-grounded features.

### Stage 15E — Cross-pixel / spatial features (~0.5 day)

New module: `atlas/spatial_features.py` (uses x_um, y_um, grid_nx, grid_ny
already in `RawRecord`).

- `pixel_variance_per_region(X_per_pixel, wn, file_ids)` → DD16, DD17
- `pixel_kurtosis_at_band(X_per_pixel, file_ids, wn, band_centers)` → DD18
- `morans_i_spatial(intensity_map, weights_matrix)` → DD19 (uses pysal)

**Deliverable:** ~10 new spatial features.

### Stage 15F — Re-train classifier with full feature set (~1 day)

Re-run Stage 5 XGBoost + LogReg + Ensemble on the full ~130-feature set.
Use:
- Feature importance ranking to identify which new features are load-bearing
- Per-class SHAP values to verify Non-STEC bias is reduced (Stage 7 finding)
- Multi-seed runs (5 seeds) for honest variance reporting

**Pre-committed targets:**
- LOSO mean parent-recall ≥ 0.50 (improvement from 0.31)
- Non-STEC LOSO recall ≥ 0.25 (currently 0.07 on LogReg, 0.31 on XGB)
- STEC default-class bias rate at α=0.5 mixtures ≤ 0.55 (currently 0.44–0.72)

If the augmented classifier hits LOSO ≥ 0.55, **revisit Stage 6 (3-channel CNN)**
which was skipped per Stage 5 stage-gate.

---

## 6. Module extensions

### Existing: `atlas/band_features.py`

Already implements: BANDS catalog, integrate_band, integrate_region,
macromolecule_vector, band_ratios, fit_peak (Lorentzian — to be replaced),
feature_frame.

Extensions per Stage 15A and 15D above. Backwards-compatible: existing
`feature_frame()` signature unchanged, new columns appended at the end.

### New: `atlas/spectral_features.py` (Stage 15B)

```python
def dwt_features(X: ndarray, wavelet: str = 'db4',
                 max_level: int = 6) -> dict[str, ndarray]
def roi_pca_scores(X: ndarray, wn: ndarray, regions: dict,
                   n_components_per_region: dict) -> dict[str, ndarray]
def sam_to_templates(X: ndarray, templates: ndarray) -> ndarray
def emsc_correct(X: ndarray, reference: ndarray) -> tuple[ndarray, dict]
def feature_frame_spectral(X, wn, **kwargs) -> pd.DataFrame
```

### New: `atlas/unmix_features.py` (Stage 15C)

```python
class MCRALSWrapper:
    def fit(self, X_per_pixel: ndarray, n_components: int = 8): ...
    def transform(self, X_new: ndarray) -> ndarray: ...
    @property
    def pure_spectra(self) -> ndarray: ...

def mcr_concentration_summary(C: ndarray, file_ids: ndarray) -> pd.DataFrame
def nmf_features(X, n_components, region: tuple[float, float] | None = None)
```

### New: `atlas/spatial_features.py` (Stage 15E)

```python
def pixel_variance_per_region(X_per_pixel, wn, file_ids, regions) -> pd.DataFrame
def morans_i_for_band(intensity_map: ndarray, coords: ndarray, weights: ndarray) -> float
def glcm_texture_features(intensity_map: ndarray, levels: int = 16) -> dict
```

### Master integration: `atlas/all_features.py` (Stage 15F)

```python
def build_full_feature_frame(X_per_pixel, wn, spec_df, **fit_results) -> pd.DataFrame
```

Combines outputs from `band_features`, `spectral_features`, `unmix_features`,
`spatial_features` into the single ~130-column DataFrame used by the Stage 15F
classifier.

---

## 7. Risk analysis

### R1 — Feature explosion overfits to 87 files

130 features ÷ 87 files = 1.5 features per file. Even with file-level
training, this is curse-of-dimensionality territory. **Mitigation:**
mutual-information feature selection per fold, target 30–40 features after
filtering. Tracked in plan/14 §9.

### R2 — Per-fold feature-fitting leaks

PCA, PLS, NMF, MCR-ALS, SAM templates all must be fit on **train fold only**.
Already-noted in plan/03 §G evaluation rules. Add explicit assertion in
each new module: fit methods raise if test fold rows are passed.

### R3 — Pseudo-Voigt fit success may not actually reach 80%

The Lorentzian's 0.2–37% rate may not be a model-form problem but a
preprocessing-residual problem (arPLS boundary artifact, SNV bringing peaks
near zero). **Mitigation:** profile failure modes first; if curvature is
fine but baseline is wrong, fix the preprocessing not the fit model.

### R4 — MCR-ALS components don't separate biology from substrate

Plan/13 §2.3 cites successful bacterial-MCR papers but those used different
instruments / preprocessing. **Mitigation:** Stage 15C reserves 1 day for
manual component inspection. If MCR doesn't separate cleanly, fall back to
guided-NMF (DD5).

### R5 — Some biology features don't show signal at 785-nm excitation

Cytochromes are weak off-resonance; carotenoids are likely null on
non-pigmented cultures. **Mitigation:** plan/15 §4.5 already deferred these
to second tier. Include them in the classifier and let feature importance
decide.

### R6 — Spatial features need ≥200 pixels per file

Some Atlas files have ~144 pixels (12×12 grids). GLCM and Moran's I are
noisy at this size. **Mitigation:** apply spatial features only to files
with `grid_nx × grid_ny ≥ 200`. Tracked per-file in plan/01.

---

## 8. Cross-references

- Stage 5 finding that motivates this doc: [07§stage5-band-classifier](07_findings.md#2026-05-17--stage5-band-classifier).
- Stage 7 STEC-bias finding: [07§stage7-mixed-sample](07_findings.md#2026-05-18--stage7-mixed-sample) — drives the "broaden Non-STEC feature representation" mandate.
- Plan/14 catalog of named bands is the input to all biology features (BIO1-BIO30).
- Plan/13 §2.3 MCR-ALS prioritization is operationalized here as Stage 15C.
- Plan/12 Bacteria-ID dataset is the prerequisite for DD15 anchor features.
- Memory: [[atlas-raman-bands]] (band assignments), [[atlas-briefing-emphasis]] (user-highlighted priorities), [[atlas-band-chemistry-roadmap]] (stage tracker).

---

## 9. Open questions for further research

1. **Is the LOSO crater a feature-engineering problem or a labeled-data problem?**
   Plan/15 commits to one more round of feature engineering. If Stage 15F
   doesn't push LOSO above 0.50, the answer is "labeled-data problem" and we
   stop adding features.
2. **Is the STEC-default class bias an artifact of class-imbalance or of feature
   geometry?** Could be either. Class-weighted XGBoost is a one-line test.
3. **Would deep learning over the raw 987-bin spectrum learn these features
   implicitly?** The vanilla CNN says no (LOSO 0.35); but a Transformer with
   patch_size=5 (existing model) hit 1.00 on ATCC25922 — it learned something
   the engineered features didn't. **Stage 15F results should be cross-compared
   with the patch=5 Transformer's per-strain table.**
4. **Are we hitting a Bayesian information-theoretic ceiling?** Per
   `tang-2026-wgan` the realistic ceiling on cross-strain transfer is ~94% with
   thousands of training spectra and modern WGAN augmentation. We have 87 files.
   Stage 15F's gain may be measured against an asymptote, not against unbounded
   improvement.

---

## Future / not-now

- **Multi-modal**: combine Raman with mass-spec or genome-derived features per file. Plan/09 §future_work.
- **Active learning**: route low-confidence files to human review per plan/09.
- **Per-strain submodels**: separate classifier per O-antigen serogroup if MCR-ALS surfaces serogroup-specific spectra.
- **Cross-strain feature transfer**: train SAM templates on one strain set, evaluate transfer to another. Operationalizes "do the engineered features describe biology that generalizes?"
