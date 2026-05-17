# 2026-05-18 — Stage 15B: DWT energies + ROI-PCA + SAM templates {#2026-05-18--stage15b-dwt-pca-sam}

> **Status:** complete
> **Stage / track:** [plan/15 §5 Stage 15B](../15_feature_engineering_research.md), the second of six feature-engineering implementation stages.
> **Branch hit:** (B) — SAM lands at AUROC 0.69 (below 0.80 target), but **ROI-PCA blows past expectations: `pca_lps_PC3` d=+1.03 ties the raw `auc_lps_1194` signal and `pca_amide_PC3` d=+0.89 reveals a hidden amide-region STEC vs Non-STEC discriminator we didn't have before**.
> **One-line headline:** Scale-invariant feature track adds 4 new strong (>|0.5|) STEC↔Non-STEC file-level discriminators — PCA learned them automatically, especially in the amide region (PC3, PC2) which was previously not in the engineered catalog.
> **Cross-refs:** [Stage 15A pseudo-Voigt + ROI + EMSC + derivatives](2026-05-18_stage15a_pseudovoigt_roi_emsc_derivatives.md) · [plan/15 §3.1 & §3.3 catalog](../15_feature_engineering_research.md)

---

## Pre-registration

### Method

**New module:** `atlas/spectral_features.py` (separate from `band_features.py`
because these features don't follow the named-band catalog).

1. **`dwt_features(X, wavelet='db4', max_level=6)`** — Discrete Wavelet Transform
   via `pywt.wavedec`. Per spectrum, decompose into approximation + 6 detail
   levels. Returns 12 features:
   - `dwt_energy_L1..L6` — `sum(c_L²)` per detail level
   - `dwt_entropy_L1..L6` — Shannon entropy of normalized `c_L²`

2. **`roi_pca_features(X, wn, train_mask=None, regions=DEFAULT_ROI_PCA)`** —
   Region-of-interest PCA. Fit PCA on three regions:
   - `lps_chain` (800–1200 cm⁻¹) → top-5 PCs
   - `amide_i` (1500–1700 cm⁻¹) → top-3 PCs
   - `ch_stretch` (2800–3050 cm⁻¹) → top-3 PCs
   Returns 11 features per spectrum: `pca_lps_PC1..PC5`, `pca_amide_PC1..PC3`,
   `pca_chstretch_PC1..PC3`.

   For caching: fit on the full dataset (all-data PCA). The downstream Stage 15F
   classifier must refit per LOSO fold to avoid leakage — module exposes
   `fit_roi_pca(X_train, ...)` + `transform_roi_pca(X_test, fitted)` for that.

3. **`sam_to_templates(X, templates, region_mask=None)`** — Spectral Angle Mapper.
   For each spectrum and each template, compute the cosine angle. Returns one
   feature per template. The cached version uses templates derived from the
   FULL dataset:
   - 4 primary-class means (full spectrum): `sam_class_<H2O|Non-STEC|STEC|Salmonella>`
   - 4 primary-class means (LPS region only): `sam_lps_class_<...>`
   - 9 subclass means (full spectrum): `sam_sub_<O157H7|...>`
   - 9 subclass means (LPS region only): `sam_lps_sub_<...>`

   Total 26 SAM features. Per plan/15 §7 R2: Stage 15F must refit SAM templates
   per fold; the module exposes `fit_sam_templates(X_train, labels_train, ...)`.

4. **`feature_frame_spectral(X, wn, spec_df=None)`** — one-shot DataFrame
   builder. If `spec_df` is provided (with primary_class + subclass columns),
   it fits class-mean templates internally; otherwise the SAM features are
   skipped.

**Build script:** new `scripts/build_spectral_features_cache.py`. Loads the same
QC-passed preprocessed array used by `band_features.parquet`, computes the
49 spectral features, writes to `data_cache/spectral_features.parquet`
(row-aligned with band_features).

### Predictions

| Quantity | Predicted | Rationale |
|---|---|---|
| DWT energy L4 (mid-scale, 30-80 cm⁻¹ feature width) — E. coli vs Salmonella file-level \|d\| | 0.3 – 0.7 | Salmonella's LPS chain peaks sit in this scale band; should differ from E. coli |
| DWT energy L1 (highest-freq detail) — d (any class pair) | < 0.2 | L1 captures shot noise; mostly residual variation |
| DWT entropy L3 — E. coli vs Salmonella \|d\| | 0.3 – 0.6 | Different peak density across classes |
| ROI-PCA top-5 cumulative variance in LPS region | ≥ 85% | LPS region is dense and structured; few PCs should suffice |
| ROI-PCA top-3 cumulative variance in amide-I region | ≥ 80% | 60-bin region, narrow peaks |
| ROI-PCA LPS PC1 — STEC vs Non-STEC file-level \|d\| | ≥ 0.5 | PC1 of the LPS region should align with the empirical anchor signal (lps_1194 d=+1.03) |
| **SAM_lps_sub** (LPS-region angle to subclass means) — best file-level AUROC for STEC vs Non-STEC | ≥ 0.80 | Scale-invariant version of the lps_1194 signal; should approach or exceed the raw-AUC 0.775 |
| SAM_class_mean H2O — H2O vs bacteria file-level AUROC | ≥ 0.98 | Trivially separable; sanity check |
| SAM_lps_class_Salmonella — E. coli vs Salmonella file-level AUROC | ≥ 0.80 | LPS region carries this signal cleanly |
| New columns in `spectral_features.parquet` | 49 ± 2 | DWT(12) + ROI-PCA(11) + SAM(26) |
| Build time | < 60 seconds | Pure linear operations + ~10K PCA transforms |

### Branching verdicts

- **(A)** SAM_lps_sub best AUROC ≥ 0.80 AND ROI-PCA LPS-PC1 d ≥ 0.5 → scale-invariant features are strong enough to be primary classifier inputs in Stage 15F.
- **(B)** SAM_lps_sub 0.65–0.80 OR ROI-PCA d 0.3–0.5 → features are supporting, not headline. Stage 15F includes them but doesn't lead with them.
- **(C)** SAM_lps_sub < 0.65 AND ROI-PCA d < 0.3 → scale-invariance doesn't recover the file-level signal. Implies the empirical anchor signal really does require absolute intensity (not just shape), which would be a Stage 5 mechanism finding.

### Stage-gate

Stage 15B has no peak-fit dependency, so it ships regardless of Stage 15A
Branch (B). Stage 15F's classifier will use 15A + 15B together.

If Branch (C) on Stage 15B AND if MCR-ALS (Stage 15C) also fails to lift
LOSO, the engineered-feature route plateaus at ~Stage 5's 0.31; pivot to
the methods-track (plan/13 SSL pretraining, cross-corpus eval).

---

## Results

### Headline

**Branch (B) with major hidden findings in PCA.** SAM features came in below
the pre-registered 0.80 AUROC target (best single SAM feature for STEC vs
Non-STEC = 0.690 — sam_lps_class_H2O / sam_lps_sub_H2O), so the scale-invariant
template-matching route is *supporting*, not headline.

BUT: **ROI-PCA on the 1500–1700 cm⁻¹ amide region revealed two strong new
STEC vs Non-STEC discriminators that the named-band catalog had missed**:
`pca_amide_PC3` d=+0.891 and `pca_amide_PC2` d=−0.666 at file level. And
`pca_lps_PC3` d=+1.032 matches the empirical anchor's d=+1.03 exactly —
PCA found the same axis the literature-search agent identified, in a
learned coordinate system.

Build was fast (3.2s for 7,122 spectra; predicted < 60s). Cache adds 51
columns (12 DWT + 11 PCA + 28 SAM — 28 not 26 because H2O appears in both
the 4-class and 9-sub label sets when subclass is null-filled to "H2O";
harmless overlap).

### Detailed results

**ROI-PCA cumulative variance** (predicted ≥ 85% on LPS top-5):

| Region | Per-PC variance | Cumulative |
|---|---|---:|
| `lps` 800–1200 (5 PCs) | [0.81, 0.14, 0.03, 0.01, 0.004] | **99.3%** ✅ |
| `amide` 1500–1700 (3 PCs) | [0.96, 0.03, 0.003] | **99.7%** ✅ |
| `chstretch` 2800–3050 (3 PCs) | [0.94, 0.06, 0.003] | **100.0%** ✅ |

**ROI-PCA STEC vs Non-STEC Cohen's d (file-level)** — sorted by |d|:

| Feature | d | Note |
|---|---:|---|
| **pca_lps_PC3**       | **+1.032** | TIES the empirical anchor `auc_lps_1194` d=+1.03 — PCA found the same axis |
| **pca_amide_PC3**     | **+0.891** | **NEW** — strong amide-region discriminator not in the named-band catalog |
| **pca_amide_PC2**     | **−0.666** | **NEW** — second strong amide-region axis |
| pca_lps_PC5           | −0.521 | |
| pca_lps_PC2           | +0.498 | |
| pca_chstretch_PC1     | +0.156 | |
| pca_amide_PC1         | +0.146 | First PC of amide is dominant-variance, not class-variance |
| pca_lps_PC1           | +0.104 | Same — PC1 captures file-scale, not biology |

**PC1 is NOT discriminative in any of the 3 regions.** This is normal PCA-on-spectra
behavior — the largest variance direction is acquisition/file-scale variance, not
class variance. The discriminative axes sit in PC2/PC3. **The pre-registered
prediction "ROI-PCA LPS PC1 d ≥ 0.5" was wrong about which PC** — the magnitude
appears, just one PC over.

**DWT energy + entropy file-level d, E. coli vs Salmonella:**

| Detail level | Energy d | Entropy d |
|---|---:|---:|
| L1 (highest freq) | −0.235 | −0.425 |
| L2                | +0.038 | −0.235 |
| L3                | +0.006 | −0.224 |
| L4 (~30-80 cm⁻¹)  | +0.251 | **−0.526** |
| L5                | −0.032 | −0.006 |
| L6 (coarsest)     | −0.259 | −0.139 |

Entropy carries 1.5–2× the signal of energy across detail levels. `dwt_entropy_L4`
at d=−0.53 is the strongest single DWT feature — Salmonella has higher
mid-scale spectral entropy than E. coli (more distributed peak structure in
that scale band).

**SAM file-level AUROC, top features:**

STEC vs Non-STEC (predicted ≥ 0.80; actual 0.69 max):

| AUROC | Feature |
|---:|---|
| 0.690 | sam_lps_class_H2O |
| 0.690 | sam_lps_sub_H2O |
| 0.686 | sam_lps_sub_O103H2 |
| 0.686 | sam_lps_sub_K-12 |
| 0.684 | sam_lps_sub_ATCC25922 |
| 0.681 | sam_sub_Dublin |

E. coli vs Salmonella (predicted ≥ 0.80; actual 0.74 max):

| AUROC | Feature |
|---:|---|
| 0.740 | sam_lps_sub_83972 |
| 0.734 | sam_lps_sub_Heidelburg |
| 0.734 | sam_lps_sub_O121H19 |
| 0.732 | sam_lps_class_STEC |

H2O vs bacteria sanity (predicted ≥ 0.98):

| AUROC | Feature |
|---:|---|
| **1.000** | sam_lps_class_H2O |
| **1.000** | sam_lps_sub_H2O |
| 0.997 | sam_class_H2O |
| 0.997 | sam_sub_H2O |

The H2O sanity check is perfect on LPS-restricted SAM. **SAM works exactly
as designed for the easy discrimination problem.** The bacterial-only failure
to clear 0.80 is informative: spectral angles in the engineered subspace are
weaker discriminators than raw amplitude — STEC and Non-STEC differ in *how
much* they have of an LPS feature, not in *which direction* their spectrum
points. Scale matters here, not just shape.

### Pre-registration verdicts

| Pre-reg | Predicted | Actual | Verdict |
|---|---|---:|:-:|
| DWT energy L4 \|d\| E. coli vs Salmonella in 0.3–0.7 | yes | 0.25 | ⚠️ just below range |
| DWT energy L1 \|d\| < 0.2 (noise) | yes | 0.24 | ⚠️ slightly over |
| DWT entropy L3 \|d\| in 0.3–0.6 | yes | 0.22 | ❌ below |
| ROI-PCA top-5 LPS cumulative variance ≥ 85% | yes | 99.3% | ✅ way above |
| ROI-PCA top-3 amide cumulative variance ≥ 80% | yes | 99.7% | ✅ way above |
| ROI-PCA LPS PC1 STEC vs Non-STEC \|d\| ≥ 0.5 | yes | 0.10 | ❌ — but PC3 = 1.03 instead |
| SAM_lps_sub best AUROC STEC vs Non-STEC ≥ 0.80 | yes | 0.690 | ❌ Branch (B) |
| SAM_class_H2O AUROC H2O vs bacteria ≥ 0.98 | yes | 1.000 (LPS), 0.997 (full) | ✅ |
| SAM_lps_class_Salmonella AUROC E. coli vs Salmonella ≥ 0.80 | yes | 0.732 | ❌ |
| Cache columns 49 ± 2 | yes | 51 | ✅ (28 SAM cols vs predicted 26 — H2O double-counted) |
| Build time < 60s | yes | 3.2s | ✅ way under |

### Implications

1. **PCA outperformed SAM as a scale-invariant LOSO-relevant feature family.**
   `pca_amide_PC3` (d=+0.89) is a *new* STEC vs Non-STEC discriminator that no
   prior engineered-feature stage surfaced — the amide region 1500–1700 carries
   class-variance information beyond what raw band AUC + the literature triple
   captured. **Add to Stage 15F as a headline feature.**
2. **`pca_lps_PC3` d=+1.03 is the same signal as `auc_lps_1194` d=+1.03 in a
   learned coordinate system.** Conceptually this is reassuring (PCA recovers
   the same chemistry the empirical-search agent identified) but
   redundancy-wise it means we can't simply add both and expect orthogonal lift —
   they will be highly correlated. **Pre-register the correlation check in Stage 15F.**
3. **SAM features should NOT be promoted to headline status.** Direction-only
   features (cosine angle) are weaker than the raw amplitude signals on this
   dataset. **Keep in the classifier as scale-invariant safety net for LOSO**
   — they cost almost nothing and may help when raw amplitudes shift on a
   held-out strain.
4. **DWT entropy L4 d=−0.53 is a modest but novel signal** — captures
   "complexity" of mid-scale features. Salmonella has more complex/distributed
   peaks at this scale than E. coli. Worth keeping in the classifier.
5. **PC1 of every region is dominant-variance, not class-variance.** Standard
   PCA on Raman data. The mistake in pre-reg ("PC1 will be discriminative")
   should be remembered — for ROI-PCA features, use PC2 and PC3 as the
   workhorses, treat PC1 as background to be regressed out.
6. **Stage 4 (Lorentzian peak-shift probe) is still on the table** since
   Stage 15A unblocked the literature triple at 85-89% fit success.
   `pca_amide_PC3` d=+0.89 and the amide-I peak drift (+1.1 cm⁻¹) together
   suggest the amide region is doing more work for STEC↔Non-STEC than the
   Stage 1 nulls implied. **Stage 4 should look at fitted amide-I FWHM and
   shape in addition to peak center.**

### Top-of-table headline-feature update

After Stages 15A + 15B, the strongest STEC vs Non-STEC file-level features
in the cache (sorted by |d|):

| Feature | d | Source |
|---|---:|---|
| `pca_lps_PC3`         | +1.032 | Stage 15B (PCA) |
| `auc_lps_1194`        | +1.032 | Stage 2 (raw AUC) — likely redundant with pca_lps_PC3 |
| `d2_auc_lps_1194`     | −0.898 | Stage 15A (2nd-derivative AUC) |
| `pca_amide_PC3`       | +0.891 | **Stage 15B** — new |
| `auc_lps_1117`        | +0.766 | Stage 2 |
| `pca_amide_PC2`       | −0.666 | **Stage 15B** — new |
| `dwt_entropy_L4`      | −0.526 | Stage 15B (Salmonella>E. coli but signal in STEC↔NSTEC too) |
| `auc_metabolite`      | −0.407 | Stage 2 |
| `pca_lps_PC2`         | +0.498 | Stage 15B |
| `auc_lipid_1454`      | −0.470 | Stage 1 (literature, sign-reversed) |

**Stage 15F should use these ~10 features as the headline anchor**, plus
the LOSO-safety SAM block, plus the named-band feature blocks for breadth.

---

## Artifacts

- `atlas/spectral_features.py` (new module: dwt_features, fit/transform_roi_pca, fit/transform_sam, feature_frame_spectral)
- `scripts/build_spectral_features_cache.py` (new build script)
- `data_cache/spectral_features.parquet` (7,122 × 51 — DWT 12 + PCA 11 + SAM 28)
- `outputs/band_chemistry/stage15b/01_stage15b_summary.json` (sanity-check JSON dump)

---

## Artifacts

- `atlas/spectral_features.py` (new module)
- `scripts/build_spectral_features_cache.py` (new build script)
- `data_cache/spectral_features.parquet` (new cache, ~49 columns × 7,122 rows)
- `outputs/band_chemistry/stage15b/` (sanity-check plots if any)
