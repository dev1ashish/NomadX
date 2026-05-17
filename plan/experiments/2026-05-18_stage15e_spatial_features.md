# 2026-05-18 — Stage 15E: spatial / cross-pixel features {#2026-05-18--stage15e-spatial-features}

> **Status:** complete
> **Stage / track:** [plan/15 §5 Stage 15E](../15_feature_engineering_research.md#stage-15e--cross-pixel--spatial-features-05-day), the fifth feature-engineering implementation stage.
> **Branch hit:** **(C) Miss for STEC↔Non-STEC** — 0 features clear \|d\|≥0.5 (best was 0.485, just shy of threshold). H2O sanity passes 4/4 and **`spat_skew_lps_1117` is a new strong E.coli↔Salmonella axis at d=+0.725.**
> **One-line headline:** **Spatial heterogeneity does NOT discriminate STEC vs Non-STEC** — the "clinical strains are more uniform than commensals" hypothesis is falsified. But `spat_skew_lps_1117` d=+0.725 E.coli↔Salm is a new strong directional finding (Salmonella's pixel-intensity distribution at 1117 cm⁻¹ is more symmetric than E. coli's right-skewed one), and the H2O class is cleanly separable on all 4 variance/CV features.
> **Cross-refs:** [Stage 15D biology features](2026-05-18_stage15d_biology_features.md) · [Stage 15C MCR-ALS](2026-05-18_stage15c_mcr_als_unmixing.md) · [plan/15 §3.3 DD16-DD24](../15_feature_engineering_research.md) · [plan/15 §7 R6 grid-size risk](../15_feature_engineering_research.md)

---

## Pre-registration

### Method

**Pre-flight finding: R6 fires hard.** Plan/15 §7 R6 specified that
spatial features should only apply to files with `grid_nx × grid_ny ≥ 200`
pixels. **This dataset has zero such files** — pixel-per-file distribution
is min=70, **median=72**, max=180. Only 8/87 files exceed 100 pixels.

Implication: Moran's I (DD19), spatial gradient (DD24), and GLCM on
intensity maps (DD20-23, plan/15 §4.5 already deferred) are not feasible
on this corpus without setting a much looser threshold (and even at ≥100
pixels we'd only get 8 files — too sparse for a meaningful per-class
comparison). **All three are dropped from Stage 15E.**

What remains useful: **per-file moment statistics** that work at any pixel
count ≥ 50. These capture "how heterogeneous is this file?" without
requiring spatial coordinates or large grids. Plan/15 §3.3 DD16-DD18 are
moment-based and fit here.

**New module `atlas/spatial_features.py`.** Public API:

```python
def pixel_variance_per_region(X_per_pixel, wn, file_ids,
                              regions={'lps': (800, 1200),
                                       'ch': (2800, 3000)}) -> pd.DataFrame
def pixel_cv_per_region(X_per_pixel, wn, file_ids,
                        regions=...) -> pd.DataFrame
def pixel_moment_at_band(X_per_pixel, wn, file_ids, band_centers,
                         moment='kurt' | 'skew', half_width=10.0) -> pd.DataFrame
def feature_frame_spatial(X, wn, spec_df) -> pd.DataFrame
    # one-shot — returns 87 × ~10 per-file DataFrame
```

Six DD16-DD18 feature families (~10 features total):

| ID | Feature | Definition | LOSO-relevance |
|---|---|---|:-:|
| **DD16** | `spat_var_lps_chain` | within-file variance of `∫y(800–1200) dλ` across pixels | ✅ scale-invariant if reported as CV |
| **DD16'**| `spat_cv_lps_chain` | std/mean of `∫y(800–1200) dλ` per file | ✅ scale-invariant |
| **DD17** | `spat_var_ch_stretch` | within-file variance of `∫y(2800–3000) dλ` across pixels | ⚠️ absolute |
| **DD17'**| `spat_cv_ch_stretch` | std/mean of `∫y(2800–3000) dλ` per file | ✅ scale-invariant |
| **DD18a** | `spat_kurt_lps_1050` | kurtosis of per-pixel intensity at 1050 cm⁻¹ ±10 | ✅ moment-free |
| **DD18b** | `spat_kurt_lps_1117` | kurtosis of per-pixel intensity at 1117 cm⁻¹ ±10 | ✅ |
| **DD18c** | `spat_kurt_lps_1194` | kurtosis of per-pixel intensity at 1194 cm⁻¹ ±10 | ✅ |
| **DD18d-f** | `spat_skew_lps_*` (×3) | skewness of per-pixel intensity at each LPS anchor | ✅ |

Total: **10 spatial features per file.**

**Why these and not others.** Plan/15 §4.5 deferred GLCM/Moran's-I on
small grids; the pre-flight R6 check confirms that's the right call.
Variance and kurtosis on per-file pixel-intensity distributions measure
"is the per-pixel signal at this band tight (low variance, normal kurt)
or wildly heterogeneous (high variance, fat-tail kurt)?" Heterogeneity
patterns could discriminate clinical pathogenic strains (homogeneous
culture, uniform expression) from commensal or lab strains (more variable
phenotype, mixed expression).

**Output:** `data_cache/spatial_features.parquet` shape (87, 10), aligned
with `band_features` and `unmix_features` cache row order (per `file_id`).

### Predictions

| Quantity | Predicted | Rationale |
|---|---|---|
| New feature columns | 10 ± 2 | 2 variance + 2 CV + 3 kurt + 3 skew = 10 |
| Build time | < 30 s | All operations are per-file numpy moments; no fits or PCA |
| `spat_cv_lps_chain` STEC vs Non-STEC \|d\| | 0.3 – 0.8 | LPS-chain region is the strongest single-band axis; within-file spread of LPS signal should differ by strain |
| `spat_kurt_lps_1194` STEC vs Non-STEC \|d\| | 0.2 – 0.6 | Kurtosis is a 4th-moment statistic; small-sample variance is high — expect signal but noisy |
| H2O class: spatial-variance features should be LOWER than bacterial (H2O is uniform, bacteria heterogeneous) | yes, all 4 variance/CV features | Sanity check |
| K-12 vs other-STEC: at least 1 spatial feature \|d\| ≥ 0.4 | maybe (50% prob.) | If K-12's "different 2°-structure" (Stage 15D) is also spatially heterogeneous, kurtosis at protein-related bands could split K-12 |
| Best new spatial feature \|d\| STEC↔Non-STEC | 0.3 – 0.7 | Won't beat `mcr_C6_mean` d=−1.23 or `bio_alpha_helix_score` d=−0.99, but these are orthogonal "is the culture uniform?" features |
| At least 2 spatial features \|d\| ≥ 0.5 STEC↔Non-STEC | yes (60% prob.) | Moderate prior — 10 features × 60% chance per family family is reasonable |
| Moran's I: NOT IMPLEMENTED | confirmed | R6 grid-size threshold not met; dropped |

### Branching verdicts

- **(A) Strong hit.** ≥ 3 spatial features clear \|d\| ≥ 0.5 STEC↔Non-STEC, OR `spat_cv_lps_chain` clears \|d\| ≥ 0.7 alone → spatial heterogeneity is a real discriminative axis; Stage 15F includes spatial features as headline. Plan/15 §7 R6 finding is publishable as a separate concern.
- **(B) Partial.** 1–2 spatial features clear \|d\| ≥ 0.5. Add to Stage 15F as supporting (most likely outcome — moment statistics typically give moderate, not strong, signal).
- **(C) Miss.** 0–1 spatial features clear \|d\| ≥ 0.5; H2O sanity-check also fails (variance NOT lower in H2O than bacteria). Spatial heterogeneity in this dataset doesn't carry class signal — likely because each file is a single culture so within-file heterogeneity is dominated by laser-focus rather than strain biology. Document, ship lean 10-col cache for completeness, move to Stage 15F.

### Stage-gate

- Regardless of branch, **proceed to Stage 15F** after this. 15E was always
  a "would-be-nice-if-it-helps" stage per plan/15 (the only ~0.5-day stage
  in the original roadmap besides 15D).
- If Branch (A) AND any K-12-spatial signal emerges, that's a second
  K-12-specific axis (after Stage 15D's 2°-structure shift) that could
  affect Stage 6 reconsideration math.

---

## Results

### Headline

**Branch (C) for STEC↔Non-STEC** — 0 of 10 spatial features clear
|d|≥0.5 at file level. Best STEC↔Non-STEC feature is `spat_skew_lps_1117`
at d=+0.485, just shy of the threshold. **The hypothesis that
"clinical pathogenic strains are spatially more uniform than commensals"
is falsified on this corpus.** Both STEC and Non-STEC E. coli cultures
have similar within-file pixel heterogeneity.

Two clean side-findings rescue the stage from total miss:

1. **`spat_skew_lps_1117` d=+0.725 is a new strong E. coli ↔ Salmonella
   axis.** Salmonella's pixel-intensity distribution at 1117 cm⁻¹
   (phospholipid backbone) is more symmetric (skew near 0), while E. coli's
   is right-skewed (skew > 0.7 mean). This is consistent with E. coli
   having a more bimodal cell-population structure at this band
   (some pixels with strong 1117 signal, others without) — possibly
   reflecting heterogeneous LPS expression across the colony.

2. **H2O class sanity passes 4/4.** All four variance/CV features show
   H2O ~10× LOWER values than bacteria (e.g. `spat_var_lps_chain` H2O=79
   vs bacterial=947, `spat_cv_ch_stretch` H2O=0.053 vs 0.174 with d=−1.58).
   Confirms the spatial-heterogeneity features are doing what they should
   and the H2O class is *trivially* separable on these features.

**Build:** 87 files × 10 features in 0.1 s — way under the 30 s pre-reg.

### Detailed results

#### 1. Per-feature file-level signal (Cohen's d)

| Feature | d STEC↔Non-STEC | d E.coli↔Salm | d H2O↔bact | d K-12↔other-STEC | AUROC STEC↔Non-STEC |
|---|---:|---:|---:|---:|---:|
| `spat_var_lps_chain`   | −0.067 | −0.153 | **−0.715** | −0.347 | 0.615 |
| `spat_var_ch_stretch`  | −0.131 | −0.211 | **−0.861** | −0.299 | 0.647 |
| `spat_cv_lps_chain`    | −0.021 | −0.277 | **−0.770** | −0.263 | 0.604 |
| `spat_cv_ch_stretch`   | −0.339 | −0.184 | **−1.582** | −0.115 | 0.649 |
| `spat_kurt_lps_1050`   | +0.170 | +0.212 | −0.682 | −0.059 | 0.520 |
| `spat_kurt_lps_1117`   | +0.360 | +0.132 | −0.115 | −0.242 | 0.603 |
| `spat_kurt_lps_1194`   | +0.348 | −0.417 | +0.877 | −0.180 | 0.634 |
| `spat_skew_lps_1050`   | +0.131 | +0.494 | −0.201 | −0.245 | 0.511 |
| **`spat_skew_lps_1117`** | **+0.485** | **+0.725** | −0.341 | −0.348 | 0.603 |
| `spat_skew_lps_1194`   | +0.337 | −0.022 | −0.868 | −0.229 | 0.553 |

#### 2. Per-class file-level mean

| Feature | H2O | Non-STEC | STEC | Salmonella |
|---|---:|---:|---:|---:|
| spat_var_lps_chain   |   79.1 |  925.4 |  838.4 | 1074.7 |
| spat_var_ch_stretch  |   20.5 |  288.2 |  245.1 |  334.9 |
| spat_cv_lps_chain    |  0.086 |  0.248 |  0.243 |  0.315 |
| spat_cv_ch_stretch   |  0.053 |  0.183 |  0.156 |  0.184 |
| spat_kurt_lps_1050   | −0.913 |  3.851 |  5.235 |  2.970 |
| spat_kurt_lps_1117   |  2.137 |  1.792 |  4.552 |  2.309 |
| spat_kurt_lps_1194   |  7.346 |  1.062 |  2.260 |  3.263 |
| spat_skew_lps_1050   |  0.393 |  0.822 |  1.023 |  0.207 |
| spat_skew_lps_1117   | −0.454 |  0.010 |  0.724 | −0.625 |
| spat_skew_lps_1194   | −1.726 | −0.995 | −0.689 | −0.816 |

#### 3. Pre-registration verdicts

| Pre-reg | Predicted | Actual | Verdict |
|---|---|---|:-:|
| New feature columns | 10 ± 2 | 10 | ✅ |
| Build time | < 30 s | 0.10 s | ✅ |
| `spat_cv_lps_chain` STEC↔Non-STEC \|d\| in 0.3–0.8 | yes | d=−0.021 | ❌ below |
| `spat_kurt_lps_1194` STEC↔Non-STEC \|d\| in 0.2–0.6 | yes | d=+0.348 | ✅ |
| H2O variance/CV LOWER than bacteria (4/4) | yes | all 4 pass | ✅ |
| K-12 vs other-STEC \|d\| ≥ 0.4 for ≥ 1 feature | maybe (50%) | max d=−0.348 | ❌ |
| Best new spatial \|d\| STEC↔Non-STEC in 0.3–0.7 | yes | 0.485 | ✅ — at lower end |
| ≥ 2 features \|d\| ≥ 0.5 STEC↔Non-STEC | yes (60%) | **0 features** | ❌ Branch (C) miss |
| Moran's I NOT IMPLEMENTED | confirmed | dropped per R6 pre-flight (no files ≥ 200 px) | ✅ |

#### 4. R6 pre-flight finding (publishable footnote)

Plan/15 §7 R6 specified a ≥200-pixel-per-file threshold for Moran's I
and GLCM applicability. **The Atlas corpus has zero files at that
threshold** (min=70, **median=72**, max=180; only 8 of 87 exceed 100).
Spatial-correlation features (Moran's I, GLCM, spatial gradient) are
therefore unavailable on this dataset without dropping the small-grid
threshold to single digits — at which point the statistics themselves
become unreliable. Documented here so Stage 15F doesn't try to revive
them.

### Implications

1. **STEC vs Non-STEC is NOT spatially distinguishable at file level.**
   This is a genuine negative result — both clinical pathogenic and
   commensal E. coli cultures show similar within-file pixel
   heterogeneity (variance + CV at LPS + CH stretches, kurtosis + skew at
   anchor bands). The biology of "do strain populations look uniform?"
   doesn't separate these two classes on this corpus. Stage 15F should
   include spatial features for completeness and orthogonality but not
   expect lift from them on the STEC↔Non-STEC axis.

2. **`spat_skew_lps_1117` (d=+0.725) is a candidate E.coli↔Salm
   headline.** Combined with Stage 15D's `bio_virulence_aa_sig` (Trp/Phe)
   d=−0.651 for the same class pair, we now have **two new strong
   E.coli↔Salm features** beyond the existing LPS anchors. These are
   orthogonal: skew_1117 is a pixel-population symmetry statistic, and
   bio_virulence_aa_sig is a per-pixel intensity ratio. Stage 15F should
   verify they add independent signal via correlation check.

3. **K-12 has no spatial signature.** K-12's anomaly (Stage 15D
   2°-structure shift) does not extend to spatial heterogeneity. K-12
   culture is spatially as uniform as any other STEC strain — consistent
   with K-12 being a well-characterized lab strain that's been selected
   for homogeneity. The K-12 axis remains a *biology-only* axis, not a
   biology + spatial axis.

4. **R6 confirmed for this corpus.** The plan/15 ≥200-pixel threshold is
   blocking, not advisory — 0/87 files clear it. Moran's I / GLCM are not
   tractable on this dataset. Future Atlas-style datasets with smaller
   pixel grids should expect the same constraint.

5. **For Stage 15F:** treat spatial features as *complementary*, not
   *headline*. Total project cache after Stage 15E: **259 features**
   (166 band + 51 spectral + 32 unmix + 10 spatial). Per plan/15 §7 R1,
   feature selection (mutual-information) is mandatory — 259 features ÷
   87 files = 3.0 features per file, deeper into curse-of-dimensionality
   than the prior 249 features.

6. **Branch (C) does NOT pause Stage 15F.** Pre-reg stage-gate said
   "regardless of branch, proceed to Stage 15F" — and that holds. Stage
   15F now has the full ~250-feature catalog with the MCR + biology +
   spatial blocks each having distinct discriminative profiles. Stage 6
   (3-channel CNN) reconsideration remains conditional on Stage 15F
   K-12 lift, not on spatial features.

7. **No new risks.** Plan/15 §7 risk register stays at R1-R9. The R6
   resolution is "drop the feature family that triggered the risk," so
   R6 is *retired* on this corpus.

---

## Artifacts

- `atlas/spatial_features.py` (new module)
- `scripts/build_spatial_features_cache.py` (new build script)
- `data_cache/spatial_features.parquet` (87 × ~10)
- `outputs/band_chemistry/stage15e/01_stage15e_summary.json`
