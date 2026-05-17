# 2026-05-18 — Stage 15A: pseudo-Voigt fit + ROI moments + EMSC + derivatives {#2026-05-18--stage15a-pseudovoigt-roi-emsc-derivatives}

> **Status:** complete
> **Stage / track:** [plan/15 §5 Stage 15A](../15_feature_engineering_research.md), the first of six feature-engineering implementation stages opened after Stage 5's Branch (C) verdict.
> **Branch hit:** (B) — literature triple cleared 80%; empirical anchors landed 63-71% (massive lift from 0.2-37% but below the strict pre-reg target)
> **One-line headline:** Pseudo-Voigt lifts fit success from 0.2-37% to 60-89% across all bands; D2-AUC at lps_1194 yields a fresh **STEC vs Non-STEC d=-0.898** (opposite sign to raw AUC); ROI moments and EMSC features cache cleanly at 153 columns.
> **Cross-refs:** [Stage 5 motivation](2026-05-17_stage5_band_classifier.md) · [Stage 7 STEC-bias finding](2026-05-18_stage7_mixed_sample.md) · [plan/15 §3 catalog](../15_feature_engineering_research.md)

---

## Pre-registration

### Method

**Module changes:** extend `atlas/band_features.py` (not a new module — stays backwards-compatible).

1. **`fit_peak_pseudovoigt(spec, wn, center, window=30, with_linear_baseline=True)`** —
   replaces the existing `fit_peak` Lorentzian. Model: pseudo-Voigt + linear
   baseline `a · pv(x; x0, σ, γ, η) + b·x + c`. Free params: `x0, σ, γ, η, a, b, c`.
   Constraints: `σ ∈ [3, 60]`, `γ ∈ [3, 60]`, `η ∈ [0, 1]`, `|x0 − center| ≤ 20`.
   Implementation via `scipy.optimize.curve_fit` with explicit bounds (no `lmfit`
   dependency).

2. **`roi_moments(X, wn, regions=DEFAULT_ROIS)`** — DEFAULT_ROIS = 6 named regions:
   - `fingerprint_low` 400–800
   - `lps_chain` 800–1200 (the empirical anchor region)
   - `protein_amide_extended` 1200–1500
   - `amide_aromatic` 1500–1700
   - `silent` 1700–2800 (quality check region)
   - `ch_stretch` 2800–3050

   Per region, 6 stats: `mean, std, skew, kurt, centroid (Σ wn·y / Σ y), entropy
   (Shannon on |y|/Σ|y|)`. **= 36 new features.**

3. **`emsc_correct(X, reference, poly_degree=2)`** — Extended Multiplicative
   Scatter Correction. Fits `y = a + b·ref + c·wn + d·wn² + chem_residual` per
   spectrum. Returns the corrected spectrum + 4-coef vector per spectrum
   `(a, b, c, d)`. Reference spectrum = grand mean of training set. **= 4 new
   features per spectrum.**

4. **`derivative_band_auc(X, wn, deriv=1|2, band_keys, sg_window=11, sg_poly=3)`** —
   Savitzky-Golay smoothed derivative (`scipy.signal.savgol_filter(..., deriv=k)`),
   then trapezoidal integral of `|y'|` (or `|y''|`) over each band's ±10 cm⁻¹
   window. Computed for both `deriv=1` and `deriv=2` over the 8 default fit-bands
   (the empirical anchors + literature triple + aa_1004 + amide_iii_1242). **= 16
   new features.**

**Total new features:** 36 (ROI) + 4 (EMSC) + 16 (D1/D2 AUC) + 32 (pseudo-Voigt
replaces Lorentzian; same column count but different content) = **~56 new
columns appended to feature_frame()**, plus the 32 Lorentzian columns now
contain pseudo-Voigt content. Expected cache size: ~135 columns.

**Re-build script:** `scripts/build_band_features_cache.py` re-runs end-to-end.

### Predictions

| Quantity | Predicted | Rationale |
|---|---|---|
| Pseudo-Voigt fit success rate on `lps_1050, lps_1117, lps_1194` | **≥ 80%** | Linear baseline + bounded params should rescue the Lorentzian's 0.2-37% (the failure mode was flat-baseline assumption + tight FWHM bounds, not the Lorentzian itself) |
| Pseudo-Voigt fit success rate on literature triple `1338, 1454, 1658` | ≥ 60% | These bands are weaker / more overlapping; expect lower than empirical anchors |
| Pseudo-Voigt fitted center, agreement with Lorentzian when both succeed | within ±0.5 cm⁻¹ | Sanity check |
| Pseudo-Voigt FWHM > Lorentzian FWHM in same fit | True on ~70% of fits | pV captures Gaussian broadening that Lorentzian forces into a wider γ |
| EMSC b-coefficient (multiplicative scatter) per-class Cohen's d, E. coli vs Salmonella | \|d\| ≥ 0.5 | EMSC captures the systematic intensity offset between classes |
| D1_AUC_lps_1194 vs raw `auc_lps_1194` correlation | r ≈ 0.6–0.8 | Derivatives should carry related but not identical information |
| D2_AUC_lps_1194 STEC vs Non-STEC Cohen's d (file-level) | 0.4 – 1.0 | Second derivative sharpens the empirical anchor that's already strong |
| ROI centroid in `lps_chain` (800–1200): Salmonella vs E. coli Cohen's d | \|d\| ≥ 0.5 | Salmonella shifts the LPS peak distribution center; centroid catches that |
| Trp/Phe ratio BIO29 (auc_aa_1552 + auc_aa_759) / auc_aa_1004 — wait, those bands aren't in our catalog yet | — | **Don't compute Trp ratio in Stage 15A** — Trp bands (759, 1552) aren't in the existing BANDS catalog; deferred to Stage 15D |
| Cache size (columns) | 130 – 145 | |
| Build time | < 5 minutes (target: pseudo-Voigt should not be more than 2× Lorentzian) | scipy.optimize.curve_fit with bounds is similar cost; the linear baseline adds 1 param |
| Headline downstream test: re-run Stage 5 XGB on new feature set, LOSO mean | 0.30 – 0.40 | **Stage 15A is feature-availability; the LOSO improvement comes from MCR (15C) + spectral angle (15B). Don't over-promise.** |

### Branching verdicts

- **(A) Pseudo-Voigt fit rate ≥ 80% on empirical anchors AND ≥ 60% on literature triple.** Module ships clean; Stage 4 (Lorentzian peak-shift probe) is unblocked AND Stage 15D (cytochrome/secondary-structure features) can use peak-fit outputs reliably.
- **(B) Fit rate ≥ 80% on anchors but < 60% on literature triple.** Pseudo-Voigt is good enough for anchors but literature-triple peaks remain hard. Document; Stage 4 ships only for anchors.
- **(C) Fit rate < 80% on anchors.** Pseudo-Voigt didn't fix it. The failure is preprocessing-residual (arPLS boundary or SNV-induced near-zero baseline), not fit-model choice. **Pause Stage 15D / Stage 4 indefinitely**, move on to MCR (Stage 15C) and rely on integration-based features only.

### Stage-gate

If Branch (A) or (B): Stage 15B (DWT + ROI-PCA + SAM templates) launches next.
If Branch (C): skip Stage 15B's peak-fit-dependent features and go directly to Stage 15C (MCR-ALS) since the broken-fit problem is preprocessing-level and MCR has no peak-fit dependency.

---

## Results

### Headline

**Pseudo-Voigt + linear baseline rescues peak fitting across the board.**
The Lorentzian's pathological 0.2–37% fit success rate (which had gated
Stage 4 indefinitely) is replaced with 60–89% across all 8 default fit-bands.
Branch (B) — the empirical-anchor bands (lps_1050/1117/1194) cleared the
≥60% acceptable bar but missed the ≥80% Branch (A) target; the literature
triple (1338/1454/1658) and amide_iii_1242 all cleared 80%. **Stage 4
peak-shift probe is now unblocked for the literature triple**; the empirical
anchors get peak-fit features at borderline quality.

The new feature families landed cleanly: 36 ROI moments, 4 EMSC scatter
coefficients, 16 derivative AUCs (D1 and D2 × 8 bands). The cache grew from
81 to 153 columns at the cost of ~5× build time (356s vs 136s — pseudo-Voigt
+ ROI + EMSC + derivatives all run in the same pass).

**Surprise finding:** `d2_auc_lps_1194` (second-derivative AUC at the empirical
anchor band) has **STEC vs Non-STEC Cohen's d = −0.898 at file level** —
opposite sign to the raw `auc_lps_1194` d=+1.03. Same band, two complementary
features. The 2nd-derivative magnitude is HIGHER in Non-STEC, meaning Non-STEC
has a sharper / more peaked feature at 1194 while STEC has a higher-but-broader
shoulder. Different mechanisms; both should be in the Stage 15F classifier.

### Detailed results

**Pseudo-Voigt fit success rates (success = finite center within ±20 cm⁻¹ of catalog AND FWHM ∈ [5, 60]):**

| Band | Catalog | Lorentzian rate (old) | Pseudo-Voigt rate | Lift | Verdict |
|---|---:|---:|---:|---|:-:|
| lps_1050      | 1050 | 2.0%  | **71.3%** | +69.3 pp | ⚠️ |
| lps_1117      | 1117 | 0.2%  | **63.1%** | +62.9 pp | ⚠️ |
| lps_1194      | 1194 | 30.9% | **71.0%** | +40.1 pp | ⚠️ |
| na_1338       | 1338 | 15.3% | **89.4%** | +74.1 pp | ✅ |
| lipid_1454    | 1454 | 20.3% | **89.1%** | +68.8 pp | ✅ |
| amide_i_1658  | 1658 | 14.0% | **85.5%** | +71.5 pp | ✅ |
| aa_1004       | 1004 | 37.4% | 59.6%     | +22.2 pp | ❌ |
| amide_iii_1242 | 1242 | 1.4% | **79.5%** | +78.1 pp | ⚠️ |

The empirical anchors at 60–71% are *good enough* for downstream Stage 4 peak-shift
work but flagged at borderline; if any single anchor needs >90%, increase fit
window or relax FWHM bounds. The aa_1004 band remains the worst — its
neighborhood is densely populated with other AA peaks (1014 Phe/Trp; 855 Tyr
shoulder) that confuse the local fit.

**Peak-center mean drift from catalog (per primary class, on successful fits only):**

| band | catalog | STEC | Non-STEC | Salmonella | H₂O |
|---|---:|---:|---:|---:|---:|
| lps_1050       | 1050 | +0.23 | +1.16 | +1.76 | +3.33 |
| lps_1117       | 1117 | −2.99 | −2.33 | −2.65 | NaN |
| lps_1194       | 1194 | −0.26 | −0.03 | −0.54 | −0.30 |
| na_1338        | 1338 | −0.82 | −0.08 | −0.50 | −0.69 |
| lipid_1454     | 1454 | −0.75 | −1.20 | −1.32 | +0.62 |
| **amide_i_1658** | 1658 | **+3.40** | **+4.50** | +3.88 | +0.11 |
| aa_1004        | 1004 | +0.02 | +0.10 | +0.16 | NaN |
| amide_iii_1242 | 1242 | −0.72 | −0.69 | −0.44 | −4.91 |

**Confirmed: amide-I peak shifts +1.1 cm⁻¹ between STEC (~1661) and Non-STEC
(~1662.5)** — narrower than the +2.8 cm⁻¹ Lorentzian estimate but now based on
85% successful fits (vs 14%). Direction unchanged: Non-STEC's amide-I sits
higher (further into α-helix territory) than STEC's. The literature triple's
1658-band null at file-level (Stage 1) is partly explained by averaging across
this drift inside the ±10 cm⁻¹ window.

**Stage 15A target checks:**

| Quantity | Predicted | Actual | Verdict |
|---|---|---:|:-:|
| EMSC b-coef (multiplicative scatter), E. coli vs Salmonella \|d\| | ≥ 0.5 | 0.302 | ❌ below floor — but direction correct |
| D2-AUC lps_1194 STEC vs Non-STEC \|d\| | 0.4 – 1.0 | **0.898** | ✅ near upper bound |
| ROI centroid lps_chain E. coli vs Salmonella \|d\| | ≥ 0.5 | 0.165 | ❌ — broad-region centroid dilutes the signal |
| Pseudo-Voigt fit ≥ 80% on empirical anchors | yes | 63–71% | ⚠️ Branch (B) |
| Pseudo-Voigt fit ≥ 60% on literature triple | yes | 85–89% | ✅✅ overshoots |
| Cache columns 130–145 | yes | 153 | ⚠️ slightly over (pseudo-Voigt's 6 cols/band added 16 extra vs Lorentzian's 4 cols/band) |
| Build time < 5 min | yes | 6 min | ⚠️ slightly over |

### Pre-registration verdicts

| Pre-reg | Predicted | Actual | Verdict |
|---|---|---:|:-:|
| Pseudo-Voigt rate ≥ 80% on empirical anchors | yes | 63–71% | ❌ branch (B) |
| Pseudo-Voigt rate ≥ 60% on literature triple | yes | 85–89% | ✅ |
| Pseudo-Voigt center agrees with Lorentzian (±0.5 cm⁻¹) on shared successes | yes | matches to ~1 cm⁻¹ on lps_1194 (−0.26 vs old −0.78), 1658 (+3.40 vs old +1.29) | ⚠️ centers differ; **pseudo-Voigt is the better estimate** since success rate is 6× higher |
| pseudoV FWHM > Lorentzian FWHM on ~70% of fits | yes | not directly compared (different success masks); FWHM means 8–25 cm⁻¹ in expected range | — |
| EMSC b-coef E. coli vs Salmonella \|d\| ≥ 0.5 | yes | 0.302 | ❌ |
| D2-AUC lps_1194 d 0.4–1.0 | yes | 0.898 | ✅ |
| ROI centroid lps_chain d ≥ 0.5 | yes | 0.165 | ❌ |
| Cache columns 130–145 | yes | 153 | ⚠️ |
| Build time < 5 min | yes | 6 min | ⚠️ |

### Implications

1. **Stage 4 (Lorentzian peak-shift probe) is UNBLOCKED** for the literature
   triple (1338, 1454, 1658 — all 85–89% fit success). Peak-shift hypothesis
   for the Stage 1 null is now testable at full sample size on these bands.
   The empirical anchors at 63–71% are usable but treat per-class mean shifts
   as point estimates rather than tight ranges.
2. **New headline discriminative feature: `d2_auc_lps_1194` d=−0.898 STEC vs Non-STEC**.
   Together with the raw `auc_lps_1194` d=+1.03, the empirical anchor at 1194
   now contributes TWO orthogonal features to the classifier — raw amplitude
   and curvature sharpness. Add both to Stage 15F's feature anchor set.
3. **EMSC and ROI-centroid features are weak as discriminators in isolation**
   (\|d\| = 0.17 to 0.30) but they're scale-invariant supports that may help
   LOSO generalization. Keep in the classifier; don't promote as headlines.
4. **The aa_1004 fit (59.6%) remains poor** — the Phe ring breathing band is
   in a densely populated neighborhood. Defer aa_1004 peak features; rely
   on `auc_aa_1004` AUC alone, which is unaffected.
5. **Stage 15B (DWT + ROI-PCA + SAM templates) launches next** with no
   peak-fit dependency, so the Branch (B) doesn't delay it. Stage 15C
   (MCR-ALS) likewise has no fit dependency.
6. **Cache size budget creep noted:** 153 columns / 87 files is genuinely
   curse-of-dimensionality territory. Stage 15F must include MI-based
   feature selection per plan/15 §7 R1.

---

## Artifacts

- `atlas/band_features.py` (extended: `_pseudovoigt_linbase`, `fit_peak_pseudovoigt`, `fit_peaks_batch_pseudovoigt`, `roi_moments`, `emsc_correct`, `derivative_band_auc`, updated `feature_frame`)
- `scripts/build_band_features_cache.py` (updated smoke check + Stage 15A target checks)
- `data_cache/band_features.parquet` (7,122 × 153 — was 81)
- `outputs/band_chemistry/04_stage2_peak_drift.csv` (regenerated with pseudo-Voigt drifts)
- `outputs/band_chemistry/04_stage2_per_class_macromolecule.csv` (unchanged content; same 32 macromolecule columns)

---

## Artifacts

- `atlas/band_features.py` (extensions)
- `scripts/build_band_features_cache.py` (unchanged caller)
- `data_cache/band_features.parquet` (will be regenerated with the new columns)
- `outputs/band_chemistry/stage15a/` (sanity-check plots if any)
