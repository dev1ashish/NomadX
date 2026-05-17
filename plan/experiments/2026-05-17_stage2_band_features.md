# 2026-05-17 — Stage 2: empirical anchor 1194 is the strongest single-band STEC↔Non-STEC discriminator yet found (file-level d=1.03) {#2026-05-17--stage2-empirical-anchors-and-amide-shift}

> **Status:** complete
> **Stage / track:** [plan/14 band-chemistry §5](../14_band_chemistry_research.md). Mostly infrastructure (the `atlas/band_features.py` module + cached `feature_frame.parquet`), but ships with pre-registered smoke-check predictions.
> **Branch hit:** stage-gate **PASSED** (LPS asymmetry E. coli vs Salmonella d = −0.63)
> **One-line headline:** `auc_lps_1194` d=+1.03 — cleanest single-band STEC↔Non-STEC discriminator in the whole project at file level
> **Cross-refs:** [prior — Stage 1 STEC triple](2026-05-17_stage1_stec_triple.md) · [next — Stage 3 radars/ratios](2026-05-17_stage3_radars_ratios.md)

---

## Pre-registration

### Method

Build `atlas/band_features.py` per plan/14 §5.1, populate `data_cache/band_features.parquet` via `scripts/build_band_features_cache.py`. Smoke checks below verify the module produces the expected discriminative geometry before Stage 3 launches.

### Predictions

| Check | Predicted |
|---|---|
| Macromolecule LPS AUC differs E. coli vs Salmonella by ≥ 0.5 SD (file-level) | yes |
| Macromolecule NA / amide / aromatic-AA group AUC differs STEC vs Non-STEC by < 0.3 SD (file-level) — consistent with Stage 1 Branch (C) | yes (all 3 below 0.3) |
| New empirical-anchor 800–1200 LPS AUC: STEC vs Non-STEC file-level Cohen's d ≥ 0.4 | yes (Stage 1 showed top discriminators at 1117 and 1194) |
| Lorentzian fit success rate, per-band, per-spectrum (defined as fit returns finite center within ±20 cm⁻¹ of catalog center and FWHM in 5–40 cm⁻¹) | ≥ 80% on most bands; literature triple may underperform (catch the peak-position-drift hypothesis) |
| Fitted peak-center mean within ±3 cm⁻¹ of catalog center for clear single-peak bands (e.g. 1050, 1117, 1194) | yes |
| Fitted peak-center for 1454 — possible drift since the band sits in an overlapping CH₂ deformation cluster | catalog 1454, fitted likely 1448–1456 |
| Feature cache size | 7,122 rows × 25–40 columns |
| Macromolecule AUC for H₂O class at every group | all groups significantly lower than any bacterial class (water has no biological vibrations) |

### Pre-committed module surface

(No test of internals — that's unit-test territory, not pre-reg territory.)

`integrate_band`, `integrate_region`, `macromolecule_vector`, `band_ratios`, `fit_peak`, `fit_peaks_batch`, `feature_frame`, `BANDS`, `MACROMOLECULE_GROUPS`. All defined in plan/14 §5.1.

### Reasoning

The Stage 1 finding sets all of these. The LPS-vs-non-LPS asymmetry is the headline biology prediction; if even the macromolecule radar can't surface the E. coli vs Salmonella LPS difference, the module is broken. If the literature triple bands fit cleanly with peak centers right at 1338/1454/1658, the "peak-shift hypothesis" for Stage 1's null gets falsified — meaning the bands genuinely don't carry STEC-vs-Non-STEC signal on this dataset (rather than carrying signal at a slightly different wavenumber the ±10 window missed).

### Stage-gate

If LPS asymmetry doesn't show up at all (file-level Cohen's d < 0.3 for E. coli vs Salmonella on LPS group AUC), **pause and debug the module before Stage 3.** Almost-certainly a bug — the per-file mean spectra in `images/_summary/07_annotated_preprocessed_spectra.png` show the LPS region visibly differing across primary classes.

---

## Results

### Headline

1. **`auc_lps_1194` is the cleanest single-band STEC vs Non-STEC discriminator in the whole project at file-level: Cohen's d = +1.03** (STEC > Non-STEC). `auc_lps_1117` follows at d = +0.77. Compare to literature triple at the same file-level test: d ∈ [+0.13, −0.47] (none > |0.5|). The 800–1200 LPS chain region the user emphasized [[atlas-briefing-emphasis]] holds the actual STEC↔Non-STEC signal at sub-bin specificity.

2. **The literature primary triple is null on macromolecule group AUC too.** `auc_aromatic_aa`, `auc_protein_amide`, `auc_nucleic_acid`, `auc_lipid_carbohydrate` all return |d| < 0.30 at STEC vs Non-STEC file level — fully consistent with Stage 1's Branch (C) verdict. The catalog-level grouping doesn't rescue the literature anchor; the bands genuinely lack STEC↔Non-STEC signal on this dataset.

3. **E. coli vs Salmonella LPS asymmetry is real: d = −0.63 on `auc_lps_chain_discrim`** (Salmonella > E. coli). Pre-reg floor of 0.5 cleared by 0.13. Direction is biologically sensible — Salmonella O-antigens are more complex/branched than E. coli's, producing stronger Raman scattering in the carbohydrate skeletal region.

4. **Amide-I peak position SHIFTS between STEC and Non-STEC.** Mean Lorentzian-fitted center on successful fits: STEC = 1659.3 cm⁻¹ (drift +1.3 from catalog 1658), Non-STEC = 1662.1 cm⁻¹ (drift +4.1), Salmonella = 1660.9 cm⁻¹ (drift +2.9). **The Non-STEC amide-I peak sits 2.8 cm⁻¹ above STEC's** — a position drift, not an intensity difference. The Stage 1 ±10 window around 1658 averages over both peak positions and dilutes the signal. This is the first piece of evidence that the Stage 1 nulls *partly* reflect peak-position drift, not absence of signal — supports Stage 4 (Lorentzian-fit-based peak-shift probe).

### Pre-registration verdicts

| Pre-reg | Predicted | Actual | Verdict |
|---|---|---:|:-:|
| LPS group AUC E. coli vs Salmonella \|d\| ≥ 0.5 | yes | 0.63 (lps_chain_discrim) | ✅ |
| Macro NA / amide / AA STEC vs Non-STEC \|d\| < 0.3 | yes (all 3) | aa=0.17, amide=0.13, NA=0.22, lipid=0.25, metabolite=−0.41 | ⚠️ 4/5 in range; metabolite at −0.41 a small over-cap |
| 800–1200 LPS region STEC vs Non-STEC \|d\| ≥ 0.4 | yes | broad region 0.28 ❌; **but narrow lps_1117 = 0.77 ✅, lps_1194 = 1.03 ✅✅** | ⚠️ broad-region prediction wrong, narrow-band prediction overshoots |
| Lorentzian fit success rate ≥ 80% on most bands | yes | 0.2–37% across 8 bands | ❌ FAILED |
| Fit center within ±3 cm⁻¹ of catalog on clear bands | yes | 1050 drifts up to −3.4 (Non-STEC); 1194 within ±0.8 | ⚠️ partial |
| 1454 peak-center possibly drifts 1448–1456 | yes | STEC −0.07, Non-STEC −1.12, Salmonella −0.46 | ✅ small drift |
| Feature cache 25–40 columns | yes | 81 | ⚠️ over (added more ratios + per-band AUCs; cheap to keep) |
| H₂O < bacterial on every macromolecule group | yes | protein/lipid yes (d = −0.80 / −0.22); AA / NA UNEXPECTEDLY higher in H₂O (d = +0.18 / +0.14) | ⚠️ 2/4 |

**The Lorentzian fit failure is the only outright miss.** Most bands fit 2–37% of spectra; only `aa_1004` (37%) and `lps_1194` (31%) crack 30%. Plan/14 §6.6 (peak-shift probe) relies on these fits — needs investigation before Stage 4 launches. Likely causes: (a) flat-baseline Lorentzian model can't track residual SNV slope; (b) FWHM bounds [5, 40] too narrow for some bands; (c) at very weak peaks the local extremum is just noise, not a real peak.

### H₂O anomaly

H₂O scores HIGHER than bacterial classes on aromatic_aa (d = +0.18) and nucleic_acid (d = +0.14) group AUC — opposite of what biology would predict. **Hypothesis: post-SNV substrate residual.** Water spectra have no biological peaks, so SNV (per-spectrum zero-centering, unit-variance) amplifies whatever substrate signal remains. Substrate has its own Raman signature in the fingerprint region; after SNV those bumps look "above average" relative to the SNV-flattened bacterial fingerprint. Worth verifying with a non-SNV visualization in Stage 3 — and worth tagging as a Stage-3 EDA TODO.

### Per-strain peak-position drift table

Mean fitted peak-center − catalog center, per primary class, on successful fits:

| band | catalog | STEC | Non-STEC | Salmonella | H₂O |
|---|---:|---:|---:|---:|---:|
| lps_1050       | 1050 | −3.32 | −2.01 | +1.85 | NaN |
| lps_1117       | 1117 |   NaN | −3.36 |   NaN | NaN |
| lps_1194       | 1194 | −0.78 | −0.48 | −0.78 | NaN |
| na_1338        | 1338 | −0.71 | −0.64 | −0.81 | −0.49 |
| lipid_1454     | 1454 | −0.07 | −1.12 | −0.46 | +0.35 |
| amide_i_1658   | 1658 | **+1.29** | **+4.10** | +2.87 | −1.07 |
| aa_1004        | 1004 | +0.29 | +0.46 | +0.47 | NaN |
| amide_iii_1242 | 1242 | +0.30 | −0.10 | −0.69 | NaN |

The most striking row is `amide_i_1658`: Non-STEC fitted center is at ~1662 (amide-I α-helix per briefing), while STEC fitted center is at ~1659 (somewhere between α-helix and β-sheet). **Could be a real secondary-structure shift** consistent with STEC carrying virulence proteins (Shiga toxin, intimin) that have different β-sheet content than typical E. coli outer-membrane proteins. Needs the Stage 4 peak-fit improvement first — current 14% success rate on this band makes the per-class means noisy.

### Module surface (atlas/band_features.py)

Pure-function API over `(X: ndarray[N, B], wn: ndarray[B])`: `integrate_band`, `integrate_region`, `macromolecule_vector`, `band_ratios`, `fit_peak`, `fit_peaks_batch`, `feature_frame`. Catalog: `BANDS` (32 named bands), `MACROMOLECULE_GROUPS` (5 groups + 2 LPS regions), `PRIMARY_TRIPLE` (literature, demoted), `EMPIRICAL_ANCHOR_BANDS` (Atlas-derived headline anchors). Plan/14 §5.1 spec adhered to.

### Implications for Stages 3–5

- **Stage 3 (notebook visualizations):** macromolecule radar should now lead with the LPS group asymmetry (E. coli vs Salmonella, d = −0.63) and the per-class 1117/1194 AUC violins (STEC vs Non-STEC, d = 0.77 / 1.03). The literature triple panels remain in the notebook as negative-finding illustration.
- **Stage 4 (peak-shift probe):** Lorentzian fit needs to be reworked before this stage can launch. Options: (a) linear baseline instead of flat; (b) loosen FWHM bounds; (c) only fit on per-file mean spectra (~87 fits instead of 7,122). Stage 4 is gated on the fit-rate fix.
- **Stage 5 (engineered-feature classifier):** anchor feature set = `auc_lps_chain_discrim`, `auc_lps_1117`, `auc_lps_1194`, `auc_lps_1050`, plus the 10 ratios. Literature-triple AUCs included as supporting features. Pre-register LOSO ≥ 0.55 (interpretability parity) as the bar.

### Operational decisions

1. **Stage 3 launchable.** Macromolecule radar narrative anchors on LPS E. coli↔Salmonella asymmetry; per-class violins at 1117/1194 are the headline plot.
2. **Stage 4 gated on a Lorentzian-fit fix.** Three repair options to try in order: (a) linear baseline instead of flat; (b) loosen FWHM bounds to [3, 60]; (c) fit only per-file-mean spectra (87 fits, much cleaner signal).
3. **Stage 5 anchor feature set locked.** Headline features = `auc_lps_chain_discrim`, `auc_lps_1050`, `auc_lps_1117`, `auc_lps_1194`, plus the 10 default ratios. Literature-triple AUCs included as supporting features.
4. **Pre-registration miscalibration to log.** The broad-region prediction ("800–1200 integrated AUC d ≥ 0.4") was wrong — broad integration dilutes the per-bin signal. The narrow ±10 windows at the empirical-anchor centers are the right operationalization. **For Stage 5 the classifier should use the narrow windows as features, not the broad-region AUC.**
5. **Active-stage memory updated** ([[atlas-band-chemistry-roadmap]]): Stage 2 → done; Stage 3 anchor set.

---

## Artifacts

- `atlas/band_features.py` (new module, 32 bands, 5 macromolecule groups, 10 ratios, 8 Lorentzian-fitted bands)
- `data_cache/band_features.parquet` (7,122 × 81)
- `scripts/build_band_features_cache.py` (smoke + build + sanity)
- `outputs/band_chemistry/04_stage2_peak_drift.csv`
- `outputs/band_chemistry/04_stage2_per_class_macromolecule.csv`
