# 2026-05-17 — Stage 1: Published STEC triple does not replicate at file-level; signal is in 800–1200 cm⁻¹ {#2026-05-17--stec-triple-does-not-replicate}

> **Status:** complete
> **Stage / track:** [plan/14 band-chemistry §6.2 + §6.3](../14_band_chemistry_research.md)
> **Branch hit:** (C) — zero of three published bands clear file-level bar
> **One-line headline:** Literature triple (1338/1454/1658) does NOT replicate at file level; entire ANOVA top-30 sits in the user-highlighted 800–1200 LPS region
> **Cross-refs:** [next — Stage 2 band_features module](2026-05-17_stage2_band_features.md) · [`cisek-2013`](../11_references.md#cisek-2013--cisek-et-al-analyst-2013-sensitive-and-specific-discrimination-of-pathogenic-and-nonpathogenic-escherichia-coli-using-raman-spectroscopy)

---

## Pre-registration

### Method

Two research questions resolved in one script run; predictions locked **before** the analysis runs. Script: `scripts/run_stage1_band_stats.py`.

**RQ4 — Bacteria-only ANOVA on preprocessed spectra (H₂O excluded).** Re-run per-bin ANOVA-F on the 3 bacterial classes only.

**RQ1 — STEC vs Non-STEC effect sizes at the primary triple.** For each of 1338, 1454, 1658: integrate ±10 cm⁻¹ AUC per spectrum, test STEC (n_files=27) vs Non-STEC (n_files=25). Both spectrum-level (n inflated by within-file similarity 0.997) and **file-level** (true independence) statistics reported.

### Predictions

**RQ4 — Bacteria-only ANOVA top-30:**

| Quantity | Predicted |
|---|---|
| Top-30 bins concentrate in fingerprint region (1100–1750 cm⁻¹) | yes (≥20 of 30) |
| At least 2 of {1338, 1454, 1658} appear in top-30 | yes |
| All 3 of {1338, 1454, 1658} appear in top-30 | possibly (50/50) |
| Top region centroid shifts away from 2880–2940 C-H stretch | yes (≤5 of 30 in C-H region) |
| LPS region (800–1200) contributes top-30 bins | yes (≥5 bins) |

**RQ1 — STEC vs Non-STEC effect sizes at primary triple:**

| Band | Predicted d (file-level) | File-level Welch t p |
|---|---|---|
| 1338 (NA, stx) | 0.30 – 0.80 | < 0.05 |
| 1454 (lipid) | 0.20 – 0.60 | < 0.10 |
| 1658 (amide-I) | 0.30 – 0.80 | < 0.05 |

Also reported: spectrum-level d (informational only — n=4600+ gives significance regardless), Mann-Whitney U, sign of the shift per band, per-strain breakdown (O157 vs O121 vs O103 vs K-12 vs ATCC25922 vs 83972), and best-single-band AUROC.

**Reasoning.** The 4-class ANOVA top-30 was *all* in 2880–2940 cm⁻¹ C-H stretch ([07§anova-bins-vs-stec-discriminative-bands](../07_findings.md#2026-05-14--anova-bins-vs-stec-discriminative-bands)) because that's the largest water-vs-bacteria difference. Removing H₂O should demote those bins and surface within-bacterial chemistry. Per `cisek-2013` the load-bearing STEC bands are 1338/1454/1658.

File-level n ≈ 25 per side; real power needs d ≥ 0.5. The spectrum-level test is foregone-significant given n inflation, so the **file-level p is the evidence bar**. Cohen's d prior comes from `cisek-2013` showing >95% pure-culture discrimination (which back-calculates to d > 1.0 under their conditions); we'd expect smaller effects on bacteria-on-substrate. A null at all 3 bands (file-level p > 0.1, d < 0.1) would imply published bands aren't load-bearing on this dataset — the most interesting finding either way.

### Branching verdicts

- **(A) All 3 bands clear file-level p < 0.05 with predicted-range d** → published bands confirmed; Stage 2 (band_features module) anchors on this set.
- **(B) 1–2 of 3 clear** → partial confirmation; document failures as a publishable negative finding; still proceed to Stage 2.
- **(C) 0 of 3 clear at file-level (p > 0.1 for all 3)** → published bands aren't the discriminator on this dataset; **pause Stage 2 and re-anchor**: use the bacteria-only ANOVA top-30 as primary anchor, treat literature bands as supporting features only.

### Per-strain subhypothesis

Predicted: at all 3 bands, the STEC vs Non-STEC mean shift is driven by **O157 + O121 + O103 collectively** (all 3 STEC strains contribute), not by O157 alone. If only O157 shifts, the briefing's "STEC vs Non-STEC" framing is really "O157 vs the rest" — narrower than implied.

---

## Results

### Headline

**Branch (C) hit.** Zero of the three published STEC↔non-STEC discriminative bands from `cisek-2013` — 1338, 1454, 1658 cm⁻¹ — clear the pre-registered file-level bar (Welch p < 0.05 AND |Cohen's d| ≥ 0.3).

| Band | Macromolecule | Cohen's d (file) | Welch p (file) | AUROC (file) | Direction | Cleared? |
|---|---|---:|---:|---:|---|:-:|
| 1338 | NA (stx, virulence)         | +0.13 | 0.64 | 0.53 | null | ❌ |
| 1454 | lipid + carbohydrate         | **−0.47** | 0.09 | 0.62 | **STEC < Non-STEC** (sign REVERSED vs Cisek) | ❌ |
| 1658 | amide-I (virulence proteins) | −0.07 | 0.79 | 0.52 | null | ❌ |

File-level n = 27 STEC vs 25 Non-STEC. Spectrum-level (n ≈ 4600) makes 1454 highly significant (p = 4e-11) and 1658 marginally (p = 1e-6), but spectrum-level n is inflated by within-file cosine ≈ 0.997 — file-level is the honest test.

### What DOES discriminate

A per-bin Welch t on E. coli-only spectra (Salmonella + H₂O excluded) lands top-30 bins almost entirely in the user-highlighted 800–1200 cm⁻¹ LPS chain region.

| Test | Top region | Strongest wavenumbers | Primary triple in top-30? |
|---|---|---|:-:|
| 3-class bacteria-only ANOVA | 800–1200 cm⁻¹ (**30/30**) | 1032, 1050, 1054 | 0/3 |
| 2-class E. coli-only t-test | 800–1200 cm⁻¹ (**30/30**); 1100–1750 (27/30) | 1117, 1194 (STEC > Non-STEC) | 0/3 |

The 3-class ANOVA top bins at ~1030–1055 are dominated by **E. coli vs Salmonella** signal (carbohydrate / PO₂⁻ region — LPS O-antigen chain backbone). The 2-class E. coli-only top bins at ~1117 and 1194 are the **STEC vs Non-STEC** signal — also in the LPS chain region, just slightly higher in the polysaccharide vibrational ladder. **Both within-bacterial discrimination problems concentrate in 800–1200**, confirming the briefing's green-penned emphasis on this region for LPS chain structure discrimination.

### Per-strain breakdown

At every primary-triple band, within-STEC variation (O157H7 vs O121H19 vs O103H2) is roughly the same magnitude as between-class shift. The "STEC class" as labeled is biochemically heterogeneous on these classical-literature bands. Most pronounced:

- **1338 (NA):** O157H7 lowest (mean ≈ −7.5), O103H2 highest (≈ −6.7) — a span larger than the STEC↔Non-STEC mean difference. O121H19 and O103H2 don't shift together against Non-STEC.
- **1454 (lipid):** All 3 STEC strains DO lie below the Non-STEC strains, generating d = −0.47, but Salmonella Typhimurium is even lower — the effect isn't STEC-specific, it's an "O157H7-and-Typhimurium-are-low" pattern that doesn't generalize.
- **1658 (amide-I):** O103H2 high, Typhimurium and 83972 low — pattern doesn't track parent class at all.

Cisek-2013 specifically compared O157:H7 against non-pathogenic E. coli C, K-12 (Hfr), HF4714. The finding here is consistent with their methodology even while falsifying their bands as universal STEC markers: their result was likely an **O157:H7-specific** signature, not a generalizable STEC vs non-STEC signal across O157 + O121 + O103.

### Pre-registration verdicts

| Quantity | Predicted | Actual | Verdict |
|---|---|---:|:-:|
| Top-30 in fingerprint (1100–1750) | ≥20 | **3** | ❌ wildly miscalibrated |
| Top-30 has ≥2 of {1338, 1454, 1658} | yes | **0** | ❌ |
| Top-30 ≤5 in C-H stretch | yes | 0 | ✅ (extreme) |
| Top-30 has ≥5 LPS (800–1200) | yes | **30** (everything!) | ✅✅ (extreme over-prediction in the right direction) |
| 1338 file Cohen's d, range 0.3–0.8 | yes | **+0.13** | ❌ null |
| 1454 file Cohen's d, range 0.2–0.6 | yes | **−0.47** | ⚠️ magnitude in range but SIGN REVERSED |
| 1658 file Cohen's d, range 0.3–0.8 | yes | **−0.07** | ❌ null |
| 1338 file Welch p < 0.05 | yes | **0.64** | ❌ |
| 1454 file Welch p < 0.10 | yes | 0.09 | ✅ borderline |
| 1658 file Welch p < 0.05 | yes | 0.79 | ❌ |

**Pre-registration miscalibration owned.** I framed the 3-class bacteria-only ANOVA as a STEC↔Non-STEC test; it isn't — it's dominated by the much larger E. coli↔Salmonella signal in 800–1200. Added an E. coli-only 2-class t-test mid-run; it also returns 0/3 on the triple, so the Branch (C) verdict survives the methodology fix. The reasoning paragraph above ("Cohen's d prior comes from cisek-2013...") was anchored on pure-culture O157 results that don't generalize to the heterogeneous "STEC class" in this dataset (O157 + O121 + O103).

**Per-strain subhypothesis: REJECTED.** All 3 STEC strains do NOT shift together at any band. O157H7 has the lowest 1338 AUC; O103H2 has the highest — within-STEC range exceeds the between-class shift. The briefing's "STEC vs Non-STEC" framing is, on this dataset's labels, three different patterns being averaged. Cisek-2013's O157:H7-specific finding doesn't generalize to O121/O103.

### Implications for Stage 2 (band_features module)

**Re-anchor.** Stage 2 was pre-committed (plan/14 §5) to building features around the literature catalog with the primary triple as headline. Updated anchor:

1. **Primary anchor: 800–1200 cm⁻¹ LPS chain region.** Integrate the whole region as one feature (E. coli vs Salmonella), plus narrow ±10 windows at 1050 (3-class top), 1117, 1194 (E. coli STEC↔Non-STEC top).
2. **Demote the literature triple to supporting features.** Still include 1338/1454/1658 AUCs and Lorentzian fits — the negative result is publishable, and the writeup story benefits from "here are the literature bands; here is what they predict; here is what we actually see."
3. **Lorentzian peak-shift probe (plan/14 §6.6) gains importance.** The 1338/1454/1658 nulls may partly reflect peak *position drift* — the literature peak at 1454 might be sitting at 1448 in our spectra, and a ±10 window around 1454 misses it. Fitting the center, not integrating the bin, recovers this if so.
4. **Macromolecule radar plot (plan/14 §6.4) is more interesting given this finding.** Radars across STEC↔Non-STEC should look similar (literature bands carry little signal); radars between E. coli and Salmonella should diverge sharply on the LPS axis.

### Operational decisions

1. **Stage 2 anchor changes.** Headline features = the LPS region 800–1200 integrated AUC + narrow windows at 1050 / 1117 / 1194 (the actual top discriminators). Literature triple kept as supporting features for the negative-finding writeup.
2. **Macromolecule radar (plan/14 §6.4) and Lorentzian peak-fit (§6.6) gain priority** — radar tests whether literature-grouped features carry any signal at all; peak-fit tests whether the triple nulls are actually peak-position drift.
3. **Active stage memory updated** ([[atlas-band-chemistry-roadmap]]): Stage 1 → done; Stage 2 anchor revised before kickoff.
4. **Plan/14 §2.3** ("primary triple as headline test bands") needs to be re-framed: keep the catalog as-is, but add a note pointing to this finding so future readers see "the catalog includes literature bands; here is the file-level evidence that they don't replicate on Atlas data."

---

## Artifacts

- `scripts/run_stage1_band_stats.py`
- `outputs/band_chemistry/01_bacteria_only_anova.png`
- `outputs/band_chemistry/01_bacteria_only_anova_top30.csv`
- `outputs/band_chemistry/01b_ecoli_only_ttest_top30.csv`
- `outputs/band_chemistry/02_primary_triple_violin.png`
- `outputs/band_chemistry/02_primary_triple_per_strain.png`
- `outputs/band_chemistry/02_primary_triple_stats.csv`
- `outputs/band_chemistry/03_stage1_summary.{md,json}`
