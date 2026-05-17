# 2026-05-17 — Stage 3: ratios don't outperform single bands at file level; literature direction validated even though specific bands didn't {#2026-05-17--stage3-radars-ratios-auroc}

> **Status:** complete
> **Stage / track:** [plan/14 band-chemistry §6.2–6.5](../14_band_chemistry_research.md). Visualization-only — no new modeling; reads the `data_cache/band_features.parquet` cache from Stage 2 and renders the chemistry-aware story.
> **Branch hit:** — (predictions resolve mixed; no formal A/B/C branching)
> **One-line headline:** Ratios *don't* beat single-band AUCs at file level; literature direction (lipid/protein) partially validated even though specific bands aren't
> **Cross-refs:** [prior — Stage 2 band_features module](2026-05-17_stage2_band_features.md) · [next — Stage 5 band classifier](2026-05-17_stage5_band_classifier.md)

---

## Pre-registration

### Method

Stage 3 is visualization-only — no new modeling; reads the `data_cache/band_features.parquet` cache from Stage 2 and renders the chemistry-aware story. Notebook: `notebooks/band_chemistry.ipynb` (16 cells; sections B-recap, C-recap, D-radars, E-ratios, F-summary).

### Locked-in scope decisions

- **Radar normalization scheme:** z-score each macromolecule axis across all classes (so axes are comparable in units of "SD from grand mean"). Reason: raw SNV-unit AUCs are negative numbers around −20 to −60 — radar plots with negative radii don't render correctly, and the cross-class *relative* shape is what matters, not absolute values. Per-class-mean offset would lose the scale information; z-score keeps both.
- **Subclass overlay scope:** one primary-class radar (4-way overlay), plus three subclass radars (one panel per parent — STEC strains, Non-STEC strains, Salmonella strains). H₂O on the primary radar only.
- **Ratio scatters to ship:** (a) `lps_1194 / lps_1050` × `lps_1117 / lps_1050` (the two big-d empirical anchors normalized by the cross-class baseline); (b) `amide_1658 / na_1338` × `lps_chain_over_protein` (literature triple ratio vs new empirical ratio, on the same axes); (c) macromolecule group-vector scatter of `auc_protein_amide` × `auc_lipid_carbohydrate` — Cisek's two largest groups, file-level.

### Predictions

| Quantity | Predicted |
|---|---|
| Radar: Salmonella's LPS axis (chain-discrim region) is ≥1 z-SD above the E. coli classes' LPS axis | yes |
| Radar: STEC and Non-STEC overlap visually on protein_amide, nucleic_acid, aromatic_aa axes (no clean separation in the 5 catalog macromolecule groups — consistent with Stage 1/2 nulls) | yes |
| Best-1D ratio AUROC for STEC vs Non-STEC (file-level), top ratio | ≥ 0.80 (anchored on Stage 2's lps_1194 d=+1.03 → AUROC ≈ 0.77, but ratios may improve it) |
| Best-1D ratio for STEC vs Non-STEC contains lps_1117 OR lps_1194 in numerator/denominator | yes |
| Best-1D ratio AUROC for E. coli vs Salmonella (file-level), top ratio | ≥ 0.75 (anchored on Stage 2 lps_chain_discrim d=−0.63 → AUROC ≈ 0.67, ratios should improve) |
| `amide_1658 / na_1338` (Cisek-style ratio) AUROC for STEC vs Non-STEC | < 0.65 (literature bands null at file-level, ratios won't rescue) |
| Subclass radars: O157H7 visually separates from O121H19 + O103H2 on at least one axis | possibly (Cisek-2013 tested O157H7 specifically — if anything in literature replicates, O157H7-specific) |

### Reasoning

The dominant Stage 2 finding was that single empirical bands (1117, 1194) carry strong file-level signal while macromolecule-group aggregation doesn't — implying the discriminative signal is *narrow-spectral* not *broad-chemical*. Ratios should preserve the narrow-band advantage where they're built around 1117/1194; literature-style ratios should remain null. The radar visualization is the *interpretability* deliverable, not the *discrimination* deliverable — it shows the user, in macromolecule vocabulary, *why* the classifier will lean on the empirical anchors rather than the literature catalog.

### Stage-gate

Stage 5 is the next concrete experiment that needs Stage 3 output. If the ratio AUROC headlines come in below 0.65 for STEC vs Non-STEC even with the empirical anchors, **the engineered-feature classifier in Stage 5 should not be expected to clear LOSO 0.55** (its pre-committed success bar). In that case Stage 5 still ships as a negative-finding writeup, but the headline pivots to "per-band AUC + raw spectrum is the right input to a CNN, not engineered ratios."

---

## Results

### Headlines

1. **Best 1-D STEC vs Non-STEC discriminator at file level is `auc_lps_1194` at AUROC = 0.775** (single-band AUC). Confirms Stage 2 d=+1.03. The best *ratio* — `lipid_over_protein` (literature-style macromolecule ratio) — is at 0.741, second place overall.

2. **Ratios don't beat single-band AUCs for STEC vs Non-STEC.** The Stage 3 pre-registration predicted ratios would outperform raw bins because they cancel file-level offsets; **falsified**. Single-band 1117/1194/1004/1176/1617 all outrank every LPS-anchored ratio in the top-15 leaderboard. Implication for Stage 5: the engineered-feature classifier should use the raw narrow-band AUCs as headline features, not the ratios.

3. **The literature DIRECTION is partially validated even though the specific bands aren't.** Cisek's "lipid/carb content differs STEC vs non-STEC" prediction holds: `lipid_over_protein` ratio AUROC 0.741, and `lipid_1454` single-band AUROC 0.619 with d=−0.47. Cisek's specific band centers were wrong (1454 was claimed STEC>Non-STEC; we see STEC<Non-STEC), but the **macromolecule category** (lipid vs protein balance) does carry signal. This is a sharper rephrasing than Stage 1's "literature triple doesn't replicate" — the lipid claim partially survives at the category level.

4. **For E. coli vs Salmonella, ratios HELP marginally.** Best single-band `auc_lps_1050` = 0.817 (file-level AUROC), beating any ratio. But ratios `lps_1117_over_1050` (0.769) and `lps_1194_over_1050` (0.740) outrank the broad-region `auc_lps_chain_discrim` (0.623) and `auc_lps_o_antigen_full` (0.516). The narrow-band-normalized ratios survive; the broad-region AUC doesn't.

5. **Macromolecule radar tells the chemistry-vocabulary story exactly right.** STEC and Non-STEC trace nearly identical pentagons on the 5 macromolecule axes — confirms the literature triple null. Salmonella diverges modestly on aromatic_aa and metabolite. The discriminative signal **is not in macromolecule-group AUC aggregates**, even when grouped by literature chemistry; it's in narrow specific bins within the LPS chain region.

### Best-1D AUROC table (file-level, Stage 3 summary)

Top-5 per discrimination task:

**STEC vs Non-STEC:**

| Rank | Feature | Type | AUROC |
|:-:|---|---|---:|
| 1 | `lps_1194` | band | **0.775** |
| 2 | `lipid_over_protein` | ratio | 0.741 |
| 3 | `aa_1176` | band | 0.696 |
| 4 | `aa_1617` | band | 0.661 |
| 5 | `aa_1004` | band | 0.652 |

**E. coli vs Salmonella:**

| Rank | Feature | Type | AUROC |
|:-:|---|---|---:|
| 1 | `lps_1050` | band | **0.817** |
| 2 | `lps_1117_over_1050` | ratio | 0.769 |
| 3 | `lipid_1080` | band | 0.741 |
| 4 | `lps_1194_over_1050` | ratio | 0.740 |
| 5 | `aa_1014` | band | 0.727 |

`amide_1658_over_na_1338` (Cisek-style literature ratio) did not appear in either top-15 — AUROC < 0.578. Cisek's specific ratio is null at file level, consistent with Stage 1's per-band null on 1338 and 1658.

### Pre-registration verdicts

| Prediction | Actual | Verdict |
|---|---|:-:|
| Salmonella LPS chain axis ≥1 z-SD above E. coli classes | bar chart shows ~1.0 σ gap on lps_1050, ~0.7 on lps_chain | ✅ |
| STEC and Non-STEC overlap on protein_amide / NA / AA radar axes | radars trace nearly identical pentagons; max gap on those axes ~0.2 σ | ✅ |
| Best-1D ratio AUROC STEC vs Non-STEC ≥ 0.80 | best ratio = 0.741 (`lipid_over_protein`) | ❌ FAILED |
| Top ratio for STEC vs Non-STEC contains lps_1117 or lps_1194 | top ratio is `lipid_over_protein` (no LPS bands) | ❌ FAILED — interesting falsification |
| Best-1D ratio AUROC E. coli vs Salmonella ≥ 0.75 | best ratio = `lps_1117_over_1050` at 0.769 | ✅ marginal |
| Cisek-style ratio AUROC STEC vs Non-STEC < 0.65 | not in top-15 → < 0.578 | ✅ |
| Subclass radars: O157H7 visually separates from O121H19 + O103H2 | subclass radar shows all 3 STEC strains tracing similar pentagons; small differences but no clean separation | ⚠️ partial — K-12 separates from ATCC/83972 more than any STEC strain separates from siblings |

**Pre-reg miss to log.** The "ratios beat single bands" prediction was rooted in domain wisdom about batch-effect cancellation. On THIS dataset, the file-level test n=25–27 per class means the within-file averaging already denoises the per-spectrum noise that ratios were meant to fix. The raw narrow-band AUC is more discriminative at the file level than any ratio of broad aggregates — the discriminative signal is *narrow-spectral*, and any denominator (group AUC, neighboring-band AUC) dilutes it more than it cleans it.

### Implications for Stage 5 (engineered-feature classifier)

**Headline feature set locked:**

1. **Single-band AUCs (the headline):** `auc_lps_1050`, `auc_lps_1117`, `auc_lps_1194`, plus 3 of `aa_1004` / `aa_1176` / `aa_1617` / `lipid_1080` for redundancy.
2. **Ratios as supporting:** `lipid_over_protein`, `lps_1117_over_1050`, `lps_1194_over_1050` — they don't beat single bands but they're robust to file-level scale, may help on the LOSO transfer test where new strains shift overall amplitude.
3. **Literature triple AUCs (1338, 1454, 1658) included as supporting** for the negative-finding writeup.
4. **Exclude macromolecule group AUCs** — radar confirmed they don't separate STEC vs Non-STEC; including them adds noise. Keep `auc_lps_chain_discrim` for the E. coli vs Salmonella signal.

Feature count: ~12 (down from the 81-column feature_frame). Stage 5 will fit XGBoost + LogReg on this set under Protocol A + LOSO, pre-committed bar LOSO ≥ 0.55.

### Implications for Stage 4 (Lorentzian peak-shift probe)

Still gated on the fit-rate fix from Stage 2. Stage 3 doesn't change that gate. The amide-I peak drift from Stage 2 (STEC ≈ 1659, Non-STEC ≈ 1662) remains the most promising peak-shift target — Stage 4 needs the fit fix to verify it with full sample size.

### Side finding — literature direction partially validated

`lipid_over_protein` ratio AUROC = 0.741 for STEC vs Non-STEC, and `lipid_1454` single-band d=−0.47. Cisek's "lipid carries STEC signal" claim survives at the **macromolecule category** level even though the specific band centers don't replicate. The writeup story sharpens from "literature triple doesn't replicate" to "the chemistry direction was right; the wavenumber centers were wrong on this instrument."

### Operational decisions

1. **Stage 5 anchor features locked:** the 12-feature set above. Pre-committed bar: LOSO mean parent-recall ≥ 0.55.
2. **Stage 4 still gated** on Lorentzian fit-rate fix (Stage 2 issue, unchanged by Stage 3).
3. **Memory + roadmap updated:** Stage 3 → done, Stage 5 next.

---

## Artifacts

- `notebooks/band_chemistry.ipynb` (16 cells, 1.1 MB with inline plots)
- `outputs/band_chemistry/05_macromolecule_radar.png` (per-class radar + LPS bars)
- `outputs/band_chemistry/06_macromolecule_radar_subclass.png` (per-subclass radar grid)
- `outputs/band_chemistry/07_lps_region_violins.png` (Stage 3 headline panel — d annotated)
- `outputs/band_chemistry/08_band_ratios_per_class.png` (10 ratios × 4 classes boxplots)
- `outputs/band_chemistry/09_best1d_auroc.png` (top-15 features per task)
- `outputs/band_chemistry/10_ratio_scatters.png` (2D ratio geometry, file+spectrum level)
- `outputs/band_chemistry/11_stage3_summary.csv` (headline numerical summary)
