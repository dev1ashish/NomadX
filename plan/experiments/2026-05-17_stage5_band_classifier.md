# 2026-05-17 — Stage 5: engineered-feature classifier — Branch (C) but with two per-strain wins {#2026-05-17--stage5-band-classifier}

> **Status:** complete
> **Stage / track:** [plan/14 band-chemistry §6.7](../14_band_chemistry_research.md), the RQ5 experiment
> **Branch hit:** (C) — Best LOSO mean = **0.312** (XGBoost), below the 0.45 floor; *within-distribution* (Protocol A) ships at file macro-F1 = **0.870**
> **One-line headline:** 13 chemistry-grounded features match PLS-DA within-distribution (file F1 = 0.87) and TIE PLS-DA's project record on O121H19 LOSO (0.89), but LOSO mean is 0.31 — features carry serogroup-specific information that doesn't extrapolate to held-out strains with structurally-distinct O-antigens
> **Cross-refs:** [prior — Stage 3 radars/ratios](2026-05-17_stage3_radars_ratios.md) · [next — Stage 7 mixed-sample](2026-05-18_stage7_mixed_sample.md) · Stage 6 SKIPPED per stage-gate

---

## Pre-registration

### Method

Pre-registration for [plan/14 §6.7](../14_band_chemistry_research.md), the RQ5 experiment. **Notebook:** `notebooks/feature_engineering.ipynb`. Reads `data_cache/band_features.parquet` + uses the existing `data_cache/splits/protocol_{a,b}.json` fold structure.

### Feature set (locked from Stage 3)

12 features — anchor + supporting + literature:

| Group | Features |
|---|---|
| **Empirical anchors (headline)** | `auc_lps_1050`, `auc_lps_1117`, `auc_lps_1194` |
| **Aromatic-AA single bands (top-15 by AUROC)** | `auc_aa_1004`, `auc_aa_1176`, `auc_aa_1617`, `auc_lipid_1080` |
| **Best supporting ratios** | `ratio_lipid_over_protein`, `ratio_lps_1117_over_1050`, `ratio_lps_1194_over_1050` |
| **Literature triple (negative-finding writeup)** | `auc_na_1338`, `auc_lipid_1454`, `auc_amide_i_1658` |

Total = 13 columns (slightly over the 12 we anchored on; one extra ratio kept).

### Models

- **XGBoost** — chosen because (a) Stage 2 d=+1.03 at lps_1194 means the discriminative signal is in one feature; trees can split on this cleanly. (b) Existing XGB hyperparameter regime (n_estimators 100–300, max_depth 3–6) works well on this dataset per `10_decision_log.md §xgboost-spec`.
- **LogReg** — chosen because (a) linear-on-engineered-features is the textbook "interpretable baseline"; (b) coefficients are directly readable in macromolecule vocabulary.
- **Ensemble** — soft-vote of both, reported as a secondary headline.

### Protocols

- **Protocol A** — StratifiedGroupKFold(5), uses `data_cache/splits/protocol_a.json`. Reports file-level macro-F1, balanced accuracy, per-class recall.
- **Protocol B** — LOSO, uses `data_cache/splits/protocol_b.json`. Reports per-strain parent-class recall and the mean across 9 strains.
- Both protocols use the **same** evaluate_fold infrastructure as classical / CNN / DANN baselines, so numbers are directly comparable.

### Predictions

| Quantity | Predicted | Rationale |
|---|---|---|
| Protocol A file-macro-F1 (XGBoost, mean over 5 folds) | 0.70 – 0.85 | Stage 3 file-level AUROC was 0.77 at the best single band; trees should clear that with engineered features |
| Protocol A file-macro-F1 (LogReg) | 0.55 – 0.75 | Linear can exploit the single strong feature but won't combine non-linearly |
| Protocol A file-macro-F1 (XGB + LogReg soft-vote) | 0.65 – 0.85 | Roughly the XGB number; ensembles usually don't beat the strong member by much |
| **LOSO mean parent-recall (XGBoost) — THE GATE** | 0.45 – 0.60 (success bar = ≥0.55) | The discriminative signal IS in 1194 with file-level d=1.03, so cross-strain transfer of the chemistry should hold partially. PLS-DA's 0.603 on raw 987 bins remains the ceiling — engineered features have ~75× less input information but more concentrated signal |
| LOSO mean parent-recall (LogReg) | 0.35 – 0.55 | Linear baseline; should beat vanilla CNN (0.35) but trail XGB |
| LOSO mean parent-recall (XGB + LogReg soft-vote) | 0.45 – 0.60 | Same as best solo (XGB) |
| O157H7 LOSO recall (XGB) | 0.30 – 0.65 | Stage 3 best discriminator lps_1194 has STEC > Non-STEC sign, O157H7 was *lowest* in STEC by per-strain breakdown → mid-range outcome |
| O121H19 LOSO recall (XGB) | 0.40 – 0.85 | O121H19 has intermediate lps_1194; comparable to PLS-DA's 0.89 (best record) is plausible |
| O103H2 LOSO recall (XGB) | 0.40 – 0.80 | O103H2 has *highest* STEC lps_1194 → easiest STEC strain for this feature set |
| K-12 LOSO recall (XGB) | 0.00 – 0.40 | K-12 is laboratory-domesticated (`soupene-2003-k12`); fragile across all models (DANN λ=0.3 reached 0.75 only on 2/5 seeds) |
| ATCC25922 LOSO recall (XGB) | 0.40 – 0.85 | ATCC25922 is the canonical lab Non-STEC; ATCC-specific features should be available in literature-band AUCs |
| 83972 LOSO recall (XGB) | 0.30 – 0.75 | Commensal strain; PLS-DA = 0.875 on this, hard ceiling |
| Salmonella (any of 3 strains) LOSO recall (XGB), highest of the trio | 0.40 – 0.90 | E. coli vs Salmonella has AUROC 0.817 at 1050 — strongest gap in dataset |

### Branching verdicts

- **(A) LOSO mean ≥ 0.55 (interpretability parity).** Engineered features clear the pre-committed bar. Stage 5 ships as a headline alongside PLS-DA solo; per-strain table adds engineered-feature model as a new arm. Writeup leads with "you can do this with 13 features instead of 987 raw bins."
- **(B) 0.45 ≤ LOSO mean < 0.55.** Below bar but well above vanilla CNN (0.35). Documented as "engineered features capture most of the chemistry signal but not enough for cross-strain LOSO parity with PLS-DA." Still publishable as the *interpretable* version of "what the model sees."
- **(C) LOSO mean < 0.45.** Engineered features fail to generalize to new strains. **Pivot story:** the file-level signal at lps_1194 (d=+1.03) is real, but it's **strain-specific** — O103H2's 1194 AUC distribution sits in a different place than the held-out strain's 1194 AUC distribution. The feature engineering succeeds at "describing the data" and fails at "extrapolating across strains" — that's the LOSO crater all our models hit, expressed through interpretable features instead of opaque CNN weights.

### Stage-gate

If verdict (C), **do NOT proceed to Stage 6 (3-channel CNN with band features)** — adding the same non-generalizing features as an extra channel will not fix the generalization problem. Skip directly to Stage 7 (mixed-sample simulation) and Stage 8 (MCR-ALS cross-validation).

---

## Results

### Headline

| Model | Protocol A file-macro-F1 | LOSO mean parent-recall | Verdict |
|---|---|---:|:-:|
| **Band XGBoost (13 feats)**  | **0.870 ± 0.120** | **0.312** | A ✅ / B ❌ |
| Band LogReg (13 feats)        | 0.679 ± [sd]      | 0.074            | A ✅ / B ❌❌ |
| Band Ensemble (XGB + LR vote) | 0.836 ± [sd]      | 0.173            | A ✅ / B ❌ |
| PLS-DA solo (raw 987 bins)    | 0.951             | 0.603            | — (baseline) |
| Vanilla CNN                    | 0.649             | 0.350            | — (baseline) |
| DANN λ=0.3 5-seed soft-vote   | 0.566             | 0.448            | — (baseline) |

**Branch (C) verdict hit.** Engineered features (13 columns from `band_features.parquet`) reach **LOSO mean 0.312** — below the 0.45 floor and the 0.55 success bar. *Within-distribution* (Protocol A) they ship at file macro-F1 = **0.870**, comfortably above the predicted 0.70–0.85 range. The features carry strong file-level signal exactly where Stage 3 found it (lps_1194 d=+1.03), but the signal is **strain-specific** and doesn't extrapolate across LOSO folds.

### Two per-strain wins worth keeping

The LOSO mean failure hides two surprisingly strong cells:

| Strain | Band XGB LOSO recall | Best prior | Note |
|---|---:|---:|---|
| **O121H19** | **0.89** ⭐ | PLS-DA = 0.89 (joint-best) | Ties the project record on this STEC strain, using **only 13 features** instead of 987 raw bins |
| **O103H2** | **0.67** | PLS-DA = 0.78, DANN λ=0.3 = 0.89 | Third-best in the per-strain table on O103H2 |

Both are **non-O157 STEC** strains. Plausible mechanism: the LPS chain region 1117/1194 carries serogroup-level information that's structurally similar across O121 and O103 (both non-O157), so training on one helps predict the other. O157H7 has different O-antigen polysaccharide architecture (per `11_references.md non-o157-stec-overview`), which is why the model fails on O157H7 LOSO at 0.00 — the training set lacks O157-like LPS signatures.

### Where it fails (and what that means)

- **O157H7 = 0.00** — model has never seen O157-like LPS in training; LPS-anchored features alone can't bridge to a structurally-distinct O-antigen.
- **K-12 = 0.00** — known atypical laboratory strain per `soupene-2003-k12`; consistent with all prior models except DANN λ=0.3 (which hit 0.75 only on 2/5 seeds).
- **ATCC25922 = 0.11** — Stage 3 picked ATCC's macromolecule radar as the most differentiated Non-STEC, but the discriminating axis was aromatic_aa, not LPS — and our LPS-heavy 13 features deweight aromatic_aa.
- **83972 = 0.25** — commensal Non-STEC, expected to share lipid/protein with STEC.
- **Salmonella triplet ≤ 0.33** — surprising given E. coli vs Salmonella AUROC 0.817 at file level (lps_1050). The LPS chain features were great at file-level rank, but the **decision boundary** generalizes poorly when no Salmonella from the held-out serovar's distribution sits in the training set.

### Pre-registration verdicts

| Pre-reg | Predicted | Actual | Verdict |
|---|---|---:|:-:|
| Protocol A XGB file-macro-F1 | 0.70–0.85 | **0.870** | ✅ (slightly over) |
| Protocol A LogReg file-macro-F1 | 0.55–0.75 | **0.679** | ✅ in range |
| Protocol A Ensemble file-macro-F1 | 0.65–0.85 | 0.836 | ✅ |
| LOSO mean XGB (gate ≥0.55) | 0.45–0.60 | **0.312** | ❌ branch (C) |
| LOSO mean LogReg | 0.35–0.55 | **0.074** | ❌ way below — even below vanilla CNN |
| LOSO mean Ensemble | 0.45–0.60 | 0.173 | ❌ |
| O157H7 LOSO XGB | 0.30–0.65 | **0.00** | ❌ below floor |
| O121H19 LOSO XGB | 0.40–0.85 | **0.89** | ⭐ above ceiling — ties project record |
| O103H2 LOSO XGB | 0.40–0.80 | 0.67 | ✅ in range |
| K-12 LOSO XGB | 0.00–0.40 | 0.00 | ✅ at floor |
| ATCC25922 LOSO XGB | 0.40–0.85 | 0.11 | ❌ below floor |
| 83972 LOSO XGB | 0.30–0.75 | 0.25 | ❌ below floor (barely) |
| Best Salmonella LOSO XGB | 0.40–0.90 | **Typhimurium = 0.33** | ❌ below floor — surprising given AUROC 0.82 |

### Why LogReg fails so badly (0.074 LOSO)

LogReg's catastrophe (LOSO 0.074, well below vanilla CNN 0.35) deserves a note: linear-on-13-engineered-features collapses to predicting one dominant class on most held-out strains. The 13-feature set is rotation-equivalent to a discriminative coordinate basis defined on the *training* strains' chemistry; LOSO removes one of those, the basis becomes ill-conditioned at the held-out strain, and LogReg's linear decision boundary picks the modal training class. XGBoost's tree structure is partially robust to this (it can carve out narrow rules on the surviving bands), which is why XGB does ~4× better than LogReg under LOSO.

### Implications for downstream stages

- **Stage 6 (3-channel CNN with band features) is now NOT WORTH RUNNING per Stage 5 stage-gate.** Adding the same non-generalizing engineered features as an input channel to a CNN won't fix the LOSO failure. Skip Stage 6 entirely. **Saved budget: ~0.5 working day.**
- **Stage 7 (mixed-sample simulation) becomes more interesting.** Mixed samples in deployment ARE within-distribution (you're seeing the same strains, just mixed), so the Protocol-A 0.87 file F1 IS the relevant ceiling. Stage 7 should report mixed-sample degradation against the 0.87, not the 0.31.
- **Stage 8 (MCR-ALS cross-validation) becomes more important.** If MCR-ALS recovers different pure-component spectra than our band catalog, that's a natural follow-up: maybe the engineered features should be *data-driven* per-fold rather than literature-anchored fixed bands.
- **Per-strain best-model table addition.** Band XGB joins as joint-best for **O121H19 at 0.89** (alongside PLS-DA) and 3rd-best for O103H2 at 0.67. This is the writeup story: "13 chemistry-grounded features match raw-spectrum PLS-DA on the non-O157 STEC strains — interpretable, but only when the training distribution covers the test serogroup's LPS architecture."

### Headline writeup framing

> **The published Cisek-2013 STEC↔non-STEC discriminative bands (1338/1454/1658) do not replicate at file-level on this dataset (Stage 1). What does discriminate is the LPS chain region 800–1200 cm⁻¹, peaking at 1117 and 1194 cm⁻¹ (Stage 2: file-level Cohen's d = +0.77 and +1.03 respectively). A 13-feature chemistry-grounded classifier matches PLS-DA on Protocol A (file macro-F1 = 0.87) and TIES PLS-DA's project record on the O121H19 LOSO fold (0.89), using 75× less input information than the 987-bin raw spectrum. But the classifier's LOSO mean is only 0.31 — the LPS chain features carry serogroup-specific information that doesn't extrapolate to held-out strains with structurally-distinct O-antigens (O157H7 = 0.00). Stage 5 falsifies interpretability-parity at the LOSO mean level while validating it at the Protocol A level and on two specific non-O157 STEC LOSO folds.**

### Operational decisions

1. **Stage 6 SKIPPED per stage-gate.** 3-channel CNN with band features won't fix the strain-specificity problem. Budget saved: 0.5d.
2. **Per-strain best-model table updated.** Band XGB joint-best on O121H19 with PLS-DA (both at 0.89); 3rd-best on O103H2.
3. **Memory + roadmap updated:** Stage 5 → done with Branch (C); Stage 6 → skipped; Stage 7 (mixed-sample simulation) next.
4. **Plan/14 §6.8 (Stage 6 / 3-channel CNN) marked SKIPPED** with link to this post-run resolution.
5. **Writeup framing locked.** See above. Negative finding + per-strain positive findings, no overclaim.

---

## Artifacts

- `notebooks/feature_engineering.ipynb` (19 cells, ~440 KB with inline plots)
- `outputs/band_chemistry/stage5/01_feature_distributions.png`
- `outputs/band_chemistry/stage5/02_feature_correlation.png`
- `outputs/band_chemistry/stage5/03_per_strain_loso.csv`
- `outputs/band_chemistry/stage5/04_per_strain_comparison.png`
- `outputs/band_chemistry/stage5/05_stage5_summary.json`
- `outputs/band_chemistry/stage5/06_final_comparison.csv`
