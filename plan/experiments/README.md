# `plan/experiments/` — per-experiment log

> **Purpose.** One file per experiment. Each file holds the full lifecycle: pre-registration, method, results, verdict, artifacts.
> **Replaces.** The bulk of `plan/07_findings.md` (post-mortems) and `plan/08_expectations.md` (pre-regs) — those are now thin stubs that redirect here.
> **Why.** The monolithic logs grew past 1500 lines each. Splitting per-experiment means:
> - adding a new experiment = one new file, never editing the giants
> - reading one experiment = ~80–200 lines, not 1500
> - pre-reg + findings live together so the lifecycle is in one place
> - back-references (the `{#anchor}` IDs the rest of the repo links to) are preserved in the stubs

---

## Conventions

- **Filename:** `YYYY-MM-DD_<short-slug>.md`. Same-date experiments take `_stage1`, `_stage2`, etc. when ordered, or unique slugs otherwise.
- **Anchor preservation.** Every experiment that used to have a `{#anchor}` in 07/08 keeps that anchor on its top H1, so external links still resolve via the stub.
- **Status.** One of `complete | running | pre-registered`. Update it inline; this is the only mutable field after writing.
- **Template.** Copy `_TEMPLATE.md` to start a new entry. Do not append to 07/08 — they are stubs now.
- **Adding to the index below:** one row, top of the list (newest first).

---

## Migrated experiments (newest first)

| Date | Stage / topic | Branch | One-line headline | File |
|---|---|---|---|---|
| 2026-05-18 | Feat-eng Stage 15F — full-feature LOSO classifier (production model) | (C) | **Branch C plateau.** Best=LogReg-L2 LOSO acc=0.436 (below 0.50 (B) bar). ATCC25922 + O121H19 = **0.889**; Typhimurium lift to **0.778**; K-12 = **0.000** (α-helix axis not MI-selected per fold). Pipeline deterministic — 5 seeds gave std=0.000. **Production artifact shipped** (~2.0 MB, LogReg pipeline + frozen MCR-ALS K=7 + ROI-PCA + SAM). | [2026-05-18_stage15f_full_classifier.md](2026-05-18_stage15f_full_classifier.md) |
| 2026-05-18 | Feat-eng Stage 15E — spatial / cross-pixel features | (C) | **STEC↔Non-STEC spatial heterogeneity null** (0 features \|d\|≥0.5, best 0.485). But **`spat_skew_lps_1117` d=+0.725 = new strong E.coli↔Salm axis** (Salmonella distribution symmetric, E. coli right-skewed). H2O sanity passes 4/4. R6 confirmed: 0/87 files clear ≥200 pixels → Moran's I / GLCM dropped | [2026-05-18_stage15e_spatial_features.md](2026-05-18_stage15e_spatial_features.md) |
| 2026-05-18 | Feat-eng Stage 15D — biology features (cyt + 2°-struct + PHB + Tyr/Trp + NA-conf) | (B) | **`bio_alpha_helix_score` d=−0.986 STEC↔Non-STEC** (strongest biology-grounded ratio in project); **PHB K-12 falsifier null** but K-12 shows a 2°-structure shift (3 features \|d\|≈0.55) — **first K-12-specific axis found** | [2026-05-18_stage15d_biology_features.md](2026-05-18_stage15d_biology_features.md) |
| 2026-05-18 | Feat-eng Stage 15C — MCR-ALS unmixing | (A) | **`mcr_C6_mean` d=−1.23 (STEC↔Non-STEC) — new strongest file-level discriminator in the project**, beating prior LPS anchor (1.03). 8/32 MCR features clear \|d\|≥0.5; 6 of 7 active components map to biology | [2026-05-18_stage15c_mcr_als_unmixing.md](2026-05-18_stage15c_mcr_als_unmixing.md) |
| 2026-05-18 | Feat-eng Stage 15B — DWT + ROI-PCA + SAM templates | (B) | SAM weaker than expected (0.69 AUROC) but **`pca_amide_PC3` d=+0.89 and `pca_lps_PC3` d=+1.03** add new headline STEC↔Non-STEC features (amide region was undertapped) | [2026-05-18_stage15b_dwt_pca_sam.md](2026-05-18_stage15b_dwt_pca_sam.md) |
| 2026-05-18 | Feat-eng Stage 15A — pseudo-Voigt + ROI + EMSC + derivatives | (B) | Pseudo-Voigt lifts fit success 0.2–37% → 60–89%; **d2_auc_lps_1194 d=−0.898 STEC vs Non-STEC** as new headline feature | [2026-05-18_stage15a_pseudovoigt_roi_emsc_derivatives.md](2026-05-18_stage15a_pseudovoigt_roi_emsc_derivatives.md) |
| 2026-05-18 | Band-chem Stage 7 — mixed-sample simulation | (B) | 2/3 pairs validate briefing 10–20% drop; classifier has STEC-default bias under uncertainty | [2026-05-18_stage7_mixed_sample.md](2026-05-18_stage7_mixed_sample.md) |
| 2026-05-17 | Band-chem Stage 5 — engineered-feature classifier | (C) | Branch C — LOSO 0.312; **but ties PLS-DA project record on O121H19 = 0.89 with 13 features** | [2026-05-17_stage5_band_classifier.md](2026-05-17_stage5_band_classifier.md) |
| 2026-05-17 | Band-chem Stage 3 — radars + ratios + best-1D AUROC | — | Ratios *don't* beat single-band AUCs at file level; literature direction (lipid/protein) partially validated | [2026-05-17_stage3_radars_ratios.md](2026-05-17_stage3_radars_ratios.md) |
| 2026-05-17 | Band-chem Stage 2 — `atlas/band_features.py` module + anchors | — | `auc_lps_1194` d=+1.03 — cleanest single-band STEC↔Non-STEC discriminator in the project | [2026-05-17_stage2_band_features.md](2026-05-17_stage2_band_features.md) |
| 2026-05-17 | Band-chem Stage 1 — published STEC triple test | (C) | Literature triple (1338/1454/1658) does NOT replicate at file-level; signal is in 800–1200 cm⁻¹ | [2026-05-17_stage1_stec_triple.md](2026-05-17_stage1_stec_triple.md) |

---

## Legacy entries (still in `07_findings.md` / `08_expectations.md`)

Not yet migrated. Listed for completeness; references still resolve via the original files.

| Date | Topic | Anchor in 07 |
|---|---|---|
| 2026-05-15 | DANN λ=0.3 5-seed verification | `#2026-05-15--dann-lam03-5seed-verification` |
| 2026-05-15 | DANN λ=0.1 5-seed robustness (headline revision) | `#2026-05-15--dann-5seed-robustness` |
| 2026-05-15 | DANN aug-regime sweep | `#2026-05-15--dann-aug-regime-sweep` |
| 2026-05-15 | Temperature-scaled soft-vote | `#2026-05-15--temperature-scaled-softvote` |
| 2026-05-15 | 4-architecture re-ensemble | `#2026-05-15--4-architecture-re-ensemble-fails-to-beat-plsda` |
| 2026-05-14 | 2nd-derivative channel CNN | `#2026-05-14--2nd-derivative-channel-second-best-loso` |
| 2026-05-14 | Patch=5 Transformer | `#2026-05-14--patch5-transformer-partially-confirms-blur-hypothesis` |
| 2026-05-14 | Per-strain λ-selector | `#2026-05-14--per-strain-lambda-selection-fails` |
| 2026-05-14 | Stacking meta-learner | `#2026-05-14--stacking-meta-learner-fails` |
| 2026-05-14 | Grouped-domain DANN | `#2026-05-14--grouped-domain-dann-rejects-hypothesis` |
| 2026-05-14 | DANN λ curve completed (λ=0.05 sweep) | `#2026-05-14--dann-lambda-curve-completed` |
| 2026-05-14 | DANN λ frontier (λ=0.1 vs λ=0.3) | `#2026-05-14--dann-lambda-frontier` |
| 2026-05-14 | DANN ablation — verdict (A) hit | `#2026-05-14--dann-ablation-clears-verdict-a` |
| 2026-05-14 | Transformer (patch=20) underperforms CNN | `#2026-05-14--transformer-underperforms-cnn` |
| 2026-05-14 | Soft-vote ensemble fails to clear PLS-DA | `#2026-05-14--ensemble-fails-to-clear-plsda` |
| 2026-05-14 | Memprobe v2 fires | `#2026-05-14--memprobe-v2-fires` |
| 2026-05-14 | CNN LOSO complementary pattern | `#2026-05-14--cnn-loso-complementary-pattern` |
| 2026-05-14 | CNN Protocol A underperforms classical | `#2026-05-14--cnn-protocol-a-underperforms-classical` |
| 2026-05-14 | CNN spec + underfit fixes | `#2026-05-14--cnn-spec-underfit-and-fixes` |
| 2026-05-14 | XGBoost complementary failure mode | `#2026-05-14--xgboost-complementary-failure-mode` |
| 2026-05-14 | Cal-date diagnostic | `#2026-05-14--cal-date-diagnostic-mixed-signal` |
| 2026-05-14 | Memorization probe (v1) | `#2026-05-14--memorization-probe-weak` |
| 2026-05-14 | LOSO per-strain pattern | `#2026-05-14--loso-per-strain-pattern` |
| 2026-05-14 | Classical results — GroupKFold vs LOSO | `#2026-05-14--classical-results-groupkfold-vs-loso` |
| 2026-05-14 | EDA findings (parser, splits, batch effects, ANOVA, etc.) | various `2026-05-14--*` anchors |

Migration of any of these → identical recipe. Copy section from 07 and 08, drop into `_TEMPLATE.md` shape, replace the 07/08 sections with a stub. Bulk migration on request.

---

## Cross-refs

- `00_status.md` is the **current snapshot** — what we ran and what we're doing now. It links into individual experiment files here.
- `10_decision_log.md` records *design-decision* changes (architecture, splits, hyperparams). Not the same as an experiment.
- `14_band_chemistry_research.md` / `15_feature_engineering_research.md` are *research plans* (what we intend to test, how features are defined). Per-experiment outcomes for those plans live here.
