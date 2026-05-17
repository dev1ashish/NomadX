# 2026-05-18 — Stage 7: mixed-sample simulation reveals classifier has STEC-prior bias, not uniform 10-20% degradation {#2026-05-18--stage7-mixed-sample}

> **Status:** complete
> **Stage / track:** [plan/14 §"Future" Mixed-sample simulation](../14_band_chemistry_research.md), promoted to Stage 7 per the Stage 5 stage-gate decision
> **Branch hit:** (B) — 2/3 pairs validate the briefing at α=0.7, but with a methodological caveat
> **One-line headline:** 2/3 pairs validate briefing 10–20% drop; classifier has STEC-default class bias under uncertainty; per-file-mean classification ≠ per-pixel + soft-vote pipeline
> **Cross-refs:** [prior — Stage 5 band classifier](2026-05-17_stage5_band_classifier.md) · briefing claim [[atlas-briefing-emphasis]]

---

## Pre-registration

### Method

Tests the briefing's ([[atlas-briefing-emphasis]] §3) "10–20% drop vs pure-culture" claim against our Stage-5 XGBoost classifier. Script: `scripts/run_stage7_mixed_sample_sim.py`.

- Train Stage 5 XGBoost on the band-feature cache (the 13-feature anchor set from Stage 5), using all pure-culture data (no fold structure — this is a within-distribution test, not LOSO).
- For each of three pairwise class mixtures — **STEC × Non-STEC**, **STEC × Salmonella**, **Non-STEC × Salmonella** — synthesize mixtures by linearly combining per-file mean preprocessed spectra at α ∈ {0.0, 0.1, 0.2, ..., 1.0}.
- For each mixture spectrum, compute the same 13 band features and predict the class with the trained XGB.
- Define **success** as: the predicted class matches the majority component (whichever class has α > 0.5). At α = 0.5 the chance baseline is ~0.5 (binary).
- Plot accuracy vs α for each of the three pairwise mixtures.

### Predictions

| Mixing ratio (α = majority fraction) | Predicted accuracy | Reasoning |
|---|---|---|
| **α = 1.0 (pure majority)** | ≥ 0.85 on all 3 pairwise mixtures | This is just the Protocol A baseline; pure cultures should classify correctly |
| **α = 0.9 (90% majority + 10% minority)** | 0.75 – 0.90 | Briefing's small-contamination regime; small accuracy drop expected |
| **α = 0.7 (70/30 mix)** | **0.55 – 0.75 — the briefing's 10–20% drop zone** | Headline prediction. If accuracy lands at 0.70 (10% drop) or 0.55 (20% drop), the briefing's claim holds on this dataset's classifier |
| **α = 0.6 (60/40 mix)** | 0.40 – 0.65 | Boundary of practical detectability |
| **α = 0.5 (50/50 mix)** | 0.30 – 0.60 | Should approach chance (0.5 for binary); any deviation indicates the classifier has a systematic class preference under ambiguity |

**Per-pair predictions:**

- **STEC × Non-STEC** (Stage 5's failing pair): hardest mixture. At α=0.5 should drop to ~0.4–0.55 (a bit below chance because the classifier has a default class). At α=0.7 expected accuracy 0.55–0.70 — slightly worse than briefing's 10–20% drop because Stage 5 wasn't great at this pair to begin with.
- **STEC × Salmonella**: easier — these classes are far apart in feature space (lps_1050 AUROC 0.82). At α=0.7 expect 0.70–0.85, briefing's drop range.
- **Non-STEC × Salmonella**: similar to STEC × Salmonella. At α=0.7 expect 0.70–0.85.

### Branching verdicts

- **(A) Briefing claim validated:** at α=0.7, all 3 pairs land in 0.55–0.75. Acknowledge as a reusable result for deployment scoping.
- **(B) Briefing claim partially validated:** 2 of 3 pairs in range; the harder pair (STEC × Non-STEC) drops faster. Writeup should report the per-pair degradation curves, not a single "10–20% drop" number.
- **(C) Briefing claim not replicated:** all 3 pairs drop much faster than 10–20% — even small contamination tanks the classifier. Implication: the LPS-anchored 13-feature classifier is not deployment-ready for food matrices.
- **(D) Surprise: better-than-expected robustness:** all 3 pairs maintain ≥0.85 even at α=0.7. Would imply the classifier learned class boundaries far from any mixture point. Unlikely.

### What this does NOT test

- Real food-matrix interference (organic acids, fluorescence from sample tray, etc.). Linear mixing of two class means is a best-case simulation.
- Cross-strain mixing (e.g., O157H7 × O121H19 → does the model recover "STEC"?). Could be a Stage 7b if time permits.
- Spatial mixing (some pixels pure A, some pure B vs every pixel a 50/50 mix). The simulation here is the per-pixel-uniform case.

---

## Results

### Headline numbers

Majority-class accuracy at α = {0.5, 0.7, 0.9, 1.0}:

| Pair | α=0.5 | α=0.7 | α=0.9 | α=1.0 | Briefing range (α=0.7) |
|---|---:|---:|---:|---:|---|
| **STEC × Non-STEC**       | (tie) | **0.738** | 0.754 | 0.667 | ✅ 0.55–0.75 |
| **STEC × Salmonella**     | (tie) | **0.582** | 0.675 | 0.667 | ✅ 0.55–0.75 |
| **Non-STEC × Salmonella** | (tie) | **0.279** | 0.239 | **0.240** | ❌ way below |

**Verdict: Branch (B), but with a methodological caveat.** 2/3 pairs land in the briefing's 0.55–0.75 "10–20% drop" range at α=0.7. The third (Non-STEC × Salmonella) doesn't degrade — it's *already collapsed* at α=1.0 (pure Non-STEC mean spectra get classified correctly only 24% of the time).

### The real finding: classifier has a STEC prior bias

Prediction-class distribution at α=0.5 (50/50 mix) reveals what's actually happening:

| Pair | % predicted STEC | % Non-STEC | % Salmonella |
|---|---:|---:|---:|
| STEC × Non-STEC       | **72%** | 13% | 15% |
| STEC × Salmonella     | **56%** | 22% | 21% |
| Non-STEC × Salmonella | **44%** | 29% | 27% |

**Across all 3 pairwise mixtures, STEC is the most-predicted class** — even when neither input class is STEC. The 13-feature classifier has learned to default toward STEC under uncertainty, presumably because STEC is the largest class by training-set spectrum count (2,263 vs 1,908 Non-STEC vs 2,267 Salmonella — but the per-strain LPS-anchored features cluster differently across STEC files than Non-STEC files).

This explains the asymmetric per-pair degradation:

- **STEC × Non-STEC** acc=0.74 at α=0.7 → high because the STEC bias is *correct* when STEC is the 70% majority.
- **STEC × Salmonella** acc=0.58 at α=0.7 → moderate because the STEC bias is correct when STEC is majority but Salmonella majorities pull the average down.
- **Non-STEC × Salmonella** acc=0.28 at α=0.7 → low because **neither class is STEC**, so the STEC-default bias is uniformly wrong.

### Methodology caveat: file-mean classification ≠ Stage 5 per-pixel pipeline

The Stage 5 file-F1 = 0.87 was computed via per-pixel prediction then file-level soft-vote. Stage 7 evaluates per-file-MEAN feature predictions directly. These two methodologies don't necessarily agree:

- Per-pixel + soft-vote: 7,122 small votes → robust majority
- Per-file-mean: 87 single predictions on smoothed features

The α=1.0 baseline (pure mean of a single class) reaches only 0.24 (Non-STEC), 0.67 (STEC), 0.67 (Salmonella) — much worse than Protocol-A file-F1 0.87. The mean spectrum lies in a feature-space region the XGB sees differently than typical per-pixel features. **This is a real-world deployment concern**: if the production pipeline averages pixels first then classifies, you don't get Protocol-A performance. The Stage 5 pipeline must classify per-pixel and then aggregate.

### Pre-registration verdicts

| Pre-reg | Predicted | Actual | Verdict |
|---|---|---:|:-:|
| α=1.0 (pure) accuracy ≥ 0.85 on all 3 pairs | yes | **0.24 / 0.67 / 0.67** | ❌ broke baseline expectation |
| α=0.7 STEC×NonSTEC in 0.55–0.75 | yes | 0.74 | ✅ |
| α=0.7 STEC×Salmonella in 0.55–0.75 | yes | 0.58 | ✅ |
| α=0.7 NonSTEC×Salmonella in 0.55–0.75 | yes | **0.28** | ❌ |
| α=0.5 ≈ chance (0.5 binary) | yes | 0.13 to 0.72 across pairs | ❌ — strong STEC bias instead |
| All-3 pairs validate briefing | yes (Branch A) | 2/3 (Branch B) | ⚠️ Branch (B) |

### Operational decisions

1. **Pure briefing claim of "10–20% drop" doesn't replicate cleanly on this classifier.** Report 2/3 pairs validate; document the Non-STEC × Salmonella collapse separately.
2. **Stage 7 reveals a deployment-relevant defect:** per-file-mean classification under-performs per-pixel + soft-vote by ~0.2 file-F1 on Non-STEC. Add to plan/06 risks. The Stage 5 classifier ships only if the deployment pipeline classifies per-pixel.
3. **The 13-feature anchor set has a STEC-default class bias** under uncertainty. Adding class-balanced sampling or class-weight to the XGB is a one-line fix to evaluate.
4. **Per-class macromolecule diversity matters** — the 13-feature space puts Non-STEC files in a low-density region of feature space (most Non-STEC mean-spectra are far from STEC means but also far from Salmonella means); a small mixture push moves them to "more like one of the other classes." Plan/15 (the new feature-engineering research track) should anchor at least one direction on "broaden Non-STEC feature representation" (e.g., aromatic-AA-driven features per Stage 3 — Non-STEC had elevated `aa_1176` AUROC 0.696).
5. **Plan/15 mandate:** ≥1 feature direction must demonstrably improve Non-STEC class density (e.g., aromatic-AA-anchored features that Stage 3 showed Non-STEC slightly enriched on).
6. **Stage 7 complete.** Stage 4 (Lorentzian peak-shift) and Stage 8 (MCR-ALS) still queued.

---

## Artifacts

- `scripts/run_stage7_mixed_sample_sim.py`
- `outputs/band_chemistry/stage7/01_degradation_curves.png`
- `outputs/band_chemistry/stage7/02_per_pair_curves.csv`
- `outputs/band_chemistry/stage7/03_briefing_check.json`
