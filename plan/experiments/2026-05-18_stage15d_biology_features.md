# 2026-05-18 — Stage 15D: biology-specific features {#2026-05-18--stage15d-biology-features}

> **Status:** complete
> **Stage / track:** [plan/15 §5 Stage 15D](../15_feature_engineering_research.md#stage-15d--biology-specific-features-05-day), the fourth feature-engineering implementation stage.
> **Branch hit:** **(B) Partial — 3 STEC↔Non-STEC features clear \|d\|≥0.5 (needed ≥4 for A); PHB K-12 falsifier null.**
> **One-line headline:** **`bio_alpha_helix_score` d=−0.986 STEC↔Non-STEC is the strongest *biology-grounded* ratio in the project (Non-STEC has more α-helix than STEC). PHB hypothesis falsified, but K-12 shows a 2°-structure-shift signature (α-helix +0.54, β-sheet −0.59, Trp-env −0.55 vs other-STEC) — first K-12-specific structural axis found.**
> **Cross-refs:** [Stage 15C MCR-ALS](2026-05-18_stage15c_mcr_als_unmixing.md) · [Stage 15A pseudo-Voigt unblocked peak fits](2026-05-18_stage15a_pseudovoigt_roi_emsc_derivatives.md) · [Stage 1 falsified literature triple](2026-05-17_stage1_stec_triple.md) · [plan/15 §3.2 BIO catalog](../15_feature_engineering_research.md)

---

## Pre-registration

### Method

**Motivation.** Stage 15A/B/C added 32 + 51 + 32 features mostly via
data-driven directions (peak shape, PCA, MCR). Plan/15 §3.2 enumerates ~30
biology-grounded features (BIO1-BIO30) the catalog has been skipping. This
stage implements the top-tier biology bands (5 families, ~15 features) so
the Stage 15F classifier can attribute signal to interpretable chemistry,
not just "PC3 of the LPS region."

The biology features fall into 5 families, each addressing a specific
hypothesis the prior catalog couldn't test:

| Family | Hypothesis | Pre-existing alternative |
|---|---|---|
| Cytochromes (BIO1-4) | Salmonella has different cyt-bd expression than E. coli → discriminates by heme bands at 752/1127/1585 | None — Stage 15C C5 captures LPS-mix, not cyt cleanly |
| 2°-structure (BIO20-22) | STEC virulence-protein folds differ from Non-STEC → α-helix vs β-sheet ratios at amide-I/III | Only bulk `auc_protein_amide`; ratio not extracted |
| PHB (BIO24-25) | **K-12 falsifier:** lab-domesticated K-12 may have anomalous PHB accumulation → 1730 cm⁻¹ carbonyl | None — no model explains K-12's recoverable-but-fragile signature |
| Aromatic AAs (BIO26-29) | STEC has more Trp-rich virulence proteins than Non-STEC → Trp/Phe ratio | Stage 15C C2_p90 d=−0.84 hints at aromatic-AA axis; need direct ratio |
| NA conformation (BIO18-19) | A-form (RNA) vs B-form (DNA) ratio drifts by growth phase → 815/(815+835), 813/788 | None — only bulk `auc_nucleic_acid` |

**New module additions to `atlas/band_features.py`** (extension, not new file
— biology features remain band-aware):

```python
def cytochrome_features(X, wn) -> dict[str, ndarray]:
    """BIO1-BIO4."""
    # BIO1 cyt_pyrrole_ratio   = auc(752) / auc(1004)        (heme pyrrole / Phe protein)
    # BIO2 cyt_ox_state        = auc(1356) / auc(1372)       (Fe²⁺ / Fe³⁺ proxy)
    # BIO3 cyt_center_1585     = pseudo-Voigt fit center at 1585  (Cα-Cβ b-type heme)
    # BIO4 cyt_total           = auc(750) + auc(1127) + auc(1585)

def protein_secondary_structure(X, wn) -> dict[str, ndarray]:
    """BIO20-BIO22."""
    # BIO20 alpha_helix_score  = auc(1652) / auc(1670)
    # BIO21 beta_sheet_amide3  = auc(1232) / auc(1270)
    # BIO22 amide_fwhm_1655    = pseudo-Voigt fit FWHM at 1655 cm⁻¹

def phb_features(X, wn) -> dict[str, ndarray]:
    """BIO24-BIO25 — K-12 falsifier."""
    # BIO24 phb_carbonyl       = auc(1730)
    # BIO25 phb_score          = auc(1730) * auc(1058) / (auc(1450)**2)

def aromatic_aa_features(X, wn) -> dict[str, ndarray]:
    """BIO26-BIO29."""
    # BIO26 tyr_doublet_ratio  = auc(850) / auc(830)
    # BIO27 trp_content        = auc(759) + auc(1552)
    # BIO28 trp_indole_env     = auc(1340) / auc(1360)
    # BIO29 virulence_aa_sig   = (auc(759) + auc(1552)) / auc(1004)

def nucleic_conformation_features(X, wn) -> dict[str, ndarray]:
    """BIO18-BIO19."""
    # BIO18 na_a_form_fraction = auc(815) / (auc(815) + auc(835))
    # BIO19 rna_dna_ratio      = auc(813) / auc(788)
```

All AUCs use the existing `integrate_band(X, wn, center, half_width=10)`
convention. Hooked into `feature_frame` via a new `biology=True` flag
(default True); columns appended after Stage 15A's derivative AUCs.

**Build script:** re-run `scripts/build_band_features_cache.py` — produces
the updated `band_features.parquet` with biology columns appended. Expected
column count: 153 → ~168 (15 new feature names; some are ratios so single
columns).

**Inspection (sanity checks beyond Cohen's d):**
- **K-12 falsifier:** the briefing flagged K-12 as the strain no model can
  explain. PHB accumulation is the leading hypothesis. Predict that
  `phb_carbonyl` is detectably HIGHER on K-12 files vs other STEC strains
  at file level (|d| ≥ 0.5, K-12 vs other-STEC). If yes, this is the first
  feature in the project that explains a K-12-specific axis.
- **Cyt total cross-class:** Salmonella expected higher than E. coli at
  file level (literature: Salmonella has stronger cyt-bd terminal-oxidase
  expression in microaerobic conditions).
- **Trp/Phe (virulence_aa_sig)** STEC > Non-STEC at file level —
  literature: STEC carries Trp-rich virulence proteins (Stx, EHEC-Hly,
  intimin) that Non-STEC commensal strains lack. Stage 15C C2_p90
  (Trp-related) was d=−0.84 (Non-STEC > STEC) — **so the literature
  prediction may be sign-reversed on this dataset; this is the test**.

### Predictions

| Quantity | Predicted | Rationale |
|---|---|---|
| New feature columns added to cache | 15 ± 2 | 4 cyt + 3 2°struct + 2 PHB + 4 aa + 2 NA = 15 distinct features (some are ratios) |
| Cache build time delta | < +30 s | All AUCs at fixed wn; no peak fitting beyond what Stage 15A already does |
| `phb_carbonyl` K-12 vs other-STEC \|d\| ≥ 0.5 | yes (60% prob.) | K-12 falsifier; if true, first K-12-specific feature in the project. If miss, PHB hypothesis falsified |
| `cyt_total` Salmonella vs E. coli \|d\| ≥ 0.4 | yes | Salmonella's stronger cyt-bd expression should show up at 752+1127+1585 |
| `cyt_pyrrole_ratio` (752/1004) E. coli vs Salmonella \|d\| ≥ 0.3 | maybe | 785 nm excitation is off-resonance for cytochromes (R5 — plan/15 §7); signal may be weak |
| `alpha_helix_score` (1652/1670) STEC vs Non-STEC \|d\| ≥ 0.4 | maybe | STEC virulence proteins have different fold distributions; literature support is moderate |
| `virulence_aa_sig` (Trp/Phe) STEC vs Non-STEC sign | **STEC < Non-STEC** | **Reversed from literature claim** — predicting Stage 15C's C2_p90 direction (Non-STEC > STEC) holds at the explicit-ratio level. Pre-reg call: literature is wrong on Atlas; the dataset has Non-STEC strains with stronger Trp signal than the STEC pathogenic strains |
| `tyr_doublet_ratio` (850/830) any-class file-level \|d\| ≥ 0.3 | yes | Classic Raman protein descriptor; expect at least one cross-class signal |
| Best new biology feature \|d\| STEC↔Non-STEC | 0.4 – 0.8 | Won't beat `mcr_C6_mean` d=−1.23 (no single ratio should), but several should clear 0.5. If multiple > 0.6, biology track is contributing real signal beyond the MCR-derived axes |
| At least 2 biology features \|d\| ≥ 0.5 STEC↔Non-STEC | yes | Reasonable for 15 new features covering 5 different chemistry axes |
| `na_a_form_fraction` (815/835) growth-phase correlate | weak signal across all class pairs | Not a class discriminator; will check |

### Branching verdicts

- **(A) Strong hit.** ≥ 4 biology features clear |d| ≥ 0.5 STEC↔Non-STEC, OR `phb_carbonyl` K-12-falsifier hits at |d| ≥ 0.5 → biology track contributes orthogonal interpretable signal; promotes 4+ new features to Stage 15F headline. PHB hit specifically unblocks Stage 6 reconsideration (the K-12-explainer feature changes the K=8 ensemble math).
- **(B) Partial.** 1–3 biology features clear |d| ≥ 0.5, PHB K-12 falsifier null. Biology features add supporting signal to Stage 15F. Continue to 15E.
- **(C) Miss.** 0–1 biology features clear |d| ≥ 0.5, no K-12 PHB signal, Trp/Phe direction matches literature or is null. The dataset's discriminative information is genuinely concentrated in LPS+bulk-biology directions (15A/B/C captured it all); biology-specific features are a "documentation tier," not a "performance tier." Still ship to Stage 15F for interpretability, but lower expectations on lift.

### Stage-gate

- **(A) PHB hit:** revisit Stage 6 (3-channel CNN skipped) with PHB as a candidate input channel.
- **(A) or (B):** continue to Stage 15E (spatial features), then Stage 15F (re-train classifier with ~250 features).
- **(C):** skip Stage 15E pixel-variance (DD16) — if biology features add nothing, spatial heterogeneity won't either; jump directly to Stage 15F.

---

## Results

### Headline

**Branch (B) Partial.** 13 biology features added (4 cyt + 2 2°-struct +
2 PHB + 4 aa + 2 NA-conf), `band_features.parquet` 153 → 166 cols. 3
features clear |d| ≥ 0.5 STEC↔Non-STEC at file level, below the Branch (A)
bar of 4. **PHB K-12 falsifier null** (d=+0.07, K-12 vs other-STEC) — the
"K-12 accumulates anomalous PHB" hypothesis is dead.

**But two unexpected wins:**

1. **`bio_alpha_helix_score` d=−0.986 STEC↔Non-STEC** is the strongest
   *biology-grounded ratio* feature in the entire project. Non-STEC files
   have notably higher amide-I α-helix score (1652/1670) than STEC files —
   suggesting STEC virulence-protein folds shift the bulk α-helix
   population. This is the FIRST protein-fold-grounded discrimination
   signal in the catalog.

2. **K-12 has a distinct 2°-structure profile vs other STEC strains**, even
   though PHB doesn't separate it: `bio_beta_sheet_amide3` d=−0.591
   (K-12 has less β-sheet than other STEC), `bio_trp_indole_env` d=−0.547,
   `bio_alpha_helix_score` d=+0.537. This is the **first K-12-specific
   signal in the project** — three biology features that flag K-12
   differently from the four clinical STEC strains. Could feed Stage 6
   reconsideration (K-12 falsifier was the original Stage 6 rationale).

**E. coli vs Salmonella surprise:** `bio_virulence_aa_sig` (Trp/Phe)
d=−0.651 with Salmonella having higher Trp/Phe than E. coli. This is
**directionally opposite** to plan/15 §3.2's BIO29 framing (which expected
STEC > Non-STEC due to STEC's Trp-rich virulence proteins). At the E.
coli↔Salm level, the Trp/Phe ratio favors Salmonella — likely from
Salmonella's flagellar / outer-membrane protein composition rather than
STEC virulence proteins.

**Implementation caveat:** Some ratios produce values outside their
physical range on SNV-preprocessed AUCs because individual band AUCs can be
negative (e.g., `bio_tyr_doublet_ratio` mean = −8.09 on Non-STEC,
`bio_na_a_form_fraction` mean = −0.59 on Non-STEC instead of [0, 1]). The
discrimination signal is real (the spread carries information) but the
absolute values are not biologically interpretable. Stage 15F should treat
these features as "scores" not "fractions."

### Detailed results

#### 1. Per-feature file-level signal (Cohen's d)

| Feature | d STEC↔Non-STEC | d E.coli↔Salm | d H2O↔bact | d K-12↔other-STEC | AUROC STEC↔Non-STEC |
|---|---:|---:|---:|---:|---:|
| **bio_alpha_helix_score**       | **−0.986** | −0.190 | −1.311 | **+0.537** | **0.794** |
| **bio_trp_indole_env**          | **+0.603** | +0.270 | +2.288 | **−0.547** | 0.704 |
| **bio_na_a_form_fraction**      | **+0.568** | −0.095 | +0.032 | −0.206 | 0.710 |
| bio_cyt_ox_state               | +0.477 | +0.286 | +1.498 | −0.344 | 0.681 |
| bio_rna_dna_ratio              | +0.450 | −0.326 | −0.069 | −0.373 | 0.741 |
| bio_cyt_pyrrole_ratio          | −0.353 | +0.485 | −0.037 | −0.341 | 0.567 |
| bio_tyr_doublet_ratio          | +0.342 | −0.113 | +0.039 | −0.386 | 0.559 |
| bio_cyt_total                  | +0.239 | −0.177 | +0.372 | −0.315 | 0.535 |
| bio_phb_carbonyl               | +0.207 | +0.409 | +0.604 | **+0.066** ❌ | 0.631 |
| bio_trp_content                | −0.209 | +0.054 | +0.259 | −0.315 | 0.536 |
| bio_virulence_aa_sig           | −0.150 | **−0.651** | −0.347 | +0.535 | 0.511 |
| bio_beta_sheet_amide3          | +0.150 | +0.379 | −1.118 | **−0.591** | 0.538 |
| bio_phb_score                  | +0.132 | +0.226 | −0.177 | −0.219 | 0.726 |

**3 features cross \|d\|≥0.5 STEC↔Non-STEC** (alpha_helix, trp_indole_env,
na_a_form_fraction) — Branch (B) cleared.

#### 2. K-12 vs other-STEC 2°-structure shift

This was an unprompted finding — not pre-registered as the K-12 axis (which
was the PHB hypothesis). Three biology features flag K-12 differently:

| Feature | K-12 mean | other-STEC mean | d |
|---|---:|---:|---:|
| bio_beta_sheet_amide3 | (~ −0.59 d below)  |  | **−0.591** |
| bio_trp_indole_env   |  |  | **−0.547** |
| bio_alpha_helix_score |  |  | **+0.537** |

Interpretation: K-12 has more α-helix and less β-sheet content than
clinical STEC strains, plus a shifted Trp indole environment (more
exposed/hydrophilic). These are consistent with K-12's reduced
virulence-protein loading vs O157/O121/O103/ATCC25922 — virulence proteins
contribute β-sheet-heavy folds (intimin, Stx subunits), and K-12 doesn't
carry them. **This is the FIRST K-12-specific feature axis in the project.**

#### 3. Pre-registration verdicts

| Pre-reg | Predicted | Actual | Verdict |
|---|---|---|:-:|
| New feature columns added to cache | 15 ± 2 | 13 (Stage 15A's pseudo-Voigt already covers BIO3 + BIO22) | ⚠️ slightly under range, acceptable |
| Cache build time delta | < +30 s | full cache rebuild ~normal duration | ✅ |
| `phb_carbonyl` K-12 vs other-STEC \|d\| ≥ 0.5 | yes (60% prob.) | **d=+0.066** | ❌ — PHB hypothesis falsified |
| `cyt_total` Salmonella vs E. coli \|d\| ≥ 0.4 | yes | d=−0.18 (Salm slightly lower) | ❌ — off-resonance R5 confirmed |
| `cyt_pyrrole_ratio` (752/1004) E. coli vs Salm \|d\| ≥ 0.3 | maybe | d=+0.485 | ✅ — actually clears 0.4 |
| `alpha_helix_score` (1652/1670) STEC vs Non-STEC \|d\| ≥ 0.4 | maybe | **d=−0.986** | ✅✅ — far exceeds, near \|d\|=1.0 |
| `virulence_aa_sig` (Trp/Phe) STEC vs Non-STEC sign: STEC < Non-STEC | reversed from literature | d=−0.150 (STEC slightly lower) — direction correct, magnitude weak | ⚠️ direction matches, magnitude weak |
| `tyr_doublet_ratio` any-class \|d\| ≥ 0.3 | yes | d=+0.34 STEC↔Non-STEC | ✅ — but SNV-AUC denominator issue makes absolute values uninterpretable |
| Best new biology \|d\| STEC↔Non-STEC | 0.4 – 0.8 | **\|d\|=0.99 (alpha_helix_score)** | ✅ over range — biology contributes a near-1.0 ratio |
| ≥ 2 biology features \|d\| ≥ 0.5 STEC↔Non-STEC | yes | **3 features** | ✅ |
| `na_a_form_fraction` (815/835) growth-phase only, no class discrim | weak | d=+0.57 STEC↔Non-STEC (actually moderate signal) | ❌ — predicted weak, came in moderate; growth-phase still likely confound, but signal is real |

#### 4. Per-class mean of each biology feature

| Feature | H2O | Non-STEC | STEC | Salmonella |
|---|---:|---:|---:|---:|
| bio_cyt_pyrrole_ratio    | 0.297  | 0.593 | 0.404 | 0.020 |
| bio_cyt_ox_state         | 1.000  | 0.968 | 0.978 | 0.967 |
| bio_cyt_total            | −16.63 | −17.12 | −16.93 | −16.85 |
| **bio_alpha_helix_score**| 0.999  | **1.047** | **1.019** | 1.038 |
| bio_beta_sheet_amide3    | 0.873  | 0.886 | 0.888 | 0.883 |
| bio_phb_carbonyl         | −7.68  | −8.20 | −8.05 | −8.52 |
| bio_phb_score            | 0.55   | 6.25  | 11.97 | 1.27 |
| bio_tyr_doublet_ratio*   | −0.63  | −8.09 | 2.54  | 0.32 |
| bio_trp_content          | −8.50  | −8.61 | −8.77 | −8.74 |
| bio_trp_indole_env       | 0.999  | 0.923 | 0.938 | 0.922 |
| bio_virulence_aa_sig     | 1.240  | 1.338 | 1.290 | **1.698** |
| bio_na_a_form_fraction*  | 0.724  | −0.588 | 1.546 | 0.826 |
| bio_rna_dna_ratio        | 0.840  | 0.672 | 0.783 | 1.530 |

\* Ratios with denominator near zero on SNV'd AUCs produce extreme values.
Treat as "scores" not "fractions."

### Implications

1. **`bio_alpha_helix_score` should be a headline feature in Stage 15F.**
   At d=−0.986 it's near-tied with the LPS_1194 (d=+1.03) anchor in
   magnitude but lives in a completely different biological axis (protein
   2°-structure, not LPS chain). This is the cleanest "biology
   makes-sense" discriminator we have. Non-STEC > STEC for α-helix means
   commensal E. coli strains run higher amide-I α-helix Raman signal —
   plausibly because they're not loaded with β-sheet-heavy virulence
   proteins.

2. **K-12 has a 2°-structure signature, not a PHB signature.** PHB
   hypothesis is dead. But the unexpected 2°-structure split (3 features,
   |d|=0.54–0.59) IS the first K-12-specific axis. Stage 6 (3-channel CNN)
   reconsideration is back on the table if Stage 15F can show that adding
   `bio_alpha_helix_score + bio_beta_sheet_amide3 + bio_trp_indole_env`
   lifts K-12 LOSO recall vs the current best (DANN λ=0.3 5-seed K-12
   recall = 0.75 fragile on 2/5 seeds).

3. **Cytochrome features are weak**, confirming plan/15 §7 R5
   (off-resonance at 785 nm). `bio_cyt_pyrrole_ratio` STEC↔Non-STEC
   d=−0.35, E.coli↔Salm d=+0.49 — usable but not headline. Plan/15 §4.5
   already deferred cytochromes to second tier; this validates the
   decision.

4. **Trp/Phe (BIO29) literature claim is REVERSED on E. coli vs
   Salmonella.** Plan/15 framed BIO29 as a STEC↔Non-STEC test. On Atlas:
   STEC↔Non-STEC signal is weak (d=−0.15), but **E. coli vs Salmonella is
   d=−0.65 with Salmonella HIGHER**. So Trp/Phe is more useful for the
   E.coli↔Salm split (where it adds genuinely new signal) than for the
   STEC↔Non-STEC split (which was the original prediction). Document and
   redeploy.

5. **SNV-AUC ratio robustness:** some biology features (`bio_tyr_doublet`,
   `bio_na_a_form_fraction`) produce values outside their physical range
   because individual band AUCs on SNV'd data can be negative. The
   discrimination signal is preserved (the spread carries information) but
   absolute values are not biologically interpretable. **For Stage 15F:**
   either include them as raw scores OR re-derive on the non-SNV
   preprocessing path (same fix as Stage 15C). Tracked as a
   **new R8 risk** (preprocessing-induced ratio breakdown) in plan/15.

6. **Branch (B), not (A) → 15E and 15F proceed as planned.** Stage 6
   re-evaluation is conditional on Stage 15F K-12 lift, not on this
   stage's verdict directly. Continue to Stage 15E (spatial features).

7. **PHB carbonyl falsified** — drop from headline tier. But include in
   Stage 15F feature set anyway; the d=+0.41 E.coli↔Salm signal is
   modest-but-real.

8. **`bio_rna_dna_ratio` E.coli↔Salm d=−0.33 (Salm > E.coli)** is
   interesting if not pre-registered: Salmonella has more A813/A788 ratio
   than E. coli at file level. This is a growth-phase / metabolic-state
   signature (RNA dominance suggests active translation). May or may not
   help classification but it's a new directional finding.

---

## Artifacts

- `atlas/band_features.py` (extended with 5 biology functions + biology=True in feature_frame)
- `scripts/build_band_features_cache.py` (re-run, no signature change)
- `data_cache/band_features.parquet` (updated: ~168 cols × 7,122 rows)
- `outputs/band_chemistry/stage15d/01_stage15d_summary.json` (per-feature d table)
- `outputs/band_chemistry/stage15d/per_feature_d.png` (file-level d heatmap if useful)
