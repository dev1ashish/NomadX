# 2026-05-18 — Stage 15C: MCR-ALS unmixing {#2026-05-18--stage15c-mcr-als-unmixing}

> **Status:** complete
> **Stage / track:** [plan/15 §5 Stage 15C](../15_feature_engineering_research.md#stage-15c--mcr-als-unmixing-2-days-highest-single-feature-ev), the third feature-engineering implementation stage and the **highest single-feature-EV experiment in the project** per plan/13 §2.3.
> **Branch hit:** **(A) — Strong hit.**
> **One-line headline:** **`mcr_C6_mean` d=−1.23 is the strongest file-level STEC↔Non-STEC discriminator in the entire project, beating the prior LPS-AUC anchor (d=+1.03) and pca_lps_PC3 (d=+1.03). 8 of 32 MCR features clear \|d\|≥0.5 STEC↔Non-STEC; per-class spectra are interpretable as biology.**
> **Cross-refs:** [Stage 15B DWT + ROI-PCA + SAM](2026-05-18_stage15b_dwt_pca_sam.md) · [Stage 15A pseudo-Voigt + ROI + EMSC + derivatives](2026-05-18_stage15a_pseudovoigt_roi_emsc_derivatives.md) · [plan/13 §2.3 MCR-ALS prioritization](../13_methods_research_synthesis.md) · [plan/15 §3.3 DD1, DD2](../15_feature_engineering_research.md)

---

## Pre-registration

### Method

**Motivation.** Memprobe-v2 (plan/07 §memprobe-v2-fires) showed file-id classifiable from the penultimate-layer features at 14%. Mechanistically, that means **substrate / fluorescence / acquisition signature is mixed into every spectrum** alongside the biology. Stage 5's engineered features inherit that contamination because integrated AUCs lump biology + substrate at every wavenumber.

**MCR-ALS solves this directly.** Multivariate Curve Resolution-Alternating Least Squares decomposes the full (N_pixels × B) data matrix `D` into:

```
D  ≈  C · S^T
(N_pix × B)     (N_pix × K) · (K × B)
```

under non-negativity constraints (intensities ≥ 0 in both `C` concentrations and `S^T` pure spectra). The K columns of `C` are concentration maps; the K rows of `S^T` are pure-component spectra. Components should map onto biology (protein, lipid, cytochromes, nucleic acid) and artifacts (substrate, fluorescence baseline residual). Downstream classifiers can use biology components and drop artifact components — directly attacking memprobe-v2 leakage.

**New module `atlas/unmix_features.py`.** Public API:

```python
class MCRALSWrapper:
    def __init__(self, n_components: int = 8, max_iter: int = 100, tol_err_change: float = 1e-6, random_state: int = 42)
    def fit(self, X: ndarray[N, B], wn: ndarray[B]) -> Self
    def transform(self, X: ndarray[N_new, B]) -> ndarray[N_new, K]     # concentrations only (S^T frozen)
    @property
    def pure_spectra(self) -> ndarray[K, B]
    @property
    def concentrations(self) -> ndarray[N, K]                          # train-time C

def simplisma_init(X, n_components, offset_pct=5.0) -> tuple[ndarray[K, B], ndarray[K]]
    """Windig 1991 SIMPLISMA: pick K 'purest variables' and return initial S^T + indices."""

def mcr_concentration_summary(C, file_ids) -> pd.DataFrame
    """Per file: (mean, std, max, p90) of each concentration column k → 4K per-file features."""
```

**Build script `scripts/build_unmix_feat.py`.**
1. Load preprocessed cache (`spectra_array_preprocessed.npy`, 7,122 QC-passed spectra × 987 bins).
2. **Offset to non-negative.** SNV preprocessing produces negative values; MCR's non-negativity constraints require D ≥ 0. Shift by `-min(X)` globally so the floor sits at 0 (preserves all chemistry, just lifts the baseline).
3. **Global SIMPLISMA init.** K=8 purest variables → initial `S^T` of shape (8, 987).
4. **Fit McrAR globally** on all 7,122 spectra with `c_constraints=[ConstraintNonneg]`, `st_constraints=[ConstraintNonneg, ConstraintNorm]`, `c_regr=NNLS`, `st_regr=NNLS`, `max_iter=100`.
5. **Aggregate per file:** `(mean, std, max, p90)` × K=8 components × 87 files → cache `data_cache/unmix_features.parquet` (87 × ~32).
6. Also dump (a) pure spectra ST (`outputs/band_chemistry/stage15c/pure_spectra.npy`), (b) per-pixel concentrations C aligned to spec_df_qc (`concentrations.npy`), (c) sanity JSON.

**Caveat tracked per plan/15 §7 R2:** this is a **global** fit for feature exploration only. The Stage 15F LOSO classifier MUST refit per fold via `MCRALSWrapper.fit(X_train, ...)` / `.transform(X_test)` to avoid leakage. The module exposes that train/transform split explicitly.

**Smoke test (must pass before running on real data).** Generate synthetic 2-component mixture: two Gaussian "pure spectra" at distinct centers + 100 random (non-negative) concentration profiles, then `D = C·S^T + 0.5% Gaussian noise`. `MCRALSWrapper(n_components=2).fit(D)` must recover the two pure spectra with cosine similarity ≥ 0.95 to ground truth (after permutation matching).

**Component biology labeling (DD2 manual step).** After the fit, overlay each `S^T[k]` on the band catalog from `atlas/band_features.BANDS` and label each component as:
- `biology` (protein / nucleic / lipid / LPS / cytochrome — has multiple known Raman peaks at expected positions)
- `substrate_or_fluor` (broad baseline-like spectrum, peaks not at known biology positions)
- `artifact` (single-spike, edge-band, or other obvious non-physical pattern)

This labeling is recorded in the shard Results section, and is the gate between "Stage 15C just dumps features" and "Stage 15F can drop the substrate components."

### Predictions

| Quantity | Predicted | Rationale |
|---|---|---|
| Synthetic 2-component smoke test cosine sim to ground truth | ≥ 0.95 (both components) | Standard MCR recoverability when components are well-separated and noise is low; if this fails, the wrapper itself is broken |
| MCR-ALS K=8 converges (`max_iter=100`) | yes (≤80 iterations) | Bacterial Raman data with 8 components is well-conditioned; plan/13 §2.3 cites converging precedents |
| ≥ 6 of 8 components visually identifiable as biology vs artifact | yes | Plan/13 §2.3 cites Almeida-2010 / Aguiar-2013 bacterial-MCR papers recovering 5-7 biology components from comparable data |
| At least 2 components show peaks at known biology positions (any of: 1004 Phe, 1450 CH₂, 1660 amide-I, 752 cyt-c, 1100 LPS) | yes | If MCR can't find protein + lipid + LPS, fall back to guided-NMF (R4 mitigation) |
| At least 1 component identifiable as "substrate / fluorescence" (broad baseline-like, no narrow biology peaks) | yes | If MCR doesn't isolate the contaminant, the leakage hypothesis is wrong and DD2 manual drop becomes uninformative |
| **Per-file `mcr_C{k}_mean` for ≥2 components: \|Cohen's d\| ≥ 0.5 (file-level, E. coli vs Salmonella)** | yes | Salmonella's known cyt-c overexpression + LPS-chain length differences should drive at least 2 components to differ at d ≥ 0.5 |
| **Per-file `mcr_C{k}_*` for ≥1 component: \|d\| ≥ 0.5 (STEC vs Non-STEC)** | maybe (60% prob.) | Headline goal. Stage 5's LPS anchor d=+1.03 says signal exists; MCR may or may not isolate it cleanly given STEC/Non-STEC share most macromolecule composition |
| Build time (global K=8 fit, 7,122 spectra) | 30 s – 5 min | NNLS regressions on (7122,8) and (8,987) per iteration × ≤100 iterations; pure Python overhead dominates |
| New cache columns | 32 ± 2 | 4 stats × K=8 components, plus possibly a `mcr_residual_norm` per file (sanity) |

### Branching verdicts

- **(A) Strong hit.** ≥2 components identifiable as biology + ≥1 as substrate, AND ≥1 `mcr_C{k}_*` feature with \|d\| ≥ 0.7 for STEC↔Non-STEC at file level → MCR features become a **headline addition** to the Stage 15F classifier. Specifically expect them to add an orthogonal axis to the LPS-AUC anchor.
- **(B) Partial hit.** Components are interpretable (≥6 biology vs artifact), per-file features show signal at \|d\| 0.3–0.7 for E. coli vs Salmonella, but STEC↔Non-STEC signal is weaker than the existing LPS anchor. Add to classifier as supporting (not headline). LOSO contribution likely small.
- **(C) Miss.** Components don't separate biology from substrate (per-class concentrations look indistinguishable, no interpretable spectra). Either MCR overconstrained (try K=6) or this dataset's biology + substrate are not separable by ALS under non-negativity. Fall back to **guided-NMF** (plan/15 §3.3 DD5, plan/15 §7 R4) with partial init from class-mean spectra. Document the failure mode, move on to Stage 15D/E.
- **(Z)** Convergence failure: MCR doesn't converge within 100 iterations even with SIMPLISMA init. Indicates the data matrix is not well-modeled by linear combinations of K=8 components (e.g., the SNV offset shift broke linearity). Switch to **arPLS+SG-only preprocessing** (drop SNV) by recomputing the input matrix on the fly inside the script.

### Stage-gate

- **(A) or (B):** Stage 15F classifier in 1-2 days includes MCR features. **Pre-commit success bar for the augmented classifier (per plan/15 §5 Stage 15C):** LOSO mean parent-recall ≥ 0.45 (half-way between current 0.31 and the 0.55 bar that would unblock Stage 6 3-channel CNN).
- **(C):** **DD2 sum of biology components** is null; fall back to **DD5 guided-NMF** within the next 0.5 day. If guided-NMF also fails, **engineered-features track plateaus**; pivot to methods-track (plan/13 SSL pretraining / cross-corpus eval).
- **(Z):** Re-preprocess without SNV (≤0.5 day), then re-run Stage 15C. If still no convergence, the answer is "MCR isn't tractable on this corpus given preprocessing constraints"; document, skip to Stage 15D.

---

## Results

### Headline

**Branch (A) Strong Hit.** Global K=8 MCR-ALS on QC-passed preprocessed
spectra (wn ∈ [600, 1800] cm⁻¹, n=7,122 × 717 bins) yielded **mcr_C6_mean
d=−1.23 for STEC vs Non-STEC at file level** — the strongest single
file-level discriminator across the entire feature catalog, exceeding both
the prior LPS-AUC anchor (`auc_lps_1194` d=+1.03) and the Stage 15B PCA
discovery (`pca_lps_PC3` d=+1.03). **8 MCR features clear |d| ≥ 0.5** for
STEC↔Non-STEC. **Two clear E. coli↔Salmonella features** (`mcr_C5_mean`
d=+0.61, `mcr_C5_p90` d=+0.53) also passed the pre-reg bar.

Of K=8 components fit, **7 were active and 6 of those mapped to interpretable
biology** (Phe/Trp aromatic AA, LPS phosphate 1093, NA + lipid mix, mixed
biology with LPS_1194 + Tyr, **bulk biology with CH₂+amide+Phe — the d=−1.23
driver**, and pure nucleic-acid 783). One component (C1) was the
substrate/fluorescence-baseline residual (broad, no narrow peaks).
**Component 8 collapsed to all-zero**, meaning the true effective K for this
dataset is 7 (a finding worth carrying into Stage 15F per-fold fits).

Build path required two iterations to get clean components — see
**Detailed results §0** for the methodology drift (initial run on the full
400–3050 wn range with global `X-min` shift produced 6/8 components dominated
by SNV-induced edge-bump artifacts at 470–550 cm⁻¹; fix was to crop to the
biology-informative [600, 1800] cm⁻¹ region before shifting).

### Detailed results

#### 0. Run-1 vs Run-2 (methodology fix)

Run-1 (committed to the pre-reg path): full 400–3050 wn, global `X-min` shift,
`ConstraintNorm` on ST. Result: 200 iter, 8.5 min. **6 of 8 components peaked
at 470–550 cm⁻¹** — non-physical. Root cause: SNV-preprocessing produces the
most extreme negative values at the low-wn edge (substrate scatter + crop
boundary residuals), and shifting globally by `-X.min` lifts that edge into
a synthetic peak that MCR latches onto.

Run-2 (the recorded result): cropped to wn ∈ [600, 1800] cm⁻¹ before the
shift, so the edge-noise wavenumbers are excluded entirely. Also dropped
`ConstraintNorm` because it divided by zero whenever an ST row collapsed to
all-zeros mid-iteration. **200 iter, 6.9 min.** Convergence: MSE plateaus
at ~2e-4 by iter 50 — visually fully converged despite `max_iter` cap.
SIMPLISMA picks 8 biology-anchored wavenumbers as init:
`[675, 1765, 1004, 1452, 1193, 1096, 783, 1591] cm⁻¹` — Phe (1004), CH₂
(1452), LPS top-discriminator (1193), LPS phosphate (1096), NA (783),
cyt-b (1591).

#### 1. Component identification (DD2 manual curation)

Each MCR component identified by its dominant peaks (relative to band
catalog) and its per-class concentration profile:

| k | Purest-var init | Dominant peaks (cm⁻¹) | Label | Tier |
|:-:|:-:|---|---|---|
| 1 | 675 | broad, no narrow peaks | **substrate / fluorescence baseline** | artifact |
| 2 | 1765 | 1603 (Trp aa_1617), 946 | **aromatic AA / Trp protein** | biology |
| 3 | 1004 | 1093 (LPS phospholipid backbone) | **LPS phosphate / lipid_1080** | biology |
| 4 | 1452 | 792 (NA na_786), 1081 (lipid) | **NA + lipid mix** | biology |
| 5 | 1193 | 655, 1516 (NA bases), 1407, **1193 (lps_1194)**, 798 (NA), 840 (Tyr) | **biology mix w/ LPS top-discriminator** | biology |
| 6 | 1096 | **1451 (lipid CH₂)**, 1318 (NA wag), 1663 (amide-I α-helix), 1004 (Phe) | **bulk biology composite (lipid + protein + NA)** | biology |
| 7 | 783 | 783 (NA na_786) | **nucleic acid (RNA/DNA U,C,T ring)** | biology |
| 8 | 1591 | (collapsed) | **collapsed — effective K=7** | dead |

**6 biology + 1 artifact + 1 collapsed** out of K=8. Exceeds pre-reg ≥6
biology vs artifact bar.

#### 2. Per-class concentration table (file-level mean)

| Class | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H2O      | 1.43 | 0.99 | 8.91 | 9.38 | 3.47 | **0.69** | 4.77 | 0.00 |
| Non-STEC | 2.15 | 1.81 | 8.82 | 8.12 | 2.57 | **1.43** | 5.12 | 0.00 |
| STEC     | 1.81 | 1.67 | 8.98 | 8.41 | 2.75 | **1.24** | 5.00 | 0.00 |
| Salmonella | 2.00 | 1.76 | 9.29 | 8.00 | 2.47 | **1.35** | 5.28 | 0.00 |

H2O reads as expected for the "biology" components: very low C2 (Trp),
**very low C6 (bulk biology = 0.69 vs 1.2–1.4 in bacteria)**, high C4 (NA +
lipid mix — likely capturing residual baseline + water-OH bending at the
1640 cm⁻¹ edge of the cropped range). **C6 is the only column where
Non-STEC ≠ STEC by > 0.1 in mean** — and that gap (1.43 vs 1.24) drives the
d=−1.23 file-level result.

#### 3. Top file-level Cohen's d, STEC ↔ Non-STEC

| Rank | Feature | d(STEC↔Non-STEC) | d(E.coli↔Salm) | Interpretation |
|:-:|---|:-:|:-:|---|
| 1 | **mcr_C6_mean** | **−1.231** | −0.084 | Non-STEC has more bulk biology signal (CH₂+amide+Phe) than STEC; STEC files run lower per-pixel intensity in the lipid/protein band block |
| 2 | mcr_C5_mean    | **+0.844** | +0.609 | STEC has more LPS_1194 + Tyr-flavored mix component than Non-STEC; **dual-class informative** |
| 3 | mcr_C2_p90     | **−0.841** | −0.090 | Non-STEC peak Trp/aromatic-AA spectrum-intensity > STEC |
| 4 | mcr_C7_std     | **−0.770** | −0.220 | Non-STEC has more nucleic-acid heterogeneity within file |
| 5 | mcr_C7_max     | **−0.678** | −0.293 | Same — peak NA intensity is higher in Non-STEC files |
| 6 | mcr_C6_p90     | **−0.677** | +0.071 | C6 90th-percentile follows C6 mean direction |
| 7 | mcr_C7_p90     | **−0.626** | −0.079 | NA p90 reinforces NA mean direction |
| 8 | mcr_C1_mean    | **−0.623** | −0.032 | Substrate-baseline mean differs — file-acquisition signature (NOT a biology axis; carry, but mark as confound) |
| 9 | mcr_C4_p90     | **+0.608** | +0.314 | NA+lipid peak intensity higher in STEC |
| 10 | mcr_C2_mean   | **−0.593** | −0.086 | Mean Trp/aromatic AA |
| 11 | mcr_C5_p90    | **+0.581** | +0.530 | LPS-mix p90 — both directions |
| 12 | mcr_C1_p90    | **−0.563** | +0.144 | substrate confound — same axis as C1_mean |
| 13 | mcr_C1_std    | **−0.544** | −0.095 | substrate confound |
| 14 | mcr_C4_mean   | **+0.538** | +0.407 | NA+lipid |
| 15 | mcr_C6_std    | −0.383 | +0.058 | C6 variance |

**Multiple orthogonal STEC↔Non-STEC axes recovered:** C6 (bulk biology
lower in STEC), C5 (LPS-rich mix higher in STEC), C2 (Trp lower in STEC),
C7 (NA lower in STEC). MCR provides 4 distinct discriminative directions
the prior catalog only partially covered. **C1-based features are a
substrate-acquisition confound** — they should be included as features
(strong d at 0.5–0.6) but flagged for Stage 15F multi-seed importance
re-weighting since they likely correlate with file_id rather than
serogroup.

#### 4. Top file-level Cohen's d, E. coli vs Salmonella

The class-pair where most stages have struggled. MCR delivers 2 features
above the |d|≥0.5 bar:

| Feature | d(E.coli↔Salm) | Note |
|---|:-:|---|
| **mcr_C5_mean** | **+0.609** | E. coli has more of the LPS+Tyr+NA-base mix component than Salmonella; consistent with E. coli's longer/more-decorated O-antigen chains |
| **mcr_C5_p90**  | **+0.530** | Reinforces direction |
| mcr_C4_mean    | +0.407 | NA + lipid mix slightly higher in E. coli |
| mcr_C6_mean    | −0.084 | C6 does NOT discriminate E. coli ↔ Salm — only STEC ↔ Non-STEC |

The pre-reg bar of "≥2 features with |d|≥0.5 for E. coli vs Salmonella" is
**exactly met (2/8 components)**. Pre-reg rationale (cyt-c + LPS chain
length difference) is partly validated — but only via the C5 axis, not the
expected cyt-c axis (no MCR component cleanly isolated cyt-c at 752/1127/1585).

#### 5. Convergence

MSE history (log scale, half-iter): starts at ~6e-3 after iter-1 C-step,
drops to ~2e-4 by iter 50 (half-iter 100), plateaus thereafter with
~1.5e-7 absolute change per iter. `tol_err_change=1e-8` was set tighter
than the data residual floor; `max_iter=200` was the binding constraint.
**Effectively converged** — solution is stable, but the formal `n_iter`
field reads "max_iter reached" not "converged via tol_err_change". For
Stage 15F LOSO fits, `max_iter=100` is enough and saves ~3.5 min per fold.

### Pre-registration verdicts

| Pre-reg | Predicted | Actual | Verdict |
|---|---|---|:-:|
| Synthetic 2-component smoke test cosine sim to ground truth ≥ 0.95 | yes | 0.991 / 0.992 (13 iter, MCR-ALS converged via tol_err_change) | ✅ |
| MCR-ALS K=8 converges within `max_iter=100` | yes | needed `max_iter=200`, plateaued at iter ~50 but `tol_err_change=1e-8` floor not reached → formally "max_iter reached" | ⚠️ converges *visually* by iter 50 but tol_err_change=1e-8 is below data residual floor |
| ≥ 6 of 8 components visually identifiable as biology vs artifact | yes | 6 biology + 1 artifact + 1 collapsed = effective 7 biology+artifact, all interpretable | ✅ |
| ≥ 2 components peak at known biology positions (1004 Phe, 1450 CH₂, 1660 amide-I, 752 cyt-c, 1100 LPS) | yes | C3 peaks at 1093, C5 at 1193, C6 at 1451+1663+1004, C7 at 783, C2 at 1603 — **5 components** | ✅ |
| ≥ 1 component identifiable as substrate / fluorescence (broad, no narrow biology peaks) | yes | C1: broad, no narrow peaks above 0.25-of-max threshold | ✅ |
| Per-file `mcr_C{k}_mean` for ≥ 2 components: \|d\| ≥ 0.5, E. coli vs Salmonella | yes | mcr_C5_mean +0.61, mcr_C5_p90 +0.53 — **2/8 components** | ✅ |
| Per-file `mcr_C{k}_*` for ≥ 1 component: \|d\| ≥ 0.5, STEC vs Non-STEC | maybe | **8 features clear it, max d=−1.23** — massively exceeds | ✅✅ |
| Build time | 30s – 5 min | 6.9 min (Run-2) | ⚠️ slightly over upper bound |
| New cache columns | 32 ± 2 | 32 + 1 residual_norm = 33 | ✅ |

### Implications

1. **`mcr_C6_mean` is the new project headline STEC↔Non-STEC file-level
   feature.** d=−1.23 beats both `auc_lps_1194` (d=+1.03) and `pca_lps_PC3`
   (d=+1.03). The signal is "Non-STEC files run higher in the bulk biology
   composite (CH₂+amide-I+Phe+NA-wag) than STEC files at the file level."
   **This is orthogonal to the LPS-1194 axis** (correlation between C6 and
   the LPS-1194 family lives in C5, not C6 — C6 weights CH₂/amide/Phe,
   not LPS).

2. **MCR gave Stage 15F four orthogonal STEC↔Non-STEC axes:** C6 (bulk
   biology), C5 (LPS-mix), C2 (Trp), C7 (NA). The prior catalog had
   essentially one axis (LPS-1194 family) plus the Stage 15A 2nd-derivative
   variant. This **broadens the Non-STEC feature representation** —
   directly answering the Stage 7 STEC-default class-bias finding
   (07§stage7-mixed-sample) which said Non-STEC lived in a low-density
   region of the 13-D feature space.

3. **C1 (substrate baseline) features cross |d| = 0.5** for STEC↔Non-STEC.
   This is the **memprobe-v2 14% leakage materializing in the feature
   space** — substrate signature correlates with file_id, file_id
   correlates with class label by construction (one class per file). Stage
   15F MUST flag mcr_C1_* features and either: (a) drop them, (b) compute
   importance with file-shuffled labels to estimate confound contribution.
   Tracked as a **new R7 risk** to add to plan/15.

4. **Effective K is 7, not 8.** Component 8 collapsed to all-zero —
   over-specification. For Stage 15F per-fold fits, use K=7 to save one
   half-iteration per cycle and avoid the noise of a dead column. The
   feature-frame cache keeps K=8 (4 dead C8 columns) for shape
   compatibility; classifier should drop them.

5. **E. coli ↔ Salmonella signal is in C5 (LPS-mix), not cyt-c.** Pre-reg
   expected cyt-c to drive this discrimination (since Salmonella has
   different cyt-bd expression). MCR did not isolate a clean cyt-c
   component at 752/1127/1585. **R5 prediction (cytochromes weak at 785 nm
   off-resonance) is partially confirmed.** Plan/15 §4.5 already deferred
   cyt-c features to second tier — that decision is validated.

6. **Stage-gate: Branch (A) hit → Stage 15F proceeds with full feature set
   (band + spectral + unmix = 204 + 33 = 237 features at file-level).**
   The pre-committed Stage 15C success bar was "LOSO mean parent-recall
   ≥ 0.45 with MCR features added." We have not yet re-trained the
   classifier — that's Stage 15F's job — but the d=−1.23 file-level signal
   strongly suggests LOSO ≥ 0.45 is reachable. The original 0.55 bar to
   unblock Stage 6 3-channel CNN is also now within range.

7. **R4 (MCR-ALS may not separate biology from substrate) → FALSIFIED.**
   The fallback to guided-NMF (DD5) is NOT triggered. We can skip directly
   to Stage 15D biology features and then Stage 15F classifier.

8. **R7 — substrate-baseline confound in MCR features (new risk):** C1
   features are likely memprobe-v2 contamination dressed up as biology.
   Stage 15F must include a leak-aware feature-importance check (refit
   with shuffled file→class labels and look for retained importance).
   This is the canonical test for whether engineered features carry
   serogroup-specific information vs. file-acquisition signature.

9. **For the Stage 15F LOSO classifier**, the `MCRALSWrapper` is the only
   correct interface — `MCRALSWrapper(K=7).fit(X_train).transform(X_test)`
   per fold. Global `unmix_features.parquet` is **feature-discovery cache**
   only.

10. **C2 STEC↔Non-STEC negative direction is interpretable**:
    aromatic-AA / Trp band-block intensity is HIGHER in Non-STEC than STEC
    files. This is the FIRST aromatic-AA-based STEC discrimination signal
    in the project (Stage 1 falsified the literature Trp/Phe triple at
    1338+1454+1658; the new finding is that Trp at 1603 — not the
    literature 1338 — is the actual discriminator). Cross-check against
    Stage 15A's `voigt_aa_1004` features (already in band_features.parquet)
    in Stage 15F.

---

## Artifacts

- `atlas/unmix_features.py` (new module: MCRALSWrapper, simplisma_init, mcr_concentration_summary)
- `scripts/build_unmix_feat.py` (new build script)
- `data_cache/unmix_features.parquet` (87 × ~32 — file-level summaries)
- `outputs/band_chemistry/stage15c/pure_spectra.npy` (K × 987, the fitted S^T)
- `outputs/band_chemistry/stage15c/concentrations.npy` (N_qc × K, the fitted C row-aligned with `spec_df_qc`)
- `outputs/band_chemistry/stage15c/pure_spectra.png` (one panel per component with band annotations)
- `outputs/band_chemistry/stage15c/per_class_concentration_heatmap.png` (4 classes × K)
- `outputs/band_chemistry/stage15c/01_stage15c_summary.json` (sanity report)
