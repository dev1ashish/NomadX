# 02 — Locked design decisions

> **Mutability:** stable. To change a decision: append a new entry to `10_decision_log.md` with the reason, then edit this file.
> **Cross-ref:** every entry below corresponds to one or more dated entries in `10_decision_log.md`.

## High-level

| Decision | Choice | Rationale |
|---|---|---|
| Deliverable format | Repo + driver notebook | Modular reuse, looks engineered. |
| Modeling depth | Classical baseline + 1D-CNN + small 1D-Transformer | Defensible floor + an upper bound + a credibility-move third arm. |
| Subclass strategy | Stratified file-level splits + per-subclass metrics + Leave-One-Strain-Out (LOSO) stress test | Directly addresses the assessment's subclass ask. |
| Primary headline metric | macro-F1 | Robust to the H₂O class imbalance. Balanced accuracy also reported. |

## Data pipeline

| Decision | Choice | Rationale |
|---|---|---|
| Canonical wavenumber axis | `linspace(76, 3499, 2048)` | Files drift in `wn[0]` by ~0.05 cm⁻¹ across calibration batches; interpolating to a fixed axis eliminates the drift. |
| Per-file pixel cap | 200 px / file, random subsample at parse time | R364 (324) and R370 (720) would otherwise dominate their subclasses. 200 preserves stats, prevents over-representation. |
| Spectral crop | Fingerprint 400–1800 cm⁻¹ + C–H stretch 2800–3050 cm⁻¹ | C–H stretches carry strong bacterial discrimination signal (confirmed empirically — see `07_findings.md`). |
| R371 partial scan | Include with `is_complete_scan=False` flag | Don't throw away usable data; flag it for downstream awareness. |
| `.txt` file at Heidelburg | Include | Identical tab-delim format; parser globs both extensions. |
| Heidelburg subclass label | Keep as-is (do not normalize to "Heidelberg") | Match input directory naming; note typo in final README. |
| Partial-scan detection | Compare `n_pixels` to `grid_nx * grid_ny` (NOT header `#NUMX*#NUMY`) | Early-batch headers are unreliable; coord-derived grid is ground truth. |

## Preprocessing

| Decision | Choice | Rationale |
|---|---|---|
| Pipeline order | cosmic-ray → arPLS baseline → Sav-Gol smoothing → crop → SNV → (optional) 2nd deriv | Standard chemometrics ordering. Baseline before smoothing. |
| arPLS parameters | `lam=1e5, max_iter=50, diff_order=2`, no `p` | Originally specified `p=0.01` — that param belongs to `asls`, not `arpls`. arpls auto-computes asymmetry via IRPLS. |
| 2nd derivative | Optional concat for classical features only; CNN consumes SNV-only | Sharpens peaks for linear models; CNN can learn equivalents from convolutions. |

## Modeling

| Decision | Choice | Rationale |
|---|---|---|
| Classical model set | LogReg, LinearSVM, RBF-SVM, RF, XGBoost, PLS-DA | Covers linear / kernel / tree / chemometrics-standard. |
| PLS-DA placement | Available but not in headline results | Chemometrics reviewers expect it; ship `configs/extra_pls.yaml`. |
| Class weighting | `class_weight="balanced"` on primary class | No SMOTE — spectra interpolation isn't physically meaningful. |
| CNN variant default | Small (~110K params) | Dataset smaller than ml-engineer's original 22K estimate (~10K actual). |
| CNN device | Auto-detect (MPS if available else CPU), pinned in resolved config | Reproducibility trade-off documented per run. |
| DANN domain adversary | Off by default | Enable only if memorization probe shows file-ID leakage. |
| Small 1D-Transformer | Third model arm after CNN | Honest benchmarking; not expected to beat CNN; credibility move. |

## Evaluation

| Decision | Choice | Rationale |
|---|---|---|
| Split protocol A | StratifiedGroupKFold(5), groups=file_id, stratify=primary_class | 87 files → ~17–18 test files/fold including H₂O. |
| Split protocol B | Leave-One-Strain-Out (9 iterations, one per bacterial subclass) | Tests "did model learn STEC-ness or strain identity?" |
| Inner HPO validation | StratifiedGroupKFold(4) within each outer-train, fold 0 fixed (not nested) | Tractable budget for HPO. |
| Per-spectrum → per-file aggregation | Soft vote (mean of probabilities) | Calibration-aware. |
| Low-support file flag | `n_qc_pixels < 50` → `low_support` | Don't trust file-level predictions from too-few pixels. |

## Plotting / diagnostics

| Decision | Choice | Rationale |
|---|---|---|
| ANOVA discriminative plot | Plot `log10(F)` instead of `-log10(p)` | Top bins underflow `p` to 0 → `inf` gaps. F-statistic ranking is identical. |
