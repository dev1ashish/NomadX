# 06 — Risk register

> **Mutability:** stable. Add new risks at the bottom; promote to "realized" when one fires.

| # | Risk | Status | Mitigation |
|---|---|---|---|
| R1 | **Batch effects** via instrument calibration date — files calibrated same day may share artifacts even after group-aware splits. | **REALIZED** (mild, ratio 0.89) — see `07_findings.md` | Use `ac_calibration_date` as a secondary group sanity check. Run memorization probe. |
| R2 | **R364 / R370 mosaics** could swamp one subclass each. | mitigated | Per-file pixel cap = 200. |
| R3 | **Dataset smaller than agents initially estimated** (~10K vs 30K). | confirmed | Default to small CNN variant; lean on classical models; strong augmentation. |
| R4 | **MPS non-determinism** breaks bitwise reproducibility. | open | Train CNN on CPU; document in README. |
| R5 | **Pixel-level leakage** if we ever split spectra instead of files. | mitigated | All splits operate on file_id; enforce via type-checked split artifacts. |
| R6 | **arPLS divergence** on low-intensity water spectra. | open | SNIP fallback wired in (function present, not exercised yet). |
| R7 | **`#NUMX/#NUMY` headers unreliable.** | mitigated | Parser derives grid dims from `unique(x_um) × unique(y_um)`. |
| R8 | **arPLS boundary artifact** at left edge of crop range (400 cm⁻¹) creates a huge spike not present in chemistry. | **REALIZED** | Next preprocessing pass: crop start at 450 cm⁻¹. |
| R9 | **Linear separability is poor** even after best-effort preprocessing (PC1 still 69.5%, silhouette still −0.23). | **REALIZED** — likely structural, not a bug | Lean on nonlinear models (CNN, RBF-SVM, XGB). Don't expect classical linear models to win. |
