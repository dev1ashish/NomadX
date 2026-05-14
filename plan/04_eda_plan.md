# 04 — EDA plan (notebook block ordering)

> **Mutability:** stable. Add new blocks at the bottom; don't renumber existing ones.
> **Notebook:** `notebooks/atlas_driver.ipynb` (built by `scripts/build_eda_notebook.py`).

## Blocks 1–8 — on raw + SNV-only data (the "before" baseline)

Preserved deliberately to show what doesn't work *without* full preprocessing.
Plots in `outputs/eda/`.

1. **Inventory** — files + pixels per class & subclass.
2. **Class mean spectra (raw)** + a SNV-cropped preview.
3. **PCA scatter** — PC1×PC2, PC1×PC3, colored by class then by subclass + scree plot.
4. **UMAP** on first 50 PCs.
5. **Per-class top-10 peak table** (`scipy.signal.find_peaks` on class means; biological annotations).
6. **ANOVA per wavenumber + mutual information** → discriminative-power plot. Plot `log10(F)` (not `-log10(p)`) so top bins don't saturate.
7. **Inter-file cosine-similarity heatmap** of mean spectra, sorted by class/subclass. The single most predictive plot for generalization difficulty.
8. **Subclass overlay + per-subclass silhouette + LDA between-subclass / total variance ratio.**

## Block 9 — same plots on full-preprocessing data (the "after" comparison)

Plots in `outputs/eda_v2/`. Lets us *see* the impact of preprocessing.

- 9.1 Class mean spectra (preprocessed).
- 9.2 PCA scatter (preprocessed).
- 9.3 UMAP (preprocessed).
- 9.4 Inter-file heatmap (preprocessed, color range stretched to vmin=0.98).
- 9.5 Silhouette by subclass (preprocessed).
- 9.6 **NEW: per-peak boxplots** at known biological bands (1004 Phe, 1450 CH₂, 1660 amide-I, 2900 C-H). Concrete chemistry, no PCA magic.
- 9.7 **NEW: PCA colored by `ac_calibration_date`** + distance ratio of same-date vs different-date file pairs. Early batch-effect probe.

## Future EDA blocks (not yet added)

Add when relevant. Don't preemptively build; each should be triggered by a finding or a need.

- **Block 10:** Per-spectrum PCA outlier scan — identify weird spectra before training.
- **Block 11:** Acquisition-time vs class regression — is there a temporal drift in addition to calibration-date batch effect?
- **Block 12:** Memorization probe — train a tiny model to predict `file_id` from a single spectrum; if it works, the encoder is leaking acquisition signature.
