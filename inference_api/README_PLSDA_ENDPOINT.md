# PLS-DA-on-raw endpoint — UI integration notes

Backend addition shipped 2026-05-19 alongside the existing Stage 15F LogReg
endpoint. Both share the same Modal app (`atlas-inference`), parsing path,
pixel cap, and abstain threshold.

## URLs

| Route | URL | Model |
|---|---|---|
| POST `/predict`       | `https://devashishthapliyal1--atlas-inference-predict.modal.run`       | Stage 15F LogReg-L2 (35 MI features) |
| POST `/predict_plsda` | `https://devashishthapliyal1--atlas-inference-predict-plsda.modal.run` | PLS-DA on raw 987-bin spectrum *(project headline)* |
| GET  `/healthz`       | `https://devashishthapliyal1--atlas-inference-healthz.modal.run`        | liveness |

Both `POST` routes accept `multipart/form-data` with a single `file` field
(`.xls` or `.txt`). CORS allow-* is set on responses; preflight `OPTIONS`
routes exist for both predict endpoints.

## Why two endpoints?

- LogReg is McNemar-significant over PLS-DA *on the same 35 engineered
  features* (p = 0.002) and is what the artifact paper deploys.
- PLS-DA on **raw 987-bin spectrum** is the **project LOSO record** (file-
  weighted balanced acc = 0.603 vs LogReg's 0.448). Different feature space,
  different winner.
- Side-by-side they answer the "where are the other models?" question
  visibly: drop a file, see two predictions with sometimes-divergent calls.

## Payload shape

Both routes return the same JSON envelope. `model` is the new discriminator
field — `"logreg_stage15f"` or `"plsda_raw"`.

```jsonc
{
  "class": "Non-STEC",
  "probabilities": {
    "H2O":        0.046,
    "Non-STEC":   0.766,
    "STEC":       0.112,
    "Salmonella": 0.076
  },
  "spectrum_mean": [...],   // 987 floats, preprocessed
  "wn":            [...],   // 987 floats, preprocessed wavenumbers
  "feature_values": { ... }, // 35 entries for logreg, EMPTY {} for plsda_raw
  "abstain": false,          // true when top_prob < 0.55
  "top2": [
    { "class": "Non-STEC", "prob": 0.766 },
    { "class": "STEC",     "prob": 0.112 }
  ],
  "n_pixels_input": 374,     // raw pixel count before 200-pixel cap
  "n_pixels_used":  200,
  "model": "plsda_raw",      // "logreg_stage15f" on /predict

  // PLS-DA interpretability surfaces (ONLY on /predict_plsda; absent on /predict)
  "loadings_per_class": {
    "STEC":       [987 floats],   // per-bin sensitivity for each class
    "Non-STEC":   [987 floats],   // sign tells you direction (positive = bin
    "Salmonella": [987 floats],   // pushes prediction toward that class)
    "H2O":        [987 floats]
  },
  "contribution_for_predicted": [987 floats]
    // per-bin contribution to log-odds of the predicted class for THIS file.
    // sum(contribution_for_predicted) ≈ log-odds of predicted class.
    // Lengths match `wn` exactly. Sign convention: positive means the bin
    // pushed the prediction toward the predicted class.
}
```

`feature_values` is `{}` on the PLS-DA route because PLS-DA-on-raw doesn't
use the engineered 35 features — the existing `FeatureContributionTable`
component should be hidden / replaced when `model === "plsda_raw"`. Replace
it with the spectral driver view described below.

### How to render `loadings_per_class` + `contribution_for_predicted`

Three visualizations the UI can offer, ranked by impact:

1. **"What made the model say X" overlay** — render the mean preprocessed
   spectrum (`spectrum_mean` vs `wn`), then overlay a colored fill where
   `contribution_for_predicted` is positive (push toward predicted class) vs
   negative (push away). Most intuitive view.

2. **Top-N driver bins panel** — sort `wn` by `|contribution_for_predicted|`
   descending, show the top 8–12 bins with their wavenumber + contribution
   value. Cross-reference each with the named bands in
   `ui/public/data/bands.json` (e.g. 1454 cm⁻¹ → "CH₂ lipid"). This gives a
   chemistry-grounded explanation per prediction.

3. **Per-class sensitivity comparison** — plot all 4 lines of
   `loadings_per_class` over `wn`, color-coded by class. Shows globally what
   the model "looks for" to identify each class. Useful as a separate
   reference panel rather than per-prediction.

Loadings are in **standardized space** (input minus StandardScaler mean
divided by scale). Don't try to interpret absolute magnitudes as raw
intensity — they're z-score units. Relative magnitudes within a class and
sign comparisons across classes are the meaningful signal.

## Smoke results (live, 2026-05-19)

| File | Truth | PLS-DA | LogReg |
|---|---|---|---|
| R357 ATCC25922 | Non-STEC | **Non-STEC** 0.77 ✓ | **Non-STEC** 0.79 ✓ |
| R372 H2O blank | H2O | **H2O** 0.88 ✓ | **H2O** 0.98 ✓ |
| R364 O157:H7 mosaic | STEC | **STEC** 0.55 ✓ (abstain line) | Salmonella 0.94 ✗ |
| R370 Dublin mosaic | Salmonella | **Salmonella** 0.55 ✓ (abstain line) | STEC 0.45 ✗ |

PLS-DA gets 4/4 in the smoke; LogReg gets 2/4. The two "saved" predictions
land at exactly the abstain threshold — calibrated uncertainty, not
overconfidence. Cold start ~6–15 s, warm ~1 s. PLS-DA's per-pixel inference
is ~8× faster than LogReg's because it skips the 259-feature engineering
chain — once warm, expect <2 s per file.

## Recommended UI wiring

1. Add `NEXT_PUBLIC_MODAL_PREDICT_PLSDA_URL` to `ui/.env.local` and
   `ui/.env.example`, pointing at the PLS-DA route.
2. Extend `lib/modal-client.ts` with `predictPlsda(file)` — copy of `predict`
   with the alternate URL.
3. Call both endpoints in parallel from `LiveInference` (Promise.all).
   Cold-start cost is paid once per model — running them concurrently won't
   double the latency by much.
4. Render two stacked banners or a side-by-side comparison view; tag each
   with the `model` field.
5. When `model === "plsda_raw"`, skip the feature-contribution table (empty
   `feature_values`) and show a "PLS-DA reads the whole 987-bin spectrum
   directly — no engineered features" note instead.
6. Disagreement is the interesting case — when the two banners pick
   different classes, highlight that visually (this is the answer to the
   "where are the other models?" question).

## How to refit / redeploy the PLS-DA artifact

```bash
# 1. Refit on the full QC-passing cache:
python scripts/fit_plsda_production.py
# Writes: artifacts/plsda_raw_classifier.joblib (~7 MB)
#         artifacts/plsda_raw_metadata.json

# 2. Push to Modal (same app, picks up the new joblib in the image):
cd inference_api && source .venv/bin/activate && modal deploy modal_app.py
```

Both endpoints rebuild from the same `image` block, so a single `modal
deploy` updates both. `n_components = 30` was chosen as the modal winner
across the 9 LOSO folds (HPO grid [5, 8, 10, 12, 15, 20, 25, 30]).
