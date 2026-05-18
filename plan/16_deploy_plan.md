# 16 — Paper + UI Deploy Plan

> **Mutability:** mutable. Sequential 3-phase plan for the post-15E phase: write the research-paper review, train Stage 15F, build the Streamlit UI, deploy to public cloud.
> **Created:** 2026-05-18.
> **Status:** approved (user kicked off Phase 1 — Stage 15F — separately).

---

## Context

Two deliverables on top of Stages 15A–15E:

1. **Research-paper-level review** of the NomadX take-home — dataset, methods, results, citations.
2. **Simple Streamlit UI** wrapping the final classifier, deployed to public cloud (Streamlit Cloud or HuggingFace Spaces), so reviewers can drop in a spectrum file and see a predicted class.

User's explicit choices:
- **Model:** Wait for Stage 15F — the consolidated 259-feature ensemble. *Gating dependency* — UI/deploy cannot start until 15F's classifier artifact exists.
- **UI framework:** Streamlit.
- **Deploy target:** Public cloud (Streamlit Cloud / HF Spaces).
- **Paper format:** Markdown + PDF (via pandoc).

User invoked `/batch`, but the work is strictly **sequential** (15F → save artifact → UI → deploy) and the paper is a single coherent document — not 5+ parallelizable units. Plan uses sequential phases with optional Phase-2 parallelism over independent draft sections only.

---

## Repository state (verified by read-only exploration on 2026-05-18)

- **No saved sklearn models exist.** `outputs/2026-05-14_*/encoder_fold_*.pt` are PyTorch baseline checkpoints, not the headline PLS-DA / Stage 5 XGB. Stage 15F must both *train and serialize* the final classifier.
- **No UI scaffolding exists** (`streamlit_app.py`, `app.py`, `Dockerfile` all absent).
- **References file rich** — `plan/11_references.md` has Cisek-2013, Tang-2026-WGAN, Soupene-2003-K12, Yuan-2024-Salmonella, Marler-Clark-Non-O157, RSCDM-2026, Sun-2025.
- **Feature caches all built:**
  - `data_cache/band_features.parquet` — 7,122 × 166 (per-pixel)
  - `data_cache/spectral_features.parquet` — 7,122 × 51 (per-pixel)
  - `data_cache/unmix_features.parquet` — 87 × 33 (per-file, MCR-ALS DD1)
  - `data_cache/spatial_features.parquet` — 87 × 10 (per-file, moment stats)
- **`joblib` installed**; `streamlit`, `gradio`, `flask`, `pandoc` NOT.

---

## Phase 1 — Stage 15F: train + serialize the final classifier (BLOCKING)

**Goal:** Produce a deployable model artifact that takes raw spectra and outputs (STEC | Non-STEC | Salmonella | H₂O) labels.

### 1.1 Training script `scripts/run_stage15f_final.py`

Aggregate the 4 feature caches to file level (mean-pool per-pixel features per `file_id`), join the 2 already-file-level caches, then:

1. **Mutual-information feature selection per fold** (R1 mitigation; 259 ÷ 87 = 3.0 features/file). Target 30–40 features after MI.
2. **Per-LOSO-fold refits** for the leaky features (MCR-ALS, PCA, SAM) using `MCRALSWrapper.fit/transform` and `fit_roi_pca/transform_roi_pca` (R2 mitigation).
3. **Shuffled-label permutation test** on `mcr_C1_*` features specifically (R7 mitigation) — if shuffled feature importance is ≥ 50% of real importance, drop them.
4. **Multi-seed runs** (5 seeds) for honest LOSO variance.
5. **Per-strain breakdown** including the K-12 lift check (does the Stage 15D α-helix axis stabilize K-12 LOSO above 0.75?).
6. **Train final production classifier** on ALL 87 files (no holdout) using the MI-selected feature set, save via `joblib.dump`:
   - `artifacts/stage15f_classifier.joblib` — fitted sklearn pipeline.
   - `artifacts/stage15f_feature_columns.json` — ordered feature names.
   - `artifacts/stage15f_mcr_global.joblib` — frozen `MCRALSWrapper` for inference.
   - `artifacts/stage15f_roi_pca.joblib` — fitted ROI-PCA dict.
   - `artifacts/stage15f_metadata.json` — LOSO mean recall, per-strain table, model type, training date.

Algorithm choice: try **XGBoost + LogReg + PLS-DA** in the same script; pick the best LOSO mean recall for the production artifact. Keep all three pickled for the paper's ablation table.

### 1.2 Inference module `atlas/inference.py`

```python
def predict_from_xls(path: Path) -> dict:
    """Parse .xls → preprocess → extract features → classify.
    Returns {'class': str, 'probabilities': dict, 'spectrum_mean': ndarray, 'wn': ndarray}.
    """
def predict_from_array(intensities: ndarray, wn: ndarray) -> dict:
    """Same but bypasses the parser — caller already has spectra."""
```

Internally: `atlas.io.parse_file` → `atlas.preprocess.preprocess_matrix` → `atlas.band_features.feature_frame` + `atlas.spectral_features.feature_frame_spectral` → frozen `MCRALSWrapper.transform` + frozen `transform_roi_pca` → file-level aggregation → MI-selected column subset → loaded pipeline.

### 1.3 Append Stage 15F shard

`plan/experiments/2026-05-19_stage15f_full_classifier.md` (or whatever date 15F lands). Same convention as 15A–E: pre-reg → run → results + verdict + branch hit.

### Phase 1 verification (e2e)

```bash
.venv/bin/python scripts/run_stage15f_final.py
.venv/bin/python -c "from atlas.inference import predict_from_xls; \
    print(predict_from_xls('Atlas Data/STEC/O157H7/<one file>.xls'))"
```

### Phase 1 cost

~2–4 hours wall-clock (LOSO retraining). Foreground.

---

## Phase 2 — Paper draft (can overlap with Phase 1 tail)

**Goal:** `PAPER.md` at repo root + `PAPER.pdf` rendered via pandoc.

### 2.1 `PAPER.md` skeleton + Sections 1–2 (Abstract, Dataset)

- Abstract (200–300 words; headline numbers from memory).
- §1 Introduction — Marler-Clark + Soupene-2003 framing.
- §2 Dataset — 87 files / 7,122 QC-passed spectra / 9 strains + water / preprocessing pipeline / class balance / file-vs-pixel levels. Pull from `plan/01_data.md`, `atlas_project.md` memory.

### 2.2 Sections 3–4 (Methods)

- §3 Baseline modeling — classical (PLS-DA, LogReg, SVM, RF, XGB), 1D-CNN, 1D-Transformer, DANN, 5 failed ensembles. Pull from `atlas_project.md` + 2026-05-14/15 shards.
- §4 Band-chemistry track (plan/14, Stages 1–7) — literature triple falsification, LPS chain region, mixed-sample sim. Cite Cisek-2013, Yuan-2024.

### 2.3 Section 5 (Feature engineering, Stages 15A–E)

One subsection per stage:
- §5.1 Stage 15A — pseudo-Voigt + ROI + EMSC + derivatives
- §5.2 Stage 15B — DWT + ROI-PCA + SAM (amide-PC3 discovery)
- §5.3 Stage 15C — MCR-ALS (`mcr_C6_mean` project record)
- §5.4 Stage 15D — biology features (α-helix, K-12 2°-structure axis)
- §5.5 Stage 15E — spatial features (R6 retirement, E.coli↔Salm side-finding)

Source: 5 experiment shards in `plan/experiments/2026-05-18_stage15*.md`.

### 2.4 Sections 6–8 (Results + Discussion + References) — AFTER Phase 1

- §6 Stage 15F results — pulls from `stage15f_metadata.json` + shard.
- §7 Discussion — limitations (87 files, LOSO ceiling per tang-2026-wgan), future work (cross-corpus eval per plan/13, SSL pretraining).
- §8 References — port `plan/11_references.md` to citation format.

### 2.5 Render PDF

`pandoc PAPER.md -o PAPER.pdf --citeproc --bibliography=plan/11_references.bib --pdf-engine=xelatex` (or simpler `wkhtmltopdf` route if LaTeX unavailable).

### Phase 2 verification

`PAPER.md` opens with all section headers; `PAPER.pdf` exists, non-empty, opens.

### Phase 2 cost

~2–3 hours. 2.1–2.3 can run in parallel via 3 subagent worktrees if useful. 2.4 + 2.5 sequential.

---

## Phase 3 — Streamlit UI + cloud deploy

> Use `voltagent-data-ai:mlops-engineer` subagent per user request, even though the agent is more cloud/infra than UI-focused. For a simple Streamlit demo, `general-purpose` would also fit.

**Goal:** Reviewer opens a public URL, uploads an Atlas `.xls`, sees predicted class + probabilities + spectrum plot in under 5 seconds.

### 3.1 `streamlit_app.py`

Single file at repo root:
- File uploader (`.xls` / `.txt` Atlas format).
- Predict button (or auto-predict on upload).
- Output: predicted class (large), per-class probability bar chart, mean spectrum plot with named band annotations.
- Sidebar: model explainer paragraph, link to PAPER.md, dataset stats.

Imports `atlas.inference.predict_from_xls`. Loads artifacts via `@st.cache_resource`.

### 3.2 `requirements.txt`

Minimal pinned set for UI + inference. NOT the full `.venv`. Streamlit Cloud + HF Spaces expect this at repo root.

### 3.3 Deploy target choice + push

Two paths (agent picks):
- **Streamlit Community Cloud**: connect GitHub repo → point at `streamlit_app.py` → free tier handles small models. Artifacts under ~100MB total (joblib pickles fine).
- **HuggingFace Spaces**: Streamlit SDK, free tier, Git-LFS if needed.

Agent writes `DEPLOY.md` with public URL + redeploy instructions.

### 3.4 `README.md` update

Top-level README adds "Try it" section:
- Link to deployed UI.
- One-paragraph project summary.
- Links to PAPER.md / PAPER.pdf.
- `streamlit run streamlit_app.py` for local.

### Phase 3 verification (e2e)

1. Local: `streamlit run streamlit_app.py` → browser → upload known STEC `.xls` → prediction = "STEC".
2. Cloud: same upload via public URL → same prediction.
3. Latency < 5 s (parse + preprocess + features + predict).

### Phase 3 cost

~3–4 hours including cloud account setup, artifact upload, smoke tests.

---

## Total estimated wall-clock

- Phase 1: 2–4 hours (blocking — Stage 15F training; user is running this now)
- Phase 2: 2–3 hours (overlaps with Phase 1 tail)
- Phase 3: 3–4 hours (after Phases 1 + 2)

**Total: ~7–11 hours.** Feasible in one focused session if Phase 1 runs overnight; otherwise spread over 2 days.

---

## Critical files / utilities to reuse

| Existing | Used by | Purpose |
|---|---|---|
| `atlas/io.py::parse_file` | inference.py | Parse Atlas `.xls`/`.txt` |
| `atlas/preprocess.py::preprocess_matrix` | inference.py | arPLS+SG+SNV pipeline |
| `atlas/band_features.py::feature_frame` | inference.py | Per-pixel band features |
| `atlas/spectral_features.py::fit_roi_pca, transform_roi_pca, fit_sam_templates, transform_sam` | training + inference | Per-fold-fit features |
| `atlas/unmix_features.py::MCRALSWrapper` | training + inference | MCR-ALS unmixing per fold |
| `atlas/spatial_features.py::feature_frame_spatial` | inference.py | Spatial moment features |
| `plan/11_references.md` | PAPER.md | Bibliography source |
| `plan/experiments/2026-05-{17,18}_stage*.md` | PAPER.md §5 | Per-stage write-ups |
| Memory files in `~/.claude/projects/.../memory/` | PAPER.md throughout | Synthesis source |

---

## Explicit non-scope

- No Stages 4 / 6 (deferred per stage-gates).
- No Dockerfile (cloud over Docker; Streamlit Cloud handles it).
- No auth / user accounts for the UI.
- No cross-corpus / Bacteria-ID transfer (plan/12 future work).
- No `/batch` parallel-worktree pattern; sequential phases explained above.

---

## Full e2e verification recipe

1. **Phase 1 done check:** `artifacts/stage15f_classifier.joblib` exists, `python -c "from atlas.inference import predict_from_xls; predict_from_xls(<a file>)"` returns the right class label.
2. **Phase 2 done check:** `PAPER.md` + `PAPER.pdf` exist at repo root; PDF opens; all 8 sections present; reference list resolves.
3. **Phase 3 done check:** public URL responds; uploading a known STEC file returns STEC; latency < 5 s; `README.md` links work.

---

## Approval-required actions (will surface as I go)

- `.venv/bin/pip install streamlit pandoc` (non-destructive install).
- New files: `scripts/run_stage15f_final.py`, `atlas/inference.py`, `streamlit_app.py`, `PAPER.md`, `requirements.txt`, `DEPLOY.md`, Stage 15F shard.
- Run `.venv/bin/python scripts/run_stage15f_final.py` (long-running; foreground).
- Commit + push to GitHub (required for Streamlit Cloud / HF Spaces).
- Cloud-deploy step: requires user to either (a) hand me a deploy token / OAuth in browser, or (b) do the final connect-repo click in Streamlit Cloud / HF Spaces UI themselves. **Agent cannot click through third-party OAuth flows** — user has to do that step. Will surface when Phase 3 hits it.

---

## How to resume after Stage 15F finishes

Once Phase 1 produces `artifacts/stage15f_classifier.joblib` and the Stage 15F shard is filled in:

1. Re-open this session (or a new one) with the user.
2. Tell me 15F is done; I'll read the shard's Results section + the metadata JSON.
3. I'll start Phase 2 (paper draft) and Phase 3 (UI build) — possibly in parallel since they're now decoupled.
