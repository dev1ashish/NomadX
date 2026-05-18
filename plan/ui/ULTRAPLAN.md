# Atlas Raman UI — Ultraplan (Next.js + Vercel + Modal)

> **Generated 2026-05-19.** Companion to `FINAL/PAPER.md`. The UI is a viewer
> over the cached Stage 15F truth — not a re-implementation of any pipeline.
> Reproducibility-first: every plot is backed by a parquet/npy/json sidecar.

**For agentic workers:** REQUIRED SUB-SKILL — Use `superpowers:subagent-driven-development` to fan out W2–W8 in parallel after W1+W9 unblock; W10 is the sequential bookend.

**Goal.** Ship a scientist-WOW 7-tab Next.js app that surfaces the Atlas
Raman story (data → preprocessing → bands → features → MCR → live inference
→ results) with NomadX visual identity. Vercel for the frontend, Modal for
the Python inference endpoint.

**Architecture.**
```
                Browser
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
  Vercel (Next.js)     Modal (Python)
  ─ static pages        ─ atlas.inference
  ─ /public/data/*.json   ─ POST /predict
  ─ talks to Modal        ─ joblib artifacts
       via CORS           ─ 2 GB image, scale-to-zero
```

**Tech Stack.**
- **Frontend:** Next.js 15 (App Router) · TypeScript · Tailwind CSS · shadcn/ui · `react-plotly.js` (dynamic import, per-tab) · Framer Motion · `next/font/local` for General Sans + JetBrains Mono
- **Inference:** Modal (`@modal.fastapi_endpoint`) wrapping `atlas.inference.predict_from_xls()` 1:1, CORS-allow `https://*.vercel.app` and `localhost:3000`
- **Build:** `scripts/build_sidecars.py` (uv) emits all JSON into `ui/public/data/`; committed so Vercel ships them as static assets
- **Package manager:** `pnpm` for the Next.js side, `uv` for Python

---

## 0. Research pass (done — verified before planning)

### 0.1 `FINAL/PAPER.md` narrative → tab mapping

| Section | What the UI surfaces |
|---|---|
| §2 Dataset (87 files / 7,122 spectra / 987 bins, 4 classes × 9 strains + H₂O) | Tab 1 (Inventory) |
| §2.3 Preprocessing pipeline (cosmic-ray → arPLS → SG → crop → SNV) | Tab 2 toggle (raw / preprocessed) |
| §2.6 30 named bands × 5 macromolecule groups; LPS regions 400–900 + 800–1200 | Tab 2 overlays + Tab 3 primer |
| §3.1 PLS-DA LOSO 0.603 = project record | Tab 7 hero + Tab 1 KPI chip |
| §4.1 Cisek triple null at file level; §4.2 LPS-chain region = empirical anchor | Tab 3 falsification panel |
| §5 Stages 15A–E: 259 features × 5 families | Tab 4 feature explorer |
| §5.3 MCR-ALS K=7; `mcr_C6_mean` d=−1.23 (global) | Tab 5 unmixing demo |
| §6 Stage 15F: LogReg-L2, 35 MI features, LOSO 0.448 fw, CI [0.345, 0.552] | Tab 6 (Live) + Tab 7 (Results) |
| §6.4 McNemar p=0.0020 LogReg > PLS-DA on engineered features | Tab 7 contingency |
| Stage 7 — 10–20% mixed-sample degradation at 25% contamination | Tab 7 deployment callout |

### 0.2 NomadX visual scan (Playwright DOM extraction, 2026-05-19)

**Design tokens — used verbatim in `ui/styles/tokens.css`:**

```css
:root {
  /* Brand palette (top color counts from www.nomadxholdings.com) */
  --nx-bg:           #000000;   /* canvas */
  --nx-fg:           #FFFFFF;   /* primary text */
  --nx-accent:       #39B8DC;   /* signature cyan (H2, CTAs) */
  --nx-accent-deep:  #135A6F;   /* secondary teal */
  --nx-bg-elev-1:    #04161B;   /* elevated panel */
  --nx-bg-elev-2:    #0C3845;   /* card hover */
  --nx-muted:        #313131;   /* borders, dividers */
  --nx-danger:       #FF0000;

  /* Class colors (Atlas-specific, harmonized) */
  --class-stec:       #D63333;
  --class-nonstec:    #1F7A4D;
  --class-salm:       #7A3D99;
  --class-h2o:        #3070B5;

  /* Type (Saans is paid; General Sans from Fontshare is the free match) */
  --font-display: 'General Sans Variable', system-ui, sans-serif;
  --font-body:    'General Sans Variable', system-ui, sans-serif;
  --font-mono:    'JetBrains Mono Variable', ui-monospace, monospace;

  /* Type scale (NomadX heading sizes: H2=49 lh53.9, H3=39 lh42.9, weight 400) */
  --t-display: clamp(2.5rem, 4vw + 1rem, 3.5rem);
  --t-h2:      3.0625rem;
  --t-h3:      2.4375rem;
  --t-body:    1rem;
  --t-mono:    0.875rem;

  /* Spacing — 4px grid */
  --s-1: 4px; --s-2: 8px; --s-3: 12px; --s-4: 16px;
  --s-5: 24px; --s-6: 32px; --s-8: 48px; --s-10: 64px;

  --radius-0: 0px;   /* NomadX buttons are radius-0 */
  --radius-sm: 4px;  /* internal */
  --hairline: 1px solid var(--nx-muted);
}
```

Tailwind config extends with these tokens so utility classes resolve
(`bg-nx-bg`, `text-nx-accent`, `font-mono`, etc.).

### 0.3 Inference contract — Modal wraps as-is

`atlas/inference.py:184 predict_from_xls(path)` returns the dict described
in its docstring. Modal endpoint accepts `multipart/form-data` with one
`file` field, writes to a NamedTemporaryFile, calls `predict_from_xls`,
returns the same dict as JSON (numpy arrays cast to `list[float]`).

### 0.4 Cost & deploy reality check (Modal)

- Modal **free tier: $30/mo compute credit**, scale-to-zero. Inference is
  ~5 s × ~few requests/day → $0.
- Modal cold start: ~3–5 s the first request after sleep; warm: ~500 ms +
  inference work.
- Atlas artifacts (~2 MB joblib) baked into the image; no S3 needed.
- Modal `add_local_dir` ships the `atlas/` package + `artifacts/` into the
  container — no separate package upload.

---

## 1. Tech stack rationale

Why **each** library, in one line:

- **Next.js 15 App Router** — file-based routing (`/inventory`, `/spectrum`, ...) is the natural fit for the 7-tab IA; SSR for fast first paint.
- **TypeScript** — sci-data types (`FileMeta`, `Band`, `Feature`, `PredictionResponse`) are too easy to get wrong in vanilla JS.
- **Tailwind** — pairs perfectly with design tokens via `@theme`; co-locates spacing/typography with markup.
- **shadcn/ui** — Radix primitives + accessible defaults; Button/Select/Card/Dialog/Tooltip/HoverCard ship the day you `npx shadcn add`.
- **`react-plotly.js`** — sci-grade interactive plots (zoom/pan/box-select) with one-line config; dynamically imported per tab so the 3 MB bundle doesn't hit tabs that don't need it.
- **Framer Motion** — orchestrated page-load reveals + tab transitions are the "WOW" delta.
- **`next/font/local`** — preloads Fontshare WOFF2s, eliminates FOIT, Vercel-edge cached.
- **Modal** — single-file Python deploy, no Docker, no Dockerfile, no `pip freeze`. `modal deploy` is the only command you'll learn.

---

## 2. Repo layout

```
NomadX/                                # repo root
├── atlas/                             # unchanged
├── artifacts/                         # unchanged — Stage 15F artifacts
├── data_cache/                        # unchanged — parquet/npy
├── FINAL/                             # unchanged — paper + images
│
├── ui/                                # NEW — Next.js frontend
│   ├── package.json
│   ├── tsconfig.json
│   ├── next.config.mjs
│   ├── tailwind.config.ts
│   ├── postcss.config.mjs
│   ├── components.json                # shadcn registry config
│   ├── .env.example
│   ├── README.md
│   ├── app/
│   │   ├── layout.tsx                 # root: fonts, TopBar, TabNav, providers
│   │   ├── page.tsx                   # / → redirect to /inventory
│   │   ├── (tabs)/
│   │   │   ├── inventory/page.tsx
│   │   │   ├── spectrum/page.tsx
│   │   │   ├── primer/page.tsx
│   │   │   ├── features/page.tsx
│   │   │   ├── mcr/page.tsx
│   │   │   ├── live/page.tsx
│   │   │   └── results/page.tsx
│   │   └── globals.css
│   ├── components/
│   │   ├── ui/                        # shadcn primitives (auto-generated)
│   │   ├── layout/
│   │   │   ├── TopBar.tsx
│   │   │   ├── TabNav.tsx
│   │   │   └── KpiStrip.tsx
│   │   ├── plots/
│   │   │   ├── PlotlyChart.tsx        # SSR-safe dynamic wrapper
│   │   │   ├── SpectrumViewer.tsx
│   │   │   ├── ConfusionMatrix.tsx
│   │   │   ├── PureSpectraStack.tsx
│   │   │   ├── FeatureViolin.tsx
│   │   │   ├── BootstrapHistogram.tsx
│   │   │   └── ProbabilityBars.tsx
│   │   └── tabs/                      # one composed component per tab
│   │       ├── InventoryHero.tsx
│   │       ├── SpectrumExplorer.tsx
│   │       ├── BandPrimer.tsx
│   │       ├── FeatureBrowser.tsx
│   │       ├── McrDemo.tsx
│   │       ├── LiveInference.tsx
│   │       └── ResultsPanel.tsx
│   ├── lib/
│   │   ├── types.ts                   # PredictionResponse, FileMeta, Band, Feature
│   │   ├── data.ts                    # typed fetchers for /data/*.json
│   │   ├── modal-client.ts            # POST to NEXT_PUBLIC_MODAL_PREDICT_URL
│   │   ├── plotly-theme.ts            # NomadX-themed layout/colorway defaults
│   │   └── cn.ts                      # className helper
│   ├── public/
│   │   ├── data/                      # built by scripts/build_sidecars.py
│   │   │   ├── inventory.json
│   │   │   ├── bands.json
│   │   │   ├── feature_catalog.json
│   │   │   ├── mcr_components.json
│   │   │   ├── stage15f.json
│   │   │   ├── confusion.json
│   │   │   ├── bootstrap.json
│   │   │   ├── mcnemar.json
│   │   │   └── spectra/<file_id>.json × 87
│   │   ├── fig/                       # copied from FINAL/images/
│   │   └── fonts/                     # General Sans + JetBrains Mono WOFF2s
│   ├── styles/
│   │   └── tokens.css                 # the var block from §0.2
│   └── scripts/
│       └── build_sidecars.py          # uv-runnable, parquet/npy → public/data/
│
└── inference_api/                     # NEW — Modal Python app
    ├── modal_app.py
    ├── pyproject.toml                 # atlas package + fastapi + modal
    └── README.md
```

---

## 3. The 7-tab IA — ASCII mockup

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ◆ NOMADX · ATLAS RAMAN                                  v0.1 · local   │  ← TopBar
├─────────────────────────────────────────────────────────────────────────┤
│  Inventory  Spectrum  Primer  Features  MCR-ALS  Live  Results          │  ← TabNav
│  ▔▔▔▔▔▔▔▔▔ (active = cyan underline + Framer Motion layoutId pill)       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   87 FILES        7,122 SPECTRA       987 BINS       LOSO 0.603         │  ← KpiStrip (Mono)
│   ───────         ─────────────       ────────       ──────────         │     fade-in stagger
│                                                                         │
│   <hero text in cyan H2>                                                │
│                                                                         │
│   ┌───────────────────────────┐  ┌────────────────────────────────┐    │
│   │  Plotly: class composition│  │  Plotly: per-strain breakdown   │    │
│   └───────────────────────────┘  └────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

Tab transitions: Framer Motion `<AnimatePresence>` cross-fade (`opacity` +
`y: 8px`, 200 ms). KpiStrip values: stagger reveal `delay = i * 0.05s` on mount.

---

## 4. Work units (parallelization plan)

```
W1 Next.js foundation
  ├──► W9 Modal endpoint (parallel)
  │
  └──► [W2..W8: one tab each, parallel]
                │
                └──► W10 integration + deploy
```

W1 publishes `lib/types.ts` first, which lets W9 (Modal) and W2–W8 (tabs)
proceed in parallel without contract churn.

### W1 — Next.js foundation (~3h)

**Files created**
- `ui/package.json` (next 15, react 19, typescript, tailwind 4, framer-motion, react-plotly.js, plotly.js-dist-min, lucide-react, clsx, @radix-ui/react-*, class-variance-authority, tailwind-merge)
- `ui/tsconfig.json`, `ui/next.config.mjs`, `ui/tailwind.config.ts`, `ui/postcss.config.mjs`, `ui/components.json`
- `ui/.env.example` → `NEXT_PUBLIC_MODAL_PREDICT_URL=https://...modal.run`
- `ui/styles/tokens.css` + `ui/app/globals.css`
- `ui/app/layout.tsx` (loads fonts, mounts `<TopBar>`, `<TabNav>`, wraps `<AnimatePresence>`)
- `ui/components/layout/{TopBar,TabNav,KpiStrip}.tsx`
- `ui/components/plots/PlotlyChart.tsx` (the SSR-safe `dynamic(() => import('react-plotly.js'), { ssr: false })` wrapper)
- `ui/lib/{types,data,modal-client,plotly-theme,cn}.ts`
- `ui/public/fonts/{GeneralSans-Variable.woff2, JetBrainsMono-Variable.woff2}`
- `ui/scripts/build_sidecars.py` (idempotent uv-runnable)
- `ui/README.md`
- `npx shadcn@latest init` + `add button card select dialog tooltip hover-card switch toast badge`

**Acceptance**
- `pnpm dev` boots on `:3000`. Black canvas, cyan wordmark, 7 tab placeholders.
- All 7 routes `/inventory`, `/spectrum`, … render an empty styled shell.
- Tab clicks animate (Framer Motion).
- Type-check passes: `pnpm tsc --noEmit`.
- `python scripts/build_sidecars.py` materializes all `/public/data/*.json` from upstream caches.
- Lighthouse perf ≥90 on empty shell.

### W9 — Modal inference endpoint (~3h, parallelizable after W1 types)

**File: `inference_api/modal_app.py`**

```python
import modal
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy", "pandas", "scikit-learn", "joblib", "pybaselines",
        "PyWavelets", "scipy", "fastapi[standard]",
    )
    .add_local_dir(str(REPO / "atlas"),      remote_path="/root/atlas")
    .add_local_dir(str(REPO / "artifacts"),  remote_path="/root/artifacts")
)

app = modal.App("atlas-inference", image=image)

@app.function(memory=2048, cpu=2.0, min_containers=0, scaledown_window=300)
@modal.fastapi_endpoint(method="POST", docs=True)
def predict(file: bytes):
    import os, sys, tempfile, json
    sys.path.insert(0, "/root")
    os.environ["ATLAS_ARTIFACTS_DIR"] = "/root/artifacts"
    from atlas.inference import predict_from_xls
    with tempfile.NamedTemporaryFile(suffix=".xls", delete=False) as tmp:
        tmp.write(file)
        path = tmp.name
    result = predict_from_xls(path)
    return {
        "class": result["class"],
        "probabilities": result["probabilities"],
        "spectrum_mean": result["spectrum_mean"].tolist(),
        "wn": result["wn"].tolist(),
        "feature_values": result["feature_values"],
    }
```

CORS handled via the `@modal.fastapi_endpoint` decorator's default middleware
+ a `custom_domains` config if user wants a stable URL.

**Acceptance**
- `modal deploy modal_app.py` → URL printed.
- `curl -F file=@Atlas\ Data/STEC/O157H7/R357_*.xls $URL/predict` returns `{"class": "STEC", ...}` in <8 s cold, <3 s warm.
- Endpoint URL exported to `ui/.env.local` as `NEXT_PUBLIC_MODAL_PREDICT_URL`.

### W2 — Inventory tab (~3h)

**Sidecar**: `inventory.json` (87 × `{file_id, class, strain, n_pix, qc_pass_rate}`).
**Plots (Plotly via dynamic import)**:
1. Class composition stacked bar.
2. Per-strain small-multiples — files × QC-passed pixels.
3. QC-retention strip.

**Wow moves**: Framer Motion stagger on KPI numbers (count-up); hover-card on strain → file list with class-color dot.

### W3 — Spectrum explorer (~4h)

**Sidecar**: `spectra/<file_id>.json` × 87 (~30 KB each, gzipped → ~2.5 MB total).
**Layout**: shadcn `<Command>` combobox file picker (group by class), shadcn `<Switch>` for raw/preprocessed, Plotly trace with 7 anchor annotations (1004, 1117, 1194, 1242, 1338, 1454, 1658, 2900). Each annotation = hoverable shadcn `<HoverCard>` with chemistry one-liner from `atlas/band_features.BANDS`.

### W4 — Band-chemistry primer (~3h)

**Sidecar**: `bands.json` (30 entries with chemistry + d-values).
**Layout**: 5 grouped shadcn `<Card>`s (Aromatic AA, Protein amide, Nucleic acid, Lipid+carb, Metabolite). Each band: center, FWHM, chemistry one-liner, "discriminates" badge.
**Special panel**: Cisek-2013 falsification — bands 1338/1454/1658 with strike-through and the d-values (+0.13, −0.47, +0.16). Framer Motion: each card fades in on scroll (`whileInView`).

### W5 — Feature engineering (~5h)

**Sidecar**: `feature_catalog.json` (259 entries: `{name, family, region, d_stec_nonstec, d_ecoli_salm, mi_rank_stage15f | null}`).
**Layout**: family chips (Band 166 / Spectral 51 / MCR 32 / Spatial 10), top-15 by |d| (Plotly bar, color by family), selected-feature panel showing per-class violin + plain-English "what this is".
**Surfacing the headline**: Stage 15F's 35 MI-selected features highlighted; "all top-10 are Stage 15A peak-fits / derivatives" callout; "0 MCR features survived per-fold MI" caveat banner.

### W6 — MCR-ALS demo (~4h)

**Sidecar**: `mcr_components.json` (K=7 pure spectra + per-class mean concentrations).
**Layout**: 7-line stacked plot with shadcn `<Switch>` per component, reconstruction view at bottom (observed vs Σ weighted components + residual).
**Caveat callout**: "MCR features did NOT survive per-fold MI in Stage 15F (§6.7). Global-fit d-values are partly a leakage artifact."

### W7 — Live inference (~3h)

Uses W9's endpoint. shadcn `<input type=file>` drag-drop + corpus dropdown (87 entries). Posts to `NEXT_PUBLIC_MODAL_PREDICT_URL` as `multipart/form-data`. Renders:
1. Big color-coded class banner (Framer Motion scale-in on result).
2. 4-bar probability chart (animated bar growth).
3. Mean spectrum with band overlays (reused `<SpectrumViewer>`).
4. 35-feature contribution table (shadcn `<Table>`).

Latency target: ≤8 s on Modal cold start, ≤3 s warm.

### W8 — Results (~4h)

**Sidecars**: `confusion.json`, `bootstrap.json`, `mcnemar.json`, `stage15f.json`.
**Layout**:
- KPI strip — PLS-DA 0.603 vs LogReg-L2 0.448 CI [0.345, 0.552].
- Plotly heatmap confusion matrix (4×4), click cell → shadcn `<Dialog>` with file list.
- Algo-compare bar (PLS-DA / LogReg / XGB).
- Bootstrap histogram with 95% CI band (Plotly hist).
- McNemar 2×2 contingency table (shadcn `<Table>`) + p=0.0020.
- Stage 7 deployment callout (10–20% drop at 25% contamination).

### W10 — Integration + deploy (~3h, sequential)

1. Cross-tab deep-links (Tab 1 file click → Tab 2 with `?file=...`; Tab 4 feature click → Tab 2 + scroll to band).
2. Run verification (§6).
3. Capture 7 screenshots to `ui/screenshots/` at 1440×900.
4. Update `ui/README.md` with the deploy steps (§5.2).
5. `vercel deploy` (preview) → verify; promote with `vercel --prod`.

---

## 5. Run + deploy steps (for the user)

### 5.1 Local dev (one-time setup)

```bash
# 1. Modal — authenticate once
cd inference_api
uv venv && source .venv/bin/activate
uv pip install modal
modal token new        # opens browser, one-time

# 2. Deploy the inference endpoint
modal deploy modal_app.py
# → prints the URL, e.g. https://yourname--atlas-inference-predict.modal.run

# 3. Next.js
cd ../ui
pnpm install
echo "NEXT_PUBLIC_MODAL_PREDICT_URL=https://yourname--atlas-inference-predict.modal.run" > .env.local

# 4. Build sidecars (one-time; idempotent)
uv run scripts/build_sidecars.py

# 5. Run dev server
pnpm dev               # → http://localhost:3000
```

### 5.2 Production deploy

```bash
# Modal (Python inference) — already deployed in step 2 above
# Re-deploy on code change:
cd inference_api && modal deploy modal_app.py

# Vercel (Next.js) — one-time link, then deploy
cd ../ui
pnpm dlx vercel link              # one-time
pnpm dlx vercel env add NEXT_PUBLIC_MODAL_PREDICT_URL production
pnpm dlx vercel --prod
# → e.g. https://atlas-raman.vercel.app
```

### 5.3 Re-deploy after changes

| What changed | What to redeploy |
|---|---|
| Pure UI / sidecar / static data | `vercel --prod` |
| `atlas/` code or `artifacts/` rebuilt | `modal deploy` (image rebuilds, ~1–2 min) |
| Both | both, any order — they're independent |

---

## 6. Worker prompt template (copy-paste for fan-out)

```
SUBJECT: Atlas Raman UI — Work Unit {W_ID} — {TAB_NAME}

You are building one tab of the Atlas Raman Next.js app. The full plan is
at `plan/ui/ULTRAPLAN.md`. The scientific narrative is in `FINAL/PAPER.md`.

## Your scope
- Plan §4 W{W_ID} is your spec.
- Read `ui/lib/types.ts` first — it has the contracts.
- Foundation (W1) ships: tokens.css, layout, TopBar, TabNav, PlotlyChart
  wrapper, plotly-theme, fetcher helpers. Use them — don't re-invent.
- Add ONE route file + ONE composed component + plot components as needed:
    - `ui/app/(tabs)/{slug}/page.tsx`
    - `ui/components/tabs/{TabName}.tsx`
    - `ui/components/plots/*.tsx` (if new)
- Data: read from `/data/*.json` via `lib/data.ts`. If your tab needs data
  not yet sidecar-ed, ADD it to `ui/scripts/build_sidecars.py` — do NOT
  fetch from parquet/npy at runtime.

## Constraints
- TypeScript strict. No `any` without a comment justifying it.
- shadcn/ui for ALL form/dialog/tooltip primitives — don't roll your own.
- Plotly imported only via `<PlotlyChart>` (the SSR-safe wrapper). Bare
  imports break SSR.
- Tailwind utility classes; refer to design tokens via `bg-nx-bg`,
  `text-nx-accent`, etc. (configured in tailwind.config.ts).
- Framer Motion for tab-entry stagger + hover/tap delights. Keep
  animations subtle — 200–300ms, ease-out.
- Mobile/tablet: graceful (1024px+); no requirement past that.

## Deliverables
- The files above.
- `pnpm tsc --noEmit` passes.
- `pnpm dev` → tab works, no console errors.
- Screenshot at 1440×900 → `ui/screenshots/{W_ID}_{slug}.png`.
- 1-paragraph note in your reply: what's done, what's not, surprises.

## Verification before "done"
1. `pnpm dev`, navigate to your tab.
2. DevTools Network: ONLY your sidecars + fonts; no parquet/npy loaded.
3. DevTools Console: zero errors, zero warnings.
4. Resize to 1024px: no horizontal scroll, no broken layout.
5. All interactions described in §4 W{W_ID} work end-to-end.

Do NOT touch atlas/, artifacts/, data_cache/, FINAL/. Do NOT add cloud
config (Vercel/Modal are managed in W10/W9). Do NOT add auth.
```

---

## 7. Verification plan

### 7.1 Golden-path test (manual, ~10 min)

1. `pnpm dev` → :3000, black canvas, cyan wordmark.
2. **Inventory**: KPI strip animates in `87 · 7,122 · 987 · 0.603`. Hover `O157H7` → file list popover.
3. **Spectrum**: pick `R357_*` → spectrum loads in <1 s. Toggle raw/preprocessed → trace updates with `transition`. Hover `1117 cm⁻¹` → chemistry popover.
4. **Primer**: scroll through 5 cards. Cisek falsification panel visible with strike-through bands.
5. **Features**: top-15 bar loads. Click `mcr_C6_mean` → violin renders. "0 MCR features survived" callout present.
6. **MCR-ALS**: 7 traces, toggle C6 → reconstruction updates. Caveat visible.
7. **Live**: pick an O157:H7 file from corpus → predicted class = STEC in <5 s warm; ≤8 s cold. Banner animates in. Top features table populates.
8. **Results**: confusion matrix renders. Click H₂O × STEC cell → dialog shows 8 files (Stage 7 bias). Bootstrap histogram + CI band visible. McNemar 2×2 + p=0.0020.

### 7.2 Reproducibility test

```bash
diff <(curl -s localhost:3000/data/stage15f.json | jq .loso_mean_acc) \
     <(jq .loso_mean_accuracy artifacts/stage15f_metadata.json)
# expected: empty diff
```

UI never invents numbers — every headline comes from a sidecar built off
the artifact JSON.

### 7.3 Performance test

- Lighthouse perf ≥85, a11y ≥90 on each tab (Vercel preview build).
- Tab-switch latency: <250 ms (no full reload, AnimatePresence).
- Spectrum tab cold load (first file): <600 ms after sidecar fetch.
- Modal endpoint warm latency p50: ≤3 s; p95: ≤6 s.

### 7.4 Inference correctness test

For 4 representative held-out files (one per class) call the Modal endpoint
and compare predicted class against
`artifacts/stage15f_loso_predictions.parquet`. Must agree (same code path).

### 7.5 Aesthetic test

Side-by-side `/inventory` hero against `https://www.nomadxholdings.com/`:
- Same background (`#000`).
- Cyan H2 within ΔE 5 of `#39B8DC`.
- General Sans character (geometric, weight 400, generous line-height).
- Avenue-Mono-equivalent for stats (JetBrains Mono).

### 7.6 Lighthouse + bundle audit

```bash
pnpm dlx @next/bundle-analyzer
# Targets:
#   First Load JS shared: <120 KB
#   Spectrum tab JS: <800 KB after Plotly dynamic import
#   Total CSS: <40 KB
```

---

## 8. Out-of-scope

- User auth — none.
- Write operations / data mutation — UI is read-only.
- Re-implementing any pipeline step. `atlas.inference.predict_from_xls()` is the only inference path.
- Mobile-first design (≥1024 px viewport is the bar).
- Training the model — Stage 15F artifacts are frozen.
- New scientific analyses — UI surfaces existing findings.

---

## 9. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Modal cold start >10 s breaks Vercel hobby tier 10s edge timeout | medium | Browser calls Modal **directly** (CORS), not via Vercel function. Vercel's timeout doesn't apply. |
| `react-plotly.js` SSR error (`window is not defined`) | high | All Plotly use must go through `<PlotlyChart>` wrapper which uses `dynamic(..., { ssr: false })`. Enforced in worker prompt. |
| Plotly bundle (3 MB) blows Lighthouse | medium | Per-tab dynamic import; route-level code-split. Spectrum + MCR + Results bundles diverge naturally. |
| Sidecar gzipped spectra/* directory too large (~2.5 MB) on Vercel | low | Well under Vercel's 1 GB static budget. Brotli compression auto-applied at edge. |
| Modal image build slow (every deploy rebuilds) | low | Modal caches layers; `pip_install` rarely re-runs. ~30 s after first deploy. |
| `pybaselines` / `PyWavelets` missing wheels for Modal's Python 3.11 | low | Both have manylinux wheels. If not, fall back to Python 3.10 in image. |
| Tailwind v4 RC bugs (it's still pre-1.0 as of cutoff) | low | Pin to a known-good `4.0.0-beta.10` or fall back to 3.4 if breaks. |
| User wants different fonts later | low | `next/font/local` + `tokens.css` — swap one block. |

---

## 10. Total estimate

- W1 foundation: 3h (sequential — blocks W2–W8)
- W9 Modal: 3h (parallel with W2–W8 after types land in W1)
- W2–W8 tabs: ~3–5h each, **parallelizable** → wallclock ~5h if 7 agents run
- W10 integration + deploy: 3h (sequential)

**Wallclock with parallel fan-out: ~14 h** (W1 3h → W2–W8 + W9 in parallel ~5h → W10 + verification 3h, +buffer). Without parallelism: ~32 h.

---

## Self-review (against the brief)

- [x] Playwright scan worked, palette + type extracted (§0.2).
- [x] PAPER.md narrative mapped to all 7 tabs (§0.1).
- [x] Tech stack recommendation with one-sentence justification per lib (§1).
- [x] Parallel work units (§4 — W1 + W9 + 7 parallel tabs + W10).
- [x] Worker prompt template ready to dispatch (§6).
- [x] Verification plan (§7 — 6 distinct tests).
- [x] Run + deploy steps for user (§5).
- [x] Out-of-scope items (§8).
- [x] Reuses `atlas.inference.predict_from_xls()` verbatim via Modal wrap (§W9).
- [x] Reuses parquet/npy/json caches — no fresh computation (sidecar build).
- [x] Vercel for frontend, Modal for Python — user deploys both themselves with `vercel --prod` and `modal deploy`.
- [x] Reuses `FINAL/images/` PNGs as static fallbacks (Primer + Results).

**Callouts (per `[[call-out-mistakes]]` memory):**

1. **Vercel alone can't host this.** The Python inference depends on
   sklearn + pybaselines + joblib artifacts. Modal is the second piece;
   that's now in the plan.

2. **You're now managing two deploys** (Vercel + Modal) instead of one
   local Streamlit. Both are CLI-driven and git-pushable, but it's
   genuinely more surface area. The original local-only FastAPI plan is
   still saved in git history if you want to fall back.

3. **`voltagent mlops-engineer`** *does* fit here for the Modal portion
   (containerization, autoscaling, cost monitoring). Previously I said it
   didn't — that was correct for the old local-only plan, but for this
   stack the mlops-engineer subagent is a reasonable owner of W9.
