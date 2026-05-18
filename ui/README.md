# Atlas Raman UI

Next.js 16 (App Router) frontend for the Atlas Raman bacterial classifier —
a 7-tab scientific dashboard wrapping the Stage 15F LogReg-L2 model and the
PLS-DA LOSO 0.603 baseline.

Companion to `FINAL/PAPER.md` and `plan/ui/ULTRAPLAN.md`.

---

## Requirements

- **Node.js** 20.9+
- **pnpm** 9+ (`npm install -g pnpm` if missing)
- **Python** 3.11+ — only for the per-tab sidecar builders under `scripts/`
- **uv** (recommended) for running Python scripts: `pip install uv`

## One-time setup

```bash
cd ui
pnpm install
cp .env.example .env.local      # then edit NEXT_PUBLIC_MODAL_PREDICT_URL
```

Build the static JSON sidecars once (reads from `../data_cache/` and
`../artifacts/`):

```bash
python scripts/build_inventory.py
python scripts/build_bands.py
python scripts/build_spectra.py        # writes 87 files under public/data/spectra/
python scripts/build_features.py
python scripts/build_mcr.py
python scripts/build_results.py
```

Re-run any of these when their upstream cache changes. They're idempotent.

## Run

```bash
pnpm dev          # http://localhost:3000  (Turbopack, hot reload)
pnpm build        # production build
pnpm start        # serve production build
pnpm tsc --noEmit # typecheck (no JS emitted)
pnpm lint         # ESLint flat-config
```

If `:3000` is occupied (e.g. the legacy `streamlit_app.py` is running),
use `pnpm dev --port 3137`.

## Environment variables

| Var | Purpose |
|---|---|
| `NEXT_PUBLIC_MODAL_PREDICT_URL` | Modal endpoint URL for `/predict`. The browser POSTs `multipart/form-data` directly (CORS-allowed by the Modal handler). Until the Modal endpoint is deployed (`../inference_api/`), the Live tab surfaces a clean error toast. |

## Deploy

Two independent surfaces:

### 1. Modal — Python inference

```bash
cd ../inference_api
uv venv && source .venv/bin/activate
uv pip install modal
modal token new                  # one-time browser auth
modal deploy modal_app.py        # prints the predict + healthz URLs
```

Paste the `/predict` URL into `ui/.env.local` (local) and into the Vercel
project's env vars (production).

Redeploy whenever `atlas/` or `artifacts/` change:

```bash
modal deploy modal_app.py        # image rebuild ~1–2 min; layer-cached afterwards
```

### 2. Vercel — Next.js frontend

```bash
cd ui
pnpm dlx vercel link             # one-time, links project
pnpm dlx vercel env add NEXT_PUBLIC_MODAL_PREDICT_URL production
pnpm dlx vercel --prod
```

The two deploys are independent — Vercel hits Modal over CORS, no shared
infra, no shared timeouts. Vercel's 10 s edge-function timeout does **not**
apply because the browser calls Modal directly.

## Stack

- **Next.js 16.2** (App Router, Turbopack)
- **React 19.2**
- **Tailwind CSS v4** — design tokens live in `app/globals.css` `@theme`
  block. A v3-style `tailwind.config.ts` is retained at the root for
  reference only.
- **shadcn/ui** (`base-nova` style, `slate` base color) — Button, Card,
  Select, Dialog, Tooltip, HoverCard, Switch, Sonner (replaces deprecated
  Toast), Badge, Separator.
- **Framer Motion** — page transitions + active-tab underline (`layoutId`).
- **react-plotly.js** — wrapped in `components/plots/PlotlyChart.tsx` with
  `dynamic(..., { ssr: false })` so it doesn't break SSR.

### Fonts

Loaded via CDN (not `next/font`) in `app/layout.tsx`:

- **General Sans** (Fontshare, variable, weights 200–700) — display + body
- **JetBrains Mono** (Google Fonts, weights 400/500/600) — stats + code

## File layout

```
ui/
├── app/                  # App Router
│   ├── layout.tsx        # TopBar + TabNav + PageTransition shell
│   ├── page.tsx          # /  → /inventory
│   └── (tabs)/<slug>/page.tsx  ×7
├── components/
│   ├── layout/           # TopBar, TabNav, KpiStrip, PageTransition
│   ├── plots/            # PlotlyChart + per-tab plot components
│   ├── tabs/             # one composed component per tab
│   └── ui/               # shadcn primitives
├── lib/                  # types, data, modal-client, plotly-theme, cn
├── public/
│   ├── data/             # static JSON sidecars (built by scripts/)
│   │   ├── inventory.json
│   │   ├── bands.json
│   │   ├── feature_catalog.json
│   │   ├── mcr_components.json
│   │   ├── stage15f.json
│   │   ├── confusion.json
│   │   ├── bootstrap.json
│   │   ├── mcnemar.json
│   │   └── spectra/<file_id>.json  ×87  (gitignored)
│   └── fig/              # FINAL/images PNGs for static fallbacks
├── scripts/
│   ├── build_inventory.py    │ build_bands.py
│   ├── build_spectra.py      │ build_features.py
│   ├── build_mcr.py          │ build_results.py
│   └── build_sidecars.py     # legacy stub; not used
└── styles/tokens.css     # NomadX design-token CSS variables
```

See `../plan/ui/ULTRAPLAN.md` §2 for the canonical layout.

## Data caveats — known issues surfaced during the build

These are findings the tab agents flagged during W2–W8. None affects
the deployed model, but each is worth understanding before reading the UI.

### 1. MCR-ALS K=7 (deployed) ≠ K=8 (paper headline)

The saved `artifacts/stage15f_mcr_global.joblib` is **K=7** — that's what
the production inference path uses. The paper's `mcr_C6_mean` d=−1.23
headline comes from a separate **K=8** fit cached in
`data_cache/unmix_features.parquet`. Component ordering does *not*
transfer between the two fits.

The MCR-ALS tab labels each saved component with its own (K=7) Cohen's d.
Largest |d| in the deployed K=7 fit is `k=4` (saved as "C5") at d=−1.19.
The headline that **0 MCR features survived per-fold MI in Stage 15F**
(§6.7) is unchanged regardless of K, so the production classifier is
unaffected by this inconsistency.

### 2. `data_cache/qc_info.json` `retention` field is mis-computed

The per-file `retention` field is stored as `kept / n_input` (a global
denominator → ~0.0225 for every 200-pixel-capped file) rather than
`kept / n_file_pixels`. The Inventory tab's `build_inventory.py`
recomputes `qc_pass_rate` directly from `qc_mask` to get the correct
per-file 0.89–0.90 values.

Anything else downstream that trusts `qc_info.retention` is reading
garbage. Out of scope for this UI but worth chasing on a future pipeline
pass.

### 3. `loso_std_accuracy` is NaN

`artifacts/stage15f_metadata.json` has `n_seeds: 1` and consequently
`loso_std_accuracy: NaN`. The Results tab surfaces the 5000-iter
file-bootstrap CI [0.345, 0.552] as the canonical uncertainty band
instead of the (undefined) seed variance.

## Roadmap reference

W1 (Next.js foundation, complete) + W9 (Modal endpoint, complete) →
W2–W8 (seven tabs, complete) → W10 (this integration). See
`../plan/ui/ULTRAPLAN.md` §4.

## License

Project-internal. See repository root.
