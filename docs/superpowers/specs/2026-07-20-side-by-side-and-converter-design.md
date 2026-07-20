# Side-by-side model comparison + in-browser XploRA converter

Date: 2026-07-20
Surface: `Initial/ui` (Next.js 16 App Router, Live tab)

## Problem

Two independent gaps on the Live tab.

**1. The comparison isn't a comparison.** `LiveInference` already fans out to both
Modal endpoints in parallel (`predict` → LogReg-L2, `predictPlsda` → PLS-DA) and
the hero copy claims "side-by-side", but the results render as two stacked
`ModelResultBlock`s. To compare LogReg's verdict against PLS-DA's you scroll past
a 35-row feature table. The most interesting case in this project — the mosaic
files where LogReg fails and PLS-DA rescues the call — is exactly the case the
current layout hides.

**2. Real instrument files are rejected.** Every file under `all-txt-data/` (392
files) is a raw XploRA export covering ~502–2699 cm⁻¹ across ~953 points. The
Atlas parser (`atlas/io.py`) hardcodes 2048 intensities per pixel row on
`linspace(76, 3499, 2048)`, so each pixel row fails the `len < 2048` guard and
`/predict` 500s. `Initial/convert_to_atlas.py` already solves this at the CLI,
but there is no path from a raw file to a prediction inside the UI.

## Non-goals

- Batch conversion or a queue. Single file at a time.
- Changing the models, the endpoints, or the preprocessing pipeline.
- Fixing the upstream `qc_info.retention` bug or the MCR K=7/K=8 mismatch.

## Part 1 — Paired-row comparison

Replace the two stacked `ModelResultBlock`s with a paired-row grid. Each row is
its own two-column grid holding the same panel type for both models, so
equivalent panels align horizontally and the eye scans across rather than down.

```
        ⚠ disagreement banner (full width, only when classes differ)
┌──────────────────────┬──────────────────────┐
│ LogReg-L2            │ PLS-DA               │  column headers
├──────────────────────┼──────────────────────┤
│ ██ verdict + prob    │ ██ verdict + prob    │
├──────────────────────┼──────────────────────┤
│ class probabilities  │ class probabilities  │
├──────────────────────┼──────────────────────┤
│ mean spectrum        │ mean spectrum        │
├──────────────────────┼──────────────────────┤
│ 35 feature values    │ spectral drivers     │  model-specific
└──────────────────────┴──────────────────────┘
```

Row-per-grid (rather than one grid with all cells) is deliberate: it keeps
alignment within a row, collapses cleanly to one column on narrow screens, and
lets a cell be absent without disturbing grid flow.

Each model's states are independent — one can still be loading or errored while
the other renders. Every cell therefore handles `loading | error | data` itself,
and each cell carries a model chip so a single-column mobile layout is never
ambiguous about which model a panel belongs to.

`ModelResultBlock` is decomposed into the per-panel pieces it currently inlines
(`VerdictCell`, `ProbabilitiesCell`, `SpectrumCell`, `DetailCell`). The existing
`ResultBanner`, `LiveProbabilityBars`, `LiveMeanSpectrum`,
`FeatureContributionTable` and `SpectralDriversPanel` are reused unchanged apart
from sizing, so the visual language stays put.

The verdict banner shrinks: at half width the current
`clamp(2.5rem, 5vw, 3.75rem)` class name overflows for "Salmonella", so the
paired variant drops to a smaller clamp and lets the probability sit beneath the
class rather than beside it.

## Part 2 — In-browser converter

### Where it runs

A TypeScript port of `convert_to_atlas.py` running in the browser
(`lib/atlas-convert.ts`). The transform is linear interpolation with edge
clamping — `np.interp`'s exact semantics — which reproduces bit-for-bit in JS.
No Modal redeploy, no server hop, no Vercel body-size ceiling, and the user sees
the coverage report before spending a cold start.

### What the corpus actually contains

Measured across all 389 `.txt` files before writing the converter — the numbers
drove two design changes:

| Property | Range observed |
|---|---|
| Header | 0 or 53 `#` lines (both layouts occur) |
| Native points | 552 – 959 |
| Native range | 350–1849 up to 502–2699 cm⁻¹ |
| Pixel rows | 9 – 109,071 |
| Source size | 14 KB – 441 MB |

Three distinct shapes: standard map with `#` header, headerless map, and 16
two-column `wavenumber <tab> intensity` files that are single averaged or
background spectra rather than maps. One file (`Info about chips.txt`) is prose
notes and is rejected. C-H coverage is **0% for every file in the corpus**.

### Contract

```ts
function convertToAtlas(file: File, opts?: ConvertOptions):
  Promise<{ file: File; stats: ConversionStats }>
```

Faithful to the Python: same `#` header passthrough, same `#Converted=` /
`#ConvertedBy=` provenance lines, same `\t\t` + 2048 wavenumber lead row, same
`%.3f` coordinates and `%.1f` intensities, same `(X, Y, spectrum)` column order,
same skip rules for short or non-finite rows.

Three browser-specific departures, each forced by the table above:

- **Encoding.** Source is latin-1; read via `TextDecoder("iso-8859-1")`. Output
  is written back as latin-1 bytes (codepoint & 0xFF, `?` above 255) so
  `atlas/io.py`'s latin-1 read round-trips.
- **Streaming.** A 441 MB source cannot be decoded whole in a tab. The file is
  read in 8 MB slices and scanned line by line; rows that won't be kept are
  never split into cells.
- **Pixel subsampling to 200.** Emitting every row would produce a 1.5 GB upload
  for the largest file — and both endpoints immediately discard all but a random
  200 rows (`PIXEL_CAP` in `modal_app.py`), as did training
  (`parse_file(pixel_cap=200)`). Rows are therefore reservoir-sampled to 200
  during the single streaming pass, using a seeded PRNG so the same file always
  yields the same subset. The chosen rows differ from the endpoint's own seed-42
  pick, but the distribution is identical. **Files at or under 200 rows keep
  every row and are byte-identical to `convert_to_atlas.py` output.**

Two-column spectrum files are converted as a single synthetic pixel at the
origin and labelled as such in both the output header and the UI.

### The out-of-distribution problem

This is the part that matters most. A file that *parses* is not a file that
*predicts*. These exports stop at 2699 cm⁻¹, so the C-H stretch window the model
crops to (2800–3050 cm⁻¹) is **100% absent** and gets edge-clamped flat. The
fingerprint window (400–1800) is also short at the low end. The model still
returns a confident-looking probability vector over four classes, and a
screenshot of that banner is indistinguishable from a real result.

So the UI states incompatibility in three places, escalating:

1. **On selection, before conversion** — the file is named as not directly
   ingestable by the model, with the reason (point count and range vs. the
   required 2048 bins over 76–3499).
2. **After conversion** — a per-band coverage readout giving measured vs.
   flat-filled percentages for both windows, not just a generic warning.
3. **Before analysis** — when either band is short, the Analyze button is gated
   behind an explicit acknowledgement that the results are not trustworthy.
   Ungated when coverage is complete.

The acknowledgement uses the existing `Switch` primitive; no new dependency.

### Flow

```
select .txt
  → decode + parse            (incompatibility stated)
  → resample to 2048 bins     (coverage report)
  → [acknowledge if short]
  → Analyze  ──► same runPrediction() the drop zone uses
                 └─► paired-row results from Part 1
```

The converted file is also downloadable, mirroring the CLI's
`<stem>__atlas2048.txt` artifact so the output can be inspected or reused.

`ConvertPanel` owns conversion and its own state; it receives `onAnalyze(file)`
from `LiveInference` and knows nothing about Modal, models, or results. Its only
coupling to the rest of the tab is that one callback.

## Files

| File | Change |
|---|---|
| `lib/atlas-convert.ts` | new — parser, resampler, latin-1 writer, coverage stats |
| `components/tabs/ConvertPanel.tsx` | new — upload area, report, gate, Analyze |
| `components/tabs/LiveInference.tsx` | paired-row results; mount `ConvertPanel`; expose `runPrediction` |

No changes to `lib/modal-client.ts`, `lib/types.ts`, the Modal app, or any
Python.

## Verification — results

- `tsc --noEmit` clean; ESLint clean; `next build` succeeds and `/live`
  prerenders.
- **Byte-parity with the Python.** A 6-row fixture converted by both
  implementations differs on exactly one line — the intentional
  `#ConvertedBy=` provenance string. All 2048 interpolated values per row and
  every `%.1f` / `%.3f` rendering match.
- **All four layouts handled**: `#`-header map (960 rows → 200 kept), headerless
  map (625 → 200), two-column spectrum (832 pts → 1 pixel), and the prose notes
  file, which raises a specific "unrecognised layout" error.
- **Scale.** The 441 MB / 108,500-row file converts to 2.96 MB in 703 ms.
- **Live endpoints accept the converted file.** Both return HTTP 200 with full
  payloads: LogReg 35 feature values, PLS-DA 987 driver bins.

The end-to-end run also confirmed why the gate is necessary. A converted
O157:H7 file — a STEC sample — came back **Non-STEC at p = 1.00** from LogReg
and Non-STEC at 0.87 from PLS-DA. Both models are confidently wrong, which is
the expected consequence of feeding them a spectrum whose C-H window is entirely
fabricated. Without the acknowledgement gate that screenshot would read as a
clean result.
