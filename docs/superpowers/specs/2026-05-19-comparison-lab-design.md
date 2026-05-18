# Comparison Lab — Design Spec

**Date:** 2026-05-19
**Owner:** Atlas UI (Next.js)
**Status:** Approved by user (2026-05-19); ready for implementation plan.
**Companion to:** `plan/ui/ULTRAPLAN.md`

---

## 1. Why

Today the Atlas UI shows one spectrum at a time (`SpectrumExplorer`) or class-average overlays. Scientists comparing samples — especially when triaging an unknown bacterium against positive and negative controls — need to stage many spectra simultaneously, line them up on a shared axis, and quickly switch between overlay, side-by-side, and difference views.

The user's handwritten brief (dated 19/05/2026) calls for:

- A new tab that can hold **4–12+ experimental graphs**, with chip-style role lanes:
  - **+ve controls** (e.g., *E. coli* K12 / MG1655)
  - **Blank** (water)
  - **Test samples** (the unknown bacterium plus any extras)
- "**Align all graphs and pick differences**" — linked axes + a diff-vs-reference mode.
- "**Overlays AND comparisons** for every kind of graph" with the individuality preserved.
- "**Scale on scroll**" — interpreted in two complementary ways:
  1. Plotly scroll-wheel zooms the y-axis on the focused panel (one-line config).
  2. Framer Motion focus-on-scroll: the row nearest viewport center scales to 1.0; rows further out scale to 0.85 and dim to 60% opacity.

## 2. Scope

### In scope
- New tab: `/compare` (the 9th tab; existing 8 are unchanged).
- Picker UI with three role lanes (+ve / blank / test).
- Five view modes over the same staged set: Grid, Overlay, Waterfall, Heatmap, Diff-vs-reference.
- Linked interactions: synchronized x-zoom, shared crosshair, click-to-highlight named bands.
- Normalization toggle (SNV / Min-Max / Raw / Mean-centered) applied uniformly to staged spectra.
- Region presets (Full / Fingerprint 800–1800 / LPS 400–900 / LPS 800–1200).
- Plotly y-axis scroll-zoom on every panel.
- Framer Motion focus-on-scroll scaling for the Grid view.

### Out of scope (deferred to v2)
- Free-form drag-and-drop canvas positioning.
- URL-shareable comparison sets.
- Re-running preprocessing on staged spectra (preprocessing is locked upstream — reproducibility-first per `ULTRAPLAN.md`).
- Uploading new files inside the Comparison Lab (live upload remains in the Live tab).
- Saving/loading named comparison sets to local storage.

## 3. User flow

1. User clicks the **Compare** tab.
2. Lab opens with an empty staging area and a `+ Add spectrum` button.
3. User adds spectra via a file picker that reads from `inventory.json`. Each added spectrum is dropped into one of the three role lanes (default: **Test**). User can re-assign by dragging a chip between lanes.
4. Up to 12 spectra by default (soft cap; configurable). Above 12, the picker greys out further adds and suggests switching to Heatmap view.
5. As soon as ≥1 spectrum is staged, the default **Grid** view renders small multiples with the staged set.
6. User flips between Grid / Overlay / Waterfall / Heatmap / Diff using the view toolbar.
7. Hovering any panel fires a synchronized crosshair across all panels (Grid + Waterfall + Diff) or all traces (Overlay), with a tooltip showing the wavenumber, the per-spectrum intensity, and the nearest named band from `bands.json`.
8. Zooming the x-axis on any panel zooms every other panel to match (toggleable via the **LINK: x-zoom** checkbox).
9. Scroll-wheel on a panel zooms its y-axis; Cmd/Ctrl + scroll-wheel zooms the x-axis (Plotly defaults).
10. As the user scrolls the Comparison Lab page in Grid view, rows nearest the viewport center are full-size; rows further away dim and shrink slightly.

## 4. Architecture

### 4.1 File layout

```
ui/app/(tabs)/compare/page.tsx            ← thin route wrapper
ui/components/tabs/ComparisonLab.tsx      ← top-level shell (state + toolbar + view switch)
ui/components/tabs/ComparisonPicker.tsx   ← role-lane chip picker
ui/components/plots/ComparisonGrid.tsx    ← small multiples (default)
ui/components/plots/ComparisonOverlay.tsx ← single-axis overlay
ui/components/plots/ComparisonWaterfall.tsx ← vertical-offset stack
ui/components/plots/ComparisonHeatmap.tsx ← rows=samples, color=intensity
ui/components/plots/ComparisonDiff.tsx    ← subtraction vs reference
ui/lib/use-linked-zoom.ts                 ← React hook syncing Plotly xaxis.range across refs
ui/lib/use-scroll-focus.ts                ← Framer Motion + IntersectionObserver hook
ui/lib/normalize.ts                       ← SNV / MinMax / mean-center helpers
ui/lib/types.ts                           ← + new types listed below
```

### 4.2 Type additions (`ui/lib/types.ts`)

```ts
export type ComparisonRole = "control_pos" | "blank" | "test";

export interface StagedSpectrum {
  file_id: string;          // resolves to /data/spectra/<file_id>.json
  role: ComparisonRole;
  display_label: string;    // editable, defaults to file_id
  visible: boolean;         // for Overlay view trace toggling
  color_override?: string;  // optional, falls back to class color
}

export type ComparisonView =
  | "grid"
  | "overlay"
  | "waterfall"
  | "heatmap"
  | "diff";

export type NormalizationMode = "snv" | "minmax" | "raw" | "mean_center";

export type RegionPreset =
  | "full"
  | "fingerprint_800_1800"
  | "lps_400_900"
  | "lps_800_1200";

export interface ComparisonState {
  staged: StagedSpectrum[];
  view: ComparisonView;
  normalization: NormalizationMode;
  region: RegionPreset;
  link_x_zoom: boolean;
  link_crosshair: boolean;
  share_y_scale: boolean;
  reference_file_id?: string;  // for Diff view; defaults to first staged blank
}
```

State lives in `ComparisonLab.tsx` via `useState` — no global store needed.

### 4.3 Data flow

```
inventory.json (already shipped)
  └─→ ComparisonPicker reads file_id list, grouped by primary_class

/data/spectra/<file_id>.json (already shipped — one sidecar per file)
  └─→ lazy-fetched on stage; cached via React Query-style swr pattern using
       `cache: "force-cache"` on fetch (matches existing data.ts convention).

bands.json (already shipped)
  └─→ used for crosshair tooltip "nearest named band" lookup and for
       click-to-highlight band regions across panels.
```

**No new sidecars.** No new Python build step. The data layer is purely composition over what `scripts/build_sidecars.py` already emits.

### 4.4 The five views

Each view is a separate component, receives `StagedSpectrum[]` + the relevant toolbar state as props, and renders one or more Plotly figures. All five panels live under one parent so the `useLinkedZoom` hook can collect refs and broadcast `relayout` events.

| View | Plotly figure shape |
|---|---|
| Grid | N subplots in a CSS grid (2-up, 3-up, 4-up based on count). Each subplot is its own `<PlotlyChart>` instance. |
| Overlay | One figure with N traces, color-coded by class+role, legend on right. |
| Waterfall | One figure with N traces, each offset on y by `i * vertical_gap`. Annotations label each row. |
| Heatmap | One `heatmapgl` trace: z[i][j] = spectrum i, wavenumber j; y-axis = row labels. |
| Diff | N−1 subplots (excluding the reference), each plotting `spectrum_i − reference`. Horizontal zero-line. |

### 4.5 Linked-zoom hook

```ts
// ui/lib/use-linked-zoom.ts (sketch)
export function useLinkedZoom(enabled: boolean) {
  const figs = useRef<Map<string, Plotly.PlotlyHTMLElement>>(new Map());
  const register = (key: string, el: Plotly.PlotlyHTMLElement | null) => { … };

  const onRelayout = (key: string, ev: Plotly.PlotRelayoutEvent) => {
    if (!enabled) return;
    const range = ev["xaxis.range"] ?? [ev["xaxis.range[0]"], ev["xaxis.range[1]"]];
    if (!range) return;
    for (const [k, el] of figs.current) {
      if (k === key) continue;
      Plotly.relayout(el, { "xaxis.range": range });
    }
  };
  return { register, onRelayout };
}
```

Each panel calls `register(file_id, plotlyDivRef.current)` on mount and passes `onRelayout(file_id, ev)` to the Plotly `onRelayout` prop.

### 4.6 Scroll-wheel y-zoom

Plotly `config={{ scrollZoom: true }}` plus `layout.yaxis.fixedrange = false`. Default Plotly behavior already zooms whichever axis the cursor is over; with no x-axis fixedrange override, this gives both axes scroll-zoom (Cmd/Ctrl modifier picks one in some browsers — we accept Plotly's default UX).

### 4.7 Focus-on-scroll (Framer Motion)

`useScrollFocus` returns a per-row `scale` (0.85–1.0) and `opacity` (0.6–1.0) driven by IntersectionObserver. Active only in **Grid** view; other views are single-figure and don't need it.

```tsx
const { scale, opacity } = useScrollFocus(rowRef);
<motion.div ref={rowRef} style={{ scale, opacity }} transition={{ duration: 0.18 }}>
  <ComparisonGridRow … />
</motion.div>
```

### 4.8 Picker UX

- Modal opens on `+ Add spectrum`.
- Top of modal: a search box filtering `inventory.json` by `file_id` or `subclass`.
- Below: a list grouped by `primary_class` (STEC / Non-STEC / Salmonella / H2O), with a small badge showing `n_pixels` and `qc_pass_rate`.
- Click a row to stage it; a role dropdown lets the user assign it to a lane before commit.
- Already-staged file_ids are greyed out.
- Soft cap of 12; over-cap shows an inline hint: "Switch to Heatmap view for >12 spectra."

## 5. Interaction matrix

| Interaction | Grid | Overlay | Waterfall | Heatmap | Diff |
|---|---|---|---|---|---|
| Hover crosshair | Synced across all panels | Single-figure crosshair | Synced across rows | Single-figure crosshair (snapped to row) | Synced across all panels |
| x-zoom link | ✓ | n/a (single fig) | ✓ | ✓ | ✓ |
| y-zoom scroll | ✓ per panel | ✓ | ✓ | n/a (categorical y) | ✓ per panel |
| Band click → highlight | ✓ all panels | ✓ vertical band | ✓ all rows | ✓ vertical band | ✓ all panels |
| Normalization | ✓ | ✓ | ✓ | ✓ | applied before subtraction |
| Region preset | ✓ | ✓ | ✓ | ✓ | ✓ |
| Focus-on-scroll | ✓ | n/a | n/a | n/a | n/a |

## 6. Visual treatment

- Inherits NomadX tokens from `ui/styles/tokens.css` (cyan accent on hover/active, deep navy panels, white traces over near-black background).
- Class colors fixed via `--class-stec / --class-nonstec / --class-salm / --class-h2o` from existing CSS variables.
- Role mapping (in addition to class color):
  - `control_pos` → solid line, full opacity, role badge "+"
  - `blank` → dashed line, 70% opacity, role badge "∅"
  - `test` → solid line, full opacity, role badge "?"
- Grid view uses `auto-fit, minmax(320px, 1fr)` so the column count adapts: 1–2 spectra side-by-side; 3–4 in a row; 8–12 wrap to 3-up or 4-up.

## 7. Error handling

- If a staged `file_id`'s spectrum sidecar fails to load → the panel renders with an inline error and a Retry button; other panels are unaffected.
- If the user removes the spectrum currently set as the Diff reference → the lab falls back to the next staged blank, or (if no blank) prompts the user to pick a reference before showing the Diff view.
- If no spectra are staged → empty state with a single call to action: "Stage at least one spectrum to begin."

## 8. Testing

Smoke tests (manual, since the existing UI is light on automated tests):
- Stage 1, 4, 8, 12 spectra in turn; verify Grid layout reflows correctly.
- Verify linked x-zoom works across all five views where applicable.
- Verify focus-on-scroll only activates in Grid.
- Verify Diff view falls back to "no reference" prompt when the reference is removed.
- Verify normalization toggle re-renders all panels.
- Verify staging more than 12 shows the heatmap hint.

If we have time during implementation, add a Playwright check that loads `/compare`, stages 4 file_ids, switches through all five views, and asserts no console errors.

## 9. Open questions

None blocking. The user has explicitly approved this design.

## 10. Decisions log (chronological)

- **2026-05-19** — User requested multi-graph comparison with role lanes (+ve / blank / test), align-and-pick-different behavior, and "scale on scroll".
- **2026-05-19** — Approved: five-view design, linked x-zoom + crosshair, Plotly scroll-wheel y-zoom, Framer Motion focus-on-scroll in Grid view.
- **2026-05-19** — Rejected during brainstorm: free-positioning Miro-style canvas (too much UX overhead for the actual comparison task); per-panel re-preprocessing (breaks reproducibility-first).
