# Comparison Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a 9th `/compare` tab in the Atlas UI that stages 4–12+ spectra into role lanes (+ve / blank / test) and exposes five linked, normalizable views (Grid / Overlay / Waterfall / Heatmap / Diff).

**Architecture:** Next.js 16 App Router. New route `app/(tabs)/compare/page.tsx`. Top-level `ComparisonLab` shell manages state (`useState`); the picker, toolbar, and view components are dumb children. Five view components each render Plotly figures via the existing `PlotlyChart` wrapper. A `useLinkedZoom` hook syncs `xaxis.range` across Plotly instances via `Plotly.relayout`; a `useScrollFocus` hook drives Framer Motion `scale`/`opacity` from IntersectionObserver in Grid view only. No new Python sidecars — consumes existing `/data/inventory.json`, `/data/bands.json`, `/data/spectra/<file_id>.json`.

**Tech Stack:** Next.js 16, React 19, TypeScript, Tailwind v4, shadcn, `react-plotly.js`, Framer Motion, pnpm.

**Testing convention:** This UI codebase has no test runner (per `ui/package.json`). Each task's verification step is `pnpm lint && pnpm build` (TypeScript + Next build = real signal) plus a brief manual browser check on the final task. Do not introduce vitest/jest as part of this feature.

**Spec:** `docs/superpowers/specs/2026-05-19-comparison-lab-design.md`

---

## File Structure

**Creating (10):**
- `ui/app/(tabs)/compare/page.tsx` — route shell, server component, renders `<ComparisonLab />`.
- `ui/components/tabs/ComparisonLab.tsx` — top-level client component: state, toolbar, view switching, empty state.
- `ui/components/tabs/ComparisonPicker.tsx` — modal that reads `inventory.json`, lets user stage files into role lanes.
- `ui/components/plots/ComparisonGrid.tsx` — small multiples (default view).
- `ui/components/plots/ComparisonOverlay.tsx` — single-axis overlay.
- `ui/components/plots/ComparisonWaterfall.tsx` — vertical-offset stack.
- `ui/components/plots/ComparisonHeatmap.tsx` — rows = samples, color = intensity.
- `ui/components/plots/ComparisonDiff.tsx` — `spectrum_i − reference` panels.
- `ui/lib/use-linked-zoom.ts` — React hook syncing Plotly `xaxis.range`.
- `ui/lib/use-scroll-focus.ts` — Framer Motion + IntersectionObserver hook.
- `ui/lib/normalize.ts` — SNV / Min-Max / Mean-center pure helpers.

**Modifying (3):**
- `ui/lib/types.ts` — add `ComparisonRole`, `StagedSpectrum`, `ComparisonView`, `NormalizationMode`, `RegionPreset`.
- `ui/components/layout/Sidebar.tsx` — add `Compare` nav item.
- `ui/components/layout/TabNav.tsx` — add `/compare` (mobile/top-nav parity).

---

## Task 1: Add types to `ui/lib/types.ts`

**Files:**
- Modify: `ui/lib/types.ts` (append at end of file)

- [ ] **Step 1: Append the new types**

Open `ui/lib/types.ts` and add the following at the bottom of the file:

```ts
// ──────────────────────────────────────────────────────────────────
// Comparison Lab (see docs/superpowers/specs/2026-05-19-comparison-lab-design.md)
// ──────────────────────────────────────────────────────────────────

export type ComparisonRole = "control_pos" | "blank" | "test";

export interface StagedSpectrum {
  /** Resolves to `/data/spectra/<file_id>.json`. */
  file_id: string;
  role: ComparisonRole;
  /** Editable label shown in legends/badges. Defaults to file_id at stage time. */
  display_label: string;
  /** Toggle visibility per-trace in Overlay view. */
  visible: boolean;
  /** Hex color override; falls back to class color from plotly-theme. */
  color_override?: string;
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

/** Per-region wavenumber windows in cm^-1. */
export const REGION_RANGES: Record<RegionPreset, [number, number] | null> = {
  full: null,
  fingerprint_800_1800: [800, 1800],
  lps_400_900: [400, 900],
  lps_800_1200: [800, 1200],
};

/** Shape of the per-file spectrum sidecar at /data/spectra/<file_id>.json. */
export interface SpectrumSidecar {
  file_id: string;
  primary_class: ClassName;
  subclass: string | null;
  n_pixels: number;
  n_qc_pass: number;
  wn_raw: number[];
  wn_pp: number[];
  mean_raw: number[];
  mean_pp: number[];
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cd /Users/devashishthapliyal/Documents/NomadX/ui && pnpm lint && pnpm build
```

Expected: `pnpm build` finishes "✓ Compiled successfully" — no TS errors.

- [ ] **Step 3: Commit**

```bash
cd /Users/devashishthapliyal/Documents/NomadX
git add ui/lib/types.ts
git commit -m "Add Comparison Lab types (StagedSpectrum, ComparisonView, etc.)"
```

---

## Task 2: Normalization helpers

**Files:**
- Create: `ui/lib/normalize.ts`

- [ ] **Step 1: Write the helper file**

Create `ui/lib/normalize.ts`:

```ts
/**
 * Pure normalization helpers for the Comparison Lab.
 *
 * All ops take a 1-D intensity vector and return a new 1-D vector of the
 * same length. NaN-free input is assumed (spectra sidecars are QC-clean).
 */
import type { NormalizationMode } from "./types";

export function meanCenter(y: number[]): number[] {
  const mean = y.reduce((s, v) => s + v, 0) / y.length;
  return y.map((v) => v - mean);
}

export function snv(y: number[]): number[] {
  const mean = y.reduce((s, v) => s + v, 0) / y.length;
  const variance =
    y.reduce((s, v) => s + (v - mean) * (v - mean), 0) / y.length;
  const std = Math.sqrt(variance);
  if (std === 0) return y.map(() => 0);
  return y.map((v) => (v - mean) / std);
}

export function minMax(y: number[]): number[] {
  let min = Infinity;
  let max = -Infinity;
  for (const v of y) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min;
  if (range === 0) return y.map(() => 0);
  return y.map((v) => (v - min) / range);
}

export function applyNormalization(
  y: number[],
  mode: NormalizationMode,
): number[] {
  switch (mode) {
    case "snv":
      return snv(y);
    case "minmax":
      return minMax(y);
    case "mean_center":
      return meanCenter(y);
    case "raw":
      return y;
  }
}
```

- [ ] **Step 2: Verify**

```bash
cd /Users/devashishthapliyal/Documents/NomadX/ui && pnpm lint && pnpm build
```

Expected: ✓ Compiled successfully.

- [ ] **Step 3: Commit**

```bash
cd /Users/devashishthapliyal/Documents/NomadX
git add ui/lib/normalize.ts
git commit -m "Add SNV / Min-Max / mean-center normalization helpers"
```

---

## Task 3: `useLinkedZoom` hook

**Files:**
- Create: `ui/lib/use-linked-zoom.ts`

- [ ] **Step 1: Write the hook**

Create `ui/lib/use-linked-zoom.ts`:

```ts
"use client";

/**
 * Sync Plotly xaxis.range across multiple chart instances.
 *
 * Each chart registers itself with a stable key (e.g., file_id) by storing
 * the Plotly div ref. When any chart fires a relayout containing an x-axis
 * range change, the hook broadcasts that range to every other registered
 * chart via `Plotly.relayout`.
 *
 * Usage:
 *   const { register, onRelayout } = useLinkedZoom(linkEnabled);
 *   <Plot
 *     onInitialized={(_, gd) => register(fileId, gd)}
 *     onUpdate={(_, gd) => register(fileId, gd)}
 *     onRelayout={(ev) => onRelayout(fileId, ev)}
 *   />
 */
import { useCallback, useRef } from "react";
import type { PlotRelayoutEvent } from "plotly.js";

type PlotlyGlobal = {
  relayout: (gd: HTMLElement, update: Record<string, unknown>) => Promise<void>;
};

declare global {
  interface Window {
    Plotly?: PlotlyGlobal;
  }
}

export function useLinkedZoom(enabled: boolean) {
  const figs = useRef<Map<string, HTMLElement>>(new Map());
  const broadcasting = useRef(false);

  const register = useCallback((key: string, el: HTMLElement | null) => {
    if (!el) {
      figs.current.delete(key);
      return;
    }
    figs.current.set(key, el);
  }, []);

  const onRelayout = useCallback(
    (key: string, ev: Readonly<PlotRelayoutEvent>) => {
      if (!enabled || broadcasting.current) return;

      const xRange = extractXRange(ev);
      if (!xRange) return;

      const Plotly = window.Plotly;
      if (!Plotly) return;

      broadcasting.current = true;
      for (const [k, el] of figs.current) {
        if (k === key) continue;
        void Plotly.relayout(el, { "xaxis.range": xRange });
      }
      // Release on next tick so cascading relayouts from peers are ignored.
      setTimeout(() => {
        broadcasting.current = false;
      }, 0);
    },
    [enabled],
  );

  return { register, onRelayout };
}

function extractXRange(
  ev: Readonly<PlotRelayoutEvent>,
): [number, number] | null {
  const pair = (ev as Record<string, unknown>)["xaxis.range"];
  if (Array.isArray(pair) && pair.length === 2) {
    return [Number(pair[0]), Number(pair[1])];
  }
  const lo = (ev as Record<string, unknown>)["xaxis.range[0]"];
  const hi = (ev as Record<string, unknown>)["xaxis.range[1]"];
  if (typeof lo === "number" && typeof hi === "number") {
    return [lo, hi];
  }
  return null;
}
```

- [ ] **Step 2: Verify**

```bash
cd /Users/devashishthapliyal/Documents/NomadX/ui && pnpm lint && pnpm build
```

Expected: ✓ Compiled successfully.

- [ ] **Step 3: Commit**

```bash
cd /Users/devashishthapliyal/Documents/NomadX
git add ui/lib/use-linked-zoom.ts
git commit -m "Add useLinkedZoom hook for synced Plotly x-axis ranges"
```

---

## Task 4: `useScrollFocus` hook

**Files:**
- Create: `ui/lib/use-scroll-focus.ts`

- [ ] **Step 1: Write the hook**

Create `ui/lib/use-scroll-focus.ts`:

```ts
"use client";

/**
 * Drive a Framer Motion-friendly `scale` (0.85 → 1.0) and `opacity`
 * (0.6 → 1.0) based on how close a row's center is to the viewport center.
 *
 * Used by ComparisonGrid rows to give a focus-on-scroll feel.
 */
import { useEffect, useRef, useState } from "react";

export interface ScrollFocusState {
  scale: number;
  opacity: number;
}

export function useScrollFocus<T extends HTMLElement>(): {
  ref: React.RefObject<T | null>;
  state: ScrollFocusState;
} {
  const ref = useRef<T>(null);
  const [state, setState] = useState<ScrollFocusState>({
    scale: 1,
    opacity: 1,
  });

  useEffect(() => {
    const el = ref.current;
    if (!el || typeof window === "undefined") return;

    let raf = 0;

    const update = () => {
      raf = 0;
      const rect = el.getBoundingClientRect();
      const rowCenter = rect.top + rect.height / 2;
      const viewportCenter = window.innerHeight / 2;
      // Distance from viewport center, normalized by half-viewport-height.
      const d = Math.min(
        1,
        Math.abs(rowCenter - viewportCenter) / (window.innerHeight / 2),
      );
      const scale = 1 - d * 0.15; // 1.0 → 0.85
      const opacity = 1 - d * 0.4; // 1.0 → 0.6
      setState({ scale, opacity });
    };

    const onScroll = () => {
      if (raf) return;
      raf = requestAnimationFrame(update);
    };

    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  return { ref, state };
}
```

- [ ] **Step 2: Verify**

```bash
cd /Users/devashishthapliyal/Documents/NomadX/ui && pnpm lint && pnpm build
```

Expected: ✓ Compiled successfully.

- [ ] **Step 3: Commit**

```bash
cd /Users/devashishthapliyal/Documents/NomadX
git add ui/lib/use-scroll-focus.ts
git commit -m "Add useScrollFocus hook for viewport-center-driven scale+opacity"
```

---

## Task 5: Route shell

**Files:**
- Create: `ui/app/(tabs)/compare/page.tsx`

- [ ] **Step 1: Write the route file**

Create `ui/app/(tabs)/compare/page.tsx`:

```tsx
import { ComparisonLab } from "@/components/tabs/ComparisonLab";

export const metadata = {
  title: "Compare · Atlas Raman",
  description:
    "Stage 4–12+ spectra, line up axes, pick differences — small multiples, overlay, waterfall, heatmap, and diff-vs-reference views.",
};

export default function ComparePage() {
  return (
    <div className="min-h-screen px-6 md:px-10 py-8 md:py-10">
      <ComparisonLab />
    </div>
  );
}
```

- [ ] **Step 2: Verify route resolves**

The shell now imports `ComparisonLab`, which doesn't exist yet — the next task creates it. Build will fail until Task 7. Skip the build verification at this step and move to Task 6 first.

- [ ] **Step 3: Commit**

```bash
cd /Users/devashishthapliyal/Documents/NomadX
git add ui/app/\(tabs\)/compare/page.tsx
git commit -m "Add /compare route shell"
```

---

## Task 6: Picker component

**Files:**
- Create: `ui/components/tabs/ComparisonPicker.tsx`

- [ ] **Step 1: Write the picker**

Create `ui/components/tabs/ComparisonPicker.tsx`:

```tsx
"use client";

/**
 * Modal picker for the Comparison Lab. Reads inventory.json, lets the user
 * stage a file into one of the three role lanes (+ve / blank / test).
 *
 * Soft cap of 12 staged spectra; over-cap, the modal disables further adds
 * and surfaces a hint to switch to Heatmap view.
 */
import { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X } from "lucide-react";
import { cn } from "@/lib/cn";
import { getInventory } from "@/lib/data";
import type { FileMeta } from "@/lib/types";
import type { ComparisonRole, StagedSpectrum } from "@/lib/types";

const SOFT_CAP = 12;

interface ComparisonPickerProps {
  open: boolean;
  onClose: () => void;
  staged: StagedSpectrum[];
  onStage: (next: StagedSpectrum) => void;
}

export function ComparisonPicker({
  open,
  onClose,
  staged,
  onStage,
}: ComparisonPickerProps) {
  const [files, setFiles] = useState<FileMeta[] | null>(null);
  const [query, setQuery] = useState("");
  const [defaultRole, setDefaultRole] = useState<ComparisonRole>("test");

  useEffect(() => {
    if (!open || files) return;
    getInventory()
      .then((inv) => setFiles(inv.files))
      .catch(() => setFiles([]));
  }, [open, files]);

  const stagedIds = useMemo(
    () => new Set(staged.map((s) => s.file_id)),
    [staged],
  );

  const filtered = useMemo(() => {
    if (!files) return [];
    const q = query.trim().toLowerCase();
    if (!q) return files;
    return files.filter(
      (f) =>
        f.file_id.toLowerCase().includes(q) ||
        (f.subclass ?? "").toLowerCase().includes(q) ||
        f.primary_class.toLowerCase().includes(q),
    );
  }, [files, query]);

  const grouped = useMemo(() => {
    const out = new Map<string, FileMeta[]>();
    for (const f of filtered) {
      const k = f.primary_class;
      const arr = out.get(k) ?? [];
      arr.push(f);
      out.set(k, arr);
    }
    return out;
  }, [filtered]);

  const atCap = staged.length >= SOFT_CAP;

  return (
    <AnimatePresence>
      {open ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.96, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.96, opacity: 0 }}
            transition={{ duration: 0.18 }}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-2xl max-h-[80vh] flex flex-col rounded-md border border-nx-muted bg-nx-bg-elev-1 text-nx-fg"
          >
            <header className="flex items-center justify-between px-5 py-3 border-b border-nx-muted">
              <div className="flex flex-col">
                <h2 className="font-display text-base">Add spectrum</h2>
                <p className="font-mono text-[0.65rem] text-nx-fg/55">
                  {staged.length}/{SOFT_CAP} staged
                </p>
              </div>
              <button
                aria-label="Close picker"
                onClick={onClose}
                className="text-nx-fg/55 hover:text-nx-fg transition-colors"
              >
                <X className="size-4" />
              </button>
            </header>

            <div className="px-5 py-3 flex items-center gap-3 border-b border-nx-muted/60">
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search file_id, subclass, class…"
                className="flex-1 bg-nx-bg px-3 py-2 rounded-sm text-sm border border-nx-muted/60 focus:border-nx-accent outline-none"
              />
              <select
                value={defaultRole}
                onChange={(e) =>
                  setDefaultRole(e.target.value as ComparisonRole)
                }
                className="bg-nx-bg px-2 py-2 rounded-sm text-sm border border-nx-muted/60"
              >
                <option value="test">Test</option>
                <option value="control_pos">+ve control</option>
                <option value="blank">Blank</option>
              </select>
            </div>

            <div className="flex-1 overflow-y-auto px-5 py-3">
              {atCap ? (
                <p className="text-amber-300 font-mono text-[0.7rem] mb-3">
                  Soft cap reached. Switch to Heatmap view for &gt;12 spectra,
                  or remove a staged file before adding more.
                </p>
              ) : null}
              {files === null ? (
                <p className="font-mono text-xs text-nx-fg/55">Loading…</p>
              ) : grouped.size === 0 ? (
                <p className="font-mono text-xs text-nx-fg/55">No matches.</p>
              ) : (
                Array.from(grouped.entries()).map(([cls, items]) => (
                  <section key={cls} className="mb-4">
                    <h3 className="font-mono text-[0.6rem] text-nx-fg/45 uppercase tracking-[0.18em] mb-1">
                      {cls} · {items.length}
                    </h3>
                    <ul className="flex flex-col gap-1">
                      {items.map((f) => {
                        const already = stagedIds.has(f.file_id);
                        const disabled = already || atCap;
                        return (
                          <li key={f.file_id}>
                            <button
                              disabled={disabled}
                              onClick={() =>
                                onStage({
                                  file_id: f.file_id,
                                  role: defaultRole,
                                  display_label: f.file_id,
                                  visible: true,
                                })
                              }
                              className={cn(
                                "w-full flex items-center justify-between px-3 py-2 text-left rounded-sm border border-transparent transition-colors",
                                disabled
                                  ? "opacity-40 cursor-not-allowed"
                                  : "hover:bg-nx-bg-elev-2/60 hover:border-nx-muted/60",
                              )}
                            >
                              <span className="flex flex-col">
                                <span className="font-mono text-xs">
                                  {f.file_id}
                                </span>
                                <span className="font-mono text-[0.6rem] text-nx-fg/45">
                                  {f.subclass ?? "—"} · {f.n_pixels} px · QC{" "}
                                  {Math.round(f.qc_pass_rate * 100)}%
                                </span>
                              </span>
                              {already ? (
                                <span className="font-mono text-[0.6rem] text-nx-accent">
                                  staged
                                </span>
                              ) : null}
                            </button>
                          </li>
                        );
                      })}
                    </ul>
                  </section>
                ))
              )}
            </div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
```

- [ ] **Step 2: Verify**

```bash
cd /Users/devashishthapliyal/Documents/NomadX/ui && pnpm lint && pnpm build
```

Expected: build still fails because `ComparisonLab` doesn't exist yet. The picker file itself should type-check — `pnpm lint` should pass. If there are TS errors *inside* `ComparisonPicker.tsx`, fix them before committing.

- [ ] **Step 3: Commit**

```bash
cd /Users/devashishthapliyal/Documents/NomadX
git add ui/components/tabs/ComparisonPicker.tsx
git commit -m "Add ComparisonPicker modal (inventory-backed role-lane picker)"
```

---

## Task 7: Lab shell + toolbar + empty state

**Files:**
- Create: `ui/components/tabs/ComparisonLab.tsx`

- [ ] **Step 1: Write the lab shell**

Create `ui/components/tabs/ComparisonLab.tsx`:

```tsx
"use client";

/**
 * Top-level shell for the Comparison Lab tab. Owns the staged-spectra state,
 * the view toolbar, and the role-lane chip strip. Renders one of five view
 * components depending on `view`.
 */
import { useState } from "react";
import { Plus, X } from "lucide-react";
import { cn } from "@/lib/cn";
import { ComparisonPicker } from "./ComparisonPicker";
import type {
  ComparisonRole,
  ComparisonView,
  NormalizationMode,
  RegionPreset,
  StagedSpectrum,
} from "@/lib/types";
import { ComparisonGrid } from "@/components/plots/ComparisonGrid";
import { ComparisonOverlay } from "@/components/plots/ComparisonOverlay";
import { ComparisonWaterfall } from "@/components/plots/ComparisonWaterfall";
import { ComparisonHeatmap } from "@/components/plots/ComparisonHeatmap";
import { ComparisonDiff } from "@/components/plots/ComparisonDiff";

const ROLE_LABELS: Record<ComparisonRole, string> = {
  control_pos: "+ve",
  blank: "blank",
  test: "test",
};

const VIEW_TABS: { key: ComparisonView; label: string }[] = [
  { key: "grid", label: "Grid" },
  { key: "overlay", label: "Overlay" },
  { key: "waterfall", label: "Waterfall" },
  { key: "heatmap", label: "Heatmap" },
  { key: "diff", label: "Diff" },
];

const NORM_TABS: { key: NormalizationMode; label: string }[] = [
  { key: "snv", label: "SNV" },
  { key: "minmax", label: "Min-Max" },
  { key: "raw", label: "Raw" },
  { key: "mean_center", label: "Mean-center" },
];

const REGION_TABS: { key: RegionPreset; label: string }[] = [
  { key: "full", label: "Full" },
  { key: "fingerprint_800_1800", label: "Fingerprint" },
  { key: "lps_400_900", label: "LPS 400–900" },
  { key: "lps_800_1200", label: "LPS 800–1200" },
];

export function ComparisonLab() {
  const [staged, setStaged] = useState<StagedSpectrum[]>([]);
  const [view, setView] = useState<ComparisonView>("grid");
  const [normalization, setNormalization] = useState<NormalizationMode>("snv");
  const [region, setRegion] = useState<RegionPreset>("full");
  const [linkXZoom, setLinkXZoom] = useState(true);
  const [shareYScale, setShareYScale] = useState(false);
  const [pickerOpen, setPickerOpen] = useState(false);

  const stage = (next: StagedSpectrum) => {
    setStaged((s) => [...s, next]);
    setPickerOpen(false);
  };
  const remove = (file_id: string) =>
    setStaged((s) => s.filter((x) => x.file_id !== file_id));
  const setRole = (file_id: string, role: ComparisonRole) =>
    setStaged((s) =>
      s.map((x) => (x.file_id === file_id ? { ...x, role } : x)),
    );
  const referenceFileId =
    staged.find((s) => s.role === "blank")?.file_id ?? staged[0]?.file_id;

  return (
    <div className="flex flex-col gap-6">
      <header className="flex items-end justify-between flex-wrap gap-4">
        <div>
          <h1 className="font-display text-3xl tracking-tight">
            Comparison Lab
          </h1>
          <p className="font-mono text-xs text-nx-fg/55 mt-1">
            Stage 4–12+ spectra · linked zoom + crosshair · five views
          </p>
        </div>
        <button
          onClick={() => setPickerOpen(true)}
          className="flex items-center gap-2 px-3 py-2 rounded-sm border border-nx-muted hover:border-nx-accent hover:text-nx-accent transition-colors text-sm"
        >
          <Plus className="size-4" /> Add spectrum
        </button>
      </header>

      {/* Staged chip strip */}
      {staged.length > 0 ? (
        <section className="rounded-md border border-nx-muted bg-nx-bg-elev-1/40 px-4 py-3">
          <p className="font-mono text-[0.6rem] text-nx-fg/45 uppercase tracking-[0.18em] mb-2">
            Staged · {staged.length}
          </p>
          <ul className="flex flex-wrap gap-2">
            {staged.map((s) => (
              <li
                key={s.file_id}
                className="flex items-center gap-2 px-2.5 py-1 rounded-sm border border-nx-muted/60 bg-nx-bg text-xs font-mono"
              >
                <select
                  value={s.role}
                  onChange={(e) =>
                    setRole(s.file_id, e.target.value as ComparisonRole)
                  }
                  className="bg-transparent text-nx-accent text-[0.65rem] outline-none"
                  aria-label={`Role for ${s.file_id}`}
                >
                  <option value="test">{ROLE_LABELS.test}</option>
                  <option value="control_pos">
                    {ROLE_LABELS.control_pos}
                  </option>
                  <option value="blank">{ROLE_LABELS.blank}</option>
                </select>
                <span>{s.display_label}</span>
                <button
                  aria-label={`Remove ${s.file_id}`}
                  onClick={() => remove(s.file_id)}
                  className="text-nx-fg/45 hover:text-nx-danger"
                >
                  <X className="size-3" />
                </button>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {/* Toolbar */}
      <section className="flex flex-wrap gap-4 items-center text-xs">
        <Toolbar label="View" tabs={VIEW_TABS} value={view} onChange={setView} />
        <Toolbar
          label="Norm"
          tabs={NORM_TABS}
          value={normalization}
          onChange={setNormalization}
        />
        <Toolbar
          label="Region"
          tabs={REGION_TABS}
          value={region}
          onChange={setRegion}
        />
        <label className="flex items-center gap-2 font-mono text-[0.7rem] text-nx-fg/70">
          <input
            type="checkbox"
            checked={linkXZoom}
            onChange={(e) => setLinkXZoom(e.target.checked)}
          />
          link x-zoom
        </label>
        <label className="flex items-center gap-2 font-mono text-[0.7rem] text-nx-fg/70">
          <input
            type="checkbox"
            checked={shareYScale}
            onChange={(e) => setShareYScale(e.target.checked)}
          />
          shared y
        </label>
      </section>

      {/* Body */}
      {staged.length === 0 ? (
        <div className="rounded-md border border-dashed border-nx-muted bg-nx-bg-elev-1/30 px-6 py-16 text-center">
          <p className="font-display text-lg text-nx-fg/80">
            Stage at least one spectrum to begin.
          </p>
          <p className="font-mono text-xs text-nx-fg/45 mt-2">
            Use <kbd className="px-1 py-0.5 rounded-sm bg-nx-bg-elev-2">Add spectrum</kbd> above.
          </p>
        </div>
      ) : (
        <ViewBody
          view={view}
          staged={staged}
          normalization={normalization}
          region={region}
          linkXZoom={linkXZoom}
          shareYScale={shareYScale}
          referenceFileId={referenceFileId}
        />
      )}

      <ComparisonPicker
        open={pickerOpen}
        onClose={() => setPickerOpen(false)}
        staged={staged}
        onStage={stage}
      />
    </div>
  );
}

function Toolbar<T extends string>({
  label,
  tabs,
  value,
  onChange,
}: {
  label: string;
  tabs: { key: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="font-mono text-[0.6rem] text-nx-fg/45 uppercase tracking-[0.18em]">
        {label}
      </span>
      <div className="flex rounded-sm border border-nx-muted overflow-hidden">
        {tabs.map((t) => (
          <button
            key={t.key}
            onClick={() => onChange(t.key)}
            className={cn(
              "px-2.5 py-1 font-mono text-[0.7rem] transition-colors",
              value === t.key
                ? "bg-nx-accent/15 text-nx-accent"
                : "text-nx-fg/65 hover:text-nx-fg",
            )}
          >
            {t.label}
          </button>
        ))}
      </div>
    </div>
  );
}

interface ViewBodyProps {
  view: ComparisonView;
  staged: StagedSpectrum[];
  normalization: NormalizationMode;
  region: RegionPreset;
  linkXZoom: boolean;
  shareYScale: boolean;
  referenceFileId?: string;
}

function ViewBody(props: ViewBodyProps) {
  switch (props.view) {
    case "grid":
      return <ComparisonGrid {...props} />;
    case "overlay":
      return <ComparisonOverlay {...props} />;
    case "waterfall":
      return <ComparisonWaterfall {...props} />;
    case "heatmap":
      return <ComparisonHeatmap {...props} />;
    case "diff":
      return <ComparisonDiff {...props} />;
  }
}
```

Note: `ComparisonGrid` and the four sibling view components don't exist yet — Tasks 8–12 create them. Build will fail until then. **For this task, write empty stub modules for the five view files so the import graph resolves**, then replace each stub in its own task:

```tsx
// ui/components/plots/ComparisonGrid.tsx  — STUB, replaced in Task 8
"use client";
import type { ViewProps } from "./view-props";
export function ComparisonGrid(_: ViewProps) {
  return <div className="font-mono text-xs text-nx-fg/45">Grid (TBD)</div>;
}
```

Also create the shared prop type at `ui/components/plots/view-props.ts`:

```ts
import type {
  NormalizationMode,
  RegionPreset,
  StagedSpectrum,
} from "@/lib/types";

export interface ViewProps {
  staged: StagedSpectrum[];
  normalization: NormalizationMode;
  region: RegionPreset;
  linkXZoom: boolean;
  shareYScale: boolean;
  referenceFileId?: string;
}
```

Update the imports in `ComparisonLab.tsx` accordingly — `ViewBody` becomes a thin pass-through (it already is).

Repeat the stub pattern for `ComparisonOverlay.tsx`, `ComparisonWaterfall.tsx`, `ComparisonHeatmap.tsx`, `ComparisonDiff.tsx`. Each stub returns `<div>Overlay (TBD)</div>`, etc. This keeps the build green while we land the real views one at a time.

- [ ] **Step 2: Verify**

```bash
cd /Users/devashishthapliyal/Documents/NomadX/ui && pnpm lint && pnpm build
```

Expected: ✓ Compiled successfully. The route `/compare` should render the empty-state, the picker should open and stage files (which then show stub view text).

- [ ] **Step 3: Commit**

```bash
cd /Users/devashishthapliyal/Documents/NomadX
git add ui/components/tabs/ComparisonLab.tsx \
        ui/components/plots/view-props.ts \
        ui/components/plots/ComparisonGrid.tsx \
        ui/components/plots/ComparisonOverlay.tsx \
        ui/components/plots/ComparisonWaterfall.tsx \
        ui/components/plots/ComparisonHeatmap.tsx \
        ui/components/plots/ComparisonDiff.tsx
git commit -m "Add ComparisonLab shell with view stubs"
```

---

## Task 8: Grid view (small multiples + scroll focus)

**Files:**
- Replace stub: `ui/components/plots/ComparisonGrid.tsx`

- [ ] **Step 1: Replace the stub with the real Grid view**

Overwrite `ui/components/plots/ComparisonGrid.tsx`:

```tsx
"use client";

/**
 * Grid view — small multiples. One Plotly subplot per staged spectrum,
 * shared x-axis range via useLinkedZoom; per-row scale+opacity via
 * useScrollFocus when scrolling.
 */
import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import type { Data, Layout } from "plotly.js";
import { PlotlyChart } from "./PlotlyChart";
import { nxColors, nxPlotlyLayout } from "@/lib/plotly-theme";
import { getSidecar } from "@/lib/data";
import { applyNormalization } from "@/lib/normalize";
import { useLinkedZoom } from "@/lib/use-linked-zoom";
import { useScrollFocus } from "@/lib/use-scroll-focus";
import { REGION_RANGES, type SpectrumSidecar } from "@/lib/types";
import type { ViewProps } from "./view-props";

const CLASS_COLOR: Record<string, string> = {
  STEC: nxColors.classStec,
  "Non-STEC": nxColors.classNonStec,
  Salmonella: nxColors.classSalm,
  H2O: nxColors.classH2o,
};

export function ComparisonGrid({
  staged,
  normalization,
  region,
  linkXZoom,
  shareYScale,
}: ViewProps) {
  const [byId, setById] = useState<Map<string, SpectrumSidecar>>(new Map());

  useEffect(() => {
    let cancelled = false;
    const toLoad = staged.filter((s) => !byId.has(s.file_id));
    if (toLoad.length === 0) return;
    Promise.all(
      toLoad.map((s) =>
        getSidecar<SpectrumSidecar>(`spectra/${s.file_id}.json`)
          .then((d) => [s.file_id, d] as const)
          .catch(() => [s.file_id, null] as const),
      ),
    ).then((results) => {
      if (cancelled) return;
      setById((prev) => {
        const next = new Map(prev);
        for (const [id, sidecar] of results) {
          if (sidecar) next.set(id, sidecar);
        }
        return next;
      });
    });
    return () => {
      cancelled = true;
    };
  }, [staged, byId]);

  const region_range = REGION_RANGES[region];
  const sharedYRange = useMemo(() => {
    if (!shareYScale) return undefined;
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of staged) {
      const sc = byId.get(s.file_id);
      if (!sc) continue;
      const y = applyNormalization(sc.mean_pp, normalization);
      for (const v of y) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    if (!isFinite(lo) || !isFinite(hi)) return undefined;
    const pad = (hi - lo) * 0.05;
    return [lo - pad, hi + pad] as [number, number];
  }, [staged, byId, normalization, shareYScale]);

  const { register, onRelayout } = useLinkedZoom(linkXZoom);

  return (
    <div className="grid gap-4 [grid-template-columns:repeat(auto-fit,minmax(320px,1fr))]">
      {staged.map((s) => {
        const sc = byId.get(s.file_id);
        return (
          <GridCell
            key={s.file_id}
            spectrum={sc}
            label={s.display_label}
            role={s.role}
            color={
              s.color_override ?? CLASS_COLOR[sc?.primary_class ?? ""] ?? nxColors.accent
            }
            normalization={normalization}
            xRange={region_range ?? undefined}
            yRange={sharedYRange}
            register={register}
            onRelayout={onRelayout}
          />
        );
      })}
    </div>
  );
}

interface GridCellProps {
  spectrum: SpectrumSidecar | undefined;
  label: string;
  role: string;
  color: string;
  normalization: import("@/lib/types").NormalizationMode;
  xRange?: [number, number];
  yRange?: [number, number];
  register: (key: string, el: HTMLElement | null) => void;
  onRelayout: (key: string, ev: Readonly<import("plotly.js").PlotRelayoutEvent>) => void;
}

function GridCell({
  spectrum,
  label,
  role,
  color,
  normalization,
  xRange,
  yRange,
  register,
  onRelayout,
}: GridCellProps) {
  const { ref, state } = useScrollFocus<HTMLDivElement>();

  if (!spectrum) {
    return (
      <div
        ref={ref}
        className="h-[280px] rounded-md border border-nx-muted bg-nx-bg-elev-1/30 flex items-center justify-center"
      >
        <span className="font-mono text-[0.7rem] text-nx-fg/45">Loading…</span>
      </div>
    );
  }

  const y = applyNormalization(spectrum.mean_pp, normalization);

  const data: Data[] = [
    {
      type: "scattergl",
      mode: "lines",
      x: spectrum.wn_pp,
      y,
      line: { color, width: 1.5 },
      hovertemplate: "<b>%{x:.1f} cm<sup>-1</sup></b><br>%{y:.4f}<extra></extra>",
    },
  ];

  const layout: Partial<Layout> = {
    ...nxPlotlyLayout,
    height: 240,
    margin: { l: 48, r: 16, t: 8, b: 36 },
    xaxis: {
      ...nxPlotlyLayout.xaxis,
      title: { text: "cm⁻¹", font: { size: 10 } },
      range: xRange,
      autorange: xRange ? false : true,
      fixedrange: false,
    },
    yaxis: {
      ...nxPlotlyLayout.yaxis,
      range: yRange,
      autorange: yRange ? false : true,
      fixedrange: false,
    },
    showlegend: false,
  };

  return (
    <motion.div
      ref={ref}
      style={{ scale: state.scale, opacity: state.opacity }}
      transition={{ duration: 0.18 }}
      className="rounded-md border border-nx-muted bg-nx-bg-elev-1/30 p-2 origin-center"
    >
      <header className="flex items-baseline justify-between px-1 pb-1">
        <span className="font-mono text-[0.7rem]">{label}</span>
        <span className="font-mono text-[0.55rem] text-nx-accent uppercase tracking-[0.18em]">
          {role.replace("_", " ")}
        </span>
      </header>
      <div className="h-[240px]">
        <PlotlyChart
          data={data}
          layout={layout}
          config={{ scrollZoom: true }}
          onInitialized={(_: unknown, gd: unknown) =>
            register(spectrum.file_id, gd as HTMLElement)
          }
          onUpdate={(_: unknown, gd: unknown) =>
            register(spectrum.file_id, gd as HTMLElement)
          }
          onRelayout={(ev: Readonly<import("plotly.js").PlotRelayoutEvent>) =>
            onRelayout(spectrum.file_id, ev)
          }
        />
      </div>
    </motion.div>
  );
}
```

**Note on the `PlotlyChart` wrapper:** the existing `PlotlyChart` component does not forward `onInitialized` / `onUpdate` / `onRelayout`. Extend its props to forward those handlers to the underlying `<Plot />`. Edit `ui/components/plots/PlotlyChart.tsx`:

Add to `PlotProps`:
```ts
onInitialized?: (figure: unknown, gd: HTMLElement) => void;
onUpdate?: (figure: unknown, gd: HTMLElement) => void;
onRelayout?: (event: Readonly<import("plotly.js").PlotRelayoutEvent>) => void;
```

And add the same three optional props to `PlotlyChartProps`, then forward them to `<Plot … onInitialized={…} onUpdate={…} onRelayout={…} />`.

- [ ] **Step 2: Verify**

```bash
cd /Users/devashishthapliyal/Documents/NomadX/ui && pnpm lint && pnpm build
```

Expected: ✓ Compiled successfully.

- [ ] **Step 3: Commit**

```bash
cd /Users/devashishthapliyal/Documents/NomadX
git add ui/components/plots/ComparisonGrid.tsx ui/components/plots/PlotlyChart.tsx
git commit -m "Add ComparisonGrid view (small multiples + linked zoom + scroll focus)"
```

---

## Task 9: Overlay view

**Files:**
- Replace stub: `ui/components/plots/ComparisonOverlay.tsx`

- [ ] **Step 1: Replace stub with overlay**

Overwrite `ui/components/plots/ComparisonOverlay.tsx`:

```tsx
"use client";

import { useEffect, useMemo, useState } from "react";
import type { Data, Layout } from "plotly.js";
import { PlotlyChart } from "./PlotlyChart";
import { nxColors, nxPlotlyLayout } from "@/lib/plotly-theme";
import { getSidecar } from "@/lib/data";
import { applyNormalization } from "@/lib/normalize";
import { REGION_RANGES, type SpectrumSidecar } from "@/lib/types";
import type { ViewProps } from "./view-props";

const CLASS_COLOR: Record<string, string> = {
  STEC: nxColors.classStec,
  "Non-STEC": nxColors.classNonStec,
  Salmonella: nxColors.classSalm,
  H2O: nxColors.classH2o,
};

const ROLE_DASH: Record<string, "solid" | "dash"> = {
  control_pos: "solid",
  test: "solid",
  blank: "dash",
};

export function ComparisonOverlay({
  staged,
  normalization,
  region,
}: ViewProps) {
  const [byId, setById] = useState<Map<string, SpectrumSidecar>>(new Map());

  useEffect(() => {
    let cancelled = false;
    const toLoad = staged.filter((s) => !byId.has(s.file_id));
    if (toLoad.length === 0) return;
    Promise.all(
      toLoad.map((s) =>
        getSidecar<SpectrumSidecar>(`spectra/${s.file_id}.json`)
          .then((d) => [s.file_id, d] as const)
          .catch(() => [s.file_id, null] as const),
      ),
    ).then((results) => {
      if (cancelled) return;
      setById((prev) => {
        const next = new Map(prev);
        for (const [id, sidecar] of results) {
          if (sidecar) next.set(id, sidecar);
        }
        return next;
      });
    });
    return () => {
      cancelled = true;
    };
  }, [staged, byId]);

  const region_range = REGION_RANGES[region];

  const data: Data[] = useMemo(() => {
    return staged
      .map((s) => {
        const sc = byId.get(s.file_id);
        if (!sc) return null;
        const y = applyNormalization(sc.mean_pp, normalization);
        return {
          type: "scattergl",
          mode: "lines",
          x: sc.wn_pp,
          y,
          name: `${s.display_label} (${s.role.replace("_", " ")})`,
          visible: s.visible !== false,
          line: {
            color:
              s.color_override ?? CLASS_COLOR[sc.primary_class] ?? nxColors.accent,
            width: 1.5,
            dash: ROLE_DASH[s.role] ?? "solid",
          },
          opacity: s.role === "blank" ? 0.7 : 1,
          hovertemplate:
            "<b>%{x:.1f} cm<sup>-1</sup></b><br>%{y:.4f}<br>%{fullData.name}<extra></extra>",
        } satisfies Data;
      })
      .filter(Boolean) as Data[];
  }, [staged, byId, normalization]);

  const layout: Partial<Layout> = {
    ...nxPlotlyLayout,
    height: 560,
    xaxis: {
      ...nxPlotlyLayout.xaxis,
      title: { text: "Wavenumber (cm⁻¹)" },
      range: region_range ?? undefined,
      autorange: region_range ? false : true,
      fixedrange: false,
    },
    yaxis: {
      ...nxPlotlyLayout.yaxis,
      title: { text: "Intensity (normalized)" },
      fixedrange: false,
    },
    hovermode: "x unified",
    showlegend: true,
  };

  return (
    <div className="rounded-md border border-nx-muted bg-nx-bg-elev-1/30 p-3">
      <div className="h-[560px]">
        <PlotlyChart data={data} layout={layout} config={{ scrollZoom: true }} />
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify**

```bash
cd /Users/devashishthapliyal/Documents/NomadX/ui && pnpm lint && pnpm build
```

Expected: ✓ Compiled successfully.

- [ ] **Step 3: Commit**

```bash
cd /Users/devashishthapliyal/Documents/NomadX
git add ui/components/plots/ComparisonOverlay.tsx
git commit -m "Add ComparisonOverlay view (single-axis multi-trace)"
```

---

## Task 10: Waterfall view

**Files:**
- Replace stub: `ui/components/plots/ComparisonWaterfall.tsx`

- [ ] **Step 1: Replace stub with waterfall**

Overwrite `ui/components/plots/ComparisonWaterfall.tsx`:

```tsx
"use client";

import { useEffect, useMemo, useState } from "react";
import type { Data, Layout } from "plotly.js";
import { PlotlyChart } from "./PlotlyChart";
import { nxColors, nxPlotlyLayout } from "@/lib/plotly-theme";
import { getSidecar } from "@/lib/data";
import { applyNormalization, minMax } from "@/lib/normalize";
import { REGION_RANGES, type SpectrumSidecar } from "@/lib/types";
import type { ViewProps } from "./view-props";

const CLASS_COLOR: Record<string, string> = {
  STEC: nxColors.classStec,
  "Non-STEC": nxColors.classNonStec,
  Salmonella: nxColors.classSalm,
  H2O: nxColors.classH2o,
};

export function ComparisonWaterfall({
  staged,
  normalization,
  region,
}: ViewProps) {
  const [byId, setById] = useState<Map<string, SpectrumSidecar>>(new Map());

  useEffect(() => {
    let cancelled = false;
    const toLoad = staged.filter((s) => !byId.has(s.file_id));
    if (toLoad.length === 0) return;
    Promise.all(
      toLoad.map((s) =>
        getSidecar<SpectrumSidecar>(`spectra/${s.file_id}.json`)
          .then((d) => [s.file_id, d] as const)
          .catch(() => [s.file_id, null] as const),
      ),
    ).then((results) => {
      if (cancelled) return;
      setById((prev) => {
        const next = new Map(prev);
        for (const [id, sidecar] of results) {
          if (sidecar) next.set(id, sidecar);
        }
        return next;
      });
    });
    return () => {
      cancelled = true;
    };
  }, [staged, byId]);

  const region_range = REGION_RANGES[region];

  const data: Data[] = useMemo(() => {
    // Use min-max scaling internally so vertical offsets are predictable,
    // then add `i * 1.1` so neighbouring traces don't overlap.
    return staged
      .map((s, i) => {
        const sc = byId.get(s.file_id);
        if (!sc) return null;
        const norm =
          normalization === "raw" ? minMax(sc.mean_pp) : applyNormalization(sc.mean_pp, normalization);
        const scaled = minMax(norm);
        const offset = i * 1.1;
        const y = scaled.map((v) => v + offset);
        return {
          type: "scattergl",
          mode: "lines",
          x: sc.wn_pp,
          y,
          name: s.display_label,
          line: {
            color:
              s.color_override ?? CLASS_COLOR[sc.primary_class] ?? nxColors.accent,
            width: 1.4,
          },
          hovertemplate: `<b>${s.display_label}</b><br>%{x:.1f} cm<sup>-1</sup><extra></extra>`,
        } satisfies Data;
      })
      .filter(Boolean) as Data[];
  }, [staged, byId, normalization]);

  const layout: Partial<Layout> = {
    ...nxPlotlyLayout,
    height: Math.max(360, staged.length * 90),
    xaxis: {
      ...nxPlotlyLayout.xaxis,
      title: { text: "Wavenumber (cm⁻¹)" },
      range: region_range ?? undefined,
      autorange: region_range ? false : true,
      fixedrange: false,
    },
    yaxis: {
      ...nxPlotlyLayout.yaxis,
      title: { text: "Intensity (stacked)" },
      tickmode: "array",
      tickvals: staged.map((_, i) => i * 1.1 + 0.5),
      ticktext: staged.map((s) => s.display_label),
      fixedrange: false,
    },
    showlegend: false,
  };

  return (
    <div className="rounded-md border border-nx-muted bg-nx-bg-elev-1/30 p-3">
      <div style={{ height: layout.height }}>
        <PlotlyChart data={data} layout={layout} config={{ scrollZoom: true }} />
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify**

```bash
cd /Users/devashishthapliyal/Documents/NomadX/ui && pnpm lint && pnpm build
```

Expected: ✓ Compiled successfully.

- [ ] **Step 3: Commit**

```bash
cd /Users/devashishthapliyal/Documents/NomadX
git add ui/components/plots/ComparisonWaterfall.tsx
git commit -m "Add ComparisonWaterfall view (vertical-offset stack)"
```

---

## Task 11: Heatmap view

**Files:**
- Replace stub: `ui/components/plots/ComparisonHeatmap.tsx`

- [ ] **Step 1: Replace stub with heatmap**

Overwrite `ui/components/plots/ComparisonHeatmap.tsx`:

```tsx
"use client";

import { useEffect, useMemo, useState } from "react";
import type { Data, Layout } from "plotly.js";
import { PlotlyChart } from "./PlotlyChart";
import { nxColors, nxPlotlyLayout } from "@/lib/plotly-theme";
import { getSidecar } from "@/lib/data";
import { applyNormalization } from "@/lib/normalize";
import { REGION_RANGES, type SpectrumSidecar } from "@/lib/types";
import type { ViewProps } from "./view-props";

export function ComparisonHeatmap({
  staged,
  normalization,
  region,
}: ViewProps) {
  const [byId, setById] = useState<Map<string, SpectrumSidecar>>(new Map());

  useEffect(() => {
    let cancelled = false;
    const toLoad = staged.filter((s) => !byId.has(s.file_id));
    if (toLoad.length === 0) return;
    Promise.all(
      toLoad.map((s) =>
        getSidecar<SpectrumSidecar>(`spectra/${s.file_id}.json`)
          .then((d) => [s.file_id, d] as const)
          .catch(() => [s.file_id, null] as const),
      ),
    ).then((results) => {
      if (cancelled) return;
      setById((prev) => {
        const next = new Map(prev);
        for (const [id, sidecar] of results) {
          if (sidecar) next.set(id, sidecar);
        }
        return next;
      });
    });
    return () => {
      cancelled = true;
    };
  }, [staged, byId]);

  const region_range = REGION_RANGES[region];

  const { data, layout } = useMemo(() => {
    const rows: number[][] = [];
    const labels: string[] = [];
    let wn: number[] = [];
    for (const s of staged) {
      const sc = byId.get(s.file_id);
      if (!sc) continue;
      const y = applyNormalization(sc.mean_pp, normalization);
      rows.push(y);
      labels.push(s.display_label);
      if (wn.length === 0) wn = sc.wn_pp;
    }
    const data: Data[] = [
      {
        type: "heatmap",
        z: rows,
        x: wn,
        y: labels,
        colorscale: "Viridis",
        colorbar: { title: { text: "Intensity" }, tickfont: { color: nxColors.fg } },
        hovertemplate:
          "<b>%{y}</b><br>%{x:.1f} cm<sup>-1</sup><br>%{z:.4f}<extra></extra>",
      },
    ];
    const layout: Partial<Layout> = {
      ...nxPlotlyLayout,
      height: Math.max(320, staged.length * 32 + 80),
      xaxis: {
        ...nxPlotlyLayout.xaxis,
        title: { text: "Wavenumber (cm⁻¹)" },
        range: region_range ?? undefined,
        autorange: region_range ? false : true,
        fixedrange: false,
      },
      yaxis: {
        ...nxPlotlyLayout.yaxis,
        autorange: "reversed",
        fixedrange: true,
      },
    };
    return { data, layout };
  }, [staged, byId, normalization, region_range]);

  return (
    <div className="rounded-md border border-nx-muted bg-nx-bg-elev-1/30 p-3">
      <div style={{ height: layout.height }}>
        <PlotlyChart data={data} layout={layout} config={{ scrollZoom: true }} />
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify**

```bash
cd /Users/devashishthapliyal/Documents/NomadX/ui && pnpm lint && pnpm build
```

Expected: ✓ Compiled successfully.

- [ ] **Step 3: Commit**

```bash
cd /Users/devashishthapliyal/Documents/NomadX
git add ui/components/plots/ComparisonHeatmap.tsx
git commit -m "Add ComparisonHeatmap view (rows=samples, color=intensity)"
```

---

## Task 12: Diff-vs-reference view

**Files:**
- Replace stub: `ui/components/plots/ComparisonDiff.tsx`

- [ ] **Step 1: Replace stub with diff view**

Overwrite `ui/components/plots/ComparisonDiff.tsx`:

```tsx
"use client";

import { useEffect, useMemo, useState } from "react";
import type { Data, Layout } from "plotly.js";
import { PlotlyChart } from "./PlotlyChart";
import { nxColors, nxPlotlyLayout } from "@/lib/plotly-theme";
import { getSidecar } from "@/lib/data";
import { applyNormalization } from "@/lib/normalize";
import { useLinkedZoom } from "@/lib/use-linked-zoom";
import { REGION_RANGES, type SpectrumSidecar } from "@/lib/types";
import type { ViewProps } from "./view-props";

const CLASS_COLOR: Record<string, string> = {
  STEC: nxColors.classStec,
  "Non-STEC": nxColors.classNonStec,
  Salmonella: nxColors.classSalm,
  H2O: nxColors.classH2o,
};

export function ComparisonDiff({
  staged,
  normalization,
  region,
  linkXZoom,
  referenceFileId,
}: ViewProps) {
  const [byId, setById] = useState<Map<string, SpectrumSidecar>>(new Map());

  useEffect(() => {
    let cancelled = false;
    const toLoad = staged.filter((s) => !byId.has(s.file_id));
    if (toLoad.length === 0) return;
    Promise.all(
      toLoad.map((s) =>
        getSidecar<SpectrumSidecar>(`spectra/${s.file_id}.json`)
          .then((d) => [s.file_id, d] as const)
          .catch(() => [s.file_id, null] as const),
      ),
    ).then((results) => {
      if (cancelled) return;
      setById((prev) => {
        const next = new Map(prev);
        for (const [id, sidecar] of results) {
          if (sidecar) next.set(id, sidecar);
        }
        return next;
      });
    });
    return () => {
      cancelled = true;
    };
  }, [staged, byId]);

  const { register, onRelayout } = useLinkedZoom(linkXZoom);

  const region_range = REGION_RANGES[region];
  const reference = referenceFileId ? byId.get(referenceFileId) : undefined;

  const refY = useMemo(() => {
    if (!reference) return null;
    return applyNormalization(reference.mean_pp, normalization);
  }, [reference, normalization]);

  if (!referenceFileId) {
    return (
      <div className="rounded-md border border-dashed border-nx-muted bg-nx-bg-elev-1/30 px-6 py-12 text-center">
        <p className="font-display text-base text-nx-fg/80">No reference set.</p>
        <p className="font-mono text-xs text-nx-fg/45 mt-2">
          Stage a Blank (water) — or any other spectrum — to use as the
          subtraction reference.
        </p>
      </div>
    );
  }
  if (!reference || !refY) {
    return (
      <div className="font-mono text-xs text-nx-fg/55">Loading reference…</div>
    );
  }

  const others = staged.filter((s) => s.file_id !== referenceFileId);

  return (
    <div className="grid gap-4 [grid-template-columns:repeat(auto-fit,minmax(360px,1fr))]">
      {others.map((s) => {
        const sc = byId.get(s.file_id);
        if (!sc) {
          return (
            <div
              key={s.file_id}
              className="h-[260px] rounded-md border border-nx-muted bg-nx-bg-elev-1/30 flex items-center justify-center"
            >
              <span className="font-mono text-[0.7rem] text-nx-fg/45">
                Loading…
              </span>
            </div>
          );
        }
        const y = applyNormalization(sc.mean_pp, normalization);
        const minLen = Math.min(y.length, refY.length);
        const diff: number[] = new Array(minLen);
        for (let i = 0; i < minLen; i++) diff[i] = y[i] - refY[i];
        const x = sc.wn_pp.slice(0, minLen);

        const data: Data[] = [
          {
            type: "scattergl",
            mode: "lines",
            x,
            y: diff,
            line: {
              color:
                s.color_override ?? CLASS_COLOR[sc.primary_class] ?? nxColors.accent,
              width: 1.5,
            },
            hovertemplate:
              "<b>%{x:.1f} cm<sup>-1</sup></b><br>Δ %{y:.4f}<extra></extra>",
          },
        ];

        const layout: Partial<Layout> = {
          ...nxPlotlyLayout,
          height: 240,
          margin: { l: 48, r: 16, t: 8, b: 36 },
          xaxis: {
            ...nxPlotlyLayout.xaxis,
            range: region_range ?? undefined,
            autorange: region_range ? false : true,
            fixedrange: false,
          },
          yaxis: {
            ...nxPlotlyLayout.yaxis,
            zeroline: true,
            zerolinecolor: nxColors.accentDeep,
            fixedrange: false,
          },
          showlegend: false,
        };

        return (
          <div
            key={s.file_id}
            className="rounded-md border border-nx-muted bg-nx-bg-elev-1/30 p-2"
          >
            <header className="flex items-baseline justify-between px-1 pb-1">
              <span className="font-mono text-[0.7rem]">
                {s.display_label} − {referenceFileId}
              </span>
              <span className="font-mono text-[0.55rem] text-nx-accent uppercase tracking-[0.18em]">
                Δ
              </span>
            </header>
            <div className="h-[240px]">
              <PlotlyChart
                data={data}
                layout={layout}
                config={{ scrollZoom: true }}
                onInitialized={(_: unknown, gd: unknown) =>
                  register(s.file_id, gd as HTMLElement)
                }
                onUpdate={(_: unknown, gd: unknown) =>
                  register(s.file_id, gd as HTMLElement)
                }
                onRelayout={(ev: Readonly<import("plotly.js").PlotRelayoutEvent>) =>
                  onRelayout(s.file_id, ev)
                }
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
```

- [ ] **Step 2: Verify**

```bash
cd /Users/devashishthapliyal/Documents/NomadX/ui && pnpm lint && pnpm build
```

Expected: ✓ Compiled successfully.

- [ ] **Step 3: Commit**

```bash
cd /Users/devashishthapliyal/Documents/NomadX
git add ui/components/plots/ComparisonDiff.tsx
git commit -m "Add ComparisonDiff view (subtraction vs reference)"
```

---

## Task 13: Wire into navigation + manual smoke test

**Files:**
- Modify: `ui/components/layout/Sidebar.tsx`
- Modify: `ui/components/layout/TabNav.tsx`

- [ ] **Step 1: Add Compare to the Sidebar**

In `ui/components/layout/Sidebar.tsx`, find the `NAV` array. Add a new entry between `Spectrum` and `Preprocessing`:

```ts
{ href: "/compare", label: "Compare", icon: Layers, hint: "Stage 4–12+ side-by-side" },
```

The `Layers` icon is already imported. If you'd prefer a different icon (e.g., `Columns` or `LayoutGrid`), import it from `lucide-react` and use it instead.

- [ ] **Step 2: Add Compare to TabNav**

Open `ui/components/layout/TabNav.tsx` and append `{ href: "/compare", label: "Compare" }` to its tabs array (mirroring the existing entries' shape).

- [ ] **Step 3: Verify build + lint**

```bash
cd /Users/devashishthapliyal/Documents/NomadX/ui && pnpm lint && pnpm build
```

Expected: ✓ Compiled successfully.

- [ ] **Step 4: Manual smoke test in the dev server**

```bash
cd /Users/devashishthapliyal/Documents/NomadX/ui && pnpm dev
```

Open `http://localhost:3000/compare` and verify the spec's §8 testing checklist:

- Empty state shows "Stage at least one spectrum to begin."
- Click **+ Add spectrum** → modal opens, lists files grouped by primary_class.
- Stage four files (mix of STEC / Non-STEC / Salmonella / H2O) — Grid view renders four small multiples in a 2x2 or 4-up depending on viewport.
- Scroll the page — rows nearest viewport center scale to 1.0; rows further away dim and shrink slightly (focus-on-scroll).
- Click **Overlay** — all four spectra render on a single axis with distinct class colors, legend on the right.
- Click **Waterfall** — vertical-offset stack, one trace per spectrum.
- Click **Heatmap** — 4-row heatmap, color = intensity, x = wavenumber.
- Click **Diff** — if no blank is staged, prompt appears; assign one staged file to role "blank" — Diff view renders three subtraction panels.
- Zoom the x-axis on any Grid panel — other panels zoom to match (linked x-zoom enabled by default).
- Stage 13 files — over-cap behavior: the picker disables further adds and shows the "switch to Heatmap" hint.
- Switch Region preset to "Fingerprint 800–1800" — all views clamp x-axis to that window.
- Switch Normalization between SNV / Min-Max / Raw / Mean-center — y-axis re-scales accordingly across all views.
- Open the browser devtools console — no React errors, no Plotly warnings.

If any step fails, fix it before continuing. Common issues:
- Plotly relayout broadcast loops → add a `broadcasting` guard (already present in `useLinkedZoom`).
- Race condition in sidecar fetching → ensure `byId` updates are functional (`setById(prev => ...)`) which the plan already does.

- [ ] **Step 5: Commit**

```bash
cd /Users/devashishthapliyal/Documents/NomadX
git add ui/components/layout/Sidebar.tsx ui/components/layout/TabNav.tsx
git commit -m "Wire Compare tab into Sidebar and TabNav"
```

---

## Self-review

**Spec coverage:**

| Spec requirement | Implemented in |
|---|---|
| New `/compare` tab | Tasks 5, 13 |
| Role-lane picker (+ve / blank / test) | Task 6 |
| Soft cap 12 | Task 6 |
| Grid / Overlay / Waterfall / Heatmap / Diff views | Tasks 8, 9, 10, 11, 12 |
| Normalization toggle (SNV / Min-Max / Raw / Mean-center) | Tasks 2, 7 (toolbar) |
| Region presets (Full / Fingerprint / LPS 400–900 / LPS 800–1200) | Tasks 1 (`REGION_RANGES`), 7 |
| Linked x-zoom | Tasks 3, 8, 12 |
| Plotly scroll-wheel y-zoom | Tasks 8–12 (`config.scrollZoom: true`) |
| Framer Motion focus-on-scroll in Grid only | Tasks 4, 8 |
| Reference fallback in Diff view | Task 12 |
| Empty state | Task 7 |
| Manual smoke matching spec §8 | Task 13 |
| Zero new Python sidecars | Confirmed — all views read existing `/data/inventory.json`, `/data/spectra/<file_id>.json` |

**Placeholder scan:** all code blocks are complete. No "TBD" outside of the deliberate stub modules in Task 7 (each replaced in its own follow-up task).

**Type consistency:** `StagedSpectrum`, `ComparisonRole`, `ComparisonView`, `NormalizationMode`, `RegionPreset`, `REGION_RANGES`, `SpectrumSidecar`, `ViewProps` — all defined in Task 1 + Task 7 and consistently used in Tasks 6–12.
