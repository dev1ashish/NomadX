"use client";

/**
 * PreprocessingPipeline (W12) — animated step-by-step visualization of the
 * 5-stage Atlas Raman preprocessing pipeline.
 *
 * Walks the user through:
 *   0. Raw 2048-bin spectrum
 *   1. Cosmic-ray removal (MAD median filter)
 *   2. arPLS baseline subtraction  ← shows the fitted baseline as a 2nd trace
 *   3. Savitzky-Golay smoothing
 *   4. Crop to fingerprint + C-H stretch (2048 → 987 bins)
 *   5. SNV normalization
 *
 * Backed by `/data/preprocessing.json`, which is emitted offline by
 * `scripts/build_preprocessing.py` (4 exemplars × 7 captured stage traces).
 *
 * UX
 * ---
 * - 4 exemplar class chips (STEC / Non-STEC / Salmonella / H2O) above the plot.
 * - 6-step horizontal stepper at the top with the active step highlighted in
 *   cyan and a Framer-Motion layout-id underline that slides between steps.
 * - Big Plotly trace below — switches `wn` axis between full 2048 and cropped
 *   987 once we cross step 4, and toggles a faint baseline overlay during
 *   step 2 (arPLS).
 * - Side panel: title + plain-English description + algorithm chip for the
 *   currently selected step.
 * - Stepper auto-advances every 2.5 s on first load (across all 6 steps), then
 *   stops on its own. Play/pause + back/next controls available below the plot.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ChevronLeft,
  ChevronRight,
  Pause,
  Play,
  RotateCcw,
} from "lucide-react";

import { PreprocessingSpectrum } from "@/components/plots/PreprocessingSpectrum";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { getSidecar } from "@/lib/data";
import { nxColors } from "@/lib/plotly-theme";
import { cn } from "@/lib/cn";
import type { ClassName } from "@/lib/types";

// ---------- sidecar contract ----------

type StageKey =
  | "raw"
  | "cosmic_ray"
  | "after_baseline"
  | "smoothed"
  | "cropped"
  | "snv";

/** ALL stage rows in the JSON — note `baseline` is sidecar-only, not a UI step. */
interface ExemplarStages {
  raw: number[];
  cosmic_ray: number[];
  baseline: number[];
  after_baseline: number[];
  smoothed: number[];
  cropped: number[];
  snv: number[];
}

interface Exemplar {
  file_id: string;
  primary_class: ClassName;
  subclass: string | null;
  wn_raw: number[];
  wn_cropped: number[];
  stages: ExemplarStages;
}

interface PipelineStepDef {
  key: StageKey;
  label: string;
  description: string;
  algorithm: string;
}

interface PreprocessingSidecar {
  exemplars: Exemplar[];
  pipeline: PipelineStepDef[];
}

// ---------- visual config ----------

const CLASS_ORDER: ClassName[] = ["STEC", "Non-STEC", "Salmonella", "H2O"];

const CLASS_COLOR: Record<ClassName, string> = {
  STEC: nxColors.classStec,
  "Non-STEC": nxColors.classNonStec,
  Salmonella: nxColors.classSalm,
  H2O: nxColors.classH2o,
};

const CLASS_STRIPE: Record<ClassName, string> = {
  STEC: "bg-class-stec",
  "Non-STEC": "bg-class-nonstec",
  Salmonella: "bg-class-salm",
  H2O: "bg-class-h2o",
};

const CLASS_RING: Record<ClassName, string> = {
  STEC: "ring-class-stec/60",
  "Non-STEC": "ring-class-nonstec/60",
  Salmonella: "ring-class-salm/60",
  H2O: "ring-class-h2o/60",
};

const CLASS_TEXT: Record<ClassName, string> = {
  STEC: "text-class-stec",
  "Non-STEC": "text-class-nonstec",
  Salmonella: "text-class-salm",
  H2O: "text-class-h2o",
};

/** Y-axis title varies by stage — the trace lives in different units. */
const Y_AXIS_LABEL: Record<StageKey, string> = {
  raw: "Intensity (raw counts)",
  cosmic_ray: "Intensity (raw counts)",
  after_baseline: "Intensity (baseline-subtracted)",
  smoothed: "Intensity (smoothed)",
  cropped: "Intensity (cropped)",
  snv: "Intensity (SNV z-score)",
};

const AUTO_ADVANCE_MS = 2500;

// ---------- component ----------

export function PreprocessingPipeline() {
  const [sidecar, setSidecar] = useState<PreprocessingSidecar | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [classIdx, setClassIdx] = useState<number>(0); // 0..3 (STEC first)
  const [stepIdx, setStepIdx] = useState<number>(0); // 0..5
  const [playing, setPlaying] = useState<boolean>(true);
  // Track whether we've already auto-cycled once so we don't loop forever.
  const completedAutoPlay = useRef<boolean>(false);

  useEffect(() => {
    let cancelled = false;
    getSidecar<PreprocessingSidecar>("preprocessing.json")
      .then((j) => {
        if (cancelled) return;
        setSidecar(j);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const pipeline = sidecar?.pipeline ?? [];
  const nSteps = pipeline.length;

  // Auto-advance loop — runs only when `playing` is true. Stops after the
  // first full traversal so we don't keep looping in the background.
  useEffect(() => {
    if (!playing || nSteps === 0) return;
    if (completedAutoPlay.current) return;
    const t = setTimeout(() => {
      setStepIdx((prev) => {
        const next = prev + 1;
        if (next >= nSteps) {
          completedAutoPlay.current = true;
          setPlaying(false);
          return prev; // hold on last step
        }
        return next;
      });
    }, AUTO_ADVANCE_MS);
    return () => clearTimeout(t);
  }, [playing, stepIdx, nSteps]);

  const handleStep = useCallback(
    (idx: number) => {
      // Manual interaction stops autoplay permanently.
      completedAutoPlay.current = true;
      setPlaying(false);
      setStepIdx(Math.max(0, Math.min(nSteps - 1, idx)));
    },
    [nSteps],
  );

  const handlePlayPause = useCallback(() => {
    if (playing) {
      setPlaying(false);
      return;
    }
    // Resume: if we'd already reached the end, restart from 0.
    if (stepIdx >= nSteps - 1) {
      setStepIdx(0);
    }
    completedAutoPlay.current = false;
    setPlaying(true);
  }, [playing, stepIdx, nSteps]);

  const handleRestart = useCallback(() => {
    setStepIdx(0);
    completedAutoPlay.current = false;
    setPlaying(true);
  }, []);

  // ---- error + loading ----
  if (error) {
    return (
      <section className="mx-auto max-w-screen-2xl px-8 lg:px-14 py-10">
        <p className="font-mono text-sm text-nx-danger">
          preprocessing.json load failed: {error}
        </p>
      </section>
    );
  }

  if (!sidecar) {
    return (
      <section className="mx-auto max-w-screen-2xl px-8 lg:px-14 py-10">
        <div className="h-12 w-72 animate-pulse rounded-md bg-nx-bg-elev-1" />
        <div className="mt-8 h-[480px] animate-pulse rounded-md bg-nx-bg-elev-1" />
      </section>
    );
  }

  // Resolve current exemplar + step.
  const exemplar =
    sidecar.exemplars[Math.min(classIdx, sidecar.exemplars.length - 1)];
  const step = pipeline[Math.min(stepIdx, pipeline.length - 1)];
  const isCroppedSpace = step.key === "cropped" || step.key === "snv";
  const isArPLSStep = step.key === "after_baseline";

  const wn = isCroppedSpace ? exemplar.wn_cropped : exemplar.wn_raw;
  const intensity = exemplar.stages[step.key];
  // Show the fitted baseline ONLY on the arPLS step; we overlay it on the
  // post-cosmic-ray trace by passing both to the spectrum plot.
  // For step `after_baseline`, the "natural" comparison is: the post-cosmic
  // trace AS the baseline-of-reference would have you see — but the most
  // useful UX is showing the OUTPUT (after-baseline) AND the baseline that
  // was subtracted overlaid in their original units. We anchor the plot to
  // the cosmic-ray-cleaned trace + baseline so the user literally sees the
  // baseline curve UNDER the data; the next step transitions to the
  // baseline-subtracted result so the user sees the visual proof.

  return (
    <section className="mx-auto max-w-screen-2xl px-8 lg:px-14 py-10">
      {/* Hero */}
      <div className="flex flex-col gap-2 mb-8">
        <span className="font-mono text-[0.65rem] uppercase tracking-[0.22em] text-nx-accent">
          ◆ Preprocessing pipeline
        </span>
        <h2 className="font-display text-[clamp(2rem,4vw,3.25rem)] leading-[1.05] text-nx-fg">
          Raw → 987-bin, in five steps
        </h2>
        <p className="text-nx-fg/55 max-w-3xl text-sm leading-relaxed">
          Every spectrum in the Atlas dataset gets the same five-stage
          treatment before it sees a model — cosmic-ray removal, fluorescence
          baseline subtraction, smoothing, fingerprint+C-H crop, and SNV
          normalization. Pick a class, then watch the trace morph through each
          stage. Step 2 overlays the arPLS-fitted baseline so you can see
          exactly what gets subtracted.
        </p>
      </div>

      {/* Exemplar chips */}
      <div className="mb-6 flex flex-wrap items-center gap-3">
        <span className="font-mono text-[0.6rem] uppercase tracking-[0.18em] text-nx-fg/45">
          Exemplar
        </span>
        {CLASS_ORDER.map((cls, idx) => {
          const ex = sidecar.exemplars.find((e) => e.primary_class === cls);
          if (!ex) return null;
          const active = idx === classIdx;
          return (
            <button
              key={cls}
              type="button"
              onClick={() => setClassIdx(idx)}
              className={cn(
                "group relative flex items-center gap-2 rounded-sm border px-3 py-1.5 text-left transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-nx-accent",
                active
                  ? cn(
                      "bg-nx-bg-elev-2/50 border-transparent ring-1",
                      CLASS_RING[cls],
                    )
                  : "border-nx-muted/40 bg-nx-bg-elev-1/40 hover:bg-nx-bg-elev-1/80",
              )}
            >
              <span
                aria-hidden
                className={cn("inline-block size-1.5 rounded-full", CLASS_STRIPE[cls])}
              />
              <span className="flex flex-col gap-0">
                <span
                  className={cn(
                    "font-mono text-[0.72rem] font-semibold tracking-wide",
                    active ? CLASS_TEXT[cls] : "text-nx-fg/85",
                  )}
                >
                  {cls}
                </span>
                <span className="font-mono text-[0.55rem] text-nx-fg/40 leading-tight">
                  {ex.subclass ?? "—"}
                </span>
              </span>
            </button>
          );
        })}
        <span className="ml-auto font-mono text-[0.6rem] text-nx-fg/40">
          {exemplar.file_id}
        </span>
      </div>

      {/* Stepper */}
      <div className="mb-4 overflow-x-auto">
        <ol className="flex min-w-full items-stretch gap-0">
          {pipeline.map((p, i) => {
            const active = i === stepIdx;
            const done = i < stepIdx;
            return (
              <li
                key={p.key}
                className="relative flex flex-1 min-w-[140px] flex-col"
              >
                <button
                  type="button"
                  onClick={() => handleStep(i)}
                  className={cn(
                    "group relative flex flex-col items-start gap-1 px-3 py-2.5 text-left transition-colors focus-visible:outline-none focus-visible:bg-nx-bg-elev-2/50",
                    active
                      ? "text-nx-fg"
                      : done
                        ? "text-nx-fg/75 hover:text-nx-fg"
                        : "text-nx-fg/45 hover:text-nx-fg/85",
                  )}
                >
                  <div className="flex items-center gap-2">
                    <span
                      className={cn(
                        "font-mono text-[0.6rem] rounded-full size-5 flex items-center justify-center tabular-nums",
                        active
                          ? "bg-nx-accent text-nx-bg"
                          : done
                            ? "bg-nx-accent-deep/60 text-nx-fg"
                            : "bg-nx-bg-elev-1 text-nx-fg/50 ring-1 ring-nx-muted/60",
                      )}
                    >
                      {i + 1}
                    </span>
                    <span
                      className={cn(
                        "font-display text-[0.78rem] leading-tight tracking-wide",
                        active && "text-nx-accent",
                      )}
                    >
                      {p.label}
                    </span>
                  </div>
                  {active ? (
                    <motion.span
                      layoutId="preprocessingStepUnderline"
                      className="absolute left-2 right-2 bottom-0 h-[2px] bg-nx-accent rounded-sm"
                      transition={{ type: "spring", stiffness: 380, damping: 32 }}
                    />
                  ) : (
                    <span aria-hidden className="absolute left-2 right-2 bottom-0 h-[2px] bg-nx-muted/30" />
                  )}
                </button>
              </li>
            );
          })}
        </ol>
      </div>

      {/* Plot + side panel */}
      <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
        {/* Plot pane */}
        <motion.div
          key={`${exemplar.file_id}-plot`}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.25 }}
          className="h-[480px] w-full rounded-md border border-nx-muted/40 bg-nx-bg-elev-1/40 p-2"
        >
          <PreprocessingSpectrum
            wn={wn}
            intensity={intensity}
            baseline={isArPLSStep ? exemplar.stages.baseline : null}
            // When showing arPLS, plot the cosmic-cleaned trace WITH the
            // baseline overlaid — visually that's the most informative frame.
            // But the `intensity` we want is whatever the *stage* output is
            // per the description (after_baseline). We accept that the
            // baseline visually doesn't sit on top of the result; it sits in
            // its original space. To keep things crisp + scientifically
            // honest, we overlay the *raw* baseline against the after-
            // baseline result and let the user see the magnitude difference.
            traceColor={CLASS_COLOR[exemplar.primary_class]}
            yAxisLabel={Y_AXIS_LABEL[step.key]}
            traceName={`${exemplar.primary_class} · step ${stepIdx + 1}`}
          />
        </motion.div>

        {/* Description panel */}
        <aside className="flex flex-col gap-3 rounded-md border border-nx-muted/40 bg-nx-bg-elev-1/40 p-5">
          <AnimatePresence mode="wait">
            <motion.div
              key={step.key}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.25 }}
              className="flex flex-col gap-3"
            >
              <div className="flex items-baseline gap-2">
                <span className="font-mono text-[0.6rem] uppercase tracking-[0.18em] text-nx-fg/40">
                  Step {stepIdx + 1} / {nSteps}
                </span>
              </div>
              <h3 className="font-display text-lg leading-snug text-nx-accent">
                {step.label}
              </h3>
              <p className="text-sm text-nx-fg/80 leading-relaxed">
                {step.description}
              </p>
              <div className="mt-1">
                <span className="font-mono text-[0.55rem] uppercase tracking-[0.18em] text-nx-fg/40 block mb-1">
                  Algorithm
                </span>
                <Badge
                  variant="outline"
                  className="font-mono text-[0.7rem] border-nx-accent/40 bg-nx-bg-elev-1 text-nx-accent whitespace-normal break-words text-left leading-snug py-1 max-w-full"
                >
                  {step.algorithm}
                </Badge>
              </div>
              {isCroppedSpace && (
                <p className="mt-1 font-mono text-[0.65rem] text-nx-fg/45 leading-snug">
                  axis: 987 bins (400-1800 + 2800-3050 cm⁻¹)
                </p>
              )}
              {!isCroppedSpace && (
                <p className="mt-1 font-mono text-[0.65rem] text-nx-fg/45 leading-snug">
                  axis: 2048 bins (~76-3499 cm⁻¹)
                </p>
              )}
            </motion.div>
          </AnimatePresence>
        </aside>
      </div>

      {/* Controls */}
      <div className="mt-5 flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-1 rounded-md border border-nx-muted/40 bg-nx-bg-elev-1/40 p-1">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => handleStep(stepIdx - 1)}
            disabled={stepIdx === 0}
            className="h-8 px-2 text-nx-fg/80 hover:text-nx-fg disabled:opacity-30"
            aria-label="Previous step"
          >
            <ChevronLeft className="size-4" />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={handlePlayPause}
            className="h-8 px-2 text-nx-accent hover:text-nx-accent"
            aria-label={playing ? "Pause autoplay" : "Play autoplay"}
          >
            {playing ? (
              <Pause className="size-4" />
            ) : (
              <Play className="size-4" />
            )}
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => handleStep(stepIdx + 1)}
            disabled={stepIdx >= nSteps - 1}
            className="h-8 px-2 text-nx-fg/80 hover:text-nx-fg disabled:opacity-30"
            aria-label="Next step"
          >
            <ChevronRight className="size-4" />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={handleRestart}
            className="h-8 px-2 text-nx-fg/60 hover:text-nx-fg"
            aria-label="Restart pipeline animation"
          >
            <RotateCcw className="size-4" />
          </Button>
        </div>
        <span className="font-mono text-[0.6rem] text-nx-fg/40">
          {playing ? `auto · ${AUTO_ADVANCE_MS / 1000}s per step` : "manual"}
        </span>
        <span className="ml-auto font-mono text-[0.6rem] text-nx-fg/40">
          source: <span className="text-nx-fg/70">atlas/preprocess.py</span> ·
          paper §2.3
        </span>
      </div>

      {/* Pipeline overview legend */}
      <PipelineLegend
        steps={pipeline}
        activeIdx={stepIdx}
        onPick={handleStep}
      />
    </section>
  );
}

// ---------------------------------------------------------------------------

function PipelineLegend({
  steps,
  activeIdx,
  onPick,
}: {
  steps: PipelineStepDef[];
  activeIdx: number;
  onPick: (i: number) => void;
}) {
  return (
    <div className="mt-10">
      <h3 className="mb-3 font-mono text-[0.6rem] uppercase tracking-[0.18em] text-nx-fg/45">
        Full pipeline reference
      </h3>
      <ul className="grid gap-2 lg:grid-cols-3 xl:grid-cols-6">
        {steps.map((s, i) => {
          const active = i === activeIdx;
          return (
            <li key={s.key}>
              <button
                type="button"
                onClick={() => onPick(i)}
                className={cn(
                  "group flex h-full w-full flex-col gap-1.5 rounded-sm border px-3 py-2.5 text-left transition-colors",
                  active
                    ? "border-nx-accent/60 bg-nx-bg-elev-2/40"
                    : "border-nx-muted/30 bg-nx-bg-elev-1/40 hover:bg-nx-bg-elev-1/80 hover:border-nx-muted/60",
                )}
              >
                <div className="flex items-center gap-2">
                  <span
                    className={cn(
                      "font-mono text-[0.55rem] rounded-full size-4 flex items-center justify-center tabular-nums",
                      active
                        ? "bg-nx-accent text-nx-bg"
                        : "bg-nx-bg-elev-1 text-nx-fg/50 ring-1 ring-nx-muted/40",
                    )}
                  >
                    {i + 1}
                  </span>
                  <span
                    className={cn(
                      "font-display text-[0.72rem] leading-tight",
                      active ? "text-nx-accent" : "text-nx-fg/90",
                    )}
                  >
                    {s.label}
                  </span>
                </div>
                <p className="text-[0.7rem] leading-snug text-nx-fg/55">
                  {s.description}
                </p>
                <span className="mt-auto font-mono text-[0.55rem] text-nx-fg/35 truncate">
                  {s.algorithm}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
