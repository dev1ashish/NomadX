"use client";

/**
 * PreprocessingSpectrum — step-aware Plotly line chart for the Preprocessing
 * pipeline tab. Renders ONE primary trace (the current stage's spectrum) and,
 * optionally, a second faint baseline trace overlay that fades in only on the
 * arPLS step (so the user can SEE what's being subtracted).
 *
 * Trace data is fed in as props — the parent <PreprocessingPipeline/>
 * remounts this component via `key={stageKey}` on each step change to get
 * Framer-Motion-style fade-in on the chart wrapper, but the Plotly trace
 * itself transitions smoothly because react-plotly.js diff-applies the new
 * data without remounting the WebGL canvas.
 */

import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";

import { PlotlyChart } from "./PlotlyChart";
import { nxColors } from "@/lib/plotly-theme";

interface PreprocessingSpectrumProps {
  /** x-axis (wavenumbers) — full 2048-bin axis or 987-bin cropped axis. */
  wn: number[];
  /** Primary y-axis trace (intensity) for the current stage. */
  intensity: number[];
  /** Optional baseline trace — same length as `wn` — shown faded on arPLS. */
  baseline?: number[] | null;
  /** Stroke color for the primary trace (class-specific). */
  traceColor: string;
  /** Y-axis title (changes per stage: counts vs. SNV-z). */
  yAxisLabel: string;
  /** Human-readable trace name (legend / hover prefix). */
  traceName: string;
}

export function PreprocessingSpectrum({
  wn,
  intensity,
  baseline = null,
  traceColor,
  yAxisLabel,
  traceName,
}: PreprocessingSpectrumProps) {
  const { data, layout } = useMemo(() => {
    const traces: Data[] = [];

    // Optional baseline trace — render UNDER the spectrum so it sits behind.
    if (baseline && baseline.length === wn.length) {
      traces.push({
        type: "scattergl",
        mode: "lines",
        x: wn,
        y: baseline,
        name: "arPLS baseline",
        line: { color: nxColors.accent, width: 1.25, dash: "dot" },
        opacity: 0.55,
        hovertemplate:
          "<b>baseline %{x:.1f} cm⁻¹</b><br>%{y:.2f}<extra></extra>",
      });
    }

    traces.push({
      type: "scattergl",
      mode: "lines",
      x: wn,
      y: intensity,
      name: traceName,
      line: { color: traceColor, width: 1.6 },
      hovertemplate:
        "<b>%{x:.1f} cm⁻¹</b><br>%{y:.4f}<extra></extra>",
    });

    const layout: Partial<Layout> = {
      autosize: true,
      hovermode: "closest",
      showlegend: Boolean(baseline),
      legend: {
        x: 0.99,
        y: 0.98,
        xanchor: "right",
        yanchor: "top",
        bgcolor: "rgba(4,22,27,0.6)",
        bordercolor: nxColors.muted,
        borderwidth: 1,
        font: { color: nxColors.fg, size: 11 },
      },
      xaxis: {
        title: { text: "Raman shift (cm⁻¹)" },
        showspikes: true,
        spikemode: "across",
        spikecolor: nxColors.accent,
        spikethickness: 1,
      },
      yaxis: {
        title: { text: yAxisLabel },
      },
      margin: { l: 64, r: 24, t: 32, b: 56 },
      transition: {
        duration: 600,
        easing: "cubic-in-out",
      },
    };

    return { data: traces, layout };
  }, [wn, intensity, baseline, traceColor, yAxisLabel, traceName]);

  return (
    <PlotlyChart
      data={data}
      layout={layout}
      style={{ height: "100%", width: "100%" }}
    />
  );
}
