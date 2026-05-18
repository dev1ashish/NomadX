"use client";

/**
 * Mean-spectrum line plot for Live inference (W7).
 *
 * Renders the file-mean preprocessed spectrum returned by Modal's `/predict`
 * (`spectrum_mean` over `wn`). Lightweight band annotations are inlined here
 * to avoid coupling to W3's `SpectrumViewer` (per worker brief).
 *
 * Plan ref: §4 W7 ("Mean spectrum plot — use the same component pattern as
 * W3 ... Re-implement minimal version here").
 */
import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";

import { PlotlyChart } from "./PlotlyChart";
import { nxColors } from "@/lib/plotly-theme";

/** Seven anchor bands used across the project (ULTRAPLAN §W3). */
const ANCHOR_BANDS: Array<{ wn: number; label: string }> = [
  { wn: 1004, label: "Phe 1004" },
  { wn: 1117, label: "LPS 1117" },
  { wn: 1194, label: "LPS 1194" },
  { wn: 1242, label: "Amide III 1242" },
  { wn: 1338, label: "1338" },
  { wn: 1454, label: "δCH₂ 1454" },
  { wn: 1658, label: "Amide I 1658" },
];

interface Props {
  wn: number[];
  spectrum: number[];
}

export function LiveMeanSpectrum({ wn, spectrum }: Props) {
  const { traces, layout } = useMemo(() => {
    const trace: Data = {
      type: "scattergl",
      mode: "lines",
      x: wn,
      y: spectrum,
      line: { color: nxColors.accent, width: 1.5 },
      hovertemplate:
        "<b>%{x:.1f} cm⁻¹</b><br>I = %{y:.3f}<extra></extra>",
      name: "mean",
    };

    // Only annotate bands inside the spectrum window.
    const wnMin = wn.length ? wn[0] : 0;
    const wnMax = wn.length ? wn[wn.length - 1] : 0;
    const lo = Math.min(wnMin, wnMax);
    const hi = Math.max(wnMin, wnMax);

    const shapes = ANCHOR_BANDS.filter((b) => b.wn >= lo && b.wn <= hi).map(
      (b) => ({
        type: "line" as const,
        xref: "x" as const,
        yref: "paper" as const,
        x0: b.wn,
        x1: b.wn,
        y0: 0,
        y1: 1,
        line: { color: nxColors.accentDeep, width: 1, dash: "dot" as const },
      }),
    );

    const annotations = ANCHOR_BANDS.filter((b) => b.wn >= lo && b.wn <= hi).map(
      (b) => ({
        x: b.wn,
        y: 1,
        xref: "x" as const,
        yref: "paper" as const,
        text: b.label,
        showarrow: false,
        yanchor: "bottom" as const,
        font: {
          family: "JetBrains Mono Variable, monospace",
          color: nxColors.accent,
          size: 9,
        },
      }),
    );

    const layout: Partial<Layout> = {
      height: 260,
      margin: { l: 56, r: 24, t: 24, b: 40 },
      xaxis: {
        title: { text: "Wavenumber (cm⁻¹)" },
        gridcolor: nxColors.muted,
        zerolinecolor: nxColors.muted,
        color: nxColors.fg,
      },
      yaxis: {
        title: { text: "Intensity (a.u.)" },
        gridcolor: nxColors.muted,
        zerolinecolor: nxColors.muted,
        color: nxColors.fg,
      },
      shapes,
      annotations,
      showlegend: false,
    };
    return { traces: [trace], layout };
  }, [wn, spectrum]);

  return (
    <PlotlyChart
      data={traces}
      layout={layout}
      style={{ width: "100%", height: 260 }}
    />
  );
}
