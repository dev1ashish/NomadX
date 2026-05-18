"use client";

/**
 * SpectrumViewer — Plotly line chart of a single file's mean Raman spectrum.
 *
 * Renders 7 (technically 8 — 1004, 1117, 1194, 1242, 1338, 1454, 1658, 2900)
 * "anchor" wavenumbers as vertical lines + invisible hoverable markers so each
 * anchor surfaces its chemistry one-liner on hover inside the Plotly canvas.
 *
 * Plan ref: ULTRAPLAN.md §W3 (Spectrum explorer).
 */

import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";

import { PlotlyChart } from "./PlotlyChart";
import { nxColors } from "@/lib/plotly-theme";

export interface BandAnchor {
  /** Wavenumber in cm^-1. */
  wn: number;
  /** Short chemistry one-liner used in hover and HoverCard. */
  chemistry: string;
}

interface SpectrumViewerProps {
  /** Wavenumber x-axis. */
  wn: number[];
  /** Mean intensity y-axis (same length as `wn`). */
  intensity: number[];
  /** Anchors to overlay as vertical lines + hover markers. */
  anchors: BandAnchor[];
  /** Class color used for the main spectrum trace stroke. */
  traceColor?: string;
  /** "Preprocessed (SNV)" or "Raw counts" — used for the y-axis title. */
  yAxisLabel: string;
  /** Plotly trace name (legend / hover prefix). */
  traceName: string;
}

export function SpectrumViewer({
  wn,
  intensity,
  anchors,
  traceColor = nxColors.accent,
  yAxisLabel,
  traceName,
}: SpectrumViewerProps) {
  const { data, layout } = useMemo(() => {
    // Main spectrum trace
    const spectrumTrace: Data = {
      type: "scattergl",
      mode: "lines",
      x: wn,
      y: intensity,
      name: traceName,
      line: { color: traceColor, width: 1.5 },
      hovertemplate:
        "<b>%{x:.1f} cm<sup>-1</sup></b><br>%{y:.4f}<extra></extra>",
    };

    // Anchor vertical-line "shapes" — drawn straight onto the plot area.
    const yMin = Math.min(...intensity);
    const yMax = Math.max(...intensity);
    const yPad = (yMax - yMin) * 0.04;
    const yLineMin = yMin - yPad;
    const yLineMax = yMax + yPad;

    const shapes: Partial<Layout>["shapes"] = anchors.map((a) => ({
      type: "line" as const,
      xref: "x" as const,
      yref: "y" as const,
      x0: a.wn,
      x1: a.wn,
      y0: yLineMin,
      y1: yLineMax,
      line: {
        color: nxColors.accent,
        width: 1,
        dash: "dot" as const,
      },
    }));

    // Annotation labels (the wavenumber, top of the chart)
    const annotations: Partial<Layout>["annotations"] = anchors.map((a) => ({
      x: a.wn,
      y: 1,
      xref: "x" as const,
      yref: "paper" as const,
      text: `<b>${a.wn}</b>`,
      showarrow: false,
      font: {
        family: "JetBrains Mono Variable, monospace",
        size: 10,
        color: nxColors.accent,
      },
      bgcolor: nxColors.bg,
      bordercolor: nxColors.accent,
      borderwidth: 1,
      borderpad: 2,
      yanchor: "bottom" as const,
      xanchor: "center" as const,
    }));

    // Invisible hover markers — one per anchor, sitting near the top of the
    // spectrum, so Plotly's hover layer can pick them up and surface the
    // chemistry string. We anchor them to the spectrum's actual y at each wn
    // (nearest-neighbor lookup) so the hover hit area sits on the curve.
    const yAtAnchors = anchors.map((a) => {
      // wn is sorted ascending → binary-ish search via findIndex (87×8 small).
      let bestIdx = 0;
      let bestDist = Infinity;
      for (let i = 0; i < wn.length; i++) {
        const d = Math.abs(wn[i] - a.wn);
        if (d < bestDist) {
          bestDist = d;
          bestIdx = i;
        }
      }
      return intensity[bestIdx];
    });

    const hoverMarkers: Data = {
      type: "scatter",
      mode: "markers",
      x: anchors.map((a) => a.wn),
      y: yAtAnchors,
      text: anchors.map((a) => a.chemistry),
      customdata: anchors.map((a) => `${a.wn} cm⁻¹`),
      hovertemplate:
        "<b>%{customdata}</b><br>%{text}<extra></extra>",
      marker: {
        size: 14,
        color: nxColors.accent,
        opacity: 0.0001, // invisible but still hoverable
        line: { width: 0 },
      },
      showlegend: false,
      name: "anchor",
    };

    const layout: Partial<Layout> = {
      shapes,
      annotations,
      autosize: true,
      hovermode: "closest",
      showlegend: false,
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
      margin: { l: 60, r: 24, t: 36, b: 56 },
    };

    return { data: [spectrumTrace, hoverMarkers], layout };
  }, [wn, intensity, anchors, traceColor, yAxisLabel, traceName]);

  return (
    <PlotlyChart
      data={data}
      layout={layout}
      style={{ height: "100%", width: "100%" }}
    />
  );
}
