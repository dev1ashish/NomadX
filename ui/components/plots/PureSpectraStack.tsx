"use client";

/**
 * PureSpectraStack — vertically stacked overlay of the K=7 MCR-ALS pure
 * spectra. Each line is rendered with a small fixed y-offset (no axis baseline
 * crowding) so all seven traces are visible at once. The caller toggles
 * visibility via the `visible` map keyed by component k (0..K-1).
 *
 * Plan ref: ULTRAPLAN.md §W6 (MCR-ALS demo).
 */

import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";

import { PlotlyChart } from "./PlotlyChart";
import { nxColors } from "@/lib/plotly-theme";

export interface PureComponent {
  k: number;
  label: string;
  spectrum: number[];
  color: string;
  global_d_stec_nonstec: number;
}

interface PureSpectraStackProps {
  wn: number[];
  components: PureComponent[];
  /** Map of component k -> visibility flag. Missing keys treated as `true`. */
  visible: Record<number, boolean>;
}

/**
 * Per-component vertical offset for the stacked layout (in normalized
 * spectrum-amplitude units). 0.06 spaces the K=7 traces cleanly inside a
 * single chart without overlapping the labels.
 */
const STACK_OFFSET = 0.06;

function normalize01(arr: number[]): number[] {
  let min = Infinity;
  let max = -Infinity;
  for (const v of arr) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const span = max - min;
  if (!Number.isFinite(span) || span <= 0) return arr.map(() => 0);
  return arr.map((v) => (v - min) / span);
}

export function PureSpectraStack({
  wn,
  components,
  visible,
}: PureSpectraStackProps) {
  const { data, layout } = useMemo(() => {
    const traces: Data[] = [];
    const annotations: Partial<Layout>["annotations"] = [];

    // Render from top to bottom so the legend reads top-down.
    const ordered = [...components].sort((a, b) => a.k - b.k);
    const N = ordered.length;
    ordered.forEach((comp, i) => {
      const isOn = visible[comp.k] !== false;
      const yOffset = (N - 1 - i) * STACK_OFFSET;
      const ys = normalize01(comp.spectrum).map((v) => v + yOffset);
      traces.push({
        type: "scattergl",
        mode: "lines",
        x: wn,
        y: ys,
        name: `C${comp.k + 1}`,
        line: { color: comp.color, width: isOn ? 1.6 : 0.6 },
        opacity: isOn ? 1 : 0.2,
        hovertemplate:
          `<b>C${comp.k + 1}</b><br>` +
          `%{x:.1f} cm<sup>-1</sup><br>` +
          `normalized intensity: %{y:.3f}<br>` +
          `d STEC↔Non-STEC: ${comp.global_d_stec_nonstec.toFixed(2)}` +
          "<extra></extra>",
        showlegend: false,
      });
      annotations.push({
        x: wn[0],
        y: yOffset + 0.6 * STACK_OFFSET,
        xref: "x" as const,
        yref: "y" as const,
        text: `C${comp.k + 1}`,
        showarrow: false,
        font: {
          family: "JetBrains Mono Variable, monospace",
          size: 11,
          color: isOn ? comp.color : nxColors.muted,
        },
        xanchor: "right" as const,
        xshift: -6,
      });
    });

    const layout: Partial<Layout> = {
      autosize: true,
      hovermode: "closest",
      showlegend: false,
      annotations,
      xaxis: {
        title: { text: "Raman shift (cm⁻¹)" },
      },
      yaxis: {
        title: { text: "Pure spectra (stacked, normalized)" },
        showticklabels: false,
        zeroline: false,
        range: [-STACK_OFFSET * 0.5, N * STACK_OFFSET + 0.4],
      },
      margin: { l: 80, r: 24, t: 24, b: 56 },
    };
    return { data: traces, layout };
  }, [wn, components, visible]);

  return (
    <PlotlyChart
      data={data}
      layout={layout}
      style={{ height: "100%", width: "100%" }}
    />
  );
}
