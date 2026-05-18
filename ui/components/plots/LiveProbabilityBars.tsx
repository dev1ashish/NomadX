"use client";

/**
 * Horizontal probability bar chart for Live inference (W7).
 *
 * Four bars, one per class, sorted descending. Colors use the canonical
 * Atlas `--class-*` palette so the highest-prob class stands out. Plotly's
 * own bar transition is suppressed; growth animation is handled by the
 * parent (`LiveInference.tsx`) via a key-based remount that re-runs Plotly's
 * initial draw, which is "good enough" without a Framer Motion wrapper
 * around an SSR-only chart.
 *
 * Plan ref: §4 W7 ("4-bar probability chart, sorted descending,
 * animated growth via Framer Motion key-based remount").
 */
import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";

import { PlotlyChart } from "./PlotlyChart";
import { nxColors } from "@/lib/plotly-theme";
import type { ClassName } from "@/lib/types";

const CLASS_COLOR: Record<ClassName, string> = {
  STEC: nxColors.classStec,
  "Non-STEC": nxColors.classNonStec,
  Salmonella: nxColors.classSalm,
  H2O: nxColors.classH2o,
};

interface Props {
  probabilities: Record<ClassName, number>;
  predicted: ClassName;
}

export function LiveProbabilityBars({ probabilities, predicted }: Props) {
  const { trace, layout } = useMemo(() => {
    // Sort ascending so Plotly's horizontal bar draws the top class at the
    // top of the y-axis (Plotly stacks the first y-value at the bottom).
    const sorted = (Object.entries(probabilities) as [ClassName, number][])
      .sort(([, a], [, b]) => a - b);

    const labels = sorted.map(([k]) => k);
    const values = sorted.map(([, v]) => v);
    const colors = sorted.map(([k]) =>
      k === predicted ? CLASS_COLOR[k] : `${CLASS_COLOR[k]}88`,
    );

    const trace: Data = {
      type: "bar",
      orientation: "h",
      x: values,
      y: labels,
      text: values.map((v) => `${(v * 100).toFixed(1)}%`),
      textposition: "outside",
      textfont: { color: nxColors.fg, family: "JetBrains Mono Variable, monospace" },
      marker: { color: colors, line: { color: nxColors.bg, width: 1 } },
      hovertemplate: "<b>%{y}</b><br>%{x:.4f}<extra></extra>",
    };

    const layout: Partial<Layout> = {
      height: 260,
      margin: { l: 96, r: 56, t: 16, b: 32 },
      showlegend: false,
      xaxis: {
        range: [0, 1.08],
        tickformat: ".0%",
        gridcolor: nxColors.muted,
        zerolinecolor: nxColors.muted,
        color: nxColors.fg,
      },
      yaxis: {
        color: nxColors.fg,
        automargin: true,
      },
      bargap: 0.35,
    };
    return { trace, layout };
  }, [probabilities, predicted]);

  return (
    <PlotlyChart
      data={[trace]}
      layout={layout}
      style={{ width: "100%", height: 260 }}
    />
  );
}
