"use client";

/**
 * Per-class distribution box for the selected feature. Renders an
 * approximate box plot from {mean, std, n} per class — the sidecar ships
 * stats rather than 7,122 raw rows.
 *
 * Each "box" is drawn as a band of [mean - std, mean + std] with a center
 * line at mean. Whiskers extend to ±2σ. Plot uses Plotly's `box` trace with
 * `q1/median/q3/lowerfence/upperfence` set directly (the precomputed-box
 * form), so this is exact box geometry — not a fake sample.
 *
 * Plan ref: §4 W5 — "Per-class distribution: render as Plotly box plot
 * using the per-class {mean, std, n} (approximate; this is faster than
 * violins)."
 */
import dynamic from "next/dynamic";
import { useMemo } from "react";
import type { ComponentType } from "react";
import type { Data, Layout, Config } from "plotly.js";
import { nxColors, nxPlotlyConfig, nxPlotlyLayout } from "@/lib/plotly-theme";
import type { ClassName } from "@/lib/types";

const CLASS_ORDER: ClassName[] = ["STEC", "Non-STEC", "Salmonella", "H2O"];

const CLASS_COLORS: Record<ClassName, string> = {
  STEC: nxColors.classStec,
  "Non-STEC": nxColors.classNonStec,
  Salmonella: nxColors.classSalm,
  H2O: nxColors.classH2o,
};

export type ClassStat = { mean: number; std: number; n: number };
export type PerClassStats = Record<ClassName, ClassStat>;

type PlotProps = {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  style?: React.CSSProperties;
  useResizeHandler?: boolean;
};

const Plot = dynamic(
  () =>
    import("react-plotly.js").then(
      (m) => m.default as ComponentType<PlotProps>,
    ),
  {
    ssr: false,
    loading: () => (
      <div className="h-[280px] w-full animate-pulse rounded-[var(--radius-sm)] bg-nx-bg-elev-1" />
    ),
  },
);

interface FeatureBoxProps {
  featureName: string;
  stats: PerClassStats;
}

export function FeatureBox({ featureName, stats }: FeatureBoxProps) {
  const { plotData, plotLayout } = useMemo(() => {
    const traces: Data[] = CLASS_ORDER.map((cls) => {
      const s = stats[cls];
      const lower = s.mean - 2 * s.std;
      const q1 = s.mean - s.std;
      const q3 = s.mean + s.std;
      const upper = s.mean + 2 * s.std;
      return {
        type: "box",
        name: cls,
        x: [cls],
        lowerfence: [lower],
        q1: [q1],
        median: [s.mean],
        q3: [q3],
        upperfence: [upper],
        // Plotly requires `mean` to be a numeric array when provided.
        mean: [s.mean],
        marker: { color: CLASS_COLORS[cls] },
        line: { color: CLASS_COLORS[cls] },
        boxpoints: false,
        hovertemplate:
          `<b>${cls}</b> (n=${s.n})<br>` +
          "mean ± σ: %{median:.4f} ± " +
          s.std.toFixed(4) +
          "<extra></extra>",
        showlegend: false,
      } as Data;
    });

    const layout: Partial<Layout> = {
      ...nxPlotlyLayout,
      title: {
        text: featureName,
        font: {
          color: nxColors.fg,
          size: 13,
          family: "JetBrains Mono Variable, monospace",
        },
      },
      margin: { l: 56, r: 16, t: 40, b: 40 },
      xaxis: {
        ...nxPlotlyLayout.xaxis,
        title: { text: "" },
      },
      yaxis: {
        ...nxPlotlyLayout.yaxis,
        title: { text: "value (mean ± 2σ)" },
        zeroline: false,
      },
      boxmode: "group",
    };

    return { plotData: traces, plotLayout: layout };
  }, [featureName, stats]);

  return (
    <div className="h-[280px] w-full">
      <Plot
        data={plotData}
        layout={plotLayout}
        config={nxPlotlyConfig}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
