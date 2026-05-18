"use client";

/**
 * Top-15 horizontal bar of Cohen's d (STEC vs Non-STEC), color-coded by
 * feature family. Click a bar to select the feature in the parent panel.
 *
 * The shared `PlotlyChart` wrapper does not forward click handlers (it's
 * intentionally minimal for SSR safety), so this component owns its own
 * `dynamic(import("react-plotly.js"))` mount to expose `onClick`.
 *
 * Plan ref: §4 W5 — "top-15 by |d_stec_nonstec|, color by family".
 */
import dynamic from "next/dynamic";
import { useMemo } from "react";
import type { ComponentType } from "react";
import type {
  Data,
  Layout,
  Config,
  PlotMouseEvent,
} from "plotly.js";
import { nxColors, nxPlotlyConfig, nxPlotlyLayout } from "@/lib/plotly-theme";
import type { Feature } from "@/lib/types";

export const FAMILY_COLORS: Record<Feature["family"], string> = {
  band: nxColors.accent,
  spectral: nxColors.classH2o,
  mcr: nxColors.classSalm,
  spatial: nxColors.classNonStec,
  bio: nxColors.classStec,
};

export const FAMILY_LABELS: Record<Feature["family"], string> = {
  band: "Band",
  spectral: "Spectral",
  mcr: "MCR",
  spatial: "Spatial",
  bio: "Bio",
};

type PlotProps = {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  style?: React.CSSProperties;
  className?: string;
  useResizeHandler?: boolean;
  onClick?: (event: Readonly<PlotMouseEvent>) => void;
};

const Plot = dynamic(
  () =>
    import("react-plotly.js").then(
      (m) => m.default as ComponentType<PlotProps>,
    ),
  {
    ssr: false,
    loading: () => (
      <div className="h-[440px] w-full animate-pulse rounded-[var(--radius-sm)] bg-nx-bg-elev-1" />
    ),
  },
);

interface FeatureBarProps {
  features: Feature[];
  selected?: string;
  onSelect?: (name: string) => void;
}

export function FeatureBar({ features, selected, onSelect }: FeatureBarProps) {
  const { plotData, plotLayout, orderedNames } = useMemo(() => {
    const ranked = [...features]
      .filter((f) => typeof f.d_stec_nonstec === "number")
      .sort(
        (a, b) =>
          Math.abs(b.d_stec_nonstec ?? 0) - Math.abs(a.d_stec_nonstec ?? 0),
      )
      .slice(0, 15)
      .reverse(); // low-to-high so the largest |d| sits at the top.

    const names = ranked.map((f) => f.name);
    const ds = ranked.map((f) => f.d_stec_nonstec ?? 0);
    const colors = ranked.map((f) =>
      f.name === selected ? nxColors.fg : FAMILY_COLORS[f.family],
    );
    const lineColors = ranked.map((f) =>
      f.name === selected ? nxColors.accent : "rgba(0,0,0,0)",
    );
    const families = ranked.map((f) => FAMILY_LABELS[f.family]);

    const data: Data[] = [
      {
        type: "bar",
        orientation: "h",
        x: ds,
        y: names,
        marker: {
          color: colors,
          line: { color: lineColors, width: 2 },
        },
        customdata: families,
        hovertemplate:
          "<b>%{y}</b><br>family: %{customdata}<br>d = %{x:.3f}<extra></extra>",
      },
    ];

    const layout: Partial<Layout> = {
      ...nxPlotlyLayout,
      title: {
        text: "Top 15 features by |Cohen's d| — STEC vs Non-STEC",
        font: { color: nxColors.fg, size: 14 },
      },
      margin: { l: 220, r: 24, t: 48, b: 40 },
      xaxis: {
        ...nxPlotlyLayout.xaxis,
        title: { text: "Cohen's d (file-level)" },
        zeroline: true,
        zerolinewidth: 1,
      },
      yaxis: {
        ...nxPlotlyLayout.yaxis,
        automargin: true,
        tickfont: { size: 10, family: "JetBrains Mono Variable, monospace" },
      },
      bargap: 0.25,
    };

    return { plotData: data, plotLayout: layout, orderedNames: names };
  }, [features, selected]);

  const mergedLayout: Partial<Layout> = { ...nxPlotlyLayout, ...plotLayout };
  const mergedConfig: Partial<Config> = { ...nxPlotlyConfig };

  return (
    <div className="h-[440px] w-full">
      <Plot
        data={plotData}
        layout={mergedLayout}
        config={mergedConfig}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
        onClick={(event) => {
          if (!onSelect) return;
          const point = event.points?.[0];
          if (!point) return;
          // `y` is the category label for horizontal bars.
          const name = typeof point.y === "string" ? point.y : undefined;
          if (name && orderedNames.includes(name)) onSelect(name);
        }}
      />
    </div>
  );
}
