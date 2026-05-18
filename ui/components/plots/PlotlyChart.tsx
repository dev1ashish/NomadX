"use client";

/**
 * SSR-safe Plotly wrapper.
 * All tabs MUST import Plotly via this component (bare `react-plotly.js`
 * breaks SSR because Plotly touches `window` at import time).
 *
 * Plan ref: §1 (react-plotly.js justification), §6 (worker prompt constraint).
 */
import dynamic from "next/dynamic";
import type { Data, Layout, Config } from "plotly.js";
import type { ComponentType } from "react";
import { nxPlotlyConfig, nxPlotlyLayout } from "@/lib/plotly-theme";

// `react-plotly.js` doesn't have a meaningful loading state; ssr:false is
// enough. Falls back to a transparent placeholder while the chunk arrives.
type PlotProps = {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  style?: React.CSSProperties;
  className?: string;
  useResizeHandler?: boolean;
  onInitialized?: (figure: unknown, gd: HTMLElement) => void;
  onUpdate?: (figure: unknown, gd: HTMLElement) => void;
  onRelayout?: (event: Readonly<import("plotly.js").PlotRelayoutEvent>) => void;
};

const Plot = dynamic(
  () => import("react-plotly.js").then((m) => m.default as ComponentType<PlotProps>),
  {
    ssr: false,
    loading: () => (
      <div className="h-[320px] w-full animate-pulse rounded-[var(--radius-sm)] bg-nx-bg-elev-1" />
    ),
  },
);

export interface PlotlyChartProps {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  className?: string;
  style?: React.CSSProperties;
  onInitialized?: (figure: unknown, gd: HTMLElement) => void;
  onUpdate?: (figure: unknown, gd: HTMLElement) => void;
  onRelayout?: (event: Readonly<import("plotly.js").PlotRelayoutEvent>) => void;
}

export function PlotlyChart({
  data,
  layout,
  config,
  className,
  style,
  onInitialized,
  onUpdate,
  onRelayout,
}: PlotlyChartProps) {
  const mergedLayout: Partial<Layout> = { ...nxPlotlyLayout, ...layout };
  const mergedConfig: Partial<Config> = { ...nxPlotlyConfig, ...config };
  return (
    <Plot
      data={data}
      layout={mergedLayout}
      config={mergedConfig}
      useResizeHandler
      className={className}
      style={{ width: "100%", height: "100%", ...style }}
      onInitialized={onInitialized}
      onUpdate={onUpdate}
      onRelayout={onRelayout}
    />
  );
}
