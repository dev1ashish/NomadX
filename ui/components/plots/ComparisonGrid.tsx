"use client";

/**
 * Grid view — small multiples. One Plotly subplot per staged spectrum,
 * shared x-axis range via useLinkedZoom; per-row scale+opacity via
 * useScrollFocus when scrolling.
 */
import { useMemo } from "react";
import { motion } from "framer-motion";
import type { Data, Layout, PlotRelayoutEvent } from "plotly.js";
import { PlotlyChart } from "./PlotlyChart";
import { nxColors, nxPlotlyLayout } from "@/lib/plotly-theme";
import { applyNormalization } from "@/lib/normalize";
import { useLinkedZoom } from "@/lib/use-linked-zoom";
import { useScrollFocus } from "@/lib/use-scroll-focus";
import { useSpectra } from "@/lib/use-spectrum";
import { REGION_RANGES, type SpectrumSidecar, type NormalizationMode } from "@/lib/types";
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
  const byId = useSpectra(staged.map((s) => s.file_id));

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
            fileId={s.file_id}
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
  fileId: string;
  spectrum: SpectrumSidecar | undefined;
  label: string;
  role: string;
  color: string;
  normalization: NormalizationMode;
  xRange?: [number, number];
  yRange?: [number, number];
  register: (key: string, el: HTMLElement | null) => void;
  onRelayout: (key: string, ev: Readonly<PlotRelayoutEvent>) => void;
}

function GridCell({
  fileId,
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
          onInitialized={(_, gd) => register(fileId, gd)}
          onUpdate={(_, gd) => register(fileId, gd)}
          onRelayout={(ev) => onRelayout(fileId, ev)}
        />
      </div>
    </motion.div>
  );
}
