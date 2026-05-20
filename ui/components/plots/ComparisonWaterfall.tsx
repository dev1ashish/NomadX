"use client";

import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";
import { PlotlyChart } from "./PlotlyChart";
import { nxColors, nxPlotlyLayout } from "@/lib/plotly-theme";
import { applyNormalization, minMax } from "@/lib/normalize";
import { useSpectra } from "@/lib/use-spectrum";
import { REGION_RANGES } from "@/lib/types";
import type { ViewProps } from "./view-props";

const CLASS_COLOR: Record<string, string> = {
  STEC: nxColors.classStec,
  "Non-STEC": nxColors.classNonStec,
  Salmonella: nxColors.classSalm,
  H2O: nxColors.classH2o,
};

export function ComparisonWaterfall({
  staged,
  normalization,
  region,
}: ViewProps) {
  const byId = useSpectra(staged.map((s) => s.file_id));

  const region_range = REGION_RANGES[region];

  const data: Data[] = useMemo(() => {
    return staged
      .map((s, i) => {
        const sc = byId.get(s.file_id);
        if (!sc) return null;
        const norm =
          normalization === "raw" ? minMax(sc.mean_pp) : applyNormalization(sc.mean_pp, normalization);
        const scaled = minMax(norm);
        const offset = i * 1.1;
        const y = scaled.map((v) => v + offset);
        return {
          type: "scattergl",
          mode: "lines",
          x: sc.wn_pp,
          y,
          name: s.display_label,
          line: {
            color:
              s.color_override ?? CLASS_COLOR[sc.primary_class] ?? nxColors.accent,
            width: 1.4,
          },
          hovertemplate: `<b>${s.display_label}</b><br>%{x:.1f} cm<sup>-1</sup><extra></extra>`,
        } satisfies Data;
      })
      .filter(Boolean) as Data[];
  }, [staged, byId, normalization]);

  const computedHeight = Math.max(360, staged.length * 90);

  const layout: Partial<Layout> = {
    ...nxPlotlyLayout,
    height: computedHeight,
    xaxis: {
      ...nxPlotlyLayout.xaxis,
      title: { text: "Wavenumber (cm⁻¹)" },
      range: region_range ?? undefined,
      autorange: region_range ? false : true,
      fixedrange: false,
    },
    yaxis: {
      ...nxPlotlyLayout.yaxis,
      title: { text: "Intensity (stacked)" },
      tickmode: "array",
      tickvals: staged.map((_, i) => i * 1.1 + 0.5),
      ticktext: staged.map((s) => s.display_label),
      fixedrange: false,
    },
    showlegend: false,
  };

  return (
    <div className="rounded-md border border-nx-muted bg-nx-bg-elev-1/30 p-3">
      <div style={{ height: computedHeight }}>
        <PlotlyChart data={data} layout={layout} config={{ scrollZoom: true }} />
      </div>
    </div>
  );
}
