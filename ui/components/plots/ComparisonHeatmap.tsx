"use client";

import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";
import { PlotlyChart } from "./PlotlyChart";
import { nxColors, nxPlotlyLayout } from "@/lib/plotly-theme";
import { applyNormalization } from "@/lib/normalize";
import { useSpectra } from "@/lib/use-spectrum";
import { REGION_RANGES } from "@/lib/types";
import type { ViewProps } from "./view-props";

export function ComparisonHeatmap({
  staged,
  normalization,
  region,
}: ViewProps) {
  const byId = useSpectra(staged.map((s) => s.file_id));

  const region_range = REGION_RANGES[region];

  const { data, layout, height } = useMemo(() => {
    const rows: number[][] = [];
    const labels: string[] = [];
    let wn: number[] = [];
    for (const s of staged) {
      const sc = byId.get(s.file_id);
      if (!sc) continue;
      const y = applyNormalization(sc.mean_pp, normalization);
      rows.push(y);
      labels.push(s.display_label);
      if (wn.length === 0) wn = sc.wn_pp;
    }
    const height = Math.max(320, staged.length * 32 + 80);
    const data: Data[] = [
      {
        type: "heatmap",
        z: rows,
        x: wn,
        y: labels,
        colorscale: "Viridis",
        colorbar: { title: { text: "Intensity" }, tickfont: { color: nxColors.fg } },
        hovertemplate:
          "<b>%{y}</b><br>%{x:.1f} cm<sup>-1</sup><br>%{z:.4f}<extra></extra>",
      },
    ];
    const layout: Partial<Layout> = {
      ...nxPlotlyLayout,
      height,
      xaxis: {
        ...nxPlotlyLayout.xaxis,
        title: { text: "Wavenumber (cm⁻¹)" },
        range: region_range ?? undefined,
        autorange: region_range ? false : true,
        fixedrange: false,
      },
      yaxis: {
        ...nxPlotlyLayout.yaxis,
        autorange: "reversed",
        fixedrange: true,
      },
    };
    return { data, layout, height };
  }, [staged, byId, normalization, region_range]);

  return (
    <div className="rounded-md border border-nx-muted bg-nx-bg-elev-1/30 p-3">
      <div style={{ height }}>
        <PlotlyChart data={data} layout={layout} config={{ scrollZoom: true }} />
      </div>
    </div>
  );
}
