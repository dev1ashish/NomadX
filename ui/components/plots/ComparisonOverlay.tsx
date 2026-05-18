"use client";

import { useEffect, useMemo, useState } from "react";
import type { Data, Layout } from "plotly.js";
import { PlotlyChart } from "./PlotlyChart";
import { nxColors, nxPlotlyLayout } from "@/lib/plotly-theme";
import { getSidecar } from "@/lib/data";
import { applyNormalization } from "@/lib/normalize";
import { REGION_RANGES, type SpectrumSidecar } from "@/lib/types";
import type { ViewProps } from "./view-props";

const CLASS_COLOR: Record<string, string> = {
  STEC: nxColors.classStec,
  "Non-STEC": nxColors.classNonStec,
  Salmonella: nxColors.classSalm,
  H2O: nxColors.classH2o,
};

const ROLE_DASH: Record<string, "solid" | "dash"> = {
  control_pos: "solid",
  test: "solid",
  blank: "dash",
};

export function ComparisonOverlay({
  staged,
  normalization,
  region,
}: ViewProps) {
  const [byId, setById] = useState<Map<string, SpectrumSidecar>>(new Map());

  useEffect(() => {
    let cancelled = false;
    const toLoad = staged.filter((s) => !byId.has(s.file_id));
    if (toLoad.length === 0) return;
    Promise.all(
      toLoad.map((s) =>
        getSidecar<SpectrumSidecar>(`spectra/${s.file_id}.json`)
          .then((d) => [s.file_id, d] as const)
          .catch(() => [s.file_id, null] as const),
      ),
    ).then((results) => {
      if (cancelled) return;
      setById((prev) => {
        const next = new Map(prev);
        for (const [id, sidecar] of results) {
          if (sidecar) next.set(id, sidecar);
        }
        return next;
      });
    });
    return () => {
      cancelled = true;
    };
  }, [staged, byId]);

  const region_range = REGION_RANGES[region];

  const data: Data[] = useMemo(() => {
    return staged
      .map((s) => {
        const sc = byId.get(s.file_id);
        if (!sc) return null;
        const y = applyNormalization(sc.mean_pp, normalization);
        return {
          type: "scattergl",
          mode: "lines",
          x: sc.wn_pp,
          y,
          name: `${s.display_label} (${s.role.replace("_", " ")})`,
          visible: s.visible !== false,
          line: {
            color:
              s.color_override ?? CLASS_COLOR[sc.primary_class] ?? nxColors.accent,
            width: 1.5,
            dash: ROLE_DASH[s.role] ?? "solid",
          },
          opacity: s.role === "blank" ? 0.7 : 1,
          hovertemplate:
            "<b>%{x:.1f} cm<sup>-1</sup></b><br>%{y:.4f}<br>%{fullData.name}<extra></extra>",
        } satisfies Data;
      })
      .filter(Boolean) as Data[];
  }, [staged, byId, normalization]);

  const layout: Partial<Layout> = {
    ...nxPlotlyLayout,
    height: 560,
    xaxis: {
      ...nxPlotlyLayout.xaxis,
      title: { text: "Wavenumber (cm⁻¹)" },
      range: region_range ?? undefined,
      autorange: region_range ? false : true,
      fixedrange: false,
    },
    yaxis: {
      ...nxPlotlyLayout.yaxis,
      title: { text: "Intensity (normalized)" },
      fixedrange: false,
    },
    hovermode: "x unified",
    showlegend: true,
  };

  return (
    <div className="rounded-md border border-nx-muted bg-nx-bg-elev-1/30 p-3">
      <div className="h-[560px]">
        <PlotlyChart data={data} layout={layout} config={{ scrollZoom: true }} />
      </div>
    </div>
  );
}
