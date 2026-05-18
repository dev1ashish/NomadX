"use client";

import { useEffect, useMemo, useState } from "react";
import type { Data, Layout, PlotRelayoutEvent } from "plotly.js";
import { PlotlyChart } from "./PlotlyChart";
import { nxColors, nxPlotlyLayout } from "@/lib/plotly-theme";
import { getSidecar } from "@/lib/data";
import { applyNormalization } from "@/lib/normalize";
import { useLinkedZoom } from "@/lib/use-linked-zoom";
import { REGION_RANGES, type SpectrumSidecar } from "@/lib/types";
import type { ViewProps } from "./view-props";

const CLASS_COLOR: Record<string, string> = {
  STEC: nxColors.classStec,
  "Non-STEC": nxColors.classNonStec,
  Salmonella: nxColors.classSalm,
  H2O: nxColors.classH2o,
};

export function ComparisonDiff({
  staged,
  normalization,
  region,
  linkXZoom,
  referenceFileId,
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

  const { register, onRelayout } = useLinkedZoom(linkXZoom);

  const region_range = REGION_RANGES[region];
  const reference = referenceFileId ? byId.get(referenceFileId) : undefined;

  const refY = useMemo(() => {
    if (!reference) return null;
    return applyNormalization(reference.mean_pp, normalization);
  }, [reference, normalization]);

  if (!referenceFileId) {
    return (
      <div className="rounded-md border border-dashed border-nx-muted bg-nx-bg-elev-1/30 px-6 py-12 text-center">
        <p className="font-display text-base text-nx-fg/80">No reference set.</p>
        <p className="font-mono text-xs text-nx-fg/45 mt-2">
          Stage a Blank (water) — or any other spectrum — to use as the
          subtraction reference.
        </p>
      </div>
    );
  }
  if (!reference || !refY) {
    return (
      <div className="font-mono text-xs text-nx-fg/55">Loading reference…</div>
    );
  }

  const others = staged.filter((s) => s.file_id !== referenceFileId);

  return (
    <div className="grid gap-4 [grid-template-columns:repeat(auto-fit,minmax(360px,1fr))]">
      {others.map((s) => {
        const sc = byId.get(s.file_id);
        if (!sc) {
          return (
            <div
              key={s.file_id}
              className="h-[260px] rounded-md border border-nx-muted bg-nx-bg-elev-1/30 flex items-center justify-center"
            >
              <span className="font-mono text-[0.7rem] text-nx-fg/45">
                Loading…
              </span>
            </div>
          );
        }
        const y = applyNormalization(sc.mean_pp, normalization);
        const minLen = Math.min(y.length, refY.length);
        const diff: number[] = new Array(minLen);
        for (let i = 0; i < minLen; i++) diff[i] = y[i] - refY[i];
        const x = sc.wn_pp.slice(0, minLen);

        const data: Data[] = [
          {
            type: "scattergl",
            mode: "lines",
            x,
            y: diff,
            line: {
              color:
                s.color_override ?? CLASS_COLOR[sc.primary_class] ?? nxColors.accent,
              width: 1.5,
            },
            hovertemplate:
              "<b>%{x:.1f} cm<sup>-1</sup></b><br>Δ %{y:.4f}<extra></extra>",
          },
        ];

        const layout: Partial<Layout> = {
          ...nxPlotlyLayout,
          height: 240,
          margin: { l: 48, r: 16, t: 8, b: 36 },
          xaxis: {
            ...nxPlotlyLayout.xaxis,
            range: region_range ?? undefined,
            autorange: region_range ? false : true,
            fixedrange: false,
          },
          yaxis: {
            ...nxPlotlyLayout.yaxis,
            zeroline: true,
            zerolinecolor: nxColors.accentDeep,
            fixedrange: false,
          },
          showlegend: false,
        };

        return (
          <div
            key={s.file_id}
            className="rounded-md border border-nx-muted bg-nx-bg-elev-1/30 p-2"
          >
            <header className="flex items-baseline justify-between px-1 pb-1">
              <span className="font-mono text-[0.7rem]">
                {s.display_label} − {referenceFileId}
              </span>
              <span className="font-mono text-[0.55rem] text-nx-accent uppercase tracking-[0.18em]">
                Δ
              </span>
            </header>
            <div className="h-[240px]">
              <PlotlyChart
                data={data}
                layout={layout}
                config={{ scrollZoom: true }}
                onInitialized={(_: unknown, gd: HTMLElement) => register(s.file_id, gd)}
                onUpdate={(_: unknown, gd: HTMLElement) => register(s.file_id, gd)}
                onRelayout={(ev: Readonly<PlotRelayoutEvent>) => onRelayout(s.file_id, ev)}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
