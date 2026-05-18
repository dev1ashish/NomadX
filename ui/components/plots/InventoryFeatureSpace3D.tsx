"use client";

/**
 * Inventory tab — 2D feature-space scatter (W11 → W13).
 *
 * Originally a 3D scatter; downgraded to 2D after a UX read-through. With 87
 * points and 4 classes, 3D's rotation buys "wow" but loses readability —
 * depth cues are weak and class separability is hard to see at a glance.
 * The 2D PC1×PC2 view conveys the same story (PC1 41% + PC2 20% = 61% of
 * variance) and is instantly legible.
 *
 * The user can toggle the axis pair (PC1×PC2 / PC1×PC3 / PC2×PC3) to inspect
 * the third dimension without leaving 2D. Variance bars on the right keep
 * the scree-plot context in view.
 *
 * Sidecar: `inventory_pca.json` (unchanged from W11).
 */
import { useEffect, useMemo, useState } from "react";
import type { Data, Layout } from "plotly.js";
import { PlotlyChart } from "./PlotlyChart";
import { nxColors } from "@/lib/plotly-theme";
import { cn } from "@/lib/cn";
import type { ClassName } from "@/lib/types";

const CLASS_ORDER: ClassName[] = ["STEC", "Non-STEC", "Salmonella", "H2O"];

const CLASS_COLOR: Record<ClassName, string> = {
  STEC: nxColors.classStec,
  "Non-STEC": nxColors.classNonStec,
  Salmonella: nxColors.classSalm,
  H2O: nxColors.classH2o,
};

const CLASS_BG: Record<ClassName, string> = {
  STEC: "bg-class-stec",
  "Non-STEC": "bg-class-nonstec",
  Salmonella: "bg-class-salm",
  H2O: "bg-class-h2o",
};

export interface InventoryPcaFile {
  file_id: string;
  primary_class: ClassName;
  subclass: string | null;
  pc1: number;
  pc2: number;
  pc3: number;
  n_pixels: number;
  qc_pass_rate: number;
}

export interface InventoryPcaSidecar {
  files: InventoryPcaFile[];
  variance_explained: [number, number, number];
  n_features: number;
}

type AxisPair = "12" | "13" | "23";

const PAIRS: { key: AxisPair; xLabel: string; yLabel: string; xIdx: 0 | 1 | 2; yIdx: 0 | 1 | 2 }[] = [
  { key: "12", xLabel: "PC1", yLabel: "PC2", xIdx: 0, yIdx: 1 },
  { key: "13", xLabel: "PC1", yLabel: "PC3", xIdx: 0, yIdx: 2 },
  { key: "23", xLabel: "PC2", yLabel: "PC3", xIdx: 1, yIdx: 2 },
];

async function fetchInventoryPcaSidecar(): Promise<InventoryPcaSidecar> {
  const res = await fetch("/data/inventory_pca.json", { cache: "force-cache" });
  if (!res.ok) {
    throw new Error(
      `Failed to fetch /data/inventory_pca.json: ${res.status} ${res.statusText}`,
    );
  }
  return (await res.json()) as InventoryPcaSidecar;
}

export interface InventoryFeatureSpace2DProps {
  preloaded?: InventoryPcaSidecar;
  onLoaded?: (sidecar: InventoryPcaSidecar) => void;
}

export default function InventoryFeatureSpace2D({
  preloaded,
  onLoaded,
}: InventoryFeatureSpace2DProps) {
  const [sidecar, setSidecar] = useState<InventoryPcaSidecar | null>(
    preloaded ?? null,
  );
  const [error, setError] = useState<string | null>(null);
  const [pair, setPair] = useState<AxisPair>("12");

  useEffect(() => {
    if (preloaded) {
      onLoaded?.(preloaded);
      return;
    }
    let cancelled = false;
    fetchInventoryPcaSidecar()
      .then((d) => {
        if (cancelled) return;
        setSidecar(d);
        onLoaded?.(d);
      })
      .catch((e: unknown) => {
        if (!cancelled)
          setError(
            e instanceof Error ? e.message : "Failed to load PCA sidecar",
          );
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preloaded]);

  const { traces, layout, varExplainedTotal } = useMemo(() => {
    if (!sidecar)
      return {
        traces: [] as Data[],
        layout: {} as Partial<Layout>,
        varExplainedTotal: 0,
      };

    const cfg = PAIRS.find((p) => p.key === pair) ?? PAIRS[0];
    const pcs = sidecar.variance_explained;
    const xVar = pcs[cfg.xIdx];
    const yVar = pcs[cfg.yIdx];
    const total = xVar + yVar;

    const grouped: Record<ClassName, InventoryPcaFile[]> = {
      STEC: [],
      "Non-STEC": [],
      Salmonella: [],
      H2O: [],
    };
    for (const f of sidecar.files) {
      grouped[f.primary_class].push(f);
    }

    const traces: Data[] = CLASS_ORDER.map((klass) => {
      const rows = grouped[klass];
      const xValues = rows.map((r) => [r.pc1, r.pc2, r.pc3][cfg.xIdx]);
      const yValues = rows.map((r) => [r.pc1, r.pc2, r.pc3][cfg.yIdx]);
      const subs = rows.map((r) => r.subclass ?? "(none)");
      return {
        type: "scatter",
        mode: "markers",
        name: klass,
        x: xValues,
        y: yValues,
        text: subs,
        customdata: rows.map((r) => [r.file_id, r.subclass ?? "(none)"]),
        marker: {
          size: 10,
          symbol: "circle",
          color: CLASS_COLOR[klass],
          opacity: 0.85,
          line: { color: "rgba(255,255,255,0.4)", width: 1 },
        },
        hovertemplate:
          "<b>%{customdata[0]}</b><br>" +
          "%{customdata[1]}<br>" +
          `${cfg.xLabel}: ` +
          "%{x:.2f}<br>" +
          `${cfg.yLabel}: ` +
          "%{y:.2f}" +
          "<extra></extra>",
      } as Data;
    });

    const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

    const layout: Partial<Layout> = {
      height: 480,
      margin: { l: 60, r: 20, t: 20, b: 50 },
      showlegend: true,
      legend: {
        x: 0.985,
        y: 0.985,
        xanchor: "right",
        yanchor: "top",
        bgcolor: "rgba(4,22,27,0.7)",
        bordercolor: "rgba(57,184,220,0.25)",
        borderwidth: 1,
        font: { color: nxColors.fg, size: 11 },
      },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      xaxis: {
        title: { text: `${cfg.xLabel} · ${pct(xVar)} variance`, standoff: 12 },
        gridcolor: "rgba(57,184,220,0.10)",
        zerolinecolor: "rgba(57,184,220,0.30)",
        zerolinewidth: 1,
        color: nxColors.fg,
        tickfont: { color: "rgba(255,255,255,0.5)" },
      },
      yaxis: {
        title: { text: `${cfg.yLabel} · ${pct(yVar)} variance`, standoff: 12 },
        gridcolor: "rgba(57,184,220,0.10)",
        zerolinecolor: "rgba(57,184,220,0.30)",
        zerolinewidth: 1,
        color: nxColors.fg,
        tickfont: { color: "rgba(255,255,255,0.5)" },
        scaleanchor: "x",
        scaleratio: 1,
      },
      hoverlabel: {
        bgcolor: "rgba(4,22,27,0.95)",
        bordercolor: "rgba(57,184,220,0.4)",
        font: { color: nxColors.fg, family: "var(--font-mono)", size: 11 },
      },
    };

    return { traces, layout, varExplainedTotal: total };
  }, [sidecar, pair]);

  if (error) {
    return <p className="font-mono text-sm text-nx-danger">{error}</p>;
  }

  if (!sidecar) {
    return (
      <div className="h-[480px] w-full animate-pulse rounded-[var(--radius-sm)] bg-nx-bg-elev-2" />
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {/* Axis-pair selector + variance summary */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-1 rounded-md border border-nx-muted/40 bg-nx-bg-elev-1/40 p-1">
          {PAIRS.map((p) => (
            <button
              key={p.key}
              type="button"
              onClick={() => setPair(p.key)}
              className={cn(
                "font-mono text-[0.7rem] px-2.5 py-1 rounded-sm transition-colors",
                pair === p.key
                  ? "bg-nx-accent text-nx-bg"
                  : "text-nx-fg/60 hover:text-nx-fg hover:bg-nx-bg-elev-2/60",
              )}
            >
              {p.xLabel} × {p.yLabel}
            </button>
          ))}
        </div>
        <div className="font-mono text-[0.65rem] text-nx-fg/50">
          showing {(varExplainedTotal * 100).toFixed(1)}% of total variance
        </div>
      </div>

      {/* Class legend chips — also act as a quick visual key */}
      <div className="flex flex-wrap items-center gap-3">
        {CLASS_ORDER.map((klass) => (
          <span
            key={klass}
            className="flex items-center gap-1.5 font-mono text-[0.65rem] text-nx-fg/65"
          >
            <span
              className={cn("inline-block size-2 rounded-full", CLASS_BG[klass])}
            />
            {klass}
          </span>
        ))}
      </div>

      {/* The scatter */}
      <PlotlyChart
        data={traces}
        layout={layout}
        style={{ width: "100%", height: 480 }}
      />

      {/* Scree-plot mini: variance per PC */}
      <div className="flex items-center gap-3 mt-1">
        <span className="font-mono text-[0.6rem] uppercase tracking-[0.16em] text-nx-fg/40">
          Variance per PC
        </span>
        <div className="flex-1 flex items-center gap-2">
          {sidecar.variance_explained.map((v, i) => (
            <div key={i} className="flex items-center gap-1.5 flex-1">
              <span className="font-mono text-[0.6rem] text-nx-fg/60 w-7">
                PC{i + 1}
              </span>
              <div className="flex-1 h-1.5 rounded-sm bg-nx-bg-elev-2 overflow-hidden">
                <div
                  className="h-full bg-nx-accent"
                  style={{ width: `${(v / sidecar.variance_explained[0]) * 100}%` }}
                />
              </div>
              <span className="font-mono text-[0.6rem] text-nx-fg/70 tabular-nums w-10">
                {(v * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
