"use client";

/**
 * Inventory tab — Plot 2: per-strain grouped bar.
 *
 * X axis = strain (subclass). Two grouped bars per strain:
 *   - files in that strain
 *   - QC-passed pixels (sum of n_pixels * qc_pass_rate)
 *
 * Strains share the bar color of their parent primary_class.
 * Plan ref: §4 W2 ("per-strain small-multiples — files × QC-passed pixels").
 */
import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";
import { PlotlyChart } from "./PlotlyChart";
import { nxColors } from "@/lib/plotly-theme";
import type { ClassName } from "@/lib/types";
import type { InventoryFile } from "@/components/tabs/inventory-data";

const CLASS_ORDER: ClassName[] = ["STEC", "Non-STEC", "Salmonella", "H2O"];

const CLASS_COLOR: Record<ClassName, string> = {
  STEC: nxColors.classStec,
  "Non-STEC": nxColors.classNonStec,
  Salmonella: nxColors.classSalm,
  H2O: nxColors.classH2o,
};

export interface Strain {
  label: string;
  primary_class: ClassName;
  files: number;
  qc_pixels: number;
  file_ids: string[];
}

export function strainBreakdown(files: InventoryFile[]): Strain[] {
  const byKey = new Map<string, Strain>();
  for (const f of files) {
    const sub = f.subclass ?? "(none)";
    const key = `${f.primary_class}::${sub}`;
    const existing = byKey.get(key);
    const qc = f.n_pixels * f.qc_pass_rate;
    if (existing) {
      existing.files += 1;
      existing.qc_pixels += qc;
      existing.file_ids.push(f.file_id);
    } else {
      byKey.set(key, {
        label: sub,
        primary_class: f.primary_class,
        files: 1,
        qc_pixels: qc,
        file_ids: [f.file_id],
      });
    }
  }
  // Order: by class order, then by label.
  return [...byKey.values()].sort((a, b) => {
    const ai = CLASS_ORDER.indexOf(a.primary_class);
    const bi = CLASS_ORDER.indexOf(b.primary_class);
    if (ai !== bi) return ai - bi;
    return a.label.localeCompare(b.label);
  });
}

interface Props {
  files: InventoryFile[];
}

export function InventoryStrainBar({ files }: Props) {
  const { traces, layout } = useMemo(() => {
    const strains = strainBreakdown(files);
    const xLabels = strains.map((s) => s.label);
    const colors = strains.map((s) => CLASS_COLOR[s.primary_class]);

    const filesTrace: Data = {
      type: "bar",
      name: "Files",
      x: xLabels,
      y: strains.map((s) => s.files),
      marker: { color: colors, line: { color: nxColors.bg, width: 1 } },
      hovertemplate: "<b>%{x}</b><br>%{y} files<extra></extra>",
      yaxis: "y",
    };
    const pxTrace: Data = {
      type: "bar",
      name: "QC-passed pixels",
      x: xLabels,
      y: strains.map((s) => Math.round(s.qc_pixels)),
      marker: {
        color: colors,
        opacity: 0.55,
        line: { color: nxColors.bg, width: 1 },
      },
      hovertemplate: "<b>%{x}</b><br>%{y} QC-passed pixels<extra></extra>",
      yaxis: "y2",
    };

    const layout: Partial<Layout> = {
      barmode: "group",
      bargap: 0.18,
      height: 360,
      showlegend: true,
      legend: { orientation: "h", x: 0, y: -0.22 },
      xaxis: { title: { text: "Strain" }, tickangle: -25 },
      yaxis: { title: { text: "Files" } },
      yaxis2: {
        title: { text: "QC-passed pixels" },
        overlaying: "y",
        side: "right",
        showgrid: false,
      },
      margin: { l: 56, r: 64, t: 32, b: 96 },
    };
    return { traces: [filesTrace, pxTrace], layout };
  }, [files]);

  return (
    <PlotlyChart
      data={traces}
      layout={layout}
      style={{ width: "100%", height: 360 }}
    />
  );
}
