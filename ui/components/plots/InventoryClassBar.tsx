"use client";

/**
 * Inventory tab — Plot 1: class composition stacked bar.
 *
 * X axis = primary class (STEC / Non-STEC / Salmonella / H2O).
 * Each x-tick is a stack of one trace per strain (subclass) so the segment
 * heights show how the 87 files break down by strain within each class.
 *
 * Plan ref: §4 W2 ("class composition stacked bar").
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

interface Props {
  files: InventoryFile[];
}

export function InventoryClassBar({ files }: Props) {
  const { traces, layout } = useMemo(() => {
    // Group: {primary_class -> {subclass -> count}}
    const grouped: Record<ClassName, Map<string, number>> = {
      STEC: new Map(),
      "Non-STEC": new Map(),
      Salmonella: new Map(),
      H2O: new Map(),
    };
    for (const f of files) {
      const sub = f.subclass ?? "(none)";
      const klass = f.primary_class;
      const m = grouped[klass];
      m.set(sub, (m.get(sub) ?? 0) + 1);
    }

    // One trace per (class, subclass) so HoverCard can drive strain detail
    // outside the plot — but for the stacked bar we use Plotly's barmode=stack
    // and one trace per subclass (re-using the class color for fill, with a
    // darker outline to differentiate strains visually).
    const traces: Data[] = [];
    for (const klass of CLASS_ORDER) {
      const subs = [...grouped[klass].entries()].sort(([a], [b]) =>
        a.localeCompare(b),
      );
      subs.forEach(([sub, count], idx) => {
        // Alternate opacity to differentiate stacks within a class.
        const opacity = 0.55 + 0.45 * (idx / Math.max(subs.length - 1, 1));
        traces.push({
          type: "bar",
          x: [klass],
          y: [count],
          name: `${klass} · ${sub}`,
          marker: {
            color: CLASS_COLOR[klass],
            opacity,
            line: { color: nxColors.bg, width: 1 },
          },
          hovertemplate:
            `<b>${klass}</b><br>${sub}<br>%{y} files<extra></extra>`,
        });
      });
    }

    const layout: Partial<Layout> = {
      barmode: "stack",
      showlegend: false,
      height: 360,
      xaxis: { title: { text: "Class" }, categoryorder: "array", categoryarray: CLASS_ORDER },
      yaxis: { title: { text: "Files" }, dtick: 5 },
    };
    return { traces, layout };
  }, [files]);

  return (
    <PlotlyChart
      data={traces}
      layout={layout}
      style={{ width: "100%", height: 360 }}
    />
  );
}
