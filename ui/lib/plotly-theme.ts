/**
 * NomadX-themed Plotly defaults.
 * Tab workers should spread `nxPlotlyLayout` into their `layout` prop on
 * `<PlotlyChart>` so every chart shares the same black canvas + cyan grid.
 */
import type { Layout, Config } from "plotly.js";

export const nxColors = {
  bg: "#000000",
  fg: "#FFFFFF",
  accent: "#39B8DC",
  accentDeep: "#135A6F",
  muted: "#313131",
  bgElev1: "#04161B",
  classStec: "#D63333",
  classNonStec: "#1F7A4D",
  classSalm: "#7A3D99",
  classH2o: "#3070B5",
} as const;

export const nxClassColorway: string[] = [
  nxColors.classStec,
  nxColors.classNonStec,
  nxColors.classSalm,
  nxColors.classH2o,
  nxColors.accent,
];

export const nxPlotlyLayout: Partial<Layout> = {
  paper_bgcolor: nxColors.bg,
  plot_bgcolor: nxColors.bg,
  font: {
    family: "JetBrains Mono Variable, ui-monospace, monospace",
    color: nxColors.fg,
    size: 12,
  },
  colorway: nxClassColorway,
  margin: { l: 56, r: 24, t: 32, b: 48 },
  xaxis: {
    gridcolor: nxColors.muted,
    zerolinecolor: nxColors.muted,
    linecolor: nxColors.muted,
    tickcolor: nxColors.muted,
    color: nxColors.fg,
  },
  yaxis: {
    gridcolor: nxColors.muted,
    zerolinecolor: nxColors.muted,
    linecolor: nxColors.muted,
    tickcolor: nxColors.muted,
    color: nxColors.fg,
  },
  legend: {
    bgcolor: "rgba(0,0,0,0)",
    font: { color: nxColors.fg },
  },
  hoverlabel: {
    bgcolor: nxColors.bgElev1,
    bordercolor: nxColors.accent,
    font: { color: nxColors.fg, family: "JetBrains Mono Variable, monospace" },
  },
};

export const nxPlotlyConfig: Partial<Config> = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
};
