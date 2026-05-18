"use client";

/**
 * ReconstructionPlot — overlays the observed mean spectrum (white) against
 * the MCR-ALS reconstruction (cyan) built from toggled-on pure components,
 * plus the residual trace below.
 *
 * Plan ref: ULTRAPLAN.md §W6 (MCR-ALS demo).
 */

import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";

import { PlotlyChart } from "./PlotlyChart";
import { nxColors } from "@/lib/plotly-theme";

interface ReconstructionPlotProps {
  wn: number[];
  observed: number[];
  /** (K, B) array of pure spectra. */
  componentSpectra: number[][];
  /** Length-K weights to scale each component by. */
  componentWeights: number[];
  /** Length-K visibility mask; component is included only if true. */
  visible: boolean[];
}

function combine(
  pureMat: number[][],
  weights: number[],
  visible: boolean[],
  B: number,
): number[] {
  const out = new Array<number>(B).fill(0);
  for (let k = 0; k < pureMat.length; k++) {
    if (!visible[k]) continue;
    const w = weights[k];
    if (!Number.isFinite(w) || w === 0) continue;
    const s = pureMat[k];
    for (let i = 0; i < B; i++) out[i] += w * s[i];
  }
  return out;
}

export function ReconstructionPlot({
  wn,
  observed,
  componentSpectra,
  componentWeights,
  visible,
}: ReconstructionPlotProps) {
  const { data, layout } = useMemo(() => {
    const B = wn.length;
    const recon = combine(componentSpectra, componentWeights, visible, B);
    // Match the absolute scale of the observed via an offset (MCR-ALS fit was
    // on (X - X.min()), so the recon sits on a different baseline). We shift
    // the reconstruction by `mean(observed) - mean(recon)` so the eye can
    // compare shapes without one trace floating off-axis.
    let muObs = 0;
    let muRec = 0;
    for (let i = 0; i < B; i++) {
      muObs += observed[i];
      muRec += recon[i];
    }
    muObs /= Math.max(B, 1);
    muRec /= Math.max(B, 1);
    const shift = muObs - muRec;
    const reconAligned = recon.map((v) => v + shift);
    const residual = observed.map((v, i) => v - reconAligned[i]);

    const observedTrace: Data = {
      type: "scattergl",
      mode: "lines",
      x: wn,
      y: observed,
      name: "Observed mean",
      line: { color: nxColors.fg, width: 1.6 },
      xaxis: "x",
      yaxis: "y",
      hovertemplate:
        "<b>Observed</b> · %{x:.1f} cm<sup>-1</sup> · %{y:.4f}<extra></extra>",
    };

    const reconTrace: Data = {
      type: "scattergl",
      mode: "lines",
      x: wn,
      y: reconAligned,
      name: "Σ pure × weight",
      line: { color: nxColors.accent, width: 1.4 },
      xaxis: "x",
      yaxis: "y",
      hovertemplate:
        "<b>Reconstruction</b> · %{x:.1f} cm<sup>-1</sup> · %{y:.4f}<extra></extra>",
    };

    const residualTrace: Data = {
      type: "scattergl",
      mode: "lines",
      x: wn,
      y: residual,
      name: "Residual",
      line: { color: nxColors.classStec, width: 1 },
      xaxis: "x2",
      yaxis: "y2",
      hovertemplate:
        "<b>Residual</b> · %{x:.1f} cm<sup>-1</sup> · %{y:.4f}<extra></extra>",
    };

    const layout: Partial<Layout> = {
      autosize: true,
      hovermode: "closest",
      showlegend: true,
      legend: {
        orientation: "h",
        x: 0,
        y: 1.06,
        bgcolor: "rgba(0,0,0,0)",
        font: { color: nxColors.fg, size: 11 },
      },
      grid: { rows: 2, columns: 1, pattern: "independent" },
      xaxis: {
        title: { text: "" },
        domain: [0, 1],
        anchor: "y",
        showticklabels: false,
      },
      yaxis: {
        title: { text: "Observed vs reconstruction" },
        domain: [0.32, 1],
        anchor: "x",
      },
      xaxis2: {
        title: { text: "Raman shift (cm⁻¹)" },
        domain: [0, 1],
        anchor: "y2",
      },
      yaxis2: {
        title: { text: "Residual" },
        domain: [0, 0.24],
        anchor: "x2",
        zerolinecolor: nxColors.muted,
      },
      margin: { l: 64, r: 24, t: 28, b: 56 },
    };
    return { data: [observedTrace, reconTrace, residualTrace], layout };
  }, [wn, observed, componentSpectra, componentWeights, visible]);

  return (
    <PlotlyChart
      data={data}
      layout={layout}
      style={{ height: "100%", width: "100%" }}
    />
  );
}
