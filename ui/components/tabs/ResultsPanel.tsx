"use client";

/**
 * Results tab (W8) — Stage 15F production-classifier panel.
 *
 * Surfaces FINAL/PAPER.md §6 numbers verbatim from the JSON sidecars built
 * by `ui/scripts/build_results.py`. UI never re-derives metrics at runtime.
 *
 * Layout:
 *   1. KPI strip — PLS-DA-on-raw 0.603, LogReg fw 0.448 + CI, McNemar p,
 *      feature count.
 *   2. Confusion matrix (4x4 Plotly heatmap, click-to-inspect file list).
 *   3. Algorithm comparison bar (PLS-DA / LogReg / XGB on engineered).
 *   4. Bootstrap histogram with CI band + point-estimate line.
 *   5. McNemar 2x2 contingency table + p-value badge.
 *   6. Stage 7 deployment callout.
 *
 * Plan ref: §4 W8.
 */
import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import type { Data, Layout } from "plotly.js";
import { PlotlyChart } from "@/components/plots/PlotlyChart";
import { nxColors } from "@/lib/plotly-theme";
import { getSidecar } from "@/lib/data";
import { KpiStrip, type Kpi } from "@/components/layout/KpiStrip";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

// ----------------------------- Sidecar types -----------------------------

interface Stage15F {
  loso_mean_acc: number;
  loso_fw_acc: number;
  bootstrap_ci_low: number;
  bootstrap_ci_high: number;
  mcnemar_p: number;
  n_features: number;
  per_strain_accuracy: Record<string, number>;
  algo_comparison: Record<
    string,
    { mean_loso_accuracy: number; mean_loso_macro_recall: number }
  >;
  plsda_raw_loso: number;
}

interface Confusion {
  classes: string[];
  matrix: number[][];
  per_cell_files: Record<string, string[]>;
  n_files: number;
}

interface Bootstrap {
  samples: number[];
  ci_low: number;
  ci_high: number;
  point_estimate: number;
  n_boot: number;
  n_files: number;
  seed: number;
}

interface Mcnemar {
  both_right: number;
  logreg_only_right: number;
  plsda_only_right: number;
  both_wrong: number;
  p_value: number;
  n_total: number;
}

// ----------------------------- Helpers -----------------------------

function fmt3(x: number): string {
  return x.toFixed(3);
}

function pFmt(p: number): string {
  // Display the published rounded value alongside the precise tie-breaker.
  return p < 0.001 ? p.toExponential(2) : p.toFixed(4);
}

// ----------------------------- Sub-components -----------------------------

function ConfusionMatrix({
  cm,
  onCellClick,
}: {
  cm: Confusion;
  onCellClick: (row: string, col: string, files: string[]) => void;
}) {
  const { classes, matrix } = cm;
  // Plotly heatmap, with annotations for cell counts. Color scale cyan → black
  // (low counts blend into canvas, hot cells pop).
  const z = matrix;
  const annotations: NonNullable<Layout["annotations"]> = [];
  // Determine a contrast threshold so dark cells get white text and bright
  // cells get black text.
  const flat = matrix.flat();
  const maxVal = Math.max(...flat, 1);
  for (let i = 0; i < classes.length; i++) {
    for (let j = 0; j < classes.length; j++) {
      const v = matrix[i][j];
      annotations.push({
        x: classes[j],
        y: classes[i],
        text: String(v),
        showarrow: false,
        font: {
          family: "JetBrains Mono Variable, monospace",
          size: 14,
          color: v / maxVal > 0.55 ? "#000000" : nxColors.fg,
        },
      });
    }
  }

  const data: Data[] = [
    {
      type: "heatmap",
      x: classes,
      y: classes,
      z,
      colorscale: [
        [0, "#000000"],
        [1, "#39B8DC"],
      ],
      showscale: true,
      hovertemplate: "True %{y} → Pred %{x}: %{z} files<extra></extra>",
      xgap: 1,
      ygap: 1,
      colorbar: {
        tickfont: {
          color: nxColors.fg,
          family: "JetBrains Mono Variable, monospace",
          size: 10,
        },
        outlinecolor: nxColors.muted,
        bordercolor: nxColors.muted,
      },
    },
  ];

  const layout: Partial<Layout> = {
    height: 420,
    annotations,
    xaxis: {
      title: { text: "Predicted class" },
      side: "bottom",
      tickfont: { color: nxColors.fg },
    },
    yaxis: {
      title: { text: "True class" },
      autorange: "reversed",
      tickfont: { color: nxColors.fg },
    },
    margin: { l: 100, r: 40, t: 24, b: 56 },
  };

  return (
    <div className="relative">
      <PlotlyChart
        data={data}
        layout={layout}
        style={{ width: "100%", height: 420 }}
      />
      {/* Invisible click grid sits above the Plotly canvas so the dialog
          opens with the file list. Plotly's onClick is non-trivial through
          the dynamic wrapper; a 4x4 absolutely-positioned grid is simpler
          and stays in sync with the heatmap cells. */}
      <div
        aria-hidden={false}
        className="absolute inset-0 grid"
        style={{
          gridTemplateColumns: `100px repeat(${classes.length}, 1fr) 60px`,
          gridTemplateRows: `24px repeat(${classes.length}, 1fr) 56px`,
        }}
      >
        {/* Top-left + top row spacers */}
        <span />
        {classes.map((c) => (
          <span key={`top-${c}`} />
        ))}
        <span />
        {classes.map((rowClass, i) => (
          <Row key={rowClass}>
            <span />
            {classes.map((colClass, j) => {
              const key = `${rowClass}_${colClass}`;
              const files = cm.per_cell_files[key] ?? [];
              return (
                <button
                  key={`${i}-${j}`}
                  type="button"
                  aria-label={`True ${rowClass} predicted ${colClass} — ${matrix[i][j]} files`}
                  onClick={() => onCellClick(rowClass, colClass, files)}
                  className="cursor-pointer bg-transparent transition-colors hover:bg-white/5 focus-visible:outline focus-visible:outline-2 focus-visible:outline-nx-accent"
                />
              );
            })}
            <span />
          </Row>
        ))}
        <span />
        {classes.map((c) => (
          <span key={`bot-${c}`} />
        ))}
        <span />
      </div>
    </div>
  );
}

function Row({ children }: { children: React.ReactNode }) {
  // Helper to keep the click grid TSX readable — flattens its children.
  return <>{children}</>;
}

function AlgoComparisonBar({ data: payload }: { data: Stage15F }) {
  const order: ReadonlyArray<{ key: string; label: string }> = [
    { key: "plsda", label: "PLS-DA (engineered)" },
    { key: "logreg", label: "LogReg-L2" },
    { key: "xgb", label: "XGBoost" },
  ];
  const ys = order.map((o) => payload.algo_comparison[o.key]?.mean_loso_accuracy ?? 0);
  const text = ys.map((v) => v.toFixed(3));
  const colors = [
    nxColors.classH2o,
    nxColors.accent,
    nxColors.classSalm,
  ];
  const data: Data[] = [
    {
      type: "bar",
      x: order.map((o) => o.label),
      y: ys,
      text,
      textposition: "outside",
      textfont: {
        color: nxColors.fg,
        family: "JetBrains Mono Variable, monospace",
        size: 12,
      },
      marker: { color: colors, line: { color: nxColors.muted, width: 1 } },
      hovertemplate: "%{x}<br>LOSO acc: %{y:.3f}<extra></extra>",
    },
    // Reference line for PLS-DA-on-raw 0.603.
    {
      type: "scatter",
      mode: "lines",
      x: [-0.5, 2.5],
      y: [payload.plsda_raw_loso, payload.plsda_raw_loso],
      line: { color: nxColors.classStec, width: 2, dash: "dash" },
      name: "PLS-DA on raw (0.603)",
      hovertemplate: "PLS-DA on raw: 0.603<extra></extra>",
    },
  ];
  const layout: Partial<Layout> = {
    height: 320,
    showlegend: false,
    yaxis: {
      title: { text: "LOSO mean accuracy" },
      range: [0, 0.7],
      tickformat: ".2f",
    },
    xaxis: { title: { text: "Algorithm (35 MI-selected features)" } },
    margin: { l: 60, r: 24, t: 24, b: 64 },
    annotations: [
      {
        x: 2.5,
        y: payload.plsda_raw_loso,
        xref: "x",
        yref: "y",
        text: "PLS-DA on raw 0.603",
        showarrow: false,
        xanchor: "right",
        yanchor: "bottom",
        font: {
          color: nxColors.classStec,
          family: "JetBrains Mono Variable, monospace",
          size: 10,
        },
      },
    ],
  };
  return (
    <PlotlyChart data={data} layout={layout} style={{ width: "100%", height: 320 }} />
  );
}

function BootstrapHistogram({ data: boot }: { data: Bootstrap }) {
  // Histogram trace + CI band as a translucent vertical rectangle (shape) +
  // point-estimate dashed line.
  const min = Math.min(...boot.samples);
  const max = Math.max(...boot.samples);
  // ~40 bins across the empirical range.
  const binSize = Math.max((max - min) / 40, 0.001);
  const data: Data[] = [
    {
      type: "histogram",
      x: boot.samples,
      autobinx: false,
      xbins: { start: min, end: max, size: binSize },
      marker: {
        color: nxColors.accent,
        line: { color: nxColors.bg, width: 1 },
      },
      opacity: 0.85,
      hovertemplate: "acc range: %{x}<br>count: %{y}<extra></extra>",
      name: "Bootstrap",
    },
  ];
  const layout: Partial<Layout> = {
    height: 320,
    showlegend: false,
    yaxis: { title: { text: "Resample count" } },
    xaxis: { title: { text: "LOSO file-weighted accuracy" }, tickformat: ".2f" },
    shapes: [
      {
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: boot.ci_low,
        x1: boot.ci_high,
        y0: 0,
        y1: 1,
        fillcolor: nxColors.accentDeep,
        opacity: 0.25,
        line: { width: 0 },
      },
      {
        type: "line",
        xref: "x",
        yref: "paper",
        x0: boot.point_estimate,
        x1: boot.point_estimate,
        y0: 0,
        y1: 1,
        line: { color: nxColors.classStec, width: 2, dash: "dash" },
      },
    ],
    annotations: [
      {
        x: boot.point_estimate,
        y: 1,
        xref: "x",
        yref: "paper",
        text: `point ${fmt3(boot.point_estimate)}`,
        showarrow: false,
        xanchor: "left",
        yanchor: "top",
        font: {
          color: nxColors.classStec,
          family: "JetBrains Mono Variable, monospace",
          size: 11,
        },
      },
      {
        x: (boot.ci_low + boot.ci_high) / 2,
        y: 0.95,
        xref: "x",
        yref: "paper",
        text: `95% CI [${fmt3(boot.ci_low)}, ${fmt3(boot.ci_high)}]`,
        showarrow: false,
        yanchor: "top",
        font: {
          color: nxColors.fg,
          family: "JetBrains Mono Variable, monospace",
          size: 11,
        },
      },
    ],
    margin: { l: 60, r: 24, t: 24, b: 56 },
  };
  return (
    <PlotlyChart data={data} layout={layout} style={{ width: "100%", height: 320 }} />
  );
}

function McnemarTable({ m }: { m: Mcnemar }) {
  return (
    <div className="overflow-hidden border border-nx-muted">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="bg-nx-bg-elev-2 text-nx-fg/70 font-mono text-[11px] uppercase tracking-[0.14em]">
            <th className="border border-nx-muted px-3 py-2 text-left">
              {/* corner cell */}
            </th>
            <th className="border border-nx-muted px-3 py-2 text-center">
              PLS-DA correct
            </th>
            <th className="border border-nx-muted px-3 py-2 text-center">
              PLS-DA wrong
            </th>
          </tr>
        </thead>
        <tbody className="font-mono">
          <tr>
            <th
              scope="row"
              className="border border-nx-muted bg-nx-bg-elev-1 px-3 py-3 text-left text-[11px] uppercase tracking-[0.14em] text-nx-fg/70"
            >
              LogReg correct
            </th>
            <td className="border border-nx-muted px-3 py-3 text-center text-nx-fg">
              <span className="text-2xl tabular-nums">{m.both_right}</span>
              <span className="ml-2 text-[10px] text-nx-fg/50">both right</span>
            </td>
            <td className="border border-nx-muted px-3 py-3 text-center text-nx-accent">
              <span className="text-2xl tabular-nums">{m.logreg_only_right}</span>
              <span className="ml-2 text-[10px] text-nx-fg/50">LogReg-only</span>
            </td>
          </tr>
          <tr>
            <th
              scope="row"
              className="border border-nx-muted bg-nx-bg-elev-1 px-3 py-3 text-left text-[11px] uppercase tracking-[0.14em] text-nx-fg/70"
            >
              LogReg wrong
            </th>
            <td className="border border-nx-muted px-3 py-3 text-center text-class-stec">
              <span className="text-2xl tabular-nums">{m.plsda_only_right}</span>
              <span className="ml-2 text-[10px] text-nx-fg/50">PLS-DA-only</span>
            </td>
            <td className="border border-nx-muted px-3 py-3 text-center text-nx-fg/60">
              <span className="text-2xl tabular-nums">{m.both_wrong}</span>
              <span className="ml-2 text-[10px] text-nx-fg/50">both wrong</span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

// ----------------------------- Main panel -----------------------------

interface CellOpen {
  row: string;
  col: string;
  files: string[];
}

export function ResultsPanel() {
  const [stage15f, setStage15f] = useState<Stage15F | null>(null);
  const [confusion, setConfusion] = useState<Confusion | null>(null);
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [mcnemar, setMcnemar] = useState<Mcnemar | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cellOpen, setCellOpen] = useState<CellOpen | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      getSidecar<Stage15F>("stage15f.json"),
      getSidecar<Confusion>("confusion.json"),
      getSidecar<Bootstrap>("bootstrap.json"),
      getSidecar<Mcnemar>("mcnemar.json"),
    ])
      .then(([a, b, c, d]) => {
        if (cancelled) return;
        setStage15f(a);
        setConfusion(b);
        setBootstrap(c);
        setMcnemar(d);
      })
      .catch((e: unknown) => {
        if (!cancelled)
          setError(e instanceof Error ? e.message : "Failed to load results sidecars");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const kpis: Kpi[] = useMemo(() => {
    if (!stage15f) return [];
    return [
      {
        label: "PLS-DA LOSO (raw)",
        value: stage15f.plsda_raw_loso.toFixed(3),
      },
      {
        label: `LogReg-L2 fw · 95% CI [${fmt3(stage15f.bootstrap_ci_low)}, ${fmt3(stage15f.bootstrap_ci_high)}]`,
        value: stage15f.loso_fw_acc.toFixed(3),
      },
      {
        label: "McNemar p (LogReg > PLS-DA)",
        value: pFmt(stage15f.mcnemar_p),
      },
      {
        label: "Feature count (MI-selected)",
        value: String(stage15f.n_features),
      },
    ];
  }, [stage15f]);

  if (error) {
    return (
      <section className="mx-auto max-w-screen-2xl px-6 py-10">
        <p className="text-nx-danger font-mono text-sm">{error}</p>
      </section>
    );
  }

  return (
    <section className="mx-auto max-w-screen-2xl px-6 py-8">
      <motion.h2
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="font-display text-nx-accent text-[3.0625rem] leading-[1.1]"
      >
        Results
      </motion.h2>
      <motion.p
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, delay: 0.08, ease: "easeOut" }}
        className="mt-3 max-w-3xl text-nx-fg/60"
      >
        Stage 15F is the deployed classifier — LogReg-L2 on 35 MI-selected
        features (Branch C, file-weighted LOSO 0.448 with 95% CI overlapping
        the Branch (B) bar). PLS-DA on the raw 987-bin spectrum remains the
        project headline at 0.603. McNemar pairs LogReg above PLS-DA on the
        engineered cache (p = 0.0020).
      </motion.p>

      {stage15f ? (
        <KpiStrip items={kpis} />
      ) : (
        <ul
          role="list"
          aria-busy
          className="grid gap-6 px-6 py-6 sm:grid-cols-2 md:grid-cols-4"
        >
          {[0, 1, 2, 3].map((i) => (
            <li
              key={i}
              className="h-12 animate-pulse rounded-[var(--radius-sm)] bg-nx-bg-elev-1"
            />
          ))}
        </ul>
      )}

      <div className="mt-2 grid gap-6 lg:grid-cols-2">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, delay: 0.18, ease: "easeOut" }}
        >
          <Card>
            <CardHeader>
              <CardTitle className="font-mono text-xs uppercase tracking-[0.16em] text-nx-fg/70">
                Confusion matrix · LogReg LOSO (87 files)
              </CardTitle>
              <CardDescription>
                Click any cell to inspect the file_ids contributing to it.
                Row 4 [8,0,0,0] reproduces the Stage 7 STEC-default bias —
                all 8 held-out H<sub>2</sub>O files predicted as STEC.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {confusion ? (
                <ConfusionMatrix
                  cm={confusion}
                  onCellClick={(row, col, files) =>
                    setCellOpen({ row, col, files })
                  }
                />
              ) : (
                <div className="h-[420px] animate-pulse bg-nx-bg-elev-2" />
              )}
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, delay: 0.26, ease: "easeOut" }}
        >
          <Card>
            <CardHeader>
              <CardTitle className="font-mono text-xs uppercase tracking-[0.16em] text-nx-fg/70">
                Algorithm comparison · 35 MI features
              </CardTitle>
              <CardDescription>
                PLS-DA collapses on pre-selected features (0.324) — LogReg-L2
                holds at 0.436. Red dashes mark the PLS-DA-on-raw baseline
                that remains the project headline.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {stage15f ? (
                <AlgoComparisonBar data={stage15f} />
              ) : (
                <div className="h-[320px] animate-pulse bg-nx-bg-elev-2" />
              )}
            </CardContent>
          </Card>
        </motion.div>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, delay: 0.32, ease: "easeOut" }}
        >
          <Card>
            <CardHeader>
              <CardTitle className="font-mono text-xs uppercase tracking-[0.16em] text-nx-fg/70">
                Bootstrap distribution · LogReg LOSO fw
              </CardTitle>
              <CardDescription>
                5,000 file-wise resamples (seed=0). 95% CI band shaded teal;
                point estimate 0.448 dashed red. CI straddles 0.50 — verdict
                "Branch (C) with overlap into (B)".
              </CardDescription>
            </CardHeader>
            <CardContent>
              {bootstrap ? (
                <BootstrapHistogram data={bootstrap} />
              ) : (
                <div className="h-[320px] animate-pulse bg-nx-bg-elev-2" />
              )}
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, delay: 0.38, ease: "easeOut" }}
        >
          <Card>
            <CardHeader>
              <CardTitle className="font-mono text-xs uppercase tracking-[0.16em] text-nx-fg/70">
                McNemar paired test · LogReg vs PLS-DA (engineered)
              </CardTitle>
              <CardDescription>
                Discordant cell counts are 10 vs 0 — every file PLS-DA got
                right, LogReg also got right.{" "}
                <Badge variant="outline" className="ml-1 font-mono">
                  p = {mcnemar ? pFmt(mcnemar.p_value) : "—"}
                </Badge>
              </CardDescription>
            </CardHeader>
            <CardContent>
              {mcnemar ? (
                <McnemarTable m={mcnemar} />
              ) : (
                <div className="h-[160px] animate-pulse bg-nx-bg-elev-2" />
              )}
            </CardContent>
          </Card>
        </motion.div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, delay: 0.46, ease: "easeOut" }}
        className="mt-6"
      >
        <Card className="border-l-4 border-l-nx-danger">
          <CardHeader>
            <CardTitle className="font-mono text-xs uppercase tracking-[0.16em] text-nx-fg/70">
              Stage 7 deployment callout · mixed-sample contamination
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-nx-fg/85">
            <p>
              <span className="font-mono text-nx-accent">10–20%</span>{" "}
              accuracy drop at{" "}
              <span className="font-mono text-nx-accent">25%</span> mixed-pixel
              contamination (Stage 7, §4.6). STEC vs Non-STEC is the most
              affected pair beyond 20%.
            </p>
            <p className="text-nx-fg/60">
              Wet-lab deployment design constraint: target &lt;10% off-target
              pixels per Raman map to keep the production classifier inside its
              validated accuracy envelope.
            </p>
          </CardContent>
        </Card>
      </motion.div>

      <Dialog
        open={cellOpen !== null}
        onOpenChange={(open) => {
          if (!open) setCellOpen(null);
        }}
      >
        <DialogContent className="max-w-md border border-nx-muted bg-nx-bg-elev-1">
          <DialogHeader>
            <DialogTitle className="font-mono text-xs uppercase tracking-[0.16em] text-nx-fg/85">
              {cellOpen
                ? `True ${cellOpen.row} → Pred ${cellOpen.col} · ${cellOpen.files.length} file${cellOpen.files.length === 1 ? "" : "s"}`
                : ""}
            </DialogTitle>
            <DialogDescription className="text-nx-fg/60">
              file_ids contributing to this confusion-matrix cell.
            </DialogDescription>
          </DialogHeader>
          <ul className="max-h-72 overflow-auto rounded-[var(--radius-sm)] border border-nx-muted bg-nx-bg p-3 font-mono text-[11px] text-nx-fg/85">
            {cellOpen && cellOpen.files.length > 0 ? (
              cellOpen.files.map((f) => (
                <li key={f} className="truncate py-0.5">
                  {f}
                </li>
              ))
            ) : (
              <li className="text-nx-fg/50">No files in this cell.</li>
            )}
          </ul>
        </DialogContent>
      </Dialog>
    </section>
  );
}
