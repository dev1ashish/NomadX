"use client";

/**
 * McrDemo (W6) — Atlas Raman MCR-ALS unmixing demo.
 *
 * Surfaces the K=7 pure-component spectra from the saved global MCR-ALS fit
 * (`artifacts/stage15f_mcr_global.joblib` → built into
 * `/data/mcr_components.json` by `scripts/build_mcr.py`) and lets the user:
 *
 *   1. Toggle each of the 7 pure components on/off via shadcn `<Switch>`.
 *   2. Pick one of 4 representative files (one per primary_class) from a
 *      shadcn `<Select>`.
 *   3. See the observed mean spectrum for that file overlaid against the
 *      reconstruction Σ (toggled-on pure spectra × per-file weights), plus
 *      the residual trace below.
 *
 * A prominent caveat card explains that the headline `mcr_C6_mean` d=-1.23
 * comes from a different K=8 fit (cached in `unmix_features.parquet`) and
 * that 0 MCR features survived Stage 15F per-fold MI selection.
 *
 * Plan ref: ULTRAPLAN.md §W6 + FINAL/PAPER.md §5.3 + §6.7.
 */

import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";

import { PureSpectraStack } from "@/components/plots/PureSpectraStack";
import { ReconstructionPlot } from "@/components/plots/ReconstructionPlot";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { getSidecar } from "@/lib/data";
import { nxColors } from "@/lib/plotly-theme";
import type { ClassName } from "@/lib/types";

interface McrComponentDef {
  k: number;
  label: string;
  spectrum: number[];
  global_d_stec_nonstec: number;
}

interface McrExample {
  primary_class: ClassName;
  file_id: string;
  observed_mean: number[];
  component_weights: number[];
}

interface McrComponentsSidecar {
  wn: number[];
  components: McrComponentDef[];
  per_class_mean_C: Record<ClassName, number[]>;
  examples: McrExample[];
  meta: {
    saved_K: number;
    paper_K: number;
    paper_headline_feature: string;
    paper_headline_d_stec_nonstec: number;
    saved_top_abs_k: number;
    note: string;
  };
}

/**
 * Colorway for the 7 components. We avoid the 4 class colors so the user
 * can't confuse a pure-component line with a class label. Cyan accent +
 * deep teal + neutral greys form the spine.
 */
const COMPONENT_COLORS: string[] = [
  "#39B8DC", // cyan accent
  "#7BC9E0", // cyan light
  "#135A6F", // accent deep
  "#9AA7AD", // grey-blue
  "#E1E6E8", // off-white
  "#F2A93B", // amber — used for the saved-top-abs |d| highlight
  "#7A3D99", // violet
];

const CLASS_ORDER: ClassName[] = ["STEC", "Non-STEC", "Salmonella", "H2O"];

const CLASS_COLOR_VAR: Record<ClassName, string> = {
  STEC: nxColors.classStec,
  "Non-STEC": nxColors.classNonStec,
  Salmonella: nxColors.classSalm,
  H2O: nxColors.classH2o,
};

const CLASS_BADGE_TW: Record<ClassName, string> = {
  STEC: "bg-class-stec/20 text-class-stec ring-1 ring-class-stec/40",
  "Non-STEC":
    "bg-class-nonstec/20 text-class-nonstec ring-1 ring-class-nonstec/40",
  Salmonella: "bg-class-salm/20 text-class-salm ring-1 ring-class-salm/40",
  H2O: "bg-class-h2o/20 text-class-h2o ring-1 ring-class-h2o/40",
};

export function McrDemo() {
  const [sidecar, setSidecar] = useState<McrComponentsSidecar | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [visible, setVisible] = useState<Record<number, boolean>>({});
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getSidecar<McrComponentsSidecar>("mcr_components.json")
      .then((j) => {
        if (cancelled) return;
        setSidecar(j);
        // Default: all components on, first example selected.
        const v: Record<number, boolean> = {};
        for (const c of j.components) v[c.k] = true;
        setVisible(v);
        if (j.examples.length > 0) {
          setSelectedFileId(j.examples[0].file_id);
        }
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const componentsWithColor = useMemo(() => {
    if (!sidecar) return [];
    return sidecar.components.map((c, i) => ({
      ...c,
      color: COMPONENT_COLORS[i % COMPONENT_COLORS.length],
    }));
  }, [sidecar]);

  const selectedExample = useMemo(() => {
    if (!sidecar || !selectedFileId) return null;
    return (
      sidecar.examples.find((e) => e.file_id === selectedFileId) ??
      sidecar.examples[0] ??
      null
    );
  }, [sidecar, selectedFileId]);

  if (error) {
    return (
      <section className="mx-auto max-w-screen-2xl px-6 py-10">
        <p className="font-mono text-sm text-nx-danger">
          MCR sidecar load failed: {error}
        </p>
      </section>
    );
  }

  if (!sidecar || componentsWithColor.length === 0) {
    return (
      <section className="mx-auto max-w-screen-2xl px-6 py-10">
        <div className="h-12 w-48 animate-pulse rounded-md bg-nx-bg-elev-1" />
        <div className="mt-8 grid gap-6 lg:grid-cols-2">
          <div className="h-[480px] animate-pulse rounded-md bg-nx-bg-elev-1" />
          <div className="h-[480px] animate-pulse rounded-md bg-nx-bg-elev-1" />
        </div>
      </section>
    );
  }

  const K = componentsWithColor.length;
  const visibleArr = componentsWithColor.map((c) => visible[c.k] !== false);
  const pureMat = componentsWithColor.map((c) => c.spectrum);
  const weights = selectedExample?.component_weights ?? new Array(K).fill(0);

  return (
    <section className="mx-auto max-w-screen-2xl px-6 py-8">
      <motion.h2
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="font-display text-nx-accent text-[3.0625rem] leading-[1.1]"
      >
        MCR-ALS
      </motion.h2>
      <motion.p
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, delay: 0.08, ease: "easeOut" }}
        className="mt-3 max-w-3xl text-nx-fg/60"
      >
        Multivariate Curve Resolution — Alternating Least Squares decomposes
        the 7,122-pixel × 987-bin preprocessed matrix into K=7 non-negative
        pure-component spectra and per-pixel concentrations. Toggle components
        below to see how a single representative file reconstructs from the
        weighted sum.
      </motion.p>

      {/* Caveat banner — required by W6 brief. */}
      <Card className="mt-6 border-l-4 border-l-nx-accent bg-nx-bg-elev-1 ring-1 ring-nx-accent/30">
        <CardHeader className="pb-2">
          <CardTitle className="flex flex-wrap items-baseline gap-2 font-mono text-sm text-nx-accent">
            <span>STAGE 15F CAVEAT</span>
            <span className="text-[10px] text-nx-fg/50">§6.7</span>
          </CardTitle>
          <CardDescription className="text-nx-fg/85 text-sm leading-relaxed">
            <strong className="text-nx-fg">
              MCR-ALS features did NOT survive per-fold MI selection in Stage
              15F.
            </strong>{" "}
            Global-fit d-values shown here are partly a leakage artifact; the
            production classifier uses{" "}
            <span className="font-mono text-nx-accent">0 MCR features</span>.
            The paper&apos;s{" "}
            <span className="font-mono">{sidecar.meta.paper_headline_feature}</span>{" "}
            d={sidecar.meta.paper_headline_d_stec_nonstec.toFixed(2)} is a
            global-fit project record from a separate K=
            {sidecar.meta.paper_K} fit cached in{" "}
            <span className="font-mono">unmix_features.parquet</span>, not a
            deployable signal. The saved K={sidecar.meta.saved_K} fit shown
            here has a different component ordering (top |d| ={" "}
            <span className="font-mono">
              C{sidecar.meta.saved_top_abs_k + 1}
            </span>
            ).
          </CardDescription>
        </CardHeader>
      </Card>

      {/* Top row: pure-spectra stack + switch column */}
      <div className="mt-8 grid gap-6 lg:grid-cols-[1fr_280px]">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, delay: 0.18, ease: "easeOut" }}
          className="h-[480px] rounded-[var(--radius-sm)] border border-nx-muted bg-nx-bg-elev-1 p-3"
        >
          <header className="mb-2 flex items-baseline justify-between px-1">
            <h3 className="font-mono text-xs uppercase tracking-[0.16em] text-nx-fg/70">
              K=7 pure spectra
            </h3>
            <span className="font-mono text-[10px] text-nx-fg/40">
              normalized · stacked
            </span>
          </header>
          <div className="h-[440px] w-full">
            <PureSpectraStack
              wn={sidecar.wn}
              components={componentsWithColor}
              visible={visible}
            />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, delay: 0.26, ease: "easeOut" }}
          className="rounded-[var(--radius-sm)] border border-nx-muted bg-nx-bg-elev-1 p-4"
        >
          <h3 className="mb-3 font-mono text-xs uppercase tracking-[0.16em] text-nx-fg/70">
            Components
          </h3>
          <ul role="list" className="space-y-3">
            {componentsWithColor.map((c) => {
              const on = visible[c.k] !== false;
              const isTopAbs = c.k === sidecar.meta.saved_top_abs_k;
              return (
                <li
                  key={c.k}
                  className="flex items-start gap-3"
                >
                  <Switch
                    checked={on}
                    onCheckedChange={(v) =>
                      setVisible((prev) => ({
                        ...prev,
                        [c.k]: Boolean(v),
                      }))
                    }
                    aria-label={`Toggle C${c.k + 1}`}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-baseline gap-2">
                      <span
                        className="font-mono text-sm"
                        style={{ color: on ? c.color : nxColors.muted }}
                      >
                        C{c.k + 1}
                      </span>
                      {isTopAbs ? (
                        <Badge className="bg-nx-accent/15 text-nx-accent ring-1 ring-nx-accent/40 font-mono uppercase text-[10px] tracking-[0.12em]">
                          top |d|
                        </Badge>
                      ) : null}
                    </div>
                    <p className="mt-0.5 text-[11px] text-nx-fg/65 leading-snug">
                      d STEC↔Non-STEC ={" "}
                      <span className="font-mono text-nx-fg/85">
                        {c.global_d_stec_nonstec >= 0 ? "+" : ""}
                        {c.global_d_stec_nonstec.toFixed(2)}
                      </span>
                    </p>
                  </div>
                </li>
              );
            })}
          </ul>
        </motion.div>
      </div>

      {/* Reconstruction row */}
      <div className="mt-8">
        <div className="flex flex-wrap items-end justify-between gap-4 border-b border-nx-muted pb-4">
          <div>
            <h3 className="font-display text-xl text-nx-fg">Reconstruction</h3>
            <p className="mt-1 max-w-2xl text-sm text-nx-fg/60">
              Observed mean (white) vs. Σ (toggled-on pure × per-file weight,
              cyan), with residual below. The reconstruction is offset to share
              the observed&apos;s baseline so shape mismatches are visually
              comparable.
            </p>
          </div>
          <div className="flex flex-col gap-2">
            <label
              htmlFor="mcr-file-select"
              className="text-xs uppercase tracking-[0.16em] text-nx-fg/60"
            >
              Example file
            </label>
            <Select
              value={selectedFileId ?? undefined}
              onValueChange={(v) => {
                if (typeof v === "string") setSelectedFileId(v);
              }}
            >
              <SelectTrigger
                id="mcr-file-select"
                className="w-[300px] bg-nx-bg-elev-1 border-nx-muted text-nx-fg"
              >
                <SelectValue placeholder="Pick a class…" />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  <SelectLabel className="text-nx-fg/60">
                    One representative file per class
                  </SelectLabel>
                  {CLASS_ORDER.map((cls) => {
                    const ex = sidecar.examples.find(
                      (e) => e.primary_class === cls,
                    );
                    if (!ex) return null;
                    return (
                      <SelectItem key={ex.file_id} value={ex.file_id}>
                        <span
                          aria-hidden
                          className="inline-block size-2 rounded-full"
                          style={{
                            backgroundColor: CLASS_COLOR_VAR[ex.primary_class],
                          }}
                        />
                        <span className="font-mono">{ex.file_id}</span>
                      </SelectItem>
                    );
                  })}
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>
        </div>

        {selectedExample ? (
          <motion.div
            key={selectedExample.file_id}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, ease: "easeOut" }}
            className="mt-4 flex items-center gap-3"
          >
            <Badge
              className={
                CLASS_BADGE_TW[selectedExample.primary_class] +
                " font-mono uppercase"
              }
            >
              {selectedExample.primary_class}
            </Badge>
            <span className="font-mono text-sm text-nx-fg/85">
              {selectedExample.file_id}
            </span>
            <span className="font-mono text-[11px] text-nx-fg/50">
              weights = [
              {selectedExample.component_weights
                .map((w) => w.toFixed(1))
                .join(", ")}
              ]
            </span>
          </motion.div>
        ) : null}

        <div className="mt-4 h-[520px] w-full rounded-[var(--radius-sm)] border border-nx-muted bg-nx-bg-elev-1 p-3">
          {selectedExample ? (
            <ReconstructionPlot
              wn={sidecar.wn}
              observed={selectedExample.observed_mean}
              componentSpectra={pureMat}
              componentWeights={weights}
              visible={visibleArr}
            />
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-nx-fg/60">
              Pick an example file to view the reconstruction.
            </div>
          )}
        </div>
      </div>

      {/* Per-class mean concentrations */}
      <div className="mt-8 grid gap-4 lg:grid-cols-4">
        {CLASS_ORDER.map((cls) => {
          const means = sidecar.per_class_mean_C[cls] ?? [];
          return (
            <Card
              key={cls}
              className="bg-nx-bg-elev-1 ring-1 ring-nx-muted"
              size="sm"
            >
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2">
                  <span
                    aria-hidden
                    className="inline-block size-2 rounded-full"
                    style={{ backgroundColor: CLASS_COLOR_VAR[cls] }}
                  />
                  <span className="font-mono text-xs uppercase tracking-[0.12em] text-nx-fg/80">
                    {cls}
                  </span>
                </CardTitle>
                <CardDescription className="text-[10px] text-nx-fg/50">
                  mean C across files
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-1">
                <ul className="space-y-1 font-mono text-[11px] text-nx-fg/80">
                  {means.map((m, i) => (
                    <li
                      key={i}
                      className="flex justify-between gap-2"
                      style={{ color: COMPONENT_COLORS[i] }}
                    >
                      <span>C{i + 1}</span>
                      <span className="text-nx-fg/80">{m.toFixed(2)}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </section>
  );
}
