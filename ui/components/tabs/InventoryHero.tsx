"use client";

/**
 * Inventory tab (W2) — hero composition.
 *
 * - KPI strip: 87 files · 7,122 spectra · 987 bins · PLS-DA LOSO 0.603
 *   (re-uses `KpiStrip` from `components/layout/`).
 * - Plotly stacked bar (class composition) + grouped bar (per-strain
 *   files & QC-passed pixels). Both via `<PlotlyChart>` (SSR-safe).
 * - Strain chips below the second plot expose a shadcn `<HoverCard>` per
 *   strain showing the file_ids in that strain (W2 brief).
 *
 * Plan ref: §3 ASCII mockup + §4 W2.
 */
import { Suspense, lazy, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { KpiStrip, type Kpi } from "@/components/layout/KpiStrip";
import { InventoryClassBar } from "@/components/plots/InventoryClassBar";
import {
  InventoryStrainBar,
  strainBreakdown,
  type Strain,
} from "@/components/plots/InventoryStrainBar";
import type { InventoryPcaSidecar } from "@/components/plots/InventoryFeatureSpace3D";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import {
  fetchInventorySidecar,
  type InventorySidecar,
} from "@/components/tabs/inventory-data";
import type { ClassName } from "@/lib/types";

// Plotly's scatter3d bundle is heavy; defer the entire 3D scatter component
// until the user scrolls/needs it. `React.lazy` + Suspense is fine inside
// this client component (we're already "use client" at the top).
const InventoryFeatureSpace3D = lazy(
  () => import("@/components/plots/InventoryFeatureSpace3D"),
);

const CLASS_SWATCH: Record<ClassName, string> = {
  STEC: "bg-class-stec",
  "Non-STEC": "bg-class-nonstec",
  Salmonella: "bg-class-salm",
  H2O: "bg-class-h2o",
};

function fmt(n: number): string {
  return n.toLocaleString();
}

function buildKpis(sidecar: InventorySidecar): Kpi[] {
  const t = sidecar.totals;
  return [
    { label: "Files", value: fmt(t.n_files) },
    { label: "Spectra", value: fmt(t.n_spectra) },
    { label: "Bins", value: fmt(t.n_bins) },
    { label: "PLS-DA LOSO", value: t.plsda_loso.toFixed(3) },
  ];
}

export function InventoryHero() {
  const [sidecar, setSidecar] = useState<InventorySidecar | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pcaSidecar, setPcaSidecar] = useState<InventoryPcaSidecar | null>(
    null,
  );

  useEffect(() => {
    let cancelled = false;
    fetchInventorySidecar()
      .then((d) => {
        if (!cancelled) setSidecar(d);
      })
      .catch((e: unknown) => {
        if (!cancelled)
          setError(e instanceof Error ? e.message : "Failed to load inventory");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const strains: Strain[] = useMemo(
    () => (sidecar ? strainBreakdown(sidecar.files) : []),
    [sidecar],
  );

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
        Inventory
      </motion.h2>
      <motion.p
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, delay: 0.08, ease: "easeOut" }}
        className="mt-3 max-w-2xl text-nx-fg/60"
      >
        87 single-cell Raman maps across four classes — STEC, Non-STEC,
        Salmonella, H2O — preprocessed to 987 bins and QC-filtered to 7,122
        clean spectra (Stage 15F corpus).
      </motion.p>

      {sidecar ? (
        <KpiStrip items={buildKpis(sidecar)} />
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
          className="rounded-[var(--radius-sm)] border border-nx-muted bg-nx-bg-elev-1 p-4"
        >
          <header className="mb-3 flex items-baseline justify-between">
            <h3 className="font-mono text-xs uppercase tracking-[0.16em] text-nx-fg/70">
              Class composition
            </h3>
            <span className="font-mono text-[10px] text-nx-fg/40">
              files · stacked by strain
            </span>
          </header>
          {sidecar ? (
            <InventoryClassBar files={sidecar.files} />
          ) : (
            <div className="h-[360px] animate-pulse rounded-[var(--radius-sm)] bg-nx-bg-elev-2" />
          )}
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, delay: 0.26, ease: "easeOut" }}
          className="rounded-[var(--radius-sm)] border border-nx-muted bg-nx-bg-elev-1 p-4"
        >
          <header className="mb-3 flex items-baseline justify-between">
            <h3 className="font-mono text-xs uppercase tracking-[0.16em] text-nx-fg/70">
              Per-strain breakdown
            </h3>
            <span className="font-mono text-[10px] text-nx-fg/40">
              files vs QC-passed pixels
            </span>
          </header>
          {sidecar ? (
            <InventoryStrainBar files={sidecar.files} />
          ) : (
            <div className="h-[360px] animate-pulse rounded-[var(--radius-sm)] bg-nx-bg-elev-2" />
          )}
        </motion.div>
      </div>

      {sidecar && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.34, ease: "easeOut" }}
          className="mt-8"
        >
          <h3 className="font-mono text-xs uppercase tracking-[0.16em] text-nx-fg/70">
            Strain detail · hover for file ids
          </h3>
          <ul role="list" className="mt-3 flex flex-wrap gap-2">
            {strains.map((s) => (
              <li key={`${s.primary_class}-${s.label}`}>
                <HoverCard>
                  <HoverCardTrigger
                    aria-label={`${s.primary_class} ${s.label} — ${s.files} files`}
                    className="inline-flex cursor-default items-center gap-2 border border-nx-muted bg-nx-bg-elev-1 px-3 py-1.5 text-[11px] font-mono uppercase tracking-[0.12em] text-nx-fg/85 transition-colors hover:bg-nx-bg-elev-2"
                  >
                    <span
                      aria-hidden
                      className={`inline-block size-2 ${CLASS_SWATCH[s.primary_class]}`}
                    />
                    <span>{s.label}</span>
                    <span className="text-nx-fg/50">·</span>
                    <span>{s.files}</span>
                  </HoverCardTrigger>
                  <HoverCardContent
                    side="top"
                    className="w-72 border border-nx-muted bg-nx-bg-elev-1 text-nx-fg"
                  >
                    <header className="mb-2 flex items-center gap-2">
                      <span
                        aria-hidden
                        className={`inline-block size-2 ${CLASS_SWATCH[s.primary_class]}`}
                      />
                      <span className="font-mono text-[11px] uppercase tracking-[0.16em] text-nx-fg/80">
                        {s.primary_class} · {s.label}
                      </span>
                    </header>
                    <p className="mb-2 text-[11px] text-nx-fg/60 font-mono">
                      {s.files} files · {Math.round(s.qc_pixels).toLocaleString()} QC-passed pixels
                    </p>
                    <ul className="max-h-48 space-y-0.5 overflow-auto font-mono text-[11px] text-nx-fg/85">
                      {s.file_ids.map((fid) => (
                        <li key={fid} className="truncate">
                          <Link
                            href={`/spectrum?file=${encodeURIComponent(fid)}`}
                            className="hover:text-nx-accent hover:underline"
                          >
                            {fid}
                          </Link>
                        </li>
                      ))}
                    </ul>
                  </HoverCardContent>
                </HoverCard>
              </li>
            ))}
          </ul>
        </motion.div>
      )}

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, delay: 0.42, ease: "easeOut" }}
        className="mt-10 rounded-[var(--radius-sm)] border border-nx-muted bg-nx-bg-elev-1 p-4"
      >
        <header className="mb-3 flex items-baseline justify-between">
          <h3 className="font-mono text-xs uppercase tracking-[0.16em] text-nx-fg/70">
            Feature space (PCA of 259 engineered features)
          </h3>
          <span className="font-mono text-[10px] text-nx-fg/40">
            87 files · standardized · 3 PCs
          </span>
        </header>
        <Suspense
          fallback={
            <div className="h-[500px] w-full animate-pulse rounded-[var(--radius-sm)] bg-nx-bg-elev-2" />
          }
        >
          <InventoryFeatureSpace3D onLoaded={setPcaSidecar} />
        </Suspense>
        <p className="mt-3 max-w-3xl text-[12px] font-mono text-nx-fg/60">
          {pcaSidecar
            ? `PCA explains ${Math.round(
                pcaSidecar.variance_explained.reduce((a, b) => a + b, 0) *
                  100,
              )}% of variance across ${pcaSidecar.n_features} features. Hover a point for file detail. Drag to rotate.`
            : "Loading PCA projection — drag to rotate once it renders."}
        </p>
      </motion.div>
    </section>
  );
}
