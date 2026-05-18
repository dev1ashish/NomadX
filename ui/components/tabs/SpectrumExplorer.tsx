"use client";

/**
 * SpectrumExplorer (W3 + sidebar redesign + browser fixes).
 *
 * Class → Strain → File browser. Click anywhere cascades the selection.
 * URL: `?file=<file_id>` for cross-tab deep-links.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { motion } from "framer-motion";
import { ChevronRightIcon } from "lucide-react";

import { SpectrumViewer } from "@/components/plots/SpectrumViewer";
import { Switch } from "@/components/ui/switch";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { nxColors } from "@/lib/plotly-theme";
import type { ClassName } from "@/lib/types";
import { cn } from "@/lib/cn";

interface SpectrumIndexEntry {
  file_id: string;
  primary_class: ClassName;
  subclass: string | null;
}

interface SpectrumPayload {
  file_id: string;
  primary_class: ClassName;
  subclass: string | null;
  n_pixels: number;
  n_qc_pass: number;
  wn_raw: number[];
  wn_pp: number[];
  mean_raw: number[];
  mean_pp: number[];
}

const ANCHORS: { wn: number; chemistry: string }[] = [
  { wn: 1004, chemistry: "Phe ring breathing — aromatic AA, total protein proxy" },
  { wn: 1117, chemistry: "LPS chain region — empirical anchor 2 (d=+0.77 STEC↔Non-STEC)" },
  { wn: 1194, chemistry: "LPS chain anchor 1 — d=+1.03 STEC↔Non-STEC (project record raw single feature)" },
  { wn: 1242, chemistry: "Amide-III β-sheet" },
  { wn: 1338, chemistry: "CH₂ wag / adenine ring — Cisek-2013 STEC triple (NULL at file level, d=+0.13)" },
  { wn: 1454, chemistry: "CH₂ deformation, lipid — Cisek-2013 STEC triple (sign-reversed, d=−0.47)" },
  { wn: 1658, chemistry: "Amide-I α-helix/β-sheet — Cisek-2013 STEC triple (NULL, d=+0.16)" },
  { wn: 2900, chemistry: "C-H stretch — membrane lipid C-H modes" },
];

const CLASS_ORDER: ClassName[] = ["STEC", "Non-STEC", "Salmonella", "H2O"];

const CLASS_COLOR_VAR: Record<ClassName, string> = {
  STEC: nxColors.classStec,
  "Non-STEC": nxColors.classNonStec,
  Salmonella: nxColors.classSalm,
  H2O: nxColors.classH2o,
};

const CLASS_STRIPE_BG: Record<ClassName, string> = {
  STEC: "bg-class-stec",
  "Non-STEC": "bg-class-nonstec",
  Salmonella: "bg-class-salm",
  H2O: "bg-class-h2o",
};

const CLASS_TEXT: Record<ClassName, string> = {
  STEC: "text-class-stec",
  "Non-STEC": "text-class-nonstec",
  Salmonella: "text-class-salm",
  H2O: "text-class-h2o",
};

const CLASS_BLURB: Record<ClassName, string> = {
  STEC: "Shiga-toxin-producing E. coli",
  "Non-STEC": "Commensal / lab E. coli",
  Salmonella: "Salmonella enterica",
  H2O: "Water blanks",
};

const H2O_PSEUDO_STRAIN = "—";

export function SpectrumExplorer() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const urlFile = searchParams.get("file");

  const [index, setIndex] = useState<SpectrumIndexEntry[] | null>(null);
  const [indexError, setIndexError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [payload, setPayload] = useState<SpectrumPayload | null>(null);
  const [payloadError, setPayloadError] = useState<string | null>(null);
  const [showPreprocessed, setShowPreprocessed] = useState<boolean>(true);

  // Load index once.
  useEffect(() => {
    let cancelled = false;
    fetch("/data/spectra_index.json", { cache: "force-cache" })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<SpectrumIndexEntry[]>;
      })
      .then((j) => {
        if (cancelled) return;
        setIndex(j);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setIndexError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Pick a default file once the index lands.
  useEffect(() => {
    if (!index || selectedFile) return;
    if (urlFile && index.some((e) => e.file_id === urlFile)) {
      setSelectedFile(urlFile);
      return;
    }
    const firstStec = index.find((e) => e.primary_class === "STEC");
    setSelectedFile((firstStec ?? index[0]).file_id);
  }, [index, urlFile, selectedFile]);

  // Fetch payload whenever selectedFile changes.
  useEffect(() => {
    if (!selectedFile) return;
    let cancelled = false;
    setPayloadError(null);
    fetch(`/data/spectra/${selectedFile}.json`, { cache: "force-cache" })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<SpectrumPayload>;
      })
      .then((j) => {
        if (cancelled) return;
        setPayload(j);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setPayloadError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [selectedFile]);

  const updateFile = useCallback(
    (fileId: string) => {
      setSelectedFile(fileId);
      const params = new URLSearchParams(Array.from(searchParams.entries()));
      params.set("file", fileId);
      router.replace(`/spectrum?${params.toString()}`, { scroll: false });
    },
    [router, searchParams],
  );

  // ---- Derived: groupings + selection-path ------------------------------

  const byClass = useMemo(() => {
    const out: Record<ClassName, SpectrumIndexEntry[]> = {
      STEC: [], "Non-STEC": [], Salmonella: [], H2O: [],
    };
    if (!index) return out;
    for (const e of index) out[e.primary_class].push(e);
    for (const k of CLASS_ORDER) out[k].sort((a, b) => a.file_id.localeCompare(b.file_id));
    return out;
  }, [index]);

  const selectedEntry = useMemo(
    () => index?.find((e) => e.file_id === selectedFile) ?? null,
    [index, selectedFile],
  );

  // Derive the browse path from the selection — single source of truth.
  const currentClass: ClassName = selectedEntry?.primary_class ?? "STEC";
  const currentStrain: string =
    selectedEntry?.subclass ?? (currentClass === "H2O" ? H2O_PSEUDO_STRAIN : "");

  const strainsInClass = useMemo(() => {
    const entries = byClass[currentClass] ?? [];
    if (currentClass === "H2O") {
      return [{ key: H2O_PSEUDO_STRAIN, count: entries.length }];
    }
    const counts = new Map<string, number>();
    for (const e of entries) {
      const k = e.subclass ?? H2O_PSEUDO_STRAIN;
      counts.set(k, (counts.get(k) ?? 0) + 1);
    }
    return [...counts.entries()]
      .map(([key, count]) => ({ key, count }))
      .sort((a, b) => a.key.localeCompare(b.key));
  }, [byClass, currentClass]);

  const filesInStrain = useMemo(() => {
    const entries = byClass[currentClass] ?? [];
    if (currentClass === "H2O") return entries;
    if (!currentStrain) return [];
    return entries.filter(
      (e) => (e.subclass ?? H2O_PSEUDO_STRAIN) === currentStrain,
    );
  }, [byClass, currentClass, currentStrain]);

  // ---- Click handlers: each cascades down to a real file selection -------

  const pickClass = useCallback(
    (klass: ClassName) => {
      const first = byClass[klass]?.[0];
      if (first) updateFile(first.file_id);
    },
    [byClass, updateFile],
  );

  const pickStrain = useCallback(
    (klass: ClassName, strain: string) => {
      const first = (byClass[klass] ?? []).find(
        (e) => (e.subclass ?? H2O_PSEUDO_STRAIN) === strain,
      );
      if (first) updateFile(first.file_id);
    },
    [byClass, updateFile],
  );

  // ---- Render -----------------------------------------------------------

  const wn = payload ? (showPreprocessed ? payload.wn_pp : payload.wn_raw) : [];
  const intensity = payload ? (showPreprocessed ? payload.mean_pp : payload.mean_raw) : [];
  const traceColor = selectedEntry
    ? CLASS_COLOR_VAR[selectedEntry.primary_class]
    : nxColors.accent;

  return (
    <section className="px-8 lg:px-14 py-10">
      {/* Header */}
      <div className="flex flex-col gap-2 mb-8">
        <span className="font-mono text-[0.65rem] uppercase tracking-[0.22em] text-nx-accent">
          📈 Per-file mean spectrum
        </span>
        <h2 className="font-display text-[clamp(2rem,4vw,3.25rem)] leading-[1.05] text-nx-fg">
          Spectrum explorer
        </h2>
        <p className="text-nx-fg/55 max-w-2xl text-sm">
          Browse the 87 training files by Class → Strain → File. Click any
          row to cascade the selection down. Toggle the preprocessing
          pipeline against raw counts; hover the 8 cyan band anchors for
          chemistry context.
        </p>
      </div>

      {/* Browser */}
      <div className="grid grid-cols-3 rounded-md overflow-hidden border border-nx-muted/40 bg-nx-bg-elev-1/30">
        {/* Column 1: Class */}
        <BrowserColumn
          title="Class"
          subtitle={`${index?.length ?? "·"} files total`}
        >
          {CLASS_ORDER.map((cls) => {
            const files = byClass[cls];
            const active = currentClass === cls;
            return (
              <BrowserItem
                key={cls}
                active={active}
                layoutGroupId="spectrum-col-class"
                stripeBg={CLASS_STRIPE_BG[cls]}
                onClick={() => pickClass(cls)}
              >
                <div className="flex flex-col items-start gap-0 min-w-0">
                  <span className={cn("font-mono text-[0.74rem] font-semibold tracking-wide", CLASS_TEXT[cls])}>
                    {cls}
                  </span>
                  <span className="text-[10px] text-nx-fg/40 leading-tight truncate">
                    {CLASS_BLURB[cls]}
                  </span>
                </div>
                <span className="ml-auto font-mono text-[10px] text-nx-fg/35 tabular-nums">
                  {files.length}
                </span>
              </BrowserItem>
            );
          })}
        </BrowserColumn>

        {/* Column 2: Strain */}
        <BrowserColumn
          title="Strain"
          subtitle={
            currentClass === "H2O"
              ? "no subclass"
              : `${strainsInClass.length} strains in ${currentClass}`
          }
          divider
        >
          {strainsInClass.map((s) => {
            const active = currentStrain === s.key;
            return (
              <BrowserItem
                key={s.key}
                active={active}
                layoutGroupId="spectrum-col-strain"
                stripeBg={CLASS_STRIPE_BG[currentClass]}
                onClick={() => pickStrain(currentClass, s.key)}
              >
                <span className="font-mono text-[0.74rem] text-nx-fg/90 truncate">
                  {s.key}
                </span>
                <span className="ml-auto font-mono text-[10px] text-nx-fg/35 tabular-nums">
                  {s.count}
                </span>
              </BrowserItem>
            );
          })}
        </BrowserColumn>

        {/* Column 3: File */}
        <BrowserColumn
          title="File"
          subtitle={`${filesInStrain.length} files`}
          divider
        >
          {filesInStrain.map((f) => {
            const active = selectedFile === f.file_id;
            return (
              <BrowserItem
                key={f.file_id}
                active={active}
                layoutGroupId="spectrum-col-file"
                stripeBg={CLASS_STRIPE_BG[currentClass]}
                onClick={() => updateFile(f.file_id)}
              >
                <span className="font-mono text-[0.72rem] text-nx-fg/90 truncate min-w-0 flex-1">
                  {f.file_id}
                </span>
              </BrowserItem>
            );
          })}
        </BrowserColumn>
      </div>

      {/* Breadcrumb + toggle */}
      <div className="mt-5 flex flex-wrap items-center gap-4">
        {selectedEntry && (
          <Breadcrumb
            klass={selectedEntry.primary_class}
            strain={selectedEntry.subclass ?? H2O_PSEUDO_STRAIN}
            fileId={selectedEntry.file_id}
            qcInfo={payload ? `${payload.n_qc_pass}/${payload.n_pixels} px QC-passed` : null}
          />
        )}

        <div className="ml-auto flex items-center gap-3 rounded-md bg-nx-bg-elev-1/40 px-3 py-1.5 border border-nx-muted/40">
          <span className="font-mono text-[0.6rem] uppercase tracking-[0.18em] text-nx-fg/45">
            Spectrum
          </span>
          <label className="flex items-center gap-2 text-xs">
            <span className={cn(showPreprocessed ? "text-nx-fg/40" : "text-nx-fg", "font-mono")}>
              Raw
            </span>
            <Switch
              checked={showPreprocessed}
              onCheckedChange={(v) => setShowPreprocessed(Boolean(v))}
              aria-label="Toggle preprocessed vs raw"
            />
            <span className={cn(showPreprocessed ? "text-nx-fg" : "text-nx-fg/40", "font-mono")}>
              Preprocessed
            </span>
          </label>
        </div>
      </div>

      {indexError && (
        <p className="mt-3 text-xs text-nx-danger font-mono">
          Index load failed: {indexError}
        </p>
      )}

      {/* Plot */}
      <motion.div
        key={selectedFile ?? "empty"}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25 }}
        className="mt-5 h-[460px] w-full rounded-md border border-nx-muted/40 bg-nx-bg-elev-1/40"
      >
        {payload ? (
          <SpectrumViewer
            wn={wn}
            intensity={intensity}
            anchors={ANCHORS}
            traceColor={traceColor}
            yAxisLabel={
              showPreprocessed
                ? "Intensity (SNV-normalized)"
                : "Intensity (raw counts)"
            }
            traceName={payload.file_id}
          />
        ) : payloadError ? (
          <div className="flex h-full items-center justify-center px-6 text-center text-sm text-nx-danger">
            Spectrum load failed: {payloadError}
          </div>
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-nx-fg/60">
            Loading spectrum…
          </div>
        )}
      </motion.div>

      {/* Band-anchor legend */}
      <div className="mt-6">
        <h3 className="mb-3 font-mono text-[0.65rem] uppercase tracking-[0.18em] text-nx-fg/45">
          Band anchors · hover for chemistry
        </h3>
        <ul className="flex flex-wrap gap-2">
          {ANCHORS.map((a) => (
            <li key={a.wn}>
              <HoverCard>
                <HoverCardTrigger
                  render={
                    <button
                      type="button"
                      className="font-mono text-xs text-nx-accent rounded-sm border border-nx-accent/40 bg-nx-bg-elev-1/60 px-2.5 py-1 transition-all hover:bg-nx-bg-elev-2 hover:scale-105 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-nx-accent"
                    >
                      {a.wn} cm⁻¹
                    </button>
                  }
                />
                <HoverCardContent
                  className="w-72 bg-nx-bg-elev-1 ring-1 ring-nx-accent/40 text-nx-fg"
                  side="top"
                >
                  <div className="space-y-1">
                    <div className="font-mono text-xs text-nx-accent">
                      {a.wn} cm⁻¹
                    </div>
                    <div className="text-sm leading-snug">{a.chemistry}</div>
                  </div>
                </HoverCardContent>
              </HoverCard>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Browser sub-components
// ---------------------------------------------------------------------------

function BrowserColumn({
  title,
  subtitle,
  children,
  divider = false,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
  divider?: boolean;
}) {
  return (
    <div
      className={cn(
        "flex flex-col",
        divider && "border-l border-nx-muted/40",
      )}
    >
      <div className="px-4 py-3 border-b border-nx-muted/30 bg-nx-bg-elev-1/40">
        <div className="font-mono text-[0.6rem] uppercase tracking-[0.18em] text-nx-accent">
          {title}
        </div>
        <div className="font-mono text-[0.6rem] text-nx-fg/35 mt-0.5">
          {subtitle}
        </div>
      </div>
      <ul className="flex flex-col py-1 max-h-[320px] overflow-y-auto">
        {children}
      </ul>
    </div>
  );
}

function BrowserItem({
  active,
  layoutGroupId,
  stripeBg,
  onClick,
  children,
}: {
  active: boolean;
  layoutGroupId: string;
  stripeBg: string;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <li>
      <button
        type="button"
        onClick={onClick}
        className={cn(
          "group relative w-full flex items-center gap-2 pl-4 pr-3 py-1.5 text-left transition-colors focus-visible:outline-none focus-visible:bg-nx-bg-elev-2/50",
          active ? "bg-nx-bg-elev-2/40" : "hover:bg-nx-bg-elev-1/70",
        )}
      >
        {active ? (
          <motion.span
            layoutId={layoutGroupId}
            className={cn("absolute left-0 top-1 bottom-1 w-[2px] rounded-r-sm", stripeBg)}
            transition={{ type: "spring", stiffness: 420, damping: 36 }}
          />
        ) : (
          <span aria-hidden className="absolute left-0 top-1 bottom-1 w-[2px]" />
        )}
        {children}
        <ChevronRightIcon
          className={cn(
            "size-3 shrink-0 transition-all",
            active ? "text-nx-fg/60 translate-x-0" : "text-nx-fg/15 -translate-x-1 group-hover:translate-x-0 group-hover:text-nx-fg/40",
          )}
          strokeWidth={1.5}
        />
      </button>
    </li>
  );
}

function Breadcrumb({
  klass,
  strain,
  fileId,
  qcInfo,
}: {
  klass: ClassName;
  strain: string;
  fileId: string;
  qcInfo: string | null;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2 font-mono text-xs">
      <span className={cn("inline-block size-1.5 rounded-full", CLASS_STRIPE_BG[klass])} />
      <span className={cn("font-semibold tracking-wide", CLASS_TEXT[klass])}>
        {klass}
      </span>
      <ChevronRightIcon className="size-3 text-nx-fg/25" />
      <span className="text-nx-fg/70">{strain}</span>
      <ChevronRightIcon className="size-3 text-nx-fg/25" />
      <span className="text-nx-fg/95">{fileId}</span>
      {qcInfo && (
        <span className="text-nx-fg/40 ml-2 text-[0.65rem]">· {qcInfo}</span>
      )}
    </div>
  );
}
