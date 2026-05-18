"use client";

/**
 * Live inference — landing hero (W7 + redesign).
 *
 * `/` now redirects here. The page is the hero, not a tab. Big drop zone,
 * class-colored sample chips, and an animated result reveal.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Toaster } from "sonner";
import { toast } from "sonner";
import {
  UploadCloudIcon,
  Loader2Icon,
  FileTextIcon,
  SparklesIcon,
  ArrowDownIcon,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { predict } from "@/lib/modal-client";
import type { ClassName, FileMeta, PredictionResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

import { LiveProbabilityBars } from "@/components/plots/LiveProbabilityBars";
import { LiveMeanSpectrum } from "@/components/plots/LiveMeanSpectrum";

// ---------------------------------------------------------------------------
// Local types + constants
// ---------------------------------------------------------------------------

const CLASS_ORDER: ClassName[] = ["STEC", "Non-STEC", "Salmonella", "H2O"];

const CLASS_BG: Record<ClassName, string> = {
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

const CLASS_BORDER: Record<ClassName, string> = {
  STEC: "border-class-stec/40 hover:border-class-stec",
  "Non-STEC": "border-class-nonstec/40 hover:border-class-nonstec",
  Salmonella: "border-class-salm/40 hover:border-class-salm",
  H2O: "border-class-h2o/40 hover:border-class-h2o",
};

const CLASS_BLURB: Record<ClassName, string> = {
  STEC: "Shiga-toxin-producing E. coli — pathogenic serogroups.",
  "Non-STEC": "Commensal / lab E. coli — same species, no Stx phage.",
  Salmonella: "Salmonella enterica — serovars Dublin, Heidelburg, Typhimurium.",
  H2O: "Water blanks — uniform substrate baseline.",
};

interface InventoryEnvelope {
  totals?: unknown;
  files?: FileMeta[];
}

async function fetchInventoryFiles(): Promise<FileMeta[]> {
  try {
    const res = await fetch("/data/inventory.json", { cache: "force-cache" });
    if (!res.ok) return [];
    const json = (await res.json()) as FileMeta[] | InventoryEnvelope;
    if (Array.isArray(json)) return json;
    return json.files ?? [];
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

type Status =
  | { kind: "idle" }
  | { kind: "loading"; filename: string }
  | { kind: "result"; filename: string; data: PredictionResponse }
  | { kind: "error"; message: string };

export function LiveInference() {
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [dragActive, setDragActive] = useState(false);
  const [corpus, setCorpus] = useState<FileMeta[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const modalConfigured =
    typeof process !== "undefined" &&
    typeof process.env?.NEXT_PUBLIC_MODAL_PREDICT_URL === "string" &&
    process.env.NEXT_PUBLIC_MODAL_PREDICT_URL.length > 0;

  useEffect(() => {
    let mounted = true;
    void fetchInventoryFiles().then((files) => {
      if (mounted) setCorpus(files);
    });
    return () => {
      mounted = false;
    };
  }, []);

  const runPrediction = useCallback(async (file: File) => {
    setStatus({ kind: "loading", filename: file.name });
    try {
      const data = await predict(file);
      setStatus({ kind: "result", filename: file.name, data });
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Modal endpoint unreachable — check NEXT_PUBLIC_MODAL_PREDICT_URL";
      const isMissingEnv = /NEXT_PUBLIC_MODAL_PREDICT_URL/.test(message);
      const display = isMissingEnv
        ? "Modal endpoint not configured — run `modal deploy` and add the URL to ui/.env.local"
        : message;
      toast.error("Inference failed", { description: display });
      setStatus({ kind: "error", message: display });
    }
  }, []);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const file = files[0];
      const lower = file.name.toLowerCase();
      if (!lower.endsWith(".xls") && !lower.endsWith(".txt")) {
        toast.error("Unsupported file type", {
          description: "Drop an .xls or .txt Raman file from your Atlas Data/ folder.",
        });
        return;
      }
      void runPrediction(file);
    },
    [runPrediction],
  );

  const onDrop = useCallback(
    (ev: React.DragEvent<HTMLDivElement>) => {
      ev.preventDefault();
      setDragActive(false);
      handleFiles(ev.dataTransfer.files);
    },
    [handleFiles],
  );

  const onDragOver = useCallback((ev: React.DragEvent<HTMLDivElement>) => {
    ev.preventDefault();
    setDragActive(true);
  }, []);

  const onDragLeave = useCallback((ev: React.DragEvent<HTMLDivElement>) => {
    ev.preventDefault();
    setDragActive(false);
  }, []);

  const grouped = useMemo(() => {
    const out: Record<ClassName, FileMeta[]> = {
      STEC: [],
      "Non-STEC": [],
      Salmonella: [],
      H2O: [],
    };
    for (const f of corpus) {
      const klass = f.primary_class;
      if (klass in out) out[klass].push(f);
    }
    for (const k of CLASS_ORDER) {
      out[k].sort((a, b) => a.file_id.localeCompare(b.file_id));
    }
    return out;
  }, [corpus]);

  const hasResult = status.kind === "result";

  return (
    <section className="relative">
      <Toaster position="bottom-right" richColors closeButton />

      {/* HERO */}
      <div className="px-8 lg:px-14 pt-14 pb-8 max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: "easeOut" }}
        >
          <div className="flex items-center gap-2 mb-4">
            <span className="font-mono text-[0.65rem] uppercase tracking-[0.22em] text-nx-accent">
              ⚡ Live classification
            </span>
            <Separator orientation="vertical" className="h-3 bg-nx-muted" />
            <span className="font-mono text-[0.65rem] uppercase tracking-[0.22em] text-nx-fg/45">
              Stage 15F · LogReg-L2 · 35 features
            </span>
          </div>
          <h1 className="font-display text-[clamp(2.5rem,5.5vw,4.75rem)] leading-[1.02] tracking-tight text-nx-fg">
            Classify a Raman map
            <br />
            <span className="text-nx-accent">in five seconds.</span>
          </h1>
          <p className="mt-6 max-w-2xl text-nx-fg/60 text-lg leading-relaxed">
            Drop one tab-delimited{" "}
            <code className="font-mono text-nx-accent">.xls</code> Raman file
            from your <code className="font-mono text-nx-fg/80">Atlas Data/</code>{" "}
            folder. The Modal endpoint runs the frozen preprocessing pipeline +
            259-feature extractor + MI-selected LogReg model and returns the
            probability over four classes.
          </p>

          {/* KPI proof bar */}
          <div className="mt-8 grid grid-cols-2 sm:grid-cols-4 gap-px bg-nx-muted/40 rounded-md overflow-hidden max-w-3xl">
            <KpiCell label="Model" value="LogReg-L2" caption="Stage 15F · 35 MI features" />
            <KpiCell label="Training data" value="87 files" caption="7,122 spectra · 9 strains + H₂O" />
            <KpiCell label="Cold start" value="3–8 s" caption="warm: ~1 s" />
            <KpiCell label="Returns" value="4 probs" caption="STEC · Non-STEC · Salm · H₂O" />
          </div>
        </motion.div>
      </div>

      {/* Modal-not-configured banner — surfaces before a click */}
      {!modalConfigured && (
        <div className="px-8 lg:px-14 pb-6">
          <div className="rounded-md border border-amber-500/40 bg-amber-950/20 px-5 py-4 flex flex-col sm:flex-row gap-3 sm:items-center">
            <span className="size-2 rounded-full bg-amber-400/90 shrink-0 mt-1 sm:mt-0" />
            <div className="flex-1 min-w-0">
              <div className="font-mono text-[0.8rem] text-amber-300/95 mb-1">
                Modal endpoint not deployed yet
              </div>
              <div className="text-[0.78rem] text-nx-fg/65 leading-relaxed">
                Inference requires the Modal Python endpoint to be running.
                From the repo root:{" "}
                <code className="font-mono text-nx-accent">cd inference_api && uv venv && source .venv/bin/activate && uv pip install modal && modal token new && modal deploy modal_app.py</code>
                . Modal prints a URL — paste it into{" "}
                <code className="font-mono text-nx-accent">ui/.env.local</code> as{" "}
                <code className="font-mono text-nx-accent">NEXT_PUBLIC_MODAL_PREDICT_URL</code>
                {" "}then restart the dev server.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* DROP ZONE — full width, taller, more prominent */}
      <div className="px-8 lg:px-14 pb-8">
        <DropZone
          active={dragActive}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onPick={() => fileInputRef.current?.click()}
          inputRef={fileInputRef}
          onChange={(e) => handleFiles(e.target.files)}
          status={status}
        />
      </div>

      {/* CLASS SAMPLE CHIPS — by-class drill-down via dialog */}
      {!hasResult && (
        <div className="px-8 lg:px-14 pb-10">
          <div className="flex items-center gap-2 mb-4">
            <ArrowDownIcon className="size-3.5 text-nx-fg/40" />
            <span className="font-mono text-[0.65rem] uppercase tracking-[0.22em] text-nx-fg/45">
              The model was trained on these
            </span>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {CLASS_ORDER.map((klass, idx) => {
              const files = grouped[klass] ?? [];
              return (
                <motion.div
                  key={klass}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.05 * idx, duration: 0.35, ease: "easeOut" }}
                >
                  <ClassSampleCard
                    klass={klass}
                    files={files}
                    total={corpus.length}
                  />
                </motion.div>
              );
            })}
          </div>
        </div>
      )}

      {/* RESULTS */}
      <AnimatePresence mode="wait">
        {hasResult && status.kind === "result" && (
          <motion.div
            key={status.filename}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 16 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
            className="px-8 lg:px-14 pb-16 flex flex-col gap-6"
          >
            <ResultBanner
              predicted={status.data.class}
              probabilities={status.data.probabilities}
              filename={status.filename}
            />

            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle className="font-mono text-nx-accent text-sm uppercase tracking-wider">
                    Class probabilities
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <LiveProbabilityBars
                    key={`${status.filename}-bars`}
                    probabilities={status.data.probabilities}
                    predicted={status.data.class}
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="font-mono text-nx-accent text-sm uppercase tracking-wider">
                    Mean preprocessed spectrum
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <LiveMeanSpectrum
                    key={`${status.filename}-spec`}
                    wn={status.data.wn}
                    spectrum={status.data.spectrum_mean}
                  />
                </CardContent>
              </Card>
            </div>

            <FeatureContributionTable
              featureValues={status.data.feature_values}
            />
          </motion.div>
        )}

        {status.kind === "error" && (
          <motion.div
            key="error"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="px-8 lg:px-14 pb-16"
          >
            <Card className="border-amber-500/40 bg-amber-950/10">
              <CardHeader>
                <CardTitle className="text-amber-400 font-mono uppercase tracking-wider text-sm">
                  Inference failed
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="font-mono text-sm text-nx-fg/80">
                  {status.message}
                </p>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}

// ---------------------------------------------------------------------------
// KPI cell
// ---------------------------------------------------------------------------

function KpiCell({
  label,
  value,
  caption,
}: {
  label: string;
  value: string;
  caption: string;
}) {
  return (
    <div className="flex flex-col gap-1 bg-nx-bg-elev-1/60 px-4 py-3 hover:bg-nx-bg-elev-2/40 transition-colors">
      <span className="font-mono text-[0.6rem] uppercase tracking-[0.18em] text-nx-fg/45">
        {label}
      </span>
      <span className="font-display text-xl text-nx-fg leading-tight tabular-nums">
        {value}
      </span>
      <span className="font-mono text-[0.65rem] text-nx-fg/40 leading-tight">
        {caption}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Drop zone
// ---------------------------------------------------------------------------

interface DropZoneProps {
  active: boolean;
  status: Status;
  onDrop: (ev: React.DragEvent<HTMLDivElement>) => void;
  onDragOver: (ev: React.DragEvent<HTMLDivElement>) => void;
  onDragLeave: (ev: React.DragEvent<HTMLDivElement>) => void;
  onPick: () => void;
  onChange: (ev: React.ChangeEvent<HTMLInputElement>) => void;
  inputRef: React.MutableRefObject<HTMLInputElement | null>;
}

const FILE_INPUT_ID = "atlas-live-file-input";

function DropZone({
  active,
  status,
  onDrop,
  onDragOver,
  onDragLeave,
  onChange,
  inputRef,
}: DropZoneProps) {
  const loading = status.kind === "loading";

  return (
    <div
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      className={cn(
        "group relative flex min-h-[300px] flex-col items-center justify-center gap-5 rounded-lg border-2 border-dashed px-8 py-12 transition-all overflow-hidden",
        active
          ? "border-nx-accent bg-nx-accent/5 scale-[1.005]"
          : "border-nx-muted/70 bg-nx-bg-elev-1/40 hover:border-nx-accent/60 hover:bg-nx-bg-elev-1/70",
      )}
    >
      <div
        aria-hidden
        className={cn(
          "absolute inset-0 pointer-events-none transition-opacity duration-500",
          active ? "opacity-100" : "opacity-0 group-hover:opacity-60",
        )}
        style={{
          background:
            "radial-gradient(circle at 50% 0%, rgba(57,184,220,0.10), transparent 70%)",
        }}
      />

      {/* Native file input — sr-only (Tailwind's visually-hidden util).
          Pair with `<label htmlFor>` below; clicking the label opens the
          native picker via the DOM association. */}
      <input
        ref={inputRef}
        id={FILE_INPUT_ID}
        type="file"
        accept=".xls,.txt"
        onChange={onChange}
        onClick={(e) => {
          // Allow re-picking the same file (Chrome won't fire change twice
          // for an identical filename otherwise).
          (e.target as HTMLInputElement).value = "";
        }}
        className="sr-only"
      />

      {loading ? (
        <div className="relative flex flex-col items-center gap-4 text-center">
          <Loader2Icon className="size-10 animate-spin text-nx-accent" />
          <div className="font-display text-xl text-nx-fg">
            Calling Modal endpoint…
          </div>
          <div className="font-mono text-xs text-nx-fg/50">
            cold start 3–8 s · warm ~1 s · {status.filename}
          </div>
        </div>
      ) : (
        <div className="relative flex flex-col items-center gap-5 text-center">
          <motion.div
            animate={active ? { y: -4 } : { y: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 18 }}
          >
            <UploadCloudIcon
              className={cn(
                "size-12 transition-colors",
                active ? "text-nx-accent" : "text-nx-fg/70 group-hover:text-nx-accent",
              )}
              strokeWidth={1.5}
            />
          </motion.div>
          <div className="flex flex-col gap-2">
            <div className="font-display text-2xl text-nx-fg">
              Drop a Raman map here
            </div>
            <div className="font-mono text-xs text-nx-fg/55 tracking-wide">
              .xls or .txt · 70-720 pixel rows · auto-detects format
            </div>
          </div>
          {/* Label-for pattern: clicking the label always opens the native
              picker, no JS click-forwarding needed. */}
          <label
            htmlFor={FILE_INPUT_ID}
            className="inline-flex items-center justify-center cursor-pointer rounded-md bg-nx-accent text-nx-bg hover:bg-nx-accent/90 font-medium px-6 py-2.5 text-base transition-colors focus-within:ring-2 focus-within:ring-nx-accent focus-within:ring-offset-2 focus-within:ring-offset-nx-bg"
          >
            Browse file…
          </label>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Class sample card (with drill-down dialog)
// ---------------------------------------------------------------------------

function ClassSampleCard({
  klass,
  files,
  total,
}: {
  klass: ClassName;
  files: FileMeta[];
  total: number;
}) {
  const pct = total > 0 ? Math.round((files.length / total) * 100) : 0;

  return (
    <Dialog>
      <DialogTrigger
        className={cn(
          "group w-full text-left rounded-md border bg-nx-bg-elev-1/40 px-4 py-4 transition-all hover:bg-nx-bg-elev-1/80 hover:translate-y-[-2px]",
          CLASS_BORDER[klass],
        )}
      >
        <div className="flex items-center gap-2 mb-2">
          <span
            className={cn("inline-block size-2 rounded-full", CLASS_BG[klass])}
          />
          <span
            className={cn(
              "font-mono text-xs font-semibold tracking-wide",
              CLASS_TEXT[klass],
            )}
          >
            {klass}
          </span>
          <span className="ml-auto font-mono text-[0.65rem] text-nx-fg/40">
            {files.length} files · {pct}%
          </span>
        </div>
        <p className="text-xs text-nx-fg/55 leading-relaxed group-hover:text-nx-fg/75 transition-colors">
          {CLASS_BLURB[klass]}
        </p>
        <div className="mt-3 flex items-center gap-1 font-mono text-[0.6rem] text-nx-fg/30 group-hover:text-nx-accent transition-colors">
          <FileTextIcon className="size-2.5" /> browse files →
        </div>
      </DialogTrigger>
      <DialogContent className="max-w-lg max-h-[70vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className={cn("font-display flex items-center gap-2", CLASS_TEXT[klass])}>
            <span
              className={cn("inline-block size-3 rounded-full", CLASS_BG[klass])}
            />
            {klass}
            <Badge variant="outline" className="font-mono ml-auto">
              {files.length}
            </Badge>
          </DialogTitle>
          <p className="text-sm text-nx-fg/60">{CLASS_BLURB[klass]}</p>
        </DialogHeader>
        <p className="text-xs text-nx-fg/50">
          File names are training-set references. Drag the actual file from
          your local <code className="font-mono">Atlas Data/{klass}/</code>{" "}
          folder to classify it.
        </p>
        <ul className="flex flex-col gap-0.5 mt-2">
          {files.map((f) => (
            <li
              key={f.file_id}
              className="font-mono text-[11px] text-nx-fg/80 px-2 py-1 rounded hover:bg-nx-bg-elev-2/40"
            >
              <span className="text-nx-fg/95">{f.file_id}</span>
              {f.subclass && (
                <span className="text-nx-fg/40"> · {f.subclass}</span>
              )}
              <span className="text-nx-fg/30 float-right">
                {f.n_pixels} px
              </span>
            </li>
          ))}
        </ul>
      </DialogContent>
    </Dialog>
  );
}

// ---------------------------------------------------------------------------
// Result banner
// ---------------------------------------------------------------------------

interface ResultBannerProps {
  predicted: ClassName;
  probabilities: Record<ClassName, number>;
  filename: string;
}

function ResultBanner({
  predicted,
  probabilities,
  filename,
}: ResultBannerProps) {
  const topProb = probabilities[predicted] ?? 0;
  return (
    <motion.div
      initial={{ scale: 0.94, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      className={cn(
        "relative flex flex-col gap-3 overflow-hidden rounded-lg px-10 py-12 text-white shadow-2xl",
        CLASS_BG[predicted],
      )}
    >
      <div
        aria-hidden
        className="absolute inset-0 bg-gradient-to-br from-white/15 via-transparent to-black/30 pointer-events-none"
      />
      <div className="relative font-mono text-[0.7rem] uppercase tracking-[0.24em] text-white/85 flex items-center gap-2">
        <SparklesIcon className="size-3" /> Prediction
      </div>
      <div className="relative flex flex-wrap items-baseline gap-5">
        <motion.span
          initial={{ y: 10, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1, duration: 0.4 }}
          className="font-display text-[clamp(3rem,6vw,4.5rem)] leading-none"
        >
          {predicted}
        </motion.span>
        <motion.span
          initial={{ y: 10, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.4 }}
          className="font-mono text-3xl text-white/90 tabular-nums"
        >
          {(topProb * 100).toFixed(1)}%
        </motion.span>
      </div>
      <div className="relative font-mono text-xs text-white/80 mt-1">
        {filename}
      </div>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Feature contribution table
// ---------------------------------------------------------------------------

interface FeatureContributionTableProps {
  featureValues: Record<string, number>;
}

function FeatureContributionTable({
  featureValues,
}: FeatureContributionTableProps) {
  const rows = useMemo(() => {
    const entries = Object.entries(featureValues);
    entries.sort(([, a], [, b]) => Math.abs(b) - Math.abs(a));
    return entries;
  }, [featureValues]);

  const max = rows.length ? Math.abs(rows[0][1]) : 1;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="font-mono text-nx-accent text-sm uppercase tracking-wider">
          Feature values · {rows.length} features
        </CardTitle>
        <p className="text-xs text-nx-fg/50">
          Stage 15F MI-selected features for this file, post-scaling. Bar
          magnitude is relative to the strongest feature in this single
          prediction.
        </p>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-[1fr_auto_2fr] items-center gap-x-4 gap-y-1.5 font-mono text-xs">
          <div className="text-nx-fg/40 uppercase tracking-wider">Feature</div>
          <div className="text-right text-nx-fg/40 uppercase tracking-wider">
            Value
          </div>
          <div className="text-nx-fg/40 uppercase tracking-wider">
            Magnitude
          </div>
          {rows.map(([name, value], idx) => {
            const pct = max > 0 ? (Math.abs(value) / max) * 100 : 0;
            const positive = value >= 0;
            return (
              <FeatureRow
                key={name}
                name={name}
                value={value}
                pct={pct}
                positive={positive}
                idx={idx}
              />
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

interface FeatureRowProps {
  name: string;
  value: number;
  pct: number;
  positive: boolean;
  idx: number;
}

function FeatureRow({ name, value, pct, positive, idx }: FeatureRowProps) {
  return (
    <>
      <div className="text-nx-fg/85 truncate" title={name}>
        {name}
      </div>
      <div
        className={cn(
          "text-right tabular-nums",
          positive ? "text-nx-accent" : "text-class-stec",
        )}
      >
        {value >= 0 ? "+" : ""}
        {value.toFixed(3)}
      </div>
      <div className="h-2 w-full rounded-sm bg-nx-bg-elev-2 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ delay: 0.02 * idx, duration: 0.5, ease: "easeOut" }}
          className={cn(
            "h-full rounded-sm",
            positive ? "bg-nx-accent" : "bg-class-stec",
          )}
        />
      </div>
    </>
  );
}
