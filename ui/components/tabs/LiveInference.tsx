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
import { Separator } from "@/components/ui/separator";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { predict, predictPlsda } from "@/lib/modal-client";
import type {
  ClassName,
  FileMeta,
  ModelName,
  PredictionResponse,
} from "@/lib/types";
import { cn } from "@/lib/utils";

import { LiveProbabilityBars } from "@/components/plots/LiveProbabilityBars";
import { LiveMeanSpectrum } from "@/components/plots/LiveMeanSpectrum";
import { ConvertPanel } from "@/components/tabs/ConvertPanel";

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

interface ModelOutcome {
  data?: PredictionResponse;
  error?: string;
  loading: boolean;
}

type Status =
  | { kind: "idle" }
  | {
      kind: "loading";
      filename: string;
      logreg: ModelOutcome;
      plsda: ModelOutcome;
    }
  | {
      kind: "result";
      filename: string;
      logreg: ModelOutcome;
      plsda: ModelOutcome;
    }
  | { kind: "error"; message: string };

const MODEL_LABEL: Record<ModelName, string> = {
  logreg_stage15f: "LogReg-L2 · 35 engineered features",
  plsda_raw: "PLS-DA · raw 987-bin spectrum (project headline)",
};

const MODEL_SHORT: Record<ModelName, string> = {
  logreg_stage15f: "LogReg-L2",
  plsda_raw: "PLS-DA",
};

export function LiveInference() {
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [dragActive, setDragActive] = useState(false);
  const [corpus, setCorpus] = useState<FileMeta[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const logregConfigured =
    typeof process !== "undefined" &&
    typeof process.env?.NEXT_PUBLIC_MODAL_PREDICT_URL === "string" &&
    process.env.NEXT_PUBLIC_MODAL_PREDICT_URL.length > 0;
  const plsdaConfigured =
    typeof process !== "undefined" &&
    typeof process.env?.NEXT_PUBLIC_MODAL_PREDICT_PLSDA_URL === "string" &&
    process.env.NEXT_PUBLIC_MODAL_PREDICT_PLSDA_URL.length > 0;
  const modalConfigured = logregConfigured && plsdaConfigured;

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
    const filename = file.name;
    setStatus({
      kind: "loading",
      filename,
      logreg: { loading: true },
      plsda: { loading: true },
    });

    const runOne = async (
      fn: (f: File) => Promise<PredictionResponse>,
      key: "logreg" | "plsda",
    ) => {
      try {
        const data = await fn(file);
        setStatus((prev) => {
          if (prev.kind !== "loading" && prev.kind !== "result") return prev;
          if (prev.filename !== filename) return prev;
          const next = { ...prev, [key]: { data, loading: false } } as Status;
          return next;
        });
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Modal endpoint unreachable";
        setStatus((prev) => {
          if (prev.kind !== "loading" && prev.kind !== "result") return prev;
          if (prev.filename !== filename) return prev;
          const next = {
            ...prev,
            [key]: { error: message, loading: false },
          } as Status;
          return next;
        });
        toast.error(`${key === "logreg" ? "LogReg" : "PLS-DA"} failed`, {
          description: message,
        });
      }
    };

    await Promise.all([runOne(predict, "logreg"), runOne(predictPlsda, "plsda")]);

    // Promote loading → result once both calls settle.
    setStatus((prev) => {
      if (prev.kind !== "loading") return prev;
      if (prev.filename !== filename) return prev;
      return { ...prev, kind: "result" } as Status;
    });
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

  const hasAnyResult =
    (status.kind === "loading" || status.kind === "result") &&
    Boolean(status.logreg.data || status.plsda.data);
  const logregData =
    status.kind === "loading" || status.kind === "result"
      ? status.logreg.data
      : undefined;
  const plsdaData =
    status.kind === "loading" || status.kind === "result"
      ? status.plsda.data
      : undefined;
  const logregLoading =
    (status.kind === "loading" || status.kind === "result") &&
    status.logreg.loading;
  const plsdaLoading =
    (status.kind === "loading" || status.kind === "result") &&
    status.plsda.loading;
  const modelsDisagree =
    logregData && plsdaData && logregData.class !== plsdaData.class;

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
              LogReg-L2 (35 features) + PLS-DA (raw spectrum) · side-by-side
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
            folder. Both models run in parallel on the same file and their
            verdicts, probabilities and explanations line up side by side. A raw
            export straight off the instrument needs converting first — the
            panel below does that in your browser.
          </p>

          {/* KPI proof bar */}
          <div className="mt-8 grid grid-cols-2 sm:grid-cols-4 gap-px bg-nx-muted/40 rounded-md overflow-hidden max-w-3xl">
            <KpiCell label="Models" value="2 in parallel" caption="LogReg-L2 + PLS-DA-raw" />
            <KpiCell label="Training data" value="87 files" caption="7,122 spectra · 9 strains + H₂O" />
            <KpiCell label="Cold start" value="3–8 s" caption="warm: ~1 s" />
            <KpiCell label="LOSO record" value="0.603" caption="PLS-DA file-weighted balanced acc" />
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
                {logregConfigured || plsdaConfigured
                  ? "Only one Modal endpoint is configured"
                  : "Modal endpoints not deployed yet"}
              </div>
              <div className="text-[0.78rem] text-nx-fg/65 leading-relaxed">
                The Live tab calls two endpoints in parallel:{" "}
                <code className="font-mono text-nx-accent">
                  NEXT_PUBLIC_MODAL_PREDICT_URL
                </code>{" "}
                (LogReg) and{" "}
                <code className="font-mono text-nx-accent">
                  NEXT_PUBLIC_MODAL_PREDICT_PLSDA_URL
                </code>{" "}
                (PLS-DA). Deploy from{" "}
                <code className="font-mono text-nx-accent">inference_api/</code>{" "}
                and paste both URLs into{" "}
                <code className="font-mono text-nx-accent">ui/.env.local</code>{" "}
                (see <code className="font-mono">.env.example</code>), then
                restart the dev server.
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

      {/* RAW FILE CONVERTER — resample an instrument export onto the canonical
          axis so the endpoints can parse it. Always mounted so a converted
          file survives a prediction round-trip. */}
      <ConvertPanel
        onAnalyze={(file) => void runPrediction(file)}
        busy={status.kind === "loading"}
      />

      {/* SUGGESTED DEMO FILES — surface the most informative cases */}
      {!hasAnyResult && status.kind !== "loading" && <DemoFilesPanel />}

      {/* CLASS SAMPLE CHIPS — by-class drill-down via dialog */}
      {!hasAnyResult && status.kind !== "loading" && (
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
        {hasAnyResult && (status.kind === "result" || status.kind === "loading") && (
          <motion.div
            key={status.filename}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 16 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
            className="px-8 lg:px-14 pb-16 flex flex-col gap-6"
          >
            {modelsDisagree && (
              <DisagreementBadge
                logregClass={logregData!.class}
                plsdaClass={plsdaData!.class}
              />
            )}

            <FileStrip
              filename={status.filename}
              data={logregData ?? plsdaData}
            />

            <ComparisonColumns
              filename={status.filename}
              logreg={{
                modelName: "logreg_stage15f",
                data: logregData,
                loading: logregLoading,
                error:
                  status.kind === "loading" || status.kind === "result"
                    ? status.logreg.error
                    : undefined,
              }}
              plsda={{
                modelName: "plsda_raw",
                data: plsdaData,
                loading: plsdaLoading,
                error:
                  status.kind === "loading" || status.kind === "result"
                    ? status.plsda.error
                    : undefined,
              }}
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
              .xls or .txt · already on the 2048-bin canonical axis
            </div>
            <div className="font-mono text-[0.68rem] text-nx-fg/35 tracking-wide">
              raw export straight off the instrument? convert it below first
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
  modelName: ModelName;
  abstain?: boolean;
}

/**
 * Verdict card, sized for a half-width column. The filename moved out to
 * `FileStrip` (it's the same file for both models) and the long model label
 * moved up to the column header, so this carries only what differs between
 * the two columns: the class and its probability.
 */
function ResultBanner({
  predicted,
  probabilities,
  modelName,
  abstain,
}: ResultBannerProps) {
  const topProb = probabilities[predicted] ?? 0;
  return (
    <motion.div
      initial={{ scale: 0.96, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      className={cn(
        "relative flex h-full flex-col gap-2 overflow-hidden rounded-lg px-7 py-7 text-white shadow-2xl",
        CLASS_BG[predicted],
      )}
    >
      <div
        aria-hidden
        className="absolute inset-0 bg-gradient-to-br from-white/15 via-transparent to-black/30 pointer-events-none"
      />
      <div className="relative flex flex-wrap items-center gap-2 font-mono text-[0.62rem] uppercase tracking-[0.24em] text-white/85">
        <SparklesIcon className="size-3" />
        <span>{MODEL_SHORT[modelName]}</span>
        {abstain && (
          <Badge
            variant="outline"
            className="ml-1 border-white/40 bg-white/10 font-mono text-[0.58rem] uppercase text-white/95"
          >
            low confidence · abstain
          </Badge>
        )}
      </div>
      <motion.span
        initial={{ y: 8, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.1, duration: 0.4 }}
        className="relative font-display text-[clamp(1.9rem,3.2vw,2.9rem)] leading-[1.05] break-words"
      >
        {predicted}
      </motion.span>
      <motion.span
        initial={{ y: 8, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.2, duration: 0.4 }}
        className="relative font-mono text-2xl tabular-nums text-white/90"
      >
        {(topProb * 100).toFixed(1)}%
      </motion.span>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Disagreement badge — when the two models pick different classes
// ---------------------------------------------------------------------------

interface DisagreementBadgeProps {
  logregClass: ClassName;
  plsdaClass: ClassName;
}

function DisagreementBadge({ logregClass, plsdaClass }: DisagreementBadgeProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="flex flex-wrap items-center gap-3 rounded-md border border-amber-500/50 bg-amber-950/25 px-5 py-3"
    >
      <span className="font-mono text-[0.65rem] uppercase tracking-[0.22em] text-amber-300/95">
        ⚠ Models disagree
      </span>
      <Separator orientation="vertical" className="h-3 bg-amber-500/40" />
      <span className="font-mono text-xs text-nx-fg/85">
        LogReg →{" "}
        <span className={cn("font-semibold", CLASS_TEXT[logregClass])}>
          {logregClass}
        </span>
        <span className="text-nx-fg/40 mx-3">·</span>
        PLS-DA →{" "}
        <span className={cn("font-semibold", CLASS_TEXT[plsdaClass])}>
          {plsdaClass}
        </span>
      </span>
      <span className="ml-auto font-mono text-[0.65rem] text-nx-fg/55 max-w-md leading-relaxed">
        Two models, two feature spaces. Disagreement = the file lives in
        the gap between engineered features and the raw spectrum.
      </span>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Per-model result block (banner + prob bars + spectrum + details)
// ---------------------------------------------------------------------------

interface ModelSlot {
  modelName: ModelName;
  data?: PredictionResponse;
  loading: boolean;
  error?: string;
}

/**
 * One comparison row: the same panel type for both models, side by side.
 *
 * Each row is its own grid rather than all cells sharing one, so a row's two
 * cells always align with each other, an absent cell can't shift the ones
 * after it, and the collapse to a single column on narrow screens keeps
 * related panels adjacent.
 */
function PairedRow({
  left,
  right,
}: {
  left: React.ReactNode;
  right: React.ReactNode;
}) {
  return (
    <div className="grid items-stretch gap-6 lg:grid-cols-2">
      {left}
      {right}
    </div>
  );
}

/** The file both columns describe — stated once, above the comparison. */
function FileStrip({
  filename,
  data,
}: {
  filename: string;
  data?: PredictionResponse;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded-md border border-nx-muted/50 bg-nx-bg-elev-1/40 px-5 py-3">
      <FileTextIcon className="size-3.5 shrink-0 text-nx-fg/40" />
      <span className="font-mono text-xs text-nx-fg/90 break-all">
        {filename}
      </span>
      {data?.n_pixels_used != null && (
        <span className="font-mono text-[0.65rem] text-nx-fg/40">
          {data.n_pixels_used} of {data.n_pixels_input ?? data.n_pixels_used}{" "}
          pixel rows used
        </span>
      )}
      <span className="ml-auto font-mono text-[0.62rem] uppercase tracking-[0.2em] text-nx-fg/35">
        two models · one file
      </span>
    </div>
  );
}

/** Column header naming the model that owns the cells beneath it. */
function ColumnHeader({ modelName }: { modelName: ModelName }) {
  return (
    <div className="flex flex-col gap-0.5 border-b border-nx-muted/50 pb-2">
      <span className="font-display text-lg leading-tight text-nx-fg">
        {MODEL_SHORT[modelName]}
      </span>
      <span className="font-mono text-[0.65rem] text-nx-fg/45">
        {MODEL_LABEL[modelName]}
      </span>
    </div>
  );
}

/**
 * Wraps a cell so every state renders at the same place in its row. A model
 * that is still loading or has failed shows that inline instead of collapsing
 * the row and pulling the other column out of alignment.
 */
function Cell({
  slot,
  title,
  children,
}: {
  slot: ModelSlot;
  title?: string;
  children: (data: PredictionResponse) => React.ReactNode;
}) {
  if (slot.data) {
    if (!title) return <>{children(slot.data)}</>;
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="font-mono text-sm uppercase tracking-wider text-nx-accent">
            {title}
          </CardTitle>
        </CardHeader>
        <CardContent>{children(slot.data)}</CardContent>
      </Card>
    );
  }
  if (slot.error) {
    return (
      <Card className="h-full border-amber-500/40 bg-amber-950/10">
        <CardHeader>
          <CardTitle className="font-mono text-sm uppercase tracking-wider text-amber-400">
            {MODEL_SHORT[slot.modelName]} failed
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="font-mono text-xs break-all text-nx-fg/80">
            {slot.error}
          </p>
        </CardContent>
      </Card>
    );
  }
  return (
    <Card className="h-full border-nx-muted/40 bg-nx-bg-elev-1/40">
      <CardContent className="flex items-center gap-3 py-6">
        <Loader2Icon className="size-4 animate-spin text-nx-accent" />
        <span className="font-mono text-xs text-nx-fg/65">
          Waiting on {MODEL_SHORT[slot.modelName]}…
        </span>
      </CardContent>
    </Card>
  );
}

/** Model-specific explanation panel — differs by model, so it has no shared title. */
function DetailCell({ slot }: { slot: ModelSlot }) {
  const { data, modelName } = slot;
  if (!data) return <div />;

  if (modelName === "logreg_stage15f") {
    if (Object.keys(data.feature_values).length === 0) return <div />;
    return <FeatureContributionTable featureValues={data.feature_values} />;
  }

  if (!data.contribution_for_predicted || !data.wn) return <div />;
  return (
    <SpectralDriversPanel
      wn={data.wn}
      contribution={data.contribution_for_predicted}
      predicted={data.class}
    />
  );
}

function ComparisonColumns({
  filename,
  logreg,
  plsda,
}: {
  filename: string;
  logreg: ModelSlot;
  plsda: ModelSlot;
}) {
  const verdict = (slot: ModelSlot) => (
    <Cell slot={slot}>
      {(data) => (
        <ResultBanner
          predicted={data.class}
          probabilities={data.probabilities}
          modelName={slot.modelName}
          abstain={data.abstain}
        />
      )}
    </Cell>
  );

  const probabilities = (slot: ModelSlot) => (
    <Cell slot={slot} title="Class probabilities">
      {(data) => (
        <LiveProbabilityBars
          key={`${filename}-${slot.modelName}-bars`}
          probabilities={data.probabilities}
          predicted={data.class}
        />
      )}
    </Cell>
  );

  const spectrum = (slot: ModelSlot) => (
    <Cell slot={slot} title="Mean preprocessed spectrum">
      {(data) => (
        <LiveMeanSpectrum
          key={`${filename}-${slot.modelName}-spec`}
          wn={data.wn}
          spectrum={data.spectrum_mean}
        />
      )}
    </Cell>
  );

  return (
    <div className="flex flex-col gap-6">
      <PairedRow
        left={<ColumnHeader modelName={logreg.modelName} />}
        right={<ColumnHeader modelName={plsda.modelName} />}
      />
      <PairedRow left={verdict(logreg)} right={verdict(plsda)} />
      <PairedRow
        left={probabilities(logreg)}
        right={probabilities(plsda)}
      />
      <PairedRow left={spectrum(logreg)} right={spectrum(plsda)} />
      <PairedRow
        left={<DetailCell slot={logreg} />}
        right={<DetailCell slot={plsda} />}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// PLS-DA spectral drivers — top wavenumber contributions cross-referenced
// with bands.json for plain-English chemistry labels
// ---------------------------------------------------------------------------

interface BandsGroup {
  key: string;
  label: string;
  biology: string;
  bands: { name: string; center: number; chemistry: string }[];
}

interface BandsJson {
  groups: BandsGroup[];
}

interface BandHit {
  center: number;
  chemistry: string;
  group: string;
  distance: number;
}

function nearestBand(
  wnTarget: number,
  flatBands: { center: number; chemistry: string; group: string }[],
  tolerance = 25,
): BandHit | null {
  let best: BandHit | null = null;
  for (const b of flatBands) {
    const d = Math.abs(b.center - wnTarget);
    if (d <= tolerance && (best == null || d < best.distance)) {
      best = { center: b.center, chemistry: b.chemistry, group: b.group, distance: d };
    }
  }
  return best;
}

interface SpectralDriversPanelProps {
  wn: number[];
  contribution: number[];
  predicted: ClassName;
}

function SpectralDriversPanel({
  wn,
  contribution,
  predicted,
}: SpectralDriversPanelProps) {
  const [bands, setBands] = useState<BandsJson | null>(null);

  useEffect(() => {
    let mounted = true;
    fetch("/data/bands.json", { cache: "force-cache" })
      .then((r) => (r.ok ? r.json() : null))
      .then((j: BandsJson | null) => {
        if (mounted && j) setBands(j);
      })
      .catch(() => {});
    return () => {
      mounted = false;
    };
  }, []);

  const flatBands = useMemo(() => {
    if (!bands) return [];
    return bands.groups.flatMap((g) =>
      g.bands.map((b) => ({ ...b, group: g.label })),
    );
  }, [bands]);

  const drivers = useMemo(() => {
    if (wn.length !== contribution.length) return [];
    const indexed = contribution.map((c, i) => ({ i, c, wn: wn[i] }));
    indexed.sort((a, b) => Math.abs(b.c) - Math.abs(a.c));
    const top = indexed.slice(0, 10);
    return top.map((row) => ({
      wn: row.wn,
      contribution: row.c,
      band: flatBands.length > 0 ? nearestBand(row.wn, flatBands) : null,
    }));
  }, [wn, contribution, flatBands]);

  const maxAbs = drivers.length > 0 ? Math.abs(drivers[0].contribution) : 1;
  const totalContrib = useMemo(
    () => contribution.reduce((a, b) => a + b, 0),
    [contribution],
  );

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="font-mono text-nx-accent text-sm uppercase tracking-wider">
          Spectral drivers · PLS-DA
        </CardTitle>
        <p className="text-xs text-nx-fg/55 leading-relaxed">
          Top wavenumber bins that pushed the prediction toward{" "}
          <span className={cn("font-semibold", CLASS_TEXT[predicted])}>
            {predicted}
          </span>
          . Each row is one of the 987 preprocessed bins; sign tells you
          whether the bin pushed toward (+) or away (−) from the predicted
          class. Sum across all 987 bins ≈ log-odds (
          <span className="font-mono tabular-nums text-nx-fg/80">
            {totalContrib >= 0 ? "+" : ""}
            {totalContrib.toFixed(2)}
          </span>
          ).
        </p>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-[auto_auto_1fr_auto] items-center gap-x-4 gap-y-2 font-mono text-xs">
          <div className="text-nx-fg/40 uppercase tracking-wider">
            Wn (cm⁻¹)
          </div>
          <div className="text-nx-fg/40 uppercase tracking-wider">Chem.</div>
          <div className="text-nx-fg/40 uppercase tracking-wider">
            Push toward {predicted}
          </div>
          <div className="text-right text-nx-fg/40 uppercase tracking-wider">
            Δ log-odds
          </div>
          {drivers.map((d, idx) => (
            <DriverRow
              key={`${d.wn}-${idx}`}
              wn={d.wn}
              chemistry={
                d.band ? d.band.chemistry : "no named band within 25 cm⁻¹"
              }
              contribution={d.contribution}
              pct={(Math.abs(d.contribution) / maxAbs) * 100}
              idx={idx}
            />
          ))}
        </div>
      </CardContent>
    </Card>
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
    <Card className="h-full">
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

interface DriverRowProps {
  wn: number;
  chemistry: string;
  contribution: number;
  pct: number;
  idx: number;
}

function DriverRow({ wn, chemistry, contribution, pct, idx }: DriverRowProps) {
  const positive = contribution >= 0;
  return (
    <>
      <div className="text-nx-fg/95 tabular-nums">{wn.toFixed(1)}</div>
      <div className="text-nx-fg/70 truncate max-w-[220px]" title={chemistry}>
        {chemistry}
      </div>
      <div className="h-2 w-full rounded-sm bg-nx-bg-elev-2 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ delay: 0.02 * idx, duration: 0.45, ease: "easeOut" }}
          className={cn(
            "h-full rounded-sm",
            positive ? "bg-nx-accent" : "bg-class-stec",
          )}
        />
      </div>
      <div
        className={cn(
          "text-right tabular-nums",
          positive ? "text-nx-accent" : "text-class-stec",
        )}
      >
        {positive ? "+" : ""}
        {contribution.toFixed(3)}
      </div>
    </>
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

// ---------------------------------------------------------------------------
// Suggested demo files — the 5 most informative drops for the dual-model UI
// ---------------------------------------------------------------------------

interface DemoFile {
  file_id: string;
  path: string;            // path relative to Atlas Data/
  klass: ClassName;
  tag: "clean" | "mosaic" | "blank";
  note: string;            // one-line "what to expect"
}

const DEMO_FILES: DemoFile[] = [
  {
    file_id: "R357_100_10000ms_260226",
    path: "Non STEC/ATCC25922/",
    klass: "Non-STEC",
    tag: "clean",
    note: "Clean ATCC25922 — both models agree Non-STEC ~0.77",
  },
  {
    file_id: "R372_100_10000ms_260306",
    path: "H20/",
    klass: "H2O",
    tag: "blank",
    note: "Water blank — both models confident H₂O (0.88 / 0.98)",
  },
  {
    file_id: "R397_100_10000ms_260310",
    path: "STEC/",
    klass: "STEC",
    tag: "clean",
    note: "Clean STEC — both correct, LogReg 0.89 / PLS-DA strong",
  },
  {
    file_id: "R364_100_10000ms_260305",
    path: "STEC/O157H7/",
    klass: "STEC",
    tag: "mosaic",
    note: "Mosaic O157:H7 — LogReg says Salmonella 0.94 ✗, PLS-DA says STEC 0.55 ✓",
  },
  {
    file_id: "R370_100_10000ms_260305",
    path: "Salmonella/Dublin/",
    klass: "Salmonella",
    tag: "mosaic",
    note: "Mosaic Dublin — LogReg says STEC ✗, PLS-DA says Salmonella 0.55 ✓",
  },
];

const TAG_COPY: Record<DemoFile["tag"], { label: string; cls: string }> = {
  clean: {
    label: "clean",
    cls: "bg-nx-accent/10 text-nx-accent border-nx-accent/30",
  },
  mosaic: {
    label: "models disagree",
    cls: "bg-amber-500/10 text-amber-300 border-amber-500/40",
  },
  blank: {
    label: "blank",
    cls: "bg-nx-fg/10 text-nx-fg/65 border-nx-fg/20",
  },
};

function DemoFilesPanel() {
  return (
    <div className="px-8 lg:px-14 pb-10">
      <div className="flex items-center gap-2 mb-4">
        <SparklesIcon className="size-3.5 text-nx-accent" />
        <span className="font-mono text-[0.65rem] uppercase tracking-[0.22em] text-nx-fg/55">
          Try one of these — most informative drops
        </span>
      </div>
      <p className="text-xs text-nx-fg/50 mb-4 max-w-3xl leading-relaxed">
        Drag these <code className="font-mono text-nx-fg/75">.xls</code> files
        from your local <code className="font-mono text-nx-fg/75">Atlas Data/</code>{" "}
        folder into the drop zone. The two mosaic files are where the models
        disagree — the deployed LogReg fails and the project-headline PLS-DA
        rescues it. That&apos;s the demo moment.
      </p>
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-3">
        {DEMO_FILES.map((d, idx) => (
          <motion.div
            key={d.file_id}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.04 * idx, duration: 0.35, ease: "easeOut" }}
            className={cn(
              "group flex flex-col gap-2 rounded-md border bg-nx-bg-elev-1/40 px-4 py-3 transition-colors hover:bg-nx-bg-elev-1/80",
              d.tag === "mosaic"
                ? "border-amber-500/30 hover:border-amber-500/60"
                : "border-nx-muted/60 hover:border-nx-accent/50",
            )}
          >
            <div className="flex items-center gap-2 flex-wrap">
              <span
                className={cn("inline-block size-2 rounded-full", CLASS_BG[d.klass])}
              />
              <span
                className={cn(
                  "font-mono text-xs font-semibold tracking-wide",
                  CLASS_TEXT[d.klass],
                )}
              >
                {d.klass}
              </span>
              <Badge
                variant="outline"
                className={cn(
                  "font-mono text-[0.6rem] uppercase tracking-wide border",
                  TAG_COPY[d.tag].cls,
                )}
              >
                {TAG_COPY[d.tag].label}
              </Badge>
            </div>
            <div className="font-mono text-[0.78rem] text-nx-fg/90 truncate" title={d.file_id}>
              {d.file_id}.xls
            </div>
            <div className="font-mono text-[0.65rem] text-nx-fg/40 truncate">
              Atlas Data/{d.path}
            </div>
            <div className="text-[0.72rem] text-nx-fg/65 leading-relaxed">
              {d.note}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
