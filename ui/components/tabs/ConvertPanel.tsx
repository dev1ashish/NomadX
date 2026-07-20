"use client";

/**
 * Raw instrument file → canonical Atlas 2048-bin converter.
 *
 * Files straight off the XploRA (everything under `all-txt-data/`) are rejected
 * by the endpoints: `atlas/io.py` requires 2048 intensities per pixel row on
 * `linspace(76, 3499, 2048)`, and these exports carry 552–959 points over a
 * much narrower range. `lib/atlas-convert.ts` resamples them so they parse.
 *
 * The important part of this panel is not the conversion — it's the coverage
 * report. Resampling makes a file *parse*; it does not make the prediction
 * *trustworthy*. Wavenumbers the instrument never measured are edge-clamped
 * flat, and for every file in this corpus the entire C-H stretch window
 * (2800–3050 cm⁻¹) is fabricated that way. So when coverage is short, Analyze
 * stays locked behind an explicit acknowledgement.
 */
import { useCallback, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  FileInputIcon,
  Loader2Icon,
  DownloadIcon,
  PlayIcon,
  AlertTriangleIcon,
  CheckCircle2Icon,
  XCircleIcon,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button, buttonVariants } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import {
  convertToAtlas,
  formatBytes,
  ConversionError,
  N_BINS,
  PIXEL_CAP,
  FINGERPRINT,
  CH_STRETCH,
  type ConversionStats,
} from "@/lib/atlas-convert";

const FILE_INPUT_ID = "atlas-convert-file-input";

type State =
  | { kind: "idle" }
  | { kind: "converting"; name: string; progress: number }
  | {
      kind: "done";
      name: string;
      file: File;
      stats: ConversionStats;
      url: string;
    }
  | { kind: "error"; name: string; message: string };

interface ConvertPanelProps {
  /** Hands the converted file to the Live tab's prediction flow. */
  onAnalyze: (file: File) => void;
  /** True while a prediction is in flight. */
  busy?: boolean;
}

export function ConvertPanel({ onAnalyze, busy }: ConvertPanelProps) {
  const [state, setState] = useState<State>({ kind: "idle" });
  const [acknowledged, setAcknowledged] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const urlRef = useRef<string | null>(null);

  const handleFile = useCallback(async (file: File) => {
    if (urlRef.current) {
      URL.revokeObjectURL(urlRef.current);
      urlRef.current = null;
    }
    setAcknowledged(false);
    setState({ kind: "converting", name: file.name, progress: 0 });

    try {
      const { file: converted, stats } = await convertToAtlas(file, {
        onProgress: (progress) =>
          setState((prev) =>
            prev.kind === "converting" && prev.name === file.name
              ? { ...prev, progress }
              : prev,
          ),
      });
      const url = URL.createObjectURL(converted);
      urlRef.current = url;
      setState({ kind: "done", name: file.name, file: converted, stats, url });
    } catch (err) {
      setState({
        kind: "error",
        name: file.name,
        message:
          err instanceof ConversionError
            ? err.message
            : err instanceof Error
              ? err.message
              : "conversion failed",
      });
    }
  }, []);

  const onPick = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      void handleFile(files[0]);
    },
    [handleFile],
  );

  const stats = state.kind === "done" ? state.stats : null;
  const short = stats
    ? stats.fingerprintCov < 100 || stats.chCov < 100
    : false;
  const canAnalyze = state.kind === "done" && (!short || acknowledged) && !busy;

  return (
    <div className="px-8 lg:px-14 pb-10">
      <div className="flex items-center gap-2 mb-4">
        <FileInputIcon className="size-3.5 text-nx-accent" />
        <span className="font-mono text-[0.65rem] uppercase tracking-[0.22em] text-nx-fg/55">
          Convert a raw instrument file
        </span>
      </div>

      <Card className="border-nx-muted/60 bg-nx-bg-elev-1/40">
        <CardHeader>
          <CardTitle className="font-mono text-nx-accent text-sm uppercase tracking-wider">
            Raw XploRA export → Atlas {N_BINS}-bin
          </CardTitle>
          <p className="text-xs text-nx-fg/55 leading-relaxed max-w-3xl">
            The models only read files carrying {N_BINS} intensities per pixel
            row on a fixed 76–3499 cm⁻¹ axis. A raw export from the instrument
            has a different point count over a narrower range, so it is{" "}
            <span className="text-nx-fg/85">not ingestable as-is</span> — the
            endpoint rejects every pixel row. Drop one here to resample it onto
            the canonical axis, then analyze.
          </p>
        </CardHeader>

        <CardContent className="flex flex-col gap-5">
          {/* Compact drop area */}
          <div
            onDrop={(ev) => {
              ev.preventDefault();
              setDragActive(false);
              onPick(ev.dataTransfer.files);
            }}
            onDragOver={(ev) => {
              ev.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={(ev) => {
              ev.preventDefault();
              setDragActive(false);
            }}
            className={cn(
              "flex flex-wrap items-center gap-4 rounded-md border border-dashed px-5 py-4 transition-colors",
              dragActive
                ? "border-nx-accent bg-nx-accent/5"
                : "border-nx-muted/70 bg-nx-bg-elev-2/20 hover:border-nx-accent/50",
            )}
          >
            <input
              ref={inputRef}
              id={FILE_INPUT_ID}
              type="file"
              accept=".txt,.xls"
              onChange={(e) => onPick(e.target.files)}
              onClick={(e) => {
                (e.target as HTMLInputElement).value = "";
              }}
              className="sr-only"
            />
            <label
              htmlFor={FILE_INPUT_ID}
              className="inline-flex cursor-pointer items-center gap-2 rounded-md border border-nx-accent/40 bg-nx-accent/10 px-4 py-2 font-mono text-xs text-nx-accent transition-colors hover:bg-nx-accent/20 focus-within:ring-2 focus-within:ring-nx-accent"
            >
              <FileInputIcon className="size-3.5" />
              Choose .txt file…
            </label>
            <span className="font-mono text-[0.7rem] text-nx-fg/45">
              or drop it here · streamed, so 400 MB maps are fine
            </span>
          </div>

          <AnimatePresence mode="wait">
            {state.kind === "converting" && (
              <motion.div
                key="converting"
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="flex flex-col gap-2"
              >
                <div className="flex items-center gap-2 font-mono text-xs text-nx-fg/70">
                  <Loader2Icon className="size-3.5 animate-spin text-nx-accent" />
                  Resampling {state.name}…
                </div>
                <div className="h-1 w-full overflow-hidden rounded-sm bg-nx-bg-elev-2">
                  <motion.div
                    className="h-full bg-nx-accent"
                    animate={{ width: `${Math.round(state.progress * 100)}%` }}
                    transition={{ duration: 0.2 }}
                  />
                </div>
              </motion.div>
            )}

            {state.kind === "error" && (
              <motion.div
                key="error"
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="flex items-start gap-3 rounded-md border border-amber-500/40 bg-amber-950/20 px-4 py-3"
              >
                <XCircleIcon className="mt-0.5 size-4 shrink-0 text-amber-400" />
                <div className="min-w-0">
                  <div className="font-mono text-xs text-amber-300/95">
                    Cannot convert {state.name}
                  </div>
                  <p className="mt-1 text-[0.72rem] leading-relaxed text-nx-fg/65">
                    {state.message}
                  </p>
                </div>
              </motion.div>
            )}

            {state.kind === "done" && stats && (
              <motion.div
                key={state.name}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="flex flex-col gap-5"
              >
                <ConversionReport name={state.name} stats={stats} />

                {short ? (
                  <OodGate
                    stats={stats}
                    acknowledged={acknowledged}
                    onToggle={setAcknowledged}
                  />
                ) : (
                  <div className="flex items-center gap-2 rounded-md border border-nx-accent/30 bg-nx-accent/5 px-4 py-3">
                    <CheckCircle2Icon className="size-4 shrink-0 text-nx-accent" />
                    <span className="text-[0.75rem] leading-relaxed text-nx-fg/75">
                      Both model windows are fully covered by real measurements.
                      No flat-fill in the regions the model reads.
                    </span>
                  </div>
                )}

                <div className="flex flex-wrap items-center gap-3">
                  <Button
                    onClick={() => onAnalyze(state.file)}
                    disabled={!canAnalyze}
                    className="gap-2"
                  >
                    {busy ? (
                      <Loader2Icon className="size-4 animate-spin" />
                    ) : (
                      <PlayIcon className="size-4" />
                    )}
                    Analyze converted file
                  </Button>
                  <a
                    href={state.url}
                    download={stats.outputName}
                    className={cn(
                      buttonVariants({ variant: "outline" }),
                      "gap-2",
                    )}
                  >
                    <DownloadIcon className="size-4" />
                    Download ({formatBytes(stats.outputBytes)})
                  </a>
                  {short && !acknowledged && (
                    <span className="font-mono text-[0.68rem] text-amber-300/80">
                      acknowledge the coverage gap to enable analysis
                    </span>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </CardContent>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Conversion report
// ---------------------------------------------------------------------------

function ConversionReport({
  name,
  stats,
}: {
  name: string;
  stats: ConversionStats;
}) {
  const [lo, hi] = stats.nativeRange;
  return (
    <div className="flex flex-col gap-4 rounded-md border border-nx-muted/50 bg-nx-bg-elev-2/25 px-5 py-4">
      <div className="flex flex-wrap items-center gap-2">
        <CheckCircle2Icon className="size-4 text-nx-accent" />
        <span className="font-mono text-xs text-nx-fg/90 truncate max-w-md">
          {name}
        </span>
        <Badge
          variant="outline"
          className="font-mono text-[0.6rem] uppercase tracking-wide"
        >
          {stats.layout === "map" ? "pixel map" : "single spectrum"}
        </Badge>
        <span className="ml-auto font-mono text-[0.65rem] text-nx-fg/40">
          {formatBytes(stats.inputBytes)} → {formatBytes(stats.outputBytes)}
        </span>
      </div>

      <Separator className="bg-nx-muted/40" />

      <div className="grid gap-x-8 gap-y-3 sm:grid-cols-2">
        <Stat
          label="Source axis"
          value={`${stats.nNative} pts`}
          caption={`${lo.toFixed(0)}–${hi.toFixed(0)} cm⁻¹`}
        />
        <Stat
          label="Converted axis"
          value={`${N_BINS} pts`}
          caption="76–3499 cm⁻¹ · canonical"
        />
        <Stat
          label={stats.layout === "map" ? "Pixel rows" : "Spectrum"}
          value={
            stats.layout === "map"
              ? `${stats.nPixelsKept.toLocaleString()} of ${stats.nPixelsTotal.toLocaleString()}`
              : "1 averaged trace"
          }
          caption={
            stats.layout === "spectrum"
              ? "two-column file · emitted as one pixel"
              : stats.subsampled
                ? `randomly subsampled — the endpoint caps at ${PIXEL_CAP} regardless`
                : "all rows kept"
          }
        />
        <Stat
          label="Rows dropped"
          value={stats.nSkipped.toLocaleString()}
          caption={
            stats.nSkipped === 0
              ? "none malformed"
              : "short, or non-numeric intensities"
          }
        />
      </div>

      <Separator className="bg-nx-muted/40" />

      <div className="flex flex-col gap-3">
        <div className="font-mono text-[0.6rem] uppercase tracking-[0.18em] text-nx-fg/45">
          Coverage of the windows the model reads
        </div>
        <CoverageBar
          label="Fingerprint"
          range={FINGERPRINT}
          pct={stats.fingerprintCov}
        />
        <CoverageBar label="C–H stretch" range={CH_STRETCH} pct={stats.chCov} />
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  caption,
}: {
  label: string;
  value: string;
  caption: string;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="font-mono text-[0.6rem] uppercase tracking-[0.18em] text-nx-fg/45">
        {label}
      </span>
      <span className="font-mono text-sm text-nx-fg/95 tabular-nums">
        {value}
      </span>
      <span className="font-mono text-[0.65rem] text-nx-fg/40 leading-tight">
        {caption}
      </span>
    </div>
  );
}

/** Measured vs. flat-filled, as a proportion of one model window. */
function CoverageBar({
  label,
  range,
  pct,
}: {
  label: string;
  range: readonly [number, number];
  pct: number;
}) {
  const complete = pct >= 100;
  const none = pct <= 0;
  return (
    <div className="grid grid-cols-[8.5rem_1fr_auto] items-center gap-x-4 font-mono text-xs">
      <div className="flex flex-col">
        <span className="text-nx-fg/85">{label}</span>
        <span className="text-[0.62rem] text-nx-fg/40">
          {range[0]}–{range[1]} cm⁻¹
        </span>
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded-sm bg-nx-bg-elev-2">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${Math.max(0, Math.min(100, pct))}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
          className={cn(
            "h-full rounded-sm",
            complete ? "bg-nx-accent" : "bg-amber-400",
          )}
        />
      </div>
      <div
        className={cn(
          "text-right tabular-nums",
          complete ? "text-nx-accent" : none ? "text-red-400" : "text-amber-300",
        )}
      >
        {pct.toFixed(0)}% real
        <span className="ml-2 text-nx-fg/35">
          {(100 - pct).toFixed(0)}% flat
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Out-of-distribution gate
// ---------------------------------------------------------------------------

function OodGate({
  stats,
  acknowledged,
  onToggle,
}: {
  stats: ConversionStats;
  acknowledged: boolean;
  onToggle: (v: boolean) => void;
}) {
  const chMissing = stats.chCov <= 0;
  return (
    <div className="flex flex-col gap-3 rounded-md border border-amber-500/45 bg-amber-950/20 px-5 py-4">
      <div className="flex items-center gap-2">
        <AlertTriangleIcon className="size-4 shrink-0 text-amber-400" />
        <span className="font-mono text-[0.65rem] uppercase tracking-[0.22em] text-amber-300/95">
          Input is not compatible with this model
        </span>
      </div>
      <p className="text-[0.75rem] leading-relaxed text-nx-fg/75 max-w-3xl">
        The file now parses, but it does not carry the measurements the model
        was trained on.{" "}
        {chMissing ? (
          <>
            The instrument stopped at{" "}
            <span className="font-mono text-nx-fg/95">
              {stats.nativeRange[1].toFixed(0)} cm⁻¹
            </span>
            , so the entire C–H stretch window ({CH_STRETCH[0]}–{CH_STRETCH[1]}{" "}
            cm⁻¹) is absent and has been filled with a flat edge-clamped value.
          </>
        ) : (
          <>
            Part of the window the model reads was never measured and has been
            filled with a flat edge-clamped value.
          </>
        )}{" "}
        The classifier will still return a confident-looking probability over
        four classes, and that number is{" "}
        <span className="text-amber-200">not trustworthy</span> — it is a
        plumbing result, not a measurement. Treat it as a demo of the pipeline,
        never as evidence about this sample.
      </p>
      <label className="flex cursor-pointer items-center gap-3 pt-1">
        <Switch checked={acknowledged} onCheckedChange={onToggle} />
        <span className="text-[0.75rem] text-nx-fg/85">
          I understand these results are not trustworthy
        </span>
      </label>
    </div>
  );
}
