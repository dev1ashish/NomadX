"use client";

/**
 * Top-level shell for the Comparison Lab tab. Owns the staged-spectra state,
 * the view toolbar, and the role-lane chip strip. Renders one of five view
 * components depending on `view`.
 */
import { useState } from "react";
import { Plus, X } from "lucide-react";
import { cn } from "@/lib/cn";
import { ComparisonPicker } from "./ComparisonPicker";
import type {
  ComparisonRole,
  ComparisonView,
  NormalizationMode,
  RegionPreset,
  StagedSpectrum,
} from "@/lib/types";
import { ComparisonGrid } from "@/components/plots/ComparisonGrid";
import { ComparisonOverlay } from "@/components/plots/ComparisonOverlay";
import { ComparisonWaterfall } from "@/components/plots/ComparisonWaterfall";
import { ComparisonHeatmap } from "@/components/plots/ComparisonHeatmap";
import { ComparisonDiff } from "@/components/plots/ComparisonDiff";

const ROLE_LABELS: Record<ComparisonRole, string> = {
  control_pos: "+ve",
  blank: "blank",
  test: "test",
};

const VIEW_TABS: { key: ComparisonView; label: string }[] = [
  { key: "grid", label: "Grid" },
  { key: "overlay", label: "Overlay" },
  { key: "waterfall", label: "Waterfall" },
  { key: "heatmap", label: "Heatmap" },
  { key: "diff", label: "Diff" },
];

const NORM_TABS: { key: NormalizationMode; label: string }[] = [
  { key: "snv", label: "SNV" },
  { key: "minmax", label: "Min-Max" },
  { key: "raw", label: "Raw" },
  { key: "mean_center", label: "Mean-center" },
];

const REGION_TABS: { key: RegionPreset; label: string }[] = [
  { key: "full", label: "Full" },
  { key: "fingerprint_800_1800", label: "Fingerprint" },
  { key: "lps_400_900", label: "LPS 400–900" },
  { key: "lps_800_1200", label: "LPS 800–1200" },
];

export function ComparisonLab() {
  const [staged, setStaged] = useState<StagedSpectrum[]>([]);
  const [view, setView] = useState<ComparisonView>("grid");
  const [normalization, setNormalization] = useState<NormalizationMode>("snv");
  const [region, setRegion] = useState<RegionPreset>("full");
  const [linkXZoom, setLinkXZoom] = useState(true);
  const [shareYScale, setShareYScale] = useState(false);
  const [pickerOpen, setPickerOpen] = useState(false);

  const stage = (next: StagedSpectrum) => {
    setStaged((s) => [...s, next]);
    setPickerOpen(false);
  };
  const remove = (file_id: string) =>
    setStaged((s) => s.filter((x) => x.file_id !== file_id));
  const setRole = (file_id: string, role: ComparisonRole) =>
    setStaged((s) =>
      s.map((x) => (x.file_id === file_id ? { ...x, role } : x)),
    );
  const referenceFileId =
    staged.find((s) => s.role === "blank")?.file_id ?? staged[0]?.file_id;

  return (
    <div className="flex flex-col gap-6">
      <header className="flex items-end justify-between flex-wrap gap-4">
        <div>
          <h1 className="font-display text-3xl tracking-tight">
            Comparison Lab
          </h1>
          <p className="font-mono text-xs text-nx-fg/55 mt-1">
            Stage 4–12+ spectra · linked zoom + crosshair · five views
          </p>
        </div>
        <button
          onClick={() => setPickerOpen(true)}
          className="flex items-center gap-2 px-3 py-2 rounded-sm border border-nx-muted hover:border-nx-accent hover:text-nx-accent transition-colors text-sm"
        >
          <Plus className="size-4" /> Add spectrum
        </button>
      </header>

      {staged.length > 0 ? (
        <section className="rounded-md border border-nx-muted bg-nx-bg-elev-1/40 px-4 py-3">
          <p className="font-mono text-[0.6rem] text-nx-fg/45 uppercase tracking-[0.18em] mb-2">
            Staged · {staged.length}
          </p>
          <ul className="flex flex-wrap gap-2">
            {staged.map((s) => (
              <li
                key={s.file_id}
                className="flex items-center gap-2 px-2.5 py-1 rounded-sm border border-nx-muted/60 bg-nx-bg text-xs font-mono"
              >
                <select
                  value={s.role}
                  onChange={(e) =>
                    setRole(s.file_id, e.target.value as ComparisonRole)
                  }
                  className="bg-transparent text-nx-accent text-[0.65rem] outline-none"
                  aria-label={`Role for ${s.file_id}`}
                >
                  <option value="test">{ROLE_LABELS.test}</option>
                  <option value="control_pos">{ROLE_LABELS.control_pos}</option>
                  <option value="blank">{ROLE_LABELS.blank}</option>
                </select>
                <span>{s.display_label}</span>
                <button
                  aria-label={`Remove ${s.file_id}`}
                  onClick={() => remove(s.file_id)}
                  className="text-nx-fg/45 hover:text-nx-danger"
                >
                  <X className="size-3" />
                </button>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <section className="flex flex-wrap gap-4 items-center text-xs">
        <Toolbar label="View" tabs={VIEW_TABS} value={view} onChange={setView} />
        <Toolbar
          label="Norm"
          tabs={NORM_TABS}
          value={normalization}
          onChange={setNormalization}
        />
        <Toolbar
          label="Region"
          tabs={REGION_TABS}
          value={region}
          onChange={setRegion}
        />
        <label className="flex items-center gap-2 font-mono text-[0.7rem] text-nx-fg/70">
          <input
            type="checkbox"
            checked={linkXZoom}
            onChange={(e) => setLinkXZoom(e.target.checked)}
          />
          link x-zoom
        </label>
        <label className="flex items-center gap-2 font-mono text-[0.7rem] text-nx-fg/70">
          <input
            type="checkbox"
            checked={shareYScale}
            onChange={(e) => setShareYScale(e.target.checked)}
          />
          shared y
        </label>
      </section>

      {staged.length === 0 ? (
        <div className="rounded-md border border-dashed border-nx-muted bg-nx-bg-elev-1/30 px-6 py-16 text-center">
          <p className="font-display text-lg text-nx-fg/80">
            Stage at least one spectrum to begin.
          </p>
          <p className="font-mono text-xs text-nx-fg/45 mt-2">
            Use <kbd className="px-1 py-0.5 rounded-sm bg-nx-bg-elev-2">Add spectrum</kbd> above.
          </p>
        </div>
      ) : (
        <ViewBody
          view={view}
          staged={staged}
          normalization={normalization}
          region={region}
          linkXZoom={linkXZoom}
          shareYScale={shareYScale}
          referenceFileId={referenceFileId}
        />
      )}

      <ComparisonPicker
        open={pickerOpen}
        onClose={() => setPickerOpen(false)}
        staged={staged}
        onStage={stage}
      />
    </div>
  );
}

function Toolbar<T extends string>({
  label,
  tabs,
  value,
  onChange,
}: {
  label: string;
  tabs: { key: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="font-mono text-[0.6rem] text-nx-fg/45 uppercase tracking-[0.18em]">
        {label}
      </span>
      <div className="flex rounded-sm border border-nx-muted overflow-hidden">
        {tabs.map((t) => (
          <button
            key={t.key}
            onClick={() => onChange(t.key)}
            className={cn(
              "px-2.5 py-1 font-mono text-[0.7rem] transition-colors",
              value === t.key
                ? "bg-nx-accent/15 text-nx-accent"
                : "text-nx-fg/65 hover:text-nx-fg",
            )}
          >
            {t.label}
          </button>
        ))}
      </div>
    </div>
  );
}

interface ViewBodyProps {
  view: ComparisonView;
  staged: StagedSpectrum[];
  normalization: NormalizationMode;
  region: RegionPreset;
  linkXZoom: boolean;
  shareYScale: boolean;
  referenceFileId?: string;
}

function ViewBody(props: ViewBodyProps) {
  const { view, ...viewProps } = props;
  switch (view) {
    case "grid":
      return <ComparisonGrid {...viewProps} />;
    case "overlay":
      return <ComparisonOverlay {...viewProps} />;
    case "waterfall":
      return <ComparisonWaterfall {...viewProps} />;
    case "heatmap":
      return <ComparisonHeatmap {...viewProps} />;
    case "diff":
      return <ComparisonDiff {...viewProps} />;
  }
}
