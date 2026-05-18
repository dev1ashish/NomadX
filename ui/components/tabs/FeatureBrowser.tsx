"use client";

/**
 * Tab 4 — Feature browser (W5).
 *
 * Renders the 259-feature Stage 15A–E catalog with:
 *  - Family chip filter row (Band 166 / Spectral 51 / MCR 32 / Spatial 10 /
 *    Bio subset).
 *  - Horizontal bar plot of top-15 by |Cohen's d| (STEC vs Non-STEC),
 *    color-coded by family.
 *  - Right-side panel with per-class box (built from {mean, std, n}),
 *    a plain-English "what is this feature?" blurb, and a metadata badge row.
 *  - Stage 15F MI-selection callout banner.
 *
 * Plan ref: `plan/ui/ULTRAPLAN.md` §4 W5. Sidecar:
 * `public/data/feature_catalog.json` built by `scripts/build_features.py`.
 */
import { useEffect, useMemo, useState } from "react";
import { SearchIcon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  FAMILY_COLORS,
  FAMILY_LABELS,
  FeatureBar,
} from "@/components/plots/FeatureBar";
import { FeatureBox, type PerClassStats } from "@/components/plots/FeatureBox";
import { getSidecar } from "@/lib/data";
import { describeFeature } from "@/lib/feature-explain";
import type { Feature } from "@/lib/types";
import { cn } from "@/lib/utils";

type SortKey = "d_stec" | "d_ecoli_salm" | "mi" | "name" | "family";

const SORT_OPTIONS: { key: SortKey; label: string }[] = [
  { key: "d_stec", label: "|d| STEC ↔ Non-STEC" },
  { key: "d_ecoli_salm", label: "|d| E.coli ↔ Salm" },
  { key: "mi", label: "Stage 15F MI rank" },
  { key: "name", label: "Name (A–Z)" },
  { key: "family", label: "Family" },
];

function sortFeatures(features: Feature[], key: SortKey): Feature[] {
  const arr = [...features];
  switch (key) {
    case "d_stec":
      return arr.sort(
        (a, b) =>
          Math.abs(b.d_stec_nonstec ?? 0) - Math.abs(a.d_stec_nonstec ?? 0),
      );
    case "d_ecoli_salm":
      return arr.sort(
        (a, b) =>
          Math.abs(b.d_ecoli_salm ?? 0) - Math.abs(a.d_ecoli_salm ?? 0),
      );
    case "mi":
      return arr.sort((a, b) => {
        const ar = a.mi_rank_stage15f ?? Number.POSITIVE_INFINITY;
        const br = b.mi_rank_stage15f ?? Number.POSITIVE_INFINITY;
        return ar - br;
      });
    case "name":
      return arr.sort((a, b) => a.name.localeCompare(b.name));
    case "family":
      return arr.sort((a, b) => {
        if (a.family !== b.family) return a.family.localeCompare(b.family);
        return (
          Math.abs(b.d_stec_nonstec ?? 0) - Math.abs(a.d_stec_nonstec ?? 0)
        );
      });
  }
}

interface FeatureCatalogPayload {
  features: Feature[];
  per_class_stats: Record<string, PerClassStats>;
  top_15_stec_nonstec: string[];
  stage15f_35: string[];
}

const FAMILY_BLURBS: Record<Feature["family"], string> = {
  band:
    "Per-band fit/area/ratio derived from one of the 30 named Raman peaks " +
    "(Stage 15A). Includes Voigt fits, AUCs, and adjacent-band ratios.",
  spectral:
    "Whole-spectrum descriptor — DWT energy/entropy by level, PCA on " +
    "anatomical regions, and SAM cosine similarity to class templates.",
  mcr:
    "MCR-ALS unmixing component statistics (Stage 15C). K=7 components, " +
    "per-file {mean, std, max, p90}. Did NOT survive per-fold MI in 15F.",
  spatial:
    "Per-map heterogeneity moments — variance, kurtosis, skew of LPS-region " +
    "peak heights across pixels (Stage 15E).",
  bio:
    "Biochemical aggregate from Stage 15D — α-helix score, Trp indole " +
    "environment, cyt-ox state, virulence amino-acid signature, etc.",
};

const FAMILY_FILTER_ORDER: ReadonlyArray<{
  key: Feature["family"];
  label: string;
}> = [
  { key: "band", label: "Band 166" },
  { key: "spectral", label: "Spectral 51" },
  { key: "mcr", label: "MCR 32" },
  { key: "spatial", label: "Spatial 10" },
  { key: "bio", label: "Bio (subset)" },
];

function formatD(d: number | undefined | null): string {
  if (typeof d !== "number" || Number.isNaN(d)) return "—";
  const sign = d >= 0 ? "+" : "";
  return `${sign}${d.toFixed(2)}`;
}

function Swatch({ family }: { family: Feature["family"] }) {
  return (
    <span
      aria-hidden
      className="inline-block h-2 w-2 rounded-full"
      style={{ background: FAMILY_COLORS[family] }}
    />
  );
}

export function FeatureBrowser() {
  const [payload, setPayload] = useState<FeatureCatalogPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeFamilies, setActiveFamilies] = useState<
    Set<Feature["family"]>
  >(new Set());
  const [selected, setSelected] = useState<string | null>(null);
  const [search, setSearch] = useState<string>("");
  const [sortKey, setSortKey] = useState<SortKey>("d_stec");
  const [miOnly, setMiOnly] = useState<boolean>(false);

  useEffect(() => {
    let cancelled = false;
    getSidecar<FeatureCatalogPayload>("feature_catalog.json")
      .then((data) => {
        if (cancelled) return;
        setPayload(data);
        // Default selection: first entry of top-15 (largest |d|).
        const first = data.top_15_stec_nonstec[0];
        setSelected(first ?? data.features[0]?.name ?? null);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const filtered = useMemo<Feature[]>(() => {
    if (!payload) return [];
    if (activeFamilies.size === 0) return payload.features;
    return payload.features.filter((f) => activeFamilies.has(f.family));
  }, [payload, activeFamilies]);

  const selectedFeature: Feature | null = useMemo(() => {
    if (!payload || !selected) return null;
    return payload.features.find((f) => f.name === selected) ?? null;
  }, [payload, selected]);

  const selectedStats: PerClassStats | null = useMemo(() => {
    if (!payload || !selected) return null;
    return payload.per_class_stats[selected] ?? null;
  }, [payload, selected]);

  const toggleFamily = (key: Feature["family"]) => {
    setActiveFamilies((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const familyCounts = useMemo(() => {
    if (!payload) return {} as Record<Feature["family"], number>;
    const counts = { band: 0, spectral: 0, mcr: 0, spatial: 0, bio: 0 };
    for (const f of payload.features) counts[f.family] += 1;
    return counts;
  }, [payload]);

  return (
    <section className="mx-auto max-w-screen-2xl px-6 py-10">
      <header className="mb-8">
        <h2 className="font-display text-nx-accent text-[3.0625rem] leading-[1.1]">
          Features
        </h2>
        <p className="mt-3 max-w-3xl text-nx-fg/70">
          259 engineered features across five families. Cohen&apos;s d is
          computed at the file level (per-pixel caches mean-pooled to file)
          so every effect size below uses the same 87-file basis as Stage
          15F.
        </p>
      </header>

      {/* Stage 15F MI callout banner ------------------------------------ */}
      <Card className="mb-8 ring-1 ring-nx-accent/40">
        <CardHeader>
          <CardTitle className="text-nx-accent">
            Stage 15F MI-selected: 35 features
          </CardTitle>
          <CardDescription className="text-nx-fg/80">
            Top-10 = all Stage 15A peak-fits + derivatives.{" "}
            <span className="font-semibold text-nx-fg">
              0 MCR features survived per-fold MI
            </span>{" "}
            (<code className="font-mono text-nx-accent">mcr_C6_mean</code>{" "}
            d=−1.23 is global-fit only — partly a leakage artifact).
          </CardDescription>
        </CardHeader>
      </Card>

      {/* Family chips --------------------------------------------------- */}
      <div className="mb-6 flex flex-wrap items-center gap-2">
        <span className="mr-2 text-xs uppercase tracking-[0.16em] text-nx-fg/50">
          Filter
        </span>
        {FAMILY_FILTER_ORDER.map(({ key, label }) => {
          const active = activeFamilies.has(key);
          const count = familyCounts[key] ?? 0;
          const baseLabel =
            key === "bio"
              ? `Bio (${count})`
              : label.replace(/\d+/, String(count));
          return (
            <button
              key={key}
              type="button"
              onClick={() => toggleFamily(key)}
              className={cn(
                "group inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-mono transition-colors",
                active
                  ? "border-nx-accent bg-nx-accent/10 text-nx-accent"
                  : "border-nx-muted bg-transparent text-nx-fg/70 hover:border-nx-accent/60 hover:text-nx-fg",
              )}
              aria-pressed={active}
            >
              <Swatch family={key} />
              {baseLabel}
            </button>
          );
        })}
        {activeFamilies.size > 0 && (
          <button
            type="button"
            onClick={() => setActiveFamilies(new Set())}
            className="ml-2 rounded-full border border-transparent px-2 py-1 text-xs text-nx-fg/50 hover:text-nx-fg"
          >
            clear
          </button>
        )}
      </div>

      {error && (
        <p className="text-sm text-nx-danger">
          Failed to load feature_catalog.json: {error}
        </p>
      )}

      {/* Main grid ------------------------------------------------------ */}
      <div className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        {/* Top-15 bar */}
        <Card className="overflow-hidden">
          <CardHeader>
            <CardTitle className="text-nx-fg">
              Top 15 features — STEC vs Non-STEC
            </CardTitle>
            <CardDescription>
              Bars colored by family. Click a bar to inspect a feature.
            </CardDescription>
          </CardHeader>
          <CardContent className="px-2">
            {payload ? (
              <FeatureBar
                features={filtered.length ? filtered : payload.features}
                selected={selected ?? undefined}
                onSelect={setSelected}
              />
            ) : (
              <div className="h-[440px] animate-pulse rounded-[var(--radius-sm)] bg-nx-bg-elev-1" />
            )}
          </CardContent>
        </Card>

        {/* Selected feature panel */}
        <Card>
          <CardHeader>
            <CardTitle className="font-mono text-nx-fg break-all">
              {selectedFeature?.name ?? "Select a feature"}
            </CardTitle>
            {selectedFeature && (
              <CardDescription className="text-nx-fg/70">
                {FAMILY_BLURBS[selectedFeature.family]}
              </CardDescription>
            )}
          </CardHeader>
          <CardContent className="space-y-4">
            {selectedFeature && selectedStats ? (
              <>
                <FeatureBox
                  featureName={selectedFeature.name}
                  stats={selectedStats}
                />
                <div className="flex flex-wrap gap-2">
                  <Badge
                    variant="outline"
                    className="border-nx-muted text-nx-fg"
                  >
                    <Swatch family={selectedFeature.family} />
                    <span className="ml-1">
                      {FAMILY_LABELS[selectedFeature.family]}
                    </span>
                  </Badge>
                  {selectedFeature.region && (
                    <Badge
                      variant="outline"
                      className="border-nx-muted font-mono text-nx-fg/80"
                    >
                      region: {selectedFeature.region}
                    </Badge>
                  )}
                  <Badge
                    variant="outline"
                    className="border-nx-muted font-mono text-nx-fg"
                  >
                    d(STEC↔Non-STEC) ={" "}
                    {formatD(selectedFeature.d_stec_nonstec)}
                  </Badge>
                  <Badge
                    variant="outline"
                    className="border-nx-muted font-mono text-nx-fg"
                  >
                    d(E.coli↔Salm) ={" "}
                    {formatD(selectedFeature.d_ecoli_salm)}
                  </Badge>
                  {typeof selectedFeature.mi_rank_stage15f === "number" && (
                    <Badge className="bg-nx-accent text-nx-bg">
                      MI rank #{selectedFeature.mi_rank_stage15f}/35
                    </Badge>
                  )}
                </div>
              </>
            ) : (
              <p className="text-sm text-nx-fg/50">
                Click a bar to inspect distribution and effect sizes.
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Browse-all list — search + sort + plain-English explainer ------- */}
      {payload && (
        <Card className="mt-8">
          <CardHeader>
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex flex-col gap-1 flex-1 min-w-[200px]">
                <CardTitle className="text-nx-fg">
                  Browse all features
                </CardTitle>
                <CardDescription>
                  {(() => {
                    const visible = (() => {
                      let arr = filtered;
                      const q = search.trim().toLowerCase();
                      if (q) arr = arr.filter((f) => f.name.toLowerCase().includes(q));
                      if (miOnly)
                        arr = arr.filter(
                          (f) => typeof f.mi_rank_stage15f === "number",
                        );
                      return arr.length;
                    })();
                    return `${visible} of ${payload.features.length} features`;
                  })()}
                </CardDescription>
              </div>

              {/* Search */}
              <div className="relative flex-1 min-w-[240px] max-w-md">
                <SearchIcon
                  className="absolute left-3 top-1/2 -translate-y-1/2 size-3.5 text-nx-fg/40"
                  strokeWidth={1.75}
                />
                <input
                  type="search"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search by name (e.g. mcr_C6, amide, lps_1117)…"
                  className="w-full bg-nx-bg-elev-1/60 border border-nx-muted/50 rounded-md pl-9 pr-3 py-1.5 text-xs font-mono text-nx-fg placeholder:text-nx-fg/35 focus:outline-none focus:border-nx-accent transition-colors"
                />
              </div>

              {/* Sort */}
              <div className="flex items-center gap-2">
                <label
                  htmlFor="feature-sort"
                  className="font-mono text-[0.6rem] uppercase tracking-[0.16em] text-nx-fg/45"
                >
                  Sort
                </label>
                <select
                  id="feature-sort"
                  value={sortKey}
                  onChange={(e) => setSortKey(e.target.value as SortKey)}
                  className="bg-nx-bg-elev-1/60 border border-nx-muted/50 rounded-md px-2 py-1.5 text-xs font-mono text-nx-fg focus:outline-none focus:border-nx-accent transition-colors"
                >
                  {SORT_OPTIONS.map((opt) => (
                    <option key={opt.key} value={opt.key}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* MI-only toggle */}
              <button
                type="button"
                onClick={() => setMiOnly((v) => !v)}
                aria-pressed={miOnly}
                className={cn(
                  "rounded-full border px-3 py-1 text-xs font-mono transition-colors",
                  miOnly
                    ? "border-nx-accent bg-nx-accent/10 text-nx-accent"
                    : "border-nx-muted text-nx-fg/60 hover:border-nx-accent/60 hover:text-nx-fg",
                )}
              >
                MI-35 only
              </button>
            </div>
          </CardHeader>

          {/* Sticky column header */}
          <div className="px-6 py-2 border-y border-nx-muted/30 bg-nx-bg-elev-1/40 sticky top-0">
            <div className="grid grid-cols-[auto_1fr_5rem_5rem_3.5rem] gap-3 items-center font-mono text-[0.6rem] uppercase tracking-[0.16em] text-nx-fg/45">
              <span className="w-3" />
              <span>Feature</span>
              <span className="text-right">d STEC↔NS</span>
              <span className="text-right">d Ec↔Sal</span>
              <span className="text-center">MI #</span>
            </div>
          </div>

          <CardContent className="p-0">
            <ul className="max-h-[520px] overflow-y-auto">
              {(() => {
                let arr = filtered;
                const q = search.trim().toLowerCase();
                if (q)
                  arr = arr.filter((f) =>
                    f.name.toLowerCase().includes(q),
                  );
                if (miOnly)
                  arr = arr.filter(
                    (f) => typeof f.mi_rank_stage15f === "number",
                  );
                return sortFeatures(arr, sortKey).map((f) => {
                  const active = f.name === selected;
                  const desc = describeFeature(f.name);
                  return (
                    <li key={f.name}>
                      <button
                        type="button"
                        onClick={() => setSelected(f.name)}
                        className={cn(
                          "group w-full text-left grid grid-cols-[auto_1fr_5rem_5rem_3.5rem] gap-3 items-center px-6 py-2 border-b border-nx-muted/15 transition-colors",
                          active
                            ? "bg-nx-accent/10"
                            : "hover:bg-nx-bg-elev-1/60",
                        )}
                      >
                        <Swatch family={f.family} />
                        <div className="flex flex-col min-w-0">
                          <span
                            className={cn(
                              "font-mono text-xs truncate",
                              active ? "text-nx-accent" : "text-nx-fg/90",
                            )}
                          >
                            {f.name}
                          </span>
                          <span className="text-[10px] text-nx-fg/45 truncate group-hover:text-nx-fg/65 transition-colors">
                            {desc}
                          </span>
                        </div>
                        <span
                          className={cn(
                            "font-mono text-xs text-right tabular-nums",
                            Math.abs(f.d_stec_nonstec ?? 0) >= 0.8
                              ? "text-nx-accent"
                              : "text-nx-fg/70",
                          )}
                        >
                          {formatD(f.d_stec_nonstec)}
                        </span>
                        <span
                          className={cn(
                            "font-mono text-xs text-right tabular-nums",
                            Math.abs(f.d_ecoli_salm ?? 0) >= 0.8
                              ? "text-nx-accent"
                              : "text-nx-fg/55",
                          )}
                        >
                          {formatD(f.d_ecoli_salm)}
                        </span>
                        <span className="text-center">
                          {typeof f.mi_rank_stage15f === "number" ? (
                            <span className="inline-flex items-center justify-center rounded-sm bg-nx-accent/15 px-1.5 py-0.5 font-mono text-[10px] text-nx-accent">
                              #{f.mi_rank_stage15f}
                            </span>
                          ) : (
                            <span className="font-mono text-[10px] text-nx-fg/20">
                              —
                            </span>
                          )}
                        </span>
                      </button>
                    </li>
                  );
                });
              })()}
            </ul>
          </CardContent>
        </Card>
      )}
    </section>
  );
}
