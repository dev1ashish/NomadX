"use client";

/**
 * Band-chemistry primer (Tab 3 / W4).
 *
 * Pure card layout — no Plotly. Renders four sections:
 *   1. LPS-chain empirical-anchor callout (Stage 1 winner, 800-1200 cm⁻¹).
 *   2. Five macromolecule-group cards (Aromatic AA / Protein amide /
 *      Nucleic acid / Lipid+carb / Metabolite), each listing the bands in
 *      its group with center, name, chemistry one-liner, and an optional
 *      "d=±X.XX STEC↔Non-STEC" badge when |d| >= 0.4.
 *   3. Cisek-2013 falsification panel (1338 / 1454 / 1658) — literature
 *      claim struck-through alongside the Atlas d-value.
 *
 * Data source: `public/data/bands.json`, built by `scripts/build_bands.py`.
 * Plan reference: `plan/ui/ULTRAPLAN.md` §4 W4.
 */
import { useEffect, useState } from "react";
import { motion } from "framer-motion";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { getSidecar } from "@/lib/data";

// ---------------------------------------------------------------------------
// Sidecar contract — mirrors the JSON emitted by `scripts/build_bands.py`.
// ---------------------------------------------------------------------------

interface BandRow {
  name: string;
  center: number;
  fwhm: number;
  chemistry: string;
  d_stec_nonstec: number | null;
}

interface BandGroup {
  key: string;
  label: string;
  biology: string;
  bands: BandRow[];
}

interface CisekRow {
  center: number;
  label: string;
  literature_claim: string;
  atlas_d: number;
  verdict: "null" | "sign-reversed" | string;
}

interface AnchorSpec {
  region: [number, number];
  label: string;
  top_band: string;
  top_d: number;
}

interface BandsSidecar {
  groups: BandGroup[];
  cisek_falsification: { headline: string; bands: CisekRow[] };
  anchors: { lps_chain: AnchorSpec };
}

const D_BADGE_THRESHOLD = 0.4;

function formatD(d: number): string {
  const sign = d >= 0 ? "+" : "";
  return `${sign}${d.toFixed(2)}`;
}

function verdictColor(verdict: string): string {
  if (verdict === "sign-reversed") return "bg-nx-danger/20 text-nx-danger border-nx-danger/40";
  return "bg-nx-muted/40 text-nx-fg/70 border-nx-muted";
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function EmpiricalAnchor({ anchor }: { anchor: AnchorSpec }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
    >
      <Card className="border-l-2 border-l-nx-accent bg-nx-bg-elev-1">
        <CardHeader>
          <div className="flex items-center gap-3 flex-wrap">
            <Badge className="bg-nx-accent text-nx-bg font-mono text-[0.7rem] uppercase tracking-wider">
              empirical anchor
            </Badge>
            <CardTitle className="text-nx-fg text-base">
              LPS chain region — {anchor.region[0]}-{anchor.region[1]} cm⁻¹
            </CardTitle>
          </div>
        </CardHeader>
        <CardContent className="text-sm text-nx-fg/80 space-y-2">
          <p>
            Where the file-level STEC ↔ Non-STEC signal actually lives. The
            literature triple (1338 / 1454 / 1658) failed to replicate on
            this corpus — see the falsification panel below.
          </p>
          <p className="font-mono text-xs text-nx-fg/70">
            Top discriminator:{" "}
            <span className="text-nx-accent">{anchor.top_band}</span>{" "}
            · d = <span className="text-nx-accent">{formatD(anchor.top_d)}</span>{" "}
            STEC ↔ Non-STEC (Stage 1)
          </p>
        </CardContent>
      </Card>
    </motion.div>
  );
}

function BandGroupCard({ group, index }: { group: BandGroup; index: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-50px" }}
      transition={{ duration: 0.3, ease: "easeOut", delay: index * 0.04 }}
    >
      <Card className="h-full bg-nx-bg-elev-1">
        <CardHeader className="border-b border-nx-muted/40 pb-3">
          <div className="flex items-baseline justify-between gap-3">
            <CardTitle className="text-nx-accent text-lg">{group.label}</CardTitle>
            <span className="font-mono text-[0.7rem] text-nx-fg/50 uppercase tracking-wider">
              {group.bands.length} bands
            </span>
          </div>
          <p className="text-xs text-nx-fg/60 mt-1">{group.biology}</p>
        </CardHeader>
        <CardContent className="space-y-2 pt-2">
          {group.bands.map((b) => (
            <BandRowItem key={b.name} band={b} />
          ))}
        </CardContent>
      </Card>
    </motion.div>
  );
}

function BandRowItem({ band }: { band: BandRow }) {
  const d = band.d_stec_nonstec;
  const showBadge = d !== null && Math.abs(d) >= D_BADGE_THRESHOLD;
  return (
    <div className="flex items-start justify-between gap-3 py-1.5 border-b border-nx-muted/20 last:border-b-0">
      <div className="flex items-start gap-2 min-w-0 flex-1">
        <span className="font-mono text-[0.72rem] text-nx-fg/90 shrink-0 w-14 pt-0.5">
          {band.center.toFixed(0)} cm⁻¹
        </span>
        <div className="min-w-0 flex-1">
          <div className="font-mono text-[0.7rem] text-nx-accent/80">{band.name}</div>
          <div className="text-xs text-nx-fg/70 leading-snug">{band.chemistry}</div>
        </div>
      </div>
      {showBadge && d !== null && (
        <Badge
          className={`shrink-0 font-mono text-[0.65rem] border ${
            d > 0
              ? "bg-class-stec/15 text-class-stec border-class-stec/40"
              : "bg-class-nonstec/15 text-class-nonstec border-class-nonstec/40"
          }`}
        >
          d={formatD(d)}
        </Badge>
      )}
    </div>
  );
}

function CisekPanel({ panel }: { panel: BandsSidecar["cisek_falsification"] }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-50px" }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="space-y-4"
    >
      <div>
        <h3 className="font-display text-nx-fg text-2xl leading-tight">
          Cisek-2013 falsification
        </h3>
        <p className="text-sm text-nx-fg/70 mt-1">{panel.headline}</p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {panel.bands.map((b, i) => (
          <motion.div
            key={b.label}
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.25, ease: "easeOut", delay: i * 0.06 }}
          >
            <Card className="h-full bg-nx-bg-elev-1">
              <CardHeader className="pb-2">
                <div className="flex items-baseline justify-between">
                  <CardTitle className="font-mono text-nx-fg text-base">
                    {b.center} cm⁻¹
                  </CardTitle>
                  <Badge className={`font-mono text-[0.65rem] uppercase tracking-wider border ${verdictColor(b.verdict)}`}>
                    {b.verdict}
                  </Badge>
                </div>
                <p className="font-mono text-[0.7rem] text-nx-accent/80 mt-1">{b.label}</p>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div>
                  <div className="text-[0.65rem] uppercase tracking-wider text-nx-fg/40 mb-1">
                    Literature claim
                  </div>
                  <p className="text-nx-fg/50 line-through decoration-nx-danger/60 decoration-1">
                    {b.literature_claim}
                  </p>
                </div>
                <div>
                  <div className="text-[0.65rem] uppercase tracking-wider text-nx-fg/40 mb-1">
                    Atlas (file-level)
                  </div>
                  <p className="font-mono text-nx-fg">
                    d = <span className={b.atlas_d < 0 ? "text-nx-danger" : "text-nx-fg"}>{formatD(b.atlas_d)}</span>
                    <span className="text-nx-fg/50 text-xs"> STEC ↔ Non-STEC</span>
                  </p>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Page-level component
// ---------------------------------------------------------------------------

export function BandPrimer() {
  const [data, setData] = useState<BandsSidecar | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    getSidecar<BandsSidecar>("bands.json")
      .then((d) => {
        if (alive) setData(d);
      })
      .catch((e: unknown) => {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      alive = false;
    };
  }, []);

  if (error) {
    return (
      <section className="mx-auto max-w-screen-2xl px-6 py-10">
        <h2 className="font-display text-nx-accent text-[3.0625rem] leading-[1.1]">Primer</h2>
        <p className="mt-4 text-nx-danger">Failed to load bands.json: {error}</p>
      </section>
    );
  }

  if (!data) {
    return (
      <section className="mx-auto max-w-screen-2xl px-6 py-10">
        <h2 className="font-display text-nx-accent text-[3.0625rem] leading-[1.1]">Primer</h2>
        <p className="mt-4 text-nx-fg/40 font-mono text-sm">Loading band catalog…</p>
      </section>
    );
  }

  const totalBands = data.groups.reduce((sum, g) => sum + g.bands.length, 0);

  return (
    <section className="mx-auto max-w-screen-2xl px-6 py-10 space-y-8">
      <header>
        <h2 className="font-display text-nx-accent text-[3.0625rem] leading-[1.1]">
          Band-chemistry primer
        </h2>
        <p className="mt-3 text-nx-fg/70 max-w-3xl">
          The 30+ named Raman bands the Atlas pipeline integrates, grouped by
          macromolecule. Discrimination strength shown as Cohen&apos;s d at file
          level (STEC ↔ Non-STEC); only |d| ≥ {D_BADGE_THRESHOLD.toFixed(1)} is
          flagged.
        </p>
        <p className="mt-2 font-mono text-xs text-nx-fg/40 uppercase tracking-wider">
          {totalBands} bands · {data.groups.length} groups
        </p>
      </header>

      <EmpiricalAnchor anchor={data.anchors.lps_chain} />

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {data.groups.map((g, i) => (
          <BandGroupCard key={g.key} group={g} index={i} />
        ))}
      </div>

      <CisekPanel panel={data.cisek_falsification} />
    </section>
  );
}
