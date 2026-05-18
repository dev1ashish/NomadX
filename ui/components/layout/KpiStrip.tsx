"use client";

/**
 * KPI strip — mono-typeface stat row that fades in on mount.
 * Plan ref: §3 ASCII mockup ("87 FILES   7,122 SPECTRA   987 BINS   LOSO 0.603").
 */
import { motion } from "framer-motion";

export interface Kpi {
  label: string;
  value: string;
}

interface KpiStripProps {
  items: Kpi[];
}

export function KpiStrip({ items }: KpiStripProps) {
  return (
    <ul
      role="list"
      className="grid gap-6 px-6 py-6 text-nx-fg sm:grid-cols-2 md:grid-cols-4"
    >
      {items.map((kpi, i) => (
        <motion.li
          key={kpi.label}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: i * 0.05, ease: "easeOut" }}
          className="flex flex-col gap-1"
        >
          <span className="font-mono text-2xl text-nx-fg">{kpi.value}</span>
          <span className="text-xs uppercase tracking-[0.16em] text-nx-fg/60">
            {kpi.label}
          </span>
          <span aria-hidden className="mt-1 h-px w-12 bg-nx-muted" />
        </motion.li>
      ))}
    </ul>
  );
}
