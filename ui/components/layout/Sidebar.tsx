"use client";

/**
 * Atlas Raman — left sidebar navigation.
 * Replaces the top TabNav. Live is the hero (first + accent).
 */
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import {
  Zap,
  Database,
  Activity,
  BookOpen,
  Sparkles,
  Layers,
  BarChart3,
  Wand2,
  HelpCircle,
  Columns2,
} from "lucide-react";
import { cn } from "@/lib/cn";

type NavItem = {
  href: string;
  label: string;
  icon: typeof Zap;
  hint?: string;
  hero?: boolean;
  divideBefore?: boolean;
};

const NAV: NavItem[] = [
  { href: "/live", label: "Live", icon: Zap, hint: "Classify a Raman map", hero: true },
  { href: "/inventory", label: "Inventory", icon: Database, hint: "87 files · 4 classes" },
  { href: "/spectrum", label: "Spectrum", icon: Activity, hint: "Per-file mean trace" },
  { href: "/compare", label: "Compare", icon: Columns2, hint: "Stage 4–12+ side-by-side" },
  { href: "/preprocessing", label: "Preprocessing", icon: Wand2, hint: "5-step pipeline" },
  { href: "/primer", label: "Primer", icon: BookOpen, hint: "30 Raman bands" },
  { href: "/features", label: "Features", icon: Sparkles, hint: "259 engineered" },
  { href: "/mcr", label: "MCR-ALS", icon: Layers, hint: "7 pure components" },
  { href: "/results", label: "Results", icon: BarChart3, hint: "LOSO + bootstrap + McNemar" },
  { href: "/about", label: "About", icon: HelpCircle, hint: "Glossary · FAQ · limits", divideBefore: true },
];

export function Sidebar() {
  const pathname = usePathname();
  const modalConfigured =
    typeof process !== "undefined" &&
    typeof process.env?.NEXT_PUBLIC_MODAL_PREDICT_URL === "string" &&
    process.env.NEXT_PUBLIC_MODAL_PREDICT_URL.length > 0;

  return (
    <aside
      aria-label="Atlas Raman navigation"
      className="hidden md:flex sticky top-0 h-screen w-64 shrink-0 flex-col border-r border-nx-muted bg-nx-bg-elev-1/30 backdrop-blur-sm"
    >
      {/* Brand */}
      <Link
        href="/live"
        className="group flex items-baseline gap-3 px-6 py-6 border-b border-nx-muted/60 hover:bg-nx-bg-elev-1/60 transition-colors"
      >
        <span
          aria-hidden
          className="text-nx-accent text-xl leading-none transition-transform group-hover:rotate-12"
          style={{ transform: "translateY(2px)" }}
        >
          ◆
        </span>
        <div className="flex flex-col gap-0.5">
          <span className="text-nx-fg text-[0.7rem] font-medium tracking-[0.22em] uppercase font-display">
            NomadX
          </span>
          <span className="text-nx-accent text-[0.7rem] font-medium tracking-[0.22em] uppercase font-display">
            Atlas Raman
          </span>
        </div>
      </Link>

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto px-3 py-4">
        <ul className="flex flex-col gap-1">
          {NAV.map((item) => {
            const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
            const Icon = item.icon;
            return (
              <li key={item.href} className={cn(
                "relative",
                item.divideBefore && "mt-3 pt-3 border-t border-nx-muted/40",
              )}>
                <Link
                  href={item.href}
                  className={cn(
                    "group relative flex items-center gap-3 rounded-sm px-3 py-2.5 text-sm transition-all",
                    active
                      ? "text-nx-fg bg-nx-bg-elev-2/40"
                      : "text-nx-fg/55 hover:text-nx-fg hover:bg-nx-bg-elev-1/50 hover:translate-x-0.5",
                    item.hero && !active && "text-nx-fg/80",
                  )}
                >
                  {active ? (
                    <motion.span
                      layoutId="sidebarActive"
                      className="absolute left-0 top-1 bottom-1 w-[3px] bg-nx-accent rounded-r-sm"
                      transition={{ type: "spring", stiffness: 380, damping: 32 }}
                    />
                  ) : null}
                  <Icon
                    aria-hidden
                    className={cn(
                      "size-4 shrink-0 transition-colors",
                      active ? "text-nx-accent" : "text-nx-fg/45 group-hover:text-nx-fg/80",
                      item.hero && !active && "text-nx-accent/70",
                    )}
                    strokeWidth={1.75}
                  />
                  <span className="flex flex-col gap-0">
                    <span className="font-display text-[0.875rem] tracking-wide leading-tight">
                      {item.label}
                    </span>
                    {item.hint ? (
                      <span className="font-mono text-[0.625rem] text-nx-fg/35 group-hover:text-nx-fg/55 leading-tight">
                        {item.hint}
                      </span>
                    ) : null}
                  </span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Footer: stats + status */}
      <div className="border-t border-nx-muted/60 px-6 py-4 flex flex-col gap-3">
        <dl className="grid grid-cols-2 gap-x-3 gap-y-2 font-mono text-[0.65rem] tracking-wide">
          <Stat label="files" value="87" />
          <Stat label="spectra" value="7,122" />
          <Stat label="bins" value="987" />
          <Stat label="LOSO" value="0.603" accent />
        </dl>
        <div className="flex items-center gap-2 pt-2 border-t border-nx-muted/40">
          <span
            aria-hidden
            className={cn(
              "size-1.5 rounded-full",
              modalConfigured ? "bg-emerald-400 animate-pulse" : "bg-amber-400/80",
            )}
          />
          <span className="font-mono text-[0.625rem] text-nx-fg/55 tracking-wide">
            {modalConfigured ? "modal · live" : "modal · not configured"}
          </span>
        </div>
        <span className="font-mono text-[0.5rem] text-nx-fg/30 tracking-[0.2em] uppercase">
          v0.1 · local
        </span>
      </div>
    </aside>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="flex flex-col gap-0.5">
      <dt className="text-nx-fg/35 uppercase tracking-[0.14em] text-[0.55rem]">
        {label}
      </dt>
      <dd className={cn("text-[0.8rem]", accent ? "text-nx-accent" : "text-nx-fg/90")}>
        {value}
      </dd>
    </div>
  );
}
