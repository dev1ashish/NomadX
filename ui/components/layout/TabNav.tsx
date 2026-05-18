"use client";

/**
 * 7-tab navigation bar.
 * Active tab gets a cyan underline that animates between tabs via
 * Framer Motion `layoutId="tabActive"`.
 * Plan ref: §3 ASCII mockup, §4 W1.
 */
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { cn } from "@/lib/cn";

interface Tab {
  href: string;
  label: string;
}

const TABS: Tab[] = [
  { href: "/inventory", label: "Inventory" },
  { href: "/spectrum", label: "Spectrum" },
  { href: "/compare", label: "Compare" },
  { href: "/primer", label: "Primer" },
  { href: "/features", label: "Features" },
  { href: "/mcr", label: "MCR-ALS" },
  { href: "/live", label: "Live" },
  { href: "/results", label: "Results" },
];

export function TabNav() {
  const pathname = usePathname();

  return (
    <nav
      aria-label="Atlas Raman tabs"
      className="border-b border-nx-muted bg-nx-bg"
    >
      <ul className="mx-auto flex max-w-screen-2xl items-stretch gap-1 px-6">
        {TABS.map((tab) => {
          const active = pathname === tab.href || pathname.startsWith(`${tab.href}/`);
          return (
            <li key={tab.href} className="relative">
              <Link
                href={tab.href}
                className={cn(
                  "relative inline-block px-4 py-3 text-sm tracking-wide transition-colors",
                  active
                    ? "text-nx-fg"
                    : "text-nx-fg/60 hover:text-nx-fg",
                )}
              >
                {tab.label}
                {active ? (
                  <motion.span
                    layoutId="tabActive"
                    className="absolute inset-x-2 -bottom-px h-[2px] bg-nx-accent"
                    transition={{ type: "spring", stiffness: 380, damping: 32 }}
                  />
                ) : null}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
