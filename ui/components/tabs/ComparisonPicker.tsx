"use client";

/**
 * Modal picker for the Comparison Lab. Reads inventory.json, lets the user
 * stage a file into one of the three role lanes (+ve / blank / test).
 *
 * Soft cap of 12 staged spectra; over-cap, the modal disables further adds
 * and surfaces a hint to switch to Heatmap view.
 *
 * Multi-select UX: stays open after each pick, staged rows are clickable to
 * remove, class-filter chips narrow the list, Done button dismisses.
 */
import { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Check, X } from "lucide-react";
import { cn } from "@/lib/cn";
import { getInventory } from "@/lib/data";
import type { ClassName, FileMeta } from "@/lib/types";
import type { ComparisonRole, StagedSpectrum } from "@/lib/types";

const SOFT_CAP = 12;

const CLASS_FILTERS: { key: ClassName | "all"; label: string }[] = [
  { key: "all", label: "All" },
  { key: "STEC", label: "STEC" },
  { key: "Non-STEC", label: "Non-STEC" },
  { key: "Salmonella", label: "Salmonella" },
  { key: "H2O", label: "H2O" },
];

interface ComparisonPickerProps {
  open: boolean;
  onClose: () => void;
  staged: StagedSpectrum[];
  onStage: (next: StagedSpectrum) => void;
  onUnstage: (file_id: string) => void;
}

export function ComparisonPicker({
  open,
  onClose,
  staged,
  onStage,
  onUnstage,
}: ComparisonPickerProps) {
  const [files, setFiles] = useState<FileMeta[] | null>(null);
  const [query, setQuery] = useState("");
  const [defaultRole, setDefaultRole] = useState<ComparisonRole>("test");
  const [classFilter, setClassFilter] = useState<ClassName | "all">("all");

  useEffect(() => {
    if (!open || files) return;
    getInventory()
      .then((inv) => setFiles(inv.files))
      .catch(() => setFiles([]));
  }, [open, files]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  const stagedIds = useMemo(
    () => new Set(staged.map((s) => s.file_id)),
    [staged],
  );

  const filtered = useMemo(() => {
    if (!files) return [];
    let list = files;
    if (classFilter !== "all") {
      list = list.filter((f) => f.primary_class === classFilter);
    }
    const q = query.trim().toLowerCase();
    if (!q) return list;
    return list.filter(
      (f) =>
        f.file_id.toLowerCase().includes(q) ||
        (f.subclass ?? "").toLowerCase().includes(q) ||
        f.primary_class.toLowerCase().includes(q),
    );
  }, [files, query, classFilter]);

  const grouped = useMemo(() => {
    const out = new Map<string, FileMeta[]>();
    for (const f of filtered) {
      const k = f.primary_class;
      const arr = out.get(k) ?? [];
      arr.push(f);
      out.set(k, arr);
    }
    return out;
  }, [filtered]);

  const atCap = staged.length >= SOFT_CAP;

  return (
    <AnimatePresence>
      {open ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.96, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.96, opacity: 0 }}
            transition={{ duration: 0.18 }}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-2xl max-h-[80vh] flex flex-col rounded-md border border-nx-muted bg-nx-bg-elev-1 text-nx-fg"
          >
            {/* Header */}
            <header className="flex items-center justify-between px-5 py-3 border-b border-nx-muted">
              <div className="flex flex-col">
                <h2 className="font-display text-base">Add spectrum</h2>
                <p className="font-mono text-[0.65rem] text-nx-fg/55">
                  {staged.length}/{SOFT_CAP} staged
                </p>
              </div>
              <button
                aria-label="Close picker"
                onClick={onClose}
                className="text-nx-fg/55 hover:text-nx-fg transition-colors"
              >
                <X className="size-4" />
              </button>
            </header>

            {/* Class-filter chips */}
            <div className="px-5 pt-3 pb-1 flex flex-wrap gap-2">
              {CLASS_FILTERS.map((chip) => (
                <button
                  key={chip.key}
                  onClick={() => setClassFilter(chip.key)}
                  className={cn(
                    "px-2.5 py-0.5 rounded-full font-mono text-[0.65rem] border transition-colors",
                    classFilter === chip.key
                      ? "bg-nx-accent/15 text-nx-accent border-nx-accent/40"
                      : "text-nx-fg/55 hover:text-nx-fg border-transparent hover:border-nx-muted/60",
                  )}
                >
                  {chip.label}
                </button>
              ))}
            </div>

            {/* Search + role select */}
            <div className="px-5 py-3 flex items-center gap-3 border-b border-nx-muted/60">
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search file_id, subclass, class…"
                className="flex-1 bg-nx-bg px-3 py-2 rounded-sm text-sm border border-nx-muted/60 focus:border-nx-accent outline-none"
              />
              <select
                value={defaultRole}
                onChange={(e) =>
                  setDefaultRole(e.target.value as ComparisonRole)
                }
                className="bg-nx-bg px-2 py-2 rounded-sm text-sm border border-nx-muted/60"
              >
                <option value="test">Test</option>
                <option value="control_pos">+ve control</option>
                <option value="blank">Blank</option>
              </select>
            </div>

            {/* Scrollable list */}
            <div className="flex-1 overflow-y-auto px-5 py-3">
              {atCap ? (
                <p className="text-amber-300 font-mono text-[0.7rem] mb-3">
                  Soft cap reached. Switch to Heatmap view for &gt;12 spectra,
                  or click a staged file to remove it before adding more.
                </p>
              ) : null}
              {files === null ? (
                <p className="font-mono text-xs text-nx-fg/55">Loading…</p>
              ) : grouped.size === 0 ? (
                <p className="font-mono text-xs text-nx-fg/55">No matches.</p>
              ) : (
                Array.from(grouped.entries()).map(([cls, items]) => (
                  <section key={cls} className="mb-4">
                    <h3 className="font-mono text-[0.6rem] text-nx-fg/45 uppercase tracking-[0.18em] mb-1">
                      {cls} · {items.length}
                    </h3>
                    <ul className="flex flex-col gap-1">
                      {items.map((f) => {
                        const already = stagedIds.has(f.file_id);
                        // Staged rows: always clickable (to remove).
                        // Unstaged rows: clickable only if !atCap.
                        const disabled = !already && atCap;
                        return (
                          <li key={f.file_id}>
                            <button
                              disabled={disabled}
                              onClick={() => {
                                if (already) {
                                  onUnstage(f.file_id);
                                } else {
                                  onStage({
                                    file_id: f.file_id,
                                    role: defaultRole,
                                    display_label: f.file_id,
                                    visible: true,
                                  });
                                }
                              }}
                              className={cn(
                                "w-full flex items-center justify-between px-3 py-2 text-left rounded-sm border border-transparent transition-colors",
                                disabled
                                  ? "opacity-40 cursor-not-allowed"
                                  : already
                                    ? "hover:bg-red-500/10 hover:border-red-500/30 cursor-pointer"
                                    : "hover:bg-nx-bg-elev-2/60 hover:border-nx-muted/60",
                              )}
                            >
                              <span className="flex flex-col">
                                <span className="font-mono text-xs">
                                  {f.file_id}
                                </span>
                                <span className="font-mono text-[0.6rem] text-nx-fg/45">
                                  {f.subclass ?? "—"} · {f.n_pixels} px · QC{" "}
                                  {Math.round(f.qc_pass_rate * 100)}%
                                </span>
                              </span>
                              {already ? (
                                <span className="flex items-center gap-1.5 font-mono text-[0.6rem] text-nx-accent group-hover:hidden">
                                  <Check className="size-3" />
                                  staged
                                </span>
                              ) : null}
                            </button>
                          </li>
                        );
                      })}
                    </ul>
                  </section>
                ))
              )}
            </div>

            {/* Sticky footer */}
            <footer className="flex items-center justify-between px-5 py-3 border-t border-nx-muted/60 bg-nx-bg-elev-1">
              <span className="font-mono text-[0.65rem] text-nx-fg/55">
                {staged.length} staged
              </span>
              <button
                onClick={onClose}
                className="flex items-center gap-2 px-3 py-2 rounded-sm border border-nx-accent text-nx-accent hover:bg-nx-accent/10 text-sm font-mono transition-colors"
              >
                Done
              </button>
            </footer>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
