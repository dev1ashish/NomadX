/**
 * Inventory tab — typed sidecar fetcher.
 *
 * Mirrors the shape emitted by `ui/scripts/build_inventory.py`:
 *
 *   {
 *     totals: { n_files, n_spectra, n_bins, plsda_loso, logreg_fw },
 *     files: FileMeta[],
 *   }
 *
 * Kept local to the Inventory feature (not in `lib/data.ts`) because the
 * shape is feature-specific — `lib/data.ts` already declares `getInventory`
 * returning the bare array form for future generic use.
 */
import type { FileMeta } from "@/lib/types";

export type InventoryFile = FileMeta;

export interface InventoryTotals {
  n_files: number;
  n_spectra: number;
  n_bins: number;
  plsda_loso: number;
  logreg_fw: number;
}

export interface InventorySidecar {
  totals: InventoryTotals;
  files: InventoryFile[];
}

export async function fetchInventorySidecar(): Promise<InventorySidecar> {
  const res = await fetch("/data/inventory.json", { cache: "force-cache" });
  if (!res.ok) {
    throw new Error(
      `Failed to fetch /data/inventory.json: ${res.status} ${res.statusText}`,
    );
  }
  return (await res.json()) as InventorySidecar;
}
