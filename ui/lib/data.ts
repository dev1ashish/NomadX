/**
 * Typed fetchers for the static sidecar JSON shipped under `public/data/`.
 *
 * Sidecars are built offline by `scripts/build_*.py` (per-tab) — the UI never
 * re-derives them at runtime. Each tab usually has a local payload type and
 * uses `getSidecar<T>("file.json")`; the named fetchers below cover the
 * shared shell-level sidecars (inventory + bands + features).
 */
import type { FileMeta, Band, Feature } from "./types";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path, { cache: "force-cache" });
  if (!res.ok) {
    throw new Error(`Failed to fetch ${path}: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

/** `inventory.json` payload — emitted by `scripts/build_inventory.py`. */
export interface InventorySidecar {
  totals: {
    n_files: number;
    n_spectra: number;
    n_bins: number;
    plsda_loso: number;
    logreg_fw: number;
  };
  files: FileMeta[];
}

/** `bands.json` payload — emitted by `scripts/build_bands.py`. */
export interface BandsSidecar {
  groups: Array<{
    key: string;
    label: string;
    biology: string;
    bands: Band[];
  }>;
  cisek_falsification: {
    headline: string;
    bands: Array<{
      center: number;
      label: string;
      literature_claim: string;
      atlas_d: number;
      verdict: "null" | "sign-reversed" | "confirmed";
    }>;
  };
  anchors: Record<
    string,
    { region: [number, number]; label: string; top_band: string; top_d: number }
  >;
}

/** `feature_catalog.json` payload — emitted by `scripts/build_features.py`. */
export interface FeatureCatalogSidecar {
  features: Feature[];
  per_class_stats: Record<
    string,
    Record<string, { mean: number; std: number; n: number }>
  >;
  top_15_stec_nonstec: string[];
  stage15f_35: string[];
}

export const getInventory = (): Promise<InventorySidecar> =>
  getJson<InventorySidecar>("/data/inventory.json");

export const getBands = (): Promise<BandsSidecar> =>
  getJson<BandsSidecar>("/data/bands.json");

export const getFeatureCatalog = (): Promise<FeatureCatalogSidecar> =>
  getJson<FeatureCatalogSidecar>("/data/feature_catalog.json");

/** Generic escape hatch for tab-specific sidecars (confusion.json, bootstrap.json,
 *  mcnemar.json, stage15f.json, mcr_components.json, spectra/*.json). */
export const getSidecar = <T>(filename: string): Promise<T> =>
  getJson<T>(`/data/${filename}`);
