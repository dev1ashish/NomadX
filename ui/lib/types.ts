/**
 * Atlas Raman — shared TypeScript contracts.
 * Mirrors the sidecar JSON shape emitted by `ui/scripts/build_sidecars.py`
 * and the Modal `/predict` response shape from `inference_api/modal_app.py`.
 *
 * See plan/ui/ULTRAPLAN.md §W1 for the canonical contract.
 */

export type ClassName = "STEC" | "Non-STEC" | "Salmonella" | "H2O";

export interface FileMeta {
  file_id: string;
  primary_class: ClassName;
  subclass: string | null;
  n_pixels: number;
  qc_pass_rate: number;
}

export interface Band {
  name: string;
  center: number;
  fwhm: number;
  group: string;
  chemistry: string;
  d_stec_nonstec?: number;
}

export interface Feature {
  name: string;
  family: "band" | "spectral" | "mcr" | "spatial" | "bio";
  region?: string;
  d_stec_nonstec?: number;
  d_ecoli_salm?: number;
  mi_rank_stage15f?: number | null;
}

export interface PredictionResponse {
  class: ClassName;
  probabilities: Record<ClassName, number>;
  spectrum_mean: number[];
  wn: number[];
  feature_values: Record<string, number>;
}

// ──────────────────────────────────────────────────────────────────
// Comparison Lab (see docs/superpowers/specs/2026-05-19-comparison-lab-design.md)
// ──────────────────────────────────────────────────────────────────

export type ComparisonRole = "control_pos" | "blank" | "test";

export interface StagedSpectrum {
  /** Resolves to `/data/spectra/<file_id>.json`. */
  file_id: string;
  role: ComparisonRole;
  /** Editable label shown in legends/badges. Defaults to file_id at stage time. */
  display_label: string;
  /** Toggle visibility per-trace in Overlay view. */
  visible: boolean;
  /** Hex color override; falls back to class color from plotly-theme. */
  color_override?: string;
}

export type ComparisonView =
  | "grid"
  | "overlay"
  | "waterfall"
  | "heatmap"
  | "diff";

export type NormalizationMode = "snv" | "minmax" | "raw" | "mean_center";

export type RegionPreset =
  | "full"
  | "fingerprint_800_1800"
  | "lps_400_900"
  | "lps_800_1200";

/** Per-region wavenumber windows in cm^-1. */
export const REGION_RANGES: Record<RegionPreset, [number, number] | null> = {
  full: null,
  fingerprint_800_1800: [800, 1800],
  lps_400_900: [400, 900],
  lps_800_1200: [800, 1200],
};

/** Shape of the per-file spectrum sidecar at /data/spectra/<file_id>.json. */
export interface SpectrumSidecar {
  file_id: string;
  primary_class: ClassName;
  subclass: string | null;
  n_pixels: number;
  n_qc_pass: number;
  wn_raw: number[];
  wn_pp: number[];
  mean_raw: number[];
  mean_pp: number[];
}
