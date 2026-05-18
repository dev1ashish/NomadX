/**
 * Pure normalization helpers for the Comparison Lab.
 *
 * All ops take a 1-D intensity vector and return a new 1-D vector of the
 * same length. NaN-free input is assumed (spectra sidecars are QC-clean).
 */
import type { NormalizationMode } from "./types";

export function meanCenter(y: number[]): number[] {
  const mean = y.reduce((s, v) => s + v, 0) / y.length;
  return y.map((v) => v - mean);
}

export function snv(y: number[]): number[] {
  const mean = y.reduce((s, v) => s + v, 0) / y.length;
  const variance =
    y.reduce((s, v) => s + (v - mean) * (v - mean), 0) / y.length;
  const std = Math.sqrt(variance);
  if (std === 0) return y.map(() => 0);
  return y.map((v) => (v - mean) / std);
}

export function minMax(y: number[]): number[] {
  let min = Infinity;
  let max = -Infinity;
  for (const v of y) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min;
  if (range === 0) return y.map(() => 0);
  return y.map((v) => (v - min) / range);
}

export function applyNormalization(
  y: number[],
  mode: NormalizationMode,
): number[] {
  switch (mode) {
    case "snv":
      return snv(y);
    case "minmax":
      return minMax(y);
    case "mean_center":
      return meanCenter(y);
    case "raw":
      return y;
  }
}
