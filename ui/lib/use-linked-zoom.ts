"use client";

/**
 * Sync Plotly xaxis.range across multiple chart instances.
 *
 * Each chart registers itself with a stable key (e.g., file_id) by storing
 * the Plotly div ref. When any chart fires a relayout containing an x-axis
 * range change, the hook broadcasts that range to every other registered
 * chart via `Plotly.relayout`.
 *
 * Usage:
 *   const { register, onRelayout } = useLinkedZoom(linkEnabled);
 *   <Plot
 *     onInitialized={(_, gd) => register(fileId, gd)}
 *     onUpdate={(_, gd) => register(fileId, gd)}
 *     onRelayout={(ev) => onRelayout(fileId, ev)}
 *   />
 */
import { useCallback, useRef } from "react";
import type { PlotRelayoutEvent } from "plotly.js";

type PlotlyGlobal = {
  relayout: (gd: HTMLElement, update: Record<string, unknown>) => Promise<void>;
};

declare global {
  interface Window {
    Plotly?: PlotlyGlobal;
  }
}

export function useLinkedZoom(enabled: boolean) {
  const figs = useRef<Map<string, HTMLElement>>(new Map());
  const broadcasting = useRef(false);

  const register = useCallback((key: string, el: HTMLElement | null) => {
    if (!el) {
      figs.current.delete(key);
      return;
    }
    figs.current.set(key, el);
  }, []);

  const onRelayout = useCallback(
    (key: string, ev: Readonly<PlotRelayoutEvent>) => {
      if (!enabled || broadcasting.current) return;

      const xRange = extractXRange(ev);
      if (!xRange) return;

      const Plotly = window.Plotly;
      if (!Plotly) return;

      broadcasting.current = true;
      for (const [k, el] of figs.current) {
        if (k === key) continue;
        void Plotly.relayout(el, { "xaxis.range": xRange });
      }
      // Release on next tick so cascading relayouts from peers are ignored.
      setTimeout(() => {
        broadcasting.current = false;
      }, 0);
    },
    [enabled],
  );

  return { register, onRelayout };
}

function extractXRange(
  ev: Readonly<PlotRelayoutEvent>,
): [number, number] | null {
  const pair = (ev as Record<string, unknown>)["xaxis.range"];
  if (Array.isArray(pair) && pair.length === 2) {
    return [Number(pair[0]), Number(pair[1])];
  }
  const lo = (ev as Record<string, unknown>)["xaxis.range[0]"];
  const hi = (ev as Record<string, unknown>)["xaxis.range[1]"];
  if (typeof lo === "number" && typeof hi === "number") {
    return [lo, hi];
  }
  return null;
}
