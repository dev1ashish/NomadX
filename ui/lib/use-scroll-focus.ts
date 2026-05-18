"use client";

/**
 * Drive a Framer Motion-friendly `scale` (0.85 → 1.0) and `opacity`
 * (0.6 → 1.0) based on how close a row's center is to the viewport center.
 *
 * Used by ComparisonGrid rows to give a focus-on-scroll feel.
 */
import { useEffect, useRef, useState } from "react";

export interface ScrollFocusState {
  scale: number;
  opacity: number;
}

export function useScrollFocus<T extends HTMLElement>(): {
  ref: React.RefObject<T | null>;
  state: ScrollFocusState;
} {
  const ref = useRef<T>(null);
  const [state, setState] = useState<ScrollFocusState>({
    scale: 1,
    opacity: 1,
  });

  useEffect(() => {
    const el = ref.current;
    if (!el || typeof window === "undefined") return;

    let raf = 0;

    const update = () => {
      raf = 0;
      const rect = el.getBoundingClientRect();
      const rowCenter = rect.top + rect.height / 2;
      const viewportCenter = window.innerHeight / 2;
      // Distance from viewport center, normalized by half-viewport-height.
      const d = Math.min(
        1,
        Math.abs(rowCenter - viewportCenter) / (window.innerHeight / 2),
      );
      const scale = 1 - d * 0.15; // 1.0 → 0.85
      const opacity = 1 - d * 0.4; // 1.0 → 0.6
      setState({ scale, opacity });
    };

    const onScroll = () => {
      if (raf) return;
      raf = requestAnimationFrame(update);
    };

    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  return { ref, state };
}
