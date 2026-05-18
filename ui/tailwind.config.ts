/**
 * Tailwind config — Tailwind v4 reads design tokens from CSS @theme blocks
 * (see `app/globals.css`). This file is kept for reference and as a fallback
 * if we ever opt back into the v3 config-driven setup. The authoritative
 * source for `nx-*` and `class-*` colors + font families is `app/globals.css`.
 *
 * Plan reference: `plan/ui/ULTRAPLAN.md` §0.2 (design tokens).
 */
import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx,mdx}",
    "./components/**/*.{ts,tsx,mdx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        "nx-bg": "#000000",
        "nx-fg": "#FFFFFF",
        "nx-accent": "#39B8DC",
        "nx-accent-deep": "#135A6F",
        "nx-bg-elev-1": "#04161B",
        "nx-bg-elev-2": "#0C3845",
        "nx-muted": "#313131",
        "nx-danger": "#FF0000",
        "class-stec": "#D63333",
        "class-nonstec": "#1F7A4D",
        "class-salm": "#7A3D99",
        "class-h2o": "#3070B5",
      },
      fontFamily: {
        display: ["General Sans Variable", "system-ui", "sans-serif"],
        body: ["General Sans Variable", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono Variable", "ui-monospace", "monospace"],
      },
    },
  },
};

export default config;
