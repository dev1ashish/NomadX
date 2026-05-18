import { Suspense } from "react";
import { SpectrumExplorer } from "@/components/tabs/SpectrumExplorer";

function SpectrumExplorerFallback() {
  return (
    <section className="mx-auto max-w-screen-2xl px-6 py-10">
      <h2 className="font-display text-nx-accent text-[3.0625rem] leading-[1.1]">
        Spectrum
      </h2>
      <p className="mt-4 text-nx-fg/60">Loading…</p>
    </section>
  );
}

export default function SpectrumPage() {
  return (
    <Suspense fallback={<SpectrumExplorerFallback />}>
      <SpectrumExplorer />
    </Suspense>
  );
}
