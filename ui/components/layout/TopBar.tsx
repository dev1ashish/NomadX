/**
 * NomadX top-bar wordmark — cyan diamond + white text + version tag.
 * Plan ref: §3 ASCII mockup.
 */
export function TopBar() {
  return (
    <header className="border-b border-nx-muted bg-nx-bg">
      <div className="mx-auto flex max-w-screen-2xl items-center justify-between px-6 py-4">
        <div className="flex items-baseline gap-3 font-display">
          <span
            aria-hidden
            className="text-nx-accent text-lg leading-none"
            style={{ transform: "translateY(2px)" }}
          >
            ◆
          </span>
          <span className="text-nx-fg text-sm font-medium tracking-[0.18em] uppercase">
            NOMADX
          </span>
          <span className="text-nx-muted">·</span>
          <span className="text-nx-fg text-sm font-medium tracking-[0.18em] uppercase">
            ATLAS RAMAN
          </span>
        </div>
        <span className="font-mono text-[0.75rem] text-nx-fg/60">
          v0.1 · local
        </span>
      </div>
    </header>
  );
}
