"""Atlas Raman UI — sidecar builder (STUB).

This script is the single source of `ui/public/data/*.json`. Each tab worker
in W2–W8 will add their own emitter here (parquet/npy/json → flat JSON files)
so the Next.js frontend never reads upstream caches at runtime.

Plan reference: `plan/ui/ULTRAPLAN.md` §2 + §W1.

Usage (one-time, idempotent):
    cd ui && python scripts/build_sidecars.py
    # or, with uv:
    cd ui && uv run scripts/build_sidecars.py
"""

from __future__ import annotations


def main() -> None:
    print(
        "Sidecar build not yet implemented — "
        "will be filled in per tab in W2-W8."
    )


if __name__ == "__main__":
    main()
