"""Parse every file under Atlas Data/ and cache results to data_cache/.

Idempotency: if data_cache/metadata.parquet exists and every file's (mtime, sha256)
matches the cache entry, we exit without re-parsing.

Run:
    .venv/bin/python scripts/build_dataset.py
    .venv/bin/python scripts/build_dataset.py --force          # rebuild
    .venv/bin/python scripts/build_dataset.py --no-cap         # skip 200-px cap
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from atlas.io import (  # noqa: E402
    discover_files,
    parse_dataset,
    write_cache,
    _sha256_of,
)


def _cache_is_fresh(data_root: Path, cache_dir: Path) -> bool:
    """True if cache is up-to-date for every discovered file."""
    meta_path = cache_dir / "metadata.parquet"
    if not meta_path.exists():
        return False
    import pandas as pd

    try:
        meta = pd.read_parquet(meta_path)
    except Exception:
        return False
    existing = {row["file_path"]: row for _, row in meta.iterrows()}
    for p in discover_files(data_root):
        rel = str(p.relative_to(data_root))
        if rel not in existing:
            return False
        st = p.stat()
        row = existing[rel]
        if abs(float(row["file_mtime"]) - st.st_mtime) > 1e-6:
            return False
        if row["file_size_bytes"] != st.st_size:
            return False
        # cheapest valid check passed — only hash if size/mtime tied
        if row["file_sha256"] != _sha256_of(p):
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(REPO_ROOT / "Atlas Data"))
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "data_cache"))
    ap.add_argument("--force", action="store_true", help="rebuild even if fresh")
    ap.add_argument(
        "--pixel-cap",
        type=int,
        default=200,
        help="Per-file pixel cap (set to 0 to disable).",
    )
    ap.add_argument("--no-cap", action="store_true", help="disable pixel cap")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    cache_dir = Path(args.cache_dir)

    if not data_root.exists():
        print(f"ERROR: data root not found: {data_root}", file=sys.stderr)
        return 2

    if not args.force and _cache_is_fresh(data_root, cache_dir):
        print(f"cache is fresh at {cache_dir} — nothing to do (use --force to rebuild)")
        return 0

    pixel_cap = None if args.no_cap or args.pixel_cap == 0 else args.pixel_cap

    print(f"data_root  = {data_root}")
    print(f"cache_dir  = {cache_dir}")
    print(f"pixel_cap  = {pixel_cap}")
    print(f"seed       = {args.seed}")

    files = discover_files(data_root)
    print(f"discovered {len(files)} files")

    records = parse_dataset(
        data_root, pixel_cap=pixel_cap, seed=args.seed, progress=True
    )

    summary = write_cache(records, cache_dir)
    print("---")
    print(json.dumps(summary, indent=2))

    # Failure surfacing
    fatal = [r for r in records if not r.is_valid]
    warned = [r for r in records if r.warnings and r.is_valid]
    if fatal:
        print("\nFATAL files (would be excluded from training):")
        for r in fatal:
            print(f"  {r.file_path}  errors={r.fatal_errors}")
    if warned:
        print(f"\nFiles with warnings (still loaded): {len(warned)}")
        for r in warned[:20]:
            print(f"  {r.file_path}  warnings={r.warnings}")
        if len(warned) > 20:
            print(f"  ... and {len(warned) - 20} more (see build.log)")
    if not fatal and not warned:
        print("\nclean parse: 0 fatal, 0 warnings")
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
