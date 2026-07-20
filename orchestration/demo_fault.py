"""Reversible fault injection for the live demo.

The "ohhh that's what's up" moment: break the data contract and watch the
blocking quality gate turn RED and stop bad data from reaching the data
scientist — then restore and watch it go green again.

  python orchestration/demo_fault.py inject    # drop 5 rows from band_features
  python orchestration/demo_fault.py restore    # put it back

What it does: backs up band_features.parquet, then writes a version with 5
fewer rows. This violates the contract invariant `band rows == qc_mask.sum()`,
so the BLOCKING check `band_rows_equal_qc_keep` fails and `feature_store_contract`
(downstream) never runs. Fully reversible; touches only a regenerable cache file.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "data_cache"
TARGET = CACHE / "band_features.parquet"
BACKUP = CACHE / "band_features.parquet.demo_bak"


def inject() -> None:
    if BACKUP.exists():
        print("Already injected (backup exists). Run 'restore' first.")
        return
    df = pd.read_parquet(TARGET)
    shutil.copy2(TARGET, BACKUP)
    df.iloc[:-5].to_parquet(TARGET, compression="snappy")
    print(f"INJECTED fault: band_features {len(df)} -> {len(df) - 5} rows.")
    print("Now re-materialize in the UI → 'band_rows_equal_qc_keep' will go RED")
    print("and block feature_store_contract. Backup at:", BACKUP.name)


def restore() -> None:
    if not BACKUP.exists():
        print("No backup found — nothing to restore.")
        return
    shutil.move(str(BACKUP), str(TARGET))
    print(f"RESTORED band_features ({len(pd.read_parquet(TARGET))} rows). Re-materialize → green.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "inject":
        inject()
    elif cmd == "restore":
        restore()
    else:
        print(__doc__)
        sys.exit(2)
