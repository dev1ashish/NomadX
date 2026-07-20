"""Print a friendly summary of the prepared feature-store dataset.

Run:  .venv/bin/python orchestration/show_dataset.py
(or via the wrapper: ./orchestration/show_dataset.sh, which also opens the CSV)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FS = Path(__file__).resolve().parent.parent / "feature_store"
PX = FS / "pixel_training_matrix.parquet"
FILE = FS / "file_level_table.parquet"

ID_COLS = ["file_id", "primary_class", "subclass", "pixel_idx", "x_um", "y_um"]


def main() -> int:
    if not PX.exists():
        print("Dataset not built yet. First run the pipeline and 'Materialize all',")
        print("or:  ./orchestration/demo_pipeline.sh   (then Materialize all in the UI)")
        return 1

    df = pd.read_parquet(PX)
    feats = [c for c in df.columns if c not in ID_COLS]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 10)

    print("=" * 70)
    print("PREPARED DATASET FOR THE DATA SCIENTIST")
    print("=" * 70)
    print(f"file : {FS.name}/pixel_training_matrix.parquet")
    print(f"shape: {df.shape[0]:,} rows x {df.shape[1]} cols "
          f"(6 identity/label + {len(feats)} features)")
    print()
    print("label (y = primary_class) — what the model predicts:")
    for k, v in df["primary_class"].value_counts().items():
        print(f"    {k:<12} {v:>5}")
    print(f"    groups (file_id): {df['file_id'].nunique()} distinct samples (use for CV)")
    print()
    print("first 6 rows (identity/label + 3 example features):")
    show = ["file_id", "primary_class", "pixel_idx"] + feats[:3]
    print(df[show].head(6).to_string(index=False))
    print()
    if FILE.exists():
        f = pd.read_parquet(FILE)
        print(f"also: {FS.name}/file_level_table.parquet — {f.shape[0]} rows "
              f"(one per sample) x {f.shape[1] + 1} cols")
    print(f"human-friendly CSV: {FS.name}/sample_for_humans.csv  (opens in Excel/Numbers)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
