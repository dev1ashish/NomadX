"""Atlas Raman UI — Inventory sidecar builder (W2).

Reads `data_cache/metadata.parquet` + `data_cache/qc_info.json` and emits
`ui/public/data/inventory.json` for the Inventory tab.

Plan reference: `plan/ui/ULTRAPLAN.md` §4 W2.

Run from the project root or `ui/`:
    cd ui && python scripts/build_inventory.py
    # or with uv:
    cd ui && uv run scripts/build_inventory.py

# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "pyarrow"]
# ///
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Stage 15F project headlines — frozen in PAPER.md §3.1 / §6.
PLSDA_LOSO = 0.603
LOGREG_FW = 0.448
N_BINS = 987


def main() -> None:
    here = Path(__file__).resolve().parent
    ui_root = here.parent
    repo_root = ui_root.parent

    metadata_path = repo_root / "data_cache" / "metadata.parquet"
    qc_path = repo_root / "data_cache" / "qc_info.json"
    out_path = ui_root / "public" / "data" / "inventory.json"

    if not metadata_path.exists():
        raise SystemExit(f"missing {metadata_path}")
    if not qc_path.exists():
        raise SystemExit(f"missing {qc_path}")

    md = pd.read_parquet(metadata_path)
    with qc_path.open() as f:
        qc = json.load(f)
    per_file: dict[str, dict] = qc["per_file"]

    # Per-file QC pass rate = kept / n  (NOT the kept/n_input ratio that
    # `qc_info.json` mis-labels as `retention`).
    files: list[dict] = []
    for row in md.sort_values("file_id").itertuples(index=False):
        pf = per_file.get(row.file_id, {})
        n = int(pf.get("n", 0)) or int(row.n_pixels)
        kept = int(pf.get("kept", 0))
        pass_rate = (kept / n) if n > 0 else 0.0
        subclass = None if (row.subclass is None or pd.isna(row.subclass)) else str(row.subclass)
        files.append(
            {
                "file_id": str(row.file_id),
                "primary_class": str(row.primary_class),
                "subclass": subclass,
                "n_pixels": int(row.n_pixels),
                "qc_pass_rate": round(float(pass_rate), 4),
            }
        )

    payload = {
        "totals": {
            "n_files": int(len(files)),
            "n_spectra": int(qc.get("n_keep", sum(int(pf.get("kept", 0)) for pf in per_file.values()))),
            "n_bins": N_BINS,
            "plsda_loso": PLSDA_LOSO,
            "logreg_fw": LOGREG_FW,
        },
        "files": files,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(
        f"wrote {out_path} — "
        f"{payload['totals']['n_files']} files, "
        f"{payload['totals']['n_spectra']} spectra, "
        f"{out_path.stat().st_size} bytes"
    )


if __name__ == "__main__":
    main()
