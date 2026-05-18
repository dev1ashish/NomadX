"""
Build live_lookup.json — per-file ground truth + recorded LOSO predictions
for all 87 corpus files. Used by the Live tab to surface ground-truth
verification AND the comparable PLS-DA / LogReg / XGB predictions from the
training run when a known file is dropped.

Sources:
  artifacts/stage15f_loso_predictions.parquet
  artifacts/stage15f_loso_summary.csv  (for per-strain stats)

Output:
  ui/public/data/live_lookup.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"
OUT = REPO_ROOT / "ui" / "public" / "data" / "live_lookup.json"

ALGOS = ["plsda", "logreg", "xgb"]
CLASS_KEYS = ["H2O", "Non-STEC", "STEC", "Salmonella"]


def main() -> None:
    preds = pd.read_parquet(ARTIFACTS / "stage15f_loso_predictions.parquet")
    print(f"Loaded {len(preds)} prediction rows")

    # Pivot: one row per file, columns are per-algo predictions.
    by_file: dict[str, dict] = {}
    for _, r in preds.iterrows():
        fid = str(r["file_id"])
        if fid not in by_file:
            by_file[fid] = {
                "file_id": fid,
                "true_class": str(r["y_true"]),
                "fold": str(r["fold"]),
                "algos": {},
            }
        algo = str(r["algo"])
        probs = {
            "H2O": float(r["proba_H2O"]),
            "Non-STEC": float(r["proba_Non-STEC"]),
            "STEC": float(r["proba_STEC"]),
            "Salmonella": float(r["proba_Salmonella"]),
        }
        by_file[fid]["algos"][algo] = {
            "predicted": str(r["y_pred"]),
            "correct": str(r["y_pred"]) == str(r["y_true"]),
            "probs": {k: round(v, 4) for k, v in probs.items()},
            "top_prob": round(probs[str(r["y_pred"])], 4),
        }

    # Add per-file consensus + difficulty score
    for fid, entry in by_file.items():
        algos_present = list(entry["algos"].keys())
        correct_count = sum(1 for a in algos_present if entry["algos"][a]["correct"])
        entry["consensus_correct_n"] = correct_count
        entry["consensus_total_n"] = len(algos_present)
        # Difficulty: all algos wrong = hard, all right = easy
        if correct_count == 0:
            entry["difficulty"] = "hard"  # all algos failed
        elif correct_count == len(algos_present):
            entry["difficulty"] = "easy"  # all algos correct
        else:
            entry["difficulty"] = "mixed"  # disagreement among algos

    payload = {
        "files": by_file,
        "algos": ALGOS,
        "classes": CLASS_KEYS,
        "meta": {
            "n_files": len(by_file),
            "easy_files": sum(1 for e in by_file.values() if e["difficulty"] == "easy"),
            "mixed_files": sum(1 for e in by_file.values() if e["difficulty"] == "mixed"),
            "hard_files": sum(1 for e in by_file.values() if e["difficulty"] == "hard"),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=None, separators=(",", ":")))

    print(f"Wrote {OUT}")
    print(f"  files:       {payload['meta']['n_files']}")
    print(f"  easy:        {payload['meta']['easy_files']} (all 3 algos correct)")
    print(f"  mixed:       {payload['meta']['mixed_files']} (algos disagree)")
    print(f"  hard:        {payload['meta']['hard_files']} (all 3 algos wrong)")
    print(f"  size:        {OUT.stat().st_size / 1024:.1f} KB")

    # Spot-check a few files
    for fid_prefix in ["R357", "R364", "R370", "R372"]:
        matching = [k for k in by_file if k.startswith(fid_prefix)]
        if matching:
            entry = by_file[matching[0]]
            print()
            print(f"  --- {fid_prefix} ---")
            print(f"    true: {entry['true_class']}, fold: {entry['fold']}, difficulty: {entry['difficulty']}")
            for algo, a in entry["algos"].items():
                mark = "✓" if a["correct"] else "✗"
                print(f"    {algo:7s} {mark} {a['predicted']:12s} (top_prob={a['top_prob']})")


if __name__ == "__main__":
    main()
