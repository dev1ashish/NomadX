#!/usr/bin/env bash
# One command to DEMO the deliverable: prints the dataset summary, then opens
# the human-friendly CSV (Excel/Numbers) and the feature_store/ folder (Finder).
#
#   ./orchestration/show_dataset.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

"$REPO/.venv/bin/python" "$REPO/orchestration/show_dataset.py" || exit 1

# Open the spreadsheet + folder so the audience SEES the table (macOS `open`).
if [[ -f feature_store/sample_for_humans.csv ]]; then
  open feature_store/sample_for_humans.csv
  open feature_store
fi
