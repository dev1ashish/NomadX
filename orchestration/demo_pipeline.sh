#!/usr/bin/env bash
# One command → the Atlas data-engineering pipeline UI.
#
#   ./orchestration/demo_pipeline.sh
#
# Opens the Dagster UI at http://localhost:3333 showing the software-defined
# asset DAG (raw files → ingest → QC → preprocess → features → DS handoff),
# per-asset lineage/quality/schema metadata, and the data-quality gates
# (asset checks). Everything is read from the real files in data_cache/.
#
# Runs in the ISOLATED .venv-dagster so the science .venv is untouched.
set -euo pipefail

# repo root = parent of this script's dir
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

VENV="$REPO/.venv-dagster"
if [[ ! -x "$VENV/bin/dagster" ]]; then
  echo "ERROR: $VENV not found. Create it with:"
  echo "  python3 -m venv .venv-dagster && .venv-dagster/bin/pip install dagster dagster-webserver numpy pandas pyarrow scipy pybaselines"
  exit 1
fi

# Persist run/check history across sessions, and make atlas + atlas_orchestration
# importable in every Dagster subprocess (multiprocess executor inherits env).
export DAGSTER_HOME="$REPO/orchestration/.dagster_home"
export PYTHONPATH="$REPO:$REPO/orchestration${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$DAGSTER_HOME"

# Default 3333 (3000 is often taken by a Next.js dev server). Override: PORT=xxxx
PORT="${PORT:-3333}"

echo "──────────────────────────────────────────────────────────────"
echo " Atlas DE pipeline  →  http://127.0.0.1:${PORT}"
echo "   • Assets tab:  the DAG + lineage (click any asset for metadata)"
echo "   • Checks:      the data-quality gates (green = passing)"
echo "   • 'Materialize all' re-runs the pipeline against data_cache/"
echo " Contract artifact: orchestration/contract/CONTRACT.md"
echo " (use http://127.0.0.1 not localhost — avoids IPv6 :: collisions)"
echo "──────────────────────────────────────────────────────────────"

exec "$VENV/bin/dagster" dev \
  -f "$REPO/orchestration/atlas_orchestration/definitions.py" \
  -h 127.0.0.1 -p "$PORT"
