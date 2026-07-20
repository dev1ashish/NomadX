"""Dagster entrypoint.

Loaded via:  dagster dev -f orchestration/atlas_orchestration/definitions.py

We make both the repo root (for `import atlas.*`) and the orchestration dir
(for `import atlas_orchestration.*`) importable before loading the assets, so
the code location resolves regardless of cwd.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_ORCH = _PKG.parent
_REPO = _ORCH.parent
for _p in (str(_REPO), str(_ORCH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dagster import Definitions  # noqa: E402

from atlas_orchestration.assets import ALL_ASSETS  # noqa: E402
from atlas_orchestration.checks import ALL_CHECKS  # noqa: E402

defs = Definitions(assets=ALL_ASSETS, asset_checks=ALL_CHECKS)
