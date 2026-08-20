"""Single source of truth for filesystem locations.

The repo root is resolved ONCE, here, so a fresh ``git clone`` works anywhere
with no path edits:

  1. ``$FAULTBENCH_ROOT`` if set (explicit override), else
  2. walk up from this file until we find the repo root -- the directory that
     contains both ``dataset/`` and ``faultbench/``.

Everything else is derived from that root. This module imports only ``os`` +
``pathlib`` (no third-party deps).
"""
from __future__ import annotations

import os
from pathlib import Path


def _resolve_root() -> Path:
    env = os.environ.get("FAULTBENCH_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
        if not (root / "dataset").is_dir():
            raise RuntimeError(
                f"FAULTBENCH_ROOT={root} has no dataset/ subdirectory")
        return root
    for parent in Path(__file__).resolve().parents:
        if (parent / "dataset").is_dir() and (parent / "faultbench").is_dir():
            return parent
    raise RuntimeError(
        f"Could not locate the repository root from {__file__!r}. "
        "Set FAULTBENCH_ROOT to your clone directory.")


ROOT = _resolve_root()

DATASET = ROOT / "dataset"
KATH_BASE = ROOT / "topology_kathara"
SADE_REPO = ROOT / "SADE-NetworkAgent"
ENV_FILE = SADE_REPO / ".env"
