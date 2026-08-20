"""Load API keys from SADE-NetworkAgent's .env into the process environment.

Every entry point must call this before the first run: the runners abort with
"ANTHROPIC_API_KEY not set" otherwise -- and only AFTER the lab has already
been deployed, so the failure is needlessly expensive.
"""
from __future__ import annotations

import os
from pathlib import Path

from .paths import ENV_FILE


def load_env(env_path: Path = ENV_FILE) -> bool:
    """Read KEY=value lines into os.environ (existing values win).

    Returns False when the file does not exist, so callers can warn that they
    are relying on already-exported variables. Comment lines, blank lines and
    placeholder values (starting with ``<``) are skipped.
    """
    if not env_path.exists():
        return False
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and not v.startswith("<"):
            os.environ.setdefault(k, v)
    return True
