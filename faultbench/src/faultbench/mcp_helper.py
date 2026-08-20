"""SADE helper-script MCP server.

Exposes SADE's diagnostic helper scripts (infra_sweep, ospf_snapshot, ...) as
a single MCP tool so the agent can run them WITHOUT host-side `Bash` (which
would let it read host topology source files — data leakage).

WHY IN-PROCESS (not `python h.py <script>` as a subprocess):
    The helpers reach the lab through SADE's own `KatharaBaseAPI` ->
    `Kathara.get_instance()` -> Docker. When that runs in a process spawned
    several levels below the Claude Code CLI on Windows (CLI -> this MCP
    server -> h.py -> script), every Docker `exec` stalls to KatharaBaseAPI's
    10s per-command `func_timeout` — so a 30-device infra_sweep "times out" at
    ~300s. The exact same script is ~4s when run from a normal shell. The
    diagnostic MCP servers (kathara_base etc.) do NOT have this problem because
    they call `KatharaBaseAPI` *in their own process*. So we mirror them: run
    the helper IN THIS PROCESS via runpy and capture its stdout, instead of
    spawning the fragile great-grandchild subprocess.

Helpers are READ-ONLY; they only gather diagnostic snapshots from the lab.
They read the target lab from the `LAB_NAME` env var, which the runner sets on
this server's environment (see runners/base.helper_mcp_config).
"""
from __future__ import annotations

import contextlib
import io
import os
import runpy
import sys
import threading
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .paths import SADE_REPO
_SKILLS = SADE_REPO / "src" / "agent" / ".claude" / "skills"
_SCRIPTS = _SKILLS / "diagnosis-methodology-skill" / "scripts"
# Special-purpose scripts that live outside the diagnosis-methodology folder
# (mirrors h.py's EXTRA_SCRIPTS).
_EXTRA = {
    "parse_large": _SKILLS / "big-return-skill" / "scripts" / "parse_large_output.py",
    "bgp_snapshot": _SKILLS / "bgp-fault-skill" / "scripts" / "bgp_snapshot.py",
}

# Serialise helper runs: each one mutates sys.argv / sys.path / stdout while it
# executes, and FastMCP may dispatch tool calls from worker threads.
_run_lock = threading.Lock()

mcp = FastMCP("helper_server")


def _resolve_script(name: str) -> Path:
    if name in _EXTRA:
        return _EXTRA[name]
    if not name.endswith(".py"):
        name += ".py"
    return _SCRIPTS / name


@mcp.tool()
def list_helpers() -> str:
    """List the available diagnostic helper scripts."""
    names = sorted(
        p.stem for p in _SCRIPTS.glob("*.py") if not p.name.startswith("_")
    )
    extra = sorted(_EXTRA.keys())
    return (
        "diagnosis-methodology helpers:\n  " + "\n  ".join(names)
        + "\nspecial:\n  " + "\n  ".join(extra)
    )


@mcp.tool()
def run_helper(script_name: str, args: list[str] | None = None) -> str:
    """Run a SADE diagnostic helper script and return its combined output.

    Args:
        script_name: e.g. "infra_sweep", "ospf_snapshot", "service_snapshot".
        args: optional extra CLI args passed through to the helper.

    The helper executes IN THIS PROCESS (see module docstring) with its stdout
    and stderr captured. It reaches the running lab via SADE's KatharaBaseAPI,
    reading the lab name from the LAB_NAME environment variable.
    """
    if not script_name or not script_name.replace("_", "").replace(".py", "").isalnum():
        return f"ERROR: invalid script_name '{script_name}'"
    clean_args: list[str] = []
    for a in (args or []):
        if not isinstance(a, str) or any(c in a for c in r";|&`$<>"):
            return f"ERROR: unsafe arg '{a}' (no shell metacharacters allowed)"
        clean_args.append(a)

    script = _resolve_script(script_name)
    if not script.is_file():
        return f"ERROR: helper not found: {script}"

    buf = io.StringIO()
    rc: int | None = 0
    with _run_lock:
        saved_argv, saved_path = sys.argv, list(sys.path)
        # Helper scripts do `from network_inventory import ...` (a sibling) and
        # import nika; ensure both the script's own dir and the shared scripts
        # dir are importable.
        for d in (str(script.parent), str(_SCRIPTS)):
            if d not in sys.path:
                sys.path.insert(0, d)
        sys.argv = [str(script), *clean_args]
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                try:
                    runpy.run_path(str(script), run_name="__main__")
                except SystemExit as exc:
                    rc = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        except Exception as exc:  # helper blew up — surface it, don't kill the server
            out = buf.getvalue()
            return f"ERROR running {script_name}: {type(exc).__name__}: {exc}\n{out[:4000]}"
        finally:
            sys.argv = saved_argv
            sys.path[:] = saved_path

    out = buf.getvalue()
    if rc not in (0, None):
        out = f"[helper exited rc={rc}]\n{out}"
    return out[:8000] if len(out) > 8000 else out


if __name__ == "__main__":
    mcp.run()
