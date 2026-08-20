"""Shared types + helpers for the three runners."""
from __future__ import annotations

import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import SADE_REPO
MCP_SERVER_DIR = SADE_REPO / "src" / "nika" / "service" / "mcp_server"
SADE_AGENT_CWD = SADE_REPO / "src" / "agent"

PYTHON_EXE = sys.executable


@dataclass
class AgentResult:
    """Result of running an agent on one scenario.

    `diagnosis_text` is the full agent final-message string. `is_anomaly` and
    `faulty_devices` are parsed out of the structured block the agent is
    instructed to write. If parsing fails the raw text is still kept.
    """
    diagnosis_text: str = ""
    is_anomaly: bool | None = None
    faulty_devices: list[str] = field(default_factory=list)
    parsed_ok: bool = False
    # Token accounting — split out so cache reads aren't conflated with fresh
    # input. `in_tokens` is fresh (uncached) input only; `cache_read_tokens` is
    # the cached portion (billed at ~10% of fresh-input rate) and is usually
    # where the bulk of context lives once the system prompt has been cached.
    in_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    out_tokens: int = 0
    cost_usd: float = 0.0           # from the API for the Claude agents; ReAct computes it from list prices
    tool_calls: int = 0
    # True when the agent exhausted its max_turns loop budget. The SDK's
    # `num_turns` is NOT a clean budget gauge (it counts every conversation
    # turn incl. tool-result rounds, so it routinely reads > max_turns), so we
    # don't surface it. This flag is the honest signal: combined with
    # parsed_ok=False it means "ran out of turns without a valid submission".
    hit_max_turns: bool = False
    raw_trace: str = ""             # the streamed reasoning trace
    error: str | None = None


class RunnerError(RuntimeError):
    pass


# --- Hermetic context for the Claude Code CLI -----------------------------------
#
# A benchmark run must be driven by the benchmark's inputs alone: the system
# prompt, the ticket, and the lab tools. Claude Code, however, also assembles
# context from the working directory it is started in -- every CLAUDE.md from
# the cwd up to the filesystem root, ~/.claude/CLAUDE.md, and the project
# memory of the cwd's git repository -- even when the Agent SDK launches it
# with a custom system prompt. The CC-Baseline CLI therefore runs from a clean
# directory outside any git repository, and both Claude runners refuse to
# start if such instruction files or project memory would be picked up. SADE
# is the one designed exception: its cwd is the SADE repo, whose routing
# CLAUDE.md is part of the agent.

def _project_key(path: Path) -> str:
    """~/.claude/projects/<key> as Claude Code derives it (separators -> '-')."""
    s = str(path.resolve())
    return re.sub(r"[:\\/]", "-", s).rstrip("-")   # D:\a\b -> D--a-b ; /u/x -> -u-x


def _git_root(path: Path) -> Path | None:
    for p in (path, *path.parents):
        if (p / ".git").exists():
            return p
    return None


def assert_context_isolated(cwd: Path, agent: str) -> None:
    """Refuse to run if a CLAUDE.md or Claude Code project memory could reach the CLI.

    SADE's own `src/agent/.claude/CLAUDE.md` is exempt: it is the agent's
    routing table, part of SADE, and lives inside its cwd (Claude Code reads
    <cwd>/.claude/CLAUDE.md as project instructions).
    """
    home = Path.home()
    problems: list[str] = []
    if (home / ".claude" / "CLAUDE.md").exists():
        problems.append(f"user-level instructions {home / '.claude' / 'CLAUDE.md'}")
    for p in (cwd, *cwd.parents):
        for cand in (p / "CLAUDE.md", p / ".claude" / "CLAUDE.md"):
            if cand.exists() and not (agent == "sade" and p == cwd):
                problems.append(f"instructions file {cand}")
    roots = {cwd}
    if (g := _git_root(cwd)) is not None:
        roots.add(g)
    for r in roots:
        mem = home / ".claude" / "projects" / _project_key(r) / "memory"
        if mem.exists() and any(mem.iterdir()):
            problems.append(f"project memory {mem}")
    if problems:
        raise RunnerError(
            f"{agent}: context outside the benchmark would reach the agent's CLI (cwd={cwd}): "
            + "; ".join(problems)
            + ". Move/rename it or pick another cwd; the run would not be a clean benchmark run.")


def isolated_cwd() -> Path:
    """A context-free working directory for the CC-Baseline CLI.

    Outside every git repository (so no project memory resolves), with no
    CLAUDE.md on the way up; created on first use under the system temp dir.
    """
    root = Path(os.environ.get("FAULTBENCH_CC_CWD")
                or Path(tempfile.gettempdir()) / "faultbench_cc_cwd")
    root.mkdir(parents=True, exist_ok=True)
    if _git_root(root) is not None:
        raise RunnerError(f"CC-Baseline cwd {root} lies inside a git repository; "
                          "set FAULTBENCH_CC_CWD to a directory outside any repo")
    assert_context_isolated(root, "cc-baseline")
    return root


_ESSENTIAL_ENV_KEYS = (
    # Windows DLL/process bootstrap
    "PATH", "SystemRoot", "ComSpec", "TEMP", "TMP", "USERPROFILE",
    "windir", "PATHEXT", "SYSTEMDRIVE", "HOMEDRIVE", "HOMEPATH",
    # Windows user identity. MUST be present: Python's getpass.getuser() checks
    # LOGNAME/USER/LNAME/USERNAME and, finding none, falls back to `import pwd`
    # (Unix-only) -> ModuleNotFoundError on Windows. Kathara's Docker lab
    # reconstruction (get_lab_from_api -> get_machines_api_objects, filtered by
    # the current user's container label) calls getuser(), so without USERNAME
    # the SADE helper's infra_sweep dies. Worse: that crash happens inside
    # KatharaBaseAPI.exec_cmd's func_timeout worker thread, where it surfaces as
    # a 10s-per-device stall rather than a clean error -> a 30-device sweep
    # "times out" at ~300s.
    "USERNAME", "USERDOMAIN",
    # Docker Desktop / credential-store config locations on Windows
    "APPDATA", "LOCALAPPDATA", "ALLUSERSPROFILE", "ProgramData",
    # Python
    "PYTHONPATH", "PYTHONHOME",
    # API keys used by helper MCP / SADE skills
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
)


def _minimal_env(**extras: str) -> dict:
    """Build a small env dict for MCP-server subprocesses.

    The Claude Code CLI serialises every MCP server's `env` into the
    `--mcp-config` JSON argument. Including the full `os.environ` (typically
    6-10 KB on Windows) for each of 3-4 MCP servers pushes the cmd line past
    Windows' 32 KB CreateProcess limit and triggers a misleading
    `CLINotFoundError`. Forwarding only the keys subprocesses actually need
    keeps us safely under the limit.
    """
    env = {k: os.environ[k] for k in _ESSENTIAL_ENV_KEYS if k in os.environ}
    env.update(extras)
    return env


def diagnostic_mcp_config(lab_name: str) -> dict:
    """MCP config for the three diagnostic servers (kathara_base/frr/telemetry).

    Servers are launched as stdio subprocesses from SADE-NetworkAgent's
    existing source files (READ-ONLY).
    """
    common_env = _minimal_env(LAB_NAME=lab_name)
    return {
        "kathara_base_mcp_server": {
            "command": PYTHON_EXE,
            "args": [str(MCP_SERVER_DIR / "kathara_base_mcp_server.py")],
            "env": common_env, "transport": "stdio",
        },
        "kathara_frr_mcp_server": {
            "command": PYTHON_EXE,
            "args": [str(MCP_SERVER_DIR / "kathara_frr_mcp_server.py")],
            "env": common_env, "transport": "stdio",
        },
        "kathara_telemetry_mcp_server": {
            "command": PYTHON_EXE,
            "args": [str(MCP_SERVER_DIR / "kathara_telemetry_mcp_server.py")],
            "env": common_env, "transport": "stdio",
        },
    }


def helper_mcp_config(lab_name: str) -> dict:
    """MCP config for SADE's helper wrapper (only used by the SADE runner).

    Lets SADE invoke its diagnostic helpers (infra_sweep, ospf_snapshot, etc.)
    without needing host-side Bash — which would leak topology source files.

    LAB_NAME MUST be forwarded here, identically to diagnostic_mcp_config.
    The helpers instantiate KatharaBaseAPI(lab_name=os.getenv("LAB_NAME",
    "ospf_enterprise_dhcp")). Without it, the helper targets the WRONG lab —
    every `exec` then fails to attach to the real containers and burns
    KatharaBaseAPI's 10s per-command func_timeout, so e.g. a 30-device
    infra_sweep stalls for ~300s instead of finishing in tens of seconds.
    """
    return {
        "helper_server": {
            "command": PYTHON_EXE,
            "args": ["-m", "faultbench.mcp_helper"],
            "env": _minimal_env(HELPER_PYTHON=PYTHON_EXE, LAB_NAME=lab_name),
            "transport": "stdio",
        },
    }


# ---- Tool whitelists (sandboxes the agent to only lab-side commands) -------

# CC-Baseline: only kathara_* MCP tools. NO Bash, Read, Glob, Grep, etc. —
# those would let the agent read host-side topology source (data leakage).
CC_BASELINE_ALLOWED_TOOLS = [
    "mcp__kathara_base_mcp_server__*",
    "mcp__kathara_frr_mcp_server__*",
    "mcp__kathara_telemetry_mcp_server__*",
]

# SADE: kathara MCP tools + helper MCP tools + Skill (for fault-family
# playbooks). Skill loads markdown from .claude/skills/ which is SADE's
# documented knowledge base (not topology source). NO Bash / Read / Glob.
SADE_ALLOWED_TOOLS = [
    "mcp__kathara_base_mcp_server__*",
    "mcp__kathara_frr_mcp_server__*",
    "mcp__kathara_telemetry_mcp_server__*",
    "mcp__helper_server__*",
    "Skill",
]

# Hard block-list — even with cwd set to SADE_AGENT_CWD (needed for Skill
# discovery) and bypassPermissions mode, these built-in tools would
# otherwise let the agent read host topology source files (data leakage).
HOST_FS_DISALLOWED_TOOLS = [
    "Read", "Bash", "Glob", "Grep", "Edit", "Write",
    "NotebookEdit", "WebFetch", "WebSearch", "Task",
]


# ---- DIAGNOSIS-block parsing -----------------------------------------------

_DIAG_RE = re.compile(
    r"===\s*DIAGNOSIS\s*===(.*?)===\s*END\s*===",
    re.DOTALL | re.IGNORECASE,
)


def parse_diagnosis(text: str) -> tuple[bool | None, list[str], str, bool]:
    """Extract is_anomaly, faulty_devices, diagnosis_text from agent's final.

    Returns (is_anomaly, faulty_devices, diagnosis_text, parsed_ok).
    If the structured block is missing, returns (None, [], full_text, False)
    so we still score on whatever the agent wrote.
    """
    if not text:
        return None, [], "", False
    m = _DIAG_RE.search(text)
    if not m:
        return None, [], text.strip(), False

    body = m.group(1)

    # is_anomaly: line
    is_anom_m = re.search(r"is_anomaly\s*:\s*(true|false|yes|no)",
                          body, re.IGNORECASE)
    is_anomaly: bool | None = None
    if is_anom_m:
        is_anomaly = is_anom_m.group(1).lower() in ("true", "yes")

    # faulty_devices: line
    devs_m = re.search(r"faulty_devices\s*:\s*([^\n]+)", body, re.IGNORECASE)
    faulty: list[str] = []
    if devs_m:
        raw = devs_m.group(1).strip()
        if raw and raw.lower() not in ("none", "[]", "n/a"):
            faulty = [d.strip().strip(",;[]\"'")
                      for d in re.split(r"[,;\s]+", raw) if d.strip()]

    # diagnosis_text: everything after the "diagnosis_text:" line
    diag_m = re.search(r"diagnosis_text\s*:\s*\n?(.+)", body,
                       re.IGNORECASE | re.DOTALL)
    diagnosis = (diag_m.group(1).strip() if diag_m else body.strip())

    return is_anomaly, faulty, diagnosis, True


def log_submission(log, final_text: str, is_anomaly, faulty_devices,
                   parsed_ok: bool) -> None:
    """Append the FULL final message + parsed verdict to the trace.

    Streaming [TEXT] lines are truncated to 800 chars, which can hide the
    === DIAGNOSIS === block of a verbose final message and makes trace.log
    look like the agent never submitted. result.json stays authoritative;
    these two lines make the trace self-contained and human-checkable.
    """
    log("SUBMITTED", f"parsed_ok={parsed_ok} is_anomaly={is_anomaly} "
                     f"faulty_devices={faulty_devices}")
    log("FINAL-TEXT", final_text)


# ---- Shared Claude Agent SDK loop ------------------------------------------

def _print_stream(line: str) -> None:
    """Live-stream a line to stdout (utf-8 safe on Windows)."""
    try:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    except UnicodeEncodeError:
        sys.stdout.write(line.encode("ascii", errors="replace").decode("ascii") + "\n")
        sys.stdout.flush()


async def stream_claude_query(prompt: str, options, stream: bool,
                              start_line: str) -> AgentResult:
    """Drive one `claude_agent_sdk.query` to completion and collect everything.

    The CC-Baseline and SADE runners differ ONLY in the ClaudeAgentOptions they
    build (system prompt, tool allowlist, cwd, MCP servers); the message loop,
    trace format, usage accounting and verdict parsing are identical and live
    here once.
    """
    from claude_agent_sdk import (
        AssistantMessage, ResultMessage, TextBlock, ToolUseBlock, UserMessage,
        query,
    )

    tool_calls = 0
    in_tok = out_tok = cache_read = cache_write = 0
    cost_usd = 0.0
    final_text = ""
    trace_lines: list[str] = []
    sdk_error: str | None = None
    hit_max_turns = False

    def log(label: str, body: str = "") -> None:
        line = f"[{label}] {body}"
        trace_lines.append(line)
        if stream:
            _print_stream(line)

    log("AGENT", start_line)

    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text = (block.text or "").strip()
                        if text:
                            log("TEXT", text[:800])
                            final_text = text  # last text wins
                    elif isinstance(block, ToolUseBlock):
                        tool_calls += 1
                        args_str = str(block.input)[:300]
                        log("TOOL", f"{block.name}({args_str})")
            elif isinstance(message, UserMessage):
                content = message.content if isinstance(message.content, list) else []
                for block in content:
                    result_text = getattr(block, "content", None)
                    if result_text is not None:
                        log("RESULT", str(result_text)[:400])
            elif isinstance(message, ResultMessage):
                usage = getattr(message, "usage", None) or {}
                if isinstance(usage, dict):
                    in_tok = usage.get("input_tokens", 0) or 0
                    out_tok = usage.get("output_tokens", 0) or 0
                    cache_read = usage.get("cache_read_input_tokens", 0) or 0
                    cache_write = usage.get("cache_creation_input_tokens", 0) or 0
                cost_usd = float(getattr(message, "total_cost_usd", 0.0) or 0.0)
                if message.result:
                    final_text = message.result
                log("FINAL", f"stop={getattr(message,'stop_reason','?')} tool_calls={tool_calls}")
    except Exception as exc:
        sdk_error = f"{type(exc).__name__}: {exc}"
        hit_max_turns = "maximum number of turns" in sdk_error.lower()
        log("ERROR", sdk_error)

    is_anom, devs, diag, ok = parse_diagnosis(final_text)
    log_submission(log, final_text, is_anom, devs, ok)
    return AgentResult(
        diagnosis_text=diag,
        is_anomaly=is_anom,
        faulty_devices=devs,
        parsed_ok=ok,
        in_tokens=in_tok,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        out_tokens=out_tok,
        cost_usd=cost_usd,
        tool_calls=tool_calls,
        hit_max_turns=hit_max_turns,
        raw_trace="\n".join(trace_lines),
        error=sdk_error,
    )
