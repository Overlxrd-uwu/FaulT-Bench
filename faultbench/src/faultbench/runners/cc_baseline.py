"""CC-Baseline runner — a plain Claude Code agent with the kathara_* MCP tools."""
from __future__ import annotations

import asyncio
import os

from claude_agent_sdk import ClaudeAgentOptions

from ..prompts.baseline_freetext import BASELINE_FREETEXT_PROMPT
from .base import (
    AgentResult, CC_BASELINE_ALLOWED_TOOLS, HOST_FS_DISALLOWED_TOOLS,
    diagnostic_mcp_config, isolated_cwd, stream_claude_query,
)


def run_cc_baseline(prompt_to_agent: str, lab_name: str,
                    model: str = "claude-sonnet-4-6",
                    max_turns: int = 20,
                    stream: bool = True) -> AgentResult:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return AgentResult(error="ANTHROPIC_API_KEY not set")

    options = ClaudeAgentOptions(
        system_prompt=BASELINE_FREETEXT_PROMPT,
        model=model,
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        mcp_servers=diagnostic_mcp_config(lab_name),
        # Whitelist route: `allowed_tools` auto-approves ONLY the kathara MCP
        # tools; `disallowed_tools` hard-blocks the host-FS/leakage built-ins.
        # CC-Baseline has no skill library, so Skill is intentionally absent
        # from the allowlist.
        # NOTE: do NOT use `tools=[]` — `--tools ""` strips the base tool set
        # (incl. MCP tools, under claude-agent-sdk 0.2.82) and yields 0-tool runs.
        allowed_tools=CC_BASELINE_ALLOWED_TOOLS,
        disallowed_tools=HOST_FS_DISALLOWED_TOOLS,
        # Clean cwd: Claude Code reads CLAUDE.md files and project memory from
        # whatever directory it starts in (see base.py); only the benchmark's
        # own inputs may reach the agent.
        cwd=str(isolated_cwd()),
        env={"ANTHROPIC_API_KEY": api_key, "ANTHROPIC_AUTH_TOKEN": ""},
    )
    return asyncio.run(stream_claude_query(
        prompt_to_agent, options, stream,
        start_line=f"cc-baseline starts on lab={lab_name} cwd={options.cwd}"))
