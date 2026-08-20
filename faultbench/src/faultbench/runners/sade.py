"""SADE runner — the SADE agent with its skill library and helper tools."""
from __future__ import annotations

import asyncio
import os

from claude_agent_sdk import ClaudeAgentOptions

from ..prompts.sade_freetext import SADE_FREETEXT_PROMPT
from .base import (
    AgentResult, HOST_FS_DISALLOWED_TOOLS, SADE_AGENT_CWD,
    SADE_ALLOWED_TOOLS, assert_context_isolated, diagnostic_mcp_config,
    helper_mcp_config, stream_claude_query,
)


def run_sade(prompt_to_agent: str, lab_name: str,
             model: str = "claude-sonnet-4-6",
             max_turns: int = 20,
             stream: bool = True) -> AgentResult:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return AgentResult(error="ANTHROPIC_API_KEY not set")

    # SADE runs from its own repo so its skills and routing CLAUDE.md load;
    # nothing else may reach it (see the hermetic-context notes in base.py).
    assert_context_isolated(SADE_AGENT_CWD, "sade")
    mcp_servers = {**diagnostic_mcp_config(lab_name), **helper_mcp_config(lab_name)}
    options = ClaudeAgentOptions(
        system_prompt=SADE_FREETEXT_PROMPT,
        model=model,
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        mcp_servers=mcp_servers,
        # Whitelist route: `allowed_tools` auto-approves the kathara/helper MCP
        # tools + Skill; `disallowed_tools` HARD-BLOCKS the host-FS/leakage
        # built-ins (Read/Bash/Glob/...) regardless of permission mode.
        # NOTE: do NOT use `tools=[]` here — `--tools ""` strips the base tool
        # set (incl. Skill and MCP tools, under claude-agent-sdk 0.2.82), which
        # yields 0-tool runs. SADE's skill library loads from the agent's cwd
        # (its .claude/skills), the same way SADE itself ships it.
        allowed_tools=SADE_ALLOWED_TOOLS,
        disallowed_tools=HOST_FS_DISALLOWED_TOOLS,
        cwd=str(SADE_AGENT_CWD),
        env={"ANTHROPIC_API_KEY": api_key, "ANTHROPIC_AUTH_TOKEN": ""},
    )
    return asyncio.run(stream_claude_query(
        prompt_to_agent, options, stream,
        start_line=f"sade starts on lab={lab_name}"))
