"""Build a compact, agent-neutral INVESTIGATION LOG from a run's trace.

Every runner writes the same trace format (one entry per line-group, labelled
[TOOL] / [RESULT] / [TEXT] / [FINAL] / [SUBMITTED] / [FINAL-TEXT]), so this
digest is uniform across agents: what the agent DID (tool calls), what the
network REPLIED (result snippets), and what the agent SAID in between (its own
interim notes). The reasoning judge consumes it to verify that the writeup's
claims are backed by observations the agent actually gathered — instead of
taking the writeup's word for it (which rewards narration and punishes terse
but well-evidenced agents).

The final writeup is NOT part of the digest (it is passed to the judge
separately), so [FINAL-TEXT] / [SUBMITTED] entries are dropped here.
"""
from __future__ import annotations

import re

_ENTRY_RE = re.compile(
    r"^\[(AGENT|TOOL|RESULT|TEXT|FINAL|SUBMITTED|FINAL-TEXT|ERROR)\]\s?(.*)$")

# Per-entry caps keep the digest inside a few thousand tokens while retaining
# EVERY tool call (the evidence trail matters more than full result bodies).
_CAP = {"TOOL": 300, "RESULT": 280, "TEXT": 600, "ERROR": 200, "AGENT": 120,
        "FINAL": 120}


def build_trace_digest(trace_text: str, max_total_chars: int = 16000) -> str:
    """Convert a raw trace into a numbered investigation log for the judge."""
    if not trace_text or not trace_text.strip():
        return ""

    # group physical lines into labelled entries (entries can span lines)
    entries: list[tuple[str, str]] = []
    label, buf = None, []
    for line in trace_text.splitlines():
        m = _ENTRY_RE.match(line)
        if m:
            if label is not None:
                entries.append((label, "\n".join(buf).strip()))
            label, buf = m.group(1), [m.group(2)]
        elif label is not None:
            buf.append(line)
    if label is not None:
        entries.append((label, "\n".join(buf).strip()))

    out: list[str] = []
    step = 0
    for label, body in entries:
        if label in ("FINAL-TEXT", "SUBMITTED", "AGENT"):
            continue  # writeup is judged separately; header adds nothing
        cap = _CAP.get(label, 300)
        text = body if len(body) <= cap else body[:cap] + " ...[cut]"
        text = text.replace("\n", " ")
        if label == "TOOL":
            step += 1
            out.append(f"{step}. RAN {text}")
        elif label == "RESULT":
            out.append(f"   -> {text}")
        elif label == "TEXT":
            out.append(f"   AGENT NOTE: {text}")
        elif label == "ERROR":
            out.append(f"   ! {text}")
        elif label == "FINAL":
            out.append(f"(end of investigation: {text})")

    digest = "\n".join(out)
    if len(digest) > max_total_chars:
        digest = digest[:max_total_chars] + "\n...[investigation log truncated]"
    return digest
