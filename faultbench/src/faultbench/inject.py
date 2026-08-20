"""Apply the [INJECTION] block to a running lab."""
from __future__ import annotations

from dataclasses import dataclass

from .lab import kathara_exec
from .scenario import Injection


@dataclass
class InjectionResult:
    applied: bool
    skipped_reason: str | None
    rc: int | None
    output: str


def apply_injection(topology: str, inj: Injection,
                    timeout: int = 60) -> InjectionResult:
    """Execute the injection. If no command (note-only / pre-existing), skip."""
    if not inj.command:
        return InjectionResult(
            applied=False,
            skipped_reason=("note-only / pre-existing fault: " +
                            (inj.note or "no cmd and no note")),
            rc=None,
            output="",
        )
    rc, out = kathara_exec(topology, inj.device, inj.command, timeout=timeout)
    return InjectionResult(
        applied=True,
        skipped_reason=None,
        rc=rc,
        output=out,
    )
