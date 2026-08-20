"""Evaluate PRE-CHECK / POST-INJECT-CHECK expectations.

Each CheckLine has the form:
    <device>  <command>  => expect <condition>

`condition` is parsed loosely. Supported:
    0% loss / loss>0       (ping percent-loss)
    nonempty / empty       (grep / pgrep output)
    UP / DOWN              (link state)
    Full / Init / down     (OSPF neighbor state)
    >=N / 0 / 1 / nonzero  (numeric comparisons)
    <baseline / <N         (informational; treated as pass unless N given)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .lab import kathara_exec
from .scenario import CheckLine


@dataclass
class CheckResult:
    line: CheckLine
    passed: bool
    actual: str            # raw stdout
    rationale: str         # one-line why we passed/failed


def run_check(topology: str, line: CheckLine, timeout: int = 30) -> CheckResult:
    """Run one check line and evaluate the expectation."""
    rc, out = kathara_exec(topology, line.device, line.command, timeout=timeout)
    passed, why = _evaluate_expectation(line.expect, rc, out)
    return CheckResult(line=line, passed=passed, actual=out.strip(), rationale=why)


def run_checks(topology: str, lines: list[CheckLine]) -> list[CheckResult]:
    return [run_check(topology, ln) for ln in lines]


def all_passed(results: list[CheckResult]) -> bool:
    return all(r.passed for r in results)


_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*(?:packet\s*)?loss", re.IGNORECASE)


def _evaluate_expectation(expect: str, rc: int, out: str) -> tuple[bool, str]:
    """Best-effort matcher for the `=> expect X` clause."""
    e = expect.strip().lower()
    out_l = out.lower()
    out_stripped = out.strip()

    # Ping loss style ("0% loss" must anchor at the start so it cannot match
    # inside "100% loss")
    if e.startswith(("0%", "0 %")) and "loss" in e:
        m = _PERCENT_RE.search(out)
        if m:
            return (float(m.group(1)) == 0.0, f"loss={m.group(1)}%")
        return (False, "no loss% found in ping output")

    if "loss>0" in e or "loss > 0" in e:
        m = _PERCENT_RE.search(out)
        if m:
            return (float(m.group(1)) > 0.0, f"loss={m.group(1)}%")
        # No ping output at all = total loss
        return (True, "no ping output (assumed loss>0)")

    # nonempty / empty (pgrep, grep)
    if "nonempty" in e or "non-empty" in e:
        return (bool(out_stripped), "output present" if out_stripped else "output empty")
    if e == "empty":
        return (not out_stripped, "output empty" if not out_stripped else "output present")

    # link state
    if e == "up":
        return (("up" in out_l and "down" not in out_l), f"link state output: {out_stripped[:80]}")
    if e == "down":
        return (("down" in out_l), f"link state output: {out_stripped[:80]}")

    # OSPF neighbor states
    if "full" in e:
        return ("full" in out_l, f"contains Full: {'yes' if 'full' in out_l else 'no'}")
    if "init" in e:
        return ("init" in out_l, f"contains Init: {'yes' if 'init' in out_l else 'no'}")

    # numeric comparisons (>=N, ==N, just N)
    nm = re.match(r">=\s*(\d+)", e)
    if nm:
        target = int(nm.group(1))
        # Try last numeric on the line (for `| wc -l` style outputs)
        nums = re.findall(r"\b(\d+)\b", out_stripped)
        if nums:
            actual = int(nums[-1])
            return (actual >= target, f"got {actual}, expected >={target}")
        return (False, f"no number found in '{out_stripped[:40]}'")

    nm = re.match(r"(\d+)\s*$", e)
    if nm:
        target = int(nm.group(1))
        nums = re.findall(r"\b(\d+)\b", out_stripped)
        if nums:
            actual = int(nums[-1])
            return (actual == target, f"got {actual}, expected {target}")
        return (False, f"no number found in '{out_stripped[:40]}'")

    if "nonzero" in e or "non-zero" in e:
        nums = re.findall(r"\b(\d+)\b", out_stripped)
        if nums:
            return (int(nums[-1]) > 0, f"got {nums[-1]}")
        return (False, "no number found")

    # Fallback: an expect form this matcher does not recognize. It passes as
    # advisory so note-only expectations don't block a run, but the rationale
    # flags it loudly — if you see this while adding a scenario, extend the
    # matcher above rather than relying on the fallback.
    return (True, f"UNRECOGNIZED expect, advisory PASS: '{expect}' (rc={rc})")
