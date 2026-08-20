"""LLM-as-judge scorer (gpt-5-mini) — THREE independent metrics.

An earlier design blended a single `overall = 0.7*outcome + 0.3*reasoning`,
where the reasoning axes were per-scenario *hardcoded* [DIAGNOSTIC-PROCESS]
steps. That penalised agents that reach the right answer by a
different-but-valid route, and it buried "did it propose the fix?" as one
inert bonus axis. The scorer therefore reports three SEPARATE numbers, none
folded into the others:

  * outcome_score   — did it reach the right conclusion?
        Weighted blend of MUST/ASSERT axes (right device + fault description)
        and BONUS axes, normalised over the axis classes present. This is the
        headline number; nothing is blended into it.

  * fix_score       — did it propose the correct remediation?
        A single 0 / 0.5 / 1 judgement of the agent's writeup against the
        scenario's [FIX] block (the canonical remediation). `None` when the
        scenario has no [FIX] (e.g. negative controls — nothing to fix).

  * reasoning_score — is the explanation SOUND, by any valid route?
        ROUTE-AGNOSTIC: the judge grades three dimensions of the agent's OWN
        reasoning toward its OWN conclusion, given the ground truth, on an
        anchored 0-10 scale (mean -> [0,1]). The dims — grounding / causality /
        coverage — follow the chain-of-thought faithfulness decomposition
        (causality + coverage; see C2-Faith / FaithCoT-Bench) plus an
        evidence-grounding axis, and the judge writes a brief analysis BEFORE
        scoring (G-Eval style). It does NOT check conformance to any prescribed
        procedure, so a correct answer reached a different way scores full marks.
        NB: the proper G-Eval continuous score (token-probability weighting) and
        self-consistency averaging are BOTH unavailable on the judge model
        (gpt-5-mini blocks logprobs, fixes temperature, and is deterministic), so
        we get granularity from the anchored 0-10 scale instead of 0/0.5/1.
        `None` if the writeup is empty.

Axis class is chosen by a leading token on each [SCORING-AXES] line:

    must-…                      mandatory; LLM-judged yes/no            (outcome)
    bonus-…                     nice-to-have; LLM-judged yes/no         (outcome)
    assert-no-anomaly           parsed is_anomaly is False              (outcome, det.)
    assert-no-faulty-devices    parsed faulty_devices is empty          (outcome, det.)
    assert-identifies: <dev>    <dev> appears in parsed faulty_devices  (outcome, det.)
    assert-not-identifies:<dev> <dev> is ABSENT from faulty_devices     (outcome, det.)
    reason-…                    ignored by the scorer (reasoning is judged
                                route-agnostically, not by per-scenario steps).

The `assert-*` family is deterministic (no LLM call) and is what makes
false-premise / negative-control scenarios scorable (`assert-no-anomaly`) and
misdirected scenarios scorable (`assert-identifies` the true culprit,
`assert-not-identifies` the decoy). `assert-*` axes join the MUST pool of the
outcome score.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class AxisJudgment:
    axis: str
    kind: str          # "must" | "bonus" | "assert" | "fix" | "reason-dim"
    score: float       # 0..1 (binary axes are exactly 0.0 or 1.0)
    rationale: str

    @property
    def passed(self) -> bool:  # back-compat for callers/readers
        return self.score >= 1.0


@dataclass
class ScoreResult:
    must_passed: int = 0
    must_total: int = 0
    bonus_passed: int = 0
    bonus_total: int = 0
    outcome_score: float = 0.0
    fix_score: float | None = None          # None => scenario has no [FIX] block
    reasoning_score: float | None = None    # None => empty writeup / not scored
    reasoning_dims: dict[str, float] = field(default_factory=dict)
    per_axis: list[AxisJudgment] = field(default_factory=list)
    judge_model: str = "gpt-5-mini"
    reasoning_mode: str = "writeup"         # "trace" when the judge saw the log
    error: str | None = None


_MUST_WEIGHT = 0.7
_BONUS_WEIGHT = 0.3


_JUDGE_SYSTEM = """You are a strict grader of network-troubleshooting writeups.

You will be shown:
  1. A ground-truth diagnosis (what we expect the engineer to identify)
  2. The agent's submitted diagnosis (free text)
  3. ONE scoring axis (a single thing to check)

Decide if the agent's diagnosis satisfies that one axis. Be strict but fair.
Output a single JSON object on one line with keys:
  "pass": true or false
  "rationale": one short sentence

Output exactly the JSON object. No prose before or after.
"""


_JUDGE_FIX_SYSTEM = """You grade whether a network-troubleshooting writeup proposes the CORRECT REMEDIATION.

You will be shown:
  1. The expected fix (the remediation that would correct the fault)
  2. The agent's submitted diagnosis (free text)

Grade ONLY the remediation the writeup proposes — not whether the diagnosis is
right. Exact wording/commands need not match; judge substance. Award:
  1.0  proposes a remediation that would actually correct the fault (matches the expected fix in substance)
  0.5  right direction but incomplete, vague, or only partially correct
  0.0  no remediation proposed, or a wrong/harmful one

Output a single JSON object on one line with keys:
  "score": 0, 0.5, or 1
  "rationale": one short sentence

Output exactly the JSON object. No prose before or after.
"""


_JUDGE_REASONING_SYSTEM = """You grade the SOUNDNESS of the reasoning in a network-troubleshooting writeup, INDEPENDENT of which diagnostic path the agent took.

You will be shown:
  1. A ground-truth diagnosis (the correct answer and why)
  2. The agent's submitted diagnosis (free text)

Do NOT reward conformance to any particular procedure or step order. A correct
conclusion reached by a different but valid route MUST score full marks. Grade
only whether the agent's OWN reasoning justifies its OWN conclusion.

First write a brief ANALYSIS, then score three route-agnostic dimensions on an
anchored 0-10 scale (0 = absent or contradicted, 3 = weak/vague, 5 = partial,
8 = mostly solid, 10 = fully and explicitly demonstrated):
  grounding: key claims are backed by SPECIFIC observations the agent actually gathered (live checks, commands, tool output), not merely asserted or assumed from the ticket.
  causality: each step follows logically from the evidence and the steps before it -- no unsupported leaps, no self-contradiction.
  coverage: the explanation contains the essential inferences needed to justify the conclusion (no critical gap) and distinguishes the real cause from plausible alternatives.

Output a single JSON object on one line:
  {"analysis": "<one or two sentences>", "grounding": <0-10>, "causality": <0-10>, "coverage": <0-10>}

Output exactly the JSON object. No prose before or after.
"""


_JUDGE_USER_TMPL = """[GROUND-TRUTH]
{ground_truth}

[AGENT-DIAGNOSIS]
{agent_diagnosis}

[AXIS]
{axis}
"""

_JUDGE_FIX_TMPL = """[EXPECTED-FIX]
{fix_spec}

[AGENT-DIAGNOSIS]
{agent_diagnosis}
"""

_JUDGE_REASONING_TMPL = """[GROUND-TRUTH]
{ground_truth}

[AGENT-DIAGNOSIS]
{agent_diagnosis}
"""


_JUDGE_REASONING_TRACE_SYSTEM = """You grade the SOUNDNESS of the reasoning behind a network-troubleshooting diagnosis, INDEPENDENT of which diagnostic path the agent took.

You will be shown:
  1. A ground-truth diagnosis (the correct answer and why)
  2. The agent's INVESTIGATION LOG -- the tools it actually ran, what the
     network returned, and the agent's own interim notes
  3. The agent's final writeup

Do NOT reward conformance to any particular procedure or step order. A correct
conclusion reached by a different but valid route MUST score full marks. Grade
whether the agent's investigation and writeup TOGETHER justify its conclusion.

The INVESTIGATION LOG is the authority on what was observed:
- A claim is GROUNDED if the log shows an observation supporting it, even when
  the writeup does not restate that evidence. Terse writeups backed by the log
  score high.
- A claim asserted in the writeup with NO supporting observation anywhere in
  the log is ungrounded, however fluent it sounds.
- Result snippets in the log may be truncated ("...[cut]"); judge from what is
  visible and do not penalise the truncation itself.

First write a brief ANALYSIS, then score three route-agnostic dimensions on an
anchored 0-10 scale (0 = absent or contradicted, 3 = weak/vague, 5 = partial,
8 = mostly solid, 10 = fully and explicitly demonstrated):
  grounding: key claims are backed by observations visible in the investigation log (commands run, outputs received), not merely asserted or assumed from the ticket.
  causality: each inference follows logically from the observed evidence and the steps before it -- no unsupported leaps, no self-contradiction.
  coverage: the investigation plus writeup contain the essential checks and inferences needed to justify the conclusion (no critical gap) and distinguish the real cause from plausible alternatives the evidence leaves open.

Output a single JSON object on one line:
  {"analysis": "<one or two sentences>", "grounding": <0-10>, "causality": <0-10>, "coverage": <0-10>}

Output exactly the JSON object. No prose before or after.
"""


_JUDGE_REASONING_TRACE_TMPL = """[GROUND-TRUTH]
{ground_truth}

[INVESTIGATION-LOG]
{trace_digest}

[FINAL-WRITEUP]
{agent_diagnosis}
"""


def score_submission(
    diagnosis_text: str,
    ground_truth: str,
    scoring_axes: list[str],
    fix_spec: str | None = None,
    model: str = "gpt-5-mini",
    is_anomaly: bool | None = None,
    faulty_devices: list[str] | None = None,
    trace_digest: str | None = None,
) -> ScoreResult:
    """Judge a submission on three independent metrics: outcome, fix, reasoning.

    `is_anomaly` / `faulty_devices` drive the deterministic `assert-*` axes.
    `fix_spec` is the scenario's [FIX] block (None => no fix metric).
    `trace_digest` (from scoring.trace_digest.build_trace_digest) makes the
    reasoning judge trace-aware; outcome and fix are unaffected by it.
    """
    axes = [a.strip() for a in scoring_axes if a and a.strip()]

    if not diagnosis_text.strip():
        return ScoreResult(
            must_total=sum(1 for a in axes if _classify(a) in ("must", "assert")),
            bonus_total=sum(1 for a in axes if _classify(a) == "bonus"),
            outcome_score=0.0,
            judge_model=model, error="empty diagnosis_text",
        )

    # assert-* axes are deterministic, but reasoning is always LLM-judged when
    # there is a writeup, so the judge client is always needed here.
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return ScoreResult(judge_model=model, error="OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    judgments: list[AxisJudgment] = []
    must_p = must_t = bonus_p = bonus_t = 0

    for axis in axes:
        kind = _classify(axis)

        if kind == "assert":
            score, rationale = _eval_assert(axis, is_anomaly, faulty_devices)
            judgments.append(AxisJudgment(axis, "assert", score, rationale))
            must_t += 1
            must_p += int(score >= 1.0)

        elif kind == "must":
            passed, rationale = _judge_binary(client, model, ground_truth,
                                              diagnosis_text, axis)
            judgments.append(AxisJudgment(axis, "must", float(passed), rationale))
            must_t += 1
            must_p += int(passed)

        elif kind == "bonus":
            passed, rationale = _judge_binary(client, model, ground_truth,
                                              diagnosis_text, axis)
            judgments.append(AxisJudgment(axis, "bonus", float(passed), rationale))
            bonus_t += 1
            bonus_p += int(passed)

        # kind == "reason": RETIRED — ignored (route-agnostic judge handles reasoning).

    outcome = _aggregate_outcome(must_p, must_t, bonus_p, bonus_t)

    # --- Fix metric (only when the scenario defines a canonical fix) ----------
    fix_score = None
    if fix_spec and fix_spec.strip():
        fix_score, fix_rat = _judge_fix(client, model, fix_spec, diagnosis_text)
        judgments.append(AxisJudgment("[FIX]", "fix", fix_score, fix_rat))

    # --- Reasoning metric (route-agnostic; trace-aware when a digest is given) --
    reasoning_score, reasoning_dims, reason_judgments = _judge_reasoning(
        client, model, ground_truth, diagnosis_text, trace_digest=trace_digest)
    judgments.extend(reason_judgments)

    return ScoreResult(
        must_passed=must_p, must_total=must_t,
        bonus_passed=bonus_p, bonus_total=bonus_t,
        outcome_score=outcome,
        fix_score=fix_score,
        reasoning_score=reasoning_score,
        reasoning_dims=reasoning_dims,
        per_axis=judgments, judge_model=model,
        reasoning_mode=("trace" if trace_digest and trace_digest.strip()
                        else "writeup"),
    )


# ---- axis classification + aggregation (pure, unit-testable) ---------------

def _classify(axis: str) -> str:
    a = axis.strip().lower()
    if a.startswith("assert-"):
        return "assert"
    if a.startswith("reason-") or a.startswith("reason:"):
        return "reason"
    if a.startswith("must"):
        return "must"
    return "bonus"


def _aggregate_outcome(must_p: int, must_t: int,
                       bonus_p: int, bonus_t: int) -> float:
    """Weighted blend normalised over the axis classes that are present.

    must-only  -> must_frac           (1.0 reachable)
    both       -> 0.7*must + 0.3*bonus (bonus always counts)
    bonus-only -> bonus_frac
    """
    parts: list[tuple[float, float]] = []
    if must_t:
        parts.append((must_p / must_t, _MUST_WEIGHT))
    if bonus_t:
        parts.append((bonus_p / bonus_t, _BONUS_WEIGHT))
    if not parts:
        return 0.0
    wsum = sum(w for _, w in parts)
    return min(1.0, sum(frac * w for frac, w in parts) / wsum)


def _assert_arg(axis: str) -> str:
    """Device argument for `assert-identifies: <dev>` / `assert-not-identifies: <dev>`."""
    if ":" in axis:
        return axis.split(":", 1)[1].strip()
    return ""


def _eval_assert(axis: str, is_anomaly: bool | None,
                 faulty_devices: list[str] | None) -> tuple[float, str]:
    a = axis.strip().lower()
    fd_text = " ".join(faulty_devices or []).lower()

    if a.startswith("assert-no-anomaly"):
        ok = (is_anomaly is False)
        return (float(ok), f"parsed is_anomaly={is_anomaly} (want False)")

    if a.startswith("assert-no-faulty-devices"):
        ok = not (faulty_devices or [])
        return (float(ok), f"parsed faulty_devices={faulty_devices or []} (want empty)")

    if a.startswith("assert-not-identifies"):
        dev = _assert_arg(axis).lower()
        if not dev:
            return (0.0, "assert-not-identifies missing ': <device>' arg")
        ok = dev not in fd_text
        return (float(ok), f"decoy '{dev}' {'absent (good)' if ok else 'PRESENT (blamed)'}")

    if a.startswith("assert-identifies"):
        dev = _assert_arg(axis).lower()
        if not dev:
            return (0.0, "assert-identifies missing ': <device>' arg")
        ok = dev in fd_text
        return (float(ok), f"culprit '{dev}' {'named (good)' if ok else 'NOT named'}")

    return (0.0, f"unknown assert axis: {axis}")


# ---- LLM judging -----------------------------------------------------------

def _judge_binary(client, model: str, ground_truth: str,
                  diagnosis_text: str, axis: str) -> tuple[bool, str]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": _JUDGE_USER_TMPL.format(
                ground_truth=ground_truth, agent_diagnosis=diagnosis_text, axis=axis)},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    return _parse_pass(raw)


def _judge_fix(client, model: str, fix_spec: str,
               diagnosis_text: str) -> tuple[float, str]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _JUDGE_FIX_SYSTEM},
            {"role": "user", "content": _JUDGE_FIX_TMPL.format(
                fix_spec=fix_spec, agent_diagnosis=diagnosis_text)},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    return _parse_score(raw)


def _judge_reasoning(client, model: str, ground_truth: str, diagnosis_text: str,
                     trace_digest: str | None = None,
                     ) -> tuple[float | None, dict[str, float], list[AxisJudgment]]:
    """Route-agnostic reasoning judge: one call, three anchored 0-10 dimensions
    (grounding / causality / coverage), normalised to [0,1].

    With `trace_digest` the judge sees the agent's actual investigation log and
    verifies grounding against it (a terse-but-evidenced writeup scores high);
    without it, it falls back to judging the writeup alone (less reliable:
    with no trace to check against, verbose writeups score higher)."""
    if trace_digest and trace_digest.strip():
        messages = [
            {"role": "system", "content": _JUDGE_REASONING_TRACE_SYSTEM},
            {"role": "user", "content": _JUDGE_REASONING_TRACE_TMPL.format(
                ground_truth=ground_truth, trace_digest=trace_digest,
                agent_diagnosis=diagnosis_text)},
        ]
    else:
        messages = [
            {"role": "system", "content": _JUDGE_REASONING_SYSTEM},
            {"role": "user", "content": _JUDGE_REASONING_TMPL.format(
                ground_truth=ground_truth, agent_diagnosis=diagnosis_text)},
        ]
    # The judge occasionally answers with prose or malformed JSON, which would
    # leave the run with reasoning_score=None for good. Retry a few times
    # before giving up -- the inputs are identical, only the reply varies.
    raw_dims, rationale = {}, ""
    for _attempt in range(_REASON_PARSE_RETRIES):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw_dims, rationale = _parse_reasoning(raw)    # each on the 0-10 scale
        if raw_dims:
            break
    if not raw_dims:
        return None, {}, []
    dims = {k: v / _REASON_SCALE for k, v in raw_dims.items()}   # -> [0,1]
    mean = sum(dims.values()) / len(dims)
    judgments = [AxisJudgment(f"reason:{k}", "reason-dim", v, rationale)
                 for k, v in dims.items()]
    return mean, dims, judgments


def _parse_pass(raw: str) -> tuple[bool, str]:
    try:
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            return bool(data.get("pass")), str(data.get("rationale") or "")
    except Exception:
        pass
    low = raw.lower()
    if "true" in low or "yes" in low or ("pass" in low and "no pass" not in low):
        return True, raw[:200]
    return False, raw[:200]


def _parse_score(raw: str) -> tuple[float, str]:
    try:
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            return _clamp_half(float(data.get("score", 0))), str(data.get("rationale") or "")
    except Exception:
        pass
    nm = re.search(r"(0(?:\.5)?|0\.5|1(?:\.0)?)", raw)
    if nm:
        try:
            return _clamp_half(float(nm.group(1))), raw[:200]
        except Exception:
            pass
    return 0.0, raw[:200]


_REASON_DIMS = ("grounding", "causality", "coverage")
_REASON_SCALE = 10.0
_REASON_PARSE_RETRIES = 3


def _parse_reasoning(raw: str) -> tuple[dict[str, float], str]:
    """Returns the three reasoning dims on the raw 0-10 scale + the analysis."""
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            dims = {k: _clamp_scale(float(data.get(k, 0)), _REASON_SCALE)
                    for k in _REASON_DIMS}
            return dims, str(data.get("analysis") or "")
    except Exception:
        pass
    return {}, raw[:200]


def _clamp_scale(x: float, hi: float) -> float:
    return max(0.0, min(hi, x))


def _clamp_half(x: float) -> float:
    if x >= 0.75:
        return 1.0
    if x >= 0.25:
        return 0.5
    return 0.0
