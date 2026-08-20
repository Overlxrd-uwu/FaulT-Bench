"""Main orchestrator — one scenario, one agent, full lifecycle."""
from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .checks import all_passed, run_checks
from .inject import apply_injection
from .lab import kathara_container_count, lclean, lstart
from .results import write_detail, write_row
from .runners.base import AgentResult
from .runners.cc_baseline import run_cc_baseline
from .runners.react_full import run_react_full
from .runners.sade import run_sade
from .scenario import Scenario, parse_scenario
from .scoring import score_submission
from .scoring.trace_digest import build_trace_digest
from .topology import summarize_topology


def _build_agent_prompt(sc: Scenario) -> str:
    """Compose what the agent actually sees: topology summary + symptom.

    The topology summary lists devices + collision domains (no IPs, no
    config) so the agent knows what device names exist and how they
    connect. Without this, agents spend their turn budget guessing device
    names instead of investigating.
    """
    topo = summarize_topology(sc.topology)
    return (
        "# Topology overview (devices + L2 connectivity)\n"
        f"{topo}\n\n"
        "# Symptom (user report)\n"
        f"{sc.prompt_to_agent}"
    )


DEFAULT_CONVERGE_WAIT = 90    # SETTLE (plain wait) after deploy so the lab converges and faults are meaningful; NOT a gated health check
DEPLOY_WAIT = 60             # max seconds to wait for all containers to launch on Docker
DEPLOY_POLL_INTERVAL = 5     # re-count containers every N s until all are up
DEFAULT_POST_INJECT_WAIT = 120
POST_INJECT_POLL_INTERVAL = 10


AGENT_RUNNERS = {
    "sade": run_sade,
    "cc-baseline": run_cc_baseline,
    "react-full": run_react_full,  # ReAct with the same 22 kathara_* tools as cc-baseline
}

AGENT_MODELS = {
    "sade": "claude-sonnet-4-6",
    "cc-baseline": "claude-sonnet-4-6",
    "react-full": "gpt-5",
}


def default_runs_root() -> Path:
    """The package's default ``runs/`` output root (sibling of ``src/``)."""
    return Path(__file__).resolve().parent.parent.parent / "runs"


def _scenario_folder(scenario_id: str, topology: str, scenario_class: str) -> str:
    """Folder name for a scenario under its topology dir.

    Prefixes the scenario class so run trees are self-classifying
    (``base__`` / ``false-premise__`` / ``wrong-device__`` / ``wrong-cause__``)
    and strips the redundant leading ``kath<N>-`` (the topology is already the
    parent dir): ``kath1-r1-ospfd-down`` -> ``base__r1-ospfd-down``. The full
    scenario_id and class are still recorded in summary.csv.
    """
    prefix = topology.lower() + "-"
    stem = scenario_id
    if scenario_id.lower().startswith(prefix):
        stem = scenario_id[len(prefix):]
    return f"{scenario_class}__{stem}"


def _uniquify(path: Path) -> Path:
    """Return ``path``, or ``path-2`` / ``path-3`` / ... if it already exists.

    Guards against two runs of the same (scenario, agent) landing in the same
    second -- astronomically unlikely (a run takes minutes) but cheap to be safe.
    """
    if not path.exists():
        return path
    i = 2
    while (path.parent / f"{path.name}-{i}").exists():
        i += 1
    return path.parent / f"{path.name}-{i}"


def run_one(scenario_path: Path, agent: str, runs_root: Path | None = None,
            converge_wait: int = DEFAULT_CONVERGE_WAIT,
            keep_lab: bool = False,
            stream: bool = True,
            run_group: str = "") -> dict:
    """Run one (scenario, agent) end-to-end. Returns the result row dict.

    Output is organised BY TOPOLOGY under ``runs_root`` (default: the package
    ``runs/`` dir):

        runs/<Topology>/summary.csv                   <- rolling, 1 row per run
        runs/<Topology>/<scenario>/<agent>/<run_id>/  <- this run's artifacts
            result.json  +  trace.log

    Each run is independent (its own folder, never overwritten); summary.csv
    accumulates every run so N>=3 repeats sit side by side. ``run_group`` tags
    the row's provenance (e.g. the bench manifest name) without changing layout.
    """
    sc = parse_scenario(scenario_path)
    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    run_id = now.strftime("%Y%m%d-%H%M%S")
    t0 = time.time()

    root = Path(runs_root) if runs_root is not None else default_runs_root()
    topo_dir = root / sc.topology
    run_dir = _uniquify(
        topo_dir / _scenario_folder(sc.scenario_id, sc.topology, sc.scenario_class)
        / agent / run_id)
    csv_path = topo_dir / "summary.csv"
    detail_path = run_dir / "result.json"
    trace_path = run_dir / "trace.log"

    row: dict = {
        "timestamp": timestamp,
        "run_id": run_dir.name,
        "run_group": run_group,
        "scenario_id": sc.scenario_id,
        "class": sc.scenario_class,
        "ticket_style": sc.ticket_style,
        "topology": sc.topology,
        "fault_family": sc.fault_family,
        "agent": agent,
        "model": AGENT_MODELS.get(agent, "unknown"),
        "status": "running",
        "submitted": False,
        "is_anomaly": "",
        "faulty_devices": [],
        "must_passed": 0, "must_total": 0,
        "bonus_passed": 0, "bonus_total": 0,
        "outcome_score": 0.0,
        "fix_score": "", "reasoning_score": "",
        "in_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0,
        "out_tokens": 0, "cost_usd": 0.0,
        "tool_calls": 0,
        "wall_seconds": 0.0, "error": "",
    }
    detail: dict = {
        "scenario": asdict(sc),
        "agent": agent,
        "timestamp": timestamp,
        "phases": {},
    }
    # The agent's trace is written to disk the moment the agent returns (see
    # below), so a failure in any LATER phase -- most likely the judge's API
    # call -- can never lose the investigation log. `ar` is kept in scope for
    # the error path.
    ar: AgentResult | None = None

    try:
        expected = lstart(sc.topology)
        # Deployment gate ONLY: wait until all containers have launched. We do
        # NOT functionally pre-check the baseline (no reachability/OSPF probes) --
        # a fully-deployed lab is assumed good; the meaningful gate is that the
        # INJECTION takes and stays alive (post-inject below).
        elapsed = 0
        n = kathara_container_count()
        while n < expected and elapsed < DEPLOY_WAIT:
            if stream:
                print(f"[DEPLOY] {n}/{expected} containers up @ {elapsed}s", flush=True)
            time.sleep(DEPLOY_POLL_INTERVAL)
            elapsed += DEPLOY_POLL_INTERVAL
            n = kathara_container_count()
        detail["phases"]["lstart"] = {"ok": n >= expected,
                                      "containers": n, "expected": expected}
        if n < expected:
            row["status"] = "deploy_incomplete"
            row["error"] = f"only {n}/{expected} containers launched after {elapsed}s"
            return _finalize(row, detail, csv_path, detail_path,
                             trace_path, "", t0, sc, keep_lab)
        # Settle: plain wait so the lab converges (faults become meaningful and
        # the agent sees a converged network). NOT a gated health check.
        if stream:
            print(f"[DEPLOY] all {n}/{expected} containers up; settling "
                  f"{converge_wait}s for convergence", flush=True)
        time.sleep(converge_wait)
        detail["phases"]["settle"] = {"converge_wait": converge_wait}

        inj = apply_injection(sc.topology, sc.injection)
        detail["phases"]["injection"] = asdict(inj)

        if inj.applied:
            post: list = []
            elapsed = 0
            while elapsed <= DEFAULT_POST_INJECT_WAIT:
                post = run_checks(sc.topology, sc.post_inject_check)
                if all_passed(post):
                    break
                time.sleep(POST_INJECT_POLL_INTERVAL)
                elapsed += POST_INJECT_POLL_INTERVAL
            detail["phases"]["post_inject_check"] = {
                "results": [asdict(r) for r in post],
                "wait_seconds": elapsed,
            }
            if not all_passed(post):
                row["status"] = "injection_failed"
                row["error"] = (f"after {elapsed}s: " + "; ".join(
                    f"{r.line.device}: {r.rationale}"
                    for r in post if not r.passed))
                return _finalize(row, detail, csv_path, detail_path,
                                 trace_path, "", t0, sc, keep_lab)
        else:
            detail["phases"]["post_inject_check"] = "skipped (note-only fault)"

        runner = AGENT_RUNNERS[agent]
        agent_prompt = _build_agent_prompt(sc)
        # Persist exactly what every agent receives so we can verify all three
        # got identical input (topology summary + symptom).
        detail["agent_input_prompt"] = agent_prompt
        ar = runner(
            prompt_to_agent=agent_prompt,
            lab_name=sc.topology,
            stream=stream,
        )
        # Persist the trace NOW, before judging: a judge-side failure must
        # never lose a completed investigation. Nothing after this line can
        # lose the trace.
        _write_trace(trace_path, ar.raw_trace)
        # Detail stores everything EXCEPT raw_trace (which goes to its own file)
        ar_dict = asdict(ar)
        ar_dict.pop("raw_trace", None)
        detail["phases"]["agent"] = ar_dict

        row["submitted"] = bool(ar.diagnosis_text)
        row["is_anomaly"] = ar.is_anomaly
        row["faulty_devices"] = ar.faulty_devices
        row["in_tokens"] = ar.in_tokens
        row["cache_read_tokens"] = ar.cache_read_tokens
        row["cache_write_tokens"] = ar.cache_write_tokens
        row["out_tokens"] = ar.out_tokens
        row["cost_usd"] = round(ar.cost_usd, 6)
        row["tool_calls"] = ar.tool_calls

        # The agent phase died before producing a verdict: the SDK surfaced an
        # API/auth/quota/transport error instead of a diagnosis (e.g. a key at
        # its usage cap returns "API Error: 400 ..." as the final text, which
        # a judge would score as a writeup). That is infrastructure, not an
        # answer and not silence: file it apart and re-run it. Turn-budget
        # exhaustion is NOT this (handled below), and a verdict that parsed is
        # kept even if the SDK then complained on the way out.
        if ar.error and not ar.hit_max_turns and not ar.parsed_ok:
            row["submitted"] = False
            row["status"] = "agent_error"
            row["error"] = ar.error + " | agent phase failed before a verdict; re-run"
            return _finalize(row, detail, csv_path, detail_path,
                             trace_path, ar.raw_trace, t0, sc, keep_lab)

        if not ar.diagnosis_text:
            row["status"] = "no_output"
            row["error"] = ar.error or "agent produced no final text"
            return _finalize(row, detail, csv_path, detail_path,
                             trace_path, ar.raw_trace, t0, sc, keep_lab)

        # Ran out of turns WITHOUT a parseable === DIAGNOSIS === block: this is
        # a non-submission, not a wrong answer. Don't score it (a partial probe
        # log would otherwise be scored as if it were an answer); flag it plainly so the
        # failure mode is "no_submission", distinct from a wrong diagnosis.
        if ar.hit_max_turns and not ar.parsed_ok:
            row["submitted"] = False
            row["status"] = "no_submission"
            row["error"] = (ar.error or "reached max_turns") + \
                " | hit max_turns with no valid === DIAGNOSIS === block"
            return _finalize(row, detail, csv_path, detail_path,
                             trace_path, ar.raw_trace, t0, sc, keep_lab)

        try:
            score = score_submission(
                diagnosis_text=ar.diagnosis_text,
                ground_truth=sc.ground_truth,
                scoring_axes=sc.scoring_axes,
                fix_spec=sc.fix_spec,
                is_anomaly=ar.is_anomaly,
                faulty_devices=ar.faulty_devices,
                # Reasoning is judged trace-aware: the judge verifies the
                # writeup's claims against what the agent actually
                # ran/observed, instead of rewarding narration (see
                # scoring/trace_digest.py).
                trace_digest=build_trace_digest(ar.raw_trace),
            )
        except Exception as exc:
            # The agent DID answer; only the judge failed (API outage, quota,
            # network). File it as its own status so it is never mistaken for
            # agent silence, and can be re-judged from the saved diagnosis
            # without re-running the agent.
            row["status"] = "judge_failed"
            row["error"] = f"judge: {type(exc).__name__}: {exc}"
            detail["phases"]["error"] = row["error"]
            return _finalize(row, detail, csv_path, detail_path,
                             trace_path, ar.raw_trace, t0, sc, keep_lab)
        detail["phases"]["score"] = asdict(score)

        row["must_passed"] = score.must_passed
        row["must_total"] = score.must_total
        row["bonus_passed"] = score.bonus_passed
        row["bonus_total"] = score.bonus_total
        row["outcome_score"] = round(score.outcome_score, 3)
        row["fix_score"] = (
            round(score.fix_score, 3)
            if score.fix_score is not None else "")
        row["reasoning_score"] = (
            round(score.reasoning_score, 3)
            if score.reasoning_score is not None else "")
        row["status"] = "scored"
        if score.error:
            row["error"] = f"scorer: {score.error}"
        if not ar.parsed_ok:
            row["error"] = (row.get("error") or "") + " | structured-block parse failed"

        return _finalize(row, detail, csv_path, detail_path,
                         trace_path, ar.raw_trace, t0, sc, keep_lab)

    except Exception as exc:
        row["status"] = "pipeline_error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        detail["phases"]["error"] = row["error"]
        # keep whatever trace exists -- never finalise with an empty one when
        # the agent phase completed
        return _finalize(row, detail, csv_path, detail_path,
                         trace_path, ar.raw_trace if ar else "", t0, sc,
                         keep_lab)


def _write_trace(trace_path: Path, trace_text: str) -> None:
    if trace_text:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(trace_text, encoding="utf-8")


def _finalize(row: dict, detail: dict, csv_path: Path, detail_path: Path,
              trace_path: Path, trace_text: str,
              t0: float, sc: Scenario, keep_lab: bool) -> dict:
    if not keep_lab:
        try:
            lclean(sc.topology)
        except Exception:
            pass
    row["wall_seconds"] = round(time.time() - t0, 1)
    write_row(csv_path, row)
    write_detail(detail_path, detail)
    _write_trace(trace_path, trace_text)   # idempotent: same content if already written
    return row
