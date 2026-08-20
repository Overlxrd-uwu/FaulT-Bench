"""Recompute every reported number from final_results/, in one file.

    python tools/compute_metrics.py           read final_results/, print the number sheet, write
                                              metrics/ (sheet, the four tables, figure, CSVs); if the
                                              paper sources are present, also write the tables and
                                              figure into the paper and check its .tex
    python tools/compute_metrics.py --check   same, but write nothing into the paper (report only)

Metric definitions are in the paper; the score implementation is
faultbench/src/faultbench/scoring/llm_judge.py.

Reads   final_results/MANIFEST.csv and every final_results/**/result.json (never modified).
Writes  metrics/numbers.txt                        the sheet: every number the paper quotes, with its rule
        metrics/table_e1.tex, _e2.tex, _e3.tex, _e4.tex   the four results tables (LaTeX): outcome by
                                                   class, outcome by persona, fix and reasoning, effort
                                                   (tool calls, tokens, API cost, spend on timeouts)
        metrics/results.pdf, metrics/results.png   the results figure (matplotlib; skipped if missing)
        metrics/cells.csv, metrics/runs.csv        arm x class x agent x pass numbers; one row per run
        acmart-primary/Paper/sections/results.tex   the effort table, between the comment markers
                                                   % BEGIN GENERATED e4 / % END GENERATED e4; the compact
                                                   E1/E2/E3 tables are hand-maintained (their cell values are checked)
        acmart-primary/Paper/Figures/results.pdf   the figure            (both only if the paper is present)
Checks  MANIFEST.csv against result.json (scores and tool calls); every scored run's outcome re-derived
        from its own judge breakdown (0.7 must + 0.3 bonus, renormalised), the deterministic axes
        re-evaluated, reasoning = mean of its three dimensions, fix in {0, .5, 1}; then, if the paper
        is present, every table cell and every prose number of the paper verbatim in the LaTeX.
        Exit code 1 if anything fails.

Policy  A score is the mean over the runs that answered. A run that used up its turn budget without
        submitting a diagnosis is a timeout: counted next to every score, never scored. Per-arm
        values are means over passes (SD across passes); "pooled" puts all runs of the arm together.
        Over-diagnosis is the share of answered runs on a healthy network that report a fault.
        Effort (tool calls, tokens, API cost) is reported per run for the runs that answered and for
        the timeouts separately, all passes pooled; input tokens = every token the model read
        (fresh + cache reads + cache writes); cost is the API's own figure for the Claude agents and
        list-price arithmetic for ReAct (gpt-5: 1.25 / 0.125 / 10 US$ per M fresh / cached / output).
        Differences quoted in the text are taken between the 3-decimal values shown in the tables,
        so the text and the tables agree.

Standard library only, except matplotlib for the figure.
"""
from __future__ import annotations

import argparse
import collections
import csv
import functools
import json
import math
import random
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "final_results"
PAPER = ROOT / "acmart-primary" / "Paper"
METRICS = ROOT / "metrics"

ARMS = ["base80", "cex120", "novice", "novice_confident", "naive", "naive_unsure", "naive_no_detail"]
AGENTS = ["sade", "cc-baseline", "react-full"]
NAME = {"sade": "SADE", "cc-baseline": "CC-Baseline", "react-full": "ReAct", "ALL": "All"}
CLASSES = [("Correct-Fault (80)", "base80", "fault"), ("False-Premise (72)", "cex120", "fp"),
           ("Wrong-Device (24)", "cex120", "wd"), ("Wrong-Cause (24)", "cex120", "wc")]
PERSONAS = [("Standard", "cex120", "3.2"), ("Naive", "naive", "2.1"), ("Novice", "novice", "1.9"),
            ("Novice-confident", "novice_confident", "1.9"), ("Naive-unsure", "naive_unsure", "2.1"),
            ("Naive, no detail", "naive_no_detail", "0.0")]          # name, arm, mean identifiers per ticket (Table 2)
PERSONA_ARMS = ["naive", "novice", "novice_confident", "naive_unsure", "naive_no_detail"]
VOICE_SUFFIXES = ["-novice-confident", "-naive-no-detail", "-naive-unsure", "-novice", "-naive"]
NIKA_TOPOS = {"dc_clos_bgp", "ospf_enterprise_static", "simple_bgp"}


# ============================================================ small helpers
def fmean(xs):
    xs = [x for x in xs if x is not None]
    return st.fmean(xs) if xs else None


def sdev(xs):
    xs = [x for x in xs if x is not None]
    return st.stdev(xs) if len(xs) >= 2 else None


def r3(x):                       # the 3-decimal value the tables show
    return None if x is None else round(x, 3)


def d3(x):                       # table style: .932, 1.000
    return "1.000" if x >= 0.9995 else f"{x:.3f}"[1:]


def pn(x):                       # whole percent as a string
    return str(round(x))


def scenario_class(arm, sid):
    if arm == "base80":
        return "fault"
    if "false-premise" in sid:
        return "fp"
    if sid.endswith("-misdirected"):
        return "wd"
    if sid.endswith("-misdescribed"):
        return "wc"
    raise ValueError(f"cannot classify {sid!r} in {arm}")


def core_scenario(sid):
    for suf in VOICE_SUFFIXES:
        if sid.endswith(suf):
            return sid[: -len(suf)]
    return sid


# ============================================================ 1. load + integrity
def outcome_formula(mp, mt, bp, bt):
    parts = []
    if mt:
        parts.append((mp / mt, 0.7))
    if bt:
        parts.append((bp / bt, 0.3))
    return min(1.0, sum(f * w for f, w in parts) / sum(w for _, w in parts)) if parts else 0.0


def assert_axis(axis, is_anom, devs):
    a = axis.strip().lower()
    fd = " ".join(devs or []).lower()
    if a.startswith("assert-no-anomaly"):
        return 1.0 if is_anom is False else 0.0
    if a.startswith("assert-no-faulty-devices"):
        return 1.0 if not (devs or []) else 0.0
    dev = axis.split(":", 1)[1].strip().lower() if ":" in axis else ""
    if a.startswith("assert-not-identifies"):
        return 1.0 if (dev and dev not in fd) else 0.0
    if a.startswith("assert-identifies"):
        return 1.0 if (dev and dev in fd) else 0.0
    return 0.0


def load(final=FINAL):
    """One dict per run + the integrity issues found while reading."""
    man = list(csv.DictReader(open(final / "MANIFEST.csv", encoding="utf-8")))
    runs, issues = [], collections.Counter()
    seen = collections.Counter((r["arm"], r["pass"], r["scenario_id"], r["agent"]) for r in man)
    if any(c > 1 for c in seen.values()):
        issues["duplicate cell in MANIFEST"] += sum(1 for c in seen.values() if c > 1)
    for r in man:
        d = final / r["arm"] / r["topology"] / r["scenario_id"] / r["agent"] / r["pass"]
        j = json.load(open(d / "result.json", encoding="utf-8"))
        ag = j["phases"].get("agent") or {}
        sc = j["phases"].get("score")
        scored = r["status"] == "scored"
        hit, parsed = bool(ag.get("hit_max_turns")), bool(ag.get("parsed_ok"))
        for k in ("outcome_score", "fix_score", "reasoning_score"):
            mv = None if r[k] in ("", None) else float(r[k])
            jv = None if sc is None else sc.get(k)
            if (mv is None) != (jv is None) or (mv is not None and abs(mv - float(jv)) > 5e-4):
                issues["MANIFEST score != result.json"] += 1
        mp = mt = bp = bt = 0          # must/bonus check counts (kept per run for the weight-sensitivity check)
        if scored:
            if sc is None:
                issues["scored run without a score object"] += 1
            else:
                ax = sc["per_axis"]
                mp = sum(1 for a in ax if a["kind"] in ("must", "assert") and a["score"] >= 1.0)
                mt = sum(1 for a in ax if a["kind"] in ("must", "assert"))
                bp = sum(1 for a in ax if a["kind"] == "bonus" and a["score"] >= 1.0)
                bt = sum(1 for a in ax if a["kind"] == "bonus")
                if abs(outcome_formula(mp, mt, bp, bt) - float(sc["outcome_score"])) > 1e-9:
                    issues["outcome != 0.7 must + 0.3 bonus over per_axis"] += 1
                if "final_score" in sc and abs(float(sc["final_score"]) - float(sc["outcome_score"])) > 1e-12:
                    issues["final_score != outcome_score"] += 1
                for a in ax:
                    if a["kind"] == "assert" and assert_axis(a["axis"], ag.get("is_anomaly"), ag.get("faulty_devices")) != a["score"]:
                        issues["deterministic axis re-evaluates differently"] += 1
                dims, rs = sc.get("reasoning_dims") or {}, sc.get("reasoning_score")
                if dims and (rs is None or abs(sum(dims.values()) / len(dims) - rs) > 1e-3):
                    issues["reasoning != mean(grounding, causality, coverage)"] += 1
                if not dims and rs is not None:
                    issues["reasoning score without dimensions"] += 1
                if rs is None:
                    issues["scored run without a reasoning score"] += 1
                fx = sc.get("fix_score")
                if fx is not None and fx not in (0.0, 0.5, 1.0):
                    issues["fix outside {0, .5, 1}"] += 1
        else:
            if sc is not None:
                issues["timeout carries a score object"] += 1
            if not (hit and not parsed):
                issues["unscored run that is not a turn-budget exhaustion"] += 1
        if int(r["tool_calls"] or 0) != int(ag.get("tool_calls") or 0):
            issues["MANIFEST tool_calls != result.json"] += 1
        tok = {k: int(ag.get(k) or 0) for k in ("in_tokens", "cache_read_tokens", "cache_write_tokens", "out_tokens")}
        if not any(tok.values()):
            issues["run without token accounting"] += 1
        runs.append({
            "arm": r["arm"], "pass": r["pass"], "topology": r["topology"],
            "family": "nika" if r["topology"] in NIKA_TOPOS else "kath",
            "scenario_id": r["scenario_id"], "core_scenario": core_scenario(r["scenario_id"]),
            "class": scenario_class(r["arm"], r["scenario_id"]), "agent": r["agent"],
            "scored": scored, "status": r["status"],
            "is_anomaly": r["is_anomaly"] == "True",
            "outcome": float(r["outcome_score"]) if r["outcome_score"] else None,
            "fix": float(r["fix_score"]) if r["fix_score"] else None,
            "reasoning": float(r["reasoning_score"]) if r["reasoning_score"] else None,
            "tool_calls": int(r["tool_calls"] or 0),
            # effort: what the model read (fresh + cache reads + cache writes) and wrote, and the API cost
            "in_tokens": tok["in_tokens"] + tok["cache_read_tokens"] + tok["cache_write_tokens"],
            "fresh_tokens": tok["in_tokens"], "cache_read_tokens": tok["cache_read_tokens"],
            "cache_write_tokens": tok["cache_write_tokens"], "out_tokens": tok["out_tokens"],
            "cost_usd": float(ag.get("cost_usd") or 0.0),
            "must": (mp, mt), "bonus": (bp, bt),
            "path": d.relative_to(final).as_posix(),
        })
    return runs, issues


# ============================================================ 2. aggregation
RUNS: list[dict] = []
SCORE_KEYS = ("outcome", "fix", "reasoning")


def block(rs):
    """The numbers for one set of runs."""
    scored = [r for r in rs if r["scored"]]
    tmo = [r for r in rs if not r["scored"]]
    out = {"n_runs": len(rs), "n_scored": len(scored), "n_timeout": len(tmo),
           "n_overdiag": sum(1 for r in scored if r["is_anomaly"])}
    out["timeout_pct"] = 100.0 * out["n_timeout"] / len(rs) if rs else None
    out["overdiag_pct"] = 100.0 * out["n_overdiag"] / len(scored) if scored else None
    for k in SCORE_KEYS:
        out[k] = fmean([r[k] for r in scored])
    # effort per run, for the runs that answered (_a) and for the timeouts (_t): mean tool calls,
    # tokens read and written, API cost; plus the cell's total spend and the share that went to timeouts
    for tag, sub in (("a", scored), ("t", tmo)):
        out["calls_" + tag] = fmean([r["tool_calls"] for r in sub])
        out["in_" + tag] = fmean([r["in_tokens"] for r in sub])
        out["out_" + tag] = fmean([r["out_tokens"] for r in sub])
        out["usd_" + tag] = fmean([r["cost_usd"] for r in sub])
    out["usd_total"] = sum(r["cost_usd"] for r in rs)
    out["usd_timeout_pct"] = 100.0 * sum(r["cost_usd"] for r in tmo) / out["usd_total"] if out["usd_total"] else None
    return out


def passes_of(arm):
    return sorted({r["pass"] for r in RUNS if r["arm"] == arm})


@functools.lru_cache(maxsize=None)
def cell(arm, cls=None, agent="ALL", pas="pooled"):
    """Stats for arm x class x agent at one pass, pooled over passes, or the
    mean over passes (pas='mean'; then also sd_<score> across passes)."""
    rs = [r for r in RUNS if r["arm"] == arm and (cls is None or r["class"] == cls)
          and (agent == "ALL" or r["agent"] == agent)]
    if pas == "pooled":
        return block(rs)
    if pas != "mean":
        return block([r for r in rs if r["pass"] == pas])
    per = [block([r for r in rs if r["pass"] == p]) for p in passes_of(arm)]
    out = {k: fmean([b[k] for b in per]) for k in per[0]}
    for k in SCORE_KEYS:
        out["sd_" + k] = sdev([b[k] for b in per])
    return out


def v(arm, cls=None, agent="ALL", pas="mean"):
    """Outcome over answered runs, at 3 decimals (what the tables show)."""
    return r3(cell(arm, cls, agent, pas)["outcome"])


def prange(arm, cls, agent):
    """(min, max) of the per-pass outcome for one agent cell."""
    xs = [r3(cell(arm, cls, agent, p)["outcome"]) for p in passes_of(arm)]
    return min(xs), max(xs)


# ============================================================ 3. the two tables
def row_cells(arm, cls, agent, with_od):
    ps = passes_of(arm)
    out = [d3(cell(arm, cls, agent, p)["outcome"]) for p in ps] + ["---"] * (3 - len(ps))
    m = cell(arm, cls, agent, "mean")
    out.append(d3(m["outcome"]) + "\\,$\\pm$\\," + d3(m["sd_outcome"]))
    out += [str(cell(arm, cls, agent, p)["n_timeout"]) for p in ps] + ["---"] * (3 - len(ps))
    p = cell(arm, cls, agent)
    out += [f"{p['n_timeout']}/{p['n_runs']}", f"{p['timeout_pct']:.0f}\\%",
            f"{round(p['overdiag_pct'])}\\%" if with_od else "---"]
    return out


def render_e1():
    cf = v("base80", "fault")
    L = [r"\begin{table*}[t]",
         r"  \caption{E1: outcome on the 200 core scenarios by class, agent and pass (1{,}800 runs).",
         r"  Columns are explained at the start of Section~\ref{sec:results}. CC-B.\ $=$ CC-Baseline.}",
         r"  \label{tab:e1-outcome}", r"  \footnotesize", r"  \setlength{\tabcolsep}{4.2pt}",
         r"  \begin{tabular}{llccccrrrrrcr}", r"    \toprule",
         r"    & & \multicolumn{4}{c}{Outcome, runs that answered} & \multicolumn{5}{c}{Timeouts} & & \\",
         r"    \cmidrule(lr){3-6}\cmidrule(lr){7-11}",
         r"    Class & Agent & P1 & P2 & P3 & mean\,$\pm$\,SD & P1 & P2 & P3 & all/$N$ & \% & Over-d. & $\Delta$ \\",
         r"    \midrule"]
    for lab, arm, cls in CLASSES:
        for i, ag in enumerate(AGENTS + ["ALL"]):
            name = "CC-B." if ag == "cc-baseline" else (r"\emph{All}" if ag == "ALL" else NAME[ag])
            delta = ""
            if ag == "ALL":
                delta = "---" if cls == "fault" else f"${v(arm, cls) - cf:+.3f}$".replace("+0.", "+.").replace("-0.", "-.")
            L.append(f"    {(lab if i == 0 else ''):<20} & {name:<10} & " + " & ".join(row_cells(arm, cls, ag, cls == "fp")) + f" & {delta} \\\\")
        if cls != "wc":
            L.append(r"    \addlinespace[2.5pt]")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}", ""]
    return "\n".join(L)


def render_e2():
    L = [r"\begin{table*}[t]",
         r"  \caption{E2: outcome on the 72 healthy networks by ticket persona, agent and pass",
         r"  (Standard: three passes; other personas: two). Columns as in Table~\ref{tab:e1-outcome};",
         r"  ID = mean verifiable identifiers per ticket (Table~\ref{tab:dataset}).}",
         r"  \label{tab:e2-outcome}", r"  \footnotesize", r"  \setlength{\tabcolsep}{4.2pt}",
         r"  \begin{tabular}{lclccccrrrrrc}", r"    \toprule",
         r"    & & & \multicolumn{4}{c}{Outcome, runs that answered} & \multicolumn{5}{c}{Timeouts} & \\",
         r"    \cmidrule(lr){4-7}\cmidrule(lr){8-12}",
         r"    Persona & ID & Agent & P1 & P2 & P3 & mean\,$\pm$\,SD & P1 & P2 & P3 & all/$N$ & \% & Over-d. \\",
         r"    \midrule"]
    for k, (lab, arm, idc) in enumerate(PERSONAS):
        for i, ag in enumerate(AGENTS + ["ALL"]):
            name = "CC-B." if ag == "cc-baseline" else (r"\emph{All}" if ag == "ALL" else NAME[ag])
            L.append(f"    {(lab if i == 0 else ''):<18} & {(idc if i == 0 else ''):<3} & {name:<10} & "
                     + " & ".join(row_cells(arm, "fp", ag, True)) + " \\\\")
        if k < len(PERSONAS) - 1:
            L.append(r"    \addlinespace[2.5pt]")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}", ""]
    return "\n".join(L)


def render_e3():
    """Table 6: fix and reasoning by class and persona x agent (mean +- SD across passes)."""
    def ms(arm, cls, ag, key):
        m = cell(arm, cls, ag, "mean")
        return "---" if m[key] is None else d3(m[key]) + "\\,$\\pm$\\," + d3(m["sd_" + key])
    L = [r"\begin{table*}[t]",
         r"  \caption{Fix and reasoning scores by class and persona (mean\,$\pm$\,SD across passes, over the",
         r"  runs that answered). Fix exists only where there is something to repair: correct-fault, wrong-device",
         r"  and wrong-cause scenarios. CC-B.\ $=$ CC-Baseline.}",
         r"  \label{tab:fix-reasoning}", r"  \footnotesize", r"  \setlength{\tabcolsep}{5pt}",
         r"  \begin{tabular}{lcccccccc}", r"    \toprule",
         r"    & \multicolumn{4}{c}{Fix} & \multicolumn{4}{c}{Reasoning} \\",
         r"    \cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"    Class / persona & SADE & CC-B. & ReAct & \emph{All} & SADE & CC-B. & ReAct & \emph{All} \\",
         r"    \midrule"]
    blocks = [(lab, arm, cls) for lab, arm, cls in CLASSES] + [(lab, arm, "fp") for lab, arm, _ in PERSONAS[1:]]
    for i, (lab, arm, cls) in enumerate(blocks):
        if i == len(CLASSES):
            L.append(r"    \addlinespace[2.5pt]")
        L.append(f"    {lab:<20} & " + " & ".join(ms(arm, cls, ag, "fix") for ag in AGENTS + ["ALL"])
                 + " & " + " & ".join(ms(arm, cls, ag, "reasoning") for ag in AGENTS + ["ALL"]) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}", ""]
    return "\n".join(L)


def render_e4():
    """Table 7: effort per run by class and persona x agent, all passes pooled: tool calls, tokens and
    API cost for the runs that answered and for the timeouts, and the share of the spend on timeouts."""
    big = lambda n: f"{n:,}".replace(",", "{,}")
    f1 = lambda x: "---" if x is None else f"{x:.1f}"
    k0 = lambda x: "---" if x is None else big(round(x / 1000))
    k1 = lambda x: "---" if x is None else f"{x / 1000:.1f}"
    usd = lambda x: "---" if x is None else f"{x:.2f}"

    def row(lab, c, ag):
        name = "CC-B." if ag == "cc-baseline" else NAME[ag]
        vals = [big(c["n_runs"]), f1(c["calls_a"]), k0(c["in_a"]), k1(c["out_a"]), usd(c["usd_a"]),
                str(c["n_timeout"]), f1(c["calls_t"]), k0(c["in_t"]), k1(c["out_t"]), usd(c["usd_t"]),
                f"{c['usd_timeout_pct']:.0f}\\%"]
        return f"    {lab:<20} & {name:<6} & " + " & ".join(vals) + r" \\"

    L = [r"\begin{table*}[t]",
         r"  \caption{Effort per run, over all passes: mean tool calls, input and output tokens (thousands)",
         r"  and API cost (US\$, list prices) for the runs that answered and for the timeouts, and the share of",
         r"  the cell's API spend that went to timeouts. Input counts every token the model read on every turn,",
         r"  cached context included. CC-B.\ $=$ CC-Baseline.}",
         r"  \label{tab:effort}", r"  \footnotesize", r"  \setlength{\tabcolsep}{4pt}",
         r"  \begin{tabular}{llrrrrrrrrrrr}", r"    \toprule",
         r"    & & & \multicolumn{4}{c}{Runs that answered} & \multicolumn{6}{c}{Timeouts} \\",
         r"    \cmidrule(lr){4-7}\cmidrule(lr){8-13}",
         r"    Class / persona & Agent & $N$ & Calls & In (k) & Out (k) & US\$ & $n$ & Calls & In (k) & Out (k) & US\$ & Spend \\",
         r"    \midrule"]
    blocks = [(lab, arm, cls) for lab, arm, cls in CLASSES] + [(lab, arm, "fp") for lab, arm, _ in PERSONAS[1:]]
    for i, (lab, arm, cls) in enumerate(blocks):
        if i:
            L.append(r"    \addlinespace[2.5pt]")
        for j, ag in enumerate(AGENTS):
            L.append(row(lab if j == 0 else "", cell(arm, cls, ag), ag))
    L.append(r"    \midrule")
    for j, ag in enumerate(AGENTS):
        L.append(row("All runs" if j == 0 else "", block([r for r in RUNS if r["agent"] == ag]), ag))
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}", ""]
    return "\n".join(L)


TABLES = {"e1": render_e1, "e2": render_e2, "e3": render_e3, "e4": render_e4}


def table_span(text, key):
    a = text.index(f"% BEGIN GENERATED {key}"); a = text.index("\n", a) + 1
    return a, text.index(f"% END GENERATED {key}")


def write_tables(results_tex: Path):
    # only the effort table (e4) is generated in place; the E1/E2/E3 tables are
    # hand-maintained in the paper and compact_values() checks their cell values
    text = results_tex.read_bytes().decode("utf-8").replace("\r\n", "\n")
    a, b = table_span(text, "e4")
    text = text[:a] + TABLES["e4"]() + text[b:]
    results_tex.write_bytes(text.encode("utf-8"))


def compact_values():
    """Every machine value the hand-maintained results tables (E1, E2, fix/reasoning) must contain."""
    out = []
    def pm(arm, cls, ag, key="outcome"):
        m = cell(arm, cls, ag, "mean")
        if m[key] is None:
            return None
        return f"${d3(m[key])} \\pm {d3(m['sd_' + key])}$"
    cf = v("base80", "fault")
    for _, arm, cls in CLASSES:
        for ag in AGENTS + ["ALL"]:
            out.append((f"e1 {cls} {ag}", pm(arm, cls, ag)))
        if cls != "fault":
            out.append((f"e1 {cls} delta", f"${v(arm, cls) - cf:+.3f}$".replace("+0.", "+.").replace("-0.", "-.")))
        p = cell(arm, cls)
        out.append((f"e1 {cls} T.O.", f"{p['timeout_pct']:.0f}\\%"))
        if cls == "fp":
            out.append((f"e1 {cls} over-d", f"{round(p['overdiag_pct'])}\\%"))
    for lab, arm, _ in PERSONAS:
        for ag in AGENTS + ["ALL"]:
            out.append((f"e2 {lab} {ag}", pm(arm, "fp", ag)))
        p = cell(arm, "fp")
        out += [(f"e2 {lab} over-d", f"{round(p['overdiag_pct'])}\\%"), (f"e2 {lab} T.O.", f"{p['timeout_pct']:.0f}\\%")]
    for lab, arm, cls in list(CLASSES) + [(lab, arm, "fp") for lab, arm, _ in PERSONAS[1:]]:
        for ag in AGENTS + ["ALL"]:
            for key in ("fix", "reasoning"):
                out.append((f"e3 {lab} {ag} {key}", pm(arm, cls, ag, key)))
    return [(lab, s) for lab, s in out if s]


def check_tables(results_tex: Path):
    text = results_tex.read_bytes().decode("utf-8").replace("\r\n", "\n")
    try:
        a, b = table_span(text, "e4"); same = text[a:b] == TABLES["e4"]()
    except ValueError:
        same = False
    print(f"  {'OK     ' if same else 'DIFFERS'} generated table e4 in {results_tex.name}")
    vals = compact_values()
    missing = [(lab, s) for lab, s in vals if s not in text]
    print(f"  {len(vals)} hand-table cell values (E1/E2/E3):", "ALL PRESENT" if not missing else f"{len(missing)} MISSING")
    for lab, s in missing:
        print(f"    {lab}: {s!r}")
    return same and not missing


# ============================================================ 4. the figure
def figure(out_dir: Path):
    try:
        import matplotlib
    except ImportError:
        print("  matplotlib not installed; figure skipped")
        return False
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
    color = {"sade": "#0072B2", "cc-baseline": "#E69F00", "react-full": "#009E73"}
    labels = ["Standard", "Naive", "Novice", "Novice-\nconfident", "Naive-\nunsure", "Naive,\nno detail"]
    arms = [p[1] for p in PERSONAS]
    ids = [p[2] for p in PERSONAS]
    x = list(range(len(arms)))

    fig = plt.figure(figsize=(7.0, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.75, wspace=0.10,
                          left=0.075, right=0.985, top=0.94, bottom=0.15)
    # (a) outcome by persona and agent
    ax = fig.add_subplot(gs[0, :])
    w = 0.25
    for i, ag in enumerate(AGENTS):
        ys = [cell(a, "fp", ag, "mean")["outcome"] for a in arms]
        sd = [cell(a, "fp", ag, "mean")["sd_outcome"] or 0 for a in arms]
        ax.bar([k + (i - 1) * w for k in x], ys, w, color=color[ag], label=NAME[ag], yerr=sd, capsize=1.6,
               error_kw={"elinewidth": 0.6, "capthick": 0.6, "ecolor": "#333"})
    for k, a in enumerate(arms):
        m = cell(a, "fp", "ALL", "mean")["outcome"]
        ax.plot([k - 0.42, k + 0.42], [m, m], color="#111", lw=1.3, zorder=5)
        top = max(cell(a, "fp", ag, "mean")["outcome"] + (cell(a, "fp", ag, "mean")["sd_outcome"] or 0) for ag in AGENTS)
        ax.text(k, top + 0.03, f"all {m:.2f}", ha="center", va="bottom", fontsize=7.4, color="#111")
    ax.plot([], [], color="#111", lw=1.3, label="all agents (mean)")
    ax.set_xticks(x); ax.set_xticklabels([f"{n}\nID {i}" for n, i in zip(labels, ids)])
    ax.set_ylim(0, 1.42); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel("outcome over answered runs")

    def bracket(x0, x1, y, text, dy=0.035):
        ax.plot([x0, x0, x1, x1], [y - dy, y, y, y - dy], color="#333", lw=0.7)
        ax.text((x0 + x1) / 2, y + 0.012, text, ha="center", va="bottom", fontsize=7.6, color="#333")
    bracket(2, 3, 1.07, f"tone only: {v('novice_confident', 'fp') - v('novice', 'fp'):+.2f}")
    bracket(4, 5, 1.07, f"identifiers removed: {v('naive_no_detail', 'fp') - v('naive_unsure', 'fp'):+.2f}")
    ax.set_title("(a) Outcome by ticket persona and agent (72 healthy networks)", loc="left")
    ax.legend(loc="upper left", ncol=4, frameon=False, handlelength=1.1, columnspacing=1.2, bbox_to_anchor=(0.0, 1.03))
    # (b) how the runs end, per agent
    cols = {"ok": "#8FBBD9", "od": "#D55E00", "tmo": "#7F7F7F"}
    for i, ag in enumerate(AGENTS):
        ax = fig.add_subplot(gs[1, i])
        for k, a in enumerate(arms):
            c = cell(a, "fp", ag)
            n = c["n_runs"]; tmo = c["n_timeout"] / n; od = c["n_overdiag"] / n; ok = (c["n_scored"] - c["n_overdiag"]) / n
            ax.bar(k, ok, 0.62, color=cols["ok"]); ax.bar(k, od, 0.62, bottom=ok, color=cols["od"])
            ax.bar(k, tmo, 0.62, bottom=ok + od, color=cols["tmo"])
            for base, h in ((ok, od), (ok + od, tmo)):
                if h >= 0.10:
                    ax.text(k, base + h / 2, f"{100 * h:.0f}%", ha="center", va="center", fontsize=6.8, color="white")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.8); ax.set_ylim(0, 1.0)
        ax.set_title(NAME[ag], loc="left", color=color[ag], fontweight="bold", fontsize=8.5, pad=3)
        if i == 0:
            ax.set_ylabel("share of runs")
        else:
            ax.set_yticklabels([])
    fig.text(0.075, 0.485, "(b) How the runs end, per agent and persona (all passes)", fontsize=9, ha="left", va="bottom")
    fig.legend(handles=[Patch(color=cols["ok"], label="reports no fault (correct)"),
                        Patch(color=cols["od"], label="reports a fault (over-diagnosis)"),
                        Patch(color=cols["tmo"], label="timeout (no diagnosis submitted)")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.2, columnspacing=1.8)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "results.pdf"); fig.savefig(out_dir / "results.png", dpi=170)
    return True


# ============================================================ 5. the numbers the paper quotes
def weight_sensitivity(weights=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
    """Recompute every answered run's outcome with the accuracy (must) weight set to each value in
    `weights` (bonus weight 1-w; renormalised when a scenario has only one kind of check) and report
    whether the comparisons the paper draws survive: the class order, the agent ranking on the three
    incorrect-reporting classes, the sign of the persona contrasts, and the largest shift of any
    class/persona mean from its 0.7 value. Means are pooled over agents and passes."""
    def out_w(r, w):
        parts = []
        if r["must"][1]:
            parts.append((r["must"][0] / r["must"][1], w))
        if r["bonus"][1]:
            parts.append((r["bonus"][0] / r["bonus"][1], 1 - w))
        return sum(f * x for f, x in parts) / sum(x for _, x in parts) if parts else 0.0
    answered = [r for r in RUNS if r["scored"]]
    mean = lambda sel, w: st.fmean(out_w(r, w) for r in answered if sel(r))
    cls_sel = {"fault": lambda r: r["arm"] == "base80" and r["class"] == "fault",
               "fp": lambda r: r["arm"] == "cex120" and r["class"] == "fp",
               "wd": lambda r: r["class"] == "wd", "wc": lambda r: r["class"] == "wc"}
    per_sel = {arm: (lambda r, arm=arm: r["arm"] == arm and r["class"] == "fp")
               for arm in ("cex120", "novice", "novice_confident", "naive", "naive_unsure", "naive_no_detail")}
    ref = {k: mean(s, 0.7) for k, s in {**cls_sel, **per_sel}.items()}
    max_shift, class_ok, rank_ok, sign_ok = 0.0, True, True, True
    contrasts = (("novice_confident", "novice"), ("naive_unsure", "naive"), ("naive_no_detail", "naive_unsure"),
                 ("novice", "cex120"))
    for w in weights:
        cur = {k: mean(s, w) for k, s in {**cls_sel, **per_sel}.items()}
        max_shift = max(max_shift, max(abs(cur[k] - ref[k]) for k in cur))
        # class order: misdirection close to Correct-Fault, False-Premise clearly below all three
        class_ok &= cur["fp"] < min(cur["fault"], cur["wd"], cur["wc"])
        for c in ("fp", "wd", "wc"):
            by_agent = lambda ww: sorted(AGENTS, key=lambda a: -mean(lambda r, a=a: cls_sel[c](r) and r["agent"] == a, ww))
            rank_ok &= by_agent(w) == by_agent(0.7)
        sign_ok &= all((cur[a] - cur[b] > 0) == (ref[a] - ref[b] > 0) for a, b in contrasts)
    return {"max_shift": max_shift, "class_order_kept": class_ok, "agent_rank_kept": rank_ok,
            "contrast_sign_kept": sign_ok, "weights": weights}


def sheet():
    """Return (rows, expect). rows: (label, value, how) printed as the number sheet.
    expect: (label, string) that must appear verbatim in the paper's LaTeX."""
    rows, expect = [], []
    add = lambda lab, val, how="": rows.append((lab, val, how))
    ex = lambda lab, s: expect.append((lab, s))
    pc = lambda x: pn(x) + "\\%"

    cf, fp, wd, wc = v("base80", "fault"), v("cex120", "fp"), v("cex120", "wd"), v("cex120", "wc")
    nov, conf = v("novice", "fp"), v("novice_confident", "fp")
    nai, plus, minus = v("naive", "fp"), v("naive_unsure", "fp"), v("naive_no_detail", "fp")

    # ---- E1: classes
    add("Correct-Fault mean", f"{cf:.3f}", "outcome over answered runs, all agents, mean over passes")
    base = [r for r in RUNS if r["arm"] == "base80" and r["scored"]]
    nika = st.fmean(r["outcome"] for r in base if r["family"] == "nika")
    kath = st.fmean(r["outcome"] for r in base if r["family"] == "kath")
    react_nika = st.fmean(r["outcome"] for r in base if r["family"] == "nika" and r["agent"] == "react-full")
    add("  NIKA networks", f"{nika:.3f}", "answered runs pooled over agents and passes"); add("  best agent on NIKA (ReAct)", f"{react_nika:.3f}")
    add("  practitioner networks", f"{kath:.3f}")
    add("Wrong-Device / Wrong-Cause shift vs Correct-Fault", f"{wd - cf:+.3f} / {wc - cf:+.3f}", "All means at 3 dp")
    add("False-Premise penalty", f"{cf - fp:.3f} = {(cf - fp) * 100:.1f} points, to {fp:.3f}")
    for ag in AGENTS:
        c = cell("cex120", "fp", ag)
        add(f"  over-diagnosis on standard False-Premise, {NAME[ag]}", f"{pn(c['overdiag_pct'])}% ({c['n_overdiag']}/{c['n_scored']})", "share of answered runs reporting a fault, pooled")
    t_e1 = cell("base80")["n_timeout"] + cell("cex120")["n_timeout"]
    t_ag = {ag: cell("base80", None, ag)["n_timeout"] + cell("cex120", None, ag)["n_timeout"] for ag in AGENTS}
    add("E1 timeouts", f"{t_e1} of 1,800 = {100 * t_e1 / 1800:.1f}%; " + ", ".join(f"{NAME[a]} {n}" for a, n in t_ag.items()) + f"; {cell('cex120', 'fp')['n_timeout']} on False-Premise")
    ex("E1 caption class means (All column)", f"({cf:.3f}, {fp:.3f}, {wd:.3f}, {wc:.3f})")
    ex("fp drop sentence", f"drops to {fp:.3f}, compared to the Correct-Fault mean of {cf:.3f}")
    c = cell("cex120", "fp", "sade"); ex("over-diagnosis SADE", f"in {pc(c['overdiag_pct'])} of its answered runs ({c['n_overdiag']}/{c['n_scored']})")
    c = cell("cex120", "fp", "react-full"); ex("over-diagnosis ReAct", f"ReAct in {pc(c['overdiag_pct'])} ({c['n_overdiag']}/{c['n_scored']})")
    c = cell("cex120", "fp", "cc-baseline"); ex("over-diagnosis CC-Baseline", f"CC-Baseline in {pc(c['overdiag_pct'])} ({c['n_overdiag']}/{c['n_scored']})")

    # ---- E1: per agent and pass
    lo_cf = min(prange("base80", "fault", ag)[0] for ag in AGENTS)
    rg_cf = max(prange("base80", "fault", ag)[1] - prange("base80", "fault", ag)[0] for ag in AGENTS)
    add("Correct-Fault per agent", ", ".join(f"{NAME[a]} {v('base80', 'fault', a):.3f}" for a in AGENTS) + f"; lowest agent-pass cell {lo_cf:.3f}; largest per-agent pass range {rg_cf:.3f}")
    add("  Correct-Fault timeouts", f"{cell('base80', 'fault')['n_timeout']} of 720 ({cell('base80', 'fault')['timeout_pct']:.0f}%); " + ", ".join(f"{NAME[a]} {cell('base80', 'fault', a)['n_timeout']}" for a in AGENTS))
    cc_wdwc = [r3(cell("cex120", c_, "cc-baseline", p)["outcome"]) for c_ in ("wd", "wc") for p in passes_of("cex120")]
    lo = {ag: min(prange("cex120", c_, ag)[0] for c_ in ("wd", "wc")) for ag in AGENTS}
    hi = {ag: max(prange("cex120", c_, ag)[1] for c_ in ("wd", "wc")) for ag in AGENTS}
    add("Wrong-Device/Cause per agent", f"CC-Baseline at 1.000 in {sum(1 for x in cc_wdwc if x >= 0.9995)} of 6 cells, min {min(cc_wdwc):.3f}; SADE {lo['sade']:.3f}-{hi['sade']:.3f}; ReAct {lo['react-full']:.3f}-{hi['react-full']:.3f}; timeouts {cell('cex120', 'wd')['n_timeout'] + cell('cex120', 'wc')['n_timeout']} of 432")
    add("  largest SDs across passes", f"SADE Wrong-Cause {cell('cex120', 'wc', 'sade', 'mean')['sd_outcome']:.3f}, ReAct Wrong-Device {cell('cex120', 'wd', 'react-full', 'mean')['sd_outcome']:.3f}")
    gap = v("cex120", "fp", "cc-baseline") - v("cex120", "fp", "sade")
    rg_fp = max(prange("cex120", "fp", ag)[1] - prange("cex120", "fp", ag)[0] for ag in AGENTS)
    odp = {ag: "/".join(pn(cell("cex120", "fp", ag, p)["overdiag_pct"]) for p in passes_of("cex120")) for ag in AGENTS}
    add("False-Premise per agent", f"SADE {v('cex120', 'fp', 'sade'):.3f} vs CC-Baseline {v('cex120', 'fp', 'cc-baseline'):.3f}, gap {gap:.3f}; largest per-agent pass range {rg_fp:.3f}; over-diagnosis per pass SADE {odp['sade']}%, ReAct {odp['react-full']}%, CC-Baseline {odp['cc-baseline']}%; CC-Baseline timeouts {cell('cex120', 'fp', 'cc-baseline')['n_timeout']} of 216 ({cell('cex120', 'fp', 'cc-baseline')['timeout_pct']:.0f}%)")
    rng_cls = {cls: prange(arm, cls, "ALL")[1] - prange(arm, cls, "ALL")[0] for _, arm, cls in CLASSES}
    big = max((prange(arm, cls, ag)[1] - prange(arm, cls, ag)[0], ag, cls) for _, arm, cls in CLASSES for ag in AGENTS)
    cells_ = collections.defaultdict(dict)
    for r in RUNS:
        if r["arm"] in ("base80", "cex120") and r["scored"]:
            cells_[(r["arm"], r["scenario_id"], r["agent"])][r["pass"]] = r["outcome"]
    elig = [c_ for c_ in cells_.values() if len(c_) >= 2]
    flips = sum(1 for c_ in elig if max(c_.values()) - min(c_.values()) >= 0.5)
    add("E1 pass stability", f"pooled class means move <= {max(rng_cls.values()):.3f} (Wrong-Cause), {rng_cls['fp']:.3f} on False-Premise; largest single agent cell {big[0]:.3f} ({big[1]}, {big[2]}); {flips} of {len(elig)} (scenario, agent) cells = {round(100 * flips / len(elig))}% move >= .5 between passes")
    ex("CF per agent", f"{v('base80', 'fault', 'sade'):.3f} (SADE), {v('base80', 'fault', 'cc-baseline'):.3f} (CC-Baseline), and {v('base80', 'fault', 'react-full'):.3f} (ReAct)")
    ex("WD All", f"average {wd:.3f} in the outcome score")
    ex("WC All", f"and {wc:.3f} when it incorrectly asserts")

    # ---- E2: personas
    add("novice vs standard", f"{(fp - nov) * 100:.1f} points, to {nov:.3f}"); add("naive vs standard", f"{(nai - fp) * 100:+.1f} points")
    add("tone control (confident - novice)", f"{conf - nov:+.3f}"); add("identifier control (no detail - unsure)", f"{minus - plus:+.3f}")
    random.seed(0)
    ci = {}
    for a, b in (("novice", "novice_confident"), ("naive_unsure", "naive_no_detail")):
        def per_ticket(arm):
            d = collections.defaultdict(list)
            for r in RUNS:
                if r["arm"] == arm and r["scored"]:
                    d[r["core_scenario"]].append(r["outcome"])
            return {k: st.fmean(x) for k, x in d.items()}
        A, B = per_ticket(a), per_ticket(b); ks = sorted(set(A) & set(B))
        diffs = [B[k] - A[k] for k in ks]
        boots = sorted(st.fmean(random.choice(diffs) for _ in ks) for _ in range(5000))
        ci[a] = (boots[125], boots[4875])
        add(f"  95% CI, {b} - {a}", f"paired mean {st.fmean(diffs):+.3f}, CI [{ci[a][0]:+.3f}, {ci[a][1]:+.3f}]", f"{len(ks)} tickets, 5000 resamples, seed 0")
    tools_ = lambda arm, cls=None: st.fmean(r["tool_calls"] for r in RUNS if r["arm"] == arm and (cls is None or r["class"] == cls))
    add("standard fp -> no detail", f"tool calls {tools_('cex120', 'fp'):.1f} -> {tools_('naive_no_detail'):.1f}; timeouts {cell('cex120', 'fp')['timeout_pct']:.1f}% -> {cell('naive_no_detail', 'fp')['timeout_pct']:.1f}%; over-diagnosis {pn(cell('cex120', 'fp')['overdiag_pct'])}% -> {pn(cell('naive_no_detail', 'fp')['overdiag_pct'])}%")
    cc_r = [cell(a, "fp", "cc-baseline")["timeout_pct"] for a in PERSONA_ARMS]
    sr_r = [cell(a, "fp", g)["timeout_pct"] for a in PERSONA_ARMS for g in ("sade", "react-full")]
    add("persona timeout rates", f"CC-Baseline {round(min(cc_r))}-{round(max(cc_r))}%, SADE/ReAct {round(min(sr_r))}-{round(max(sr_r))}%")
    add("CC-Baseline standard fp -> no detail", f"{v('cex120', 'fp', 'cc-baseline'):.3f} -> {v('naive_no_detail', 'fp', 'cc-baseline'):.3f}, timeouts {pn(cell('naive_no_detail', 'fp', 'cc-baseline')['timeout_pct'])}%")
    ex("novice falls", f"the average outcome score falls to {nov:.3f}")
    ex("tone control (bullet)", f"slightly by ${conf - nov:+.3f}$")
    ex("tone control (summary)", f"the outcome rises by ${conf - nov:+.3f}$")
    ex("tone over-d", f"over-diagnosis falls from {pc(cell('novice', 'fp')['overdiag_pct'])} to {pc(cell('novice_confident', 'fp')['overdiag_pct'])}")
    # the "not significant" claim in the paper rests on this interval (recomputed
    # above); the interval itself is not quoted in the text
    ex("cause control", f"drops the average outcome by {nai - plus:.3f}")
    ex("cause over-d", f"over-diagnosis from {pc(cell('naive', 'fp')['overdiag_pct'])} to {pc(cell('naive_unsure', 'fp')['overdiag_pct'])}")
    ex("uncertainty chain outcome", f"from {nai:.3f} to {plus:.3f} and then to {minus:.3f}")
    ex("uncertainty chain over-d", f"raises over-diagnosis from {pc(cell('naive', 'fp')['overdiag_pct'])} to {pc(cell('naive_unsure', 'fp')['overdiag_pct'])} and {pc(cell('naive_no_detail', 'fp')['overdiag_pct'])}")
    ex("identifier control", f"a further {plus - minus:.3f}")
    ex("identifier over-d", f"over-diagnosis to {pc(cell('naive_no_detail', 'fp')['overdiag_pct'])}")

    # ---- E2: per agent and pass
    dm = lambda a, b, g: v(b, "fp", g) - v(a, "fp", g)
    od = lambda arm, g: cell(arm, "fp", g)["overdiag_pct"]
    for lab, a, b in (("standard -> naive", "cex120", "naive"), ("standard -> novice", "cex120", "novice"),
                      ("novice -> confident (tone)", "novice", "novice_confident"),
                      ("naive-unsure -> no detail (identifiers)", "naive_unsure", "naive_no_detail"),
                      ("naive -> naive-unsure (certainty; two campaigns)", "naive", "naive_unsure")):
        parts = []
        for g in AGENTS:
            signs = "".join("+" if cell(b, "fp", g, p)["outcome"] > cell(a, "fp", g, p)["outcome"] else "-" for p in ("p1", "p2"))
            parts.append(f"{NAME[g]} {dm(a, b, g):+.3f} [{signs}] over-d {pn(od(a, g))}->{pn(od(b, g))}% timeouts {cell(a, 'fp', g)['n_timeout']}->{cell(b, 'fp', g)['n_timeout']}")
        add(f"E2 {lab}", "; ".join(parts), "arm-mean delta per agent [sign of the per-pass deltas p1 p2]; over-diag % of answered; timeouts pooled")
    pt = {g: (sum(cell(a, "fp", g)["n_timeout"] for a in PERSONA_ARMS), sum(cell(a, "fp", g)["n_runs"] for a in PERSONA_ARMS)) for g in AGENTS}
    add("E2 persona-arm timeouts by agent", ", ".join(f"{NAME[g]} {n}/{N} ({round(100 * n / N)}%)" for g, (n, N) in pt.items()))
    sd_all = [cell(a, "fp", "ALL", "mean")["sd_outcome"] for a in PERSONA_ARMS]
    sd_big = max((cell(a, "fp", g, "mean")["sd_outcome"], g, a) for a in PERSONA_ARMS for g in AGENTS)
    add("E2 pass-to-pass SD", f"pooled persona rows {min(sd_all):.3f}-{max(sd_all):.3f}; largest agent cell {sd_big[0]:.3f} ({sd_big[1]}, {sd_big[2]})")
    # ---- Section 5.4 (per-agent failure behaviour) and the signatures table
    arms6 = ["cex120"] + PERSONA_ARMS
    ex("CC standard best", f"({v('cex120', 'fp', 'cc-baseline'):.3f} on the Standard row)")
    ex("CC hedging drop", f"drops its outcome by {-dm('naive', 'naive_unsure', 'cc-baseline'):.3f}")
    ex("CC hedging over-d", f"from {pc(od('naive', 'cc-baseline'))} to {pc(od('naive_unsure', 'cc-baseline'))} of answered runs")
    ex("CC no-detail", f"reaches {v('naive_no_detail', 'fp', 'cc-baseline'):.3f} and times out in {pc(cell('naive_no_detail', 'fp', 'cc-baseline')['timeout_pct'])} of runs")
    ex("CC range", f"the {v('cex120', 'fp', 'cc-baseline') - v('naive_no_detail', 'fp', 'cc-baseline'):.3f} range")
    co6 = [od(a, "cc-baseline") for a in arms6]
    ex("signatures CC row", f"Outcome {v('cex120', 'fp', 'cc-baseline'):.3f} to {v('naive_no_detail', 'fp', 'cc-baseline'):.3f} across arms; timeouts {pc(cell('cex120', 'fp', 'cc-baseline')['timeout_pct'])} to {pc(cell('naive_no_detail', 'fp', 'cc-baseline')['timeout_pct'])}; over-d.\\ {pc(min(co6))} to {pc(max(co6))}")
    ex("signatures CC largest change", f"(${dm('naive', 'naive_unsure', 'cc-baseline'):+.3f}$)")
    rt = [cell(a, "fp", "react-full")["timeout_pct"] for a in arms6]
    ex("ReAct timeouts", f"timeouts between {round(min(rt))}\\% and {round(max(rt))}\\% on every arm")
    ex("ReAct novice drop", f"${dm('cex120', 'novice', 'react-full'):+.3f}$ from the Standard to the Novice row")
    ex("signatures ReAct novice", f"${dm('cex120', 'novice', 'react-full'):+.3f}$ on the novice arm")
    s6 = [v(a, "fp", "sade") for a in arms6]
    st6 = [cell(a, "fp", "sade")["timeout_pct"] for a in arms6]
    so6 = [od(a, "sade") for a in arms6]
    ex("SADE range and timeouts", f"within {min(s6):.3f}--{max(s6):.3f} and its timeouts within {round(min(st6))}--{round(max(st6))}\\%")
    ex("SADE over-d band", f"in {round(min(so6))}--{round(max(so6))}\\% of answered runs")
    ex("signatures SADE row", f"Over-d.\\ {round(min(so6))}--{round(max(so6))}\\% on every arm; outcome within {min(s6):.3f}--{max(s6):.3f} on all six arms; timeouts {round(min(st6))}--{round(max(st6))}\\%")

    # ---- fix and reasoning (Table 6 + its paragraph)
    fx = lambda arm, cls, ag="ALL": cell(arm, cls, ag, "mean")["fix"]
    rz = lambda arm, cls, ag="ALL": cell(arm, cls, ag, "mean")["reasoning"]
    n_fix = {cls: cell(arm, cls)["n_scored"] for _, arm, cls in CLASSES}
    add("fix: answered runs with a fix", f"Correct-Fault {n_fix['fault']}, Wrong-Device {n_fix['wd']}, Wrong-Cause {n_fix['wc']}; False-Premise has none")
    add("fix, All", ", ".join(f"{c} {fx(a, c):.3f}" for _, a, c in CLASSES if c != "fp"), "mean over answered runs, mean over passes")
    for ag in AGENTS:
        add(f"  fix, {NAME[ag]}", ", ".join(f"{c} {fx(a, c, ag):.3f}" for _, a, c in CLASSES if c != "fp"))
    for ag in AGENTS + ["ALL"]:
        add(f"reasoning, {NAME[ag]}", ", ".join(f"{c} {rz(a, c, ag):.3f}" for _, a, c in CLASSES)
            + "; personas " + ", ".join(f"{arm} {rz(arm, 'fp', ag):.3f}" for arm in PERSONA_ARMS))
    ex("fix scores (5.1)", f"average fix scores of {fx('base80', 'fault'):.3f}, {fx('cex120', 'wd'):.3f}, and {fx('cex120', 'wc'):.3f}")
    ex("reasoning fp per agent", f"({rz('cex120', 'fp', 'sade'):.3f}, against {rz('cex120', 'fp', 'cc-baseline'):.3f} for CC-Baseline and {rz('cex120', 'fp', 'react-full'):.3f} for ReAct")
    rr = [rz(arm, cls, "react-full") for _, arm, cls in CLASSES]
    ex("ReAct class reasoning range", f"every scenario class ({min(rr):.3f}--{max(rr):.3f}, Table")
    ex("signatures ReAct reasoning", f"lowest reasoning score on every class ({min(rr):.3f}--{max(rr):.3f})")

    # ---- effort and the cost of timeouts (Table 7 + its paragraph)
    tot = {ag: block([r for r in RUNS if r["agent"] == ag]) for ag in AGENTS}
    allb = block(RUNS)
    core = sum(r["cost_usd"] for r in RUNS if r["arm"] in ("base80", "cex120"))
    k0 = lambda x: f"{x / 1000:.0f}"
    add("API spend, all runs", f"US${allb['usd_total']:,.0f} = " + " + ".join(f"{NAME[a]} {tot[a]['usd_total']:,.0f}" for a in AGENTS)
        + f"; core benchmark (E1) per pass US${core / 3:.1f}", "cost_usd summed over result.json; the API's figure for the Claude agents, list-price arithmetic for ReAct")
    for a in AGENTS:
        add(f"  answered run, {NAME[a]}", f"in {tot[a]['in_a']:,.0f} tokens, out {tot[a]['out_a']:,.0f}, US${tot[a]['usd_a']:.3f}, {tot[a]['calls_a']:.1f} tool calls; timeout: in {tot[a]['in_t']:,.0f}, out {tot[a]['out_t']:,.0f}, US${tot[a]['usd_t']:.3f}, {tot[a]['calls_t']:.1f} calls")
    fault_cells = [cell(arm, cls, ag) for _, arm, cls in CLASSES if cls != "fp" for ag in AGENTS]
    add("fault classes, answered runs", f"tool calls {min(c['calls_a'] for c in fault_cells):.1f}-{max(c['calls_a'] for c in fault_cells):.1f}, US${min(c['usd_a'] for c in fault_cells):.2f}-{max(c['usd_a'] for c in fault_cells):.2f}")
    cc_nd, cc_cf = cell("naive_no_detail", "fp", "cc-baseline"), cell("base80", "fault", "cc-baseline")
    add("CC-Baseline answered, no detail vs Correct-Fault", f"in {cc_nd['in_a']:,.0f} vs {cc_cf['in_a']:,.0f} tokens (x{cc_nd['in_a'] / cc_cf['in_a']:.1f}); calls {cc_nd['calls_a']:.1f} vs {cc_cf['calls_a']:.1f}")
    add("timeout vs answered, all runs", f"calls {allb['calls_t']:.1f} vs {allb['calls_a']:.1f}; US${allb['usd_t']:.3f} vs {allb['usd_a']:.3f} (x{allb['usd_t'] / allb['usd_a']:.2f}); {allb['n_timeout']} timeouts = {allb['timeout_pct']:.1f}% of runs, {allb['usd_timeout_pct']:.1f}% of spend")
    add("share of spend on timeouts, by agent", ", ".join(f"{NAME[a]} {tot[a]['usd_timeout_pct']:.1f}%" for a in AGENTS)
        + f"; CC-Baseline on naive-unsure {cell('naive_unsure', 'fp', 'cc-baseline')['usd_timeout_pct']:.1f}%, no detail {cc_nd['usd_timeout_pct']:.1f}%")
    ex("fault classes effort", f"{round(min(c['calls_a'] for c in fault_cells))}--{round(max(c['calls_a'] for c in fault_cells))} tool calls and US\\${min(c['usd_a'] for c in fault_cells):.2f}--{max(c['usd_a'] for c in fault_cells):.2f} per answered run")
    cats = [(arm, cls) for _, arm, cls in CLASSES] + [(arm, "fp") for _, arm, _ in PERSONAS[1:]]
    top = {ag: max((cell(arm, cls, ag)["usd_a"], arm) for arm, cls in cats) for ag in AGENTS}
    add("most expensive answered run, by agent", "; ".join(f"{NAME[a]} US${u:.3f} on {arm}" for a, (u, arm) in top.items()))
    persona_t = sum(cell(arm, "fp")["n_timeout"] for _, arm, _ in PERSONAS[1:])
    add("timeouts on the persona arms", f"{persona_t} of {allb['n_timeout']} ({100 * persona_t / allb['n_timeout']:.0f}%); CC-Baseline naive-unsure {cell('naive_unsure', 'fp', 'cc-baseline')['n_timeout']}, no detail {cc_nd['n_timeout']} of 144; Correct-Fault {cc_cf['n_timeout']} of {cc_cf['n_runs']}")
    ex("persona share of timeouts", f"{persona_t} of them occur on the persona rewrites")
    words = {2: "twice", 3: "three times", 4: "four times", 5: "five times"}
    ex("CC no-detail vs CF", f"read roughly {words[round(cc_nd['in_a'] / cc_cf['in_a'])]} the tokens of its Correct-Fault runs ({k0(cc_nd['in_a'])}k against {k0(cc_cf['in_a'])}k) and make {round(cc_nd['calls_a'])} tool calls against {round(cc_cf['calls_a'])}")
    ex("timeout vs answered", f"makes {round(allb['calls_t'])} tool calls on average, against {round(allb['calls_a'])} for an answered run, and costs {allb['usd_t'] / allb['usd_a']:.1f} times as much (US\\${allb['usd_t']:.2f} against US\\${allb['usd_a']:.2f})")
    big = lambda n: f"{round(n):,}".replace(",", "{,}")
    ex("timeouts, share of runs and spend", f"The {allb['n_timeout']} timeouts are {pc(allb['timeout_pct'])} of the {big(allb['n_runs'])} runs but account for {pc(allb['usd_timeout_pct'])} of the US\\${big(allb['usd_total'])} total API spend")
    ex("persona timeouts by agent", f"times out in {pt['cc-baseline'][0]} of {pt['cc-baseline'][1]} runs ({round(100 * pt['cc-baseline'][0] / pt['cc-baseline'][1])}\\%), against {pt['sade'][0]} for SADE ({round(100 * pt['sade'][0] / pt['sade'][1])}\\%) and {pt['react-full'][0]} for ReAct ({round(100 * pt['react-full'][0] / pt['react-full'][1])}\\%)")
    ex("timeout spend by agent", f"{pc(tot['cc-baseline']['usd_timeout_pct'])} for CC-Baseline, rising to {pn(cell('naive_unsure', 'fp', 'cc-baseline')['usd_timeout_pct'])}--{pn(cc_nd['usd_timeout_pct'])}\\% on the two identifier-control arms, against {pc(tot['sade']['usd_timeout_pct'])} and {pc(tot['react-full']['usd_timeout_pct'])} for the others")
    ex("core pass cost (experiments)", f"US\\${round(core / 3)} of API usage")

    # ---- method: the 0.7/0.3 must/bonus weighting does not drive any comparison
    ws = weight_sensitivity()
    add("outcome weight sensitivity (accuracy weight 0.5-1.0)",
        f"max shift of a class/persona mean vs the 0.7 value {ws['max_shift']:.3f}; class order kept {ws['class_order_kept']}; "
        f"agent ranking on FP/WD/WC kept {ws['agent_rank_kept']}; persona contrasts keep sign {ws['contrast_sign_kept']}",
        "outcome recomputed per run from its must/bonus pass counts, pooled over agents and passes")
    assert ws["class_order_kept"] and ws["agent_rank_kept"] and ws["contrast_sign_kept"], ws
    # the weight-sensitivity computation and assert keep the robustness claim on
    # record in numbers.txt; the paper does not quote the number directly

    # ---- abstract, introduction, experiments, conclusion
    add("abstract / intro / conclusion", f"fp {(cf - fp) * 100:.1f} points; novice {(fp - nov) * 100:.1f} points; identifiers {(plus - minus) * 100:.1f} points (14 rounded); WD/WC {100 * (wd - cf):+.1f}/{100 * (wc - cf):+.1f} points; over-diagnosis up to {pn(od('cex120', 'sade'))}% (SADE) vs {pn(od('cex120', 'cc-baseline'))}% (Claude Code)")
    ex("abstract fp", f"by {(cf - fp) * 100:.1f} points"); ex("abstract novice", f"further {(fp - nov) * 100:.1f} points")
    ex("abstract identifiers", f"costs {(plus - minus) * 100:.1f} points")
    ex("intro WD/WC", f"${100 * (wd - cf):+.1f}$ and ${100 * (wc - cf):+.1f}$")
    ex("intro over-d", f"up to {pc(od('cex120', 'sade'))} of healthy networks (SADE {pc(od('cex120', 'sade'))} versus Claude Code {pc(od('cex120', 'cc-baseline'))})")
    ex("intro phrasing", f"by up to {(fp - nov) * 100:.1f} percentage points")
    ex("conclusion identifiers", f"from a ticket costs {round((plus - minus) * 100)} points")
    ex("experiments cells flipping", f"with {round(100 * flips / len(elig))}\\% of (scenario, agent) cells")
    return rows, expect


def visible_tex(raw):
    """The text as it renders: drop % comments, unwrap \\textcolor{red|green}{...} review markup."""
    raw = "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in raw.splitlines())
    pat = re.compile(r"\\textcolor\{(?:red|green)\}\{")
    while True:
        m = pat.search(raw)
        if not m:
            return raw
        j, depth = m.end(), 1
        while j < len(raw) and depth:
            depth += {"{": 1, "}": -1}.get(raw[j], 0)
            j += 1
        raw = raw[:m.start()] + raw[m.end():j - 1] + raw[j:]


def check_prose(paper: Path, expect):
    tex = ""
    for f in sorted(paper.rglob("*.tex")):
        tex += re.sub(r"\s+", " ", visible_tex(f.read_text(encoding="utf-8", errors="replace"))) + "\n"
    missing = [(lab, s) for lab, s in expect if re.sub(r"\s+", " ", s) not in tex]
    print(f"  {len(expect)} prose numbers checked verbatim:", "ALL PRESENT" if not missing else f"{len(missing)} MISSING")
    for lab, s in missing:
        print(f"    {lab}: {s!r}")
    return not missing


# ============================================================ 6. optional CSV export
def write_outputs(out: Path, rows):
    """metrics/: the sheet, both tables, the figure, cells.csv, runs.csv, README.md."""
    out.mkdir(parents=True, exist_ok=True)
    w = max(len(r[0]) for r in rows)
    (out / "numbers.txt").write_text(
        "every number the paper quotes, recomputed from final_results/ (label | value | how)\n\n"
        + "\n".join(f"{lab:<{w}}  {val}" + (f"   [{how}]" if how else "") for lab, val, how in rows) + "\n",
        encoding="utf-8")
    for key, fn in TABLES.items():
        (out / f"table_{key}.tex").write_bytes(fn().encode("utf-8"))
    figure(out)
    keys = ["arm", "pass", "topology", "family", "scenario_id", "core_scenario", "class", "agent", "status",
            "scored", "is_anomaly", "outcome", "fix", "reasoning", "tool_calls", "in_tokens", "fresh_tokens",
            "cache_read_tokens", "cache_write_tokens", "out_tokens", "cost_usd", "path"]
    with open(out / "runs.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in RUNS:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in keys})
    cols = ["arm", "class", "agent", "pass", "n_runs", "n_scored", "n_timeout", "timeout_pct", "n_overdiag",
            "overdiag_pct", "outcome", "sd_outcome", "fix", "sd_fix", "reasoning", "sd_reasoning",
            "calls_a", "in_a", "out_a", "usd_a", "calls_t", "in_t", "out_t", "usd_t", "usd_total", "usd_timeout_pct"]
    fmt = lambda x: "" if x is None else (f"{x:.4f}" if isinstance(x, float) else x)
    with open(out / "cells.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for arm in ARMS:
            classes = sorted({r["class"] for r in RUNS if r["arm"] == arm}, key=["fault", "fp", "wd", "wc"].index)
            for cls in ([None] + classes if len(classes) > 1 else classes):
                for ag in AGENTS + ["ALL"]:
                    for pas in passes_of(arm) + ["mean", "pooled"]:
                        c = cell(arm, cls, ag, pas)
                        w.writerow({"arm": arm, "class": cls or "all", "agent": ag, "pass": pas,
                                    **{k: fmt(c.get(k)) for k in cols[4:]}})
    (out / "README.md").write_text(
        "# metrics\n\nWritten by `python tools/compute_metrics.py` from `final_results/`.\n\n"
        "* `numbers.txt`: every number the paper quotes, with the rule that produced it.\n"
        "* `table_e1.tex`, `table_e2.tex`, `table_e3.tex`, `table_e4.tex`: the paper's results tables (LaTeX, "
        "booktabs): outcome by class, outcome by persona, fix and reasoning, effort (tool calls, tokens, API cost "
        "per run for the runs that answered and for the timeouts, and the share of the spend on timeouts).\n"
        "* `results.pdf` / `results.png`: the paper's results figure.\n"
        "* `runs.csv`: one row per run (3,960): arm, pass, topology, scenario, class (fault / fp / wd / wc), agent, "
        "status, scored (True = the agent submitted a diagnosis; False = timeout), is_anomaly (the diagnosis "
        "reports a fault), outcome / fix / reasoning scores, tool calls, tokens (in_tokens = everything the model "
        "read = fresh + cache_read + cache_write; out_tokens = written), cost_usd (the API's figure for the Claude "
        "agents, list-price arithmetic for ReAct), path in final_results/.\n"
        "* `cells.csv`: arm x class x agent x pass. `pass` = p1..p3 (one pass), `mean` (mean over passes; "
        "`sd_*` = SD across passes), `pooled` (all runs together). Scores are means over the runs that answered; "
        "`n_timeout` / `timeout_pct` count the runs that did not; `overdiag_pct` = share of answered runs that "
        "report a fault (meaningful on healthy networks, class fp). `calls_a` / `in_a` / `out_a` / `usd_a` = mean "
        "tool calls, input tokens, output tokens and cost per run over the runs that answered, `*_t` the same over "
        "the timeouts; `usd_total` = the cell's spend, `usd_timeout_pct` = the share of it that went to timeouts. "
        "`class = all` pools the classes of an arm.\n\n"
        "Scores are means over the runs that answered; a run that used up its turn budget without submitting a "
        "diagnosis is a timeout, counted next to every score and not scored. Per-arm values are means over "
        "passes; `sd_*` is the SD across passes; `pooled` puts all runs of the arm together.\n", encoding="utf-8")


# ============================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="report only; do not write into the paper")
    ap.add_argument("--final-results", type=Path, default=FINAL)
    ap.add_argument("--paper", type=Path, default=PAPER, help="directory with main.tex, sections/, Figures/")
    ap.add_argument("--out", type=Path, default=METRICS, help="output directory (default metrics/)")
    a = ap.parse_args()
    global RUNS
    ok = True

    print(f"reading {a.final_results} ...")
    RUNS, issues = load(a.final_results)
    n_sc = sum(r["scored"] for r in RUNS)
    print(f"  {len(RUNS)} runs = {n_sc} answered + {len(RUNS) - n_sc} timeouts; "
          + ", ".join(f"{arm} {sum(r['arm'] == arm for r in RUNS)}" for arm in ARMS))
    print("  integrity:", "no issues" if not issues else "; ".join(f"{k}: {n}" for k, n in issues.items()))
    ok &= not issues
    if a.check:
        print("  (--check: nothing is written into the paper)")

    print("\nnumbers (label | value | how)")
    rows, expect = sheet()
    w = max(len(r[0]) for r in rows)
    for lab, val, how in rows:
        print(f"  {lab:<{w}}  {val}" + (f"   [{how}]" if how else ""))

    write_outputs(a.out, rows)
    print(f"\nwrote {a.out}: numbers.txt, table_e1.tex, table_e2.tex, table_e3.tex, table_e4.tex, results.pdf, cells.csv, runs.csv")

    results_tex = a.paper / "sections" / "results.tex"
    if not results_tex.exists():
        print(f"\npaper not found at {a.paper} (no sections/results.tex): tables and figure are in "
              f"{a.out}; the LaTeX write/check is skipped")
        print("\nRESULT:", "numbers reproduced from final_results" if ok else "INTEGRITY ISSUES, see above")
        return 0 if ok else 1
    if not a.check:
        write_tables(results_tex)
        print(f"wrote table e4 into {results_tex} (e1/e2/e3 are hand-maintained)")
        print("wrote figure" if figure(a.paper / "Figures") else "figure skipped", a.paper / "Figures" / "results.pdf")

    print("\npaper check")
    ok &= check_tables(results_tex)
    ok &= check_prose(a.paper, expect)
    print("\nRESULT:", "everything in the paper matches final_results" if ok else "MISMATCH, see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
