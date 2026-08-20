<div align="center">
<h1>FaulT-Bench: Evaluating LLM Troubleshooting Agents on Live Networks with Realistic Trouble Tickets</h1>

[Overview](#overview) ·
[What FaulT-Bench adds](#what-fault-bench-adds) ·
[Repo layout](#repo-layout) ·
[Verify the paper's numbers](#verify-the-papers-numbers) ·
[Installation](#installation) ·
[Running the benchmark](#running-the-benchmark) ·
[The dataset](#the-dataset) ·
[The released runs](#the-released-runs) ·
[Topology sources](#topology-sources) ·
[Acknowledgements](#acknowledgements) ·
[License](#license)

</div>

> **Works together with the SADE-NetworkAgent repository.**
> The three evaluated agents (SADE, CC-Baseline, ReAct + GPT-5) and the Kathará
> MCP tool layer come from that repository, cloned as a sibling directory
> (see [Acknowledgements](#acknowledgements)).
> This repository contains everything else: the dataset, the live-network
> labs, the evaluation pipeline, the 3,960 frozen runs behind the paper, and
> the tools that recompute every reported number from them.

## Overview

FaulT-Bench evaluates LLM troubleshooting agents on **live emulated networks**
(Kathará/FRR) with **realistic trouble tickets** — including tickets that are
wrong: reports of faults that do not exist (false premise), tickets that blame
the wrong device, tickets that assert the wrong cause, and rewrites of the
same false report in five reporter voices. Existing benchmarks always inject
a real fault, leaving an agent that verifies the ticket indistinguishable
from an agent that blindly trusts it; FaulT-Bench separates them.

One run = one (scenario, agent): boot the scenario's Kathará lab, inject the
fault if there is one and verify it holds, run the agent, judge the free-text
diagnosis with an LLM judge on three separate scores (outcome / fix /
reasoning), archive everything, tear down.

## What FaulT-Bench adds

1. **A 560-scenario dataset** (`dataset/`, CC BY 4.0): 200 core scenarios —
   80 correct-fault, 72 false-premise, 24 wrong-device, 24 wrong-cause —
   across 8 topologies, plus 360 persona rewrites of the 72 false-premise
   tickets (five reporter voices over identical, verified-healthy networks).
2. **Eight live topologies**: five Kathará translations of publicly published
   practitioner networks (`topology_kathara/Kath1`–`Kath5`) and the three
   NIKA reference topologies (built in code by the NIKA package).
3. **An evaluation pipeline** (`faultbench/`): deploy → inject → verify the
   fault holds → agent → trace-aware three-score LLM judge → archive, with a
   CLI for single runs, manifest benchmarks, and dataset-layer sweeps.
   Timeouts are counted and reported separately, never folded into scores.
4. **The 3,960 runs behind every number in the paper** (`final_results/`):
   full diagnosis, per-axis judge breakdown, and investigation trace for each.
5. **Two verification tools** (`tools/`) that recompute every reported table
   and number from the raw runs, with no Docker and no API keys.

## Repo layout

```
dataset/            560 scenario files + dataset_index.csv (see "The dataset")
topology_kathara/   the five Kathará labs Kath1-Kath5 (lab.conf, *.startup)
faultbench/         the evaluation pipeline
  fb.cmd, fb.sh       the launcher: one command per run (see "Running")
  src/faultbench/     CLI (run / sweep / bench / list), lab lifecycle,
                      scenario parser, agent runners, LLM judge
  benchmarks/         one manifest CSV per experiment category
  runs/               output tree (generated, gitignored)
final_results/      the 3,960 frozen runs + MANIFEST.csv (see "The released runs")
tools/              compute_metrics.py + dataset_stats.py (see below)
```

## Verify the paper's numbers

No Docker, no API keys — two commands over the shipped data:

```bash
python tools/compute_metrics.py --check   # every reported number, from final_results/
python tools/dataset_stats.py  --check    # the ticket statistics, from dataset/
```

`compute_metrics.py` reads `final_results/MANIFEST.csv` and every run's
`result.json`, recomputes all metrics (tables, figure, per-pass values and
standard deviations into `metrics/`), and re-verifies the stored data against
itself — every scored run's outcome is re-derived from its own per-axis judge
breakdown. `dataset_stats.py` recomputes the ticket statistics and the
persona-design invariants from the scenario files alone. Both exit non-zero
with `--check` if anything disagrees. Metric definitions are in the paper;
the score implementation is the judge,
[`faultbench/src/faultbench/scoring/llm_judge.py`](faultbench/src/faultbench/scoring/llm_judge.py).
(The figure needs matplotlib and is skipped without it; every number needs
only the standard library.)

> **Windows note:** some run paths inside `final_results/` are long. Clone to
> a short directory (e.g. `D:\Fault`) or enable long paths first:
> `git config --global core.longpaths true`.

## Installation

Needed only for **live benchmark runs** (booting labs and calling paid LLM
APIs). Verifying the shipped data needs none of this.

**Requirements**

- **OS:** Linux, or Windows 11 with Docker Desktop (the released results were
  produced on Windows 11).
- **Docker** running, Engine **28.x** — 29.x hangs Kathará 3.8.3 deploys
  (containers stick in `Created` and the deploy never returns).
- **Kathará 3.x** — CLI + Python API.
- **Python 3.12+** and **uv**.
- **SADE-NetworkAgent** cloned as a sibling directory — it provides the
  agents and the Kathara MCP tool layer (`get_reachability`, `exec_shell`,
  ...) used by **all** agents.
- **Anthropic + OpenAI API keys** (the latter also drives the `gpt-5-mini`
  judge).

**Setup.** Four pieces sit side by side under one root; paths auto-resolve
(the harness walks up to the directory holding both `dataset/` and
`faultbench/`; set `FAULTBENCH_ROOT` only to override):

```
<root>/
  dataset/  topology_kathara/  SADE-NetworkAgent/  faultbench/
```

```powershell
# from <root> — reuse SADE's venv, which already has Kathara + the agent SDKs
cd SADE-NetworkAgent
uv sync
uv pip install -e ../faultbench

# verify both halves
.\.venv\Scripts\python.exe -c "import faultbench; print('harness OK')"
.\.venv\Scripts\python.exe -c "from Kathara.manager.Kathara import Kathara; print('kathara OK')"
```

The exact package versions that produced `final_results/` are pinned in
[`requirements.txt`](requirements.txt).

Kathará boots each device from a Docker image; three are stock, one is custom
(Kath1's four VRRP switches only) and builds in seconds:

```powershell
docker pull kathara/frr ; docker pull kathara/base ; docker pull kathara/dhcp
docker build -t frr-with-vrrp-relay `
  -f topology_kathara\Kath1\Dockerfile.relay topology_kathara\Kath1
```

Create **`SADE-NetworkAgent/.env`** — auto-loaded by the harness:

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
BASE_DIR=/absolute/path/to/SADE-NetworkAgent   # the NIKA topologies need it
```

**Hermetic agent context.** A run is driven by the benchmark's inputs alone:
the system prompt, the ticket, and the lab tools. Because Claude Code also
reads `CLAUDE.md` files and project memory from the directory it starts in,
the harness runs CC-Baseline from a clean working directory outside any git
repository (`FAULTBENCH_CC_CWD` overrides the default) and refuses to start
either Claude agent if such files would reach it — the error names the file
to move. SADE's own routing `CLAUDE.md` inside the SADE repo is part of the
agent and is the one designed exception.

## Running the benchmark

Every command below is run from `faultbench/` through the `fb` launcher —
`fb.cmd` (Windows) or `fb.sh` (Linux/macOS) — which finds the sibling SADE
venv and sets `PYTHONUTF8=1` for you (without it, a Unicode banner can crash
a sweep under a pipe while still reporting exit 0):

```powershell
cd ..\faultbench              # the Setup steps left you in SADE-NetworkAgent\
.\fb list --scope base        # smoke test: prints the 80 base scenario ids
```

On Linux/macOS, write `./fb.sh` wherever the examples below say `.\fb`, and
flip the `\` path separators to `/`.

### One run (~9 min per agent)

`--agents` takes a comma-separated list; omitting it runs all three:

```powershell
.\fb bench --scenarios kath3-router0-ripd-down --agents cc-baseline        # one agent
.\fb bench --scenarios kath3-router0-ripd-down --agents sade,react-full    # any subset
.\fb bench --scenarios kath3-router0-ripd-down                             # all three agents
```

Each run deploys the lab, injects and verifies the fault, runs the agent,
judges the diagnosis, archives, and tears down. With several agents, each
gets its own freshly deployed lab, one after another, and the closing
summary table shows them side by side — the three-agent comparison the
paper reports. Add `--dry-run` to any `bench` command to preview its plan
without executing anything. (There is also a lower-level `run` subcommand taking a scenario
*file path* and a single agent; `bench --scenarios <id>` is usually more
convenient.)

### One category — a manifest times your agents

Each CSV in `faultbench/benchmarks/` lists the tickets of one experiment
category:

| manifest | tickets | what they are |
|---|---:|---|
| `base80.csv` | 80 | a real fault is present |
| `cex120.csv` | 120 | 72 false-premise + 24 wrong-device + 24 wrong-cause |
| `novice72.csv` | 72 | false-premise, vague, names no cause |
| `naive72.csv` | 72 | false-premise, states a definite wrong cause |
| `novice_confident72.csv` | 72 | the novice tickets rewritten demanding, content fixed |
| `naive_unsure72.csv` | 72 | naive with the certainty withdrawn, identifiers kept |
| `naive_no_detail72.csv` | 72 | the same, with every identifier deleted |

```powershell
.\fb bench --manifest benchmarks\cex120.csv --runs-root runs\cex_p1
```

`--agents` defaults to all three (`sade,cc-baseline,react-full`); name fewer
to restrict. Run count = tickets × agents (here 120 × 3 = 360). `--runs-root`
names the output tree for this category; without it, results land in `runs\`
directly.

### Repeating a category — passes

A **pass** is one full trip through a category. To repeat one, run the *same*
command with a **new `--runs-root`** — the root is the pass:

```powershell
.\fb bench --manifest benchmarks\base80.csv --runs-root runs\base_p1
.\fb bench --manifest benchmarks\base80.csv --runs-root runs\base_p2
.\fb bench --manifest benchmarks\base80.csv --runs-root runs\base_p3
```

Never reuse a root to repeat runs: a root's `summary.csv` accumulates every
run appended to it; repeats inside one root therefore pile up in the same table.
One fresh root per pass keeps the passes cleanly separate; aggregate across
them afterwards.

### The paper's full experiment

* **E1** — `base80.csv` and `cex120.csv`, 3 agents, 3 passes → 1,800 runs.
* **E2** — the five persona manifests, 3 agents, 2 passes → 2,160 runs.

The whole experiment is the one command above, repeated per (category, pass):

```powershell
.\fb bench --manifest benchmarks\<category>.csv --runs-root runs\<category>_p<N>
```

with `<category>` = `base80`, `cex120` at `<N>` = 1–3 (E1), and the five
persona manifests at `<N>` = 1–2 (E2). A run takes ~9 minutes including lab
deploy and teardown; reproduce one category at a time rather than in one
sitting, and compare against the frozen runs in `final_results/`.

### Re-running specific cells

Put just those scenarios in a manifest and bench it with just the affected
agent into a fresh root. `list` writes a manifest you can prune (delete or
`#`-prefix rows); a manifest is only a CSV with a `scenario` column of ids
or paths — writing one by hand is also fine:

```powershell
.\fb list --scope cex --out benchmarks\subset.csv
.\fb bench --manifest benchmarks\subset.csv --agents react-full --runs-root runs\redo
```

### Sweeping a dataset layer without a manifest

`sweep --scope` discovers the scenarios from `dataset/` itself and therefore
can never drift out of sync with the dataset; `--topology` restricts to one
lab:

```powershell
.\fb sweep --scope naive --topology Kath2 --agents sade
```

`--scope` takes `base | cex | novice | naive | novice_confident |
naive_unsure | naive_no_detail`, or `all` for `base` + `cex` together (the
200 core scenarios).

### Long jobs — launch detached

A full category is a multi-hour job, and a run started normally dies with
its terminal. This starts it in a hidden background shell instead, letting
the terminal be closed; the log ends with a `DONE` summary when the job
finishes:

```powershell
mkdir -Force runs | Out-Null
Start-Process powershell -WindowStyle Hidden -WorkingDirectory $PWD -ArgumentList "-Command", `
  ".\fb bench --manifest benchmarks\cex120.csv --runs-root runs\cex_p1 *> runs\cex_p1.log"
```

One more Windows note, already handled in code: manifest CSVs written by
PowerShell 5.1 carry a UTF-8 BOM; the loader reads `utf-8-sig`.

### Output

```
runs/<Topology>/summary.csv                     one row per run
runs/<Topology>/<class>__<scenario-stem>/<agent>/<run_id>/
    result.json    diagnosis + per-axis judge breakdown  (authoritative)
    trace.log      the agent's reasoning trace (chunks truncated to 800 chars;
                   never parse a verdict from it, read result.json)
```

(The frozen runs in `final_results/` are the same records re-filed by
(category, pass) for analysis; fresh runs always use the `runs/` layout
above.)

`summary.csv` `status` values:

| status | meaning | what to do |
|---|---|---|
| `scored` | valid verdict within the turn budget, judged | — |
| `no_submission` | turn budget exhausted, no valid `=== DIAGNOSIS ===` block | keep; a timeout, reported next to the scores, never scored |
| `no_output` | turn budget exhausted, no final text at all | keep; the same event as `no_submission` |
| `judge_failed` | the agent answered but the judge's API call failed | re-judge from the saved diagnosis; do **not** re-run the agent |
| `agent_error` | the agent's SDK errored (key at its cap, auth, transport) before any verdict | re-run the cell; never score it |
| `deploy_incomplete`, `injection_failed`, `pipeline_error` | infrastructure; the agent never ran or the lab was wrong | re-run |

A run that ends with no verdict is not necessarily a failure: an agent that
probed and never concluded is a real result (a timeout). A run whose agent
phase died with ~zero tool calls is an infrastructure failure — re-run it.
Leftover containers after a killed run:
`docker ps -a --filter "name=kathara" -q | %{ docker rm -f $_ }`.

### Adding a scenario

Write a `.txt` under `dataset/<topology>_Q/{base,cex}/` in the section format
below — the lab comes from the file's `Topology:` line — then run it like any
other scenario. The pipeline itself verifies the injection on every run: a
fault that does not take and hold fails the `[POST-INJECT-CHECK]` and the run
is discarded before the agent starts.

## The dataset

560 free-text scenario files. Every topology folder has the same seven
layers; each `.txt` is one scenario, and the pipeline reads the target
topology from the file's `Topology:` line:

```
dataset/
  kath1_Q/ ... kath5_Q/  nika_bgp_Q/ nika_clos_Q/ nika_ospf_Q/
    base/                injection faults (a real fault is present)      10 each
    cex/                 counter-examples                                15 each
    novice/ naive/ novice_confident/ naive_unsure/ naive_no_detail/      9 each
  dataset_index.csv      index of the 200 core scenarios
```

**The two kinds of core scenario.** `base` (80): a genuine fault is injected
— a daemon killed, a link downed, a route removed — and the agent should find
it. `cex` (120): the discriminating contribution — per topology, 9
**false-premise** tickets (the network is healthy; the correct answer is
"nothing is wrong"), 3 **wrong-device** (`*-misdirected`: a real fault, the
ticket blames the wrong device), and 3 **wrong-cause** (`*-misdescribed`: a
real fault, the ticket mis-states the mechanism).

**The persona layers** rewrite the same 72 false-premise tickets while
keeping everything else (network, ground truth, rubric) identical; every
comparison is therefore a within-case paired contrast: `novice` (vague, no technical
vocabulary, no cause named) and `naive` (confidently wrong technical framing,
names a cause) are the naturalistic voices; `novice_confident` is the tone
control (novice content, demanding instead of apologetic); `naive_unsure`
and `naive_no_detail` are the identifier controls (certainty withdrawn with
identifiers kept, then every identifier deleted). Every rewrite carries
`Ticket style:` and `Variant-of: <parent-scenario-id>` headers, and
`novice_confident` additionally names its novice parent via
`Confidence-flip-of:`. Two design invariants are machine-checked:
`novice_confident` keeps the novice identifiers exactly, and
`naive_no_detail` contains no identifier at all
(`tools/dataset_stats.py --check` verifies both).

**Scenario file format** — section-delimited plain text; open any file to
read it:

```
[PROMPT-TO-AGENT]      the ticket shown to the AI (the only thing it sees)
[INJECTION]            the fault applied to the live lab (none for false-premise)
[POST-INJECT-CHECK]    how the harness confirms the fault took and HOLDS
[GROUND-TRUTH]         the correct diagnosis, in prose (judge context)
[FIX]                  the canonical remediation, lab-verified (fix metric;
                       only in scenarios that contain a fault)
[DIAGNOSTIC-PROCESS]   an example diagnostic route (informational only)
[SCORING-AXES]         the per-scenario grading rubric (outcome metric)
[NIKA-LABEL]           relation to NIKA's 54-label fault taxonomy
```

`[DIAGNOSTIC-PROCESS]` is a worked example of ONE valid route, kept for human
review — the reasoning judge is route-agnostic and never reads it.
`[NIKA-LABEL]` only feeds `dataset_index.csv`: the base faults map to NIKA's
closed 54-label taxonomy where a label exists, while the counter-examples are
outside any closed taxonomy by design (a false-premise ticket has no fault to
label) — which is why the benchmark judges free text rather than matching
labels. `dataset_index.csv` lists all 200 core scenarios with their topology,
layer, class, fault family, and NIKA-label columns (100 in-taxonomy,
100 novel).

## The released runs

`final_results/` holds the 3,960 runs: the 200 core scenarios at three passes
(E1) and the five persona categories at two passes (E2), each run by all
three agents. One directory per run:

```
<category>/<topology>/<scenario_id>/<agent>/<pass>/
    result.json    the authoritative record: scenario blocks, the exact prompt
                   the agent received, agent phase with tokens and cost, the
                   full per-axis judge breakdown
    trace.log      the agent's investigation trace
```

`MANIFEST.csv` is the registry — one row per run, keyed `arm, pass, topology,
scenario_id, agent` (the `arm` column holds the category name), with
`model`, `status`, `submitted`, `is_anomaly`, the
three scores, `tool_calls`, the four token columns, `cost_usd`, and
`wall_seconds`. It is what `tools/compute_metrics.py` reads, and `--check`
re-verifies every row against the run's own `result.json`.

| category (`arm`) | runs | scored | no_output | no_submission |
|---|---:|---:|---:|---:|
| base80 | 720 | 703 | 3 | 14 |
| cex120 | 1080 | 1047 | 9 | 24 |
| novice | 432 | 385 | 9 | 38 |
| novice_confident | 432 | 377 | 16 | 39 |
| naive | 432 | 394 | 8 | 30 |
| naive_unsure | 432 | 349 | 10 | 73 |
| naive_no_detail | 432 | 339 | 12 | 81 |
| **total** | **3960** | | | |

`no_submission` and `no_output` are both **timeouts** — the 20-turn budget
ran out without a valid verdict — differing only in whether the backend
returned stray final text. Timeouts are agent behaviour, not infrastructure
failure: they are kept, counted, and reported next to every score. Runs whose
agent phase died on an infrastructure error are not in this set; every cell
here ran to the agent's own conclusion. (Three SADE runs wrote the structured
verdict with markdown decoration the parser cannot read; they are kept as
scored — a structured verdict the harness cannot parse is a wrong answer —
and are not counted as over-diagnosis.)

## Topology sources

Kath1–Kath5 are our Kathará/FRR translations of publicly published lab
designs, used with attribution; the original source configuration files are
not redistributed here.

* **Kath1** — Jeremy's IT Lab, *CCNA Mega Lab* (free CCNA 200-301 course), https://courses.jeremysitlab.com/
* **Kath2** — katejay, *College-Network*, https://github.com/katejay/College-Network
* **Kath3** — imsiddhant, *Computer Networking Project 1*, https://github.com/imsiddhant/Computer-Networking-Project-1
* **Kath4** — imsiddhant, *Computer Networking Project 2*, https://github.com/imsiddhant/Computer-Networking-Project-2
* **Kath5** — Internetworks, *OSPF CCNA CCNP Lab* (The CCIE lab blog), https://ccie.internetworks.in/2024/10/ospf-ccna-ccnp-lab.html
* **nika_ospf / nika_bgp / nika_clos** — the NIKA benchmark's reference topologies.

## Acknowledgements

The three evaluated agents and the Kathara MCP tool layer come from
**SADE-NetworkAgent**, which in turn builds on
[NIKA](https://github.com/sands-lab/nika) — obtain SADE-NetworkAgent from its
authors and place it as a sibling directory. The NIKA package also provides
the three reference topologies used alongside our five translated labs.

## License

Code is MIT ([LICENSE](LICENSE)). The scenario dataset (`dataset/`) is
CC BY 4.0 ([dataset/LICENSE](dataset/LICENSE)).
