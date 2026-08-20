"""CLI entry point: `python -m faultbench ...`.

This module is the command-line layer only. Scenario discovery/resolution
lives in `scenario.py`, .env loading in `envfile.py`, and the run lifecycle in
`orchestrator.py`.
"""
from __future__ import annotations

import csv
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .envfile import load_env
from .orchestrator import AGENT_RUNNERS, run_one
from .paths import ENV_FILE
from .scenario import LAYERS, discover_scenarios, load_manifest, resolve_scenario

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

_SCOPE_HELP = " | ".join(LAYERS) + " | all (= base + cex, the 200-scenario dataset)"


@app.callback()
def _main(ctx: typer.Context) -> None:
    """Load API keys from .env before any subcommand runs."""
    if not load_env():
        console.print(f"[yellow]WARNING[/yellow]: {ENV_FILE} not found; "
                      "relying on already-set env vars")


def _runs_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "runs"


def _check_agents(agents: str) -> list[str]:
    agent_list = [a.strip() for a in agents.split(",") if a.strip()]
    bad = [a for a in agent_list if a not in AGENT_RUNNERS]
    if bad:
        typer.echo(f"Unknown agents: {bad}. Choose from {list(AGENT_RUNNERS)}")
        raise typer.Exit(2)
    return agent_list


def _discover_or_exit(scope: str, topology: str | None) -> list[Path]:
    try:
        scenarios = discover_scenarios(scope, topology)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(2)
    if not scenarios:
        typer.echo("No scenarios found.")
        raise typer.Exit(1)
    return scenarios


@app.command()
def run(
    scenario: Path = typer.Option(..., exists=True, dir_okay=False,
                                  help="Path to the scenario .txt file"),
    agent: str = typer.Option(..., help=f"One of: {', '.join(AGENT_RUNNERS)}"),
    runs_root: Path = typer.Option(
        None, "--runs-root",
        help="Results root (default: package runs/). Output is filed under "
             "<runs-root>/<Topology>/."),
    keep_lab: bool = typer.Option(False, help="Don't kathara lclean after"),
) -> None:
    """Run ONE scenario on ONE agent. Appends to <Topology>/summary.csv + per-run folder."""
    if agent not in AGENT_RUNNERS:
        typer.echo(f"Unknown agent '{agent}'. Choose from {list(AGENT_RUNNERS)}")
        raise typer.Exit(2)
    runs_root = runs_root or _runs_dir()
    console.print(f"[bold]Running[/bold] {scenario.name} on {agent}")
    row = run_one(scenario, agent, runs_root, keep_lab=keep_lab, run_group="run")
    _print_row(row)
    console.print(f"[dim]-> {runs_root / row['topology'] / 'summary.csv'}[/dim]")


@app.command()
def sweep(
    agents: str = typer.Option("sade,cc-baseline,react-full",
                               help="Comma-separated list of agents"),
    topology: str = typer.Option(None,
                                 help="Restrict to one topology (e.g. Kath1)"),
    scope: str = typer.Option("cex", help=f"Which set to run: {_SCOPE_HELP}"),
    runs_root: Path = typer.Option(None, "--runs-root"),
    keep_lab: bool = typer.Option(False),
) -> None:
    """Run a whole dataset layer on the chosen agents (discovery globs the dataset)."""
    agent_list = _check_agents(agents)
    runs_root = runs_root or _runs_dir()
    scenarios = _discover_or_exit(scope, topology)

    console.print(f"[bold]Sweep[/bold] {len(scenarios)} scenarios "
                  f"x {len(agent_list)} agents = "
                  f"{len(scenarios)*len(agent_list)} runs -> {runs_root}")

    for sc_path in scenarios:
        for agent in agent_list:
            console.rule(f"{sc_path.name} on {agent}")
            row = run_one(sc_path, agent, runs_root, keep_lab=keep_lab,
                          run_group="sweep")
            _print_row(row)


@app.command()
def bench(
    manifest: Path = typer.Option(
        None, dir_okay=False,
        help="Manifest of scenarios to run: CSV with a 'scenario' column, "
             "or a plain line-list (# comments allowed)."),
    scenarios: str = typer.Option(
        None, help="Inline comma-separated scenario ids (instead of --manifest)."),
    agents: str = typer.Option(
        "sade,cc-baseline,react-full",
        help="Comma-separated agents to run on every scenario."),
    runs_root: Path = typer.Option(None, "--runs-root"),
    keep_lab: bool = typer.Option(False),
    converge_wait: int = typer.Option(
        90, "--converge-wait",
        help="Seconds to SETTLE after all containers deploy (lets the lab "
             "converge before injection). No functional pre-check is run."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Resolve + print the run plan, don't execute."),
) -> None:
    """Run a CURATED set of scenarios x agents (a named benchmark).

    Unlike `sweep` (every scenario of a layer), `bench` runs exactly the
    scenarios you list -- from a CSV manifest or an inline id list.

        python -m faultbench bench --manifest benchmarks/cex120.csv
        python -m faultbench bench --scenarios kath2-router0-ripd-down --agents sade
    """
    agent_list = _check_agents(agents)

    if bool(manifest) == bool(scenarios):
        typer.echo("Provide exactly one of --manifest or --scenarios.")
        raise typer.Exit(2)
    if manifest and not manifest.exists():
        typer.echo(f"Manifest not found: {manifest}")
        raise typer.Exit(2)

    refs = (load_manifest(manifest) if manifest
            else [s.strip() for s in scenarios.split(",") if s.strip()])
    if not refs:
        typer.echo("No scenarios to run.")
        raise typer.Exit(1)

    try:
        sc_paths = [resolve_scenario(r) for r in refs]
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Manifest error: {exc}")
        raise typer.Exit(2)

    tag = manifest.stem if manifest else "adhoc"
    runs_root = runs_root or _runs_dir()

    total = len(sc_paths) * len(agent_list)
    console.print(f"[bold]Bench[/bold] {len(sc_paths)} scenarios x "
                  f"{len(agent_list)} agents = {total} runs -> {runs_root} "
                  f"(group={tag})")

    if dry_run:
        plan = Table(title="Run plan (dry-run)")
        plan.add_column("#", style="dim"); plan.add_column("scenario")
        plan.add_column("agents")
        for i, p in enumerate(sc_paths, 1):
            plan.add_row(str(i), p.stem, ", ".join(agent_list))
        console.print(plan)
        return

    rows: list[dict] = []
    for sc_path in sc_paths:
        for agent in agent_list:
            console.rule(f"{sc_path.stem} on {agent}")
            row = run_one(sc_path, agent, runs_root, keep_lab=keep_lab,
                          converge_wait=converge_wait, run_group=tag)
            rows.append(row)
            _print_row(row)

    _print_summary(rows, agent_list)
    console.print("\n[bold green]DONE[/bold green] -> per-topology summaries:")
    for tp in sorted({r["topology"] for r in rows}):
        console.print(f"  {runs_root / tp / 'summary.csv'}")


@app.command(name="list")
def list_scenarios(
    scope: str = typer.Option("cex", help=_SCOPE_HELP),
    topology: str = typer.Option(None, help="restrict to one topology (e.g. Kath2)"),
    out: Path = typer.Option(
        None, "--out",
        help="write a bench manifest CSV here (a 'scenario' column); "
             "otherwise just print the ids."),
) -> None:
    """List scenario ids for a scope -- a menu to copy, or a ready bench manifest.

    The ids come straight from the dataset folders, so the list is always
    current. Generate a manifest, prune the rows you don't want, then run it:

        python -m faultbench list --scope all --out benchmarks/selected.csv
        # edit benchmarks/selected.csv: delete (or #-prefix) the rows to skip
        python -m faultbench bench --manifest benchmarks/selected.csv
    """
    ids = [p.stem for p in _discover_or_exit(scope, topology)]
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["scenario"])
            for i in ids:
                w.writerow([i])
        console.print(f"wrote {len(ids)} scenarios -> {out}")
        console.print(f"[dim]edit it, then: python -m faultbench bench "
                      f"--manifest {out}[/dim]")
    else:
        for i in ids:
            print(i)


def _print_row(row: dict) -> None:
    t = Table(show_header=False)
    t.add_column("k", style="dim"); t.add_column("v")
    for k in ("scenario_id", "agent", "status", "submitted", "is_anomaly",
              "faulty_devices", "must_passed", "must_total",
              "bonus_passed", "bonus_total",
              "outcome_score", "fix_score", "reasoning_score",
              "wall_seconds", "error"):
        t.add_row(k, str(row.get(k, "")))
    console.print(t)


def _print_summary(rows: list[dict], agent_list: list[str]) -> None:
    """Compact scenario x agent matrix: outcome / fix / reasoning, or status."""
    by_scenario: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_scenario.setdefault(r["scenario_id"], {})[r["agent"]] = r

    t = Table(title="Benchmark summary  (o=outcome f=fix r=reasoning | status)")
    t.add_column("scenario", style="bold")
    for a in agent_list:
        t.add_column(a, justify="center")
    for sid, per_agent in by_scenario.items():
        cells = [sid]
        for a in agent_list:
            r = per_agent.get(a)
            if not r:
                cells.append("-")
            elif r.get("status") == "scored":
                rs = r.get("reasoning_score", "")
                osc = r.get("outcome_score", "")
                fx = r.get("fix_score", "")
                sub = " ".join(p for p in (
                    f"f{fx}" if fx not in ("", None) else "",
                    f"r{rs}" if rs not in ("", None) else "") if p)
                cells.append(f"o{osc}" + (f" ({sub})" if sub else ""))
            else:
                cells.append(f"[red]{r.get('status', '?')}[/red]")
        t.add_row(*cells)
    console.print(t)


if __name__ == "__main__":
    app()
