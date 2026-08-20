"""Kathara lab lifecycle wrappers.

Uses the Kathara Python API (NOT the CLI `kathara lstart`) so we can
explicitly set `lab.name = <topology>`. NIKA's MCP servers find the lab
via `Kathara.get_instance().get_lab_from_api(lab_name=...)` — that only
works for labs registered with a known name; CLI-started labs get an
auto-generated name the MCP servers cannot discover.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from func_timeout import FunctionTimedOut, func_timeout
from Kathara.manager.Kathara import Kathara
from Kathara.parser.netkit.LabParser import LabParser

from .paths import KATH_BASE


def lab_dir(topology: str) -> Path:
    """Map 'Kath3' -> the lab directory path."""
    return KATH_BASE / topology


def _is_nika_topology(topology: str) -> bool:
    """True if `topology` is a programmatic NIKA net_env (not a Kath* dir lab).

    NIKA labs are built in Python (nika.net_env) rather than parsed from a
    lab.conf dir, so they take a different deploy path. Imported lazily so the
    pipeline still runs Kath-only without nika installed.
    """
    try:
        from nika.net_env.net_env_pool import list_all_net_envs
        return topology in list_all_net_envs()
    except Exception:
        return False


def _kathara() -> Kathara:
    return Kathara.get_instance()


def lstart(topology: str) -> int:
    """Boot a Kathara lab via Python API with `lab.name = topology`.

    Always lclean first to clear any stale Docker state from a previous
    crashed run (avoids MachineAlreadyExistsError). Returns the lab's device
    count (= the expected running-container count) so the caller can confirm
    the deployment is complete.
    """
    if _is_nika_topology(topology):
        return _lstart_nika(topology)
    cwd = lab_dir(topology)
    if not cwd.exists():
        raise FileNotFoundError(f"Topology {topology} not found at {cwd}")
    lclean(topology)
    lab = LabParser().parse(str(cwd))
    lab.name = topology
    _kathara().deploy_lab(lab)
    return len(lab.machines)


def _lstart_nika(topology: str) -> int:
    """Deploy a programmatic NIKA net_env under Kathara, named `topology`.

    The NIKA env builds its own Lab (already named LAB_NAME == topology), so we
    lclean stale state and deploy it. Everything downstream (exec, checks,
    agents, teardown) then works by lab_name exactly like a Kath lab.
    """
    from nika.net_env.net_env_pool import get_net_env_instance
    lclean(topology)
    env = get_net_env_instance(topology)
    _kathara().deploy_lab(lab=env.lab)
    return len(env.lab.machines)


def kathara_container_count() -> int:
    """Count running Kathara lab containers (pipeline runs one lab at a time)."""
    try:
        out = subprocess.run(
            ["docker", "ps", "-q", "--filter", "name=kathara_"],
            check=False, timeout=15,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        ).stdout
        return len([ln for ln in out.splitlines() if ln.strip()])
    except Exception:
        return -1


def prune_kathara_networks() -> int:
    """Remove orphan kathara_ collision-domain networks left after teardown.

    Kathara's undeploy can leave collision-domain networks behind; across a long
    sweep (hundreds of deploy/teardown cycles) these accumulate and exhaust
    Docker's default address pool, wedging the daemon (new containers stick in
    'Created' and never start). Pruning ONLY kathara_-named, currently-unused
    networks after each teardown keeps the pool clear. Best-effort; swallows
    errors and never touches non-kathara or in-use networks. Returns the number
    of networks pruned (-1 if Docker could not be queried).
    """
    try:
        ids = subprocess.run(
            ["docker", "network", "ls", "--filter", "name=kathara", "-q"],
            check=False, timeout=30, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True,
        ).stdout.split()
        if ids:
            subprocess.run(["docker", "network", "rm", *ids], check=False,
                           timeout=90, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        return len(ids)
    except Exception:
        return -1


def lclean(topology: str, timeout: int = 300) -> None:
    """Tear down a Kathara lab by name. Belt-and-suspenders with CLI fallback."""
    try:
        _kathara().undeploy_lab(lab_name=topology)
    except Exception:
        pass
    # CLI fallback for any orphaned containers
    try:
        subprocess.run(
            ["kathara", "lclean"], cwd=lab_dir(topology),
            check=False, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
    except Exception:
        pass
    # Wait until Docker has actually removed the lab's containers before
    # returning, so a subsequent lstart deploys into a CLEAN state. Rapid
    # deploy/undeploy cycles (e.g. a multi-scenario bench) otherwise leave a
    # degraded Docker network state that stalls OSPF convergence on the next
    # lab — a fresh single lab converges in ~60s, a churned one may never.
    _wait_docker_clean()
    prune_kathara_networks()


def _wait_docker_clean(timeout: int = 90, settle: int = 4) -> None:
    """Poll until no `kathara_` containers remain, then a brief settle."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            left = subprocess.run(
                ["docker", "ps", "-q", "--filter", "name=kathara_"],
                check=False, timeout=15,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            ).stdout.strip()
        except Exception:
            left = ""
        if not left:
            break
        time.sleep(2)
    time.sleep(settle)


def kathara_exec(topology: str, device: str, command: str,
                 timeout: int = 30) -> tuple[int, str]:
    """Run a shell command inside a lab device. Returns (exit_code, stdout+stderr).

    Mirrors NIKA's `_run_cmd` pattern (see SADE-NetworkAgent/src/nika/service/
    kathara/base_api.py). The Python API's `exec()` returns a generator
    yielding items (bytes / str / int / None) — we collect non-empty
    decoded strings and concatenate.
    """
    def _exec_collect() -> tuple[int, str]:
        instance = _kathara()
        # MachineNotFoundError if device doesn't exist in the lab
        wrapped = "/bin/bash -c '{}'".format(
            command.replace("'", "'\\''")
        )
        out_parts: list[str] = []
        for item in instance.exec(
            machine_name=device, command=wrapped,
            lab_name=topology, stream=False,
        ):
            if not item or isinstance(item, int) or item == b"" or item == "None":
                continue
            if isinstance(item, bytes):
                out_parts.append(item.decode("utf-8", errors="replace"))
            else:
                out_parts.append(str(item))
        return 0, "".join(out_parts).strip()

    try:
        # The Kathara Python API's exec() generator blocks indefinitely on a
        # command that never returns (e.g. `tcpdump -c N` on a quiet link, or
        # an interactive prompt), so bound it with func_timeout (the same
        # mechanism SADE's base_api uses).
        return func_timeout(timeout, _exec_collect)
    except FunctionTimedOut:
        return 124, f"[TIMEOUT] command exceeded {timeout}s on {device}"
    except Exception as exc:
        # CLI fallback (rare — only when Python API breaks)
        env = os.environ.copy()
        env["MSYS_NO_PATHCONV"] = "1"
        try:
            proc = subprocess.run(
                ["kathara", "exec", device, "--", "sh", "-c", command],
                cwd=lab_dir(topology), env=env, timeout=timeout,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace",
            )
            return proc.returncode, proc.stdout or ""
        except Exception as exc2:
            return 1, f"[exec error] {type(exc).__name__}: {exc} | fallback: {exc2}"
