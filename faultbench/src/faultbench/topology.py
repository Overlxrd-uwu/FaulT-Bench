"""Generate a textual topology summary from a lab.conf.

The summary lists devices and the collision domains they share. NO IP
addresses or other configuration details — those the agent must discover
by running diagnostic commands inside the lab. The summary just gives the
agent a starting map of "what devices exist and how they connect."

Mirrors NIKA's approach of feeding the agent a high-level topology
overview alongside the symptom prompt.
"""
from __future__ import annotations

import re
from collections import defaultdict

from .lab import lab_dir, _is_nika_topology


_LAB_CONF_RE = re.compile(
    r'^(?P<device>[A-Za-z0-9_]+)\[(?P<key>\d+)\]\s*=\s*"?(?P<cd>[^"\s]+)"?'
)


def summarize_topology(topology: str) -> str:
    """Produce a human-readable topology summary for the given Kath{N} lab.

    Output structure:
        Topology: KathN

        Devices:
          <name>: eth0=<cd>, eth1=<cd>, ...

        Collision domains (shared L2 links):
          <cd>: <dev1>.eth0, <dev2>.eth2, ...

    Devices without numeric-keyed entries (e.g. ones that only have
    `[image]=...`) are still listed if they appear anywhere in lab.conf.

    NIKA topologies (built in code, no lab.conf) get an equivalent summary from
    the net_env's device lists + link topology — same "names + connectivity, no
    IPs" contract.
    """
    if _is_nika_topology(topology):
        return _summarize_nika(topology)

    lab_conf = lab_dir(topology) / "lab.conf"
    if not lab_conf.exists():
        return f"Topology: {topology}\n(lab.conf not found at {lab_conf})"

    devices: dict[str, list[tuple[int, str]]] = defaultdict(list)
    cds: dict[str, list[tuple[str, int]]] = defaultdict(list)
    seen_devices: set[str] = set()

    for raw in lab_conf.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LAB_CONF_RE.match(line)
        if not m:
            # Catch [image]=... and similar declarations
            m2 = re.match(r"^([A-Za-z0-9_]+)\[", line)
            if m2:
                seen_devices.add(m2.group(1))
            continue
        dev = m.group("device")
        n = int(m.group("key"))
        cd = m.group("cd")
        devices[dev].append((n, cd))
        cds[cd].append((dev, n))
        seen_devices.add(dev)

    out: list[str] = [f"Topology: {topology}", ""]

    out.append("Devices:")
    for dev in sorted(seen_devices):
        if devices.get(dev):
            ifaces = ", ".join(f"eth{n}={cd}" for n, cd in sorted(devices[dev]))
            out.append(f"  {dev}: {ifaces}")
        else:
            out.append(f"  {dev}: (no link assignments in lab.conf)")
    out.append("")

    out.append("Collision domains (shared L2 segments):")
    for cd in sorted(cds):
        members = ", ".join(f"{dev}.eth{n}" for dev, n in sorted(cds[cd]))
        out.append(f"  {cd}: {members}")

    return "\n".join(out)


def _summarize_nika(topology: str) -> str:
    """Devices + link topology for a NIKA net_env (names + connectivity, no IPs)."""
    from nika.net_env.net_env_pool import get_net_env_instance

    env = get_net_env_instance(topology)
    env.load_machines()
    out: list[str] = [f"Topology: {topology}", ""]

    out.append("Devices:")
    for label, names in (("Routers (FRR)", env.routers),
                         ("Switches", env.switches),
                         ("Hosts", env.hosts)):
        if names:
            out.append(f"  {label}: {', '.join(sorted(names))}")
    for stype, names in (env.servers or {}).items():
        if names:
            out.append(f"  {stype.capitalize()} servers: {', '.join(sorted(names))}")
    out.append("")

    out.append("Links (device:iface <-> device:iface):")
    for a, b in env.get_topology():
        out.append(f"  {a} <-> {b}")

    return "\n".join(out)
