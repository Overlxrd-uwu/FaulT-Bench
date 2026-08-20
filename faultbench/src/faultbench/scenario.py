"""Parser for the .txt section-header scenario format.

Each scenario file has blocks delimited by `[SECTION-NAME]` lines:
    [PROMPT-TO-AGENT]        the trouble ticket (the only block the agent sees)
    [INJECTION]              commands that create the fault (empty if none)
    [POST-INJECT-CHECK]      probe that must confirm the fault holds
    [GROUND-TRUTH]           what is actually wrong, or that nothing is
    [FIX]                    canonical remediation (grades the fix score)
    [SCORING-AXES]           must/bonus/assert axes (grade the outcome)
    [DIAGNOSTIC-PROCESS]     reference walkthrough (documentation only)
    [NIKA-LABEL]             taxonomy label where one applies

The header at the top declares scenario id, topology, fault family and
ticket style. Unknown sections are parsed and kept; a [PRE-CHECK] block is
also accepted for compatibility.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


_HEADER_RE = re.compile(r"^Scenario:\s*(\S+)\s*$", re.MULTILINE)
_TOPOLOGY_RE = re.compile(r"^Topology:\s*(\S+)\s*$", re.MULTILINE)
_FAMILY_RE = re.compile(r"^Fault family:\s*(.+?)\s*$", re.MULTILINE)
_STYLE_RE = re.compile(r"^Ticket style:\s*(\S+)\s*$", re.MULTILINE)
_SECTION_RE = re.compile(r"^\[([A-Z][A-Z0-9-]*)\]\s*$", re.MULTILINE)


@dataclass
class CheckLine:
    """One line of a PRE-CHECK or POST-INJECT-CHECK block."""

    device: str
    command: str
    expect: str  # right-hand side of `=> expect X`


@dataclass
class Injection:
    """The [INJECTION] block."""

    device: str
    action: str
    command: str | None = None  # the actual cmd to run via `kathara exec`
    note: str | None = None     # for pre-existing/policy faults


def classify_scenario(scenario_id: str) -> str:
    """Canonical scenario class, derived from the id naming convention.

    - contains "false-premise"   -> "false-premise" (no-fault control; the
                                     ticket reports a fault that isn't there)
    - ends with "-misdirected"   -> "wrong-device"  (ticket blames the wrong device)
    - ends with "-misdescribed"  -> "wrong-cause"   (ticket mis-states the mechanism)
    - otherwise                  -> "base"          (accurate ticket)
    """
    sid = scenario_id.lower()
    if "false-premise" in sid:
        return "false-premise"
    if sid.endswith("-misdirected"):
        return "wrong-device"
    if sid.endswith("-misdescribed"):
        return "wrong-cause"
    return "base"


@dataclass
class Scenario:
    scenario_id: str
    topology: str          # e.g. "Kath3"
    fault_family: str
    prompt_to_agent: str
    injection: Injection
    pre_check: list[CheckLine] = field(default_factory=list)
    post_inject_check: list[CheckLine] = field(default_factory=list)
    ground_truth: str = ""
    fix_spec: str = ""     # [FIX] block — canonical remediation; "" for no-fault controls
    scoring_axes: list[str] = field(default_factory=list)
    source_path: Path | None = None
    scenario_class: str = ""  # base / false-premise / wrong-device / wrong-cause
    ticket_style: str = "standard"  # standard, or a persona layer (novice, naive, ...)

    def __post_init__(self) -> None:
        if not self.scenario_class:
            self.scenario_class = classify_scenario(self.scenario_id)


def parse_scenario(path: str | Path) -> Scenario:
    """Read a .txt scenario file and return a Scenario object."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    # Header parsing
    m = _HEADER_RE.search(text)
    if not m:
        raise ValueError(f"{path}: missing 'Scenario:' header")
    scenario_id = m.group(1)

    m = _TOPOLOGY_RE.search(text)
    if not m:
        raise ValueError(f"{path}: missing 'Topology:' header")
    topology = m.group(1)

    m = _FAMILY_RE.search(text)
    fault_family = m.group(1) if m else "unspecified"

    m = _STYLE_RE.search(text)
    ticket_style = m.group(1) if m else "standard"

    # Section parsing
    sections = _split_sections(text)

    inj = _parse_injection(sections.get("INJECTION", ""))
    pre = _parse_checks(sections.get("PRE-CHECK", ""))
    post = _parse_checks(sections.get("POST-INJECT-CHECK", ""))

    return Scenario(
        scenario_id=scenario_id,
        topology=topology,
        fault_family=fault_family,
        prompt_to_agent=sections.get("PROMPT-TO-AGENT", "").strip(),
        injection=inj,
        pre_check=pre,
        post_inject_check=post,
        ground_truth=sections.get("GROUND-TRUTH", "").strip(),
        fix_spec=sections.get("FIX", "").strip(),
        scoring_axes=_parse_scoring_axes(sections.get("SCORING-AXES", "")),
        source_path=path,
        ticket_style=ticket_style,
    )


def _split_sections(text: str) -> dict[str, str]:
    """Split the text into a dict of {SECTION-NAME: body_text}."""
    matches = list(_SECTION_RE.finditer(text))
    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[name] = text[start:end].strip()
    return out


def _parse_injection(body: str) -> Injection:
    """Parse [INJECTION] block (YAML-ish key:value)."""
    device = ""
    action = ""
    command: str | None = None
    note: str | None = None
    in_cmd = False
    in_note = False
    cmd_lines: list[str] = []
    note_lines: list[str] = []

    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("device:"):
            device = stripped.split(":", 1)[1].strip()
            in_cmd = in_note = False
        elif stripped.startswith("action:"):
            action = stripped.split(":", 1)[1].strip()
            in_cmd = in_note = False
        elif stripped.startswith("cmd:"):
            rest = stripped.split(":", 1)[1].strip()
            if rest == "|":
                in_cmd = True
                in_note = False
            else:
                command = rest
                in_cmd = in_note = False
        elif stripped.startswith("note:"):
            rest = stripped.split(":", 1)[1].strip()
            if rest == "|":
                in_note = True
                in_cmd = False
            else:
                note = rest
                in_cmd = in_note = False
        elif in_cmd:
            cmd_lines.append(line)
        elif in_note:
            note_lines.append(line)

    if cmd_lines:
        command = "\n".join(cmd_lines).strip()
    if note_lines:
        note = "\n".join(note_lines).strip()

    return Injection(device=device, action=action, command=command, note=note)


_CHECK_LINE_RE = re.compile(
    r"^(?P<device>\S+)\s+(?P<cmd>.+?)\s*=>\s*expect\s+(?P<expect>.+?)\s*$"
)


def _parse_checks(body: str) -> list[CheckLine]:
    """Parse a PRE-CHECK or POST-INJECT-CHECK block into CheckLine objects."""
    out: list[CheckLine] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("("):
            continue
        m = _CHECK_LINE_RE.match(line)
        if not m:
            continue
        out.append(
            CheckLine(
                device=m.group("device"),
                command=m.group("cmd").strip(),
                expect=m.group("expect").strip(),
            )
        )
    return out


def _parse_scoring_axes(body: str) -> list[str]:
    """Parse [SCORING-AXES] body — one axis per non-empty line."""
    return [
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


# ---- locating scenario files ------------------------------------------------

LAYERS = ("base", "cex", "novice", "naive", "novice_confident",
          "naive_unsure", "naive_no_detail")


def discover_scenarios(scope: str, topology: str | None = None) -> list[Path]:
    """Scenario .txt files for a scope, straight from the dataset tree.

    Dataset layout is dataset/<topology>_Q/<layer>/*.txt; selection is by GLOB,
    so it can never drift out of sync with the files on disk. ``all`` is the
    core dataset (base + cex = 200 scenarios); the five persona layers are
    separate scopes on purpose, so a full-dataset rerun cannot silently
    multiply into the rewrites too. Raises ValueError on an unknown scope.
    """
    from .paths import DATASET
    if scope == "all":
        found = sorted(DATASET.glob("*_Q/base/*.txt")) + sorted(
            DATASET.glob("*_Q/cex/*.txt"))
    elif scope in LAYERS:
        found = sorted(DATASET.glob(f"*_Q/{scope}/*.txt"))
    else:
        raise ValueError(
            f"unknown scope {scope!r}; choose from: {', '.join(LAYERS)}, all")
    if topology:
        found = [s for s in found
                 if any(p.name.lower() == f"{topology.lower()}_q" for p in s.parents)]
    return found


_ID_INDEX: dict[str, list[Path]] | None = None


def _scenario_id_index() -> dict[str, list[Path]]:
    """scenario id (the `Scenario:` header line) -> files, built once."""
    global _ID_INDEX
    if _ID_INDEX is None:
        from .paths import DATASET
        idx: dict[str, list[Path]] = {}
        for f in DATASET.glob("*_Q/**/*.txt"):
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines()[:10]:
                if line.startswith("Scenario:"):
                    idx.setdefault(line.split(":", 1)[1].strip(), []).append(f)
                    break
        _ID_INDEX = idx
    return _ID_INDEX


def resolve_scenario(ref: str) -> Path:
    """Resolve a reference (file stem, scenario id, or .txt path) to a real file.

    File stem first (what the CLI's `list` writes into manifests); if nothing
    matches, fall back to the `Scenario:` header id -- what every summary.csv
    row carries. The two are identical for every file in the released dataset;
    the fallback keeps older cell lists resolvable. Raises FileNotFoundError
    when nothing matches and ValueError when a reference is ambiguous.
    """
    from .paths import DATASET
    ref = ref.strip()
    p = Path(ref)
    if p.suffix == ".txt" and p.exists():
        return p
    stem = p.stem if p.suffix == ".txt" else ref
    matches = sorted(DATASET.glob(f"*_Q/**/{stem}.txt"))
    if not matches:
        matches = _scenario_id_index().get(stem, [])
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"scenario not found: {ref!r}")
    raise ValueError(f"ambiguous scenario id {ref!r}: {[str(m) for m in matches]}")


def load_manifest(path: Path) -> list[str]:
    """Scenario refs from a manifest file. A .csv needs a 'scenario' column;
    any other file is a line-list (blank lines + '#' comments ignored)."""
    import csv
    if path.suffix.lower() == ".csv":
        # utf-8-sig: tolerate the BOM that PowerShell 5.1 Out-File prepends
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if "scenario" not in (reader.fieldnames or []):
                raise ValueError(
                    f"{path}: CSV needs a 'scenario' column; got {reader.fieldnames}")
            return [s for row in reader
                    if (s := (row.get("scenario") or "").strip())
                    and not s.startswith("#")]
    return [s for line in path.read_text(encoding="utf-8").splitlines()
            if (s := line.strip()) and not s.startswith("#")]
