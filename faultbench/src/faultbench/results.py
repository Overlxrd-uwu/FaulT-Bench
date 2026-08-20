"""Per-run output writers — JSON (detailed) + CSV (flat, one row per agent×scenario)."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

CSV_COLUMNS = [
    "timestamp", "run_id", "run_group",
    "scenario_id", "class", "ticket_style", "topology", "fault_family",
    "agent", "model", "status",
    "submitted", "is_anomaly", "faulty_devices",
    "must_passed", "must_total", "bonus_passed", "bonus_total",
    "outcome_score", "fix_score", "reasoning_score",
    "in_tokens", "cache_read_tokens", "cache_write_tokens",
    "out_tokens", "cost_usd", "tool_calls",
    "wall_seconds", "error",
]


def write_row(csv_path: Path, row: dict) -> None:
    """Append a single result row to the CSV, creating header if needed.

    When appending to a pre-existing CSV, reuse ITS header as the field list
    (older files may predate newer CSV_COLUMNS entries such as ``class``) so
    rows never shift out of alignment with the header.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    fieldnames = CSV_COLUMNS
    if exists:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header:
            fieldnames = header
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            w.writeheader()
        # Coerce list/dict values to JSON strings for CSV
        row = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
               for k, v in row.items()}
        w.writerow(row)


def write_detail(detail_path: Path, payload: dict) -> None:
    """Write the full per-scenario JSON with trace, ground truth, and judgments."""
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path.write_text(_to_json_str(payload), encoding="utf-8")


def _to_json_str(obj) -> str:
    return json.dumps(obj, indent=2, default=_default)


def _default(o):
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(f"not serializable: {type(o)}")
