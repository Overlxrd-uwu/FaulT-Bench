"""Ticket statistics of the dataset, from dataset/ alone.

    python tools/dataset_stats.py            print the statistics and the integrity checks
    python tools/dataset_stats.py --check    same, and exit 1 if any value differs from the documented values

How the counting is defined: the MARK phrase lists and the identifier
regexes in this file.

Reads the 72 false-premise scenarios and their five persona rewrites
(dataset/<topology>_Q/{cex,novice,novice_confident,naive,naive_unsure,naive_no_detail}/).
Prints, per persona:
  * ID  = mean number of distinct verifiable identifiers per ticket (Table 2, "ID" column):
          IPv4 addresses and prefixes (also the 192.168.3.x placeholder form the tickets use),
          domain names, device names, interface names, VLAN ids and AS numbers, counted once
          each per ticket, case-insensitively, over the [PROMPT-TO-AGENT] block only;
  * for information, the share of tickets that name a cause, carry an explicit certainty
    marker, or prescribe a repair (not quoted in the paper; the marker lists are in MARK);
and checks the invariants the persona design promises: 72 tickets per persona, every
rewrite a no-injection negative control that carries a Variant-of lineage, every block
other than the ticket byte-identical to the parent scenario, novice-confident keeps the
novice identifiers exactly, and naive-no-detail contains no identifier at all.

Standard library only. dataset/ is never modified.
"""
from __future__ import annotations

import argparse
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"

PERSONAS = [  # name, sub-directory, file suffix after the parent stem
    ("standard", "cex", ""),
    ("novice", "novice", "-novice"),
    ("novice-confident", "novice_confident", "-novice-confident"),
    ("naive", "naive", "-naive"),
    ("naive-unsure", "naive_unsure", "-naive-unsure"),
    ("naive-no-detail", "naive_no_detail", "-naive-no-detail"),
]
PARENT = {"novice-confident": "novice", "naive-unsure": "naive", "naive-no-detail": "naive-unsure"}

# The documented per-persona identifier means (Table 2 of the paper); --check compares against these.
EXPECTED_ID = {"standard": 3.2, "novice": 1.9, "novice-confident": 1.9, "naive": 2.1,
            "naive-unsure": 2.1, "naive-no-detail": 0.0}

# ---- verifiable identifiers -------------------------------------------------------------
PAT = {
    "ipv4":   r"\b\d{1,3}(?:\.(?:\d{1,3}|x{1,3}))+(?:/\d{1,2})?\b",
    "domain": r"\b[a-z0-9][a-z0-9\-]*\.(?:com|net|org|local|lab)\b",
    "device": r"\b(?:pc|laptop|srv|server|host|router|sw|switch|asw|dsw|csw|"
              r"isp|core|leaf|spine|edge|fw|r)[-_]?\d+[a-z]?\d*\b",
    "iface":  r"\b(?:eth|gi|gig|fa|se|lo|br|bond)\s?\d+(?:[./]\d+)*\b",
    "vlan":   r"\bvlan\s?\d+\b",
    "asn":    r"\bAS\s?\d{3,6}\b",
}
STOP = {"re", "r", "router", "we", "were"}
MARK = {
    "cause": re.compile(r"\b(suspect\w*|obviously|clearly|definitely|must be|has (?:gone|fallen)|"
                        r"is failing|is down|has died|has crashed|one hundred percent|100%|because|"
                        r"it is the|it's the|blame\w*)\b", re.I),
    "certainty": re.compile(r"\b(obviously|clearly|definitely|one hundred percent|100%|certainly|"
                            r"surely|no doubt)\b", re.I),
    "repair": re.compile(r"\b(please\s+)?(restart|reboot|reset|reconnect|turn .{0,12}back on|"
                         r"bring .{0,12}back up|re-?enable)\b", re.I),
}


def identifiers(ticket: str) -> set[tuple[str, str]]:
    found = set()
    for kind, rx in PAT.items():
        for m in re.findall(rx, ticket, re.I):
            t = m.lower().strip()
            if kind == "device" and t in STOP:
                continue
            found.add((kind, t))
    return found


def block(text: str, name: str) -> str | None:
    m = re.search(rf"\[{name}\]\n(.*?)(?=\n\[|\Z)", text, re.S)
    return m.group(1).strip() if m else None


def ticket(text: str) -> str:
    b = block(text, "PROMPT-TO-AGENT")
    return " ".join(b.split()) if b else ""


def load():
    """{persona: {stem: file text}} for the 72 false-premise stems."""
    std = sorted(DATASET.glob("*_Q/cex/*false-premise*.txt"))
    stems = {p.stem: p.parent.parent.name for p in std}
    out = {}
    for name, sub, suf in PERSONAS:
        out[name] = {}
        for stem, q in stems.items():
            p = DATASET / q / sub / f"{stem}{suf}.txt"
            if p.exists():
                out[name][stem] = p.read_text(encoding="utf-8")
    return stems, out


def main() -> int:
    global DATASET
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if a value differs from the documented values")
    ap.add_argument("--dataset", type=Path, default=DATASET)
    a = ap.parse_args()
    DATASET = a.dataset
    stems, T = load()
    ok = True

    print(f"reading {DATASET} ... {len(stems)} false-premise stems")
    print("\nintegrity")
    for name, _, _ in PERSONAS:
        n = len(T[name])
        print(f"  {'OK  ' if n == 72 else 'FAIL'} {name:<17} {n}/72 tickets")
        ok &= n == 72
    bad_inj = [(n, s) for n, _, _ in PERSONAS for s, t in T[n].items()
               if "device: none" not in (block(t, "INJECTION") or "").lower()
               and "no fault" not in (block(t, "INJECTION") or "").lower()]
    print(f"  {'OK  ' if not bad_inj else 'FAIL'} every ticket is a no-injection negative control" + (f" {bad_inj[:3]}" if bad_inj else ""))
    ok &= not bad_inj
    bad_lin = [(n, s) for n, _, _ in PERSONAS if n != "standard" for s, t in T[n].items() if "Variant-of:" not in t]
    print(f"  {'OK  ' if not bad_lin else 'FAIL'} every rewrite carries a Variant-of lineage header" + (f" {bad_lin[:3]}" if bad_lin else ""))
    ok &= not bad_lin
    diff = []
    for name, _, _ in PERSONAS[1:]:
        for s, t in T[name].items():
            parent = T["standard"].get(s, "")
            for blk in ("INJECTION", "POST-INJECT-CHECK", "GROUND-TRUTH", "SCORING-AXES", "FIX", "NIKA-LABEL"):
                if (block(t, blk) or "") != (block(parent, blk) or ""):
                    diff.append((name, s, blk))
    print(f"  {'OK  ' if not diff else 'FAIL'} every block other than the ticket is byte-identical to the standard scenario" + (f" {diff[:3]}" if diff else ""))
    ok &= not diff
    changed = [s for s in stems if s in T["novice"] and s in T["novice-confident"]
               and identifiers(ticket(T["novice"][s])) != identifiers(ticket(T["novice-confident"][s]))]
    print(f"  {'OK  ' if not changed else 'FAIL'} novice-confident keeps the novice identifiers exactly (0 pairs differ)" + (f" {changed[:3]}" if changed else ""))
    ok &= not changed
    nz = [s for s, t in T["naive-no-detail"].items() if identifiers(ticket(t))]
    print(f"  {'OK  ' if not nz else 'FAIL'} naive-no-detail contains no identifier ({len(nz)} tickets with any)")
    ok &= not nz

    print("\nper persona (Table 2 ID column; the marker shares are informational)")
    print(f"  {'persona':<17} {'ID mean':>8} {'min':>4} {'max':>4} {'cause':>7} {'certainty':>10} {'repair':>7}")
    for name, _, _ in PERSONAS:
        tk = [ticket(t) for t in T[name].values()]
        counts = [len(identifiers(x)) for x in tk]
        pct = {k: 100.0 * sum(bool(rx.search(x)) for x in tk) / len(tk) for k, rx in MARK.items()}
        mean = st.fmean(counts)
        flag = ""
        if name in EXPECTED_ID and round(mean, 1) != EXPECTED_ID[name]:
            flag += f"  ID differs from the documented value ({EXPECTED_ID[name]})"; ok = False
        hu = lambda x: int(x + 0.5)   # whole percent, halves rounded up (12.5 -> 13), as the paper quotes them
        print(f"  {name:<17} {mean:>8.2f} {min(counts):>4} {max(counts):>4} {hu(pct['cause']):>6d}% {hu(pct['certainty']):>9d}% {hu(pct['repair']):>6d}%{flag}")
    print("\n  identifiers = distinct IPv4 addresses/prefixes, domain names, device names, interface names, VLAN ids, AS numbers in the ticket text;"
          "\n  cause / certainty / repair = the ticket contains a phrase from the corresponding marker list (see MARK in this file)")
    print("\nRESULT:", "matches the documented statistics" if ok else "MISMATCH, see above")
    return 0 if (ok or not a.check) else 1


if __name__ == "__main__":
    sys.exit(main())
