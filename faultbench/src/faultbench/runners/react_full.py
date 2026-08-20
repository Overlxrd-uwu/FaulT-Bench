"""ReAct (gpt-5) runner ("react-full"): the GPT arm of the benchmark.

Plain Reason/Act loop over OpenAI function calling, exposing the SAME
22 Kathara tools the Claude agents (SADE, cc-baseline) get -- the three
`kathara_*` MCP servers -- as OpenAI function tools. This makes ReAct a fair
peer to cc-baseline: identical toolset, different model.

Evidence parity: every tool handler calls the SAME backend the MCP servers
wrap -- `nika.service.kathara.KatharaAPIALL` (the combined Base+FRR+Telemetry+
TC class) -- and the SAME method each MCP tool calls. So the bytes ReAct sees
== what SADE/cc-baseline see via MCP. We construct ONE KatharaAPIALL per run
(lab_name fixed for the run) and reuse it across tool calls.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

from openai import OpenAI

from ..prompts.react_freetext import REACT_FULL_FREETEXT_PROMPT
from .base import AgentResult, log_submission, parse_diagnosis


# ---- OpenAI function-tool schemas: one per kathara_* MCP tool ---------------

_STR = {"type": "string"}
_INT = {"type": "integer"}


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


ALL_TOOLS = [
    # --- kathara_base (13) ---
    _tool("get_reachability",
          "Topology-wide ping matrix: ping from each host to all other hosts "
          "(a subset is tested if there are too many). Best first action to map "
          "connectivity.", {}, []),
    _tool("ping_pair", "Ping from one host to another in the lab.",
          {"host_a": _STR, "host_b": _STR, "count": _INT, "args": _STR},
          ["host_a", "host_b"]),
    _tool("systemctl_ops",
          "Run systemctl (start/stop/restart/status) for a service on a host.",
          {"host_name": _STR, "service_name": _STR, "operation": _STR},
          ["host_name", "service_name", "operation"]),
    _tool("get_host_net_config",
          "Get a host's network configuration (ifconfig, ip addr, ip route).",
          {"host_name": _STR}, ["host_name"]),
    _tool("get_tc_statistics",
          "Get traffic-control (tc) statistics for an interface on a host.",
          {"host_name": _STR, "intf_name": _STR}, ["host_name", "intf_name"]),
    _tool("netstat", "Run netstat on a host (default args -tuln).",
          {"host_name": _STR, "args": _STR}, ["host_name"]),
    _tool("ip_addr_statistics", "Get IP address statistics of a host.",
          {"host_name": _STR}, ["host_name"]),
    _tool("ethtool", "Run ethtool on a host interface with given args.",
          {"host_name": _STR, "interface": _STR, "args": _STR},
          ["host_name", "interface", "args"]),
    _tool("curl_web_test",
          "Curl a URL from a host several times; returns timing stats "
          "(lookup/connect/TTFB/total).",
          {"host_name": _STR, "url": _STR, "times": _INT},
          ["host_name", "url"]),
    _tool("iperf_test", "Run an iperf throughput test between two hosts.",
          {"client_host_name": _STR, "server_host_name": _STR, "duration": _INT,
           "client_args": _STR, "server_args": _STR},
          ["client_host_name", "server_host_name"]),
    _tool("cat_file", "Show the contents of a file on a host.",
          {"host_name": _STR, "file_path": _STR}, ["host_name", "file_path"]),
    _tool("exec_shell", "Execute an arbitrary shell command on a host.",
          {"host_name": _STR, "command": _STR}, ["host_name", "command"]),
    _tool("exec_shell_dual",
          "Execute shell commands on two hosts concurrently.",
          {"host1": _STR, "cmd1": _STR, "host2": _STR, "cmd2": _STR},
          ["host1", "cmd1", "host2", "cmd2"]),
    # --- kathara_frr (5) ---
    _tool("frr_get_bgp_conf", "Get the BGP configuration from an FRR router.",
          {"router_name": _STR}, ["router_name"]),
    _tool("frr_show_running_config",
          "Get the running configuration from an FRR router.",
          {"router_name": _STR}, ["router_name"]),
    _tool("frr_show_ip_route", "Get the IP routing table from an FRR router.",
          {"router_name": _STR}, ["router_name"]),
    _tool("frr_get_ospf_conf", "Get the OSPF configuration from an FRR router.",
          {"router_name": _STR}, ["router_name"]),
    _tool("frr_exec", "Execute a vtysh command on an FRR router.",
          {"router_name": _STR, "command": _STR}, ["router_name", "command"]),
    # --- kathara_telemetry (4) ---
    _tool("influx_list_buckets", "List all InfluxDB buckets.", {}, []),
    _tool("influx_get_measurements", "List all InfluxDB measurements.", {}, []),
    _tool("influx_count_measurements",
          "Count records in an InfluxDB measurement.",
          {"measurement": _STR}, ["measurement"]),
    _tool("influx_query_measurement",
          "Query an InfluxDB measurement (use count first; LIMIT/OFFSET for big "
          "results).",
          {"measurement": _STR, "limit": _INT, "offset": _INT}, ["measurement"]),
]


def _dispatch(api, name: str, a: dict):
    """Route a tool call to the SAME KatharaAPIALL method the MCP server uses."""
    if name == "get_reachability":
        return asyncio.run(api.get_reachability())
    if name == "ping_pair":
        return api.ping_pair(host_a=a["host_a"], host_b=a["host_b"],
                             count=a.get("count", 4), args=a.get("args", ""))
    if name == "systemctl_ops":
        return api.systemctl_ops(host_name=a["host_name"],
                                 service_name=a["service_name"],
                                 operation=a["operation"])
    if name == "get_host_net_config":
        return api.get_host_net_config(host_name=a["host_name"])
    if name == "get_tc_statistics":
        return api.tc_show_statistics(host_name=a["host_name"],
                                      intf_name=a["intf_name"])
    if name == "netstat":
        return api.netstat(host_name=a["host_name"], args=a.get("args", "-tuln"))
    if name == "ip_addr_statistics":
        return api.ip_addr_statistics(host_name=a["host_name"])
    if name == "ethtool":
        return api.ethtool(host_name=a["host_name"], interface=a["interface"],
                           args=a.get("args", ""))
    if name == "curl_web_test":
        return api.curl_web_test(host_name=a["host_name"], url=a["url"],
                                 times=a.get("times", 5))
    if name == "iperf_test":
        return api.iperf_test(client_host_name=a["client_host_name"],
                              server_host_name=a["server_host_name"],
                              duration=a.get("duration", 10),
                              client_args=a.get("client_args", ""),
                              server_args=a.get("server_args", ""))
    if name == "cat_file":
        return api.exec_cmd(host_name=a["host_name"],
                            command=f"cat {a['file_path']}")
    if name == "exec_shell":
        return api.exec_cmd(a["host_name"], a["command"])
    if name == "exec_shell_dual":
        async def _dual():
            return await asyncio.gather(
                api.exec_cmd_async(a["host1"], a["cmd1"]),
                api.exec_cmd_async(a["host2"], a["cmd2"]),
            )
        r1, r2 = asyncio.run(_dual())
        return {"host1": [r1], "host2": [r2]}
    if name == "frr_get_bgp_conf":
        return api.frr_get_bgp_conf(a["router_name"])
    if name == "frr_show_running_config":
        return api.frr_show_running_config(a["router_name"])
    if name == "frr_show_ip_route":
        return api.frr_show_route(a["router_name"])
    if name == "frr_get_ospf_conf":
        return api.frr_get_ospf_conf(a["router_name"])
    if name == "frr_exec":
        return api.frr_exec(a["router_name"], a["command"])
    if name == "influx_list_buckets":
        return api.influx_list_buckets()
    if name == "influx_get_measurements":
        return api.influx_get_measurements()
    if name == "influx_count_measurements":
        return api.influx_count_measurements(a["measurement"])
    if name == "influx_query_measurement":
        return api.influx_query_measurement(a["measurement"],
                                            limit=a.get("limit", 10),
                                            offset=a.get("offset", 0))
    raise KeyError(f"unknown tool {name}")


def _stringify(result, cap: int = 8000) -> str:
    if not isinstance(result, str):
        try:
            result = json.dumps(result, default=str)
        except Exception:
            result = str(result)
    return result if len(result) <= cap else result[:cap] + "\n[...truncated]"


def _print_stream(line: str) -> None:
    try:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    except UnicodeEncodeError:
        sys.stdout.write(line.encode("ascii", errors="replace").decode("ascii") + "\n")
        sys.stdout.flush()


def run_react_full(prompt_to_agent: str, lab_name: str,
                   model: str = "gpt-5",
                   max_turns: int = 20,
                   stream: bool = True) -> AgentResult:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return AgentResult(error="OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    # Same backend the MCP servers wrap; lab_name fixed for the run.
    from nika.service.kathara import KatharaAPIALL
    api = KatharaAPIALL(lab_name=lab_name)

    messages = [
        {"role": "system", "content": REACT_FULL_FREETEXT_PROMPT},
        {"role": "user", "content": prompt_to_agent},
    ]

    in_tok = out_tok = cache_read = tool_calls = 0
    hit_max_turns = False
    final_text = ""
    trace_lines: list[str] = []

    def log(label: str, body: str = "") -> None:
        line = f"[{label}] {body}"
        trace_lines.append(line)
        if stream:
            _print_stream(line)

    log("AGENT", f"react-full ({model}) starts on lab={lab_name} "
                 f"with {len(ALL_TOOLS)} tools")

    for _ in range(max_turns):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=ALL_TOOLS,
            tool_choice="auto",
        )
        usage = getattr(resp, "usage", None)
        if usage:
            prompt = usage.prompt_tokens or 0
            cached = (
                getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
                if getattr(usage, "prompt_tokens_details", None) is not None
                else 0
            )
            in_tok += max(0, prompt - cached)
            cache_read += cached
            out_tok += usage.completion_tokens or 0

        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if msg.content:
            log("TEXT", msg.content[:800])
            final_text = msg.content

        if not msg.tool_calls:
            log("FINAL", f"stop=no-more-tool-calls tool_calls={tool_calls}")
            break

        for tc in msg.tool_calls:
            tool_calls += 1
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            log("TOOL", f"{name}({json.dumps(args)[:300]})")

            try:
                result = _stringify(_dispatch(api, name, args))
            except KeyError:
                result = f"ERROR: unknown tool {name}"
            except Exception as exc:  # tool blew up — surface it, keep going
                result = f"ERROR running {name}: {type(exc).__name__}: {exc}"
            log("RESULT", result[:400])

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
    else:
        hit_max_turns = True
        log("FINAL", f"stop=max-turns tool_calls={tool_calls}")

    is_anom, devs, diag, ok = parse_diagnosis(final_text)
    log_submission(log, final_text, is_anom, devs, ok)
    # gpt-5 pricing (USD per 1M tokens): fresh $1.25, cached $0.125, output $10.00.
    # Verify against current OpenAI pricing before trusting cost.
    cost_usd = (in_tok * 1.25 + cache_read * 0.125 + out_tok * 10.00) / 1_000_000.0
    return AgentResult(
        diagnosis_text=diag,
        is_anomaly=is_anom,
        faulty_devices=devs,
        parsed_ok=ok,
        in_tokens=in_tok,
        cache_read_tokens=cache_read,
        cache_write_tokens=0,
        out_tokens=out_tok,
        cost_usd=cost_usd,
        tool_calls=tool_calls,
        hit_max_turns=hit_max_turns,
        raw_trace="\n".join(trace_lines),
    )
