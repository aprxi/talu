#!/usr/bin/env python3
"""Benchmark harness for talu serve endpoints.

Usage:
    python bench/run.py list
    python bench/run.py <scenario> [--config NAME] [--rounds N] [--set k=v ...]

Examples:
    python bench/run.py list
    python bench/run.py responses/perf/hello --config cuda
    python bench/run.py responses/perf/hello --config cuda --rounds 3
    python bench/run.py responses/perf/hello --config cuda --set precision=original,GAF4
    python bench/run.py responses/perf/hello --config cuda --set model_uri=Qwen/Qwen3-0.6B,Qwen/Qwen3.5-2B
    python bench/run.py responses/perf/hello --config cuda --env BACKEND=cpu
"""

from __future__ import annotations

import argparse
import datetime
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from server import TaluServer

# Import scenarios — triggers registration via __init_subclass__.
import responses  # noqa: F401

from scenario import (
    list_scenarios,
    list_configs,
    get_scenario,
    scenario_names,
    load_config,
)

console = Console()


# ---------------------------------------------------------------------------
# Version & hardware detection
# ---------------------------------------------------------------------------

def _run_quiet(*args: str) -> str:
    """Run a command, return stdout or empty string on failure."""
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _get_version(binary: Path) -> str:
    """Get talu version from binary -V output."""
    return _run_quiet(str(binary), "-V") or "unknown"


def _detect_hardware(env: dict[str, str]) -> str:
    """Return compact one-line hardware summary: CPU (· GPU when BACKEND=cuda)."""
    parts: list[str] = []

    # CPU.
    cpu = ""
    if platform.system() == "Darwin":
        cpu = _run_quiet("sysctl", "-n", "machdep.cpu.brand_string")
    else:
        for line in _run_quiet("lscpu").splitlines():
            if line.strip().startswith("Model name:"):
                cpu = line.split(":", 1)[1].strip()
                break
    if cpu:
        # Strip common noise suffixes.
        import re
        cpu = re.sub(r"\s+\d+-Core Processor$", "", cpu)
        parts.append(cpu)

    # GPU — only when the benchmark actually uses it.
    if env.get("BACKEND") == "cuda":
        gpu_name = _run_quiet(
            "nvidia-smi", "--query-gpu=name",
            "--format=csv,noheader", "--id=0",
        )
        if gpu_name:
            parts.append(gpu_name)

    return " · ".join(parts) if parts else "unknown"


def cmd_list() -> None:
    """Print available scenarios, defaults, and usage examples."""
    from scenario import _DEFAULT_CONFIG

    groups = list_scenarios()
    if not groups:
        print("No scenarios registered.")
        return

    # Scenarios.
    print("Scenarios:")
    for group, items in groups.items():
        configs = list_configs(items[0].name)
        conf_str = f"  configs: {', '.join(configs)}" if configs else ""
        print(f"  {group}/{conf_str}")

        # Collapse scenarios that share a family into one line.
        families: dict[str, list[type]] = {}
        standalone: list[type] = []
        for cls in items:
            if cls.family:
                families.setdefault(cls.family, []).append(cls)
            else:
                standalone.append(cls)

        for fam, members in sorted(families.items()):
            suffixes = sorted(
                (c.name.rsplit("/", 1)[-1][len(fam):] for c in members),
                key=lambda s: (int(s) if s.isdigit() else s),
            )
            label = f"{fam}{{{','.join(suffixes)}}}"
            desc = members[0].description
            print(f"    {label:<26s} {desc}")

        for cls in standalone:
            short = cls.name.rsplit("/", 1)[-1]
            print(f"    {short:<26s} {cls.description}")

    # Sampling presets.
    from scenario import SAMPLING_PRESETS
    print()
    print("Sampling presets:")
    for name, params in SAMPLING_PRESETS.items():
        parts = " ".join(f"{k}={v}" for k, v in params.items())
        marker = " (default)" if name == "general" else ""
        print(f"  {name:<15s} {parts}{marker}")

    # Defaults.
    cfg = _DEFAULT_CONFIG
    models = ",".join(cfg["model_uri"])
    precs = ",".join(cfg["precision"])
    print()
    print("Defaults:")
    print(f"  model_uri={models}  precision={precs}")
    print(f"  seed={cfg['seed']}  temperature={cfg['temperature']}  top_p={cfg['top_p']}  top_k={cfg['top_k']}")

    # Usage.
    first = next(iter(scenario_names()), "responses/perf/pp512")
    print()
    print("Usage:")
    argv0 = sys.argv[0]
    print(f"  python {argv0} {first}")
    print(f"  python {argv0} {first} --config cuda --rounds 3")
    print(f"  python {argv0} {first} --set model_uri=Qwen/Qwen3-0.6B,Qwen/Qwen3.5-0.8B")
    print(f"  python {argv0} {first} --set precision=original,GAF4")
    print(f"  python {argv0} {first} --set preset=coding")
    print(f"  python {argv0} {first} --set preset=coding --set temperature=0.3")
    print()


def _print_expanded_cmd(args: argparse.Namespace, config: dict) -> None:
    """Print the full CLI command with all resolved defaults."""
    parts = [f"python {sys.argv[0]}", args.scenario]
    if args.config:
        parts.append(f"--config {args.config}")
    parts.append(f"--rounds {args.rounds}")
    models = ",".join(config.get("model_uri", []))
    precs = ",".join(config.get("precision", []))
    parts.append(f"--set model_uri={models}")
    parts.append(f"--set precision={precs}")
    for key in ("seed", "temperature", "top_p", "top_k", "presence_penalty"):
        if key in config:
            parts.append(f"--set {key}={config[key]}")
    env_vars = config.get("env", {})
    for k, v in sorted(env_vars.items()):
        parts.append(f"--env {k}={v}")
    print(" \\\n    ".join(parts), flush=True)
    print()


def cmd_run(args: argparse.Namespace) -> None:
    """Run a single scenario."""
    cls = get_scenario(args.scenario)
    if cls is None:
        print(f"Unknown scenario: {args.scenario}", file=sys.stderr)
        print(f"Available: {', '.join(scenario_names())}", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.scenario, args.config, args.set, args.env)

    # Show the fully-expanded CLI command so users can reproduce/modify.
    _print_expanded_cmd(args, config)

    scenario = cls()
    env_vars = config.get("env", {})
    srv = TaluServer(port=args.port, extra_args=scenario.server_args(config), env=env_vars)

    version = _get_version(srv.binary)
    hardware = _detect_hardware(env_vars)

    results: list[dict] = []
    try:
        print(f"Starting server (port {args.port}) ...", flush=True)
        srv.start(timeout=30)
        print(f"Server ready (pid={srv.pid}).\n", flush=True)
        results = scenario.run(srv.base_url, args.rounds, config)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("\nStopping server ...", flush=True)
        srv.stop()

    if results:
        print()
        print_report(args.scenario, args.config, config, args.rounds, results,
                     version, hardware)


# ---------------------------------------------------------------------------
# Report formatting (Rich)
# ---------------------------------------------------------------------------

def print_report(
    scenario_name: str,
    config_name: str | None,
    config: dict,
    rounds: int,
    results: list[dict],
    version: str,
    hardware: str,
) -> None:
    """Print a presentable benchmark report using Rich tables."""
    env_vars = config.get("env", {})
    date = datetime.date.today().isoformat()

    # -- Header --
    console.print()
    console.rule(f"[bold]talu bench · {scenario_name}[/bold]", style="bright_blue")
    console.print()

    # Line 1: essentials.
    info_parts = [version, hardware, date]
    if config_name:
        info_parts.append(f"config={config_name}")
    console.print(f"  [dim]{' · '.join(info_parts)}[/]")

    # Line 2: parameters.
    param_parts = [f"rounds={rounds}"]
    for key in ("seed", "temperature", "top_p", "top_k", "presence_penalty", "max_tokens"):
        if key in config:
            param_parts.append(f"{key}={config[key]}")
    for k, v in sorted(env_vars.items()):
        param_parts.append(f"{k}={v}")
    console.print(f"  [dim]{' · '.join(param_parts)}[/]")
    console.print()

    # -- Group results by model, then by quant --
    by_model: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_model[r["model"]][r.get("scheme", "original")].append(r)

    for model, scheme_groups in by_model.items():
        _print_model_table(console, model, scheme_groups)


def _print_model_table(
    console: Console,
    model: str,
    scheme_groups: dict[str, list[dict]],
) -> None:
    """Print a results table for one model with grouped column headers."""
    # Column definitions: (header, group, right-align).
    col_defs = [
        ("Precision", "",               False),
        ("Size",  "",               True),
        ("tok",   "Prefill",    False),
        ("avg t/s", "Prefill",  True),
        ("min",   "Prefill",    True),
        ("max",   "Prefill",    True),
        ("tok",   "Generate", True),
        ("avg t/s", "Generate", True),
        ("min",   "Generate", True),
        ("max",   "Generate", True),
        ("avg",   "TTFT",       True),
        ("min",   "TTFT",       True),
        ("max",   "TTFT",       True),
    ]
    gap_inner = 2   # within a group
    gap_group = 5   # between groups

    # Build data rows.
    data_rows: list[list[str]] = []
    for scheme, rows in scheme_groups.items():
        # Model info is per-precision (file size differs per variant).
        mi: dict = rows[0].get("model_info", {}) or {}
        fs = mi.get("file_size_bytes", 0)
        if fs >= 1 << 30:
            size_str = f"{fs / (1 << 30):.1f} GB"
        elif fs > 0:
            size_str = f"{fs / (1 << 20):.0f} MB"
        else:
            size_str = "—"
        rates = [r["engine_tok_s"] for r in rows if r["engine_tok_s"] > 0]
        in_tok = str(rows[0].get("input_tokens", 0))
        out_tok = str(rows[0].get("output_tokens", 0))
        if rates:
            avg = f"{sum(rates) / len(rates):.1f}"
            mn = f"{min(rates):.1f}"
            mx = f"{max(rates):.1f}"
        else:
            avg = mn = mx = "—"
        prefill_rates = [r["prefill_tok_s"] for r in rows if r.get("prefill_tok_s", 0) > 0]
        if prefill_rates:
            prefill_avg = f"{sum(prefill_rates)/len(prefill_rates):.1f}"
            prefill_mn = f"{min(prefill_rates):.1f}"
            prefill_mx = f"{max(prefill_rates):.1f}"
        else:
            prefill_avg = prefill_mn = prefill_mx = "—"
        ttft_vals = [r["ttft_ms"] for r in rows if r.get("ttft_ms", 0) > 0]
        if ttft_vals:
            ttft_avg = f"{sum(ttft_vals)/len(ttft_vals):.0f}ms"
            ttft_mn = f"{min(ttft_vals):.0f}ms"
            ttft_mx = f"{max(ttft_vals):.0f}ms"
        else:
            ttft_avg = ttft_mn = ttft_mx = "—"
        # Derive precision label from model_info (includes group size for GAF).
        dtype = mi.get("weight_dtype", "")
        gs = mi.get("gaffine_group_size", 0)
        if dtype and dtype.startswith("GAF") and gs > 0:
            label = f"{dtype}_{gs}"
        elif dtype:
            label = dtype
        else:
            label = scheme if scheme != "original" else "n/a"
        data_rows.append([label, size_str, in_tok, prefill_avg, prefill_mn, prefill_mx, out_tok, avg, mn, mx, ttft_avg, ttft_mn, ttft_mx])

    # Compute per-column gaps: wider between groups, tighter within.
    gaps = []
    for i in range(1, len(col_defs)):
        gaps.append(gap_group if col_defs[i][1] != col_defs[i - 1][1] else gap_inner)

    def join_parts(parts: list[str]) -> str:
        """Join column parts using per-column gaps."""
        out = parts[0]
        for i, p in enumerate(parts[1:]):
            out += " " * gaps[i] + p
        return out

    # Compute column widths.
    widths = [len(cd[0]) for cd in col_defs]
    for row in data_rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    # Identify groups and their column ranges.
    group_ranges: list[tuple[str, int, int]] = []  # (name, start, end)
    cur_group = col_defs[0][1]
    cur_start = 0
    for i in range(1, len(col_defs)):
        g = col_defs[i][1]
        if g != cur_group:
            group_ranges.append((cur_group, cur_start, i - 1))
            cur_group = g
            cur_start = i
    group_ranges.append((cur_group, cur_start, len(col_defs) - 1))

    # Widen columns if group label is wider than the span.
    for name, start, end in group_ranges:
        if not name:
            continue
        label_width = len(name)
        inner_gaps = sum(gaps[start:end]) if end > start else 0
        span_width = sum(widths[start:end + 1]) + inner_gaps
        if label_width > span_width:
            widths[start] += label_width - span_width

    # Build spanning group line.
    group_parts: list[str] = []
    for name, start, end in group_ranges:
        inner_gaps = sum(gaps[start:end]) if end > start else 0
        w = sum(widths[start:end + 1]) + inner_gaps
        if name:
            side = (w - len(name)) // 2
            span = " " * side + name + " " * (w - side - len(name))
            group_parts.append(span)
        else:
            group_parts.append(" " * w)
    # Join group parts with the gaps between groups.
    group_line_parts: list[str] = []
    gi = 0
    for idx, (name, start, end) in enumerate(group_ranges):
        group_line_parts.append(group_parts[idx])
        if idx < len(group_ranges) - 1:
            group_line_parts.append(" " * gaps[end])  # gap after this group
    group_line = "".join(group_line_parts)

    # Column headers + separator.
    header_parts = [
        f"{cd[0]:>{w}}" if cd[2] else f"{cd[0]:<{w}}"
        for cd, w in zip(col_defs, widths)
    ]
    header_line = join_parts(header_parts)
    sep_line = join_parts(["─" * w for w in widths])

    # Data lines with Rich markup.
    content_lines = [
        f"[bold]{group_line}[/]",
        f"[dim]{header_line}[/]",
        f"[dim]{sep_line}[/]",
    ]
    for row in data_rows:
        parts: list[str] = []
        for i, (val, w) in enumerate(zip(row, widths)):
            cell = f"{val:>{w}}" if col_defs[i][2] else f"{val:<{w}}"
            if i == 0:
                cell = f"[cyan bold]{cell}[/]"
            elif i in (3, 7):
                cell = f"[green bold]{cell}[/]"
            elif i in (4, 5, 8, 9):
                cell = f"[dim]{cell}[/]"
            parts.append(cell)
        content_lines.append(join_parts(parts))

    panel = Panel(
        "\n".join(content_lines),
        title=f"[bold]{model}[/bold]",
        title_align="left",
        border_style="blue",
        padding=(0, 1),
        expand=False,
    )
    console.print(panel)
    console.print()


def main() -> None:
    # Bare scenario name shorthand: `run.py responses/hello` → `run.py run responses/hello`
    if len(sys.argv) > 1 and sys.argv[1] not in ("list", "run", "-h", "--help"):
        if get_scenario(sys.argv[1]) is not None:
            sys.argv.insert(1, "run")

    parser = argparse.ArgumentParser(
        description="Benchmark harness for talu serve endpoints.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List available scenarios and configs.")

    run_p = sub.add_parser("run", help="Run a scenario.")
    run_p.add_argument("scenario", help="Scenario name (e.g. responses/perf/hello).")
    run_p.add_argument("--config", "-c", default=None, help="Config name (e.g. cpu).")
    run_p.add_argument("--port", type=int, default=18258, help="Server port (default: 18258).")
    run_p.add_argument("--rounds", "-n", type=int, default=5, help="Rounds per variant (default: 5).")
    run_p.add_argument("--set", "-s", action="append", default=[], metavar="KEY=VALUE",
                        help="Override config value. Parsed as JSON, fallback to string. Repeatable.")
    run_p.add_argument("--env", "-e", action="append", default=[], metavar="KEY=VALUE",
                        help="Set environment variable for the server process. Repeatable.")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list()
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
