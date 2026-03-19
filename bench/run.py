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
    python bench/run.py responses/evals/mmlu --samples 100
    python bench/run.py responses/evals/gpqa --samples 50
    python bench/run.py db/perf/sql_select1 --set requests=200 --set concurrency=8
"""

from __future__ import annotations

import argparse
import datetime
import platform
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from server import TaluServer

# Import scenarios — triggers registration via __init_subclass__.
import responses  # noqa: F401
import db  # noqa: F401

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


def _infer_tg_shape(scenario_name: str) -> tuple[int, int] | None:
    """Infer (max_output_tokens, async in-flight) from tg scenario names."""
    prefix = "responses/perf/tg"
    if not scenario_name.startswith(prefix):
        return None

    suffix = scenario_name[len(prefix):]
    if not suffix:
        return None

    i = 0
    while i < len(suffix) and suffix[i].isdigit():
        i += 1
    if i == 0:
        return None
    tg_tokens = int(suffix[:i])
    if i == len(suffix):
        return tg_tokens, 1
    if suffix[i] != "b":
        return None

    batch_part = suffix[i + 1:]
    if not batch_part or not batch_part.isdigit():
        return None

    batch = int(batch_part)
    return tg_tokens, batch if batch > 0 else 1


def _coerce_bool(value: object) -> bool:
    """Parse bool-like config values from JSON/CLI overrides."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        if v in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)

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
                key=lambda s: (0, int(s)) if s.isdigit() else (1, s),
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
    names = scenario_names()
    first = next(
        (n for n in names if n.startswith("responses/perf/")),
        names[0] if names else "responses/perf/pp512",
    )
    first_db = next((n for n in names if n.startswith("db/perf/")), None)
    print()
    print("Usage:")
    argv0 = sys.argv[0]
    print(f"  python {argv0} {first}")
    print(f"  python {argv0} {first} --config cuda --rounds 3")
    print(f"  python {argv0} {first} --set model_uri=Qwen/Qwen3-0.6B,Qwen/Qwen3.5-0.8B")
    print(f"  python {argv0} {first} --set precision=original,GAF4")
    print(f"  python {argv0} {first} --set preset=coding")
    print(f"  python {argv0} {first} --set preset=coding --set temperature=0.3")
    if first_db:
        print(f"  python {argv0} {first_db} --config local --set requests=200 --set concurrency=8")
    print()


def _print_expanded_cmd(args: argparse.Namespace, config: dict) -> None:
    """Print the full CLI command with all resolved defaults."""
    parts = [f"python {sys.argv[0]}", args.scenario]
    if args.config:
        parts.append(f"--config {args.config}")
    if args.samples is not None:
        parts.append(f"--samples {args.samples}")
    is_eval = "evals/" in args.scenario
    if not is_eval:
        parts.append(f"--rounds {args.rounds}")
    models = ",".join(config.get("model_uri", []))
    precs = ",".join(config.get("precision", []))
    parts.append(f"--set model_uri={models}")
    parts.append(f"--set precision={precs}")
    # Show all tunable params — explicit values and discoverable defaults.
    _EVAL_DEFAULTS = {
        "max_reasoning_tokens": 0,
        "seed": None,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "presence_penalty": None,
    }
    for key, default in _EVAL_DEFAULTS.items():
        val = config.get(key, default if is_eval else None)
        if val is not None:
            # Format lists as comma-separated (e.g. max_reasoning_tokens=10,20).
            if isinstance(val, list):
                val = ",".join(str(v) for v in val)
            parts.append(f"--set {key}={val}")
    env_vars = config.get("env", {})
    for k, v in sorted(env_vars.items()):
        parts.append(f"--env {k}={v}")
    print(" \\\n    ".join(parts), flush=True)
    print()


def _is_eval_scenario(scenario_name: str, results: list[dict]) -> bool:
    """Detect eval scenarios by name prefix or result shape."""
    if "evals/" in scenario_name:
        return True
    return bool(results) and "accuracy" in results[0]


def _is_db_scenario(scenario_name: str, results: list[dict]) -> bool:
    """Detect DB perf scenarios by name prefix or result shape."""
    if scenario_name.startswith("db/"):
        return True
    if not results:
        return False
    first = results[0]
    return "rps" in first and "p95_ms" in first


def cmd_run(args: argparse.Namespace) -> None:
    """Run a single scenario."""
    cls = get_scenario(args.scenario)
    if cls is None:
        print(f"Unknown scenario: {args.scenario}", file=sys.stderr)
        print(f"Available: {', '.join(scenario_names())}", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.scenario, args.config, args.set, args.env)

    # Eval scenarios: default to original precision, rounds=1, pass --samples.
    if "evals/" in args.scenario:
        if args.samples is not None:
            config["samples"] = args.samples
        # Default to original only (user can override with --set precision=...).
        precision_explicit = any(s.startswith("precision=") for s in args.set)
        if not precision_explicit:
            config["precision"] = ["original"]
        args.rounds = 1

    # For tg* perf scenarios, align CUDA scheduler capacity and sequence cap
    # with the scenario shape unless explicitly overridden by user/config.
    tg_shape = _infer_tg_shape(args.scenario)
    if tg_shape is not None:
        tg_tokens, auto_batch = tg_shape
        # Keep a conservative envelope above prompt + output while remaining
        # much smaller than model defaults (often 32k+).
        auto_max_seq = max(1024, tg_tokens * 2)
        env_cfg = config.setdefault("env", {})
        if "TALU_CUDA_MAX_BATCH_SIZE" not in env_cfg:
            env_cfg["TALU_CUDA_MAX_BATCH_SIZE"] = str(auto_batch)
            print(
                f"[auto] TALU_CUDA_MAX_BATCH_SIZE={auto_batch} "
                f"(inferred from scenario {args.scenario})",
                flush=True,
            )
        if "TALU_CUDA_MAX_SEQ_LEN" not in env_cfg:
            env_cfg["TALU_CUDA_MAX_SEQ_LEN"] = str(auto_max_seq)
            print(
                f"[auto] TALU_CUDA_MAX_SEQ_LEN={auto_max_seq} "
                f"(inferred from scenario {args.scenario})",
                flush=True,
            )
        if "TALU_BATCH_MAX_INFLIGHT" not in env_cfg:
            # Keep server queue bounded to intended async fan-out for clearer
            # throughput/latency characteristics in tg perf runs. This is host-
            # side backpressure, not a direct GPU memory guarantee.
            env_cfg["TALU_BATCH_MAX_INFLIGHT"] = str(auto_batch)
            print(
                f"[auto] TALU_BATCH_MAX_INFLIGHT={auto_batch} "
                f"(inferred from scenario {args.scenario})",
                flush=True,
            )

    # Optional fixed-allocation mode for predictable-memory perf runs:
    # --set fixed_alloc=true
    if "fixed_alloc" in config:
        fixed_alloc = _coerce_bool(config["fixed_alloc"])
        env_cfg = config.setdefault("env", {})
        if "TALU_CUDA_FIXED_ALLOC" not in env_cfg:
            env_cfg["TALU_CUDA_FIXED_ALLOC"] = "1" if fixed_alloc else "0"
            print(
                f"[auto] TALU_CUDA_FIXED_ALLOC={env_cfg['TALU_CUDA_FIXED_ALLOC']} "
                "(from --set fixed_alloc=...)",
                flush=True,
            )
        if fixed_alloc and "TALU_CUDA_REQUIRE_FIT" not in env_cfg:
            env_cfg["TALU_CUDA_REQUIRE_FIT"] = "1"
            print(
                "[auto] TALU_CUDA_REQUIRE_FIT=1 (fixed_alloc enabled)",
                flush=True,
            )

    scenario = cls()
    if not scenario.uses_model_matrix:
        # DB scenarios benchmark endpoint behavior, not model variants.
        config["model_uri"] = ["n/a"]
        config["precision"] = ["original"]

    # Show the fully-expanded CLI command so users can reproduce/modify.
    _print_expanded_cmd(args, config)

    env_vars = config.get("env", {})
    temp_bucket: tempfile.TemporaryDirectory[str] | None = None
    bucket_path: Path | None = None
    if scenario.requires_storage:
        cfg_bucket = config.get("bucket_path")
        if isinstance(cfg_bucket, str) and cfg_bucket.strip():
            bucket_path = Path(cfg_bucket).expanduser()
            bucket_path.mkdir(parents=True, exist_ok=True)
        else:
            temp_bucket = tempfile.TemporaryDirectory(prefix="talu-bench-db-")
            bucket_path = Path(temp_bucket.name)
        print(f"[auto] DB benchmark bucket={bucket_path}", flush=True)

    # Version and hardware detection (no server needed).
    _probe = TaluServer(port=args.port)
    version = _get_version(_probe.binary)
    hardware = _detect_hardware(env_vars)

    # Restart the server for each model × precision combo so the previous
    # model is fully unloaded from GPU memory before the next one loads.
    model_uris: list[str] = config.get("model_uri", ["Qwen/Qwen3.5-0.8B"])
    precisions: list[str] = config.get("precision", ["original"])

    results: list[dict] = []
    try:
        for model in model_uris:
            for precision in precisions:
                combo_config = dict(config)
                combo_config["model_uri"] = [model]
                combo_config["precision"] = [precision]

                srv = TaluServer(
                    port=args.port,
                    no_bucket=not scenario.requires_storage,
                    bucket=bucket_path,
                    extra_args=scenario.server_args(combo_config),
                    env=env_vars,
                )
                try:
                    print(f"Starting server (port {args.port}) ...", flush=True)
                    srv.start(timeout=30)
                    print(f"Server ready (pid={srv.pid}).\n", flush=True)
                    combo_results = scenario.run(
                        srv.base_url, args.rounds, combo_config,
                    )
                    results.extend(combo_results)
                finally:
                    print("\nStopping server ...", flush=True)
                    srv.stop()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if temp_bucket is not None:
            temp_bucket.cleanup()

    if results:
        print()
        if _is_eval_scenario(args.scenario, results):
            print_eval_report(args.scenario, args.config, config, results,
                              version, hardware)
        elif scenario.report_type == "db" or _is_db_scenario(args.scenario, results):
            print_db_report(args.scenario, args.config, config, args.rounds, results,
                            version, hardware)
        else:
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


def print_db_report(
    scenario_name: str,
    config_name: str | None,
    config: dict,
    rounds: int,
    results: list[dict],
    version: str,
    hardware: str,
) -> None:
    """Print DB performance summary tables."""
    env_vars = config.get("env", {})
    date = datetime.date.today().isoformat()

    console.print()
    console.rule(f"[bold]talu bench · {scenario_name}[/bold]", style="bright_blue")
    console.print()

    info_parts = [version, hardware, date]
    if config_name:
        info_parts.append(f"config={config_name}")
    console.print(f"  [dim]{' · '.join(info_parts)}[/]")

    param_parts = [f"rounds={rounds}"]
    for key in ("requests", "concurrency", "batch_size", "seed_rows"):
        if key in config:
            param_parts.append(f"{key}={config[key]}")
    for k, v in sorted(env_vars.items()):
        param_parts.append(f"{k}={v}")
    console.print(f"  [dim]{' · '.join(param_parts)}[/]")
    console.print()

    by_op: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_op[row.get("op", "unknown")].append(row)

    table = Table(title="DB Performance", box=box.SIMPLE, header_style="bold cyan")
    table.add_column("Operation", style="cyan")
    table.add_column("Concurrency", justify="right")
    table.add_column("Requests", justify="right")
    table.add_column("Success %", justify="right")
    table.add_column("RPS avg", justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("p99 ms", justify="right")

    def _avg(rows: list[dict], key: str) -> float:
        vals = [float(r.get(key, 0.0)) for r in rows]
        return (sum(vals) / len(vals)) if vals else 0.0

    for op in sorted(by_op):
        rows = by_op[op]
        total_requests = int(sum(int(r.get("requests", 0)) for r in rows))
        total_ok = int(sum(int(r.get("ok", 0)) for r in rows))
        success_pct = (100.0 * total_ok / total_requests) if total_requests > 0 else 0.0
        conc_values = sorted({int(r.get("concurrency", 1)) for r in rows})
        conc_label = ",".join(str(v) for v in conc_values)
        table.add_row(
            op,
            conc_label,
            str(total_requests),
            f"{success_pct:.1f}",
            f"{_avg(rows, 'rps'):.1f}",
            f"{_avg(rows, 'p50_ms'):.2f}",
            f"{_avg(rows, 'p95_ms'):.2f}",
            f"{_avg(rows, 'p99_ms'):.2f}",
        )

    console.print(table)
    console.print()


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


# ---------------------------------------------------------------------------
# Eval report formatting (Rich)
# ---------------------------------------------------------------------------

def print_eval_report(
    scenario_name: str,
    config_name: str | None,
    config: dict,
    results: list[dict],
    version: str,
    hardware: str,
) -> None:
    """Print an accuracy-focused eval report and write JSONL log."""
    env_vars = config.get("env", {})
    date = datetime.date.today().isoformat()

    # -- Collect log paths (scenarios log during run(), no double-logging) --
    bench_name = results[0].get("bench", scenario_name.rsplit("/", 1)[-1]) if results else "eval"
    from log import eval_log_path
    log_paths: list[Path] = []
    for r in results:
        r_mrt = int(r.get("max_reasoning_tokens", 0))
        lp = eval_log_path(bench_name, r.get("model_uri", r["model"]),
                           config.get("samples"), r_mrt)
        if lp not in log_paths:
            log_paths.append(lp)

    # -- Header --
    bench_label = results[0].get("bench", scenario_name.rsplit("/", 1)[-1]).upper() if results else scenario_name
    console.print()
    console.rule(f"[bold]talu eval · {bench_label}[/bold]", style="bright_blue")
    console.print()

    info_parts = [version, hardware, date]
    if config_name:
        info_parts.append(f"config={config_name}")
    console.print(f"  [dim]{' · '.join(info_parts)}[/]")

    param_parts = []
    samples = config.get("samples")
    if samples is not None:
        param_parts.append(f"samples={samples}")
    for key in ("seed", "temperature", "max_tokens"):
        if key in config:
            param_parts.append(f"{key}={config[key]}")
    # Show reasoning budgets from actual results (config may be stale after runner loop).
    mrt_vals = sorted(set(r.get("max_reasoning_tokens", 0) for r in results))
    mrt_str = ",".join(str(v) for v in mrt_vals)
    param_parts.append(f"reasoning={mrt_str}")
    for k, v in sorted(env_vars.items()):
        param_parts.append(f"{k}={v}")
    if param_parts:
        console.print(f"  [dim]{' · '.join(param_parts)}[/]")
    console.print()

    # -- Group results by model --
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    for model, rows in by_model.items():
        _print_eval_model_table(console, model, rows, bench_label)

    for lp in log_paths:
        console.print(f"  [dim]Log: {lp}[/]")
    console.print()


def _print_eval_model_table(
    console: Console,
    model: str,
    rows: list[dict],
    bench_label: str = "",
) -> None:
    """Print an accuracy table for one model across precision variants.

    Uses grouped columns similar to the perf report for a consistent look.
    """
    has_perf = any(r.get("avg_gen_tok_s", 0) > 0 for r in rows)
    has_errors = any(r.get("errors", 0) > 0 for r in rows)
    has_ifeval = any("prompt_strict_acc" in r for r in rows)

    # Column definitions: (header, group, right-align).
    col_defs: list[tuple[str, str, bool]] = [
        ("Precision", "",          False),
        ("Size",      "",          True),
        ("Reasoning", "",          True),
    ]
    if has_ifeval:
        col_defs += [
            ("Prompt",      "Accuracy",  True),
            ("Instruction", "Accuracy",  True),
        ]
    else:
        col_defs += [
            ("Score",     "Accuracy",  True),
            ("%",         "Accuracy",  True),
        ]
    if has_perf:
        col_defs += [
            ("tokens",  "Prefill",   True),
            ("avg t/s", "Prefill",   True),
            ("tokens",  "Generate",  True),
            ("avg t/s", "Generate",  True),
        ]
    if has_errors:
        col_defs.append(("Errors", "", True))

    gap_inner = 2
    gap_group = 5

    # Build data rows.
    data_rows: list[list[str]] = []
    for r in rows:
        mi: dict = r.get("model_info", {}) or {}
        # Precision label.
        dtype = mi.get("weight_dtype", "")
        gs = mi.get("gaffine_group_size", 0)
        if dtype and dtype.startswith("GAF") and gs > 0:
            label = f"{dtype}_{gs}"
        elif dtype:
            label = dtype
        else:
            scheme = r.get("scheme", "original")
            label = scheme if scheme != "original" else "n/a"
        # Size.
        fs = mi.get("file_size_bytes", 0)
        if fs >= 1 << 30:
            size_str = f"{fs / (1 << 30):.1f} GB"
        elif fs > 0:
            size_str = f"{fs / (1 << 20):.0f} MB"
        else:
            size_str = "—"
        # Reasoning budget.
        budget = r.get("max_reasoning_tokens", 0)
        reasoning_str = f"{budget:,}" if budget > 0 else "—"
        # Accuracy columns.
        if has_ifeval:
            row_data = [
                label, size_str, reasoning_str,
                f"{r.get('prompt_strict_acc', 0):.1f}%",
                f"{r.get('inst_strict_acc', 0):.1f}%",
            ]
        else:
            correct = r.get("correct_count", 0)
            total = r.get("total", 0)
            pct = r.get("accuracy", 0)
            row_data = [label, size_str, reasoning_str, f"{correct}/{total}", f"{pct:.1f}%"]
        if has_perf:
            # Prefill.
            in_tok = r.get("total_input_tokens", 0)
            row_data.append(f"{in_tok:,}" if in_tok > 0 else "—")
            avg_pre = r.get("avg_prefill_tok_s", 0)
            row_data.append(f"{avg_pre:.1f}" if avg_pre > 0 else "—")
            # Generate.
            out_tok = r.get("total_output_tokens", 0)
            row_data.append(f"{out_tok:,}" if out_tok > 0 else "—")
            avg_gen = r.get("avg_gen_tok_s", 0)
            row_data.append(f"{avg_gen:.1f}" if avg_gen > 0 else "—")
        if has_errors:
            errs = r.get("errors", 0)
            row_data.append(str(errs) if errs > 0 else "—")
        data_rows.append(row_data)

    # Compute per-column gaps: wider between groups, tighter within.
    gaps: list[int] = []
    for i in range(1, len(col_defs)):
        gaps.append(gap_group if col_defs[i][1] != col_defs[i - 1][1] else gap_inner)

    def join_parts(parts: list[str]) -> str:
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
    group_ranges: list[tuple[str, int, int]] = []
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
        inner_gaps = sum(gaps[start:end]) if end > start else 0
        span_width = sum(widths[start:end + 1]) + inner_gaps
        if len(name) > span_width:
            widths[start] += len(name) - span_width

    # Build spanning group line.
    group_line_parts: list[str] = []
    for idx, (name, start, end) in enumerate(group_ranges):
        inner_gaps = sum(gaps[start:end]) if end > start else 0
        w = sum(widths[start:end + 1]) + inner_gaps
        if name:
            side = (w - len(name)) // 2
            span = " " * side + name + " " * (w - side - len(name))
            group_line_parts.append(span)
        else:
            group_line_parts.append(" " * w)
        if idx < len(group_ranges) - 1:
            group_line_parts.append(" " * gaps[end])
    group_line = "".join(group_line_parts)

    # Column headers + separator.
    header_parts = [
        f"{cd[0]:>{w}}" if cd[2] else f"{cd[0]:<{w}}"
        for cd, w in zip(col_defs, widths)
    ]
    header_line = join_parts(header_parts)
    sep_line = join_parts(["─" * w for w in widths])

    # Find column indices for styling.
    if has_ifeval:
        # IFEval: 2 accuracy columns (Prompt %, Instruction %).
        accuracy_indices = {3, 4}
        perf_offset = 5
    else:
        # MCQ evals: single "%" column.
        accuracy_indices = {4}
        perf_offset = 5
    prefill_avg_idx = perf_offset + 1 if has_perf else -1
    gen_avg_idx = perf_offset + 3 if has_perf else -1
    errors_idx = len(col_defs) - 1 if has_errors else -1

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
            elif i in accuracy_indices:
                cell = f"[green bold]{cell}[/]"
            elif i in (prefill_avg_idx, gen_avg_idx):
                cell = f"[green bold]{cell}[/]"
            elif i == errors_idx and val != "—":
                cell = f"[red]{cell}[/]"
            parts.append(cell)
        content_lines.append(join_parts(parts))

    # Compute visible content width (strip Rich markup for measurement).
    import re
    _strip_markup = re.compile(r"\[/?[^\]]*\]").sub
    content_width = max(len(_strip_markup("", line)) for line in content_lines)
    # Build title: model left, bench label right on the top border.
    if bench_label:
        # -6 accounts for border decoration + spaces around the dash fill.
        pad = content_width - len(model) - len(bench_label) - 6
        if pad < 1:
            pad = 1
        title = f"[bold]{model}[/bold] [blue]{'─' * pad}[/] [bold]{bench_label}[/bold]"
    else:
        title = f"[bold]{model}[/bold]"

    panel = Panel(
        "\n".join(content_lines),
        title=title,
        title_align="left",
        border_style="blue",
        padding=(0, 1),
        expand=False,
    )
    console.print(panel)

    # Per-category breakdown for BFCL.
    for r in rows:
        cat_acc = r.get("category_accuracy")
        if not cat_acc:
            continue
        scheme = r.get("scheme", "original")
        label = scheme if scheme != "original" else ""
        prefix = f"  {label}  " if label else "  "
        parts: list[str] = []
        for cat, acc in cat_acc.items():
            parts.append(f"[dim]{cat}[/] [bold]{acc:.1f}%[/]")
        console.print(f"{prefix}{'  '.join(parts)}")

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
    run_p.add_argument("--rounds", "-r", type=int, default=5, help="Rounds per variant (default: 5).")
    run_p.add_argument("--samples", "-n", type=int, default=None,
                        help="Number of eval samples (default: all). For eval scenarios only.")
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
