#!/usr/bin/env python3
"""Pretty-print cargo-criterion benchmark results.

Usage:
    cargo criterion --output-format quiet --message-format json 2>/dev/null \
        | python3 scripts/bench-summary.py

For rigorous statistical benchmarks (slow, ~5 min):
    cargo bench
"""
import json
import sys


def format_ns(ns: float) -> str:
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.2f} µs"
    return f"{ns:.0f} ns"


def main() -> None:
    rows: list[tuple[str, str]] = []

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") != "benchmark-complete":
            continue

        name = msg["id"]
        estimate = msg["median"]["estimate"]
        unit = msg["median"]["unit"]

        if unit == "ns":
            display = format_ns(estimate)
        else:
            display = f"{estimate:.2f} {unit}"

        rows.append((name, display))

    if not rows:
        print("No benchmark results found.", file=sys.stderr)
        sys.exit(1)

    name_w = max(len(r[0]) for r in rows)
    val_w = max(len(r[1]) for r in rows)

    print(f"{'Benchmark':<{name_w}}  {'Median':>{val_w}}")
    print(f"{'-' * name_w}  {'-' * val_w}")
    for name, val in rows:
        print(f"{name:<{name_w}}  {val:>{val_w}}")


if __name__ == "__main__":
    main()
