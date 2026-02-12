#!/usr/bin/env python3
"""
Run end-to-end model parity diagnosis:
1) capture PyTorch reference NPZ
2) dump talu NPZ
3) compare and report first divergence

Usage:
    uv run python -m diagnose qwen3_moe --prompt "Hello"
    uv run python -m diagnose qwen3_moe --model-id Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
        --talu-model /path/to/local/model --prompt "Hello"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from capture import capture_reference
from compare import compare_npz
from trace import ARCHITECTURES


def _default_model_id(arch: str) -> str:
    info = ARCHITECTURES.get(arch)
    if not info or not info.get("model_ids"):
        raise ValueError(f"No default model_id for architecture: {arch}")
    return info["model_ids"][0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture reference NPZ + talu dump NPZ + compare in one command."
    )
    parser.add_argument("arch", help="Architecture name from tools/archs/trace.py")
    parser.add_argument("--prompt", default="Hello", help="Prompt text (must match on both sides)")
    parser.add_argument("--model-id", help="Model ID/path used by PyTorch capture (defaults from arch registry)")
    parser.add_argument(
        "--talu-model",
        help="Model path/ID passed to talu-dump (defaults to --model-id)",
    )
    parser.add_argument(
        "--ref-npz",
        help="Output path for reference NPZ (default: /tmp/ref_<arch>.npz)",
    )
    parser.add_argument(
        "--talu-npz",
        help="Output path for talu NPZ (default: /tmp/talu_<arch>.npz)",
    )
    parser.add_argument(
        "--dump-bin",
        default="./zig-out/bin/talu-dump",
        help="Path to talu-dump binary",
    )
    parser.add_argument("--tokens", type=int, default=1, help="Max tokens for talu-dump (default: 1)")
    parser.add_argument("--threshold", type=float, default=0.01, help="compare.py threshold")
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable tokenizer.apply_chat_template during reference capture",
    )
    parser.add_argument("--skip-capture", action="store_true", help="Skip reference capture")
    parser.add_argument("--skip-dump", action="store_true", help="Skip talu dump")
    parser.add_argument("--skip-compare", action="store_true", help="Skip NPZ compare")
    args = parser.parse_args()

    if args.arch not in ARCHITECTURES:
        print(f"Unknown architecture: {args.arch}")
        print(f"Available: {', '.join(sorted(ARCHITECTURES.keys()))}")
        return 2

    model_id = args.model_id or _default_model_id(args.arch)
    talu_model = args.talu_model or model_id
    ref_npz = args.ref_npz or f"/tmp/ref_{args.arch}.npz"
    talu_npz = args.talu_npz or f"/tmp/talu_{args.arch}.npz"

    print(f"[diag] arch={args.arch}")
    print(f"[diag] prompt={args.prompt!r}")
    print(f"[diag] model_id={model_id}")
    print(f"[diag] talu_model={talu_model}")
    print(f"[diag] ref_npz={ref_npz}")
    print(f"[diag] talu_npz={talu_npz}")

    if not args.skip_capture:
        print("[diag] capturing PyTorch reference NPZ...")
        capture = capture_reference(
            args.arch,
            args.prompt,
            model_id=model_id,
            use_chat_template=not args.no_chat_template,
        )
        Path(ref_npz).parent.mkdir(parents=True, exist_ok=True)
        capture.save(ref_npz)

    if not args.skip_dump:
        print("[diag] running talu-dump...")
        cmd = [
            args.dump_bin,
            "-m",
            talu_model,
            "-p",
            args.prompt,
            "-o",
            talu_npz,
            "-n",
            str(args.tokens),
        ]
        subprocess.run(cmd, check=True)

    if args.skip_compare:
        return 0

    print("[diag] comparing NPZ files...")
    divergence = compare_npz(ref_npz, talu_npz, threshold=args.threshold, verbose=True)
    return 1 if divergence else 0


if __name__ == "__main__":
    raise SystemExit(main())
