#!/usr/bin/env python3
"""
Compare reference NPZ against talu NPZ to find divergence.

This tool loads two NPZ files (typically from PyTorch reference and talu)
and compares tensors at each trace point. It reports the first point
where divergence exceeds a threshold, making it easy to pinpoint issues.

Usage:
    uv run python -m compare _reference/qwen3.npz /tmp/talu.npz
    uv run python -m compare _reference/qwen3.npz /tmp/talu.npz --threshold 0.001
"""

import argparse
import sys
from typing import Optional

import numpy as np


def _canonical_key(key: str) -> str:
    """Normalize key naming differences across capture/export tools."""
    out = key
    out = out.replace(".layer_attn_norm", ".attn_norm")
    out = out.replace(".layer_ffn_norm", ".ffn_norm")
    out = out.replace(".block.out", ".block_out")
    return out


def _load_npz_canonical(path: str) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    npz = np.load(path)
    data: dict[str, np.ndarray] = {}
    original: dict[str, str] = {}
    for k in npz.files:
        ck = _canonical_key(k)
        if ck not in data:
            data[ck] = npz[k]
            original[ck] = k
    return data, original


def compare_npz(ref_path: str, talu_path: str, threshold: float = 0.01,
                verbose: bool = True) -> Optional[str]:
    """
    Compare two NPZ files and find first divergence.

    Args:
        ref_path: Path to reference NPZ (from PyTorch)
        talu_path: Path to talu NPZ
        threshold: Maximum allowed absolute difference
        verbose: Print comparison results

    Returns:
        Name of first divergent tensor, or None if all match
    """
    ref, ref_orig = _load_npz_canonical(ref_path)
    talu, talu_orig = _load_npz_canonical(talu_path)

    # Check for missing/extra keys
    ref_keys = set(ref.keys())
    talu_keys = set(talu.keys())

    missing_in_talu = ref_keys - talu_keys
    missing_in_ref = talu_keys - ref_keys

    if missing_in_talu and verbose:
        print(f"WARNING: Missing in talu: {sorted(missing_in_talu)}")
    if missing_in_ref and verbose:
        print(f"WARNING: Missing in ref: {sorted(missing_in_ref)}")

    # Compare common keys in sorted order
    common = sorted(ref_keys & talu_keys, key=_sort_key)

    first_divergence = None

    for key in common:
        ref_tensor = ref[key]
        talu_tensor = talu[key]

        # Check shapes
        if ref_tensor.shape != talu_tensor.shape:
            if verbose:
                print(f"✗ {key}: SHAPE MISMATCH ref={ref_tensor.shape} talu={talu_tensor.shape}")
            if first_divergence is None:
                first_divergence = key
            continue

        # Check for NaN
        if np.any(np.isnan(talu_tensor)):
            if verbose:
                nan_count = np.sum(np.isnan(talu_tensor))
                print(f"✗ {key}: talu has {nan_count} NaN values")
            if first_divergence is None:
                first_divergence = key
            continue

        # Check for Inf
        if np.any(np.isinf(talu_tensor)):
            if verbose:
                inf_count = np.sum(np.isinf(talu_tensor))
                print(f"✗ {key}: talu has {inf_count} Inf values")
            if first_divergence is None:
                first_divergence = key
            continue

        # Compute difference stats
        diff = np.abs(ref_tensor.astype(np.float64) - talu_tensor.astype(np.float64))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        # Also compute relative error for context
        ref_abs = np.abs(ref_tensor.astype(np.float64))
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.where(ref_abs > 1e-8, diff / ref_abs, 0)
            max_rel_diff = np.max(rel_diff)

        if max_diff > threshold:
            if verbose:
                ref_name = ref_orig.get(key, key)
                talu_name = talu_orig.get(key, key)
                print(f"✗ {key:30s} max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  max_rel={max_rel_diff:.4f}")
                if ref_name != key or talu_name != key:
                    print(f"    mapped: ref={ref_name}  talu={talu_name}")
                # Show first few values for debugging
                ref_flat = ref_tensor.flatten()
                talu_flat = talu_tensor.flatten()
                print(f"    ref:  {ref_flat[:5]}...")
                print(f"    talu: {talu_flat[:5]}...")
                # Find location of max difference
                max_idx = np.argmax(diff)
                max_loc = np.unravel_index(max_idx, diff.shape)
                print(f"    max diff at {max_loc}: ref={ref_tensor[max_loc]:.6f} talu={talu_tensor[max_loc]:.6f}")
            if first_divergence is None:
                first_divergence = key
        else:
            if verbose:
                print(f"✓ {key:30s} max_diff={max_diff:.2e}")

    # Summary
    if verbose:
        print()
        if first_divergence:
            print(f"FIRST DIVERGENCE: {first_divergence}")
            print(f"\nThis is the first trace point where talu output differs from PyTorch.")
            print(f"Investigate the computation leading to this point.")
        else:
            print("All tensors match within threshold")

    return first_divergence


def _sort_key(key: str) -> tuple:
    """Sort keys by layer number, then by point."""
    if key.startswith("layer"):
        parts = key.split(".")
        layer = int(parts[0][5:])
        point = parts[1] if len(parts) > 1 else ""
        # Order within layer: attn_norm, attn_out, ffn_norm, ffn_down, block_out
        point_order = {
            "attn_norm": 0, "attn_out": 1,
            "ffn_norm": 2, "ffn_down": 3,
            "block_out": 4
        }
        return (1, layer, point_order.get(point, 99))
    elif key == "embed":
        return (0, 0, 0)
    elif key == "final_norm":
        return (2, 0, 0)
    elif key == "lm_head":
        return (2, 0, 1)
    else:
        return (3, 0, key)


def main():
    parser = argparse.ArgumentParser(
        description="Compare reference NPZ against talu NPZ to find divergence."
    )
    parser.add_argument(
        "ref",
        help="Reference NPZ file (from PyTorch)"
    )
    parser.add_argument(
        "talu",
        help="talu NPZ file"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.01,
        help="Divergence threshold (default: 0.01)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Only output first divergence (no per-tensor output)"
    )
    args = parser.parse_args()

    divergence = compare_npz(args.ref, args.talu, args.threshold, verbose=not args.quiet)

    if args.quiet and divergence:
        print(divergence)

    sys.exit(1 if divergence else 0)


if __name__ == "__main__":
    main()
