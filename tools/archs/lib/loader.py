"""
Fast model loading utilities.

Provides direct safetensors loading which is faster than HuggingFace's
snapshot_download + AutoModel pattern. Useful for inference and validation.
"""

import json
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file


def load_config(model_path: str, config_name: str = "config.json") -> dict:
    """Load model config from a directory or HuggingFace cache path.

    Args:
        model_path: Path to model directory or HuggingFace cache snapshot
        config_name: Name of config file (default: config.json)

    Returns:
        Config dictionary
    """
    path = Path(model_path)
    config_path = path / config_name

    if not config_path.exists():
        # Try HuggingFace cache structure
        snapshots = path / "snapshots"
        if snapshots.exists():
            # Get most recent snapshot
            snapshot_dirs = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if snapshot_dirs:
                config_path = snapshot_dirs[0] / config_name

    with open(config_path) as f:
        return json.load(f)


def find_safetensor_files(model_path: str) -> tuple[Path, list[str]]:
    """Find safetensor files in a model directory.

    Args:
        model_path: Path to model directory

    Returns:
        Tuple of (base_path, list of safetensor filenames)
    """
    path = Path(model_path)

    # Check for index file (sharded model)
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        files = sorted(set(weight_map.values()))
        return path, files

    # Single file model
    single_file = path / "model.safetensors"
    if single_file.exists():
        return path, ["model.safetensors"]

    # Try HuggingFace cache structure
    snapshots = path / "snapshots"
    if snapshots.exists():
        snapshot_dirs = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if snapshot_dirs:
            return find_safetensor_files(str(snapshot_dirs[0]))

    raise FileNotFoundError(f"No safetensor files found in {model_path}")


def load_safetensors_fast(
    model: nn.Module,
    model_path: str,
    *,
    pre_load_hook: Optional[Callable[[nn.Module], None]] = None,
    weight_filter: Optional[Callable[[str, torch.Tensor, nn.Module], Optional[torch.Tensor]]] = None,
    verbose: bool = True,
) -> None:
    """Load safetensor weights directly into a model.

    This is faster than HuggingFace's from_pretrained because it:
    - Skips snapshot_download overhead
    - Uses assign=True for efficient weight loading
    - Loads shards sequentially to minimize memory

    Args:
        model: The model to load weights into
        model_path: Path to model directory containing safetensor files
        pre_load_hook: Optional function called before loading (e.g., to register buffers)
        weight_filter: Optional function to filter/transform weights before loading.
                       Called with (name, tensor, model) -> tensor or None to skip.
        verbose: Print progress messages
    """
    base_path, files = find_safetensor_files(model_path)

    if verbose:
        print(f"  Found {len(files)} checkpoint files to load")

    if pre_load_hook:
        pre_load_hook(model)

    for i, filename in enumerate(files):
        filepath = base_path / filename
        if verbose:
            print(f"  [{i + 1}/{len(files)}] Loading {filename}...")

        file_weights = load_file(str(filepath))

        if verbose:
            print(f"  [{i + 1}/{len(files)}] Loaded {len(file_weights)} tensors")

        # Apply weight filter if provided
        if weight_filter:
            filtered_weights = {}
            for name, tensor in list(file_weights.items()):
                result = weight_filter(name, tensor, model)
                if result is not None:
                    filtered_weights[name] = result
            file_weights = filtered_weights

        # Load remaining weights
        model.load_state_dict(file_weights, strict=False, assign=True)

        if verbose:
            print(f"  [{i + 1}/{len(files)}] Assigned weights")

        del file_weights

    if verbose:
        print("  All weights loaded")


def apply_post_load_hooks(model: nn.Module) -> None:
    """Call post_load_hook() on all modules that have one.

    Useful for finalizing quantization state after weights are loaded.
    """
    for module in model.modules():
        hook = getattr(module, "post_load_hook", None)
        if callable(hook):
            hook()


# === GPT-OSS specific helpers (can be used as examples for other models) ===

def gpt_oss_pre_load_hook(model: nn.Module) -> None:
    """Ensure Expert modules have required buffers before loading."""
    from .moe import Experts

    for _, module in model.named_modules():
        if isinstance(module, Experts):
            for buf_name in ["gate_up_proj_blocks", "gate_up_proj_scales",
                           "down_proj_blocks", "down_proj_scales"]:
                if not hasattr(module, buf_name) or getattr(module, buf_name) is None:
                    use_uint8 = "blocks" in buf_name or "scales" in buf_name
                    module.register_buffer(
                        buf_name,
                        torch.empty(0, dtype=torch.uint8 if use_uint8 else torch.bfloat16)
                    )


def gpt_oss_weight_filter(name: str, tensor: torch.Tensor, model: nn.Module) -> Optional[torch.Tensor]:
    """Handle Expert buffer weights specially for GPT-OSS."""
    from .moe import Experts

    # Check if this is an Expert buffer weight
    expert_buffers = ["gate_up_proj_blocks", "gate_up_proj_scales",
                     "down_proj_blocks", "down_proj_scales"]

    for buf_name in expert_buffers:
        if name.endswith(buf_name):
            # Find the module and set directly
            module_name = name.rsplit(".", 1)[0]
            parts = module_name.split(".")
            module = model
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break

            if isinstance(module, Experts):
                setattr(module, buf_name, tensor)
                return None  # Don't include in load_state_dict

    return tensor  # Normal weight, include in load_state_dict
