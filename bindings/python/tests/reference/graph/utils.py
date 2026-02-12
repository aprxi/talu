import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add tools/archs to path so 'trace' module and arch submodules can be imported.
# Path: bindings/python/tests/reference/graph/utils.py -> parents[5] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[5]
_ARCHS_ROOT = _REPO_ROOT / "tools" / "archs"
if str(_ARCHS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ARCHS_ROOT))


def get_arch_definition(arch: str) -> dict[str, Any]:
    import trace as trace_mod

    if arch not in trace_mod.ARCHITECTURES:
        raise ValueError(f"Unknown arch: {arch}")
    return trace_mod.ARCHITECTURES[arch]


def get_block_class(arch: str):
    arch_def = get_arch_definition(arch)
    module = importlib.import_module(arch_def["module"])
    return getattr(module, arch_def["block_class"])


def block_kwargs_for_arch(arch: str, config: dict[str, Any]) -> dict[str, Any]:
    import trace as trace_mod

    block_class = get_block_class(arch)
    return trace_mod._block_kwargs(block_class, config)


def write_graph_for_arch(
    out_dir: Path,
    arch: str,
    model_type: str,
    block_config: dict[str, Any],
    primitives_only: bool,
) -> Path:
    import trace as trace_mod

    block_class = get_block_class(arch)
    prev_mode = trace_mod.PRIMITIVES_ONLY
    trace_mod.PRIMITIVES_ONLY = primitives_only
    try:
        block_ops = trace_mod.trace_block(block_class, block_config)
    finally:
        trace_mod.PRIMITIVES_ONLY = prev_mode

    # Get eps from block_config if available
    eps = block_config.get("eps", 1e-6)

    graph = {
        "name": model_type,
        "model_types": [model_type],
        # Pre-block: embedding lookup
        "pre_block": [
            {
                "op": "embedding",
                "inputs": [{"tensor": "input_ids"}],
                "outputs": ["_t0"],
            }
        ],
        # Transformer block operations
        "block": block_ops,
        # Post-block: final layer norm
        "post_block": [
            {
                "op": "norm",
                "inputs": [{"tensor": "_t_last"}],
                "outputs": ["_t_out"],
                "eps": eps,
            }
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_type}.json"
    with open(path, "w") as f:
        json.dump(graph, f, indent=2)
        f.write("\n")
    return path


def build_block_weights(
    block_class,
    block_config: dict[str, Any],
    prefix: str,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Build weights for a transformer block.

    Uses small random values for projection weights to ensure the model
    actually computes something meaningful (not just zeros).
    """
    rng = np.random.default_rng(seed)
    block = block_class(**block_config)
    weights: dict[str, np.ndarray] = {}
    for name, param in block.named_parameters():
        shape = param.shape
        if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
            # Norm weights should be ~1.0
            arr = np.ones(shape, dtype=np.float32)
        else:
            # Use small random weights (Xavier-like initialization)
            # Scale by 1/sqrt(fan_in) to keep activations reasonable
            fan_in = shape[-1] if len(shape) > 1 else shape[0]
            scale = 1.0 / np.sqrt(fan_in)
            arr = rng.normal(0, scale, size=shape).astype(np.float32)
        weights[f"{prefix}{name}"] = arr
    return weights


def llama2_block_config(config: dict[str, Any]) -> dict[str, Any]:
    hidden_size = config["hidden_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    head_dim = config.get("head_dim", hidden_size // num_heads)
    return {
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": config["intermediate_size"],
        "rope_theta": config.get("rope_theta", 10000.0),
        "eps": config.get("rms_norm_eps", 1e-6),
    }


def write_llama2_graph(
    out_dir: Path,
    model_type: str,
    block_config: dict[str, Any],
    primitives_only: bool,
) -> Path:
    import trace as trace_mod

    from llama.llama2 import Block

    prev_mode = trace_mod.PRIMITIVES_ONLY
    trace_mod.PRIMITIVES_ONLY = primitives_only
    try:
        block_ops = trace_mod.trace_block(Block, block_config)
    finally:
        trace_mod.PRIMITIVES_ONLY = prev_mode

    # Get eps from block_config if available
    eps = block_config.get("eps", 1e-6)

    graph = {
        "name": model_type,
        "model_types": [model_type],
        # Pre-block: embedding lookup
        "pre_block": [
            {
                "op": "embedding",
                "inputs": [{"tensor": "input_ids"}],
                "outputs": ["_t0"],
            }
        ],
        # Transformer block operations
        "block": block_ops,
        # Post-block: final layer norm
        "post_block": [
            {
                "op": "norm",
                "inputs": [{"tensor": "_t_last"}],
                "outputs": ["_t_out"],
                "eps": eps,
            }
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_type}.json"
    with open(path, "w") as f:
        json.dump(graph, f, indent=2)
        f.write("\n")
    return path
