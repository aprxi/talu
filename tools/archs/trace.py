"""
Generate compute graphs from Python model definitions using torch.fx tracing.

This script imports each model, traces the Block.forward() using torch.fx,
and converts the traced graph to JSON for the Zig runtime.

Usage:
    python -m talu.trace qwen3
    python -m talu.trace --all
"""

import json
import importlib
import inspect
import operator
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import torch.fx as fx

from lib.utils import load_config

# Environment variable to force primitives-only mode
# When set, all ops are traced into primitives (no fused ops)
# Useful for debugging or when models have non-standard attention/mlp
PRIMITIVES_ONLY = os.environ.get("TALU_PRIMITIVES_ONLY", "0") == "1"


# =============================================================================
# SDPA Helpers (for muP and primitive path support)
# =============================================================================

def is_sdpa_node(node: fx.Node) -> bool:
    """
    Check if node is a scaled_dot_product_attention call.

    FX targets can be Python function objects, torch._ops.OpOverload, or other
    callables. We check multiple conditions to handle all cases.
    """
    if node.op != "call_function":
        return False
    t = node.target
    # Direct function reference
    if t is torch.nn.functional.scaled_dot_product_attention:
        return True
    # Check __name__ attribute
    if getattr(t, "__name__", "") == "scaled_dot_product_attention":
        return True
    # Check string representation (covers OpOverload forms)
    if "scaled_dot_product_attention" in str(t):
        return True
    return False


def get_sdpa_param(node: fx.Node, param_name: str, pos_index: int, default=None):
    """
    Get SDPA parameter from kwargs or positional args.

    PyTorch SDPA signature:
        scaled_dot_product_attention(query, key, value, attn_mask=None,
            dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False)

    Parameters may arrive as positional args OR kwargs.
    """
    # Try kwargs first
    if param_name in node.kwargs:
        return node.kwargs[param_name]
    # Fall back to positional args
    if len(node.args) > pos_index:
        return node.args[pos_index]
    return default


def extract_sdpa_scale(node: fx.Node, warn_state: dict) -> Optional[float]:
    """
    Extract SDPA scale with proper error handling.

    Policy:
    - Normal mode: Warn once if scale is present but non-constant
    - PRIMITIVES_ONLY mode: Raise error (this mode exists to verify correctness)
    """
    # Check BOTH kwargs and positional args (scale is arg[6])
    scale = get_sdpa_param(node, "scale", 6, None)
    if scale is None:
        return None

    try:
        if hasattr(scale, "item"):
            return float(scale.item())
        return float(scale)
    except (TypeError, RuntimeError, ValueError) as e:
        if PRIMITIVES_ONLY:
            # Strict mode: non-constant scale is an error
            raise ValueError(
                f"SDPA scale must be a constant in PRIMITIVES_ONLY mode, "
                f"got {type(scale).__name__}: {e}"
            )
        # Normal mode: warn once PER TRACE RUN
        if not warn_state.get("scale_warned"):
            import warnings
            warnings.warn(
                f"SDPA scale is non-constant ({type(scale).__name__}), "
                f"using default 1/sqrt(head_dim). This may affect muP correctness."
            )
            warn_state["scale_warned"] = True
        return None


def check_sdpa_enable_gqa(node: fx.Node, warn_state: dict):
    """
    Check enable_gqa and warn if True.

    Our runtime handles GQA via head count ratios, not this flag.
    """
    enable_gqa = get_sdpa_param(node, "enable_gqa", 7, False)
    if enable_gqa and not warn_state.get("enable_gqa_warned"):
        import warnings
        warnings.warn(
            "SDPA enable_gqa=True is set but ignored. "
            "GQA is handled automatically via head count configuration."
        )
        warn_state["enable_gqa_warned"] = True


# =============================================================================
# Mamba Helpers (for Mamba2/SSM tracing)
# =============================================================================

def extract_mamba_config(submod: nn.Module, node: fx.Node) -> dict:
    """
    Extract Mamba config with required attribute validation.

    Required attributes error if missing (don't silently default to 0).
    Also validates d_inner consistency: expand * d_model == n_heads * d_head.
    """
    # REQUIRED attributes - error if missing
    REQUIRED_ATTRS = ["d_model", "d_state", "d_conv", "n_heads", "d_head"]
    config = {}

    for attr in REQUIRED_ATTRS:
        val = getattr(submod, attr, None)
        if val is None:
            raise ValueError(
                f"Mamba module '{node.target}' missing required attribute '{attr}'. "
                f"Ensure @fusable(kernel='mamba') is on the correct module."
            )
        # Normalize to int for JSON serialization
        config[attr] = int(val.item() if hasattr(val, "item") else val)

    # OPTIONAL attributes with sensible defaults
    n_groups = getattr(submod, "n_groups", 1)
    config["n_groups"] = int(n_groups.item() if hasattr(n_groups, "item") else n_groups)

    expand = getattr(submod, "expand", 2)
    config["expand"] = int(expand.item() if hasattr(expand, "item") else expand)

    # CONSISTENCY CHECK: d_inner can be derived two ways
    # d_inner = expand * d_model  (from expansion factor)
    # d_inner = n_heads * d_head  (from head structure)
    # These MUST match or the kernel config is inconsistent
    d_inner_from_expand = config["expand"] * config["d_model"]
    d_inner_from_heads = config["n_heads"] * config["d_head"]

    if d_inner_from_expand != d_inner_from_heads:
        raise ValueError(
            f"Mamba d_inner inconsistency: "
            f"expand * d_model = {config['expand']} * {config['d_model']} = {d_inner_from_expand}, "
            f"but n_heads * d_head = {config['n_heads']} * {config['d_head']} = {d_inner_from_heads}. "
            f"These must be equal."
        )

    # Emit d_inner explicitly (avoids Zig kernel needing to derive it)
    config["d_inner"] = d_inner_from_heads

    return config

# =============================================================================
# Architecture Registry
# =============================================================================

ARCHITECTURES = {
    "qwen3": {
        "name": "qwen3",
        "model_types": ["qwen3", "qwen3_vl", "qwen2.5", "qwen2", "qwen"],
        "module": "qwen.qwen3",
        "block_class": "Block",
        "model_ids": ["Qwen/Qwen3-0.6B"],
        "eps": 1e-6,
    },
    "qwen3_moe": {
        "name": "qwen3_moe",
        "model_types": ["qwen3_moe"],
        "module": "qwen.qwen3_moe",
        "block_class": "Block",
        "model_ids": ["Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"],
        "eps": 1e-6,
    },
    "qwen3_next": {
        "name": "qwen3_next",
        "model_types": ["qwen3_next"],
        "module": "qwen.qwen3_next",
        "block_class": "HybridBlock",
        "model_ids": ["Qwen/Qwen3-Coder-Next-FP8"],
        "eps": 1e-6,
        "heterogeneous": True,
        "layer_types_key": "layer_types",
    },
    "llama2": {
        "name": "llama2",
        "model_types": ["llama2", "mistral", "yi", "vicuna", "tinyllama"],
        "module": "llama.llama2",
        "block_class": "Block",
        "model_ids": ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
        "eps": 1e-6,
    },
    "llama3": {
        "name": "llama3",
        "model_types": ["llama", "llama3", "llama3.1", "llama3.2", "idefics3"],
        "module": "llama.llama3",
        "block_class": "Block",
        "model_ids": ["meta-llama/Llama-3.2-1B"],
        "eps": 1e-5,
        "weight_prefixes": [
            "model.layers.{d}.",
            "model.text_model.layers.{d}.",
            "layers.{d}.",
            "transformer.h.{d}.",
            "backbone.layers.{d}.",
            "language_model.model.layers.{d}.",
        ],
        "global_weights": [
            {
                "id": "token_embeddings",
                "candidates": [
                    "model.embed_tokens.weight",
                    "model.text_model.embed_tokens.weight",
                    "embed_tokens.weight",
                    "transformer.wte.weight",
                    "backbone.embedding.weight",
                    "language_model.model.embed_tokens.weight",
                ],
                "module_type": "Embedding",
                "layout": "embedding",
                "dtype": "float32",
                "required": True,
            },
            {
                "id": "ln_final",
                "candidates": [
                    "model.norm.weight",
                    "model.text_model.norm.weight",
                    "norm.weight",
                    "transformer.ln_f.weight",
                    "backbone.norm.weight",
                    "language_model.model.norm.weight",
                    "model.embedding_norm.weight",
                ],
                "module_type": "RMSNorm",
                "layout": "none",
                "dtype": "float32",
                "required": True,
            },
            {
                "id": "lm_head",
                "candidates": [
                    "lm_head.weight",
                    "model.text_model.lm_head.weight",
                    "output.weight",
                    "transformer.lm_head.weight",
                    "language_model.lm_head.weight",
                ],
                "module_type": "Linear",
                "layout": "linear",
                "dtype": "float32",
                "required": False,
            },
        ],
    },
    "gemma3": {
        "name": "gemma3",
        "model_types": ["gemma3", "gemma3_text", "gemma2", "gemma"],
        "module": "gemma.gemma3",
        "block_class": "Block",
        "model_ids": ["google/gemma-3-1b-it"],
        "eps": 1e-6,
        "embedding_scale": 45.254833995939045,
    },
    "phi4": {
        "name": "phi",
        "model_types": ["phi3", "phi4", "phi"],
        "module": "phi.phi4",
        "block_class": "Block",
        "model_ids": ["microsoft/Phi-4-mini-instruct"],
        "eps": 1e-5,
    },
    "granite3": {
        "name": "granite3",
        "model_types": ["granite"],
        "module": "granite.granite3",
        "block_class": "Block",
        "model_ids": ["ibm-granite/granite-3.3-2b-instruct"],
        "eps": 1e-5,
    },
    "ministral3": {
        "name": "ministral3",
        "model_types": ["ministral3", "mistral3"],
        "module": "mistral.ministral3",
        "block_class": "Block",
        "model_ids": ["mistralai/Ministral-3-3B-Base-2512"],
        "eps": 1e-5,
    },
    "granite_hybrid": {
        "name": "granite_hybrid",
        "model_types": ["granite_hybrid", "granitehybrid", "granitemoehybrid"],
        "module": "granite.granite_hybrid",
        "block_class": "HybridBlock",
        "model_ids": ["ibm-granite/granite-4.0-h-350m"],
        "eps": 1e-5,
        # Heterogeneous: trace both "mamba" and "attention" variants
        "heterogeneous": True,
        "layer_types_key": "layer_types",  # config key that determines layer type per layer
    },
    "lfm2": {
        "name": "lfm2",
        "model_types": ["lfm2"],
        "module": "lfm2.lfm2",
        "block_class": "HybridBlock",
        "model_ids": ["LiquidAI/LFM2-350M"],
        "eps": 1e-5,
        # Heterogeneous: trace both "shortconv" and "attention" variants
        "heterogeneous": True,
        "layer_types_key": "full_attn_idxs",  # config key that determines layer type per layer
    },
    "lfm2_5": {
        "name": "lfm2_5",
        "model_types": ["lfm2_5"],
        "module": "lfm2.lfm2_5",
        "block_class": "HybridBlock",
        "model_ids": ["LiquidAI/LFM2.5-1.2B-Thinking"],
        "eps": 1e-5,
        # Heterogeneous: trace both "conv" and "full_attention" variants
        "heterogeneous": True,
        "layer_types_key": "layer_types",
    },
    "gpt_oss": {
        "name": "gpt_oss",
        "model_types": ["gpt_oss"],
        "module": "gpt_oss.gpt_oss",
        "block_class": "HybridBlock",
        "model_ids": ["openai/gpt-oss-20b"],
        "eps": 1e-5,
        # Heterogeneous: sliding_attention and full_attention layers
        "heterogeneous": True,
        "layer_types_key": "layer_types",
    },
    "youtu_vl": {
        "name": "youtu_vl",
        "model_types": ["youtu_vl"],
        "module": "youtu_vl.youtu_vl",
        "block_class": "Block",
        "model_ids": ["tencent/Youtu-VL-4B-Instruct"],
        "eps": 1e-6,
        # VLM: uses different weight prefixes
        "weight_prefixes": [
            "model.layers.{d}.",
            "language_model.model.layers.{d}.",
        ],
    },
    "minilm": {
        "name": "minilm",
        "model_types": ["bert", "minilm"],
        "module": "bert.minilm",
        "block_class": "Block",
        "model_ids": ["sentence-transformers/all-MiniLM-L6-v2"],
        "eps": 1e-12,
        # BERT uses different weight prefixes than decoder LLMs
        "weight_prefixes": [
            "bert.encoder.layer.{d}.",
            "encoder.layer.{d}.",
        ],
        # BERT embedding model: 3 embedding tables + embedding LayerNorm, no lm_head
        "pre_block": [
            {"op": "embedding", "inputs": [{"tensor": "input_ids"}], "outputs": ["_t0"],
             "name": "word_embeddings"},
            {"op": "embedding", "inputs": [{"tensor": "position_ids"}], "outputs": ["_t1"],
             "name": "position_embeddings"},
            {"op": "embedding", "inputs": [{"tensor": "token_type_ids"}], "outputs": ["_t2"],
             "name": "token_type_embeddings"},
            {"op": "add", "inputs": [{"tensor": "_t0"}, {"tensor": "_t1"}], "outputs": ["_t3"]},
            {"op": "add", "inputs": [{"tensor": "_t3"}, {"tensor": "_t2"}], "outputs": ["_t4"]},
            {"op": "norm", "inputs": [{"tensor": "_t4"}], "outputs": ["_t5"], "eps": 1e-12},
        ],
        # Embedding model: no final norm or lm_head projection
        "post_block": [],
        "global_weights": [
            {
                "id": "token_embeddings",
                "candidates": [
                    "bert.embeddings.word_embeddings.weight",
                    "embeddings.word_embeddings.weight",
                ],
                "module_type": "Embedding",
                "layout": "embedding",
                "dtype": "float32",
                "required": True,
            },
            {
                "id": "position_embeddings",
                "candidates": [
                    "bert.embeddings.position_embeddings.weight",
                    "embeddings.position_embeddings.weight",
                ],
                "module_type": "Embedding",
                "layout": "embedding",
                "dtype": "float32",
                "required": True,
            },
            {
                "id": "token_type_embeddings",
                "candidates": [
                    "bert.embeddings.token_type_embeddings.weight",
                    "embeddings.token_type_embeddings.weight",
                ],
                "module_type": "Embedding",
                "layout": "embedding",
                "dtype": "float32",
                "required": True,
            },
            {
                "id": "embedding_ln",
                "candidates": [
                    "bert.embeddings.LayerNorm.weight",
                    "embeddings.LayerNorm.weight",
                ],
                "module_type": "LayerNorm",
                "layout": "none",
                "dtype": "float32",
                "required": True,
            },
            {
                "id": "embedding_ln_bias",
                "candidates": [
                    "bert.embeddings.LayerNorm.bias",
                    "embeddings.LayerNorm.bias",
                ],
                "module_type": "LayerNorm",
                "layout": "none",
                "dtype": "float32",
                "required": True,
            },
        ],
    },
}


# =============================================================================
# FX Graph to JSON conversion
# =============================================================================

def fx_to_ops(graph_module: fx.GraphModule, module: nn.Module) -> List[Dict[str, Any]]:
    """
    Convert a torch.fx GraphModule to our JSON op format.

    We only emit ops that Zig needs to execute:
    - linear (with weight name)
    - norm (with eps)
    - split (for fused projections)
    - scaled_dot_product_attention
    - silu/gelu (activations)
    - add/mul (residual connections)

    We skip internal ops that Zig handles implicitly:
    - view/reshape/transpose (Zig handles internally)
    - RoPE computations (Zig has RoPE built into attention)
    - Internal norm calculations (rsqrt, pow, mean)
    """
    ops = []
    tensor_map = {}  # Maps fx node names to our tensor names
    tensor_counter = [0]  # Mutable counter
    split_counter = [0]  # Counter for split outputs

    # Warn-once state scoped to THIS trace run (not module-level!)
    warn_state = {
        "scale_warned": False,
        "enable_gqa_warned": False,
    }

    def get_tensor_name(node_name: str) -> str:
        """Get or create a tensor name for a node."""
        if node_name not in tensor_map:
            tensor_map[node_name] = f"_t{tensor_counter[0]}"
            tensor_counter[0] += 1
        return tensor_map[node_name]

    def get_input_ref(node) -> Dict[str, Any]:
        """Convert fx node to input reference."""
        if isinstance(node, fx.Node):
            name = tensor_map.get(node.name, get_tensor_name(node.name))
            return {"tensor": name}
        elif isinstance(node, (int, float)):
            return {"scalar": float(node)}
        else:
            return {"scalar": 0}  # Fallback

    def get_module_name(target: str) -> str:
        """Extract clean module name from fx target."""
        return target

    def get_submodule(target: str) -> nn.Module:
        """Get submodule from target path."""
        parts = target.split(".")
        mod = module
        for p in parts:
            mod = getattr(mod, p)
        return mod

    # Track the "main" tensor flow (skip internal computations)
    main_tensor = None

    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            tensor_map[node.name] = node.name
            main_tensor = node.name

        elif node.op == "call_module":
            target = node.target
            submod = get_submodule(target)
            mod_name = get_module_name(target)

            if getattr(submod, "_fusable_kernel", None) == "moe":
                input_node = node.args[0]
                out_name = get_tensor_name(node.name)
                op = {
                    "op": "moe",
                    "inputs": [get_input_ref(input_node)],
                    "outputs": [out_name],
                    "num_experts": getattr(submod, "num_experts", 0),
                    "experts_per_token": getattr(submod, "num_experts_per_tok", 0),
                }
                # Check for SwiGLU variant (alpha=1.702, clipping, (up+1) formulation)
                # Used by GPT-OSS experts
                experts = getattr(submod, "experts", None)
                if experts and hasattr(experts, "ALPHA") and experts.ALPHA == 1.702:
                    op["activation"] = "swiglu_oss"
                ops.append(op)
                tensor_map[node.name] = out_name
                continue

            if getattr(submod, "_fusable_kernel", None) == "norm":
                # Fused norm op - emits single op instead of pow/mean/rsqrt/mul primitives
                input_node = node.args[0]
                out_name = get_tensor_name(node.name)

                # Get weight name from module path
                weight_name = f"{mod_name}.weight"

                # Check for weight_offset (Gemma-style norms use 1 + weight)
                weight_offset = getattr(submod, "weight_offset", 0.0)

                op = {
                    "op": "norm",
                    "inputs": [get_input_ref(input_node), {"tensor": weight_name}],
                    "outputs": [out_name],
                    "eps": getattr(submod, "eps", 1e-6),
                    "name": mod_name,
                }
                if weight_offset != 0.0:
                    op["weight_offset"] = weight_offset

                ops.append(op)
                tensor_map[node.name] = out_name
                continue

            if getattr(submod, "_fusable_kernel", None) == "attention":
                # Fused attention op
                input_node = node.args[0]
                out_name = get_tensor_name(node.name)

                # Check for config flags
                config = getattr(submod, "_fusable_config", [])
                has_qk_norm = "qk_norm" in config
                has_fused_qkv = "fused_qkv" in config
                has_mla = "mla" in config

                op = {
                    "op": "multihead_attention",
                    "inputs": [get_input_ref(input_node)],
                    "outputs": [out_name],
                    "qk_norm": has_qk_norm,
                    "fused_qkv": has_fused_qkv,
                }

                # MLA (Multi-Latent Attention) - extract compression parameters
                if has_mla:
                    op["mla"] = True
                    op["q_lora_rank"] = getattr(submod, "q_lora_rank", 0)
                    op["kv_lora_rank"] = getattr(submod, "kv_lora_rank", 0)
                    op["qk_head_dim"] = getattr(submod, "qk_head_dim", 0)
                    op["qk_rope_head_dim"] = getattr(submod, "qk_rope_head_dim", 0)
                    op["qk_nope_head_dim"] = getattr(submod, "qk_nope_head_dim", 0)
                    op["v_head_dim"] = getattr(submod, "v_head_dim", 0)
                    op["rope_interleave"] = getattr(submod, "rope_interleave", True)

                # Sliding window attention (for heterogeneous models like GPT-OSS)
                sliding_window = getattr(submod, "sliding_window", 0)
                if sliding_window and sliding_window > 0:
                    op["sliding_window"] = int(sliding_window)

                ops.append(op)
                tensor_map[node.name] = out_name
                continue

            if getattr(submod, "_fusable_kernel", None) == "mlp":
                # Fused MLP/FFN op
                input_node = node.args[0]
                out_name = get_tensor_name(node.name)

                # Check for config flags
                config = getattr(submod, "_fusable_config", [])
                activation = "gelu" if "gelu" in config else "silu"
                has_fused_gate_up = "fused_gate_up" in config

                op = {
                    "op": "mlp",
                    "inputs": [get_input_ref(input_node)],
                    "outputs": [out_name],
                    "activation": activation,
                    "fused_gate_up": has_fused_gate_up,
                }
                ops.append(op)
                tensor_map[node.name] = out_name
                continue

            if getattr(submod, "_fusable_kernel", None) == "mamba":
                # Mamba2 mixer - monolithic op (not decomposed into conv1d/softplus/etc)
                input_node = node.args[0]
                out_name = get_tensor_name(node.name)

                # Extract config with validation (errors on missing required attrs)
                mamba_config = extract_mamba_config(submod, node)

                op = {
                    "op": "mamba_mixer",
                    "inputs": [get_input_ref(input_node)],
                    "outputs": [out_name],
                    "name": mod_name,
                    **mamba_config,  # All validated config fields
                }
                ops.append(op)
                tensor_map[node.name] = out_name
                continue

            if getattr(submod, "_fusable_kernel", None) == "shortconv":
                # LFM2 ShortConv - monolithic op (not decomposed into conv1d/chunking/etc)
                input_node = node.args[0]
                out_name = get_tensor_name(node.name)

                # Extract config from ShortConv module
                op = {
                    "op": "shortconv",
                    "inputs": [get_input_ref(input_node)],
                    "outputs": [out_name],
                    "name": mod_name,
                    "d_conv": getattr(submod, "L_cache", 3),
                    "conv_dim": getattr(submod, "conv_dim", None),
                    "conv_dim_out": getattr(submod, "conv_dim_out", None),
                    "conv_bias": submod.conv.bias is not None,
                }
                ops.append(op)
                tensor_map[node.name] = out_name
                continue

            if isinstance(submod, nn.Linear):
                # Get input - find the first Node arg
                input_node = node.args[0]
                out_name = get_tensor_name(node.name)

                op = {
                    "op": "linear",
                    "inputs": [get_input_ref(input_node)],
                    "outputs": [out_name],
                    "name": mod_name,
                    "bias": submod.bias is not None,
                }
                ops.append(op)
                tensor_map[node.name] = out_name

            elif isinstance(submod, nn.Embedding):
                input_node = node.args[0]
                out_name = get_tensor_name(node.name)
                op = {
                    "op": "embedding",
                    "inputs": [get_input_ref(input_node)],
                    "outputs": [out_name],
                }
                ops.append(op)
                tensor_map[node.name] = out_name

            elif "RotaryEmbedding" in type(submod).__name__:
                # RoPE - emit rope ops for q and k
                if len(node.args) >= 2:
                    q_in = node.args[0]
                    k_in = node.args[1]
                    q_out = get_tensor_name(f"{node.name}_q")
                    k_out = get_tensor_name(f"{node.name}_k")

                    # Emit rope for query
                    ops.append({
                        "op": "rope",
                        "inputs": [get_input_ref(q_in)],
                        "outputs": [q_out],
                        "target": "query",
                    })
                    # Emit rope for key
                    ops.append({
                        "op": "rope",
                        "inputs": [get_input_ref(k_in)],
                        "outputs": [k_out],
                        "target": "key",
                    })

                    # Returns tuple (q, k) - will be unpacked by getitem
                    tensor_map[node.name] = ("_rope", [q_out, k_out])

            else:
                # Skip other modules
                # Pass through - map output to input
                if node.args and isinstance(node.args[0], fx.Node):
                    tensor_map[node.name] = tensor_map.get(node.args[0].name, get_tensor_name(node.args[0].name))

        elif node.op == "get_attr":
            # Getting a parameter (like norm.weight)
            tensor_map[node.name] = get_module_name(node.target)

        elif node.op == "call_function":
            fn = node.target
            fn_name = getattr(fn, "__name__", str(fn))

            if fn_name == "split":
                # torch.split - this is important for fused QKV/gate_up
                input_node = node.args[0]
                split_sizes = node.args[1]

                # Handle both list and fx.Node split_sizes
                if isinstance(split_sizes, (list, tuple)):
                    sizes = list(split_sizes)
                elif isinstance(split_sizes, fx.Node):
                    # Dynamic split - skip for now, will be computed at runtime
                    tensor_map[node.name] = "_split_tuple"
                    continue
                else:
                    sizes = [int(split_sizes)]

                dim = -1
                if len(node.args) > 2:
                    dim = node.args[2] if isinstance(node.args[2], int) else -1
                dim = node.kwargs.get("dim", dim)

                num_outputs = len(sizes)
                output_names = [f"_split_{split_counter[0]}_{i}" for i in range(num_outputs)]
                split_counter[0] += 1

                op = {
                    "op": "split",
                    "inputs": [get_input_ref(input_node)],
                    "outputs": output_names,
                    "split_sizes": sizes,
                    "dim": dim,
                    "num_outputs": num_outputs,
                }
                ops.append(op)

                # Mark for getitem extraction
                tensor_map[node.name] = ("_split", output_names)

            elif fn_name == "scaled_dot_product_attention" or is_sdpa_node(node):
                q, k, v = node.args[0], node.args[1], node.args[2]
                out_name = get_tensor_name(node.name)

                # Check enable_gqa (warn if True, we handle GQA differently)
                check_sdpa_enable_gqa(node, warn_state)

                # Extract params from BOTH positional and keyword args
                op = {
                    "op": "scaled_dot_product_attention",
                    "inputs": [get_input_ref(q), get_input_ref(k), get_input_ref(v)],
                    "outputs": [out_name],
                    "dropout_p": get_sdpa_param(node, "dropout_p", 4, 0.0),
                    "is_causal": get_sdpa_param(node, "is_causal", 5, True),  # Intentional default for decoder LLMs
                }
                # Extract scale with proper policy (for muP support)
                sdpa_scale = extract_sdpa_scale(node, warn_state)
                if sdpa_scale is not None:
                    op["sdpa_scale"] = sdpa_scale
                ops.append(op)
                tensor_map[node.name] = out_name

            elif fn_name == "silu":
                input_node = node.args[0]
                out_name = get_tensor_name(node.name)
                op = {
                    "op": "silu",
                    "inputs": [get_input_ref(input_node)],
                    "outputs": [out_name],
                }
                ops.append(op)
                tensor_map[node.name] = out_name

            elif fn_name == "gelu":
                input_node = node.args[0]
                out_name = get_tensor_name(node.name)
                op = {
                    "op": "gelu",
                    "inputs": [get_input_ref(input_node)],
                    "outputs": [out_name],
                }
                ops.append(op)
                tensor_map[node.name] = out_name

            elif fn_name == "getitem":
                container = node.args[0]
                idx = node.args[1]
                container_val = tensor_map.get(container.name)

                if isinstance(container_val, tuple) and container_val[0] == "_split":
                    # Extracting from split result
                    output_names = container_val[1]
                    if isinstance(idx, int) and idx < len(output_names):
                        tensor_map[node.name] = output_names[idx]
                    else:
                        tensor_map[node.name] = get_tensor_name(node.name)
                elif isinstance(container_val, tuple) and container_val[0] == "_rope":
                    # Extracting from RoPE result (q, k tuple)
                    output_names = container_val[1]
                    if isinstance(idx, int) and idx < len(output_names):
                        tensor_map[node.name] = output_names[idx]
                    else:
                        tensor_map[node.name] = get_tensor_name(node.name)
                else:
                    # Pass through
                    tensor_map[node.name] = get_tensor_name(node.name)

            elif fn_name == "add":
                a, b = node.args[0], node.args[1]
                out_name = get_tensor_name(node.name)

                op = {
                    "op": "add",
                    "inputs": [get_input_ref(a), get_input_ref(b)],
                    "outputs": [out_name],
                }
                ops.append(op)
                tensor_map[node.name] = out_name

            elif fn_name == "mul":
                a, b = node.args[0], node.args[1]
                out_name = get_tensor_name(node.name)

                op = {
                    "op": "mul",
                    "inputs": [get_input_ref(a), get_input_ref(b)],
                    "outputs": [out_name],
                }
                ops.append(op)
                tensor_map[node.name] = out_name

            elif fn_name == "rsqrt":
                input_node = node.args[0]
                out_name = get_tensor_name(node.name)
                op = {
                    "op": "rsqrt",
                    "inputs": [get_input_ref(input_node)],
                    "outputs": [out_name],
                }
                ops.append(op)
                tensor_map[node.name] = out_name

            else:
                # Skip other functions (rsqrt, cat for RoPE, etc.)
                # Pass through first tensor arg if any
                for arg in node.args:
                    if isinstance(arg, fx.Node):
                        tensor_map[node.name] = tensor_map.get(arg.name, get_tensor_name(arg.name))
                        break
                else:
                    tensor_map[node.name] = get_tensor_name(node.name)

        elif node.op == "call_method":
            method_name = node.target
            out_name = get_tensor_name(node.name)

            if method_name in ("view", "reshape"):
                # Emit reshape op
                input_node = node.args[0]
                # Extract shape from args (skip the input tensor)
                shape_args = node.args[1:]
                shape = []
                shape_idx = 0
                for s in shape_args:
                    if isinstance(s, int):
                        shape.append(s)
                    elif isinstance(s, fx.Node):
                        # Dynamic dims from x.shape - first is B, second is T
                        if shape_idx == 0:
                            shape.append(-2)  # B
                        elif shape_idx == 1:
                            shape.append(-3)  # T
                        else:
                            shape.append(-1)
                    else:
                        shape.append(-1)
                    shape_idx += 1
                op = {
                    "op": "reshape",
                    "inputs": [get_input_ref(input_node)],
                    "outputs": [out_name],
                    "shape": shape,
                }
                ops.append(op)
                tensor_map[node.name] = out_name

            elif method_name == "transpose":
                input_node = node.args[0]
                dim0 = node.args[1] if len(node.args) > 1 else 0
                dim1 = node.args[2] if len(node.args) > 2 else 1
                op = {
                    "op": "transpose",
                    "inputs": [get_input_ref(input_node)],
                    "outputs": [out_name],
                    "dim0": dim0,
                    "dim1": dim1,
                }
                ops.append(op)
                tensor_map[node.name] = out_name

            elif method_name == "repeat_interleave":
                # GQA expansion - Zig handles this internally in attention
                # Just pass through the tensor
                input_node = node.args[0]
                tensor_map[node.name] = tensor_map.get(input_node.name, get_tensor_name(input_node.name))

            elif method_name == "contiguous":
                # Pass through
                input_node = node.args[0]
                tensor_map[node.name] = tensor_map.get(input_node.name, get_tensor_name(input_node.name))

            elif method_name == "mean":
                input_node = node.args[0]
                dim = node.args[1] if len(node.args) > 1 else -1
                keepdim = node.kwargs.get("keepdim", False)
                op = {
                    "op": "mean",
                    "inputs": [get_input_ref(input_node)],
                    "outputs": [out_name],
                    "dim": dim,
                    "keepdim": keepdim,
                }
                ops.append(op)
                tensor_map[node.name] = out_name

            elif method_name == "pow":
                input_node = node.args[0]
                exponent = node.args[1] if len(node.args) > 1 else 1
                if isinstance(exponent, fx.Node):
                    exponent = 1
                op = {
                    "op": "pow",
                    "inputs": [get_input_ref(input_node)],
                    "outputs": [out_name],
                    "exponent": float(exponent),
                }
                ops.append(op)
                tensor_map[node.name] = out_name

            else:
                # Pass through other methods
                if node.args and isinstance(node.args[0], fx.Node):
                    tensor_map[node.name] = tensor_map.get(node.args[0].name, get_tensor_name(node.args[0].name))
                else:
                    tensor_map[node.name] = out_name

        elif node.op == "output":
            pass

    return ops


class TaluTracer(fx.Tracer):
    """
    Custom tracer that treats certain modules as leaves (opaque).

    Default mode (fused ops for speed):
    - RMSNorm/LayerNorm - Zig has optimized SIMD norm kernels
    - Attention with @fusable - Zig has optimized attention kernel
    - MLP with @fusable - Zig has optimized FFN kernel
    - MoE - special routing logic
    - RotaryEmbedding - Zig handles RoPE internally

    Primitives-only mode (TALU_PRIMITIVES_ONLY=1):
    - All ops traced into primitives (linear, split, sdpa, etc.)
    - Guaranteed to work with any model architecture
    - Useful for debugging or non-standard models
    """

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if PRIMITIVES_ONLY:
            # In primitives-only mode, only MoE, Mamba, and RotaryEmbedding are leaves
            # (these have special handling that can't be expressed as primitives)
            kernel = getattr(m, "_fusable_kernel", None)
            if kernel == "moe":
                return True
            # Mamba MUST be a leaf even in primitives mode - it contains ops
            # that fx_to_ops doesn't support (conv1d, softplus, einsum, slicing)
            if kernel == "mamba":
                return True
            # ShortConv MUST be a leaf even in primitives mode - it contains
            # ops that fx_to_ops doesn't support (conv1d, chunking)
            if kernel == "shortconv":
                return True
            name = type(m).__name__
            if name == "RotaryEmbedding":
                return True
            return super().is_leaf_module(m, module_qualified_name)

        # Default mode: fuse everything we can
        # MoE is a leaf (special routing logic)
        kernel = getattr(m, "_fusable_kernel", None)
        if kernel == "moe":
            return True
        # Attention with @fusable decorator is a leaf
        if kernel == "attention":
            return True
        # MLP with @fusable decorator is a leaf
        if kernel == "mlp":
            return True
        # Mamba with @fusable decorator is a leaf (monolithic mamba_mixer op)
        if kernel == "mamba":
            return True
        # ShortConv with @fusable decorator is a leaf (monolithic shortconv op)
        if kernel == "shortconv":
            return True
        # RMSNorm/LayerNorm are leaves (Zig has optimized SIMD kernels)
        name = type(m).__name__
        if name in ("RMSNorm", "LayerNorm"):
            return True
        # RotaryEmbedding is a leaf (Zig handles RoPE internally)
        if name == "RotaryEmbedding":
            return True
        return super().is_leaf_module(m, module_qualified_name)


def trace_block_instance(block: nn.Module) -> List[Dict[str, Any]]:
    """
    Trace a block instance using torch.fx to extract operations.
    """
    try:
        tracer = TaluTracer()
        graph = tracer.trace(block)
        traced = fx.GraphModule(block, graph)
    except Exception as e:
        raise RuntimeError(f"Failed to trace {type(block).__name__}: {e}")

    return fx_to_ops(traced, block)


def trace_block(block_class, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Trace a Block class using torch.fx to extract operations.

    Args:
        block_class: The Block nn.Module class
        config: Model config dict for instantiation

    Returns:
        List of ops in JSON format
    """
    block = block_class(**config)
    return trace_block_instance(block)


def _dedupe_candidates(candidates: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        deduped.append(cand)
    return deduped


def generate_weight_candidates(param_name: str) -> List[str]:
    """
    Generate standard HuggingFace name candidates for a weight.
    """
    return [f"{prefix}{param_name}" for prefix in WEIGHT_PREFIXES]


# Standard prefixes for block weight names (HuggingFace-style).
# These are stored once at the graph root level; block weights only need the suffix.
WEIGHT_PREFIXES = [
    "model.layers.{d}.",
    "model.language_model.layers.{d}.",
    "layers.{d}.",
    "transformer.h.{d}.",
    "backbone.layers.{d}.",
    "language_model.model.layers.{d}.",
]


def generate_global_weight_specs() -> List[Dict[str, Any]]:
    """
    Generate weight specifications for global (non-per-layer) weights.

    These are: token_embeddings, ln_final (final norm), lm_head (output projection).
    """
    return [
        {
            "id": "token_embeddings",
            "candidates": [
                "model.embed_tokens.weight",
                "model.language_model.embed_tokens.weight",
                "embed_tokens.weight",
                "transformer.wte.weight",
                "backbone.embedding.weight",
                "language_model.model.embed_tokens.weight",
            ],
            "module_type": "Embedding",
            "layout": "embedding",
            "dtype": "float32",
            "required": True,
        },
        {
            "id": "ln_final",
            "candidates": [
                "model.norm.weight",
                "model.language_model.norm.weight",
                "norm.weight",
                "transformer.ln_f.weight",
                "backbone.norm.weight",
                "language_model.model.norm.weight",
                "model.embedding_norm.weight",  # LFM2
            ],
            "module_type": "RMSNorm",
            "layout": "none",
            "dtype": "float32",
            "required": True,
        },
        {
            "id": "lm_head",
            "candidates": [
                "lm_head.weight",
                "output.weight",
                "transformer.lm_head.weight",
                "language_model.lm_head.weight",
            ],
            "module_type": "Linear",
            "layout": "linear",
            "dtype": "float32",
            "required": False,  # May be tied to embeddings
        },
    ]


def infer_layout(module_type: str) -> str:
    """
    Infer weight layout from module type.
    """
    if module_type == "Linear":
        return "linear"
    if module_type == "Conv1d":
        return "conv1d_depthwise"
    if module_type == "Embedding":
        return "embedding"
    return "none"


def _get_param_parent(block: nn.Module, param_name: str) -> nn.Module:
    parts = param_name.split(".")
    if len(parts) == 1:
        return block
    module_path = ".".join(parts[:-1])
    try:
        return block.get_submodule(module_path)
    except AttributeError:
        parent = block
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent


def _should_skip_weight(param_name: str, module_type: str) -> bool:
    _ = module_type
    # RoPE inv_freq buffers are recomputed at runtime; checkpoints may omit them.
    if param_name.endswith("inv_freq") and "rotary" in param_name:
        return True
    # MXFP4 expert weights (blocks, scales) are loaded by the MoE kernel directly,
    # not the generic weight loader. Skip them to avoid UnexpectedDType errors.
    if "experts" in param_name and ("_blocks" in param_name or "_scales" in param_name):
        return True
    return False


def extract_weight_specs(block: nn.Module) -> List[Dict[str, Any]]:
    """
    Extract weight specifications from a block instance.

    Block weights use prefix factorization: the graph stores weight_prefixes
    once at the root, and each weight spec just has the id (suffix).
    The runtime joins prefix + id to generate candidate paths.

    Only weights with overrides get explicit candidates (for non-standard paths).
    """
    specs: List[Dict[str, Any]] = []
    overrides = getattr(block, "weight_map_overrides", None) or {}

    for param_name, param in block.state_dict().items():
        parent = _get_param_parent(block, param_name)
        module_type = type(parent).__name__
        if _should_skip_weight(param_name, module_type):
            continue

        # 1D tensors (biases) use "none" layout to ensure f32 conversion.
        # 2D+ tensors use layout based on module type.
        layout = "none" if param.dim() == 1 else infer_layout(module_type)

        spec: Dict[str, Any] = {
            "id": param_name,
            "module_type": module_type,
            "layout": layout,
            "dtype": str(param.dtype).replace("torch.", ""),
            "required": True,
            # Note: expected_shape intentionally omitted. Weight shapes depend on
            # model size (d_model, n_heads, etc.) and the same graph serves all
            # sizes of an architecture. The loader infers orientation from d_model.
        }

        # Only include explicit candidates for overridden weights
        if param_name in overrides:
            override_candidates = list(overrides[param_name])
            standard_candidates = generate_weight_candidates(param_name)
            spec["candidates"] = _dedupe_candidates(override_candidates + standard_candidates)

        specs.append(spec)

    return specs


def _config_value(config: Dict[str, Any], key: str) -> Optional[Any]:
    if key in config:
        return config[key]
    if key == "rope_theta":
        rope_params = config.get("rope_parameters", {})
        if isinstance(rope_params, dict) and "rope_theta" in rope_params:
            return rope_params["rope_theta"]
    return None


def _block_kwargs(block_class, config: Dict[str, Any]) -> Dict[str, Any]:
    alias_map = {
        "num_heads": ["num_attention_heads"],
        "num_kv_heads": ["num_key_value_heads"],
        "head_dim": ["head_dim"],
        "hidden_size": ["hidden_size"],
        "intermediate_size": ["intermediate_size"],
        "rope_theta": ["rope_theta"],
        "eps": ["rms_norm_eps", "layer_norm_eps", "eps"],
        "num_experts": ["num_local_experts", "num_experts"],
        "num_experts_per_tok": ["num_experts_per_tok"],
        # MLA (Multi-Latent Attention) specific
        "q_lora_rank": ["q_lora_rank"],
        "kv_lora_rank": ["kv_lora_rank"],
        "qk_head_dim": ["qk_head_dim"],
        "qk_rope_head_dim": ["qk_rope_head_dim"],
        "qk_nope_head_dim": ["qk_nope_head_dim"],
        "v_head_dim": ["v_head_dim"],
        "rope_interleave": ["rope_interleave"],
    }

    kwargs = {}
    sig = inspect.signature(block_class.__init__)
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        value = None
        if name in config:
            value = config[name]
        else:
            for alias in alias_map.get(name, []):
                value = _config_value(config, alias)
                if value is not None:
                    break
        if value is None:
            if name == "num_kv_heads":
                value = _config_value(config, "num_attention_heads")
            elif name == "head_dim":
                n_heads = _config_value(config, "num_attention_heads")
                hidden = _config_value(config, "hidden_size")
                if n_heads and hidden:
                    value = hidden // n_heads
            elif name == "num_experts":
                value = _config_value(config, "num_local_experts") or 8
            elif name == "num_experts_per_tok":
                value = _config_value(config, "num_experts_per_tok") or 2
        if value is None:
            if param.default is inspect._empty:
                raise ValueError(f"Missing config value for Block param '{name}'")
            continue
        kwargs[name] = value
    return kwargs


def _trace_hybrid_block(block_class, config: Dict[str, Any], layer_idx: int) -> List[Dict[str, Any]]:
    """
    Trace a HybridBlock at a specific layer index.

    HybridBlock takes (config, layer_idx) and the layer type is determined by
    config["layer_types"][layer_idx].
    """
    block = block_class(config, layer_idx)
    return trace_block_instance(block)


def generate_graph(model_name: str) -> Dict[str, Any]:
    """
    Generate graph for a model by tracing its Block class with torch.fx.

    For heterogeneous models (e.g., granite_hybrid), traces each variant
    separately and generates block_variants + layer_map.
    """
    registry = ARCHITECTURES
    if model_name not in registry:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(registry.keys())}")

    model_def = registry[model_name]

    # Import the model module
    module = importlib.import_module(model_def['module'])

    # Get the Block class
    block_class = getattr(module, model_def["block_class"])

    # Load config
    model_id = model_def["model_ids"][0]
    config = load_config(model_id)

    # Build pre/post block
    eps = _config_value(config, "rms_norm_eps") or _config_value(config, "layer_norm_eps") or model_def.get("eps", 1e-6)

    # Allow architecture overrides for weight prefixes and global weights
    weight_prefixes = model_def.get("weight_prefixes", WEIGHT_PREFIXES)
    global_weights = model_def.get("global_weights", generate_global_weight_specs())

    # Allow architecture-level pre_block/post_block overrides
    if "pre_block" in model_def:
        pre_block = model_def["pre_block"]
    else:
        pre_block = [
            {"op": "embedding", "inputs": [{"tensor": "input_ids"}], "outputs": ["_t0"]}
        ]

        # Handle embedding_multiplier for muP models
        embedding_multiplier = config.get("embedding_multiplier")
        if embedding_multiplier and embedding_multiplier != 1.0:
            pre_block.append({
                "op": "mul",
                "inputs": [{"tensor": "_t0"}, {"scalar": embedding_multiplier}],
                "outputs": ["_t1"]
            })
        elif "embedding_scale" in model_def:
            pre_block.append({
                "op": "mul",
                "inputs": [{"tensor": "_t0"}, {"scalar": model_def["embedding_scale"]}],
                "outputs": ["_t1"]
            })

    if "post_block" in model_def:
        post_block = model_def["post_block"]
    else:
        post_block = [
            {"op": "norm", "inputs": [{"tensor": "_t_last"}], "outputs": ["_t_out"], "eps": eps}
        ]

    # Check if this is a heterogeneous model
    if model_def.get("heterogeneous"):
        # Heterogeneous model: trace each variant and build layer_map
        layer_types_key = model_def.get("layer_types_key", "layer_types")
        layer_types_raw = config.get(layer_types_key, [])

        if not layer_types_raw:
            # Some heterogeneous models encode variants via full_attention_interval
            # instead of an explicit per-layer list.
            interval = config.get("full_attention_interval")
            n_layers = config.get("num_hidden_layers", config.get("n_layers"))
            if interval and n_layers:
                layer_types_raw = [
                    "full_attention" if (idx + 1) % int(interval) == 0 else "linear_attention"
                    for idx in range(int(n_layers))
                ]
            else:
                raise ValueError(f"Heterogeneous model {model_name} requires '{layer_types_key}' in config")

        # Handle different config formats:
        # - layer_types: ["mamba", "attention", "mamba", ...] - per-layer type array
        # - full_attn_idxs: [2, 5, 8, ...] - list of layer indices that are attention
        if layer_types_key == "full_attn_idxs":
            # LFM2-style: full_attn_idxs is a list of indices for attention layers
            n_layers = config.get("num_hidden_layers", config.get("n_layers", 16))
            attn_indices = set(layer_types_raw)
            # Layers in attn_indices are "attention", others are "shortconv"
            layer_types = ["attention" if i in attn_indices else "shortconv" for i in range(n_layers)]
        else:
            # Granite-style: layer_types is a per-layer array
            layer_types = layer_types_raw

        # Find unique variants and a representative layer for each
        variant_layers = {}  # variant_name -> first layer_idx with that type
        for idx, layer_type in enumerate(layer_types):
            if layer_type not in variant_layers:
                variant_layers[layer_type] = idx

        # Trace each variant
        block_variants = {}
        variant_to_index = {}  # For layer_map (use indices for O(1) lookup)
        variant_names = [str(k) for k in variant_layers.keys()]
        for variant_idx, (variant_key, layer_idx) in enumerate(variant_layers.items()):
            variant_name = str(variant_key)  # Ensure string keys for JSON/Zig
            print(f"  Tracing variant '{variant_name}' at layer {layer_idx}...")
            block_instance = block_class(config, layer_idx)
            ops = trace_block_instance(block_instance)
            weights = extract_weight_specs(block_instance)
            block_variants[variant_name] = {"ops": ops, "weights": weights}
            variant_to_index[variant_key] = variant_idx  # Keep original key for lookup

        # Build layer_map as indices (not strings) for O(1) lookup
        layer_map = [variant_to_index[lt] for lt in layer_types]

        return {
            "name": model_def["name"],
            "model_types": model_def["model_types"],
            "weight_prefixes": weight_prefixes,
            "block_variants": block_variants,
            "variant_names": variant_names,
            "layer_map": layer_map,
            "pre_block": pre_block,
            "post_block": post_block,
            "global_weights": global_weights,
        }

    # Homogeneous model: single block type
    block_instance = block_class(**_block_kwargs(block_class, config))
    block_ops = trace_block_instance(block_instance)
    block_weights = extract_weight_specs(block_instance)

    return {
        "name": model_def["name"],
        "model_types": model_def["model_types"],
        "weight_prefixes": weight_prefixes,
        "block": block_ops,
        "block_weights": block_weights,
        "pre_block": pre_block,
        "post_block": post_block,
        "global_weights": global_weights,
    }


def main():
    import sys

    args = sys.argv[1:]

    if not args or args[0] in ["-h", "--help"]:
        print("Usage: python -m talu.trace <model_name>")
        print("       python -m talu.trace --all")
        print(f"\nAvailable: {list(ARCHITECTURES.keys())}")
        return

    # Output to tools/archs/_graphs (same directory as this script)
    graphs_dir = Path(__file__).parent / "_graphs"
    graphs_dir.mkdir(exist_ok=True)

    names = list(ARCHITECTURES.keys()) if args[0] == "--all" else args

    for name in names:
        if name not in ARCHITECTURES:
            print(f"Unknown: {name}")
            print(f"Available: {list(ARCHITECTURES.keys())}")
            sys.exit(1)

        try:
            graph = generate_graph(name)
            out_path = graphs_dir / f"{name}.json"
            with open(out_path, "w") as f:
                json.dump(graph, f, indent=2)
                f.write("\n")
            print(f"OK: {name} -> {out_path}")
        except Exception as e:
            print(f"FAIL: {name} - {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
