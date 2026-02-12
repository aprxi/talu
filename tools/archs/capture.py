#!/usr/bin/env python3
"""
Capture PyTorch reference tensors to NPZ for debugging comparison.

This tool runs a PyTorch reference model and captures intermediate tensors
at trace points that match talu's trace.emit() points. The resulting NPZ
can be compared against talu's export to find where divergence occurs.

Usage:
    uv run python -m capture qwen3 "Hello" -o _reference/qwen3.npz
    uv run python -m capture --list
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn

from lib.utils import from_pretrained

# Import ARCHITECTURES for model lookup
from trace import ARCHITECTURES


# Trace points matching talu's trace.emit() points
TRACE_POINTS = [
    "embed",
    "attn_norm", "attn_out",
    "ffn_norm", "ffn_down",
    "block_out",
    "final_norm", "lm_head",
]


class TensorCapture:
    """Captures tensors at trace points during forward pass."""

    def __init__(self):
        self.tensors: Dict[str, np.ndarray] = {}
        self._hooks = []

    def capture(self, name: str, tensor: torch.Tensor):
        """Capture a tensor at a trace point."""
        # Convert to numpy (always float32 for comparison stability)
        self.tensors[name] = tensor.detach().cpu().float().numpy()

    def save(self, path: str):
        """Save all captured tensors to NPZ."""
        np.savez_compressed(path, **self.tensors)
        print(f"Saved {len(self.tensors)} tensors to {path}")

    def _register_hook(self, module: nn.Module, name: str):
        """Register a forward hook to capture output."""
        def hook(mod, inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            self.capture(name, tensor)
        handle = module.register_forward_hook(hook)
        self._hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def _find_module(model: nn.Module, candidates: list[str]) -> Optional[nn.Module]:
    """Find a module by trying candidate names."""
    for name in candidates:
        try:
            return model.get_submodule(name)
        except AttributeError:
            continue
    return None


def capture_standard_model(
    model: nn.Module,
    tokenizer,
    prompt: str,
    use_chat_template: bool = True,
) -> TensorCapture:
    """
    Capture tensors from a standard decoder model.

    Works with models that have:
    - embed_tokens (embedding layer)
    - layers (list of blocks)
    - norm (final norm)
    - lm_head (output projection)

    Each block should have:
    - input_layernorm (or self_attn_layer_norm)
    - self_attn
    - post_attention_layernorm (or ffn_layer_norm, mlp_layer_norm)
    - mlp (or feed_forward)
    """
    capture = TensorCapture()
    model.eval()

    # Find embedding module
    embed = _find_module(model, ["embed_tokens", "embedding", "wte", "word_embeddings"])
    if embed is None:
        raise ValueError("Could not find embedding module")

    # Find final norm
    final_norm = _find_module(model, ["norm", "ln_f", "final_layer_norm", "embedding_norm"])
    if final_norm is None:
        raise ValueError("Could not find final norm module")

    # Find lm_head
    lm_head = _find_module(model, ["lm_head", "output", "lm_proj"])
    if lm_head is None:
        print("Warning: No lm_head found (embedding model?)")

    # Find layers
    layers = _find_module(model, ["layers", "h", "transformer.h", "encoder.layer"])
    if layers is None:
        raise ValueError("Could not find layers module")

    # Register hooks for embedding
    capture._register_hook(embed, "embed")

    # Register hooks for each layer
    for i, layer in enumerate(layers):
        # Attention input norm
        attn_norm = _find_module(layer, [
            "input_layernorm", "self_attn_layer_norm", "ln_1",
            "attention.layer_norm", "pre_attention_layernorm"
        ])
        if attn_norm:
            capture._register_hook(attn_norm, f"layer{i}.attn_norm")

        # Attention output
        attn = _find_module(layer, [
            "self_attn", "attention", "attn", "self_attention"
        ])
        if attn:
            capture._register_hook(attn, f"layer{i}.attn_out")

        # FFN input norm
        ffn_norm = _find_module(layer, [
            "post_attention_layernorm", "mlp_layer_norm", "ln_2",
            "ffn_layer_norm", "pre_mlp_layernorm"
        ])
        if ffn_norm:
            capture._register_hook(ffn_norm, f"layer{i}.ffn_norm")

        # FFN output (down projection)
        mlp = _find_module(layer, ["mlp", "feed_forward", "ffn", "MLP"])
        if mlp:
            capture._register_hook(mlp, f"layer{i}.ffn_down")

        # Block output - capture by hooking the whole layer
        capture._register_hook(layer, f"layer{i}.block_out")

    # Register hooks for final norm and lm_head
    capture._register_hook(final_norm, "final_norm")
    if lm_head:
        capture._register_hook(lm_head, "lm_head")

    # Tokenize
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        # Use chat template if available
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt
    else:
        text = prompt

    if hasattr(tokenizer, "encode"):
        if hasattr(tokenizer.encode(text), "ids"):
            # tokenizers library
            ids = tokenizer.encode(text).ids
        else:
            # transformers tokenizer
            ids = tokenizer.encode(text)
    else:
        raise ValueError("Unknown tokenizer type")

    input_ids = torch.tensor([ids])

    # Run forward pass
    with torch.inference_mode():
        _ = model(input_ids)

    capture.remove_hooks()
    return capture


def capture_reference(
    arch: str,
    prompt: str,
    model_id: Optional[str] = None,
    use_chat_template: bool = True,
) -> TensorCapture:
    """
    Run PyTorch reference with capture hooks.

    Args:
        arch: Architecture name from ARCHITECTURES registry
        prompt: Input prompt text
        model_id: Optional HuggingFace model ID (default: from ARCHITECTURES)

    Returns:
        TensorCapture with all captured tensors
    """
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {arch}. Available: {list(ARCHITECTURES.keys())}")

    arch_info = ARCHITECTURES[arch]

    # Import model module
    import importlib
    module = importlib.import_module(arch_info["module"])

    # Get model class - try common names
    model_class = None
    for class_name in ["Qwen3", "Llama3", "Llama2", "Phi4", "Gemma3", "Granite3",
                       "GraniteHybrid", "LFM2", "LFM2_5", "MiniLM", "Ministral3",
                       "GptOss", "YoutuVl", "Qwen3Moe"]:
        if hasattr(module, class_name):
            model_class = getattr(module, class_name)
            break

    # Also try the module name capitalized
    if model_class is None:
        module_name = arch_info["module"].split(".")[-1]
        # Convert snake_case to CamelCase
        class_name = "".join(word.capitalize() for word in module_name.split("_"))
        if hasattr(module, class_name):
            model_class = getattr(module, class_name)

    if model_class is None:
        raise ValueError(f"Could not find model class in {arch_info['module']}")

    # Load model
    model_id = model_id or arch_info["model_ids"][0]
    print(f"Loading {model_id}...")
    model, tokenizer = model_class.from_pretrained(model_id)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Capture tensors
    print(f"Capturing tensors for prompt: {prompt!r}")
    return capture_standard_model(model, tokenizer, prompt, use_chat_template=use_chat_template)


def main():
    parser = argparse.ArgumentParser(
        description="Capture PyTorch reference tensors to NPZ for debugging comparison."
    )
    parser.add_argument(
        "arch",
        nargs="?",
        help="Architecture name (e.g., qwen3, llama3)"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Hello",
        help="Input prompt (default: 'Hello')"
    )
    parser.add_argument(
        "-m", "--model",
        help="Model ID (default: from ARCHITECTURES registry)"
    )
    parser.add_argument(
        "-o", "--output",
        default="_reference/{arch}.npz",
        help="Output NPZ path (default: _reference/{arch}.npz)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available architectures"
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable tokenizer.apply_chat_template and use raw prompt text",
    )
    args = parser.parse_args()

    if args.list or not args.arch:
        print("Available architectures:")
        for name, info in ARCHITECTURES.items():
            model_id = info["model_ids"][0] if info["model_ids"] else "N/A"
            print(f"  {name:20s} {model_id}")
        return

    # Resolve output path
    output = args.output.format(arch=args.arch)
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Capturing {args.arch} reference tensors...")
    capture = capture_reference(
        args.arch,
        args.prompt,
        args.model,
        use_chat_template=not args.no_chat_template,
    )
    capture.save(output)

    # Show what was captured
    print(f"\nCaptured {len(capture.tensors)} trace points:")
    for name in sorted(capture.tensors.keys(), key=_sort_key):
        shape = capture.tensors[name].shape
        print(f"  {name:30s} {shape}")


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


if __name__ == "__main__":
    main()
