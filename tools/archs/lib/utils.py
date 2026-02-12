"""
Shared utilities for Talu models.
"""

import json
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import torch


def _find_scale_tensor(weights, hf_name):
    if not hf_name.endswith(".weight"):
        return None
    base = hf_name[: -len(".weight")]
    candidates = [
        f"{base}.weight_scale_inv",
        f"{base}_scale_inv",
    ]
    for key in candidates:
        if key in weights:
            return weights[key]
    return None


def _dequantize_fp8_with_scale(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize FP8 tensor using inverse scales.

    Supports scalar/1D and per-block scales over the last two dimensions.
    """
    w = weight.to(torch.float32)
    s = scale_inv.to(torch.float32)

    if s.numel() == 1:
        return (w * s.reshape(1)).to(out_dtype)

    if w.ndim < 2:
        # Fallback for unexpected rank
        return (w * s.reshape(1)).to(out_dtype)

    rows = w.shape[-2]
    cols = w.shape[-1]

    if s.ndim == 1:
        return (w * s.reshape(1)).to(out_dtype)

    # Expected per-block scales over last two dims.
    scale_rows = s.shape[-2]
    scale_cols = s.shape[-1]
    if scale_rows <= 0 or scale_cols <= 0:
        return w.to(out_dtype)
    if rows % scale_rows != 0 or cols % scale_cols != 0:
        return w.to(out_dtype)

    block_m = rows // scale_rows
    block_n = cols // scale_cols

    reshaped = w.reshape(-1, scale_rows, block_m, scale_cols, block_n)
    expanded = s.reshape(-1, scale_rows, scale_cols).unsqueeze(-1).unsqueeze(2)
    dequant = reshaped * expanded
    return dequant.reshape(w.shape).to(out_dtype)


def _is_fp8_tensor(t: torch.Tensor) -> bool:
    return str(t.dtype).startswith("torch.float8_")


def from_pretrained(model_class, model_id, model_dtype=None):
    """
    Load a pretrained model from HuggingFace.

    Args:
        model_class: The model class (e.g., Qwen3, Phi4)
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-0.6B")

    Returns:
        (model, tokenizer) tuple
    """
    path = Path(snapshot_download(model_id))

    with open(path / "config.json") as f:
        config = json.load(f)

    # Handle multimodal models with nested text_config
    if "text_config" in config:
        config = config["text_config"]

    if model_dtype is not None:
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(model_dtype)
        try:
            model = model_class(config)
        finally:
            torch.set_default_dtype(prev_dtype)
    else:
        model = model_class(config)

    state = model.state_dict()

    # Load shard-by-shard to avoid materializing full checkpoints and mapped
    # tensors in memory simultaneously.
    for fp in sorted(path.glob("*.safetensors")):
        shard = load_file(fp)
        mapped = {}
        for hf_name, value in shard.items():
            # Scale tensors are consumed when dequantizing the paired weight.
            if hf_name.endswith(".weight_scale_inv") or hf_name.endswith("_scale_inv"):
                continue

            state_name = hf_name[len("model."):] if hf_name.startswith("model.") else hf_name
            if state_name not in state:
                continue

            target_dtype = state[state_name].dtype
            if _is_fp8_tensor(value):
                scale = _find_scale_tensor(shard, hf_name)
                if scale is not None:
                    value = _dequantize_fp8_with_scale(value, scale, out_dtype=target_dtype)
                else:
                    value = value.to(target_dtype)
            else:
                value = value.to(target_dtype)

            mapped[state_name] = value

        if mapped:
            model.load_state_dict(mapped, strict=False)
        del mapped
        del shard

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    except ValueError as e:
        if "TokenizersBackend" in str(e):
            # Mistral models use mistral-common tokenizer
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            tokenizer = MistralTokenizer.from_file(str(path / "tokenizer.json"))
        else:
            raise
    return model, tokenizer


def load_config(model_id):
    """Load config.json for a HuggingFace model (downloads only that file)."""
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(model_id, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    if "text_config" in config:
        config = config["text_config"]
    return config
