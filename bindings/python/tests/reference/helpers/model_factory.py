"""
Synthetic model generation for testing.

Creates minimal model files (config.json, tokenizer.json, model.safetensors)
for testing model loading without network dependency.
"""

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SyntheticModel:
    """
    A synthetic model for testing.

    Contains minimal config, tokenizer, and weight files.
    """

    path: Path
    weights: dict[str, np.ndarray] = field(default_factory=dict)
    vocab_size: int = 1000
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 2
    intermediate_size: int = 128
    max_seq_len: int = 512
    _temp_dir: tempfile.TemporaryDirectory | None = field(default=None, repr=False)

    def cleanup(self):
        """Clean up temporary files."""
        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


def create_minimal_config(
    vocab_size: int = 1000,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_heads: int = 2,
    intermediate_size: int = 128,
    max_seq_len: int = 512,
    model_type: str = "llama",
) -> dict[str, Any]:
    """
    Create a minimal config.json for testing.

    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        intermediate_size: FFN intermediate dimension
        max_seq_len: Maximum sequence length
        model_type: Model type string

    Returns:
        Config dictionary
    """
    return {
        "architectures": [f"{model_type.title()}ForCausalLM"],
        "model_type": model_type,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_heads,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": max_seq_len,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }


def create_minimal_tokenizer(vocab_size: int = 1000) -> dict[str, Any]:
    """
    Create a minimal tokenizer.json for testing.

    Uses a simple byte-level vocabulary.

    Args:
        vocab_size: Number of tokens

    Returns:
        Tokenizer config dictionary
    """
    # Create a simple vocabulary
    vocab = {}
    merges = []

    # Special tokens
    vocab["<pad>"] = 0
    vocab["<s>"] = 1
    vocab["</s>"] = 2
    vocab["<unk>"] = 3

    # ASCII characters
    for i in range(256):
        if i + 4 < vocab_size:
            vocab[chr(i) if 32 <= i < 127 else f"<0x{i:02X}>"] = i + 4

    # Fill remaining with fake tokens
    for i in range(260, vocab_size):
        vocab[f"token_{i}"] = i

    return {
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": merges,
        },
        "added_tokens": [
            {"id": 0, "content": "<pad>", "special": True},
            {"id": 1, "content": "<s>", "special": True},
            {"id": 2, "content": "</s>", "special": True},
            {"id": 3, "content": "<unk>", "special": True},
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
        "post_processor": None,
        "decoder": {"type": "ByteLevel"},
    }


def create_minimal_weights(
    vocab_size: int = 1000,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_heads: int = 2,
    intermediate_size: int = 128,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Create minimal random weights for a transformer model.

    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        intermediate_size: FFN intermediate dimension
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping weight names to NumPy arrays
    """
    rng = np.random.default_rng(seed)

    weights = {}

    # Embeddings
    weights["model.embed_tokens.weight"] = (
        rng.standard_normal((vocab_size, hidden_size)).astype(np.float32) * 0.02
    )

    # Per-layer weights
    for i in range(num_layers):
        prefix = f"model.layers.{i}"

        # Attention weights
        weights[f"{prefix}.self_attn.q_proj.weight"] = (
            rng.standard_normal((hidden_size, hidden_size)).astype(np.float32) * 0.02
        )

        weights[f"{prefix}.self_attn.k_proj.weight"] = (
            rng.standard_normal((hidden_size, hidden_size)).astype(np.float32) * 0.02
        )

        weights[f"{prefix}.self_attn.v_proj.weight"] = (
            rng.standard_normal((hidden_size, hidden_size)).astype(np.float32) * 0.02
        )

        weights[f"{prefix}.self_attn.o_proj.weight"] = (
            rng.standard_normal((hidden_size, hidden_size)).astype(np.float32) * 0.02
        )

        # FFN weights
        weights[f"{prefix}.mlp.gate_proj.weight"] = (
            rng.standard_normal((intermediate_size, hidden_size)).astype(np.float32) * 0.02
        )

        weights[f"{prefix}.mlp.up_proj.weight"] = (
            rng.standard_normal((intermediate_size, hidden_size)).astype(np.float32) * 0.02
        )

        weights[f"{prefix}.mlp.down_proj.weight"] = (
            rng.standard_normal((hidden_size, intermediate_size)).astype(np.float32) * 0.02
        )

        # Layer norms
        weights[f"{prefix}.input_layernorm.weight"] = np.ones(hidden_size, dtype=np.float32)
        weights[f"{prefix}.post_attention_layernorm.weight"] = np.ones(
            hidden_size, dtype=np.float32
        )

    # Final layer norm
    weights["model.norm.weight"] = np.ones(hidden_size, dtype=np.float32)

    return weights


def save_safetensors(path: Path, weights: dict[str, np.ndarray]) -> None:
    """
    Save weights in safetensors format.

    Args:
        path: Output file path
        weights: Dictionary mapping weight names to NumPy arrays
    """
    try:
        from safetensors.numpy import save_file

        save_file(weights, str(path))
    except ImportError:
        # Fallback: create a minimal safetensors file manually
        # This is a simplified implementation for testing
        _save_safetensors_manual(path, weights)


def _save_safetensors_manual(path: Path, weights: dict[str, np.ndarray]) -> None:
    """
    Manual safetensors writer (simplified, for testing only).

    Creates a valid safetensors file without the safetensors library.
    """
    import struct

    # Build header
    header = {}
    offset = 0

    # Calculate total data size and build metadata
    data_parts = []
    for name, arr in weights.items():
        arr = np.ascontiguousarray(arr)
        data = arr.tobytes()
        data_parts.append(data)

        dtype_map = {
            np.float32: "F32",
            np.float64: "F64",
            np.int32: "I32",
            np.int64: "I64",
            np.float16: "F16",
        }
        dtype_str = dtype_map.get(arr.dtype.type, "F32")

        header[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)

    # Add metadata
    header["__metadata__"] = {"format": "pt"}

    # Serialize header
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")

    # Write file
    with open(path, "wb") as f:
        # Header size (8 bytes, little endian)
        f.write(struct.pack("<Q", len(header_json)))
        # Header JSON
        f.write(header_json)
        # Tensor data
        for data in data_parts:
            f.write(data)


def create_minimal_model(
    vocab_size: int = 1000,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_heads: int = 2,
    intermediate_size: int = 128,
    max_seq_len: int = 512,
    model_type: str = "llama",
    seed: int = 42,
    output_dir: Path | None = None,
) -> SyntheticModel:
    """
    Create a complete minimal model for testing.

    Creates config.json, tokenizer.json, and model.safetensors in a
    temporary directory (or specified output_dir).

    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        intermediate_size: FFN intermediate dimension
        max_seq_len: Maximum sequence length
        model_type: Model type string
        seed: Random seed for weight generation
        output_dir: Optional output directory (uses temp dir if None)

    Returns:
        SyntheticModel with path to the created model

    Example:
        with create_minimal_model() as model:
            session = talu.Chat(str(model.path))
            # ... run tests ...
        # Cleanup happens automatically
    """
    if output_dir is not None:
        temp_dir = None
        model_dir = Path(output_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="talu_test_")
        model_dir = Path(temp_dir.name)

    # Write config.json
    config = create_minimal_config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        max_seq_len=max_seq_len,
        model_type=model_type,
    )
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Write tokenizer.json
    tokenizer = create_minimal_tokenizer(vocab_size)
    with open(model_dir / "tokenizer.json", "w") as f:
        json.dump(tokenizer, f, indent=2)

    # Write tokenizer_config.json (needed for chat template)
    tokenizer_config = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "User: {{ message['content'] }}\n"
            "{% elif message['role'] == 'assistant' %}"
            "Assistant: {{ message['content'] }}\n"
            "{% elif message['role'] == 'system' %}"
            "System: {{ message['content'] }}\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "Assistant:"
            "{% endif %}"
        ),
    }
    with open(model_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    # Write model.safetensors
    weights = create_minimal_weights(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        seed=seed,
    )
    save_safetensors(model_dir / "model.safetensors", weights)

    return SyntheticModel(
        path=model_dir,
        weights=weights,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        max_seq_len=max_seq_len,
        _temp_dir=temp_dir,
    )
