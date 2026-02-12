"""
Storage-related fixture factories.

Provides factories for creating fake HuggingFace cache structures.
"""

import json
from pathlib import Path


def fake_hf_cache_factory(tmp_path: Path, monkeypatch) -> dict:
    """
    Create a fake HuggingFace cache with known models.

    This creates a temporary directory structure that mimics the
    HuggingFace Hub cache, enabling deterministic testing without
    relying on real cached models.

    The factory:
    1. Creates a fake cache at tmp_path/huggingface/hub
    2. Populates it with two test models (test/model-a and org/model-b)
    3. Sets HF_HOME to point to the fake cache

    Args:
        tmp_path: pytest tmp_path fixture
        monkeypatch: pytest monkeypatch fixture

    Returns:
        dict with keys:
        - cache_dir: Path to the hub directory
        - hf_home: Path to the huggingface directory
        - models: List of model IDs in the cache
        - info: Dict mapping model_id -> {"path": str, "config": dict}
    """
    hf_home = tmp_path / "huggingface"
    cache_dir = hf_home / "hub"

    models_info = {}

    # Model A: Simple model with config.json
    # Note: Uses "test/model-a" as ID - tests should reference this ID
    model_a_id = "test/model-a"
    model_a_dir = cache_dir / "models--test--model-a"
    model_a_snapshot = model_a_dir / "snapshots" / "abc123def456"
    model_a_snapshot.mkdir(parents=True)

    config_a = {
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,  # Required for describe() head_dim calculation
        "num_key_value_heads": 4,  # GQA heads
        "intermediate_size": 128,  # FFN dimension
        "max_position_embeddings": 512,
        "vocab_size": 1000,
        "rms_norm_eps": 1e-6,
    }
    (model_a_snapshot / "config.json").write_text(json.dumps(config_a))

    tokenizer_a = {"version": "1.0", "model": {"type": "BPE"}}
    (model_a_snapshot / "tokenizer.json").write_text(json.dumps(tokenizer_a))

    # Weights file required for Zig storage to consider model "cached"
    fake_weights_a = b"\x00" * 512
    (model_a_snapshot / "model.safetensors").write_bytes(fake_weights_a)

    refs_dir = model_a_dir / "refs"
    refs_dir.mkdir(parents=True)
    (refs_dir / "main").write_text("abc123def456")

    models_info[model_a_id] = {
        "path": str(model_a_snapshot),
        "config": config_a,
    }

    # Model B: Model with different structure
    model_b_id = "org/model-b"
    model_b_dir = cache_dir / "models--org--model-b"
    model_b_snapshot = model_b_dir / "snapshots" / "xyz789"
    model_b_snapshot.mkdir(parents=True)

    config_b = {
        "model_type": "qwen",
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "intermediate_size": 256,
        "max_position_embeddings": 1024,
        "vocab_size": 2000,
        "rms_norm_eps": 1e-6,
    }
    (model_b_snapshot / "config.json").write_text(json.dumps(config_b))

    tokenizer_b = {"version": "1.0", "model": {"type": "BPE"}}
    (model_b_snapshot / "tokenizer.json").write_text(json.dumps(tokenizer_b))

    fake_weights_b = b"\x00" * 1024  # 1KB for size testing
    (model_b_snapshot / "model.safetensors").write_bytes(fake_weights_b)

    refs_dir_b = model_b_dir / "refs"
    refs_dir_b.mkdir(parents=True)
    (refs_dir_b / "main").write_text("xyz789")

    models_info[model_b_id] = {
        "path": str(model_b_snapshot),
        "config": config_b,
    }

    # Isolate both caches so real models don't leak into tests
    monkeypatch.setenv("HF_HOME", str(hf_home))
    talu_home = tmp_path / "talu"
    talu_home.mkdir(parents=True)
    monkeypatch.setenv("TALU_HOME", str(talu_home))

    return {
        "cache_dir": cache_dir,
        "hf_home": hf_home,
        "models": [model_a_id, model_b_id],
        "info": models_info,
    }


def empty_hf_cache_factory(tmp_path: Path, monkeypatch) -> dict:
    """
    Create an empty HuggingFace cache.

    Useful for testing behavior when no models are cached.

    Args:
        tmp_path: pytest tmp_path fixture
        monkeypatch: pytest monkeypatch fixture

    Returns:
        dict with keys:
        - cache_dir: Path to the hub directory
        - hf_home: Path to the huggingface directory
        - models: Empty list
    """
    hf_home = tmp_path / "huggingface"
    cache_dir = hf_home / "hub"
    cache_dir.mkdir(parents=True)

    monkeypatch.setenv("HF_HOME", str(hf_home))
    talu_home = tmp_path / "talu"
    talu_home.mkdir(parents=True)
    monkeypatch.setenv("TALU_HOME", str(talu_home))

    return {
        "cache_dir": cache_dir,
        "hf_home": hf_home,
        "models": [],
    }


def incomplete_model_factory(tmp_path: Path, monkeypatch, missing: str = "weights") -> dict:
    """
    Create a cache with an incomplete model (missing tokenizer.json or weights).

    Useful for testing graceful handling of malformed cache entries.

    Args:
        tmp_path: pytest tmp_path fixture
        monkeypatch: pytest monkeypatch fixture
        missing: What to omit - "weights", "tokenizer", or "config"

    Returns:
        dict with keys:
        - cache_dir: Path to the hub directory
        - hf_home: Path to the huggingface directory
        - model_id: The incomplete model ID
        - model_path: Path to the model snapshot
        - missing: What was omitted
    """
    hf_home = tmp_path / "huggingface"
    cache_dir = hf_home / "hub"

    model_id = f"test/incomplete-{missing}"
    model_dir = cache_dir / f"models--test--incomplete-{missing}"
    snapshot = model_dir / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)

    # Always create config unless that's what we're testing
    if missing != "config":
        config = {
            "model_type": "llama",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 256,
            "vocab_size": 1000,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-5,
        }
        (snapshot / "config.json").write_text(json.dumps(config))

    # Create tokenizer unless that's what we're testing
    if missing != "tokenizer":
        tokenizer = {"version": "1.0", "model": {"type": "BPE"}}
        (snapshot / "tokenizer.json").write_text(json.dumps(tokenizer))

    # Create weights unless that's what we're testing
    if missing != "weights":
        (snapshot / "model.safetensors").write_bytes(b"\x00" * 512)

    # Create refs
    refs_dir = model_dir / "refs"
    refs_dir.mkdir(parents=True)
    (refs_dir / "main").write_text("abc123")

    monkeypatch.setenv("HF_HOME", str(hf_home))
    talu_home = tmp_path / "talu"
    talu_home.mkdir(parents=True)
    monkeypatch.setenv("TALU_HOME", str(talu_home))

    return {
        "cache_dir": cache_dir,
        "hf_home": hf_home,
        "model_id": model_id,
        "model_path": str(snapshot),
        "missing": missing,
    }
