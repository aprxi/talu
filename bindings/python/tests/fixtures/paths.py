"""
Path-finding utilities for cached models.

These utilities don't require torch/numpy - they just find paths.
"""

import os
from pathlib import Path


def find_cached_model_path(hf_id: str) -> str | None:
    """
    Find a cached HuggingFace model path.

    Checks multiple locations:
    - HF_HOME environment variable
    - ~/.cache/huggingface/hub/models--org--name/snapshots/
    - ./cache/models--org--name/snapshots/
    - ./models/models--org--name/snapshots/

    Args:
        hf_id: HuggingFace model ID (e.g., "org/model-name")

    Returns:
        Path to model snapshot directory, or None if not found
    """
    org, name = hf_id.split("/", 1)
    dir_name = f"models--{org}--{name}"

    # Check HF_HOME first
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates = [Path(hf_home) / "hub" / dir_name]
    else:
        candidates = []

    # Standard locations
    candidates.extend(
        [
            Path.home() / ".cache" / "huggingface" / "hub" / dir_name,
            Path("cache") / dir_name,
            Path("models") / dir_name,
        ]
    )

    for model_dir in candidates:
        if not model_dir.exists():
            continue

        # Check refs/main for commit hash
        ref_main = model_dir / "refs" / "main"
        if ref_main.exists():
            try:
                commit_hash = ref_main.read_text().strip()
                snapshot = model_dir / "snapshots" / commit_hash
                if snapshot.exists() and (snapshot / "config.json").exists():
                    return str(snapshot)
            except Exception:
                pass

        # Fallback to any snapshot with config.json
        snapshots = model_dir / "snapshots"
        if snapshots.exists():
            for p in snapshots.iterdir():
                if p.is_dir() and (p / "config.json").exists():
                    return str(p)

        # Direct format (no snapshots directory)
        if (model_dir / "config.json").exists():
            return str(model_dir)

    return None


def find_unquantized_model_path() -> str | None:
    """
    Find an unquantized model for conversion tests.

    Conversion tests require an unquantized source model (f16/bf16/f32).
    MLX-4bit models are already quantized and can't be re-quantized.

    Checks:
    - TALU_TEST_MODEL_UNQUANTIZED environment variable
    - Common unquantized model locations

    Returns:
        Path to unquantized model, or None if not found
    """
    # Check environment variable first
    model_path = os.environ.get("TALU_TEST_MODEL_UNQUANTIZED")
    if model_path and Path(model_path).exists():
        return model_path

    # Fall back to TEST_MODEL_URI_TEXT_RANDOM (may or may not be unquantized)
    from tests.conftest import TEST_MODEL_URI_TEXT_RANDOM

    path = find_cached_model_path(TEST_MODEL_URI_TEXT_RANDOM)
    if path:
        return path

    return None
