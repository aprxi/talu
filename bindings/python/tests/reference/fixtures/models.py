"""
Model-related fixture factories (requires torch).

Provides factories for creating synthetic models.

Synthetic Model Structure:
    A synthetic model created by synthetic_model_factory() contains:
    - config.json: Model configuration (model_type, hidden_size, etc.)
    - tokenizer.json: BPE tokenizer with ASCII + special tokens
    - tokenizer_config.json: Chat template and special token definitions
    - model.safetensors: Random weights in safetensors format

    The model directory structure mimics a standard HuggingFace model.
    See tests/reference/helpers/model_factory.py for implementation details.

Usage:
    from tests.reference.fixtures import synthetic_model_factory

    model = synthetic_model_factory(vocab_size=1000, hidden_size=64)
    session = talu.Chat(str(model.path))
    model.cleanup()  # Or use as context manager
"""

from tests.reference.helpers import create_minimal_model


def synthetic_model_factory(
    vocab_size: int = 1000,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_heads: int = 2,
    intermediate_size: int = 128,
    model_type: str = "llama",
):
    """
    Create a synthetic model for deterministic testing.

    This creates a minimal model with known weights that can be used
    for testing without requiring external model downloads.

    Args:
        vocab_size: Vocabulary size (default: 1000, must be >= 100)
        hidden_size: Hidden dimension (default: 64, must be divisible by num_heads)
        num_layers: Number of transformer layers (default: 2, must be >= 1)
        num_heads: Number of attention heads (default: 2, must be >= 1)
        intermediate_size: FFN intermediate size (default: 128, must be >= hidden_size)
        model_type: Model type string (default: "llama")

    Returns:
        SyntheticModel with .path attribute pointing to model directory

    Raises:
        ValueError: If parameter constraints are violated
    """
    # Validate parameters to catch misconfigured tests early
    if vocab_size < 100:
        raise ValueError(f"vocab_size must be >= 100, got {vocab_size}")
    if num_heads < 1:
        raise ValueError(f"num_heads must be >= 1, got {num_heads}")
    if hidden_size % num_heads != 0:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        )
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if intermediate_size < hidden_size:
        raise ValueError(
            f"intermediate_size ({intermediate_size}) should be >= hidden_size ({hidden_size})"
        )

    model = create_minimal_model(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        model_type=model_type,
    )

    # Post-creation validation: verify expected files exist
    assert (model.path / "config.json").exists(), "config.json not created"
    assert (model.path / "tokenizer.json").exists(), "tokenizer.json not created"
    assert (model.path / "model.safetensors").exists(), "model.safetensors not created"

    return model
