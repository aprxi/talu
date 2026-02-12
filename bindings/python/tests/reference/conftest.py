"""
Validation test fixtures requiring PyTorch/NumPy.

These tests run on Python 3.14+ and validate talu against reference implementations.
"""

import sys

import pytest

from tests.conftest import TEST_MODEL_URI_TEXT

# =============================================================================
# Python Version Enforcement
# =============================================================================

# Validation tests require Python 3.14+ (enforced via .python-version file)
# This check provides a clear error if someone bypasses .python-version
if sys.version_info < (3, 14):
    pytest.exit(
        f"Validation tests require Python 3.14+, got {'.'.join(map(str, sys.version_info[:2]))}.\n"
        "The project uses .python-version to enforce Python 3.14.\n"
        "Run with: uv run pytest tests/reference/",
        returncode=1,
    )


# =============================================================================
# PyTorch Fixtures (for correctness tests)
# =============================================================================


@pytest.fixture(scope="session")
def torch():
    """Import and return torch (required dependency)."""
    import torch

    return torch


@pytest.fixture(scope="session")
def numpy():
    """Import and return numpy (required dependency)."""
    import numpy as np

    return np


# =============================================================================
# Model Path Override
# =============================================================================
# Reference tests validate against real models (HuggingFace transformers, etc.)
# and need a capable model, not the tiny-random default from TEST_MODEL_URI_TEXT.


@pytest.fixture(scope="session")
def test_model_path():
    """Override test_model_path for reference tests to use the reasoning model."""
    return TEST_MODEL_URI_TEXT


# =============================================================================
# Tokenizer Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def tokenizer(talu, test_model_path):
    """Get talu Tokenizer for the test model."""
    return talu.Tokenizer(test_model_path)


# =============================================================================
# DLPack Interchange Fixtures
# =============================================================================


@pytest.fixture
def torch_tensor_factory(torch):
    """
    Factory for creating PyTorch tensors with specific properties.

    Returns a function that creates tensors with given shape, dtype, etc.
    """

    def factory(
        shape,
        dtype=None,
        fill=None,
        requires_grad=False,
        seed=None,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        if dtype is None:
            dtype = torch.float32

        if fill is not None:
            tensor = torch.full(shape, fill, dtype=dtype)
        else:
            tensor = torch.randn(shape, dtype=dtype)

        tensor.requires_grad = requires_grad
        return tensor

    return factory


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def activation_test_values(numpy):
    """Test values for activation function validation."""
    np = numpy
    return {
        "positive": np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float32),
        "negative": np.array([-0.5, -1.0, -2.0, -3.0], dtype=np.float32),
        "mixed": np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "edge": np.array([-1e6, -1e-6, 0.0, 1e-6, 1e6], dtype=np.float32),
    }


# =============================================================================
# Tolerance Fixtures
# =============================================================================


@pytest.fixture
def float32_tolerance():
    """Tolerance for float32 comparisons (element-wise ops).

    For element-wise operations like activations, normalization.
    """
    return {"rtol": 1e-4, "atol": 1e-5}


@pytest.fixture
def matmul_tolerance():
    """Tolerance for matmul/linear operations.

    SIMD-optimized matmul implementations may have larger numerical
    differences (~1e-4 to 1e-3) compared to PyTorch due to different
    accumulation order. Errors scale with matrix size.
    """
    return {"rtol": 2e-3, "atol": 1e-3}


@pytest.fixture
def float16_tolerance():
    """Tolerance for float16 comparisons (looser than float32)."""
    return {"rtol": 1e-3, "atol": 1e-4}


@pytest.fixture
def bfloat16_tolerance():
    """Tolerance for bfloat16 comparisons."""
    return {"rtol": 1e-2, "atol": 1e-3}


@pytest.fixture
def quantized_tolerance():
    """Tolerance for quantized operation comparisons (much looser)."""
    return {"rtol": 0.1, "atol": 0.05}


@pytest.fixture
def mxfp4_tolerance():
    """Tolerance for MXFP4 microscaling format (4-bit with E8M0 scales).

    MXFP4 uses per-group E8M0 scaling which provides better precision
    than block-based Q4 formats. The 4-bit values range from -7 to +7
    (16 levels), so quantization error is bounded by 1/14 of the group's
    max value (±3.5% of max for well-scaled values).

    Mathematical bounds for E8M0 microscaling:
    - Each group of 32 elements shares an E8M0 scale (8-bit exponent only)
    - 4-bit values: {-7, -6, ..., 0, ..., +6, +7} (15 levels around zero)
    - Max quantization step: scale / 7

    The high rtol was due to dividing by small expected values near zero.
    Using atol handles near-zero cases, rtol handles larger values.

    Bounds:
    - atol=0.25: Absolute tolerance for near-zero expected values.
      Covers max observed diff ~0.18 with 40% headroom.
    - rtol=0.15: Relative tolerance (15%) for larger values.
      MXFP4 should achieve ~10% relative error for well-scaled data.
      15% provides headroom for edge cases without accepting garbage.

    NOTE: If tests fail with these tighter bounds, investigate:
    1. Are inputs well-scaled (values not too small relative to group max)?
    2. Is the quantization implementation correct?
    """
    return {"rtol": 0.15, "atol": 0.25}


@pytest.fixture
def q4_tolerance():
    """Tolerance for Q4 block quantized operations (extra loose).

    Q4 formats use block-level scaling which can accumulate more
    quantization error than microscaling approaches like MXFP4.

    Empirically validated bounds (seed=42, 27 configurations tested):
    - atol=1.5: Covers max absolute diff of ~1.29 for test sizes (32x8)
      Note: Larger matrices (128x32) can have diffs up to ~3.4
    - rtol=0.35: Relative tolerance. High rtol values occur when
      expected values approach zero (div-by-small-number effect)

    These tolerances are appropriate for the small test matrices used.
    Production code should validate with representative model sizes.
    """
    return {"rtol": 0.35, "atol": 1.5}


@pytest.fixture
def attention_shapes():
    """Common attention tensor shapes for testing."""
    return {
        "tiny": {
            "batch": 1,
            "n_heads": 2,
            "seq_len": 4,
            "head_dim": 8,
        },
        "small": {
            "batch": 1,
            "n_heads": 4,
            "seq_len": 16,
            "head_dim": 32,
        },
        "medium": {
            "batch": 2,
            "n_heads": 8,
            "seq_len": 64,
            "head_dim": 64,
        },
    }


@pytest.fixture
def matmul_shapes():
    """Common matmul tensor shapes for testing."""
    return [
        # (M, K, N)
        (1, 32, 32),  # Single token
        (4, 64, 64),  # Small batch
        (16, 128, 256),  # Medium
        (32, 512, 512),  # Larger
        (1, 4096, 4096),  # Single token, large hidden dim
    ]
