"""
Fixtures for convert tests.

Fixture Categories:
- Unit fixtures: Used for API/parameter validation without real models
  - convert_func, Converter, ConvertError, temp_output_dir

- Deterministic fixtures: Synthetic models with known properties
  - synthetic_model: Creates a minimal model with fixed seed for reproducible tests

- Integration fixtures: Require real cached models (mark tests with @pytest.mark.requires_model)
  - small_test_model: Finds unquantized model in cache, returns None if not found
  - small_hf_model_id: HuggingFace ID for download tests (mark with @pytest.mark.network)
"""

import shutil
import tempfile

import pytest

from tests.fixtures.storage import (
    fake_hf_cache_factory,
    incomplete_model_factory,
)
from tests.reference.fixtures import find_unquantized_model_path, synthetic_model_factory

# =============================================================================
# Unit Test Fixtures (no external dependencies)
# =============================================================================


@pytest.fixture
def temp_output_dir():
    """
    Create a temporary directory for conversion output.

    Automatically cleaned up after test.
    """
    temp_dir = tempfile.mkdtemp(prefix="talu_convert_test_")
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def convert_func(talu):
    """Return the convert function for API testing."""
    return talu.convert


@pytest.fixture
def ConvertError():
    """Return the ConvertError exception class."""
    from talu.exceptions import ConvertError as CE

    return CE


# =============================================================================
# Deterministic Fixtures (synthetic models with known properties)
# =============================================================================


@pytest.fixture
def synthetic_model():
    """
    Create a synthetic model for deterministic conversion tests.

    Uses fixed seed (42) for reproducible weight generation.
    No network access required.
    """
    return synthetic_model_factory()


# =============================================================================
# Integration Fixtures (require cached models)
# =============================================================================


@pytest.fixture
def small_test_model():
    """
    Get path to a small UNQUANTIZED test model for conversion tests.

    Conversion tests require an unquantized source model (f16/bf16/f32).
    Returns None if no suitable model is found.

    Tests using this fixture should:
    - Be marked with @pytest.mark.requires_model
    - Use pytest.skip() if small_test_model is None (infrastructure unavailable)
    """
    return find_unquantized_model_path()


@pytest.fixture
def small_hf_model_id():
    """
    Return a small HuggingFace model ID for testing downloads.

    This model will be downloaded if not cached.

    Tests using this fixture should be marked with @pytest.mark.network.
    """
    from tests.conftest import TEST_MODEL_URI_TEXT_RANDOM

    return TEST_MODEL_URI_TEXT_RANDOM


# =============================================================================
# Fake HF Cache Fixtures (for describe tests)
# =============================================================================


@pytest.fixture
def fake_hf_cache(tmp_path, monkeypatch):
    """Create a fake HuggingFace cache with known models."""
    return fake_hf_cache_factory(tmp_path, monkeypatch)


@pytest.fixture
def incomplete_model_missing_config(tmp_path, monkeypatch):
    """Create a cache entry missing config.json."""
    return incomplete_model_factory(tmp_path, monkeypatch, missing="config")


@pytest.fixture
def incomplete_model_missing_tokenizer(tmp_path, monkeypatch):
    """Create a cache entry missing tokenizer.json."""
    return incomplete_model_factory(tmp_path, monkeypatch, missing="tokenizer")
