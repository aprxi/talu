"""
Repository test fixtures.

Provides fixtures for deterministic Repository testing without relying on
real HuggingFace cache or network access.
"""

import pytest

from tests.fixtures.storage import (
    empty_hf_cache_factory,
    fake_hf_cache_factory,
    incomplete_model_factory,
)


@pytest.fixture
def fake_hf_cache(tmp_path, monkeypatch):
    """Create a fake HuggingFace cache with known models."""
    return fake_hf_cache_factory(tmp_path, monkeypatch)


@pytest.fixture
def empty_hf_cache(tmp_path, monkeypatch):
    """Create an empty HuggingFace cache."""
    return empty_hf_cache_factory(tmp_path, monkeypatch)


@pytest.fixture
def incomplete_model_missing_config(tmp_path, monkeypatch):
    """Create a cache entry missing config.json."""
    return incomplete_model_factory(tmp_path, monkeypatch, missing="config")


@pytest.fixture
def incomplete_model_missing_tokenizer(tmp_path, monkeypatch):
    """Create a cache entry missing tokenizer.json."""
    return incomplete_model_factory(tmp_path, monkeypatch, missing="tokenizer")


@pytest.fixture
def incomplete_model_missing_weights(tmp_path, monkeypatch):
    """Create a cache entry missing model weights."""
    return incomplete_model_factory(tmp_path, monkeypatch, missing="weights")
