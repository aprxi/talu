"""
Shared test fixtures for talu (Python 3.10 compatible).

This module provides storage-related fixtures that don't require torch/numpy.
For model-related fixtures (synthetic models), see tests/reference/fixtures/.

Maps to: N/A (shared test fixtures)
"""

from .paths import (
    find_cached_model_path,
    find_unquantized_model_path,
)
from .storage import (
    empty_hf_cache_factory,
    fake_hf_cache_factory,
    incomplete_model_factory,
)

__all__ = [
    # Path utilities
    "find_cached_model_path",
    "find_unquantized_model_path",
    # Storage fixtures
    "fake_hf_cache_factory",
    "empty_hf_cache_factory",
    "incomplete_model_factory",
]
