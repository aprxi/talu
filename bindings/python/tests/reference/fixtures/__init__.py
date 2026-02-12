"""
Model-related test fixtures (requires torch/numpy).

These fixtures run on Python 3.14+ and are used for validation tests.

Maps to: N/A (reference test fixtures)
"""

from tests.fixtures import find_cached_model_path, find_unquantized_model_path

from .models import synthetic_model_factory

__all__ = [
    "synthetic_model_factory",
    "find_cached_model_path",
    "find_unquantized_model_path",
]
