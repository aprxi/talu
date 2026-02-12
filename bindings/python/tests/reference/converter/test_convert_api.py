"""
Tests for convert API with real models.

These tests require a model and validate actual conversion behavior.
API parameter validation tests are in tests/converter/test_api.py.
"""

import pytest


class TestConvertApiWithModel:
    """Tests for convert() that require a real model."""

    @pytest.mark.slow
    @pytest.mark.requires_model
    def test_returns_string_path(self, convert_func, small_test_model, temp_output_dir):
        """convert() returns a string path."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result = convert_func(
            small_test_model,
            output_dir=temp_output_dir,
            force=True,
        )
        assert isinstance(result, str)


class TestConvertForceOverwrite:
    """Tests for force overwrite functionality."""

    @pytest.mark.slow
    @pytest.mark.requires_model
    def test_force_overwrites_existing(self, convert_func, small_test_model, temp_output_dir):
        """force=True overwrites existing output."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        # First conversion
        path1 = convert_func(
            small_test_model,
            output_dir=temp_output_dir,
            force=True,
        )

        # Second conversion with force should succeed
        path2 = convert_func(
            small_test_model,
            output_dir=temp_output_dir,
            force=True,
        )

        # Both should return valid paths
        assert path1
        assert path2

    @pytest.mark.slow
    @pytest.mark.requires_model
    def test_without_force_is_idempotent(self, convert_func, small_test_model, temp_output_dir):
        """Without force=True, convert is idempotent (returns existing path)."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        # First conversion
        path1 = convert_func(
            small_test_model,
            output_dir=temp_output_dir,
            force=True,
        )

        # Second conversion without force returns same path (no error)
        path2 = convert_func(
            small_test_model,
            output_dir=temp_output_dir,
            force=False,
        )
        assert path2 == path1


class TestConvertWithModel:
    """Tests for convert() function that require a real model."""

    @pytest.mark.slow
    @pytest.mark.requires_model
    def test_convert_returns_string(self, convert_func, small_test_model, temp_output_dir):
        """convert() returns string path."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result = convert_func(small_test_model, output_dir=temp_output_dir, force=True)
        assert isinstance(result, str)
