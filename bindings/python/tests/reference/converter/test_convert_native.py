"""
Tests for GAF scheme conversion.

Tests conversion using grouped affine quantization schemes (gaf4_32, gaf4_64, gaf4_128, etc.).
GAF schemes are the default and offer excellent quality with MLX compatibility.
"""

import json
from pathlib import Path

import pytest

# All tests in this module are slow integration tests requiring a cached model
pytestmark = [pytest.mark.slow, pytest.mark.requires_model]


class TestGAFSchemeConversion:
    """Tests for GAF scheme conversion."""

    @pytest.mark.slow
    def test_convert_gaf4_64(self, convert_func, small_test_model, temp_output_dir):
        """Convert to GAF4_64 scheme (default)."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="gaf4_64",
            output_dir=temp_output_dir,
            force=True,
        )

        assert Path(result_path).exists()
        # Should have config.json
        assert (Path(result_path) / "config.json").exists()
        # Should have model weights
        assert (Path(result_path) / "model.safetensors").exists()

    @pytest.mark.slow
    def test_convert_gaf4_32(self, convert_func, small_test_model, temp_output_dir):
        """Convert to GAF4_32 scheme (higher accuracy)."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="gaf4_32",
            output_dir=temp_output_dir,
            force=True,
        )

        assert Path(result_path).exists()
        assert (Path(result_path) / "config.json").exists()

    @pytest.mark.slow
    def test_convert_gaf8_64(self, convert_func, small_test_model, temp_output_dir):
        """Convert to GAF8_64 scheme (near-lossless)."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="gaf8_64",
            output_dir=temp_output_dir,
            force=True,
        )

        assert Path(result_path).exists()
        assert (Path(result_path) / "config.json").exists()

    @pytest.mark.slow
    def test_convert_gaf4_128(self, convert_func, small_test_model, temp_output_dir):
        """Convert to GAF4_128 scheme (smallest)."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="gaf4_128",
            output_dir=temp_output_dir,
            force=True,
        )

        assert Path(result_path).exists()


class TestGAFSchemeOutputStructure:
    """Tests for GAF scheme output structure."""

    @pytest.mark.slow
    def test_output_has_required_files(self, convert_func, small_test_model, temp_output_dir):
        """GAF scheme output has all required files."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="gaf4_64",
            output_dir=temp_output_dir,
            force=True,
        )

        result_dir = Path(result_path)
        assert result_dir.exists()

        # Required files
        assert (result_dir / "config.json").exists()
        assert (result_dir / "model.safetensors").exists()
        assert (result_dir / "tokenizer.json").exists()

    @pytest.mark.slow
    def test_config_contains_model_info(self, convert_func, small_test_model, temp_output_dir):
        """Config should contain model information (copied from original)."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="gaf4_64",
            output_dir=temp_output_dir,
            force=True,
        )

        config_path = Path(result_path) / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        # Config should have core model info (copied from source)
        assert "model_type" in config
        assert "vocab_size" in config
        assert "hidden_size" in config

    @pytest.mark.slow
    def test_gaf_scheme_smaller_than_original(self, convert_func, synthetic_model, temp_output_dir):
        """Quantized model should be significantly smaller than original.

        STRICT SIZE CHECK for GAF4_64 quantization:

        Expected compression ratio:
        - Original: 32-bit (float32) = 4 bytes per weight
        - GAF4_64: 4-bit + scales/biases = ~0.5-0.6 bytes per weight
        - Expected ratio: 12-20% of original

        We use 35% threshold (< original_size * 0.35) which:
        - Catches bugs that accidentally preserve too much precision
        - Allows for quantization overhead (scales, biases, metadata)
        - Allows for small models where overhead is proportionally larger

        Uses synthetic_model (hidden_size=64) instead of a cached model
        to ensure dimensions are divisible by the GAF4_64 group size (64).
        Models with hidden_size < 64 cannot be quantized with this scheme.

        If this test fails, investigate:
        1. Is quantization actually being applied?
        2. Are scales/biases being stored efficiently?
        3. Is there duplicate data in the safetensors?
        """
        # Get original size
        original_weights = synthetic_model.path / "model.safetensors"
        original_size = original_weights.stat().st_size

        # Convert to 4-bit
        result_path = convert_func(
            str(synthetic_model.path),
            scheme="gaf4_64",
            output_dir=temp_output_dir,
            force=True,
        )

        # Get converted size
        converted_weights = Path(result_path) / "model.safetensors"
        converted_size = converted_weights.stat().st_size

        # Calculate actual compression ratio for diagnostics
        actual_ratio = converted_size / original_size

        # GAF4_64 should achieve significant compression
        # We use 35% threshold to allow for overhead while catching conversion bugs
        assert converted_size < original_size * 0.35, (
            f"GAF4_64 conversion achieved only {actual_ratio:.1%} compression "
            f"(converted: {converted_size / 1024:.1f}KB, original: {original_size / 1024:.1f}KB). "
            f"Expected <35% - quantization may not be applied correctly."
        )


class TestDefaultScheme:
    """Tests for default scheme behavior."""

    @pytest.mark.slow
    def test_default_uses_gaf4_64(self, convert_func, small_test_model, temp_output_dir):
        """Without explicit scheme, uses default gaf4_64 quantization."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            # No scheme= uses default (gaf4_64)
            output_dir=temp_output_dir,
            force=True,
        )

        assert Path(result_path).exists()
        # Default is GAF4 quantization
        assert "GAF4" in result_path or "gaf4" in result_path.lower()
