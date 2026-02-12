"""
Tests for GAF scheme (grouped affine) conversion.

Tests conversion using grouped affine quantization schemes (gaf4_*, gaf8_*).
GAF schemes are compatible with MLX framework on Mac.
"""

import json
from pathlib import Path

import pytest

# All tests in this module are slow integration tests requiring a cached model
pytestmark = [pytest.mark.slow, pytest.mark.requires_model]


class TestGAFSchemeConversion:
    """Tests for GAF scheme (grouped affine) conversion."""

    @pytest.mark.slow
    def test_convert_gaf4_64(self, convert_func, small_test_model, temp_output_dir):
        """Convert to GAF4_64 scheme (MLX default)."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="gaf4_64",
            output_dir=temp_output_dir,
            force=True,
        )

        assert Path(result_path).exists()
        assert (Path(result_path) / "config.json").exists()
        assert (Path(result_path) / "model.safetensors").exists()

    @pytest.mark.slow
    def test_convert_gaf8_64(self, convert_func, small_test_model, temp_output_dir):
        """Convert to GAF8_64 scheme."""
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
    def test_convert_gaf4_32(self, convert_func, small_test_model, temp_output_dir):
        """Convert to GAF4_32 scheme (highest accuracy)."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="gaf4_32",
            output_dir=temp_output_dir,
            force=True,
        )

        assert Path(result_path).exists()

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

    @pytest.mark.slow
    def test_convert_gaf8_32(self, convert_func, small_test_model, temp_output_dir):
        """Convert to GAF8_32 scheme (8-bit, highest accuracy)."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="gaf8_32",
            output_dir=temp_output_dir,
            force=True,
        )

        assert Path(result_path).exists()
        assert (Path(result_path) / "config.json").exists()

    @pytest.mark.slow
    def test_convert_gaf8_128(self, convert_func, small_test_model, temp_output_dir):
        """Convert to GAF8_128 scheme (8-bit, smallest)."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="gaf8_128",
            output_dir=temp_output_dir,
            force=True,
        )

        assert Path(result_path).exists()
        assert (Path(result_path) / "config.json").exists()


class TestGAFSchemeConstraints:
    """Tests for GAF scheme constraints."""

    def test_gaf_scheme_rejects_overrides(self, convert_func, ConvertError):
        """GAF schemes cannot use overrides parameter."""
        with pytest.raises(ValueError, match="not supported"):
            convert_func(
                "dummy/model",
                scheme="gaf4_64",
                overrides={"model.layers.*": "gaf4_64"},
            )


class TestGAFSchemeOutputStructure:
    """Tests for GAF scheme output structure."""

    @pytest.mark.slow
    def test_gaf_output_has_scales_and_biases(
        self, convert_func, small_test_model, temp_output_dir
    ):
        """GAF scheme output has separate scales and biases tensors."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="gaf4_64",
            output_dir=temp_output_dir,
            force=True,
        )

        # Load the safetensors and check for scales/biases
        try:
            from safetensors import safe_open

            weights_path = Path(result_path) / "model.safetensors"
            with safe_open(weights_path, framework="pt") as f:
                tensor_names = f.keys()

            # Should have .scales and .biases for quantized layers
            has_scales = any(".scales" in name for name in tensor_names)
            has_biases = any(".biases" in name for name in tensor_names)

            assert has_scales, "GAF scheme should have .scales tensors"
            assert has_biases, "GAF scheme should have .biases tensors"

        except ImportError:
            pytest.skip("safetensors library not available")

    @pytest.mark.slow
    def test_gaf_config_has_quantization_info(
        self, convert_func, small_test_model, temp_output_dir
    ):
        """GAF scheme config has quantization section."""
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

        # GAF scheme uses "quantization" key (MLX format)
        assert "quantization" in config, "GAF scheme config should have quantization section"

        quant_config = config["quantization"]
        assert quant_config.get("bits") == 4
        assert quant_config.get("group_size") == 64

    @pytest.mark.slow
    def test_gaf_output_naming_convention(self, convert_func, small_test_model, temp_output_dir):
        """GAF scheme output follows naming convention."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="gaf4_64",
            output_dir=temp_output_dir,
            force=True,
        )

        # GAF schemes use GAF naming (e.g., GAF4, GAF8-G128)
        assert "GAF4" in result_path or "gaf4" in result_path.lower()


class TestGAFGroupSizeVariations:
    """Tests for different GAF group sizes."""

    @pytest.mark.slow
    def test_gaf4_32_config(self, convert_func, small_test_model, temp_output_dir):
        """GAF4_32 scheme with group_size=32."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="gaf4_32",
            output_dir=temp_output_dir,
            force=True,
        )

        config_path = Path(result_path) / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        assert config.get("quantization", {}).get("group_size") == 32

    @pytest.mark.slow
    def test_gaf4_128_config(self, convert_func, small_test_model, temp_output_dir):
        """GAF4_128 scheme with group_size=128."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="gaf4_128",
            output_dir=temp_output_dir,
            force=True,
        )

        config_path = Path(result_path) / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        assert config.get("quantization", {}).get("group_size") == 128

    @pytest.mark.slow
    def test_gaf8_64_config(self, convert_func, small_test_model, temp_output_dir):
        """GAF8_64 scheme with 8-bit quantization."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="gaf8_64",
            output_dir=temp_output_dir,
            force=True,
        )

        config_path = Path(result_path) / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        quant_config = config.get("quantization", {})
        assert quant_config.get("bits") == 8
        assert quant_config.get("group_size") == 64

    @pytest.mark.slow
    def test_gaf8_32_config(self, convert_func, small_test_model, temp_output_dir):
        """GAF8_32 scheme with 8-bit, group_size=32."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="gaf8_32",
            output_dir=temp_output_dir,
            force=True,
        )

        config_path = Path(result_path) / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        quant_config = config.get("quantization", {})
        assert quant_config.get("bits") == 8
        assert quant_config.get("group_size") == 32

    @pytest.mark.slow
    def test_gaf8_128_config(self, convert_func, small_test_model, temp_output_dir):
        """GAF8_128 scheme with 8-bit, group_size=128."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="gaf8_128",
            output_dir=temp_output_dir,
            force=True,
        )

        config_path = Path(result_path) / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        quant_config = config.get("quantization", {})
        assert quant_config.get("bits") == 8
        assert quant_config.get("group_size") == 128
