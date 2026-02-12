"""
Deterministic conversion tests using synthetic models.

These tests don't require external model downloads and run in any environment.
They verify conversion behavior with known inputs.
"""

import json
from pathlib import Path

import pytest


class TestConvertSyntheticModel:
    """Tests conversion using synthetic models with known properties."""

    def test_convert_synthetic_produces_output(
        self, convert_func, synthetic_model, temp_output_dir
    ):
        """Conversion of synthetic model produces output directory."""
        result_path = convert_func(
            str(synthetic_model.path),
            scheme="gaf4_64",
            output_dir=temp_output_dir,
            force=True,
        )

        assert Path(result_path).exists()
        assert Path(result_path).is_dir()

    def test_convert_synthetic_has_config(self, convert_func, synthetic_model, temp_output_dir):
        """Converted model has config.json."""
        result_path = convert_func(
            str(synthetic_model.path),
            scheme="gaf4_64",
            output_dir=temp_output_dir,
            force=True,
        )

        config_path = Path(result_path) / "config.json"
        assert config_path.exists()

    def test_convert_synthetic_has_weights(self, convert_func, synthetic_model, temp_output_dir):
        """Converted model has model.safetensors."""
        result_path = convert_func(
            str(synthetic_model.path),
            scheme="gaf4_64",
            output_dir=temp_output_dir,
            force=True,
        )

        weights_path = Path(result_path) / "model.safetensors"
        assert weights_path.exists()

    def test_convert_synthetic_has_tokenizer(self, convert_func, synthetic_model, temp_output_dir):
        """Converted model has tokenizer.json."""
        result_path = convert_func(
            str(synthetic_model.path),
            scheme="gaf4_64",
            output_dir=temp_output_dir,
            force=True,
        )

        tokenizer_path = Path(result_path) / "tokenizer.json"
        assert tokenizer_path.exists()

    def test_convert_preserves_config_values(self, convert_func, synthetic_model, temp_output_dir):
        """Conversion preserves config values from source model."""
        result_path = convert_func(
            str(synthetic_model.path),
            scheme="gaf4_64",
            output_dir=temp_output_dir,
            force=True,
        )

        with open(Path(result_path) / "config.json") as f:
            config = json.load(f)

        # Values should match synthetic model
        assert config["vocab_size"] == synthetic_model.vocab_size
        assert config["hidden_size"] == synthetic_model.hidden_size
        assert config["num_hidden_layers"] == synthetic_model.num_layers
        assert config["num_attention_heads"] == synthetic_model.num_heads


class TestConvertSchemes:
    """Test different quantization schemes with synthetic models."""

    @pytest.mark.parametrize(
        "scheme", ["gaf4_32", "gaf4_64", "gaf4_128", "gaf8_32", "gaf8_64", "gaf8_128"]
    )
    def test_gaf_scheme_produces_output(
        self, convert_func, synthetic_model, temp_output_dir, scheme
    ):
        """Each GAF scheme produces valid output."""
        result_path = convert_func(
            str(synthetic_model.path),
            scheme=scheme,
            output_dir=temp_output_dir,
            force=True,
        )

        assert Path(result_path).exists()
        assert (Path(result_path) / "config.json").exists()
        assert (Path(result_path) / "model.safetensors").exists()


class TestConvertOutputNaming:
    """Test output directory naming conventions."""

    def test_output_name_contains_scheme(self, convert_func, synthetic_model, temp_output_dir):
        """Output directory name includes scheme type."""
        result_path = convert_func(
            str(synthetic_model.path),
            scheme="gaf4_64",
            output_dir=temp_output_dir,
            force=True,
        )

        # GAF schemes use GAF naming (e.g., GAF4, GAF8-G128)
        assert "GAF4" in result_path or "gaf4" in result_path.lower()


class TestConvertMetadataStability:
    """Tests that conversion produces identical metadata across runs.

    For reproducible builds, config.json and tokenizer.json should be
    byte-for-byte identical across multiple conversions of the same model.
    """

    def test_config_json_identical_across_runs(
        self, convert_func, synthetic_model, temp_output_dir
    ):
        """config.json is identical across two conversions."""
        import tempfile
        from pathlib import Path

        # First conversion
        path1 = convert_func(
            str(synthetic_model.path),
            scheme="gaf4_64",
            output_dir=temp_output_dir,
            force=True,
        )
        config1 = (Path(path1) / "config.json").read_text()

        # Second conversion to different directory
        with tempfile.TemporaryDirectory() as temp_dir2:
            path2 = convert_func(
                str(synthetic_model.path),
                scheme="gaf4_64",
                output_dir=temp_dir2,
                force=True,
            )
            config2 = (Path(path2) / "config.json").read_text()

        assert config1 == config2, (
            "config.json differs between runs - conversion is non-deterministic"
        )

    def test_tokenizer_json_identical_across_runs(
        self, convert_func, synthetic_model, temp_output_dir
    ):
        """tokenizer.json is identical across two conversions."""
        import tempfile
        from pathlib import Path

        # First conversion
        path1 = convert_func(
            str(synthetic_model.path),
            scheme="gaf4_64",
            output_dir=temp_output_dir,
            force=True,
        )
        tokenizer1 = (Path(path1) / "tokenizer.json").read_text()

        # Second conversion to different directory
        with tempfile.TemporaryDirectory() as temp_dir2:
            path2 = convert_func(
                str(synthetic_model.path),
                scheme="gaf4_64",
                output_dir=temp_dir2,
                force=True,
            )
            tokenizer2 = (Path(path2) / "tokenizer.json").read_text()

        assert tokenizer1 == tokenizer2, (
            "tokenizer.json differs between runs - conversion is non-deterministic"
        )
