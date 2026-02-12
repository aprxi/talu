"""
Model description tests.

Tests for talu.converter.describe() and ModelInfo.
"""

import pytest

import talu
from talu.converter import describe
from tests.reference.helpers import create_minimal_model


class TestDescribe:
    """Tests for describe function."""

    @pytest.mark.requires_model
    def test_describe_returns_model_info(self, test_model_path):
        """describe() returns ModelInfo object."""
        info = describe(test_model_path)

        assert info is not None
        assert hasattr(info, "vocab_size")
        assert hasattr(info, "hidden_size")
        assert hasattr(info, "num_layers")
        assert hasattr(info, "num_heads")

    @pytest.mark.requires_model
    def test_describe_valid_values(self, test_model_path):
        """describe() returns valid configuration values."""
        info = describe(test_model_path)

        assert info.vocab_size > 0
        assert info.hidden_size > 0
        assert info.num_layers > 0
        assert info.num_heads > 0
        assert info.head_dim > 0

    def test_describe_invalid_path(self):
        """describe() raises error for invalid path."""
        with pytest.raises(talu.ModelError):
            describe("/nonexistent/model/path")

    @pytest.mark.requires_model
    def test_model_info_properties(self, test_model_path):
        """ModelInfo has expected properties.

        CONTRACT: For real models (test_model_path), model_type must be
        a non-empty string identifying the architecture family.
        """
        info = describe(test_model_path)

        # Check boolean properties are actual bools
        assert isinstance(info.is_quantized, bool)
        assert isinstance(info.is_moe, bool)

        # CONTRACT: Real models must have model_type defined
        assert isinstance(info.model_type, str), (
            f"model_type must be a string, got {type(info.model_type)}"
        )
        assert len(info.model_type) > 0, "model_type must not be empty for a valid model"

        # architecture may be None (derived from model_type) or string
        assert info.architecture is None or isinstance(info.architecture, str)

    @pytest.mark.requires_model
    def test_model_info_repr(self, test_model_path):
        """ModelInfo has informative repr."""
        info = describe(test_model_path)

        repr_str = repr(info)

        assert "ModelInfo" in repr_str
        assert "layers=" in repr_str
        assert "hidden=" in repr_str


class TestDescribeDeterministic:
    """Deterministic tests using synthetic models with known config values."""

    def test_describe_exact_values(self):
        """describe() returns exact values from config."""
        model = create_minimal_model(
            vocab_size=1234,
            hidden_size=256,
            num_layers=4,
            num_heads=8,
            intermediate_size=512,
        )

        info = describe(str(model.path))

        assert info.vocab_size == 1234
        assert info.hidden_size == 256
        assert info.num_layers == 4
        assert info.num_heads == 8
        assert info.head_dim == 256 // 8  # hidden_size / num_heads

    def test_model_info_invariants(self):
        """ModelInfo maintains consistent invariants across field values."""
        model = create_minimal_model(
            vocab_size=5000,
            hidden_size=512,
            num_layers=6,
            num_heads=8,
        )

        info = describe(str(model.path))

        # head_dim = hidden_size / num_heads
        assert info.head_dim == info.hidden_size // info.num_heads

        # All numeric fields must be positive
        assert info.vocab_size > 0
        assert info.hidden_size > 0
        assert info.num_layers > 0
        assert info.num_heads > 0
        assert info.head_dim > 0

        # Boolean fields are actual bools, not truthy values
        assert info.is_quantized is False or info.is_quantized is True
        assert info.is_moe is False or info.is_moe is True

    def test_describe_different_configs(self):
        """describe() works with various config values."""
        configs = [
            {"vocab_size": 500, "hidden_size": 32, "num_layers": 1, "num_heads": 1},
            {"vocab_size": 10000, "hidden_size": 512, "num_layers": 8, "num_heads": 16},
            {"vocab_size": 32000, "hidden_size": 128, "num_layers": 2, "num_heads": 4},
        ]

        for cfg in configs:
            model = create_minimal_model(**cfg)
            info = describe(str(model.path))

            assert info.vocab_size == cfg["vocab_size"]
            assert info.hidden_size == cfg["hidden_size"]
            assert info.num_layers == cfg["num_layers"]
            assert info.num_heads == cfg["num_heads"]

    def test_describe_model_type(self):
        """describe() returns correct model_type."""
        model = create_minimal_model(model_type="llama")
        info = describe(str(model.path))

        assert info.model_type == "llama"

    def test_describe_not_quantized(self):
        """describe() correctly identifies non-quantized model."""
        model = create_minimal_model()
        info = describe(str(model.path))

        assert info.is_quantized is False

    def test_describe_not_moe(self):
        """describe() correctly identifies non-MoE model."""
        model = create_minimal_model()
        info = describe(str(model.path))

        assert info.is_moe is False


class TestDescribeEdgeCases:
    """Edge case and error handling tests."""

    def test_describe_empty_path(self):
        """describe() raises error for empty path."""
        with pytest.raises(talu.ModelError):
            describe("")

    def test_describe_directory_without_config(self, tmp_path):
        """describe() raises error for directory without config.json."""
        # Create empty directory
        model_dir = tmp_path / "empty_model"
        model_dir.mkdir()

        with pytest.raises(talu.ModelError):
            describe(str(model_dir))

    def test_describe_file_instead_of_directory(self, tmp_path):
        """describe() raises error when given a file instead of directory."""
        # Create a file instead of directory
        file_path = tmp_path / "not_a_model.txt"
        file_path.write_text("not a model")

        with pytest.raises(talu.ModelError):
            describe(str(file_path))


class TestDescribeFromFakeCache:
    """Tests for describe() using fake HF cache with exact known values.

    Uses model IDs defined in tests/fixtures/storage.py:
    - test/model-a: llama, hidden=64, layers=2, heads=4, vocab=1000
    - org/model-b: qwen, hidden=128, layers=4, vocab=2000
    """

    def test_describe_exact_values_from_fake_cache(self, fake_hf_cache):
        """describe() returns exact values from fake cache config."""
        # Use the model path directly from the fixture info
        # This avoids relying on LocalStorage which may not see the fake cache
        model_path = fake_hf_cache["info"]["test/model-a"]["path"]

        info = describe(model_path)

        # Assert exact values from fake_hf_cache_factory
        # (these are the values defined in tests/fixtures/storage.py)
        assert info.vocab_size == 1000, f"Expected vocab_size=1000, got {info.vocab_size}"
        assert info.hidden_size == 64, f"Expected hidden_size=64, got {info.hidden_size}"
        assert info.num_layers == 2, f"Expected num_layers=2, got {info.num_layers}"
        assert info.num_heads == 4, f"Expected num_heads=4, got {info.num_heads}"
        assert info.head_dim == 16, f"Expected head_dim=16 (64/4), got {info.head_dim}"

    def test_describe_model_type_from_fake_cache(self, fake_hf_cache):
        """describe() returns correct model_type from fake cache."""
        # Use the model path directly from the fixture info
        model_path = fake_hf_cache["info"]["test/model-a"]["path"]

        info = describe(model_path)

        # fake_hf_cache_factory uses model_type="llama" for test/model-a
        assert info.model_type == "llama", f"Expected model_type='llama', got {info.model_type}"

    def test_describe_second_model_from_fake_cache(self, fake_hf_cache):
        """describe() works with different model in fake cache."""
        # Use org/model-b which has different config values
        model_path = fake_hf_cache["info"]["org/model-b"]["path"]

        info = describe(model_path)

        # org/model-b has: hidden=128, layers=4, vocab=2000, type=qwen
        assert info.vocab_size == 2000, f"Expected vocab_size=2000, got {info.vocab_size}"
        assert info.hidden_size == 128, f"Expected hidden_size=128, got {info.hidden_size}"
        assert info.num_layers == 4, f"Expected num_layers=4, got {info.num_layers}"
        assert info.model_type == "qwen", f"Expected model_type='qwen', got {info.model_type}"


class TestDescribeIncompleteModels:
    """Tests for describe() with incomplete model cache entries."""

    def test_missing_config_raises_error(self, incomplete_model_missing_config):
        """describe() raises ModelError when config.json is missing."""
        model_path = incomplete_model_missing_config["model_path"]

        with pytest.raises(talu.ModelError) as exc_info:
            describe(model_path)

        # Error message should mention config or file not found
        error_msg = str(exc_info.value).lower()
        assert "config" in error_msg or "not found" in error_msg or "file" in error_msg, (
            f"Error should mention config.json, got: {exc_info.value}"
        )

    def test_missing_tokenizer_still_describes(self, incomplete_model_missing_tokenizer):
        """describe() works without tokenizer.json (only needs config)."""
        model_path = incomplete_model_missing_tokenizer["model_path"]

        # describe() only needs config.json, not tokenizer
        info = describe(model_path)

        assert info is not None
        assert info.vocab_size > 0
        assert info.hidden_size > 0

    def test_missing_config_error_type(self, incomplete_model_missing_config):
        """Missing config raises appropriate error type."""
        model_path = incomplete_model_missing_config["model_path"]

        # Should raise ModelError (which is a TaluError subclass)
        with pytest.raises(talu.ModelError):
            describe(model_path)
