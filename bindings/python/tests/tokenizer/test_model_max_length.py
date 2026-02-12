"""Tests for model_max_length and automatic truncation defaults.

Tests the tokenizer's ability to:
1. Read model_max_length from tokenizer_config.json
2. Use it as default when truncation=True without explicit max_length
3. Handle models that don't have model_max_length defined

This provides a user-friendly truncation experience where users can simply
pass truncation=True without needing to know the model's context length.
"""

import pytest

from talu.tokenizer import Tokenizer


class TestModelMaxLength:
    """Tests for model_max_length property."""

    @pytest.fixture
    def tokenizer(self, test_model_path):
        """Create a Tokenizer for testing."""
        return Tokenizer(test_model_path)

    def test_model_max_length_exists(self, tokenizer):
        """model_max_length property exists and returns an integer."""
        max_len = tokenizer.model_max_length
        assert isinstance(max_len, int)
        assert max_len > 0

    def test_model_max_length_reasonable_value(self, tokenizer):
        """model_max_length is a reasonable context length (not 0 or negative)."""
        max_len = tokenizer.model_max_length
        # Most models have context lengths between 512 and 1M tokens
        assert 512 <= max_len <= 2_000_000

    def test_model_max_length_matches_config(self, tokenizer):
        """model_max_length matches value from tokenizer_config.json."""
        import json
        from pathlib import Path

        config_path = Path(tokenizer.model_path) / "tokenizer_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            if "model_max_length" in config:
                expected = config["model_max_length"]
                assert tokenizer.model_max_length == expected


class TestTruncationWithModelMaxLength:
    """Tests for automatic truncation using model_max_length."""

    @pytest.fixture
    def tokenizer(self, test_model_path):
        """Create a Tokenizer for testing."""
        return Tokenizer(test_model_path)

    def test_truncation_true_without_max_length_uses_model_max_length(self, tokenizer):
        """truncation=True without max_length uses model_max_length."""
        # This should NOT raise an error anymore
        result = tokenizer("Hello world", truncation=True)

        # Should return a valid BatchEncoding
        from talu.tokenizer import BatchEncoding

        assert isinstance(result, BatchEncoding)

    def test_truncation_true_with_explicit_max_length_uses_explicit(self, tokenizer):
        """truncation=True with explicit max_length uses the explicit value."""
        # Use encode() for single string truncation (batch encoding truncates at tensor level)
        result = tokenizer.encode(
            "Hello world this is a longer text", truncation=True, max_length=5
        )

        # Get the token IDs
        ids = result.tolist()
        assert len(ids) == 5

    def test_truncation_actually_truncates_long_text(self, tokenizer):
        """Long text is actually truncated when truncation=True."""
        # Create a very long text
        long_text = "Hello world. " * 10000  # Should be way over most context limits

        # With truncation, should work without error
        result = tokenizer(long_text, truncation=True)

        # Result should be at most model_max_length
        ids = result.to_list()["input_ids"][0]
        assert len(ids) <= tokenizer.model_max_length

    def test_encode_truncation_without_max_length_uses_model_max_length(self, tokenizer):
        """encode() with truncation=True without max_length uses model_max_length."""
        long_text = "Hello world. " * 10000

        # Should not raise
        tokens = tokenizer.encode(long_text, truncation=True)

        # Should be truncated to model_max_length
        assert len(tokens) <= tokenizer.model_max_length

    def test_batch_encode_truncation_without_max_length(self, tokenizer):
        """Batch encode with truncation=True without max_length works."""
        texts = [
            "Short text",
            "Hello world. " * 10000,  # Very long
            "Medium length text here",
        ]

        # Should not raise
        result = tokenizer.encode(texts, truncation=True)

        # All sequences should be at most model_max_length
        for i, length in enumerate(result.lengths()):
            assert length <= tokenizer.model_max_length, f"Sequence {i} too long: {length}"


class TestModelMaxLengthEdgeCases:
    """Edge case tests for model_max_length."""

    @pytest.fixture
    def tokenizer(self, test_model_path):
        """Create a Tokenizer for testing."""
        return Tokenizer(test_model_path)

    def test_explicit_max_length_smaller_than_model_max(self, tokenizer):
        """Explicit max_length smaller than model_max_length is respected."""
        small_max = 10
        assert small_max < tokenizer.model_max_length

        result = tokenizer("Hello world this is text", truncation=True, max_length=small_max)
        ids = result.to_list()["input_ids"][0]
        assert len(ids) <= small_max

    def test_explicit_max_length_larger_than_model_max(self, tokenizer):
        """Explicit max_length larger than model_max_length is still respected."""
        # User explicitly requests a larger max_length - we respect it
        # (The model might not be able to process it, but that's the user's choice)
        large_max = tokenizer.model_max_length + 1000

        # Short text - should work fine
        result = tokenizer("Hi", truncation=True, max_length=large_max)
        ids = result.to_list()["input_ids"][0]
        # Short text won't hit the limit anyway
        assert len(ids) < large_max

    def test_truncation_false_ignores_model_max_length(self, tokenizer):
        """truncation=False does not apply model_max_length limit."""
        # Medium-length text that won't hit model_max_length
        text = "Hello world " * 100

        result_no_trunc = tokenizer(text, truncation=False)
        result_with_trunc = tokenizer(text, truncation=True)

        # Both should work
        ids_no_trunc = result_no_trunc.to_list()["input_ids"][0]
        ids_with_trunc = result_with_trunc.to_list()["input_ids"][0]

        # Without truncation: full length
        # With truncation: might be same if under limit, or truncated if over
        assert len(ids_no_trunc) >= len(ids_with_trunc)

    def test_truncation_without_max_length_raises_when_no_model_max(self, tokenizer):
        """truncation=True without max_length raises when model_max_length is 0."""
        from unittest.mock import PropertyMock, patch

        from talu.exceptions import ValidationError

        # Mock model_max_length to return 0
        with patch.object(type(tokenizer), "model_max_length", new_callable=PropertyMock) as mock:
            mock.return_value = 0

            with pytest.raises(ValidationError, match="truncation=True requires max_length"):
                tokenizer("Hello world", truncation=True)

    def test_unknown_kwargs_logged_as_warning(self, tokenizer):
        """Unknown kwargs are logged as warnings."""
        from unittest.mock import patch

        # Capture log messages
        with patch("talu.tokenizer.tokenizer.logger") as mock_logger:
            tokenizer("Hello", unknown_arg=42)

            # Check that warning was called with unknown argument message
            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args
            assert "Unknown argument" in call_args[0][0]
