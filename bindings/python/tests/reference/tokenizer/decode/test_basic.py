"""
Basic decoding tests.

Tests for talu.Tokenizer.decode() with various input types.
"""

import pytest


class TestDecodeBasic:
    """Basic decoding functionality tests."""

    @pytest.mark.requires_model
    def test_decode_token_array(self, tokenizer):
        """decode() accepts TokenArray and produces original content."""
        original = "Hello"
        tokens = tokenizer.encode(original)
        decoded = tokenizer.decode(tokens)

        assert isinstance(decoded, str)
        # Decoded text should contain original content (may have spacing differences)
        assert original.lower() in decoded.lower(), (
            f"Decoded '{decoded}' should contain '{original}'"
        )

    @pytest.mark.requires_model
    def test_decode_list_of_ints(self, tokenizer):
        """decode() accepts list of ints and produces correct output."""
        original = "Hello"
        tokens = tokenizer.encode(original)
        token_list = tokens.tolist()

        # Decode from list
        decoded = tokenizer.decode(token_list)

        assert isinstance(decoded, str)
        # Decoded text should match TokenArray decode
        decoded_from_array = tokenizer.decode(tokens)
        assert decoded == decoded_from_array, (
            f"List decode '{decoded}' should match array decode '{decoded_from_array}'"
        )

    @pytest.mark.requires_model
    def test_decode_empty_list(self, tokenizer):
        """decode() handles empty list."""
        decoded = tokenizer.decode([])

        assert decoded == ""

    @pytest.mark.requires_model
    def test_decode_single_token(self, tokenizer):
        """decode() handles single token."""
        tokens = tokenizer.encode("A")
        if len(tokens) > 0:
            decoded = tokenizer.decode([tokens[0]])
            assert isinstance(decoded, str)


class TestDecodeCorrectness:
    """Correctness tests for decoding."""

    @pytest.mark.requires_model
    def test_decode_matches_transformers(self, tokenizer, hf_tokenizer):
        """Decoding produces same result as transformers."""
        text = "Hello, world!"

        # Encode with both
        talu_tokens = tokenizer.encode(text).tolist()
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        # Decode with both
        talu_decoded = tokenizer.decode(talu_tokens)
        hf_decoded = hf_tokenizer.decode(hf_tokens)

        # Should match (or be close - some whitespace normalization may occur)
        assert talu_decoded.strip() == hf_decoded.strip(), (
            f"Decode mismatch:\n  talu: '{talu_decoded}'\n  transformers: '{hf_decoded}'"
        )

    @pytest.mark.requires_model
    @pytest.mark.parametrize(
        "text",
        [
            "Hello, world!",
            "What is 2+2?",
            "The quick brown fox.",
            "def foo(): return 42",
        ],
    )
    def test_decode_various_texts(self, tokenizer, hf_tokenizer, text):
        """Various texts decode correctly."""
        tokens = tokenizer.encode(text).tolist()

        talu_decoded = tokenizer.decode(tokens)
        hf_decoded = hf_tokenizer.decode(tokens)

        assert talu_decoded == hf_decoded


class TestDecodeEdgeCases:
    """Edge case tests for decoding."""

    @pytest.mark.requires_model
    def test_decode_invalid_tokens(self, tokenizer, talu):
        """Decoding invalid token IDs raises descriptive TokenizerError.

        CONTRACT: Invalid token IDs MUST raise TokenizerError with a message
        that includes one of:
        - "token" - indicates the error is token-related
        - "invalid" - indicates the value is invalid
        - "out of range" or "out of bounds" - indicates the value is too large
        """
        # Very large token ID (guaranteed invalid for any vocabulary)
        invalid_token_id = 999999999

        with pytest.raises(talu.TokenizerError) as exc_info:
            tokenizer.decode([invalid_token_id])

        error_msg = str(exc_info.value).lower()

        # CONTRACT: Error message must be descriptive about the token issue
        descriptive_keywords = ["token", "invalid", "out of range", "out of bounds"]
        has_descriptive_message = any(kw in error_msg for kw in descriptive_keywords)

        assert has_descriptive_message, (
            f"TokenizerError for invalid token should contain one of {descriptive_keywords}, "
            f"got: '{exc_info.value}'"
        )

    @pytest.mark.requires_model
    def test_decode_repeated_tokens(self, tokenizer):
        """Decoding repeated tokens produces repeated text."""
        tokens = tokenizer.encode("hello")
        if len(tokens) > 0:
            first_token = tokens[0]
            # Decode single token to get base text
            single_decoded = tokenizer.decode([first_token])

            # Repeat token 5 times
            repeated = [first_token] * 5
            decoded = tokenizer.decode(repeated)

            # Repeated decode should be longer or equal to single
            assert len(decoded) >= len(single_decoded), (
                f"Repeated tokens should produce longer output: "
                f"single='{single_decoded}', repeated='{decoded}'"
            )

    @pytest.mark.requires_model
    def test_decode_preserves_unicode(self, tokenizer):
        """Decoding preserves Unicode characters."""
        text = "Hello 世界!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        # The text should be preserved (possibly with whitespace changes)
        assert "世界" in decoded or decoded.strip() == text.strip()


class TestDecodeTypes:
    """Tests for various input types to decode()."""

    @pytest.mark.requires_model
    def test_decode_tuple(self, tokenizer):
        """decode() accepts tuple converted to list.

        Contract: Tuples can be converted to lists for decoding.
        Since we explicitly convert to list, this must always succeed.
        """
        tokens = tokenizer.encode("Hello")
        token_tuple = tuple(tokens.tolist())

        # Convert tuple to list - this always works
        decoded = tokenizer.decode(list(token_tuple))
        assert isinstance(decoded, str)
        assert len(decoded) > 0, "Decoded tuple should produce non-empty string"
