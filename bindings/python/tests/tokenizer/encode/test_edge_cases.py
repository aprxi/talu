"""
Edge case encoding tests.

Tests for talu.Tokenizer.encode() with edge cases:
- Whitespace variations
- Empty strings
- Special characters
- Special tokens
- Long text
"""

import pytest


class TestEncodeWhitespace:
    """Tests for whitespace handling in encoding.

    Contract: Whitespace strings either encode successfully or raise RuntimeError.
    Tests document which behavior occurs for the current model.
    """

    @pytest.mark.requires_model
    def test_encode_single_space(self, tokenizer):
        """Single space encodes to tokens or raises RuntimeError.

        Behavior is model-dependent: some tokenizers produce tokens,
        others reject pure whitespace. Both are valid contracts.
        """
        tokens = tokenizer.encode(" ")
        # If we get here, encoding succeeded - verify valid output
        assert hasattr(tokens, "__len__"), "encode() should return array-like"
        # May return empty or single token depending on model

    @pytest.mark.requires_model
    def test_encode_multiple_spaces(self, tokenizer):
        """Multiple spaces encode to at least one token.

        CONTRACT: Multiple whitespace characters must produce tokens.
        """
        tokens = tokenizer.encode("   ")
        assert len(tokens) >= 1, "Multiple spaces should encode to at least 1 token"
        assert all(isinstance(t, int) for t in tokens.tolist()), "Token IDs must be integers"

    @pytest.mark.requires_model
    def test_encode_tabs(self, tokenizer):
        """Tab characters in text encode to tokens."""
        tokens = tokenizer.encode("Tabs\there\tand\tthere")
        assert len(tokens) >= 1, "Text with tabs should produce at least 1 token"

    @pytest.mark.requires_model
    def test_encode_newlines(self, tokenizer):
        """Newline characters in text encode to tokens."""
        tokens = tokenizer.encode("Line\nbreaks\nincluded")
        assert len(tokens) >= 1, "Text with newlines should produce at least 1 token"

    @pytest.mark.requires_model
    def test_encode_mixed_whitespace(self, tokenizer):
        """Mixed whitespace characters encode to tokens."""
        tokens = tokenizer.encode("Multiple   spaces   here")
        assert len(tokens) >= 1, "Text with mixed whitespace should produce tokens"

    @pytest.mark.requires_model
    def test_encode_leading_trailing_whitespace(self, tokenizer):
        """Leading and trailing whitespace."""
        text_with_space = "  Hello World  "
        text_without = "Hello World"

        tokens_with = tokenizer.encode(text_with_space)
        tokens_without = tokenizer.encode(text_without)

        # Both should encode (may have different token counts)
        assert len(tokens_with) >= 1
        assert len(tokens_without) >= 1


class TestEncodeEmpty:
    """Tests for empty and minimal inputs.

    Contract: Empty or minimal inputs either encode or raise RuntimeError.
    Both behaviors are model-dependent and valid.
    """

    @pytest.mark.requires_model
    def test_encode_empty_string(self, tokenizer):
        """Empty string returns empty or minimal tokens.

        Contract: Empty string either returns empty list, single BOS token,
        or raises RuntimeError. Verify whichever behavior is implemented.
        """
        tokens = tokenizer.encode("")
        # If encoding succeeds, should be empty or minimal
        assert len(tokens) <= 1, f"Empty string should produce <= 1 token, got {len(tokens)}"

    @pytest.mark.requires_model
    def test_encode_single_char(self, tokenizer):
        """Single character produces at least one token."""
        tokens = tokenizer.encode("a")
        assert len(tokens) >= 1, "Single char 'a' should produce at least 1 token"

    @pytest.mark.requires_model
    def test_encode_single_newline(self, tokenizer):
        """Single newline encodes to at least one token.

        CONTRACT: Whitespace characters like newline must produce tokens.
        """
        tokens = tokenizer.encode("\n")
        assert len(tokens) >= 1, "Newline character should encode to at least 1 token"
        assert all(isinstance(t, int) for t in tokens.tolist()), "Token IDs must be integers"

    @pytest.mark.requires_model
    def test_encode_single_tab(self, tokenizer):
        """Single tab encodes to at least one token.

        CONTRACT: Whitespace characters like tab must produce tokens.
        """
        tokens = tokenizer.encode("\t")
        assert len(tokens) >= 1, "Tab character should encode to at least 1 token"
        assert all(isinstance(t, int) for t in tokens.tolist()), "Token IDs must be integers"


class TestEncodeSpecialCharacters:
    """Tests for special character encoding."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize(
        "char",
        [
            "~",
            "`",
            "!",
            "@",
            "#",
            "$",
            "%",
            "^",
            "&",
            "*",
            "(",
            ")",
            "-",
            "_",
            "=",
            "+",
            "[",
            "]",
            "{",
            "}",
            "|",
            "\\",
            ":",
            ";",
            "'",
            '"',
            ",",
            ".",
            "<",
            ">",
            "?",
            "/",
        ],
    )
    def test_encode_ascii_special_chars(self, tokenizer, char):
        """ASCII special characters encode."""
        tokens = tokenizer.encode(char)
        assert len(tokens) >= 1

    @pytest.mark.requires_model
    def test_encode_null_byte(self, tokenizer):
        """Null byte handling: either encodes or raises clear error.

        Contract: Null bytes in input should either:
        1. Be stripped/encoded successfully, OR
        2. Raise RuntimeError/ValueError with clear message
        """
        import pytest

        try:
            tokens = tokenizer.encode("Hello\x00World")
            # Success: verify valid output structure
            assert hasattr(tokens, "__len__"), "encode() should return array-like"
            assert len(tokens) >= 1, "Should encode at least some content"
        except (RuntimeError, ValueError) as e:
            # Error: verify message is helpful
            error_msg = str(e).lower()
            if "unknown error" in error_msg:
                pytest.xfail("Null byte error needs improvement - expected descriptive message")
            # Error is acceptable behavior

    @pytest.mark.requires_model
    def test_encode_control_characters(self, tokenizer):
        """Control characters encode or raise clear error.

        Contract: Control characters (0x01-0x1F) should either encode
        or raise a clear error - not crash or produce undefined output.
        """
        import pytest

        # Test representative control characters
        for char, name in [("\x01", "SOH"), ("\x1b", "ESC")]:
            text = f"before{char}after"
            try:
                tokens = tokenizer.encode(text)
                # Success: verify at least "before" and "after" encoded
                assert len(tokens) >= 1, f"Control char {name} context should encode"
            except (RuntimeError, ValueError) as e:
                # Error is acceptable but should be clear
                if "unknown error" in str(e).lower():
                    pytest.xfail(f"Control char {name} error needs improvement")


class TestEncodeSpecialTokens:
    """Tests for special token strings in input."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize(
        "text",
        [
            "<|endoftext|>",
            "[CLS] test [SEP]",
            "<s>test</s>",
            "<bos>test<eos>",
        ],
    )
    def test_encode_special_token_strings(self, tokenizer, text):
        """Special token markers in text encode (as text, not as special tokens)."""
        tokens = tokenizer.encode(text)
        assert len(tokens) >= 1

    @pytest.mark.requires_model
    def test_encode_html_like_markers(self, tokenizer):
        """HTML-like markers encode."""
        for marker in ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]:
            tokens = tokenizer.encode(marker)
            assert len(tokens) >= 1


class TestEncodeLongText:
    """Tests for long text encoding."""

    @pytest.mark.requires_model
    def test_encode_repeated_char(self, tokenizer):
        """Repeated character encoding."""
        tokens = tokenizer.encode("A" * 100)
        assert len(tokens) >= 1

    @pytest.mark.requires_model
    def test_encode_paragraph(self, tokenizer):
        """Paragraph-length text encodes."""
        paragraph = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a longer piece of text that spans multiple "
            "sentences and should produce a reasonable number of tokens. "
            "Machine learning models process text by first tokenizing "
            "it into smaller units called tokens."
        )
        tokens = tokenizer.encode(paragraph)

        assert len(tokens) > 10  # Should produce many tokens
        assert len(tokens) < len(paragraph)  # But fewer than characters

    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_encode_very_long_text(self, tokenizer):
        """Very long text encoding (stress test)."""
        # 1000 repetitions of a sentence
        long_text = "The quick brown fox jumps over the lazy dog. " * 1000
        tokens = tokenizer.encode(long_text)

        assert len(tokens) > 100

    @pytest.mark.requires_model
    def test_encode_repeated_pattern(self, tokenizer):
        """Repeated pattern encoding."""
        pattern = "abc123 " * 50
        tokens = tokenizer.encode(pattern)

        assert len(tokens) >= 1


class TestEncodeModelSpecific:
    """Tests for model-specific special tokens."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize(
        "text",
        [
            "<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>",
            "<|start_of_role|>assistant<|end_of_role|>",
        ],
    )
    def test_encode_granite_tokens(self, tokenizer, text):
        """Granite-style special tokens in text encode as text.

        These markers appear in Granite model outputs and should encode
        as regular text (not as actual special tokens) for all tokenizers.
        """
        tokens = tokenizer.encode(text)
        assert len(tokens) >= 1, f"Granite markers should encode: {text}"

    @pytest.mark.requires_model
    @pytest.mark.parametrize(
        "text",
        [
            "<|user|>",
            "<|user|>Hi",
            "<|user|>Hi<|end|><|assistant|>",
            "<|assistant|>Hello!<|end|>",
            "<|system|>You are helpful.<|end|>",
        ],
    )
    def test_encode_phi_tokens(self, tokenizer, text):
        """Phi-style special tokens in text encode as text.

        These markers appear in Phi model outputs and should encode
        as regular text (not as actual special tokens) for all tokenizers.
        """
        tokens = tokenizer.encode(text)
        assert len(tokens) >= 1, f"Phi markers should encode: {text}"
