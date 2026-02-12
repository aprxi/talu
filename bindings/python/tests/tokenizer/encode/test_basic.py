"""
Basic encoding tests.

Tests for talu.Tokenizer.encode() with basic text inputs.
"""

import pytest


class TestEncodeBasic:
    """Basic encoding functionality tests."""

    @pytest.mark.requires_model
    def test_encode_simple_text(self, tokenizer):
        """Encode simple text produces tokens."""
        tokens = tokenizer.encode("Hello, world!")

        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens.tolist())

    @pytest.mark.requires_model
    @pytest.mark.parametrize(
        "text",
        [
            "Hello, world!",
            "What is the capital of France?",
            "The quick brown fox jumps over the lazy dog.",
        ],
    )
    def test_encode_basic_strings(self, tokenizer, text):
        """Encode basic strings produces non-empty tokens."""
        tokens = tokenizer.encode(text)

        assert len(tokens) > 0
        assert len(tokens) < len(text)  # Tokens should be compressed

    @pytest.mark.requires_model
    def test_encode_single_char(self, tokenizer):
        """Single character encoding produces at least one token."""
        tokens = tokenizer.encode("a")

        assert len(tokens) >= 1

    @pytest.mark.requires_model
    def test_encode_returns_token_array(self, tokenizer, talu):
        """encode() returns TokenArray type."""
        tokens = tokenizer.encode("Hello")

        assert type(tokens).__name__ == "TokenArray"
        assert hasattr(tokens, "tolist")
        assert hasattr(tokens, "__len__")
        assert hasattr(tokens, "__getitem__")

    @pytest.mark.requires_model
    def test_encode_deterministic(self, tokenizer):
        """Same text produces same tokens every time."""
        text = "Test text for determinism"

        results = [tokenizer.encode(text).tolist() for _ in range(5)]

        assert all(r == results[0] for r in results)


class TestEncodeNumbers:
    """Tests for encoding numbers and math expressions."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize(
        "text",
        [
            "What is 2+2?",
            "The answer is 42.",
            "3.14159 is approximately pi.",
            "100,000 people attended.",
        ],
    )
    def test_encode_numbers(self, tokenizer, text):
        """Number strings encode successfully."""
        tokens = tokenizer.encode(text)

        assert len(tokens) > 0

    @pytest.mark.requires_model
    def test_encode_pure_numbers(self, tokenizer):
        """Pure numeric strings encode."""
        for num_str in ["123", "3.14", "-42", "1,000,000"]:
            tokens = tokenizer.encode(num_str)
            assert len(tokens) >= 1


class TestEncodePunctuation:
    """Tests for encoding punctuation-heavy text."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize(
        "text",
        [
            "Wait... what?!",
            'He said: "Hello!"',
            "It's a test's test.",
            "email@example.com",
            "https://example.com/path?query=value&other=123",
        ],
    )
    def test_encode_punctuation(self, tokenizer, text):
        """Punctuation-heavy text encodes successfully."""
        tokens = tokenizer.encode(text)

        assert len(tokens) > 0


class TestEncodeContractions:
    """Tests for encoding contractions (important for GPT-2 style tokenizers)."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize(
        "text",
        [
            "I'm going to the store.",
            "We've been waiting.",
            "They're not here.",
            "It's John's book.",
            "I'd like that.",
            "We'll see.",
        ],
    )
    def test_encode_contractions(self, tokenizer, text):
        """Contractions encode successfully."""
        tokens = tokenizer.encode(text)

        assert len(tokens) > 0

    @pytest.mark.requires_model
    def test_contraction_vs_full(self, tokenizer):
        """Contractions may tokenize differently than full forms."""
        contracted = tokenizer.encode("I'm")
        full = tokenizer.encode("I am")

        # Just verify both encode - whether they match depends on tokenizer
        assert len(contracted) >= 1
        assert len(full) >= 1


class TestEncodeCode:
    """Tests for encoding code snippets."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize(
        "code",
        [
            "def foo(): return 42",
            "if (x > 0) { print(x); }",
            "<html><body>Test</body></html>",
            "for i in range(10): print(i)",
            "x = lambda y: y * 2",
            "class MyClass:\n    pass",
        ],
    )
    def test_encode_code(self, tokenizer, code):
        """Code snippets encode successfully."""
        tokens = tokenizer.encode(code)

        assert len(tokens) > 0

    @pytest.mark.requires_model
    def test_encode_multiline_code(self, tokenizer):
        """Multiline code encodes correctly."""
        code = """def hello():
    print("Hello, World!")

hello()"""
        tokens = tokenizer.encode(code)

        assert len(tokens) > 0
