"""
Stress tests for encoding.

Tests for handling large inputs and edge cases:
- Very long text
- Many tokens
- Repeated patterns
- Random content
"""

import string

import pytest

from tests.tokenizer.conftest import (
    get_available_models,
    load_hf_tokenizer,
    load_tokenizer,
)

# =============================================================================
# Models that can't be tested for exact token matching
# =============================================================================
#
# ministral3:
#   - TokenizersBackend requires transformers v5 (not yet released)
#   - Talu works fine with Ministral, just can't compare against HF
#
SKIP_MODELS = {"ministral3"}

# Known model-specific issues for stress tests
KNOWN_ISSUES = {
    # No known stress test issues currently - all tests pass
}


def get_testable_models():
    """Get models that can be tested."""
    available = get_available_models()
    return [(name, hf_id, path) for name, hf_id, path in available if name not in SKIP_MODELS]


def should_xfail(model_name, category):
    """Check if a test should be marked as xfail based on known issues."""
    if (model_name, category) in KNOWN_ISSUES:
        return KNOWN_ISSUES[(model_name, category)]
    return None


class TestLongText:
    """Tests for long text encoding."""

    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_encode_1000_words(self, tokenizer, hf_tokenizer):
        """Encode 1000 words correctly."""
        # Generate 1000 words
        words = ["word"] * 1000
        text = " ".join(words)

        talu_tokens = tokenizer.encode(text).tolist()
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"Token mismatch for 1000 words:\n"
            f"  talu length: {len(talu_tokens)}\n"
            f"  hf length: {len(hf_tokens)}"
        )

    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_encode_10000_chars(self, tokenizer, hf_tokenizer):
        """Encode 10000 characters correctly."""
        # Generate 10000 chars of mixed content
        base = "Hello World 123 日本語 "
        text = (base * (10000 // len(base) + 1))[:10000]

        talu_tokens = tokenizer.encode(text).tolist()
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens

    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_encode_paragraph_multiple_times(self, tokenizer, hf_tokenizer):
        """Repeated encoding produces consistent results."""
        paragraph = (
            "The quick brown fox jumps over the lazy dog. "
            "Machine learning models process text by tokenization. "
            "This is a test of the emergency broadcast system."
        )

        for _ in range(10):
            talu_tokens = tokenizer.encode(paragraph).tolist()
            hf_tokens = hf_tokenizer.encode(paragraph, add_special_tokens=False)
            assert talu_tokens == hf_tokens


class TestRepeatedPatterns:
    """Tests for repeated pattern encoding."""

    @pytest.mark.requires_model
    def test_single_char_repeated(self, tokenizer, hf_tokenizer):
        """Repeated single character encodes correctly."""
        for char in ["a", "A", "1", " ", "."]:
            text = char * 100

            talu_tokens = tokenizer.encode(text).tolist()
            hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

            assert talu_tokens == hf_tokens, f"Mismatch for '{char}' x 100"

    @pytest.mark.requires_model
    def test_word_repeated(self, tokenizer, hf_tokenizer):
        """Repeated word encodes correctly."""
        for word in ["hello", "test", "123"]:
            text = (word + " ") * 50

            talu_tokens = tokenizer.encode(text).tolist()
            hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

            assert talu_tokens == hf_tokens, f"Mismatch for '{word}' x 50"

    @pytest.mark.requires_model
    def test_pattern_repeated(self, tokenizer, hf_tokenizer):
        """Repeated pattern encodes correctly."""
        patterns = [
            "abc123",
            "Hello, World!",
            "日本語テスト",
            "🎉🚀",
        ]

        for pattern in patterns:
            text = pattern * 20

            talu_tokens = tokenizer.encode(text).tolist()
            hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

            assert talu_tokens == hf_tokens, f"Mismatch for pattern '{pattern[:10]}' x 20"


class TestMixedContent:
    """Tests for mixed content encoding."""

    @pytest.mark.requires_model
    def test_alternating_languages(self, tokenizer, hf_tokenizer):
        """Alternating language text encodes correctly."""
        text = "Hello 你好 Bonjour こんにちは Привет مرحبا " * 10

        talu_tokens = tokenizer.encode(text).tolist()
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens

    @pytest.mark.requires_model
    def test_code_and_prose(self, tokenizer, hf_tokenizer):
        """Mixed code and prose encodes correctly."""
        text = """
        Here is some text explaining the code below:

        def hello_world():
            print("Hello, World!")

        The function above prints a greeting. Now let's look at another example:

        for i in range(10):
            print(i * 2)

        This loop prints even numbers from 0 to 18.
        """

        talu_tokens = tokenizer.encode(text).tolist()
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens

    @pytest.mark.requires_model
    def test_numbers_and_text(self, tokenizer, hf_tokenizer):
        """Mixed numbers and text encodes correctly."""
        text = "The year 2024 has 365 days. Pi is approximately 3.14159. "
        text += "There are 1,000,000 microseconds in a second. "
        text += "Scientific notation: 6.022e23 atoms per mole."

        talu_tokens = tokenizer.encode(text).tolist()
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens


class TestAllModelsStress:
    """Stress tests across all models."""

    @pytest.mark.requires_model
    @pytest.mark.slow
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    def test_long_text_all_models(
        self,
        model_name,
        hf_id,
        model_path,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """Long text encodes correctly across all models."""
        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        # 500 words
        text = " ".join(["testing"] * 500)

        talu_tokens = tok.encode(text).tolist()
        hf_tokens = hf_tok.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"[{model_name}] Token mismatch for long text:\n"
            f"  talu length: {len(talu_tokens)}\n"
            f"  hf length: {len(hf_tokens)}"
        )

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    def test_all_ascii_printable(
        self,
        model_name,
        hf_id,
        model_path,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """All ASCII printable characters encode correctly."""
        reason = should_xfail(model_name, "ascii_printable")
        if reason:
            pytest.xfail(reason)

        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        # All printable ASCII
        text = string.printable

        talu_tokens = tok.encode(text).tolist()
        hf_tokens = hf_tok.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, f"[{model_name}] Token mismatch for ASCII printable"

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    def test_unicode_blocks(
        self,
        model_name,
        hf_id,
        model_path,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """Various Unicode blocks encode correctly."""
        reason = should_xfail(model_name, "unicode_blocks")
        if reason:
            pytest.xfail(reason)

        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        # Sample from various Unicode blocks
        samples = [
            "ÀÁÂÃÄÅÆÇÈÉÊË",  # Latin Extended
            "ΑΒΓΔΕΖΗΘΙΚΛΜ",  # Greek
            "АБВГДЕЁЖЗИЙК",  # Cyrillic
            "אבגדהוזחטיכל",  # Hebrew
            "あいうえおかきくけこ",  # Hiragana
            "アイウエオカキクケコ",  # Katakana
            "一二三四五六七八九十",  # CJK
        ]

        for sample in samples:
            talu_tokens = tok.encode(sample).tolist()
            hf_tokens = hf_tok.encode(sample, add_special_tokens=False)

            assert talu_tokens == hf_tokens, (
                f"[{model_name}] Token mismatch for Unicode: '{sample[:10]}...'"
            )


class TestEdgeCasesStress:
    """Edge case stress tests."""

    @pytest.mark.requires_model
    def test_many_newlines(self, tokenizer, hf_tokenizer):
        """Many newlines encode correctly."""
        text = "\n".join(["line"] * 100)

        talu_tokens = tokenizer.encode(text).tolist()
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens

    @pytest.mark.requires_model
    def test_many_tabs(self, tokenizer, hf_tokenizer):
        """Many tabs encode correctly."""
        text = "\t".join(["word"] * 100)

        talu_tokens = tokenizer.encode(text).tolist()
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens

    @pytest.mark.requires_model
    def test_mixed_whitespace(self, tokenizer, hf_tokenizer):
        """Mixed whitespace patterns encode correctly."""
        text = "a  b   c\td\t\te\n\nf\r\ng"

        talu_tokens = tokenizer.encode(text).tolist()
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens

    @pytest.mark.requires_model
    def test_emoji_sequences(self, tokenizer, hf_tokenizer):
        """Complex emoji sequences encode correctly."""
        emojis = [
            "👨‍👩‍👧‍👦",  # Family
            "🏳️‍🌈",  # Rainbow flag
            "👍🏽",  # Thumbs up with skin tone
            "🇺🇸",  # Flag
            "👩‍💻",  # Woman technologist
        ]
        text = " ".join(emojis * 5)

        talu_tokens = tokenizer.encode(text).tolist()
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens


class TestNullBytes:
    """Tests for null byte handling in tokenization.

    These tests verify that null bytes (0x00) in input are properly
    tokenized instead of being treated as string terminators.
    """

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    def test_null_byte_in_input(
        self,
        model_name,
        hf_id,
        model_path,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """Null byte should encode correctly."""
        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        text = "\x00"
        talu_tokens = tok.encode(text).tolist()
        hf_tokens = hf_tok.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"[{model_name}] Null byte encoding mismatch:\n  talu: {talu_tokens}\n  hf: {hf_tokens}"
        )

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    def test_null_byte_in_middle(
        self,
        model_name,
        hf_id,
        model_path,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """Null byte in middle of text should be tokenized correctly."""
        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        text = "Hello\x00World"
        talu_tokens = tok.encode(text).tolist()
        hf_tokens = hf_tok.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"[{model_name}] Null byte in middle mismatch:\n"
            f"  talu: {talu_tokens}\n"
            f"  hf: {hf_tokens}"
        )

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    def test_multiple_null_bytes(
        self,
        model_name,
        hf_id,
        model_path,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """Multiple null bytes should each be tokenized."""
        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        text = "\x00\x00\x00"
        talu_tokens = tok.encode(text).tolist()
        hf_tokens = hf_tok.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"[{model_name}] Multiple null bytes mismatch:\n"
            f"  talu: {talu_tokens}\n"
            f"  hf: {hf_tokens}"
        )
