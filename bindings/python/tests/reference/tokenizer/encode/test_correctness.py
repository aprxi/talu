"""
Tokenizer correctness tests against transformers.

Compares talu.Tokenizer.encode() output against HuggingFace transformers
for exact token matching. This is the ground truth validation.
"""

import pytest

from tests.tokenizer.conftest import (
    BASIC_STRINGS,
    CODE_STRINGS,
    CONTRACTION_STRINGS,
    EDGE_CASE_STRINGS,
    MULTILINGUAL_STRINGS,
    NUMBER_STRINGS,
    PUNCTUATION_STRINGS,
    SPECIAL_TOKEN_STRINGS,
    UNICODE_STRINGS,
    WHITESPACE_STRINGS,
)


class TestCorrectnessBasic:
    """Correctness tests for basic text."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", BASIC_STRINGS)
    def test_basic_strings_match(self, tokenizer, hf_tokenizer, text):
        """Basic strings tokenize identically to transformers."""
        talu_tokens = tokenizer.encode(text).tolist()
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"Token mismatch for '{text[:50]}...'\n"
            f"  talu ({len(talu_tokens)}): {talu_tokens[:10]}...\n"
            f"  transformers ({len(hf_tokens)}): {hf_tokens[:10]}..."
        )


class TestCorrectnessNumbers:
    """Correctness tests for numbers and math."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", NUMBER_STRINGS)
    def test_number_strings_match(self, tokenizer, hf_tokenizer, text):
        """Number strings tokenize identically to transformers."""
        talu_tokens = tokenizer.encode(text).tolist()
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"Token mismatch for '{text[:50]}'\n  talu: {talu_tokens}\n  transformers: {hf_tokens}"
        )


class TestCorrectnessPunctuation:
    """Correctness tests for punctuation."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", PUNCTUATION_STRINGS)
    def test_punctuation_strings_match(self, tokenizer, hf_tokenizer, text):
        """Punctuation strings tokenize identically to transformers."""
        talu_tokens = tokenizer.encode(text).tolist()
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"Token mismatch for '{text[:50]}'\n  talu: {talu_tokens}\n  transformers: {hf_tokens}"
        )


class TestCorrectnessContractions:
    """Correctness tests for contractions (critical for GPT-2 tokenizers)."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", CONTRACTION_STRINGS)
    def test_contraction_strings_match(self, tokenizer, hf_tokenizer, text):
        """Contraction strings tokenize identically to transformers."""
        talu_tokens = tokenizer.encode(text).tolist()
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"Token mismatch for '{text[:50]}'\n  talu: {talu_tokens}\n  transformers: {hf_tokens}"
        )


class TestCorrectnessWhitespace:
    """Correctness tests for whitespace variations."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", WHITESPACE_STRINGS)
    def test_whitespace_strings_match(self, tokenizer, hf_tokenizer, text):
        """Whitespace strings tokenize identically to transformers."""
        talu_tokens = tokenizer.encode(text).tolist()
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"Token mismatch for '{repr(text)[:50]}'\n"
            f"  talu: {talu_tokens}\n"
            f"  transformers: {hf_tokens}"
        )


class TestCorrectnessUnicode:
    """Correctness tests for Unicode text."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", UNICODE_STRINGS)
    def test_unicode_strings_match(self, tokenizer, hf_tokenizer, text):
        """Unicode strings tokenize identically to transformers."""
        talu_tokens = tokenizer.encode(text).tolist()
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"Token mismatch for '{text[:50]}'\n  talu: {talu_tokens}\n  transformers: {hf_tokens}"
        )


class TestCorrectnessMultilingual:
    """Correctness tests for multilingual text."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("lang,text", MULTILINGUAL_STRINGS)
    def test_multilingual_strings_match(self, tokenizer, hf_tokenizer, lang, text):
        """Multilingual strings tokenize identically to transformers."""
        talu_tokens = tokenizer.encode(text).tolist()
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"Token mismatch for {lang}: '{text[:50]}'\n"
            f"  talu: {talu_tokens}\n"
            f"  transformers: {hf_tokens}"
        )


class TestCorrectnessCode:
    """Correctness tests for code snippets."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", CODE_STRINGS)
    def test_code_strings_match(self, tokenizer, hf_tokenizer, text):
        """Code strings tokenize identically to transformers."""
        talu_tokens = tokenizer.encode(text).tolist()
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"Token mismatch for '{text[:50]}'\n  talu: {talu_tokens}\n  transformers: {hf_tokens}"
        )


class TestCorrectnessEdgeCases:
    """Correctness tests for edge cases."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", [t for t in EDGE_CASE_STRINGS if t])
    def test_edge_case_strings_match(self, tokenizer, hf_tokenizer, text):
        """Non-empty edge case strings tokenize identically to transformers."""
        talu_tokens = tokenizer.encode(text).tolist()
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"Token mismatch for '{repr(text)[:50]}'\n"
            f"  talu: {talu_tokens}\n"
            f"  transformers: {hf_tokens}"
        )


class TestCorrectnessSpecialTokens:
    """Correctness tests for special token strings."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", SPECIAL_TOKEN_STRINGS)
    def test_special_token_strings_match(self, tokenizer, hf_tokenizer, text):
        """Special token strings tokenize identically to transformers."""
        talu_tokens = tokenizer.encode(text).tolist()
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"Token mismatch for '{text[:50]}'\n  talu: {talu_tokens}\n  transformers: {hf_tokens}"
        )


class TestCorrectnessComprehensive:
    """Comprehensive correctness tests combining all categories."""

    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_all_strings_match(self, tokenizer, hf_tokenizer, all_test_strings):
        """All test strings tokenize identically to transformers."""
        failures = []

        for text in all_test_strings:
            if not text:  # Skip empty
                continue

            try:
                talu_tokens = tokenizer.encode(text).tolist()
                hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

                if talu_tokens != hf_tokens:
                    failures.append(
                        {
                            "text": text,
                            "talu": talu_tokens,
                            "hf": hf_tokens,
                        }
                    )
            except Exception as e:
                failures.append(
                    {
                        "text": text,
                        "error": str(e),
                    }
                )

        if failures:
            msg = f"{len(failures)} mismatches:\n"
            for f in failures[:5]:  # Show first 5
                if "error" in f:
                    msg += f"  '{f['text'][:30]}...': ERROR {f['error']}\n"
                else:
                    msg += f"  '{f['text'][:30]}...': talu={f['talu'][:5]}... hf={f['hf'][:5]}...\n"
            pytest.fail(msg)

    @pytest.mark.requires_model
    @pytest.mark.parametrize(
        "text",
        [
            "Hello",
            "Hello World",
            "The quick brown fox",
            "A" * 100,
        ],
        ids=["single_word", "two_words", "sentence", "repeated_char"],
    )
    def test_token_count_reasonable(self, tokenizer, hf_tokenizer, text):
        """Token counts match transformers exactly."""
        talu_tokens = tokenizer.encode(text)
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        # Token counts should match exactly
        assert len(talu_tokens) == len(hf_tokens), (
            f"Token count mismatch for '{text[:20]}': talu={len(talu_tokens)}, hf={len(hf_tokens)}"
        )
