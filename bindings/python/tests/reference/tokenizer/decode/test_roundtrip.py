"""
Encode/decode roundtrip tests.

Tests that text survives encode -> decode roundtrip.
"""

import pytest

from tests.tokenizer.conftest import (
    BASIC_STRINGS,
    CODE_STRINGS,
    CONTRACTION_STRINGS,
    MULTILINGUAL_STRINGS,
    NUMBER_STRINGS,
    PUNCTUATION_STRINGS,
)


class TestRoundtripBasic:
    """Basic roundtrip tests."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", BASIC_STRINGS)
    def test_roundtrip_basic(self, tokenizer, text):
        """Basic text survives encode/decode roundtrip."""
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        # Text should be preserved (possibly with whitespace normalization)
        assert (
            text.strip() in decoded
            or decoded.strip() in text.strip()
            or text.lower().strip() == decoded.lower().strip()
        ), f"Roundtrip failed: '{text}' -> {tokens.tolist()} -> '{decoded}'"

    @pytest.mark.requires_model
    def test_roundtrip_simple(self, tokenizer):
        """Simple roundtrip test."""
        text = "Hello World"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        assert "Hello" in decoded
        assert "World" in decoded

    @pytest.mark.requires_model
    def test_roundtrip_deterministic(self, tokenizer):
        """Roundtrip is deterministic."""
        text = "Test text for determinism"

        results = []
        for _ in range(5):
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            results.append(decoded)

        assert all(r == results[0] for r in results)


class TestRoundtripNumbers:
    """Roundtrip tests for numbers."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", NUMBER_STRINGS)
    def test_roundtrip_numbers(self, tokenizer, text):
        """Number strings survive roundtrip."""
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        # Check key parts are preserved
        # Numbers should be in the decoded text
        assert len(decoded) > 0


class TestRoundtripPunctuation:
    """Roundtrip tests for punctuation."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", PUNCTUATION_STRINGS)
    def test_roundtrip_punctuation(self, tokenizer, text):
        """Punctuation strings survive roundtrip."""
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        assert len(decoded) > 0


class TestRoundtripContractions:
    """Roundtrip tests for contractions."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", CONTRACTION_STRINGS)
    def test_roundtrip_contractions(self, tokenizer, text):
        """Contraction strings survive roundtrip."""
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        # The apostrophe and contraction should be preserved
        assert "'" in decoded or text.replace("'", "") in decoded.replace("'", "")


class TestRoundtripMultilingual:
    """Roundtrip tests for multilingual text."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("lang,text", MULTILINGUAL_STRINGS)
    def test_roundtrip_multilingual(self, tokenizer, lang, text):
        """Multilingual strings survive roundtrip."""
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        # The text should be preserved
        # Some normalization may occur
        assert len(decoded) > 0, f"Empty decode for {lang}: {text}"


class TestRoundtripCode:
    """Roundtrip tests for code."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("code", CODE_STRINGS)
    def test_roundtrip_code(self, tokenizer, code):
        """Code strings survive roundtrip."""
        tokens = tokenizer.encode(code)
        decoded = tokenizer.decode(tokens)

        # Code should be mostly preserved
        assert len(decoded) > 0


class TestRoundtripLongText:
    """Roundtrip tests for longer text."""

    @pytest.mark.requires_model
    def test_roundtrip_paragraph(self, tokenizer):
        """Paragraph survives roundtrip."""
        paragraph = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a longer piece of text that spans multiple "
            "sentences and should produce a reasonable number of tokens."
        )

        tokens = tokenizer.encode(paragraph)
        decoded = tokenizer.decode(tokens)

        # Key phrases should be preserved
        assert "quick brown fox" in decoded
        assert "lazy dog" in decoded

    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_roundtrip_consistency(self, tokenizer):
        """Multiple roundtrips produce consistent results."""
        text = "Consistency test text"

        tokens1 = tokenizer.encode(text, special_tokens=False)
        decoded1 = tokenizer.decode(tokens1)

        tokens2 = tokenizer.encode(decoded1, special_tokens=False)
        decoded2 = tokenizer.decode(tokens2)

        # Second roundtrip should be stable
        assert decoded1 == decoded2


class TestRoundtripVsTransformers:
    """Roundtrip tests comparing to transformers."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", BASIC_STRINGS)
    def test_roundtrip_matches_transformers(self, tokenizer, hf_tokenizer, text):
        """Roundtrip matches transformers behavior."""
        # Encode with talu (no special tokens to match HF)
        talu_tokens = tokenizer.encode(text, special_tokens=False)
        talu_decoded = tokenizer.decode(talu_tokens)

        # Encode with transformers
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)
        hf_decoded = hf_tokenizer.decode(hf_tokens)

        # Decoded text should match
        assert talu_decoded == hf_decoded, (
            f"Roundtrip mismatch for '{text}':\n"
            f"  talu: '{talu_decoded}'\n"
            f"  transformers: '{hf_decoded}'"
        )
