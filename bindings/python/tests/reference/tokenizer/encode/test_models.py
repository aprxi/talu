"""
Multi-model encode correctness tests.

Tests talu.Tokenizer.encode() across all available models,
comparing against HuggingFace transformers for exact token matching.
"""

import pytest

from tests.tokenizer.conftest import (
    BASIC_STRINGS,
    CODE_STRINGS,
    CONTRACTION_STRINGS,
    MULTILINGUAL_STRINGS,
    NUMBER_STRINGS,
    PUNCTUATION_STRINGS,
    UNICODE_STRINGS,
    WHITESPACE_STRINGS,
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

# =============================================================================
# Known model-specific issues (xfail pattern)
# =============================================================================
#
# Format: (model_name, category, text_pattern) -> reason_string
#
# Keys:
#   model_name: Model identifier from MODEL_REGISTRY (e.g., "llama2", "qwen3")
#   category: Test category matching class name suffix (e.g., "basic", "unicode")
#   text_pattern: Substring match in test text, or None to match all in category
#
# Examples:
#   ("llama2", "unicode", "emoji"): "LLaMA2 tokenizer splits emojis differently"
#   ("qwen3", "whitespace", None): "All whitespace tests fail for Qwen3"
#   ("phi4", "code", "lambda"): "Phi4 BPE differs on lambda expressions"
#
# Policy: Add entries here when a test fails due to known talu limitation.
# These appear as XFAIL in pytest output and alert when fixed (XPASS).
# Include actionable fix hints where possible (e.g., "needs X normalization").
#
# Issue tracking: Optionally add GitHub issue link in reason string:
#   ("model", "cat", "text"): "Description - see #123"
#
KNOWN_ISSUES = {
    # No known encode issues currently - all tests pass
    # When adding entries, follow the format above with explicit reason strings.
}


def get_testable_models():
    """Get models that can be tested (excluding known problematic ones)."""
    available = get_available_models()
    return [(name, hf_id, path) for name, hf_id, path in available if name not in SKIP_MODELS]


def should_xfail(model_name, category, text=None, lang=None):
    """Check if a test should be marked as xfail based on known issues."""
    # Check exact matches
    if (model_name, category, text) in KNOWN_ISSUES:
        return KNOWN_ISSUES[(model_name, category, text)]

    # Check pattern matches
    for (m, c, pattern), reason in KNOWN_ISSUES.items():
        if m == model_name and c == category:
            if pattern is None:
                return reason
            if text and pattern in text:
                return reason
            if lang and pattern in lang:
                return reason

    return None


class TestAllModelsEncodeBasic:
    """Test basic encoding across all models."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    @pytest.mark.parametrize("text", BASIC_STRINGS)
    def test_basic_strings_match(
        self,
        model_name,
        hf_id,
        model_path,
        text,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """Basic strings encode identically across all models."""
        reason = should_xfail(model_name, "basic", text)
        if reason:
            pytest.xfail(reason)

        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        talu_tokens = tok.encode(text).tolist()
        hf_tokens = hf_tok.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"[{model_name}] Token mismatch for '{text[:50]}'\n"
            f"  talu ({len(talu_tokens)}): {talu_tokens[:10]}...\n"
            f"  transformers ({len(hf_tokens)}): {hf_tokens[:10]}..."
        )


class TestAllModelsEncodeNumbers:
    """Test number encoding across all models."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    @pytest.mark.parametrize("text", NUMBER_STRINGS)
    def test_number_strings_match(
        self,
        model_name,
        hf_id,
        model_path,
        text,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """Number strings encode identically across all models."""
        reason = should_xfail(model_name, "numbers", text)
        if reason:
            pytest.xfail(reason)

        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        talu_tokens = tok.encode(text).tolist()
        hf_tokens = hf_tok.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"[{model_name}] Token mismatch for '{text}'\n"
            f"  talu: {talu_tokens}\n"
            f"  transformers: {hf_tokens}"
        )


class TestAllModelsEncodePunctuation:
    """Test punctuation encoding across all models."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    @pytest.mark.parametrize("text", PUNCTUATION_STRINGS)
    def test_punctuation_strings_match(
        self,
        model_name,
        hf_id,
        model_path,
        text,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """Punctuation strings encode identically across all models."""
        reason = should_xfail(model_name, "punctuation", text)
        if reason:
            pytest.xfail(reason)

        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        talu_tokens = tok.encode(text).tolist()
        hf_tokens = hf_tok.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"[{model_name}] Token mismatch for '{text}'\n"
            f"  talu: {talu_tokens}\n"
            f"  transformers: {hf_tokens}"
        )


class TestAllModelsEncodeContractions:
    """Test contraction encoding across all models."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    @pytest.mark.parametrize("text", CONTRACTION_STRINGS)
    def test_contraction_strings_match(
        self,
        model_name,
        hf_id,
        model_path,
        text,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """Contraction strings encode identically across all models."""
        reason = should_xfail(model_name, "contractions", text)
        if reason:
            pytest.xfail(reason)

        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        talu_tokens = tok.encode(text).tolist()
        hf_tokens = hf_tok.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"[{model_name}] Token mismatch for '{text}'\n"
            f"  talu: {talu_tokens}\n"
            f"  transformers: {hf_tokens}"
        )


class TestAllModelsEncodeWhitespace:
    """Test whitespace encoding across all models."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    @pytest.mark.parametrize("text", WHITESPACE_STRINGS)
    def test_whitespace_strings_match(
        self,
        model_name,
        hf_id,
        model_path,
        text,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """Whitespace strings encode identically across all models."""
        reason = should_xfail(model_name, "whitespace", text)
        if reason:
            pytest.xfail(reason)

        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        talu_tokens = tok.encode(text).tolist()
        hf_tokens = hf_tok.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"[{model_name}] Token mismatch for '{repr(text)}'\n"
            f"  talu: {talu_tokens}\n"
            f"  transformers: {hf_tokens}"
        )


class TestAllModelsEncodeUnicode:
    """Test Unicode encoding across all models."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    @pytest.mark.parametrize("text", UNICODE_STRINGS)
    def test_unicode_strings_match(
        self,
        model_name,
        hf_id,
        model_path,
        text,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """Unicode strings encode identically across all models."""
        reason = should_xfail(model_name, "unicode", text)
        if reason:
            pytest.xfail(reason)

        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        talu_tokens = tok.encode(text).tolist()
        hf_tokens = hf_tok.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"[{model_name}] Token mismatch for '{text}'\n"
            f"  talu: {talu_tokens}\n"
            f"  transformers: {hf_tokens}"
        )


class TestAllModelsEncodeMultilingual:
    """Test multilingual encoding across all models."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    @pytest.mark.parametrize("lang,text", MULTILINGUAL_STRINGS)
    def test_multilingual_strings_match(
        self,
        model_name,
        hf_id,
        model_path,
        lang,
        text,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """Multilingual strings encode identically across all models."""
        reason = should_xfail(model_name, "multilingual", text, lang)
        if reason:
            pytest.xfail(reason)

        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        talu_tokens = tok.encode(text).tolist()
        hf_tokens = hf_tok.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"[{model_name}] Token mismatch for {lang}: '{text}'\n"
            f"  talu: {talu_tokens}\n"
            f"  transformers: {hf_tokens}"
        )


class TestAllModelsEncodeCode:
    """Test code encoding across all models."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    @pytest.mark.parametrize("text", CODE_STRINGS)
    def test_code_strings_match(
        self,
        model_name,
        hf_id,
        model_path,
        text,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """Code strings encode identically across all models."""
        reason = should_xfail(model_name, "code", text)
        if reason:
            pytest.xfail(reason)

        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        talu_tokens = tok.encode(text).tolist()
        hf_tokens = hf_tok.encode(text, add_special_tokens=False)

        assert talu_tokens == hf_tokens, (
            f"[{model_name}] Token mismatch for '{text}'\n"
            f"  talu: {talu_tokens}\n"
            f"  transformers: {hf_tokens}"
        )


class TestAllModelsRoundtrip:
    """Test encode/decode roundtrip across all models."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    @pytest.mark.parametrize("text", BASIC_STRINGS + CODE_STRINGS)
    def test_roundtrip_matches_hf(
        self,
        model_name,
        hf_id,
        model_path,
        text,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """Encode/decode roundtrip matches HuggingFace across all models."""
        reason = should_xfail(model_name, "roundtrip", text)
        if reason:
            pytest.xfail(reason)

        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        # Roundtrip with talu
        talu_tokens = tok.encode(text)
        talu_decoded = tok.decode(talu_tokens)

        # Roundtrip with HF
        hf_tokens = hf_tok.encode(text, add_special_tokens=False)
        hf_decoded = hf_tok.decode(hf_tokens)

        assert talu_decoded == hf_decoded, (
            f"[{model_name}] Roundtrip mismatch for '{text[:50]}'\n"
            f"  talu: '{talu_decoded}'\n"
            f"  transformers: '{hf_decoded}'"
        )
