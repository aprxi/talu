"""
Multi-model decode correctness tests.

Tests talu.Tokenizer.decode() across all available models,
comparing against HuggingFace transformers for exact string matching.
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
#   ("llama2", "unicode", "emoji"): "LLaMA2 decode adds extra space before emoji"
#   ("qwen3", "whitespace", None): "All whitespace decode tests fail for Qwen3"
#   ("gemma3", "single_tokens", None): "Gemma3 special tokens decode differently"
#
# Policy: Add entries here when a test fails due to known talu limitation.
# These appear as XFAIL in pytest output and alert when fixed (XPASS).
# Include actionable fix hints where possible (e.g., "needs byte fallback handling").
#
# Issue tracking: Optionally add GitHub issue link in reason string:
#   ("model", "cat", "text"): "Description - see #123"
#
KNOWN_ISSUES = {
    # No known decode issues currently - all tests pass
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


class TestAllModelsDecodeBasic:
    """Test basic decoding across all models."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    @pytest.mark.parametrize("text", BASIC_STRINGS)
    def test_basic_strings_decode_match(
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
        """Basic strings decode identically across all models."""
        reason = should_xfail(model_name, "basic", text)
        if reason:
            pytest.xfail(reason)

        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        # Use HF tokens as ground truth
        hf_tokens = hf_tok.encode(text, add_special_tokens=False)

        talu_decoded = tok.decode(hf_tokens)
        hf_decoded = hf_tok.decode(hf_tokens)

        assert talu_decoded == hf_decoded, (
            f"[{model_name}] Decode mismatch for '{text[:50]}'\n"
            f"  talu: '{talu_decoded}'\n"
            f"  transformers: '{hf_decoded}'"
        )


class TestAllModelsDecodeNumbers:
    """Test number decoding across all models."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    @pytest.mark.parametrize("text", NUMBER_STRINGS)
    def test_number_strings_decode_match(
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
        """Number strings decode identically across all models."""
        reason = should_xfail(model_name, "numbers", text)
        if reason:
            pytest.xfail(reason)

        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        hf_tokens = hf_tok.encode(text, add_special_tokens=False)

        talu_decoded = tok.decode(hf_tokens)
        hf_decoded = hf_tok.decode(hf_tokens)

        assert talu_decoded == hf_decoded, (
            f"[{model_name}] Decode mismatch for '{text}'\n"
            f"  talu: '{talu_decoded}'\n"
            f"  transformers: '{hf_decoded}'"
        )


class TestAllModelsDecodePunctuation:
    """Test punctuation decoding across all models."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    @pytest.mark.parametrize("text", PUNCTUATION_STRINGS)
    def test_punctuation_strings_decode_match(
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
        """Punctuation strings decode identically across all models."""
        reason = should_xfail(model_name, "punctuation", text)
        if reason:
            pytest.xfail(reason)

        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        hf_tokens = hf_tok.encode(text, add_special_tokens=False)

        talu_decoded = tok.decode(hf_tokens)
        hf_decoded = hf_tok.decode(hf_tokens)

        assert talu_decoded == hf_decoded, (
            f"[{model_name}] Decode mismatch for '{text}'\n"
            f"  talu: '{talu_decoded}'\n"
            f"  transformers: '{hf_decoded}'"
        )


class TestAllModelsDecodeContractions:
    """Test contraction decoding across all models."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    @pytest.mark.parametrize("text", CONTRACTION_STRINGS)
    def test_contraction_strings_decode_match(
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
        """Contraction strings decode identically across all models."""
        reason = should_xfail(model_name, "contractions", text)
        if reason:
            pytest.xfail(reason)

        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        hf_tokens = hf_tok.encode(text, add_special_tokens=False)

        talu_decoded = tok.decode(hf_tokens)
        hf_decoded = hf_tok.decode(hf_tokens)

        assert talu_decoded == hf_decoded, (
            f"[{model_name}] Decode mismatch for '{text}'\n"
            f"  talu: '{talu_decoded}'\n"
            f"  transformers: '{hf_decoded}'"
        )


class TestAllModelsDecodeWhitespace:
    """Test whitespace decoding across all models."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    @pytest.mark.parametrize("text", WHITESPACE_STRINGS)
    def test_whitespace_strings_decode_match(
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
        """Whitespace strings decode identically across all models."""
        reason = should_xfail(model_name, "whitespace", text)
        if reason:
            pytest.xfail(reason)

        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        hf_tokens = hf_tok.encode(text, add_special_tokens=False)

        talu_decoded = tok.decode(hf_tokens)
        hf_decoded = hf_tok.decode(hf_tokens)

        assert talu_decoded == hf_decoded, (
            f"[{model_name}] Decode mismatch for '{repr(text)}'\n"
            f"  talu: '{repr(talu_decoded)}'\n"
            f"  transformers: '{repr(hf_decoded)}'"
        )


class TestAllModelsDecodeUnicode:
    """Test Unicode decoding across all models."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    @pytest.mark.parametrize("text", UNICODE_STRINGS)
    def test_unicode_strings_decode_match(
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
        """Unicode strings decode identically across all models."""
        reason = should_xfail(model_name, "unicode", text)
        if reason:
            pytest.xfail(reason)

        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        hf_tokens = hf_tok.encode(text, add_special_tokens=False)

        talu_decoded = tok.decode(hf_tokens)
        hf_decoded = hf_tok.decode(hf_tokens)

        assert talu_decoded == hf_decoded, (
            f"[{model_name}] Decode mismatch for '{text}'\n"
            f"  talu: '{talu_decoded}'\n"
            f"  transformers: '{hf_decoded}'"
        )


class TestAllModelsDecodeMultilingual:
    """Test multilingual decoding across all models."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    @pytest.mark.parametrize("lang,text", MULTILINGUAL_STRINGS)
    def test_multilingual_strings_decode_match(
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
        """Multilingual strings decode identically across all models."""
        reason = should_xfail(model_name, "multilingual", text, lang)
        if reason:
            pytest.xfail(reason)

        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        hf_tokens = hf_tok.encode(text, add_special_tokens=False)

        talu_decoded = tok.decode(hf_tokens)
        hf_decoded = hf_tok.decode(hf_tokens)

        assert talu_decoded == hf_decoded, (
            f"[{model_name}] Decode mismatch for {lang}: '{text}'\n"
            f"  talu: '{talu_decoded}'\n"
            f"  transformers: '{hf_decoded}'"
        )


class TestAllModelsDecodeCode:
    """Test code decoding across all models."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    @pytest.mark.parametrize("text", CODE_STRINGS)
    def test_code_strings_decode_match(
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
        """Code strings decode identically across all models."""
        reason = should_xfail(model_name, "code", text)
        if reason:
            pytest.xfail(reason)

        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        hf_tokens = hf_tok.encode(text, add_special_tokens=False)

        talu_decoded = tok.decode(hf_tokens)
        hf_decoded = hf_tok.decode(hf_tokens)

        assert talu_decoded == hf_decoded, (
            f"[{model_name}] Decode mismatch for '{text}'\n"
            f"  talu: '{talu_decoded}'\n"
            f"  transformers: '{hf_decoded}'"
        )


class TestAllModelsDecodeSingleTokens:
    """Test single token decoding across all models."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    def test_single_tokens_match(
        self,
        model_name,
        hf_id,
        model_path,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """Single token decoding matches across all models."""
        reason = should_xfail(model_name, "single_tokens")
        if reason:
            pytest.xfail(reason)

        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        failures = []
        # Test first 50 token IDs
        for token_id in range(50):
            try:
                talu_decoded = tok.decode([token_id])
                hf_decoded = hf_tok.decode([token_id])

                if talu_decoded != hf_decoded:
                    failures.append(
                        {
                            "token_id": token_id,
                            "talu": repr(talu_decoded),
                            "hf": repr(hf_decoded),
                        }
                    )
            except Exception:
                pass  # Some token IDs may be invalid

        if failures:
            msg = f"[{model_name}] {len(failures)} single token mismatches:\n"
            for f in failures[:5]:
                msg += f"  token {f['token_id']}: got={f['talu']} expected={f['hf']}\n"
            pytest.fail(msg)


class TestAllModelsDecodeEmpty:
    """Test empty decoding across all models."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    def test_empty_decode_match(
        self,
        model_name,
        hf_id,
        model_path,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """Empty token list decodes to empty string across all models."""
        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        talu_decoded = tok.decode([])
        hf_decoded = hf_tok.decode([])

        assert talu_decoded == hf_decoded == "", (
            f"[{model_name}] Empty decode mismatch:\n"
            f"  talu: '{repr(talu_decoded)}'\n"
            f"  transformers: '{repr(hf_decoded)}'"
        )
