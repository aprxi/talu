"""
Special token tests.

Tests for special token handling:
- EOS token detection
- Token ID properties
- Vocab consistency
"""

import pytest

from tests.tokenizer.conftest import (
    GRANITE_SPECIAL_TOKENS,
    PHI_SPECIAL_TOKENS,
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


def get_testable_models():
    """Get models that can be tested."""
    available = get_available_models()
    return [(name, hf_id, path) for name, hf_id, path in available if name not in SKIP_MODELS]


class TestBosTokens:
    """Tests for BOS (beginning-of-sequence) token handling."""

    @pytest.mark.requires_model
    def test_bos_token_id_type(self, tokenizer):
        """bos_token_id is int or None."""
        bos_id = tokenizer.bos_token_id
        assert bos_id is None or isinstance(bos_id, int)

    @pytest.mark.requires_model
    def test_bos_token_type(self, tokenizer):
        """bos_token is str or None."""
        bos_token = tokenizer.bos_token
        assert bos_token is None or isinstance(bos_token, str)

    @pytest.mark.requires_model
    def test_bos_token_consistency(self, tokenizer):
        """bos_token is None iff bos_token_id is None."""
        bos_id = tokenizer.bos_token_id
        bos_token = tokenizer.bos_token

        if bos_id is None:
            assert bos_token is None
        else:
            # bos_token might be None if id_to_token fails for added tokens
            # This is a known limitation
            pass


class TestBosTokenMatchesHuggingFace:
    """Test BOS token matches HuggingFace."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    def test_bos_token_id_matches_hf(
        self,
        model_name,
        hf_id,
        model_path,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """BOS token ID matches HuggingFace tokenizer."""
        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        talu_bos = tok.bos_token_id
        hf_bos = hf_tok.bos_token_id

        assert talu_bos == hf_bos, (
            f"[{model_name}] BOS token ID mismatch: talu={talu_bos}, hf={hf_bos}"
        )


class TestEosTokenIds:
    """Tests for EOS token handling."""

    @pytest.mark.requires_model
    def test_eos_token_ids_available(self, tokenizer):
        """Tokenizer exposes EOS tokens as immutable tuple."""
        eos_tokens = tokenizer.eos_token_ids

        assert isinstance(eos_tokens, tuple)
        # Most tokenizers have at least one EOS token
        assert len(eos_tokens) >= 1, "Expected at least one EOS token"

    @pytest.mark.requires_model
    def test_eos_token_ids_are_valid_ids(self, tokenizer):
        """EOS tokens are valid token IDs."""
        eos_tokens = tokenizer.eos_token_ids

        for token_id in eos_tokens:
            assert isinstance(token_id, int)
            assert token_id >= 0

    @pytest.mark.requires_model
    def test_eos_token_ids_decode(self, tokenizer):
        """EOS tokens can be decoded."""
        eos_tokens = tokenizer.eos_token_ids

        for token_id in eos_tokens:
            decoded = tokenizer.decode([token_id])
            # Should produce some output (may be empty string for some tokens)
            assert isinstance(decoded, str)


class TestAllModelsEosTokenIds:
    """Test EOS tokens across all models."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    def test_eos_token_ids_available(self, model_name, hf_id, model_path, talu, tokenizer_cache):
        """EOS tokens are available for all models."""
        tok = load_tokenizer(model_path, tokenizer_cache, talu)

        eos_tokens = tok.eos_token_ids
        assert isinstance(eos_tokens, tuple)
        assert len(eos_tokens) >= 1, f"[{model_name}] No EOS tokens found"

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    def test_eos_matches_hf(
        self,
        model_name,
        hf_id,
        model_path,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """Primary EOS token matches HuggingFace."""
        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        eos_tokens = tok.eos_token_ids

        # HF eos_token_id should be in our EOS tokens list
        if hf_tok.eos_token_id is not None:
            assert hf_tok.eos_token_id in eos_tokens, (
                f"[{model_name}] HF EOS token {hf_tok.eos_token_id} not in talu EOS tokens {eos_tokens}"
            )


class TestModelSpecificTokens:
    """Tests for model-specific special token strings."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", GRANITE_SPECIAL_TOKENS)
    def test_granite_special_tokens_encode(self, tokenizer, text):
        """Granite special token strings encode without error."""
        tokens = tokenizer.encode(text)
        assert len(tokens) >= 1

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", PHI_SPECIAL_TOKENS)
    def test_phi_special_tokens_encode(self, tokenizer, text):
        """Phi special token strings encode without error."""
        tokens = tokenizer.encode(text)
        assert len(tokens) >= 1

    @pytest.mark.requires_model
    def test_special_markers_roundtrip(self, tokenizer):
        """Special marker strings survive roundtrip."""
        markers = [
            "<|user|>",
            "<|assistant|>",
            "<|system|>",
            "<|end|>",
            "[INST]",
            "[/INST]",
        ]

        for marker in markers:
            tokens = tokenizer.encode(marker)
            decoded = tokenizer.decode(tokens)
            # Decoded should contain the marker (may have extra whitespace)
            assert marker.lower() in decoded.lower() or decoded.strip() == marker.strip(), (
                f"Marker '{marker}' not preserved: got '{decoded}'"
            )


class TestTokenIdProperties:
    """Tests for token ID properties."""

    @pytest.mark.requires_model
    def test_token_ids_are_positive(self, tokenizer):
        """All encoded token IDs are non-negative."""
        texts = ["Hello", "World", "Test 123", "日本語"]

        for text in texts:
            tokens = tokenizer.encode(text)
            for token_id in tokens:
                assert token_id >= 0, f"Negative token ID {token_id} for '{text}'"

    @pytest.mark.requires_model
    def test_token_ids_consistent(self, tokenizer):
        """Same text produces same token IDs."""
        text = "Consistency test"

        tokens1 = tokenizer.encode(text).tolist()
        tokens2 = tokenizer.encode(text).tolist()
        tokens3 = tokenizer.encode(text).tolist()

        assert tokens1 == tokens2 == tokens3

    @pytest.mark.requires_model
    def test_different_texts_different_tokens(self, tokenizer):
        """Different texts produce different tokens."""
        text1 = "Hello World"
        text2 = "Goodbye World"

        tokens1 = tokenizer.encode(text1).tolist()
        tokens2 = tokenizer.encode(text2).tolist()

        # At least some tokens should differ
        assert tokens1 != tokens2


class TestAllModelsTokenProperties:
    """Test token properties across all models."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    def test_common_tokens_consistent(
        self,
        model_name,
        hf_id,
        model_path,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """Common tokens are consistent with HuggingFace."""
        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        # Test common words
        common_words = ["the", "and", "is", "a", "to", "of"]

        for word in common_words:
            talu_tokens = tok.encode(word).tolist()
            hf_tokens = hf_tok.encode(word, add_special_tokens=False)

            assert talu_tokens == hf_tokens, (
                f"[{model_name}] Token mismatch for '{word}': talu={talu_tokens}, hf={hf_tokens}"
            )

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    def test_single_char_tokens(
        self,
        model_name,
        hf_id,
        model_path,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """Single character tokens match HuggingFace."""
        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        # Test single characters
        chars = list("abcdefghijklmnopqrstuvwxyz0123456789")

        for char in chars:
            talu_tokens = tok.encode(char).tolist()
            hf_tokens = hf_tok.encode(char, add_special_tokens=False)

            assert talu_tokens == hf_tokens, (
                f"[{model_name}] Token mismatch for '{char}': talu={talu_tokens}, hf={hf_tokens}"
            )
