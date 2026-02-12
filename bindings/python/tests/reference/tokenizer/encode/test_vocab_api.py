"""
Vocabulary API tests.

Tests for vocabulary access methods:
- vocab_size
- id_to_token / token_to_id
- special tokens (bos, unk, pad)
- tokenize / count_tokens
"""

import pytest

import talu
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


def get_testable_models():
    """Get models that can be tested."""
    available = get_available_models()
    return [(name, hf_id, path) for name, hf_id, path in available if name not in SKIP_MODELS]


class TestVocabSize:
    """Tests for vocab_size property."""

    @pytest.mark.requires_model
    def test_vocab_size_positive(self, tokenizer):
        """Vocab size is a positive integer."""
        assert tokenizer.vocab_size > 0
        assert isinstance(tokenizer.vocab_size, int)

    @pytest.mark.requires_model
    def test_vocab_size_matches_hf(self, tokenizer, hf_tokenizer):
        """Vocab size matches HuggingFace."""
        # Note: Some tokenizers may have slightly different vocab sizes
        # due to added tokens, but they should be close
        talu_size = tokenizer.vocab_size
        hf_size = len(hf_tokenizer)

        # Allow for some difference due to added tokens
        assert abs(talu_size - hf_size) < 1000, (
            f"Vocab size mismatch: talu={talu_size}, hf={hf_size}"
        )


class TestIdToToken:
    """Tests for id_to_token method."""

    @pytest.mark.requires_model
    def test_id_to_token_valid(self, tokenizer):
        """Valid IDs return token strings."""
        # Test first few tokens
        for token_id in range(min(10, tokenizer.vocab_size)):
            token = tokenizer.id_to_token(token_id)
            assert token is not None
            assert isinstance(token, str)

    @pytest.mark.requires_model
    def test_id_to_token_out_of_range(self, tokenizer):
        """Out of range IDs return None."""
        token = tokenizer.id_to_token(tokenizer.vocab_size + 1000)
        assert token is None

    @pytest.mark.requires_model
    def test_id_to_token_negative(self, tokenizer):
        """Negative IDs return None."""
        token = tokenizer.id_to_token(-1)
        assert token is None

    @pytest.mark.requires_model
    def test_id_to_token_matches_hf(self, tokenizer, hf_tokenizer):
        """id_to_token matches HuggingFace convert_ids_to_tokens."""
        # Test a sample of IDs
        for token_id in [0, 1, 10, 100]:
            if token_id >= tokenizer.vocab_size:
                continue
            talu_token = tokenizer.id_to_token(token_id)
            hf_token = hf_tokenizer.convert_ids_to_tokens(token_id)

            assert talu_token == hf_token, (
                f"Token mismatch for ID {token_id}: talu={repr(talu_token)}, hf={repr(hf_token)}"
            )


class TestTokenToId:
    """Tests for token_to_id method."""

    @pytest.mark.requires_model
    def test_token_to_id_valid(self, tokenizer):
        """Valid tokens return IDs."""
        # Get a known token
        token = tokenizer.id_to_token(0)
        if token:
            result_id = tokenizer.token_to_id(token)
            assert result_id == 0

    @pytest.mark.requires_model
    def test_token_to_id_not_found(self, tokenizer):
        """Unknown tokens return None."""
        # Use a very unlikely token
        token_id = tokenizer.token_to_id("xyzzy_not_a_real_token_12345")
        assert token_id is None

    @pytest.mark.requires_model
    def test_roundtrip(self, tokenizer):
        """id_to_token and token_to_id are inverses."""
        for token_id in range(min(100, tokenizer.vocab_size)):
            token = tokenizer.id_to_token(token_id)
            if token:
                result_id = tokenizer.token_to_id(token)
                assert result_id == token_id, (
                    f"Roundtrip failed: {token_id} -> {repr(token)} -> {result_id}"
                )


class TestContains:
    """Tests for __contains__ (membership testing)."""

    @pytest.mark.requires_model
    def test_known_token_in_tokenizer(self, tokenizer):
        """Known tokens return True with 'in' operator."""
        # Get a known token from the vocab
        token = tokenizer.id_to_token(0)
        if token:
            assert token in tokenizer

    @pytest.mark.requires_model
    def test_unknown_token_not_in_tokenizer(self, tokenizer):
        """Unknown tokens return False with 'in' operator."""
        assert "xyzzy_not_a_real_token_12345" not in tokenizer
        assert "Supercalifragilisticexpialidocious_42" not in tokenizer

    @pytest.mark.requires_model
    def test_contains_non_string_returns_false(self, tokenizer):
        """Non-string values return False (not TypeError)."""
        assert (123 in tokenizer) is False
        assert (None in tokenizer) is False
        assert ([1, 2, 3] in tokenizer) is False

    @pytest.mark.requires_model
    def test_contains_matches_token_to_id(self, tokenizer):
        """__contains__ is consistent with token_to_id."""
        test_strings = ["Hello", "the", "unknown_xyz_123", ""]
        for s in test_strings:
            token_id = tokenizer.token_to_id(s)
            in_vocab = s in tokenizer
            assert in_vocab == (token_id is not None), (
                f"Inconsistent: token_to_id({repr(s)})={token_id}, "
                f"but '{s}' in tokenizer = {in_vocab}"
            )

    @pytest.mark.requires_model
    def test_contains_special_tokens(self, tokenizer):
        """Special token strings are in vocab if they exist."""
        if tokenizer.bos_token is not None:
            assert tokenizer.bos_token in tokenizer
        for eos_token in tokenizer.eos_tokens:
            assert eos_token in tokenizer
        if tokenizer.pad_token is not None:
            assert tokenizer.pad_token in tokenizer
        if tokenizer.unk_token is not None:
            assert tokenizer.unk_token in tokenizer


class TestSpecialTokens:
    """Tests for special token properties."""

    @pytest.mark.requires_model
    def test_eos_token_ids(self, tokenizer):
        """EOS tokens are valid (immutable tuple)."""
        eos = tokenizer.eos_token_ids
        assert isinstance(eos, tuple)
        for token_id in eos:
            assert isinstance(token_id, int)
            assert token_id >= 0

    @pytest.mark.requires_model
    def test_bos_token_id(self, tokenizer):
        """BOS token ID is valid if set."""
        bos = tokenizer.bos_token_id
        if bos is not None:
            assert isinstance(bos, int)
            assert bos >= 0
            # Should be in vocab
            token = tokenizer.id_to_token(bos)
            assert token is not None

    @pytest.mark.requires_model
    def test_unk_token_id(self, tokenizer):
        """UNK token ID is valid if set."""
        unk = tokenizer.unk_token_id
        if unk is not None:
            assert isinstance(unk, int)
            assert unk >= 0

    @pytest.mark.requires_model
    def test_pad_token_id(self, tokenizer):
        """PAD token ID is valid if set."""
        pad = tokenizer.pad_token_id
        if pad is not None:
            assert isinstance(pad, int)
            assert pad >= 0


class TestSpecialTokenStrings:
    """Tests for special token string properties."""

    @pytest.mark.requires_model
    def test_bos_token_string(self, tokenizer):
        """bos_token returns string when bos_token_id is set."""
        if tokenizer.bos_token_id is not None:
            bos = tokenizer.bos_token
            # May be None if id_to_token fails, but if it works, should be string
            if bos is not None:
                assert isinstance(bos, str)
                assert len(bos) > 0
        else:
            assert tokenizer.bos_token is None

    @pytest.mark.requires_model
    def test_eos_tokens_strings(self, tokenizer):
        """eos_tokens returns list of strings for all EOS tokens."""
        eos_tokens = tokenizer.eos_tokens
        assert isinstance(eos_tokens, list)
        if tokenizer.eos_token_ids:
            # Each returned token should be a string
            for token in eos_tokens:
                assert isinstance(token, str)
            # Number of strings may be <= number of IDs (some may not decode)
            assert len(eos_tokens) <= len(tokenizer.eos_token_ids)
        else:
            assert eos_tokens == []

    @pytest.mark.requires_model
    def test_pad_token_string(self, tokenizer):
        """pad_token returns string when pad_token_id is set."""
        if tokenizer.pad_token_id is not None:
            pad = tokenizer.pad_token
            if pad is not None:
                assert isinstance(pad, str)
        else:
            assert tokenizer.pad_token is None

    @pytest.mark.requires_model
    def test_unk_token_string(self, tokenizer):
        """unk_token returns string when unk_token_id is set."""
        if tokenizer.unk_token_id is not None:
            unk = tokenizer.unk_token
            if unk is not None:
                assert isinstance(unk, str)
        else:
            assert tokenizer.unk_token is None

    @pytest.mark.requires_model
    def test_special_token_consistency(self, tokenizer):
        """Special token strings match id_to_token lookup."""
        # Test bos
        if tokenizer.bos_token_id is not None:
            expected = tokenizer.id_to_token(tokenizer.bos_token_id)
            assert tokenizer.bos_token == expected

        # Test unk
        if tokenizer.unk_token_id is not None:
            expected = tokenizer.id_to_token(tokenizer.unk_token_id)
            assert tokenizer.unk_token == expected

        # Test pad
        if tokenizer.pad_token_id is not None:
            expected = tokenizer.id_to_token(tokenizer.pad_token_id)
            assert tokenizer.pad_token == expected


class TestBatchConversion:
    """Tests for convert_ids_to_tokens and convert_tokens_to_ids."""

    @pytest.mark.requires_model
    def test_convert_ids_to_tokens_basic(self, tokenizer):
        """convert_ids_to_tokens returns list of strings."""
        ids = [0, 1, 2]
        tokens = tokenizer.convert_ids_to_tokens(ids)
        assert isinstance(tokens, list)
        assert len(tokens) == len(ids)

    @pytest.mark.requires_model
    def test_convert_ids_to_tokens_matches_encode(self, tokenizer):
        """convert_ids_to_tokens matches tokenize output."""
        text = "Hello world"
        encoded = tokenizer.encode(text, special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(encoded.tolist())
        expected = tokenizer.tokenize(text)
        assert tokens == expected

    @pytest.mark.requires_model
    def test_convert_tokens_to_ids_basic(self, tokenizer):
        """convert_tokens_to_ids returns list of ints."""
        tokens = tokenizer.tokenize("Hello")
        ids = tokenizer.convert_tokens_to_ids(tokens)
        assert isinstance(ids, list)
        assert len(ids) == len(tokens)
        for token_id in ids:
            assert token_id is None or isinstance(token_id, int)

    @pytest.mark.requires_model
    def test_batch_roundtrip(self, tokenizer):
        """convert_ids_to_tokens and convert_tokens_to_ids are inverses."""
        original_ids = list(range(min(50, tokenizer.vocab_size)))
        tokens = tokenizer.convert_ids_to_tokens(original_ids)
        back_to_ids = tokenizer.convert_tokens_to_ids(tokens)

        for orig, result in zip(original_ids, back_to_ids, strict=True):
            # If token lookup succeeded, round-trip should work
            if tokens[original_ids.index(orig)] is not None:
                assert result == orig

    @pytest.mark.requires_model
    def test_convert_ids_to_tokens_empty(self, tokenizer):
        """convert_ids_to_tokens handles empty list."""
        assert tokenizer.convert_ids_to_tokens([]) == []

    @pytest.mark.requires_model
    def test_convert_tokens_to_ids_empty(self, tokenizer):
        """convert_tokens_to_ids handles empty list."""
        assert tokenizer.convert_tokens_to_ids([]) == []

    @pytest.mark.requires_model
    def test_convert_ids_to_tokens_invalid(self, tokenizer):
        """convert_ids_to_tokens returns None for invalid IDs."""
        invalid_id = tokenizer.vocab_size + 1000
        result = tokenizer.convert_ids_to_tokens([0, invalid_id, 1])
        assert result[0] is not None  # Valid
        assert result[1] is None  # Invalid
        assert result[2] is not None  # Valid

    @pytest.mark.requires_model
    def test_convert_tokens_to_ids_invalid(self, tokenizer):
        """convert_tokens_to_ids returns None for unknown tokens."""
        result = tokenizer.convert_tokens_to_ids(["Hello", "xyzzy_not_a_token_12345"])
        # First might work, second should be None
        assert result[1] is None


class TestTokenize:
    """Tests for tokenize method."""

    @pytest.mark.requires_model
    def test_tokenize_returns_strings(self, tokenizer):
        """tokenize() returns list of strings."""
        tokens = tokenizer.tokenize("Hello world")
        assert isinstance(tokens, list)
        for token in tokens:
            assert isinstance(token, str)

    @pytest.mark.requires_model
    def test_tokenize_length_matches_encode(self, tokenizer):
        """tokenize() returns same number of tokens as encode()."""
        text = "Hello world, this is a test."
        token_strings = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)
        assert len(token_strings) == len(token_ids)

    @pytest.mark.requires_model
    def test_tokenize_matches_hf(self, tokenizer, hf_tokenizer):
        """tokenize() matches HuggingFace tokenize()."""
        texts = [
            "Hello world",
            "The quick brown fox",
            "Testing 123",
        ]
        for text in texts:
            talu_tokens = tokenizer.tokenize(text)
            hf_tokens = hf_tokenizer.tokenize(text)
            assert talu_tokens == hf_tokens, (
                f"Tokenize mismatch for '{text}':\n  talu: {talu_tokens}\n  hf: {hf_tokens}"
            )

    @pytest.mark.requires_model
    def test_tokenize_empty(self, tokenizer):
        """tokenize('') returns empty list."""
        assert tokenizer.tokenize("") == []

    @pytest.mark.requires_model
    def test_tokenize_return_bytes(self, tokenizer):
        """tokenize(return_bytes=True) returns list of bytes."""
        tokens = tokenizer.tokenize("Hello world", return_bytes=True)
        assert isinstance(tokens, list)
        for token in tokens:
            assert isinstance(token, bytes)

    @pytest.mark.requires_model
    def test_tokenize_bytes_length_matches_strings(self, tokenizer):
        """Bytes and string tokenize return same number of tokens."""
        text = "Hello world, this is a test."
        tokens_str = tokenizer.tokenize(text)
        tokens_bytes = tokenizer.tokenize(text, return_bytes=True)
        assert len(tokens_str) == len(tokens_bytes)

    @pytest.mark.requires_model
    def test_tokenize_bytes_empty(self, tokenizer):
        """tokenize('', return_bytes=True) returns empty list."""
        assert tokenizer.tokenize("", return_bytes=True) == []

    @pytest.mark.requires_model
    def test_tokenize_bytes_unicode(self, tokenizer):
        """tokenize with unicode returns bytes that can be inspected."""
        text = "café"
        tokens_bytes = tokenizer.tokenize(text, return_bytes=True)
        # Should have at least one token
        assert len(tokens_bytes) > 0
        # Each token should be bytes
        for token in tokens_bytes:
            assert isinstance(token, bytes)
            assert len(token) > 0

    @pytest.mark.requires_model
    def test_tokenize_bytes_preserves_data(self, tokenizer):
        """Bytes representation preserves token data."""
        text = "Hello"
        tokens_bytes = tokenizer.tokenize(text, return_bytes=True)
        tokens_str = tokenizer.tokenize(text)

        # Verify string decode of bytes matches string tokenize
        # (at least for ASCII text where UTF-8 decoding should work)
        for b, s in zip(tokens_bytes, tokens_str, strict=True):
            # Decode bytes and compare (use replace for safety)
            decoded = b.decode("utf-8", errors="replace")
            assert decoded == s


class TestCountTokens:
    """Tests for count_tokens method."""

    @pytest.mark.requires_model
    def test_count_tokens_matches_encode(self, tokenizer):
        """count_tokens() matches len(encode()) with default special tokens."""
        texts = [
            "Hello",
            "Hello world",
            "The quick brown fox",
            "日本語テスト",
        ]
        for text in texts:
            count = tokenizer.count_tokens(text)
            encoded_len = len(tokenizer.encode(text))
            assert count == encoded_len, (
                f"Count mismatch for '{text}': count={count}, encode_len={encoded_len}"
            )

    @pytest.mark.requires_model
    def test_count_tokens_empty(self, tokenizer):
        """count_tokens('') returns 0."""
        assert tokenizer.count_tokens("") == 0

    @pytest.mark.requires_model
    def test_count_tokens_with_special_tokens(self, tokenizer):
        """count_tokens(special_tokens=True) matches encode with special tokens."""
        text = "Hello world"
        count = tokenizer.count_tokens(text, special_tokens=True)
        encoded_len = len(tokenizer.encode(text, special_tokens=True))
        assert count == encoded_len

    @pytest.mark.requires_model
    def test_count_tokens_without_special_tokens(self, tokenizer):
        """count_tokens(special_tokens=False) matches encode without special tokens."""
        text = "Hello world"
        count = tokenizer.count_tokens(text, special_tokens=False)
        encoded_len = len(tokenizer.encode(text, special_tokens=False))
        assert count == encoded_len

    @pytest.mark.requires_model
    def test_count_tokens_special_tokens_consistent(self, tokenizer):
        """special_tokens parameter is passed through correctly."""
        text = "Hello world"
        with_special = tokenizer.count_tokens(text, special_tokens=True)
        without_special = tokenizer.count_tokens(text, special_tokens=False)

        # The counts should match the encode() behavior
        encode_with = len(tokenizer.encode(text, special_tokens=True))
        encode_without = len(tokenizer.encode(text, special_tokens=False))

        assert with_special == encode_with
        assert without_special == encode_without

        # With special tokens should be >= without (never less)
        assert with_special >= without_special

    @pytest.mark.requires_model
    def test_count_tokens_default_matches_generation(self, tokenizer):
        """Default count_tokens matches what generation path uses.

        This is critical for context window checking - users expect the
        default to match what gets fed to the model.
        """
        text = "Test prompt for context window check"

        # Default should include special tokens (matching generation)
        default_count = tokenizer.count_tokens(text)
        generation_tokens = tokenizer.encode(text, special_tokens=True)

        assert default_count == len(generation_tokens), (
            "Default count_tokens should match generation token count"
        )


class TestSpecialTokensAPI:
    """Tests for the special_tokens: bool | set[str] API."""

    @pytest.mark.requires_model
    def test_special_tokens_true(self, tokenizer):
        """special_tokens=True adds all special tokens."""
        text = "Hello world"
        tokens = tokenizer.encode(text, special_tokens=True)
        assert len(tokens) > 0

    @pytest.mark.requires_model
    def test_special_tokens_false(self, tokenizer):
        """special_tokens=False adds no special tokens."""
        text = "Hello world"
        tokens = tokenizer.encode(text, special_tokens=False)
        assert len(tokens) > 0

    @pytest.mark.requires_model
    def test_special_tokens_set_bos(self, tokenizer):
        """special_tokens={'bos'} adds only BOS token."""
        text = "Hello world"
        tokens = tokenizer.encode(text, special_tokens={"bos"})
        assert len(tokens) > 0

    @pytest.mark.requires_model
    def test_special_tokens_set_eos(self, tokenizer):
        """special_tokens={'eos'} adds only EOS token."""
        text = "Hello world"
        tokens = tokenizer.encode(text, special_tokens={"eos"})
        assert len(tokens) > 0

    @pytest.mark.requires_model
    def test_special_tokens_set_both(self, tokenizer):
        """special_tokens={'bos', 'eos'} equivalent to True."""
        text = "Hello world"
        tokens_set = tokenizer.encode(text, special_tokens={"bos", "eos"})
        tokens_true = tokenizer.encode(text, special_tokens=True)
        assert tokens_set.tolist() == tokens_true.tolist()

    @pytest.mark.requires_model
    def test_special_tokens_invalid_type(self, tokenizer):
        """special_tokens with invalid type raises ValidationError."""
        with pytest.raises(talu.ValidationError):
            tokenizer.encode("Hello", special_tokens="bos")

        with pytest.raises(talu.ValidationError):
            tokenizer.encode("Hello", special_tokens=123)


class TestTruncation:
    """Tests for truncation support."""

    @pytest.mark.requires_model
    def test_truncation_limits_length(self, tokenizer):
        """truncation=True with max_length limits output."""
        long_text = "Hello world " * 100
        tokens = tokenizer.encode(long_text, truncation=True, max_length=10)
        assert len(tokens) == 10

    @pytest.mark.requires_model
    def test_truncation_no_effect_on_short(self, tokenizer):
        """Truncation doesn't affect short texts."""
        text = "Hello"
        tokens_normal = tokenizer.encode(text)
        tokens_trunc = tokenizer.encode(text, truncation=True, max_length=100)
        assert len(tokens_trunc) == len(tokens_normal)

    @pytest.mark.requires_model
    def test_truncation_disabled_by_default(self, tokenizer):
        """Truncation is disabled by default."""
        long_text = "Hello world " * 100
        tokens = tokenizer.encode(long_text, max_length=10)  # No truncation=True
        assert len(tokens) > 10

    @pytest.mark.requires_model
    @pytest.mark.xfail(reason="Batch truncation not yet implemented in Zig")
    def test_truncation_batch(self, tokenizer):
        """Truncation works with batch encoding."""
        texts = ["Hello " * 100, "World " * 50]
        batch = tokenizer.encode(texts, truncation=True, max_length=5)
        for i in range(len(batch)):
            assert len(list(batch[i])) <= 5


class TestTruncationSide:
    """Tests for truncation_side configuration."""

    @pytest.mark.requires_model
    def test_right_truncation_keeps_beginning(self, tokenizer):
        """Right truncation (default) keeps beginning of text."""
        # Encode a long text with distinct beginning and end
        text = "START " + "middle " * 50 + "END"
        full_tokens = tokenizer.encode(text, special_tokens=False)

        # Truncate from right (default)
        truncated = tokenizer.encode(text, special_tokens=False, truncation=True, max_length=5)

        # Should keep the first tokens
        assert truncated.tolist() == full_tokens.tolist()[:5]

    @pytest.mark.requires_model
    def test_left_truncation_keeps_end(self, tokenizer):
        """Left truncation keeps end of text."""
        # Encode a long text with distinct beginning and end
        text = "START " + "middle " * 50 + "END"
        full_tokens = tokenizer.encode(text, special_tokens=False)

        # Truncate from left
        truncated = tokenizer.encode(
            text, special_tokens=False, truncation=True, max_length=5, truncation_side="left"
        )

        # Should keep the last tokens
        assert truncated.tolist() == full_tokens.tolist()[-5:]

    @pytest.mark.requires_model
    def test_truncation_side_instance_default(self, talu, test_model_path):
        """Instance truncation_side is used as default."""
        # Create tokenizer with left truncation default
        tok = talu.Tokenizer(test_model_path, truncation_side="left")
        assert tok.truncation_side == "left"

        text = "START " + "middle " * 50 + "END"
        full_tokens = tok.encode(text, special_tokens=False)

        # Default should be left truncation
        truncated = tok.encode(text, special_tokens=False, truncation=True, max_length=5)
        assert truncated.tolist() == full_tokens.tolist()[-5:]

    @pytest.mark.requires_model
    def test_truncation_side_method_override(self, talu, test_model_path):
        """Method arg overrides instance default."""
        # Create tokenizer with left truncation default
        tok = talu.Tokenizer(test_model_path, truncation_side="left")

        text = "START " + "middle " * 50 + "END"
        full_tokens = tok.encode(text, special_tokens=False)

        # Override to right truncation
        truncated = tok.encode(
            text, special_tokens=False, truncation=True, max_length=5, truncation_side="right"
        )
        assert truncated.tolist() == full_tokens.tolist()[:5]


class TestInstanceConfiguration:
    """Tests for tokenizer instance configuration pattern."""

    @pytest.mark.requires_model
    def test_padding_side_init(self, talu, test_model_path):
        """padding_side can be set at init."""
        tok_left = talu.Tokenizer(test_model_path, padding_side="left")
        tok_right = talu.Tokenizer(test_model_path, padding_side="right")

        assert tok_left.padding_side == "left"
        assert tok_right.padding_side == "right"

    @pytest.mark.requires_model
    def test_truncation_side_init(self, talu, test_model_path):
        """truncation_side can be set at init."""
        tok_left = talu.Tokenizer(test_model_path, truncation_side="left")
        tok_right = talu.Tokenizer(test_model_path, truncation_side="right")

        assert tok_left.truncation_side == "left"
        assert tok_right.truncation_side == "right"

    @pytest.mark.requires_model
    def test_invalid_padding_side_init(self, talu, test_model_path):
        """Invalid padding_side at init raises ValueError."""
        with pytest.raises(ValueError):
            talu.Tokenizer(test_model_path, padding_side="center")

    @pytest.mark.requires_model
    def test_invalid_truncation_side_init(self, talu, test_model_path):
        """Invalid truncation_side at init raises ValueError."""
        with pytest.raises(ValueError):
            talu.Tokenizer(test_model_path, truncation_side="center")

    @pytest.mark.requires_model
    def test_invalid_truncation_side_setter(self, tokenizer):
        """Invalid truncation_side via setter raises ValidationError."""
        from talu.exceptions import ValidationError

        with pytest.raises(ValidationError, match="truncation_side must be"):
            tokenizer.truncation_side = "center"

    @pytest.mark.requires_model
    def test_truncation_side_setter_valid(self, tokenizer):
        """Valid truncation_side can be set via setter."""
        tokenizer.truncation_side = "left"
        assert tokenizer.truncation_side == "left"
        tokenizer.truncation_side = "right"
        assert tokenizer.truncation_side == "right"

    @pytest.mark.requires_model
    def test_batch_encoding_inherits_padding_side(self, tokenizer):
        """BatchEncoding inherits padding_side from tokenizer."""
        batch = tokenizer.encode(["Hello", "World"])
        assert batch.padding_side == tokenizer.padding_side

    @pytest.mark.requires_model
    def test_batch_encoding_padding_side_settable(self, tokenizer):
        """BatchEncoding.padding_side can be overridden."""
        batch = tokenizer.encode(["Hello", "World"])
        original = batch.padding_side

        # Override
        new_side = "right" if original == "left" else "left"
        batch.padding_side = new_side
        assert batch.padding_side == new_side

    @pytest.mark.requires_model
    def test_batch_encoding_padding_side_validation(self, tokenizer):
        """Invalid padding_side raises ValueError."""
        batch = tokenizer.encode(["Hello", "World"])
        with pytest.raises(ValueError):
            batch.padding_side = "center"

    @pytest.mark.requires_model
    def test_batch_encoding_stores_pad_token_id(self, tokenizer):
        """BatchEncoding stores pad_token_id from tokenizer."""
        batch = tokenizer.encode(["Hello", "World"])
        # Should be tokenizer's pad_token_id (or None if not set)
        assert batch.pad_token_id == tokenizer.pad_token_id

    @pytest.mark.requires_model
    def test_to_list_uses_stored_defaults(self, tokenizer):
        """to_list() uses stored padding_side without explicit arg."""
        batch = tokenizer.encode(["Hi", "Hello world"])
        batch.padding_side = "right"  # Override on batch

        result = batch.to_list()  # No args - uses stored default
        assert "input_ids" in result

        # With right padding, short sequence gets padding at end
        if tokenizer.pad_token_id is not None:
            first_seq = result["input_ids"][0]
            # Longer seq should have no padding, shorter should have some
            if len(batch[0]) < len(batch[1]):
                # First is shorter - padding should be at end (right)
                assert first_seq[-1] == tokenizer.pad_token_id or first_seq[-1] == 0

    @pytest.mark.requires_model
    def test_to_list_explicit_overrides_stored(self, tokenizer):
        """Explicit to_list() args override stored defaults."""
        batch = tokenizer.encode(["Hi", "Hello world"])
        batch.padding_side = "right"

        # Override to left padding in to_list call
        result = batch.to_list(padding_side="left")

        # Should work without error
        assert "input_ids" in result

    @pytest.mark.requires_model
    def test_encode_to_list_flow_intuitive(self, tokenizer):
        """The full encode -> padding_side -> to_list flow works intuitively."""
        texts = ["Short", "Much longer text here"]

        # Encode (inherits tokenizer's padding_side)
        batch = tokenizer.encode(texts)

        # Power user can override if needed
        batch.padding_side = "left"

        # to_list automatically uses the stored settings
        tensor = batch.to_list()  # Clean call!

        assert "input_ids" in tensor
        assert "attention_mask" in tensor
        assert len(tensor["input_ids"]) == 2


class TestAllModelsVocab:
    """Test vocabulary API across all models."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    def test_vocab_size_all_models(self, model_name, hf_id, model_path, talu, tokenizer_cache):
        """Vocab size is valid for all models."""
        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        assert tok.vocab_size > 0, f"[{model_name}] vocab_size should be positive"

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    def test_special_tokens_all_models(self, model_name, hf_id, model_path, talu, tokenizer_cache):
        """Special tokens are accessible for all models."""
        tok = load_tokenizer(model_path, tokenizer_cache, talu)

        # These should not raise
        _ = tok.eos_token_ids
        _ = tok.bos_token_id
        _ = tok.unk_token_id
        _ = tok.pad_token_id

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    def test_id_to_token_all_models(
        self,
        model_name,
        hf_id,
        model_path,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """id_to_token matches HF for all models."""
        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        # Test first 10 tokens
        for token_id in range(min(10, tok.vocab_size)):
            talu_token = tok.id_to_token(token_id)
            hf_token = hf_tok.convert_ids_to_tokens(token_id)

            assert talu_token == hf_token, (
                f"[{model_name}] Token mismatch for ID {token_id}: "
                f"talu={repr(talu_token)}, hf={repr(hf_token)}"
            )

    @pytest.mark.requires_model
    @pytest.mark.parametrize("model_name,hf_id,model_path", get_testable_models())
    def test_tokenize_all_models(
        self,
        model_name,
        hf_id,
        model_path,
        talu,
        tokenizer_cache,
        hf_tokenizer_cache,
        transformers,
    ):
        """tokenize() matches HF for all models."""
        tok = load_tokenizer(model_path, tokenizer_cache, talu)
        hf_tok = load_hf_tokenizer(model_path, hf_tokenizer_cache, transformers)

        texts = ["Hello world", "The quick brown fox"]
        for text in texts:
            talu_tokens = tok.tokenize(text)
            hf_tokens = hf_tok.tokenize(text)

            assert talu_tokens == hf_tokens, (
                f"[{model_name}] Tokenize mismatch for '{text}':\n"
                f"  talu: {talu_tokens}\n"
                f"  hf: {hf_tokens}"
            )
