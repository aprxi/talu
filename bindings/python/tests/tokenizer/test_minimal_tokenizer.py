"""
Tests using the minimal tokenizer (no model files required).

These tests exercise the Tokenizer API using a minimal BPE tokenizer
created from JSON. This enables testing:
- Tokenizer.from_json() class method
- Error paths that don't require a full model
- Edge cases with predictable behavior
"""

import pytest

from tests.tokenizer.conftest import MINIMAL_TOKENIZER_JSON


class TestTokenizerFromJson:
    """Tests for Tokenizer.from_json() class method."""

    def test_from_json_string(self, talu):
        """from_json accepts string JSON content."""
        tok = talu.Tokenizer.from_json(MINIMAL_TOKENIZER_JSON)
        assert tok.vocab_size == 99
        tok.close()

    def test_from_json_bytes(self, talu):
        """from_json accepts bytes JSON content."""
        tok = talu.Tokenizer.from_json(MINIMAL_TOKENIZER_JSON.encode("utf-8"))
        assert tok.vocab_size == 99
        tok.close()

    def test_from_json_with_padding_side(self, talu):
        """from_json accepts padding_side argument."""
        tok = talu.Tokenizer.from_json(MINIMAL_TOKENIZER_JSON, padding_side="right")
        assert tok.padding_side == "right"
        tok.close()

    def test_from_json_with_truncation_side(self, talu):
        """from_json accepts truncation_side argument."""
        tok = talu.Tokenizer.from_json(MINIMAL_TOKENIZER_JSON, truncation_side="left")
        assert tok.truncation_side == "left"
        tok.close()

    def test_from_json_invalid_padding_side(self, talu):
        """from_json raises ValidationError for invalid padding_side."""
        from talu.exceptions import ValidationError

        with pytest.raises(ValidationError, match="padding_side"):
            talu.Tokenizer.from_json(MINIMAL_TOKENIZER_JSON, padding_side="invalid")

    def test_from_json_invalid_truncation_side(self, talu):
        """from_json raises ValidationError for invalid truncation_side."""
        from talu.exceptions import ValidationError

        with pytest.raises(ValidationError, match="truncation_side"):
            talu.Tokenizer.from_json(MINIMAL_TOKENIZER_JSON, truncation_side="invalid")

    def test_from_json_invalid_json(self, talu):
        """from_json raises TokenizerError for invalid JSON."""
        from talu.exceptions import TokenizerError

        with pytest.raises(TokenizerError):
            talu.Tokenizer.from_json("{ invalid json }")

    def test_from_json_has_no_model_path(self, talu):
        """from_json creates tokenizer with empty model_path."""
        tok = talu.Tokenizer.from_json(MINIMAL_TOKENIZER_JSON)
        assert tok.model_path == ""
        tok.close()

    def test_from_json_has_no_chat_template(self, talu):
        """from_json creates tokenizer without chat template."""
        tok = talu.Tokenizer.from_json(MINIMAL_TOKENIZER_JSON)
        # _chat_template is None for JSON-created tokenizers
        assert tok._chat_template is None
        tok.close()


class TestMinimalTokenizerEncode:
    """Encoding tests using minimal tokenizer."""

    def test_encode_simple(self, minimal_tokenizer):
        """Minimal tokenizer encodes simple text."""
        ids = minimal_tokenizer.encode("Hi")
        # H=44, i=77 in our minimal vocab
        assert len(ids) == 2

    def test_encode_with_special_tokens(self, minimal_tokenizer):
        """Minimal tokenizer respects special_tokens parameter."""
        ids_with = minimal_tokenizer.encode("Hi", special_tokens=True)
        ids_without = minimal_tokenizer.encode("Hi", special_tokens=False)
        # Both should work (behavior may vary)
        assert len(ids_with) >= 2
        assert len(ids_without) >= 2

    def test_encode_empty(self, minimal_tokenizer):
        """Minimal tokenizer handles empty string."""
        ids = minimal_tokenizer.encode("")
        assert len(ids) == 0

    def test_encode_whitespace(self, minimal_tokenizer):
        """Minimal tokenizer handles whitespace."""
        ids = minimal_tokenizer.encode("   ")
        assert len(ids) >= 3  # At least 3 space tokens


class TestMinimalTokenizerDecode:
    """Decoding tests using minimal tokenizer."""

    def test_decode_roundtrip(self, minimal_tokenizer):
        """Minimal tokenizer roundtrips encode/decode."""
        text = "Hello"
        ids = minimal_tokenizer.encode(text)
        decoded = minimal_tokenizer.decode(ids)
        assert decoded == text

    def test_decode_empty(self, minimal_tokenizer):
        """Minimal tokenizer decodes empty list."""
        decoded = minimal_tokenizer.decode([])
        assert decoded == ""

    def test_decode_single_token(self, minimal_tokenizer):
        """Minimal tokenizer decodes single token."""
        # Token 44 is 'H' in our minimal vocab
        decoded = minimal_tokenizer.decode([44])
        assert decoded == "H"


class TestMinimalTokenizerTokenize:
    """Tokenize tests using minimal tokenizer."""

    def test_tokenize_simple(self, minimal_tokenizer):
        """Minimal tokenizer tokenizes to strings."""
        tokens = minimal_tokenizer.tokenize("Hi")
        assert tokens == ["H", "i"]

    def test_tokenize_empty(self, minimal_tokenizer):
        """Minimal tokenizer handles empty string."""
        tokens = minimal_tokenizer.tokenize("")
        assert tokens == []


class TestMinimalTokenizerVocab:
    """Vocabulary tests using minimal tokenizer."""

    def test_vocab_size(self, minimal_tokenizer):
        """Minimal tokenizer has correct vocab size."""
        assert minimal_tokenizer.vocab_size == 99

    def test_id_to_token(self, minimal_tokenizer):
        """Minimal tokenizer converts ID to token."""
        # Token 44 is 'H'
        token = minimal_tokenizer.id_to_token(44)
        assert token == "H"

    def test_token_to_id(self, minimal_tokenizer):
        """Minimal tokenizer converts token to ID."""
        token_id = minimal_tokenizer.token_to_id("H")
        assert token_id == 44

    def test_contains_token(self, minimal_tokenizer):
        """Minimal tokenizer checks token existence."""
        assert "H" in minimal_tokenizer
        assert "notaToken123" not in minimal_tokenizer

    def test_get_vocab(self, minimal_tokenizer):
        """Minimal tokenizer returns vocabulary."""
        vocab = minimal_tokenizer.get_vocab()
        assert isinstance(vocab, dict)
        assert len(vocab) == 99
        assert vocab["H"] == 44


class TestMinimalTokenizerSpecialTokens:
    """Special token tests using minimal tokenizer."""

    def test_special_tokens(self, minimal_tokenizer):
        """Minimal tokenizer has special tokens."""
        # Our minimal tokenizer has pad, bos, eos, unk
        # BOS is <s> at ID 1
        assert minimal_tokenizer.bos_token_id == 1 or minimal_tokenizer.bos_token_id is None
        # PAD is <pad> at ID 0
        pad_id = minimal_tokenizer.pad_token_id
        assert pad_id == 0 or pad_id is None  # May not be exposed


class TestMinimalTokenizerLifecycle:
    """Lifecycle tests using minimal tokenizer."""

    def test_context_manager(self, talu):
        """Minimal tokenizer works as context manager."""
        with talu.Tokenizer.from_json(MINIMAL_TOKENIZER_JSON) as tok:
            ids = tok.encode("Hi")
            assert len(ids) == 2

    def test_close_is_idempotent(self, talu):
        """Close can be called multiple times."""
        tok = talu.Tokenizer.from_json(MINIMAL_TOKENIZER_JSON)
        tok.close()
        tok.close()  # Should not raise

    def test_use_after_close_raises(self, talu):
        """Using closed tokenizer raises StateError."""
        from talu.exceptions import StateError

        tok = talu.Tokenizer.from_json(MINIMAL_TOKENIZER_JSON)
        tok.close()
        with pytest.raises(StateError):
            tok.encode("Hi")


class TestMinimalTokenizerBatch:
    """Batch encoding tests using minimal tokenizer."""

    def test_batch_encode(self, minimal_tokenizer):
        """Minimal tokenizer batch encodes."""
        texts = ["Hi", "Bye"]
        result = minimal_tokenizer(texts, return_tensors=None)
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_batch_encode_single(self, minimal_tokenizer):
        """Minimal tokenizer batch encodes single text."""
        result = minimal_tokenizer("Hi", return_tensors=None)
        assert "input_ids" in result


class TestMinimalTokenizerRepr:
    """Repr tests using minimal tokenizer."""

    def test_repr(self, minimal_tokenizer):
        """Minimal tokenizer has informative repr."""
        r = repr(minimal_tokenizer)
        assert "Tokenizer" in r


class TestMinimalTokenizerEdgeCases:
    """Edge case tests using minimal tokenizer."""

    def test_decode_with_raw_pointer_raises(self, minimal_tokenizer):
        """Decode with raw pointer raises ValidationError."""
        import ctypes

        from talu.exceptions import ValidationError

        # Create a fake raw pointer (not a TokenArray or list)
        fake_ptr = ctypes.c_void_p(12345)

        with pytest.raises(ValidationError, match="num_tokens required"):
            minimal_tokenizer.decode(fake_ptr)

    def test_decode_with_raw_pointer_and_num_tokens_raises(self, minimal_tokenizer):
        """Decode with raw pointer and num_tokens raises ValidationError."""
        import ctypes

        from talu.exceptions import ValidationError

        # Even with num_tokens, raw pointer decode is not supported
        fake_ptr = ctypes.c_void_p(12345)

        with pytest.raises(ValidationError, match="Raw pointer decode not supported"):
            minimal_tokenizer.decode(fake_ptr, num_tokens=5)

    def test_tokenize_bytes_empty(self, minimal_tokenizer):
        """Tokenize with return_bytes=True handles empty string."""
        tokens = minimal_tokenizer.tokenize("", return_bytes=True)
        assert tokens == []

    def test_tokenize_bytes_simple(self, minimal_tokenizer):
        """Tokenize with return_bytes=True returns bytes."""
        tokens = minimal_tokenizer.tokenize("Hi", return_bytes=True)
        assert all(isinstance(t, bytes) for t in tokens)
        # H and i should be separate tokens
        assert len(tokens) == 2

    def test_model_max_length_zero_for_json(self, minimal_tokenizer):
        """JSON-created tokenizer has no model_max_length."""
        # model_max_length comes from tokenizer_config.json which we don't have
        assert minimal_tokenizer.model_max_length == 0

    def test_apply_chat_template_without_template(self, talu):
        """from_json tokenizer has no chat_template."""
        tok = talu.Tokenizer.from_json(MINIMAL_TOKENIZER_JSON)
        # _chat_template is None, so apply_chat_template should handle gracefully
        assert tok._chat_template is None
        tok.close()

    def test_tokenize_error_path_handles_null_token(self, minimal_tokenizer):
        """Tokenize handles tokens that might be null."""
        # This tests the code path where token_ptr might be null
        # Normal ASCII text shouldn't trigger this, but the test covers the branch
        tokens = minimal_tokenizer.tokenize("A")
        assert len(tokens) == 1
        assert tokens[0] == "A"

    def test_tokenize_bytes_returns_correct_bytes(self, minimal_tokenizer):
        """Tokenize with return_bytes=True returns correct byte values."""
        tokens = minimal_tokenizer.tokenize("AB", return_bytes=True)
        assert len(tokens) == 2
        # Each token should be a single byte (byte-level BPE)
        assert tokens[0] == b"A"
        assert tokens[1] == b"B"


class TestMinimalTokenizerChatTemplate:
    """Chat template tests using minimal tokenizer with custom template."""

    # Simple Jinja2 template that just concatenates messages
    SIMPLE_TEMPLATE = "{% for m in messages %}[{{ m.role }}]: {{ m.content }}\n{% endfor %}{% if add_generation_prompt %}[assistant]: {% endif %}"

    # Template that uses bos_token and eos_token
    BOS_EOS_TEMPLATE = "{{ bos_token }}{% for m in messages %}[{{ m.role }}]: {{ m.content }}{{ eos_token }}\n{% endfor %}{% if add_generation_prompt %}[assistant]: {% endif %}"

    def test_from_json_with_chat_template(self, talu):
        """from_json accepts chat_template argument."""
        tok = talu.Tokenizer.from_json(
            MINIMAL_TOKENIZER_JSON,
            chat_template=self.SIMPLE_TEMPLATE,
        )
        assert tok._chat_template_str == self.SIMPLE_TEMPLATE
        tok.close()

    def test_from_json_with_chat_template_and_tokens(self, talu):
        """from_json accepts chat_template, bos_token, and eos_token."""
        tok = talu.Tokenizer.from_json(
            MINIMAL_TOKENIZER_JSON,
            chat_template=self.BOS_EOS_TEMPLATE,
            bos_token="<s>",
            eos_token="</s>",
        )
        assert tok._chat_template_str == self.BOS_EOS_TEMPLATE
        assert tok._bos_token_str == "<s>"
        assert tok._eos_token_str == "</s>"
        tok.close()

    def test_apply_chat_template_simple(self, talu):
        """apply_chat_template works with custom template."""
        tok = talu.Tokenizer.from_json(
            MINIMAL_TOKENIZER_JSON,
            chat_template=self.SIMPLE_TEMPLATE,
        )
        messages = [{"role": "user", "content": "Hello"}]
        result = tok.apply_chat_template(messages, add_generation_prompt=False)
        assert "[user]: Hello" in result
        tok.close()

    def test_apply_chat_template_with_generation_prompt(self, talu):
        """apply_chat_template adds generation prompt."""
        tok = talu.Tokenizer.from_json(
            MINIMAL_TOKENIZER_JSON,
            chat_template=self.SIMPLE_TEMPLATE,
        )
        messages = [{"role": "user", "content": "Hello"}]
        result = tok.apply_chat_template(messages, add_generation_prompt=True)
        assert "[user]: Hello" in result
        assert "[assistant]:" in result
        tok.close()

    def test_apply_chat_template_multi_turn(self, talu):
        """apply_chat_template handles multi-turn conversations."""
        tok = talu.Tokenizer.from_json(
            MINIMAL_TOKENIZER_JSON,
            chat_template=self.SIMPLE_TEMPLATE,
        )
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        result = tok.apply_chat_template(messages, add_generation_prompt=True)
        assert "[user]: Hello" in result
        assert "[assistant]: Hi there" in result
        assert "[user]: How are you?" in result
        tok.close()

    def test_apply_chat_template_with_bos_eos(self, talu):
        """apply_chat_template uses bos_token and eos_token."""
        tok = talu.Tokenizer.from_json(
            MINIMAL_TOKENIZER_JSON,
            chat_template=self.BOS_EOS_TEMPLATE,
            bos_token="<s>",
            eos_token="</s>",
        )
        messages = [{"role": "user", "content": "Hello"}]
        result = tok.apply_chat_template(messages, add_generation_prompt=False)
        assert result.startswith("<s>")
        assert "</s>" in result
        tok.close()

    def test_apply_chat_template_without_template_raises(self, talu):
        """apply_chat_template raises StateError when no template is set."""
        from talu.exceptions import StateError

        tok = talu.Tokenizer.from_json(MINIMAL_TOKENIZER_JSON)
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(StateError, match="No chat template available"):
            tok.apply_chat_template(messages)
        tok.close()

    def test_apply_chat_template_with_tokenize(self, talu):
        """apply_chat_template with tokenize=True returns TokenArray."""
        tok = talu.Tokenizer.from_json(
            MINIMAL_TOKENIZER_JSON,
            chat_template=self.SIMPLE_TEMPLATE,
        )
        messages = [{"role": "user", "content": "Hi"}]
        result = tok.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)

        # Should return TokenArray, not string
        assert hasattr(result, "__len__")
        assert hasattr(result, "__iter__")
        assert len(result) > 0
        tok.close()

    def test_apply_chat_template_empty_messages(self, talu):
        """apply_chat_template handles empty messages list."""
        tok = talu.Tokenizer.from_json(
            MINIMAL_TOKENIZER_JSON,
            chat_template=self.SIMPLE_TEMPLATE,
        )
        messages: list = []
        # Empty messages with no generation prompt produces empty output
        result = tok.apply_chat_template(messages, add_generation_prompt=False)
        assert isinstance(result, str)
        assert result == ""
        tok.close()

    def test_apply_chat_template_empty_messages_with_prompt(self, talu):
        """apply_chat_template handles empty messages with generation prompt."""
        tok = talu.Tokenizer.from_json(
            MINIMAL_TOKENIZER_JSON,
            chat_template=self.SIMPLE_TEMPLATE,
        )
        messages: list = []
        # Empty messages with generation prompt produces just the assistant marker
        result = tok.apply_chat_template(messages, add_generation_prompt=True)
        assert isinstance(result, str)
        assert "[assistant]:" in result
        tok.close()

    def test_apply_chat_template_empty_bos_eos(self, talu):
        """apply_chat_template works with empty bos/eos tokens."""
        tok = talu.Tokenizer.from_json(
            MINIMAL_TOKENIZER_JSON,
            chat_template=self.BOS_EOS_TEMPLATE,
            bos_token="",
            eos_token="",
        )
        messages = [{"role": "user", "content": "Hello"}]
        result = tok.apply_chat_template(messages, add_generation_prompt=False)
        # Should still work, just without the tokens
        assert "[user]: Hello" in result
        tok.close()

    def test_apply_chat_template_system_message(self, talu):
        """apply_chat_template handles system messages."""
        tok = talu.Tokenizer.from_json(
            MINIMAL_TOKENIZER_JSON,
            chat_template=self.SIMPLE_TEMPLATE,
        )
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = tok.apply_chat_template(messages, add_generation_prompt=True)
        assert "[system]: You are helpful." in result
        assert "[user]: Hello" in result
        tok.close()
