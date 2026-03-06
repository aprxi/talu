"""Regression tests for generated tokenizer ctypes field types."""

import ctypes


class TestTokenizerGeneratedFieldTypes:
    """Generated tokenizer result structs must preserve typed pointer fields."""

    def test_encode_result_uses_typed_pointers(self):
        from talu._native import EncodeResult, TokenOffset

        fields = dict(EncodeResult._fields_)
        assert fields["ids"] == ctypes.POINTER(ctypes.c_uint32)
        assert fields["offsets"] == ctypes.POINTER(TokenOffset)
        assert fields["attention_mask"] == ctypes.POINTER(ctypes.c_uint32)
        assert fields["special_tokens_mask"] == ctypes.POINTER(ctypes.c_uint32)

    def test_tokenize_result_uses_typed_pointers(self):
        from talu._native import TokenizeBytesResult, TokenizeResult

        tokenize_fields = dict(TokenizeResult._fields_)
        tokenize_bytes_fields = dict(TokenizeBytesResult._fields_)

        assert tokenize_fields["tokens"] == ctypes.POINTER(ctypes.c_char_p)
        assert tokenize_bytes_fields["data"] == ctypes.POINTER(ctypes.c_uint8)
        assert tokenize_bytes_fields["offsets"] == ctypes.POINTER(ctypes.c_size_t)

    def test_eos_batch_and_vocab_results_use_typed_pointers(self):
        from talu._native import BatchEncodeResult, EosTokenResult, VocabResult

        eos_fields = dict(EosTokenResult._fields_)
        batch_fields = dict(BatchEncodeResult._fields_)
        vocab_fields = dict(VocabResult._fields_)

        assert eos_fields["tokens"] == ctypes.POINTER(ctypes.c_uint32)
        assert batch_fields["ids"] == ctypes.POINTER(ctypes.c_uint32)
        assert batch_fields["offsets"] == ctypes.POINTER(ctypes.c_size_t)
        assert vocab_fields["tokens"] == ctypes.POINTER(ctypes.c_char_p)
        assert vocab_fields["lengths"] == ctypes.POINTER(ctypes.c_uint32)
        assert vocab_fields["ids"] == ctypes.POINTER(ctypes.c_uint32)
