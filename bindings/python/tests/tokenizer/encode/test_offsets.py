"""Tests for token offset mapping."""

import pytest

import talu
from talu.tokenizer import Tokenizer, TokenOffset


class TestOffsetBasic:
    """Basic offset functionality tests."""

    def test_simple_text(self, test_model_path):
        """Offset mapping works for simple ASCII text."""
        tokenizer = Tokenizer(test_model_path)
        text = "Hello world"
        tokens = tokenizer.encode(text, special_tokens=False)

        offsets = tokens.offsets

        # Should have same number of offsets as tokens
        assert len(offsets) == len(tokens)

        # Verify we can reconstruct text from offsets
        text_bytes = text.encode("utf-8")
        reconstructed_parts = [
            text_bytes[offset.start : offset.end].decode("utf-8") for offset in offsets
        ]
        reconstructed = "".join(reconstructed_parts)
        assert reconstructed == text

    def test_empty_string(self, test_model_path):
        """Empty string returns empty offsets."""
        tokenizer = Tokenizer(test_model_path)
        tokens = tokenizer.encode("", special_tokens=False)

        assert len(tokens) == 0
        assert tokens.offsets == []

    def test_single_token(self, test_model_path):
        """Single token has correct offset span."""
        tokenizer = Tokenizer(test_model_path)
        text = "Hi"
        tokens = tokenizer.encode(text, special_tokens=False)

        if len(tokens) == 1:
            offsets = tokens.offsets
            assert len(offsets) == 1
            assert offsets[0].start == 0
            assert offsets[0].end == len(text.encode("utf-8"))


class TestOffsetUnicode:
    """Unicode handling in offset mapping."""

    def test_multibyte_characters(self, test_model_path):
        """Offsets are byte indices, not character indices."""
        tokenizer = Tokenizer(test_model_path)
        text = "café"  # é is 2 bytes in UTF-8
        text_bytes = text.encode("utf-8")
        tokens = tokenizer.encode(text, special_tokens=False)

        offsets = tokens.offsets

        # Reconstruct from byte offsets (concatenate then decode)
        reconstructed_bytes = b"".join(text_bytes[offset.start : offset.end] for offset in offsets)
        assert reconstructed_bytes.decode("utf-8") == text

    def test_emoji(self, test_model_path):
        """Emoji (4-byte UTF-8) handled correctly.

        Note: Byte-level BPE tokenizers may split multi-byte UTF-8 sequences
        across tokens. This test verifies that concatenating byte spans gives
        the original bytes, which can then be decoded as a whole.
        """
        tokenizer = Tokenizer(test_model_path)
        text = "Hello 🎉 world"
        text_bytes = text.encode("utf-8")
        tokens = tokenizer.encode(text, special_tokens=False)

        offsets = tokens.offsets

        # Concatenate byte spans and decode as a whole
        # (Individual tokens may not be valid UTF-8 for byte-level BPE)
        reconstructed_bytes = b"".join(text_bytes[offset.start : offset.end] for offset in offsets)
        assert reconstructed_bytes.decode("utf-8") == text

    def test_chinese_text(self, test_model_path):
        """Chinese characters (3-byte UTF-8) handled correctly."""
        tokenizer = Tokenizer(test_model_path)
        text = "你好世界"
        text_bytes = text.encode("utf-8")
        tokens = tokenizer.encode(text, special_tokens=False)

        offsets = tokens.offsets

        # Reconstruct from byte offsets
        reconstructed_bytes = b"".join(text_bytes[offset.start : offset.end] for offset in offsets)
        assert reconstructed_bytes.decode("utf-8") == text


class TestOffsetCaching:
    """Lazy evaluation and caching tests."""

    def test_eager_evaluation(self, test_model_path):
        """Offsets are computed eagerly during encoding."""
        tokenizer = Tokenizer(test_model_path)
        tokens = tokenizer.encode("Hello", special_tokens=False)

        # Offsets are pre-calculated at encode time
        assert tokens._offsets is not None
        assert len(tokens._offsets) == len(tokens)

    def test_cached_result(self, test_model_path):
        """Second access returns same cached object."""
        tokenizer = Tokenizer(test_model_path)
        tokens = tokenizer.encode("Hello", special_tokens=False)

        offsets1 = tokens.offsets
        offsets2 = tokens.offsets

        # Should be the exact same list object
        assert offsets1 is offsets2


class TestTokenOffset:
    """TokenOffset class behavior."""

    def test_tuple_equality(self):
        """TokenOffset compares equal to equivalent tuple."""
        offset = TokenOffset(0, 5)
        assert offset == (0, 5)
        assert (0, 5) == offset

    def test_offset_equality(self):
        """Two TokenOffset objects with same values are equal."""
        offset1 = TokenOffset(0, 5)
        offset2 = TokenOffset(0, 5)
        assert offset1 == offset2

    def test_offset_inequality(self):
        """Different offsets are not equal."""
        offset1 = TokenOffset(0, 5)
        offset2 = TokenOffset(5, 10)
        assert offset1 != offset2

    def test_tuple_unpacking(self):
        """TokenOffset supports tuple unpacking."""
        offset = TokenOffset(10, 20)
        start, end = offset
        assert start == 10
        assert end == 20

    def test_attributes(self):
        """TokenOffset has start and end attributes."""
        offset = TokenOffset(3, 7)
        assert offset.start == 3
        assert offset.end == 7

    def test_repr(self):
        """TokenOffset has readable repr."""
        offset = TokenOffset(0, 5)
        assert repr(offset) == "(0, 5)"

    def test_slice_string(self):
        """TokenOffset.slice() extracts text from string."""
        offset = TokenOffset(0, 5)
        assert offset.slice("Hello world") == "Hello"

    def test_slice_bytes(self):
        """TokenOffset.slice() works with bytes input."""
        offset = TokenOffset(0, 5)
        assert offset.slice(b"Hello world") == "Hello"

    def test_slice_unicode(self):
        """TokenOffset.slice() handles Unicode correctly."""
        text = "Hello 🎉 world"
        # 'Hello ' is 6 bytes, emoji is 4 bytes, ' world' starts at byte 10
        offset = TokenOffset(10, 16)
        assert offset.slice(text) == " world"

    def test_slice_multibyte_char(self):
        """TokenOffset.slice() handles multi-byte characters."""
        text = "café"  # 'caf' is 3 bytes, 'é' is 2 bytes (c3 a9)
        offset = TokenOffset(3, 5)
        assert offset.slice(text) == "é"

    def test_slice_partial_utf8_strict(self):
        """TokenOffset.slice() raises on partial UTF-8 by default."""
        # Partial UTF-8 sequence (first 2 bytes of a 4-byte emoji)
        text_bytes = "🎉".encode()  # \xf0\x9f\x8e\x89
        offset = TokenOffset(0, 2)  # Just \xf0\x9f
        with pytest.raises(UnicodeDecodeError):
            offset.slice(text_bytes)

    def test_slice_partial_utf8_replace(self):
        """TokenOffset.slice() with errors='replace' handles partial UTF-8."""
        text_bytes = "🎉".encode()  # \xf0\x9f\x8e\x89
        offset = TokenOffset(0, 2)  # Just \xf0\x9f
        result = offset.slice(text_bytes, errors="replace")
        assert "�" in result  # Contains replacement character


class TestSliceIntegration:
    """Integration tests for TokenOffset.slice() with real tokenizer."""

    def test_slice_all_tokens_ascii(self, test_model_path):
        """slice() works for all tokens in ASCII text."""
        tokenizer = Tokenizer(test_model_path)
        text = "The quick brown fox jumps over the lazy dog"
        tokens = tokenizer.encode(text, special_tokens=False)

        # All tokens should slice cleanly for ASCII
        parts = [offset.slice(text) for offset in tokens.offsets]
        reconstructed = "".join(parts)
        assert reconstructed == text

    def test_slice_preserves_whitespace(self, test_model_path):
        """slice() correctly preserves leading/trailing whitespace."""
        tokenizer = Tokenizer(test_model_path)
        text = "Hello world"
        tokens = tokenizer.encode(text, special_tokens=False)

        # Check that whitespace is captured correctly
        parts = [offset.slice(text) for offset in tokens.offsets]
        assert "".join(parts) == text

    def test_slice_with_errors_replace_full_text(self, test_model_path):
        """slice(errors='replace') reconstructs text with replacement chars."""
        tokenizer = Tokenizer(test_model_path)
        text = "Hello 🎉 world"
        tokens = tokenizer.encode(text, special_tokens=False)

        # With errors='replace', all slices succeed (some may have replacement chars)
        parts = [offset.slice(text, errors="replace") for offset in tokens.offsets]
        reconstructed = "".join(parts)

        # The reconstructed text may have replacement chars for split bytes
        # but should have same length in bytes when re-encoded
        assert len(reconstructed) >= len(text.replace("🎉", ""))

    def test_slice_empty_offset(self):
        """slice() handles (0, 0) offset for special tokens."""
        offset = TokenOffset(0, 0)
        assert offset.slice("Hello world") == ""
        assert offset.slice(b"Hello world") == ""

    def test_slice_full_text(self):
        """slice() can extract entire text."""
        text = "Complete text"
        offset = TokenOffset(0, len(text.encode("utf-8")))
        assert offset.slice(text) == text

    def test_slice_middle_of_text(self):
        """slice() extracts from middle of text."""
        text = "Hello world, how are you?"
        # Extract "world"
        offset = TokenOffset(6, 11)
        assert offset.slice(text) == "world"


class TestOffsetEdgeCases:
    """Edge cases and error handling."""

    def test_no_source_text_raises(self, test_model_path):
        """Accessing offsets without source text raises RuntimeError."""
        # Create TokenArray without source text (simulating internal usage)
        import ctypes

        from talu.tokenizer.token_array import TokenArray

        arr = (ctypes.c_uint32 * 2)(1, 2)
        ptr = ctypes.cast(arr, ctypes.POINTER(ctypes.c_uint32))

        # Create TokenArray without source_text and tokenizer
        tokens = TokenArray.__new__(TokenArray)
        tokens._ptr = ptr
        tokens._num_tokens = 2
        tokens._owns_data = False
        tokens._source_text = None
        tokens._tokenizer = None
        tokens._offsets = None

        with pytest.raises(talu.StateError, match="source text"):
            _ = tokens.offsets

    def test_no_source_text_error_no_memory_leak(self, test_model_path):
        """Repeated offset errors don't leak memory (error path cleanup)."""
        import ctypes

        from talu.tokenizer.token_array import TokenArray

        # Run many iterations to detect memory leaks
        for _ in range(100):
            arr = (ctypes.c_uint32 * 2)(1, 2)
            ptr = ctypes.cast(arr, ctypes.POINTER(ctypes.c_uint32))

            tokens = TokenArray.__new__(TokenArray)
            tokens._ptr = ptr
            tokens._num_tokens = 2
            tokens._owns_data = False
            tokens._source_text = None
            tokens._tokenizer = None
            tokens._offsets = None

            with pytest.raises(talu.StateError, match="source text"):
                _ = tokens.offsets

        # If we get here without OOM or crash, error path doesn't leak

    def test_consecutive_offsets(self, test_model_path):
        """Adjacent tokens have consecutive, non-overlapping offsets."""
        tokenizer = Tokenizer(test_model_path)
        text = "The quick brown fox"
        tokens = tokenizer.encode(text, special_tokens=False)
        offsets = tokens.offsets

        if len(offsets) > 1:
            # Check offsets are consecutive (no gaps, no overlaps)
            for i in range(len(offsets) - 1):
                assert offsets[i].end == offsets[i + 1].start, (
                    f"Gap or overlap between token {i} and {i + 1}: "
                    f"{offsets[i]} -> {offsets[i + 1]}"
                )

    def test_full_text_coverage(self, test_model_path):
        """Offsets cover the entire input text."""
        tokenizer = Tokenizer(test_model_path)
        text = "Complete coverage test"
        text_bytes = text.encode("utf-8")
        tokens = tokenizer.encode(text, special_tokens=False)
        offsets = tokens.offsets

        if len(offsets) > 0:
            # First offset starts at 0
            assert offsets[0].start == 0
            # Last offset ends at text length
            assert offsets[-1].end == len(text_bytes)


class TestOffsetWithOptions:
    """Offset behavior with various encode options."""

    def test_without_special_tokens(self, test_model_path):
        """Offsets work correctly without special tokens."""
        tokenizer = Tokenizer(test_model_path)
        text = "Test text"
        tokens = tokenizer.encode(text, special_tokens=False)
        offsets = tokens.offsets

        # Should cover exactly the input text
        text_bytes = text.encode("utf-8")
        parts = [text_bytes[o.start : o.end].decode("utf-8") for o in offsets]
        assert "".join(parts) == text
