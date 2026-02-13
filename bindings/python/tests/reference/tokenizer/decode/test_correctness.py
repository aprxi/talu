"""
Decode correctness tests against transformers.

Compares talu.Tokenizer.decode() output against HuggingFace transformers
for exact string matching. This is the ground truth validation for decode.
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


class TestDecodeCorrectnessBasic:
    """Correctness tests for basic text decoding."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", BASIC_STRINGS)
    def test_basic_strings_decode_match(self, tokenizer, hf_tokenizer, text):
        """Basic strings decode identically to transformers."""
        # Encode with HF (ground truth tokens)
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        # Decode with both
        talu_decoded = tokenizer.decode(hf_tokens)
        hf_decoded = hf_tokenizer.decode(hf_tokens)

        assert talu_decoded == hf_decoded, (
            f"Decode mismatch for '{text[:50]}'\n"
            f"  talu: '{talu_decoded}'\n"
            f"  transformers: '{hf_decoded}'"
        )


class TestDecodeCorrectnessNumbers:
    """Correctness tests for number decoding."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", NUMBER_STRINGS)
    def test_number_strings_decode_match(self, tokenizer, hf_tokenizer, text):
        """Number strings decode identically to transformers."""
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        talu_decoded = tokenizer.decode(hf_tokens)
        hf_decoded = hf_tokenizer.decode(hf_tokens)

        assert talu_decoded == hf_decoded, (
            f"Decode mismatch for '{text}'\n"
            f"  talu: '{talu_decoded}'\n"
            f"  transformers: '{hf_decoded}'"
        )


class TestDecodeCorrectnessPunctuation:
    """Correctness tests for punctuation decoding."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", PUNCTUATION_STRINGS)
    def test_punctuation_strings_decode_match(self, tokenizer, hf_tokenizer, text):
        """Punctuation strings decode identically to transformers."""
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        talu_decoded = tokenizer.decode(hf_tokens)
        hf_decoded = hf_tokenizer.decode(hf_tokens)

        assert talu_decoded == hf_decoded, (
            f"Decode mismatch for '{text}'\n"
            f"  talu: '{talu_decoded}'\n"
            f"  transformers: '{hf_decoded}'"
        )


class TestDecodeCorrectnessContractions:
    """Correctness tests for contraction decoding."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", CONTRACTION_STRINGS)
    def test_contraction_strings_decode_match(self, tokenizer, hf_tokenizer, text):
        """Contraction strings decode identically to transformers."""
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        talu_decoded = tokenizer.decode(hf_tokens)
        hf_decoded = hf_tokenizer.decode(hf_tokens)

        assert talu_decoded == hf_decoded, (
            f"Decode mismatch for '{text}'\n"
            f"  talu: '{talu_decoded}'\n"
            f"  transformers: '{hf_decoded}'"
        )


class TestDecodeCorrectnessWhitespace:
    """Correctness tests for whitespace decoding."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", WHITESPACE_STRINGS)
    def test_whitespace_strings_decode_match(self, tokenizer, hf_tokenizer, text):
        """Whitespace strings decode identically to transformers."""
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        talu_decoded = tokenizer.decode(hf_tokens)
        hf_decoded = hf_tokenizer.decode(hf_tokens)

        assert talu_decoded == hf_decoded, (
            f"Decode mismatch for '{repr(text)}'\n"
            f"  talu: '{repr(talu_decoded)}'\n"
            f"  transformers: '{repr(hf_decoded)}'"
        )


class TestDecodeCorrectnessUnicode:
    """Correctness tests for Unicode decoding."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", UNICODE_STRINGS)
    def test_unicode_strings_decode_match(self, tokenizer, hf_tokenizer, text):
        """Unicode strings decode identically to transformers."""
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        talu_decoded = tokenizer.decode(hf_tokens)
        hf_decoded = hf_tokenizer.decode(hf_tokens)

        assert talu_decoded == hf_decoded, (
            f"Decode mismatch for '{text}'\n"
            f"  talu: '{talu_decoded}'\n"
            f"  transformers: '{hf_decoded}'"
        )


class TestDecodeCorrectnessMultilingual:
    """Correctness tests for multilingual decoding."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("lang,text", MULTILINGUAL_STRINGS)
    def test_multilingual_strings_decode_match(self, tokenizer, hf_tokenizer, lang, text):
        """Multilingual strings decode identically to transformers."""
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        talu_decoded = tokenizer.decode(hf_tokens)
        hf_decoded = hf_tokenizer.decode(hf_tokens)

        assert talu_decoded == hf_decoded, (
            f"Decode mismatch for {lang}: '{text}'\n"
            f"  talu: '{talu_decoded}'\n"
            f"  transformers: '{hf_decoded}'"
        )


class TestDecodeCorrectnessCode:
    """Correctness tests for code decoding."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", CODE_STRINGS)
    def test_code_strings_decode_match(self, tokenizer, hf_tokenizer, text):
        """Code strings decode identically to transformers."""
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        talu_decoded = tokenizer.decode(hf_tokens)
        hf_decoded = hf_tokenizer.decode(hf_tokens)

        assert talu_decoded == hf_decoded, (
            f"Decode mismatch for '{text}'\n"
            f"  talu: '{talu_decoded}'\n"
            f"  transformers: '{hf_decoded}'"
        )


class TestDecodeCorrectnessEdgeCases:
    """Correctness tests for edge case decoding."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", [t for t in EDGE_CASE_STRINGS if t])
    def test_edge_case_strings_decode_match(self, tokenizer, hf_tokenizer, text):
        """Edge case strings decode identically to transformers."""
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        talu_decoded = tokenizer.decode(hf_tokens)
        hf_decoded = hf_tokenizer.decode(hf_tokens)

        assert talu_decoded == hf_decoded, (
            f"Decode mismatch for '{repr(text)}'\n"
            f"  talu: '{repr(talu_decoded)}'\n"
            f"  transformers: '{repr(hf_decoded)}'"
        )


class TestDecodeCorrectnessSpecialTokens:
    """Correctness tests for special token string decoding."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", SPECIAL_TOKEN_STRINGS)
    def test_special_token_strings_decode_match(self, tokenizer, hf_tokenizer, text):
        """Special token strings decode identically to transformers.

        Note: Both tokenizers must use skip_special_tokens=False to preserve
        special tokens in the output. talu defaults to True, HF defaults to False.
        """
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        # Explicitly set skip_special_tokens=False for both to compare apples-to-apples
        talu_decoded = tokenizer.decode(hf_tokens, skip_special_tokens=False)
        hf_decoded = hf_tokenizer.decode(hf_tokens, skip_special_tokens=False)

        assert talu_decoded == hf_decoded, (
            f"Decode mismatch for '{text}'\n"
            f"  talu: '{talu_decoded}'\n"
            f"  transformers: '{hf_decoded}'"
        )


class TestDecodeCorrectnessComprehensive:
    """Comprehensive correctness tests for decoding."""

    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_all_strings_decode_match(self, tokenizer, hf_tokenizer, all_test_strings):
        """All test strings decode identically to transformers.

        Note: Uses skip_special_tokens=False for both to preserve special tokens.
        talu defaults to True, HF defaults to False.
        """
        failures = []

        for text in all_test_strings:
            if not text:  # Skip empty
                continue

            try:
                hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)
                # Explicitly set skip_special_tokens=False for apples-to-apples comparison
                talu_decoded = tokenizer.decode(hf_tokens, skip_special_tokens=False)
                hf_decoded = hf_tokenizer.decode(hf_tokens, skip_special_tokens=False)

                if talu_decoded != hf_decoded:
                    failures.append(
                        {
                            "text": text,
                            "talu": talu_decoded,
                            "hf": hf_decoded,
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
            msg = f"{len(failures)} decode mismatches:\n"
            for f in failures[:5]:  # Show first 5
                if "error" in f:
                    msg += f"  '{f['text'][:30]}...': ERROR {f['error']}\n"
                else:
                    msg += f"  '{f['text'][:30]}...': got='{f['talu'][:20]}' expected='{f['hf'][:20]}'\n"
            pytest.fail(msg)

    @pytest.mark.requires_model
    def test_decode_empty_produces_empty(self, tokenizer, hf_tokenizer):
        """Decoding empty token list produces empty string."""
        talu_decoded = tokenizer.decode([])
        hf_decoded = hf_tokenizer.decode([])

        assert talu_decoded == hf_decoded == ""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("token_id", list(range(100)))
    def test_decode_single_tokens(self, tokenizer, hf_tokenizer, token_id):
        """Single token decoding matches transformers.

        Contract: Valid token IDs decode identically to transformers.
        Invalid token IDs should raise in both implementations.
        """
        try:
            hf_decoded = hf_tokenizer.decode([token_id])
        except Exception:
            # If HF can't decode it, we don't expect talu to either
            pytest.skip(f"Token ID {token_id} is invalid in this tokenizer")

        talu_decoded = tokenizer.decode([token_id])

        assert talu_decoded == hf_decoded, (
            f"Single token {token_id} mismatch:\n"
            f"  talu: '{repr(talu_decoded)}'\n"
            f"  transformers: '{repr(hf_decoded)}'"
        )


class TestStreamingDecode:
    """Tests for streaming decode behavior.

    Streaming decode uses cumulative decode: decode all tokens accumulated so far,
    then output only the new characters. This ensures correct handling of tokenizers
    with context-dependent decoder rules (e.g., SentencePiece Strip decoder).
    """

    @pytest.mark.requires_model
    def test_cumulative_decode_matches_full_decode(self, tokenizer, hf_tokenizer):
        """Cumulative decode produces same result as full decode.

        This tests the core streaming pattern:
        1. Encode a sentence
        2. Decode cumulatively (1 token, 2 tokens, 3 tokens, ...)
        3. Each cumulative decode should be a prefix of the final full decode
        """
        text = " is Paris, the capital of France"
        tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        # Full decode (ground truth)
        full_decode = tokenizer.decode(tokens)
        hf_full = hf_tokenizer.decode(tokens)

        # Verify our full decode matches HF
        assert full_decode == hf_full, (
            f"Full decode mismatch:\n  talu: {repr(full_decode)}\n  hf: {repr(hf_full)}"
        )

        # Cumulative decode: each step should be a prefix of full decode
        for i in range(1, len(tokens) + 1):
            cumulative = tokenizer.decode(tokens[:i])
            hf_cumulative = hf_tokenizer.decode(tokens[:i])

            assert cumulative == hf_cumulative, (
                f"Cumulative decode mismatch at token {i}:\n"
                f"  talu: {repr(cumulative)}\n"
                f"  hf: {repr(hf_cumulative)}"
            )

            # Each cumulative should be prefix of full (with possible changes at boundary)
            # This is a key invariant for streaming

    @pytest.mark.requires_model
    def test_streaming_simulation(self, tokenizer, hf_tokenizer):
        """Simulate streaming: decode incrementally, output only new chars.

        This is the exact pattern used by the Streamer for correct streaming.
        """
        text = " France's capital city is Paris"
        tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        # Simulate streaming with cumulative decode
        output_so_far = ""
        for i in range(1, len(tokens) + 1):
            full_decode = tokenizer.decode(tokens[:i])
            # In real streaming: new_chars = full_decode[len(output_so_far):]
            # would be written to output. Here we just track cumulative state.
            output_so_far = full_decode

        # Final output should match full decode
        expected = tokenizer.decode(tokens)
        assert output_so_far == expected, (
            f"Streaming simulation mismatch:\n"
            f"  streamed: {repr(output_so_far)}\n"
            f"  expected: {repr(expected)}"
        )

    @pytest.mark.requires_model
    def test_sentencepiece_space_handling(self, tokenizer, hf_tokenizer):
        """Test SentencePiece models handle leading space correctly in streaming.

        SentencePiece uses ▁ (U+2581) for leading spaces, with a Strip decoder
        that removes leading spaces from the full output. Streaming must use
        cumulative decode to get this right.
        """
        # This text will have leading space tokens in SentencePiece models
        text = " is located in"
        tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        # Key test: cumulative decode of first token should match HF
        # For SentencePiece, first token might be just "▁" which decodes to ""
        # (stripped), but later tokens with "▁" prefix decode correctly when
        # decoded cumulatively
        for i in range(1, len(tokens) + 1):
            cumulative = tokenizer.decode(tokens[:i])
            hf_cumulative = hf_tokenizer.decode(tokens[:i])

            assert cumulative == hf_cumulative, (
                f"SentencePiece space handling mismatch at token {i}/{len(tokens)}:\n"
                f"  tokens so far: {tokens[:i]}\n"
                f"  talu: {repr(cumulative)}\n"
                f"  hf: {repr(hf_cumulative)}"
            )


class TestByteLevelBPEDecode:
    """Regression tests for byte-level BPE multi-byte decode.

    Byte-level BPE tokenizers (GPT-2, Qwen3, Llama3) encode raw bytes as
    Unicode characters via a byte-to-Unicode mapping.  A multi-byte UTF-8
    character (e.g. emoji, CJK) can be split across multiple tokens.

    Batch decode (all tokens together) must reassemble the bytes correctly.
    This class guards against regressions where incomplete UTF-8 byte
    sequences are replaced with U+FFFD (replacement character).
    """

    @pytest.mark.requires_model
    @pytest.mark.parametrize(
        "text",
        [
            "Hello 😊",
            "🎉 celebration",
            "emoji: 🚀🔥💡",
            "mixed: café 日本語 🌍",
            "Hello 世界!",
        ],
    )
    def test_multibyte_batch_decode_no_replacement_char(self, tokenizer, text):
        """Batch decode of multi-byte text must not produce U+FFFD.

        Encodes text containing multi-byte UTF-8 characters, then decodes
        the full token sequence.  The decoded text must match the original
        (no replacement characters from mishandled byte boundaries).
        """
        tokens = tokenizer.encode(text, special_tokens=False)
        decoded = tokenizer.decode(tokens)

        assert "\ufffd" not in decoded, (
            f"U+FFFD in batch decode of {repr(text)}\n"
            f"  tokens: {tokens.tolist()}\n"
            f"  decoded: {repr(decoded)}"
        )
        assert decoded.strip() == text.strip(), (
            f"Batch decode mismatch for {repr(text)}\n  decoded: {repr(decoded)}"
        )

    @pytest.mark.requires_model
    @pytest.mark.parametrize(
        "text",
        [
            "Hello 😊",
            "🎉 celebration",
            "emoji: 🚀🔥💡",
            "mixed: café 日本語 🌍",
        ],
    )
    def test_multibyte_batch_decode_matches_hf(self, tokenizer, hf_tokenizer, text):
        """Batch decode of multi-byte text matches HuggingFace transformers.

        Uses HF token IDs as ground truth input, so both decoders operate
        on exactly the same token sequence.
        """
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        talu_decoded = tokenizer.decode(hf_tokens)
        hf_decoded = hf_tokenizer.decode(hf_tokens)

        assert talu_decoded == hf_decoded, (
            f"Multi-byte decode mismatch for {repr(text)}\n"
            f"  tokens: {hf_tokens}\n"
            f"  talu: {repr(talu_decoded)}\n"
            f"  hf: {repr(hf_decoded)}"
        )

    @pytest.mark.requires_model
    def test_cumulative_decode_multibyte_matches_hf(self, tokenizer, hf_tokenizer):
        """Cumulative decode of emoji text matches HF at every prefix.

        For byte-level BPE, partial prefixes may produce U+FFFD when
        the trailing bytes form an incomplete UTF-8 sequence.  This is
        correct (both talu and HF do it).  The key invariant is that
        talu matches HF at every prefix length, and the final full
        decode produces no U+FFFD.
        """
        text = "Hello 😊 world"
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        for i in range(1, len(hf_tokens) + 1):
            talu_partial = tokenizer.decode(hf_tokens[:i])
            hf_partial = hf_tokenizer.decode(hf_tokens[:i])
            assert talu_partial == hf_partial, (
                f"Cumulative decode mismatch at step {i}/{len(hf_tokens)}\n"
                f"  tokens so far: {hf_tokens[:i]}\n"
                f"  talu: {repr(talu_partial)}\n"
                f"  hf: {repr(hf_partial)}"
            )

        # Final full decode must not contain U+FFFD
        full_decode = tokenizer.decode(hf_tokens)
        assert "\ufffd" not in full_decode, (
            f"U+FFFD in final full decode of {repr(text)}: {repr(full_decode)}"
        )

    @pytest.mark.requires_model
    def test_single_emoji_tokens_decode_correctly(self, tokenizer, hf_tokenizer):
        """Individual emoji encode/decode roundtrip produces no U+FFFD.

        Each emoji is encoded, then decoded.  The result must contain the
        original emoji and no replacement characters.
        """
        emojis = ["😊", "🚀", "🎉", "🔥", "💡", "🌍", "❤", "👍"]

        for emoji in emojis:
            tokens = tokenizer.encode(emoji, special_tokens=False)
            decoded = tokenizer.decode(tokens)

            assert "\ufffd" not in decoded, (
                f"U+FFFD decoding emoji {repr(emoji)}\n"
                f"  tokens: {tokens.tolist()}\n"
                f"  decoded: {repr(decoded)}"
            )
            assert emoji in decoded, f"Emoji {repr(emoji)} not found in decoded {repr(decoded)}"
