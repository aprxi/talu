"""
Byte-level BPE multi-byte decode reference tests.

Validates that talu's tokenizer decodes multi-byte UTF-8 characters
identically to HuggingFace transformers, including edge cases where
byte-level BPE splits a single character across multiple tokens.

Regression guard: before the fix in api.zig (decodeRawBytes) and
iterator.zig (UTF-8 streaming buffer), decoding individual tokens
that produced incomplete UTF-8 byte sequences would replace them
with U+FFFD replacement characters.
"""

import pytest
from transformers import AutoTokenizer

# Texts with multi-byte UTF-8 characters that exercise byte-level BPE
# token boundary splitting.
EMOJI_TEXTS = [
    "Hello 😊",
    "🎉 celebration 🎊",
    "emoji: 🚀🔥💡",
    "Space emoji: 😊 end",
    "Multiple: 🌍🌎🌏",
]

CJK_TEXTS = [
    "Hello 世界!",
    "日本語テスト",
    "中文测试",
    "한국어 테스트",
]

MIXED_SCRIPT_TEXTS = [
    "café résumé naïve",
    "mixed: café 日本語 🌍",
    "Привет мир 🇷🇺",
]


@pytest.fixture(scope="module")
def hf_tokenizer(test_model_path):
    """HuggingFace tokenizer for the test model."""
    from talu.repository import resolve_path

    local_path = resolve_path(test_model_path)
    return AutoTokenizer.from_pretrained(
        local_path,
        trust_remote_code=True,
        local_files_only=True,
    )


class TestBPEMultibyteBatchDecode:
    """Batch decode of multi-byte text must match HuggingFace exactly."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", EMOJI_TEXTS)
    def test_emoji_decode_matches_hf(self, tokenizer, hf_tokenizer, text):
        """Emoji-containing text decodes identically to transformers."""
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        talu_decoded = tokenizer.decode(hf_tokens)
        hf_decoded = hf_tokenizer.decode(hf_tokens)

        assert talu_decoded == hf_decoded, (
            f"Emoji decode mismatch for {repr(text)}\n"
            f"  tokens: {hf_tokens}\n"
            f"  talu: {repr(talu_decoded)}\n"
            f"  hf: {repr(hf_decoded)}"
        )

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", CJK_TEXTS)
    def test_cjk_decode_matches_hf(self, tokenizer, hf_tokenizer, text):
        """CJK text decodes identically to transformers."""
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        talu_decoded = tokenizer.decode(hf_tokens)
        hf_decoded = hf_tokenizer.decode(hf_tokens)

        assert talu_decoded == hf_decoded, (
            f"CJK decode mismatch for {repr(text)}\n"
            f"  tokens: {hf_tokens}\n"
            f"  talu: {repr(talu_decoded)}\n"
            f"  hf: {repr(hf_decoded)}"
        )

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", MIXED_SCRIPT_TEXTS)
    def test_mixed_script_decode_matches_hf(self, tokenizer, hf_tokenizer, text):
        """Mixed-script text decodes identically to transformers."""
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        talu_decoded = tokenizer.decode(hf_tokens)
        hf_decoded = hf_tokenizer.decode(hf_tokens)

        assert talu_decoded == hf_decoded, (
            f"Mixed script decode mismatch for {repr(text)}\n"
            f"  tokens: {hf_tokens}\n"
            f"  talu: {repr(talu_decoded)}\n"
            f"  hf: {repr(hf_decoded)}"
        )


class TestBPEMultibyteSingleTokenDecode:
    """Per-token decode must not produce U+FFFD for valid byte sequences.

    For byte-level BPE, a single token may decode to an incomplete UTF-8
    byte sequence (e.g., the first 3 bytes of a 4-byte emoji).  When
    decoded individually, the batch decode API sanitizes incomplete
    sequences.  This is expected behavior for the batch API (the caller
    has all the tokens), but the test verifies that multi-token characters
    decode correctly when all their tokens are present.
    """

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", EMOJI_TEXTS + CJK_TEXTS)
    def test_full_sequence_decode_no_replacement(self, tokenizer, hf_tokenizer, text):
        """Full token sequence decode produces no U+FFFD."""
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(hf_tokens)

        assert "\ufffd" not in decoded, (
            f"U+FFFD in full decode of {repr(text)}\n"
            f"  tokens: {hf_tokens}\n"
            f"  decoded: {repr(decoded)}"
        )

    @pytest.mark.requires_model
    @pytest.mark.parametrize("text", EMOJI_TEXTS)
    def test_cumulative_decode_matches_hf(self, tokenizer, hf_tokenizer, text):
        """Cumulative decode at each step matches HuggingFace.

        This is the key streaming invariant: decode(tokens[:i]) must
        match hf.decode(tokens[:i]) for every prefix length.
        """
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        for i in range(1, len(hf_tokens) + 1):
            talu_partial = tokenizer.decode(hf_tokens[:i])
            hf_partial = hf_tokenizer.decode(hf_tokens[:i])

            assert talu_partial == hf_partial, (
                f"Cumulative decode mismatch at step {i}/{len(hf_tokens)} "
                f"for {repr(text)}\n"
                f"  tokens[:i]: {hf_tokens[:i]}\n"
                f"  talu: {repr(talu_partial)}\n"
                f"  hf: {repr(hf_partial)}"
            )
