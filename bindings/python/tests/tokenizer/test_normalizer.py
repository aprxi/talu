"""
Tests for tokenizer normalizer pipeline.

Verifies that normalizer types (Lowercase, StripAccents, BertNormalizer) and
Unicode normalization forms (NFC, NFD, NFKC, NFKD) are correctly applied during
encoding. Uses the minimal byte-level BPE tokenizer with custom normalizer JSON.
"""

import json

from tests.tokenizer.conftest import MINIMAL_TOKENIZER_JSON


def _with_normalizer(normalizer):
    """Build minimal tokenizer JSON with a custom normalizer section."""
    config = json.loads(MINIMAL_TOKENIZER_JSON)
    config["normalizer"] = normalizer
    return json.dumps(config)


# =============================================================================
# Type dispatch (Lowercase, StripAccents, BertNormalizer)
# =============================================================================


class TestNormalizerTypeDispatch:
    """Normalizer types that were silently ignored by the fast-path loader."""

    def test_type_lowercase(self, talu):
        """Type 'Lowercase' lowercases input before encoding."""
        tok = talu.Tokenizer.from_json(_with_normalizer({"type": "Lowercase"}))
        assert tok.encode("HELLO").tolist() == tok.encode("hello").tolist()
        tok.close()

    def test_type_lowercase_roundtrip(self, talu):
        """Lowercase normalizer: decode(encode(UPPER)) == lower."""
        tok = talu.Tokenizer.from_json(_with_normalizer({"type": "Lowercase"}))
        assert tok.decode(tok.encode("HELLO")) == "hello"
        tok.close()

    def test_type_strip_accents(self, talu):
        """Type 'StripAccents' removes accents before encoding."""
        tok = talu.Tokenizer.from_json(_with_normalizer({"type": "StripAccents"}))
        # Composed e-acute (U+00E9) → 'e'
        assert tok.decode(tok.encode("caf\u00e9")) == "cafe"
        tok.close()

    def test_type_bert_normalizer(self, talu):
        """Type 'BertNormalizer' lowercases AND strips accents."""
        tok = talu.Tokenizer.from_json(_with_normalizer({"type": "BertNormalizer"}))
        # Uppercase E-acute (U+00C9) → lowercase + strip → 'e'
        assert tok.decode(tok.encode("CAF\u00c9")) == "cafe"
        tok.close()

    def test_type_bert_cjk_spacing(self, talu):
        """BertNormalizer adds spaces around CJK characters."""
        tok_bert = talu.Tokenizer.from_json(
            _with_normalizer({"type": "BertNormalizer"})
        )
        tok_none = talu.Tokenizer.from_json(_with_normalizer(None))
        # CJK char gets surrounding spaces → more tokens
        bert_count = len(tok_bert.encode("a\u65e5b"))
        none_count = len(tok_none.encode("a\u65e5b"))
        assert bert_count > none_count
        tok_bert.close()
        tok_none.close()

    def test_sequence_with_type_lowercase(self, talu):
        """Type-based normalizers work inside a Sequence."""
        normalizer = {
            "type": "Sequence",
            "normalizers": [{"type": "Lowercase"}, {"type": "NFC"}],
        }
        tok = talu.Tokenizer.from_json(_with_normalizer(normalizer))
        assert tok.decode(tok.encode("HELLO")) == "hello"
        tok.close()


# =============================================================================
# Unicode normalization flags (NFD, NFKC, NFKD)
# =============================================================================


class TestNormalizerUnicodeNormalization:
    """Unicode normalization forms that were parsed but never applied."""

    def test_nfkc_fullwidth_to_ascii(self, talu):
        """NFKC normalizes fullwidth 'A' (U+FF21) to ASCII 'A'."""
        tok = talu.Tokenizer.from_json(_with_normalizer({"type": "NFKC"}))
        assert tok.decode(tok.encode("\uff21")) == "A"
        tok.close()

    def test_nfkc_ligature(self, talu):
        """NFKC decomposes fi-ligature (U+FB01) to 'fi'."""
        tok = talu.Tokenizer.from_json(_with_normalizer({"type": "NFKC"}))
        assert tok.decode(tok.encode("\ufb01")) == "fi"
        tok.close()

    def test_nfd_decomposes(self, talu):
        """NFD decomposes composed e-acute into more bytes (more tokens)."""
        tok_nfd = talu.Tokenizer.from_json(_with_normalizer({"type": "NFD"}))
        tok_none = talu.Tokenizer.from_json(_with_normalizer(None))
        # Composed e-acute (2 UTF-8 bytes) → decomposed e + combining acute (3 bytes)
        nfd_count = len(tok_nfd.encode("\u00e9"))
        none_count = len(tok_none.encode("\u00e9"))
        assert nfd_count > none_count
        tok_nfd.close()
        tok_none.close()

    def test_nfkd_fullwidth(self, talu):
        """NFKD normalizes fullwidth 'A' (U+FF21) to ASCII 'A'."""
        tok = talu.Tokenizer.from_json(_with_normalizer({"type": "NFKD"}))
        assert tok.decode(tok.encode("\uff21")) == "A"
        tok.close()

    def test_nfc_composes(self, talu):
        """NFC composes decomposed e-acute into fewer bytes (fewer tokens)."""
        tok_nfc = talu.Tokenizer.from_json(_with_normalizer({"type": "NFC"}))
        tok_none = talu.Tokenizer.from_json(_with_normalizer(None))
        # Decomposed: e (1 byte) + combining acute (2 bytes) = 3 bytes
        # NFC composed: e-acute (2 bytes)
        nfc_count = len(tok_nfc.encode("e\u0301"))
        none_count = len(tok_none.encode("e\u0301"))
        assert nfc_count < none_count
        tok_nfc.close()
        tok_none.close()
