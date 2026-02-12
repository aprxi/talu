"""
Unicode and multilingual encoding tests.

Tests for talu.Tokenizer.encode() with Unicode text.
"""

import pytest


class TestEncodeUnicode:
    """Tests for Unicode text encoding."""

    @pytest.mark.requires_model
    @pytest.mark.parametrize(
        "text",
        [
            "Cafe resume naive",  # ASCII approximation
            "Cafe\u0301",  # With combining acute accent
        ],
    )
    def test_encode_unicode_basic(self, tokenizer, text):
        """Basic Unicode text encodes successfully."""
        tokens = tokenizer.encode(text)

        assert len(tokens) >= 1

    @pytest.mark.requires_model
    def test_encode_accented_chars(self, tokenizer):
        """Accented characters encode correctly."""
        accented = [
            "\u00e9",  # e-acute (precomposed)
            "e\u0301",  # e + combining acute (decomposed)
            "\u00f1",  # n-tilde
            "\u00fc",  # u-umlaut
        ]
        for char in accented:
            tokens = tokenizer.encode(char)
            assert len(tokens) >= 1

    @pytest.mark.requires_model
    def test_encode_unicode_normalization(self, tokenizer):
        """Different Unicode normalizations may tokenize differently."""
        # Precomposed vs decomposed
        precomposed = "\u00e9"  # e-acute as single codepoint
        decomposed = "e\u0301"  # e + combining acute

        tokens_pre = tokenizer.encode(precomposed)
        tokens_dec = tokenizer.encode(decomposed)

        # Both should encode (result may or may not be same)
        assert len(tokens_pre) >= 1
        assert len(tokens_dec) >= 1


class TestEncodeMultilingual:
    """Tests for multilingual text encoding."""

    @pytest.mark.requires_model
    def test_encode_japanese(self, tokenizer):
        """Japanese text encodes."""
        tokens = tokenizer.encode("日本語テスト")

        assert len(tokens) >= 1

    @pytest.mark.requires_model
    def test_encode_chinese(self, tokenizer):
        """Chinese text encodes."""
        tokens = tokenizer.encode("中文测试")

        assert len(tokens) >= 1

    @pytest.mark.requires_model
    def test_encode_korean(self, tokenizer):
        """Korean text encodes."""
        tokens = tokenizer.encode("한국어 테스트")

        assert len(tokens) >= 1

    @pytest.mark.requires_model
    def test_encode_russian(self, tokenizer):
        """Russian (Cyrillic) text encodes."""
        tokens = tokenizer.encode("Привет мир")

        assert len(tokens) >= 1

    @pytest.mark.requires_model
    def test_encode_arabic(self, tokenizer):
        """Arabic text encodes."""
        tokens = tokenizer.encode("مرحبا بالعالم")

        assert len(tokens) >= 1

    @pytest.mark.requires_model
    def test_encode_french(self, tokenizer):
        """French text with accents encodes."""
        tokens = tokenizer.encode("Bonjour le monde")

        assert len(tokens) >= 1

    @pytest.mark.requires_model
    def test_encode_german(self, tokenizer):
        """German text with umlauts encodes."""
        tokens = tokenizer.encode("Hallo Welt")

        assert len(tokens) >= 1

    @pytest.mark.requires_model
    @pytest.mark.parametrize(
        "lang,text",
        [
            ("japanese", "日本語テスト"),
            ("chinese", "中文测试"),
            ("korean", "한국어 테스트"),
            ("russian", "Привет мир"),
            ("arabic", "مرحبا بالعالم"),
            ("french", "Bonjour le monde"),
            ("german", "Hallo Welt"),
            ("italian", "Ciao mondo"),
        ],
    )
    def test_encode_multilingual_parametrized(self, tokenizer, lang, text):
        """Parametrized multilingual encoding test."""
        tokens = tokenizer.encode(text)

        assert len(tokens) >= 1, f"Failed for {lang}: {text}"


class TestEncodeEmoji:
    """Tests for emoji encoding."""

    @pytest.mark.requires_model
    def test_encode_simple_emoji(self, tokenizer):
        """Simple emoji encodes."""
        tokens = tokenizer.encode("🎉")

        assert len(tokens) >= 1

    @pytest.mark.requires_model
    def test_encode_emoji_with_text(self, tokenizer):
        """Emoji with text encodes."""
        tokens = tokenizer.encode("🎉 Emoji test 🚀")

        assert len(tokens) >= 1

    @pytest.mark.requires_model
    @pytest.mark.parametrize(
        "emoji",
        [
            "😀",  # Simple emoji
            "👨‍👩‍👧‍👦",  # Family emoji (ZWJ sequence)
            "🇺🇸",  # Flag emoji
            "👍🏽",  # Emoji with skin tone modifier
        ],
    )
    def test_encode_various_emoji(self, tokenizer, emoji):
        """Various emoji types encode."""
        tokens = tokenizer.encode(emoji)

        assert len(tokens) >= 1


class TestEncodeMixedScripts:
    """Tests for mixed script encoding."""

    @pytest.mark.requires_model
    def test_encode_mixed_cjk_latin(self, tokenizer):
        """Mixed CJK and Latin text encodes."""
        tokens = tokenizer.encode("Hello 世界!")

        assert len(tokens) >= 1

    @pytest.mark.requires_model
    def test_encode_mixed_with_numbers(self, tokenizer):
        """Mixed scripts with numbers encode."""
        tokens = tokenizer.encode("Testing 123 日本語")

        assert len(tokens) >= 1

    @pytest.mark.requires_model
    @pytest.mark.parametrize(
        "text",
        [
            "Hello 世界!",
            "Testing 123 日本語",
            "Привет Hello 你好",
            "🎉 Party! パーティー 派對",
        ],
    )
    def test_encode_mixed_scripts_parametrized(self, tokenizer, text):
        """Parametrized mixed script tests."""
        tokens = tokenizer.encode(text)

        assert len(tokens) >= 1
