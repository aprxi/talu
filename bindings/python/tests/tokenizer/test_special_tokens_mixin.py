"""Tests for talu.tokenizer.special_tokens module.

Tests for the SpecialTokensMixin class that provides special token
properties and methods.
"""

from talu.tokenizer.special_tokens import SpecialTokensMixin


class MockTokenizer(SpecialTokensMixin):
    """Mock tokenizer for testing SpecialTokensMixin."""

    def __init__(
        self,
        eos_tokens: tuple[int, ...] = (),
        bos_token_id: int | None = None,
        unk_token_id: int | None = None,
        pad_token_id: int | None = None,
        token_map: dict[int, str] | None = None,
    ):
        self._eos_tokens = eos_tokens
        self._bos_token_id = bos_token_id
        self._unk_token_id = unk_token_id
        self._pad_token_id = pad_token_id
        self._token_map = token_map or {}

    def id_to_token(self, token_id: int) -> str | None:
        """Get token string from ID."""
        return self._token_map.get(token_id)


class TestEosTokenIds:
    """Tests for eos_token_ids property."""

    def test_empty_eos_tokens(self):
        """Returns empty tuple when no EOS tokens."""
        tok = MockTokenizer()
        assert tok.eos_token_ids == ()

    def test_single_eos_token(self):
        """Returns tuple with single EOS token."""
        tok = MockTokenizer(eos_tokens=(100,))
        assert tok.eos_token_ids == (100,)

    def test_multiple_eos_tokens(self):
        """Returns tuple with multiple EOS tokens."""
        tok = MockTokenizer(eos_tokens=(100, 101, 102))
        assert tok.eos_token_ids == (100, 101, 102)

    def test_eos_tokens_immutable(self):
        """EOS tokens tuple is immutable."""
        tok = MockTokenizer(eos_tokens=(100,))
        eos = tok.eos_token_ids
        assert isinstance(eos, tuple)


class TestEosTokens:
    """Tests for eos_tokens property (string representations)."""

    def test_empty_eos_tokens(self):
        """Returns empty list when no EOS tokens."""
        tok = MockTokenizer()
        assert tok.eos_tokens == []

    def test_returns_token_strings(self):
        """Returns list of token strings."""
        tok = MockTokenizer(eos_tokens=(100, 101), token_map={100: "<|eos|>", 101: "<|end|>"})
        assert tok.eos_tokens == ["<|eos|>", "<|end|>"]

    def test_skips_none_tokens(self):
        """Skips tokens that can't be mapped to strings."""
        tok = MockTokenizer(
            eos_tokens=(100, 101, 102),
            token_map={100: "<|eos|>", 102: "<|end|>"},  # 101 missing
        )
        assert tok.eos_tokens == ["<|eos|>", "<|end|>"]

    def test_all_tokens_unmapped(self):
        """Returns empty list when all tokens unmapped."""
        tok = MockTokenizer(eos_tokens=(100, 101))
        assert tok.eos_tokens == []


class TestBosToken:
    """Tests for bos_token_id and bos_token properties."""

    def test_bos_token_id_none(self):
        """Returns None when no BOS token."""
        tok = MockTokenizer()
        assert tok.bos_token_id is None

    def test_bos_token_id_present(self):
        """Returns BOS token ID when set."""
        tok = MockTokenizer(bos_token_id=200)
        assert tok.bos_token_id == 200

    def test_bos_token_none_when_id_none(self):
        """Returns None for bos_token when bos_token_id is None."""
        tok = MockTokenizer()
        assert tok.bos_token is None

    def test_bos_token_returns_string(self):
        """Returns BOS token string when available."""
        tok = MockTokenizer(bos_token_id=200, token_map={200: "<|bos|>"})
        assert tok.bos_token == "<|bos|>"

    def test_bos_token_none_when_not_in_map(self):
        """Returns None for bos_token when not in token map."""
        tok = MockTokenizer(bos_token_id=200)
        assert tok.bos_token is None


class TestUnkToken:
    """Tests for unk_token_id and unk_token properties."""

    def test_unk_token_id_none(self):
        """Returns None when no UNK token."""
        tok = MockTokenizer()
        assert tok.unk_token_id is None

    def test_unk_token_id_present(self):
        """Returns UNK token ID when set."""
        tok = MockTokenizer(unk_token_id=300)
        assert tok.unk_token_id == 300

    def test_unk_token_none_when_id_none(self):
        """Returns None for unk_token when unk_token_id is None."""
        tok = MockTokenizer()
        assert tok.unk_token is None

    def test_unk_token_returns_string(self):
        """Returns UNK token string when available."""
        tok = MockTokenizer(unk_token_id=300, token_map={300: "<|unk|>"})
        assert tok.unk_token == "<|unk|>"


class TestPadToken:
    """Tests for pad_token_id and pad_token properties."""

    def test_pad_token_id_none(self):
        """Returns None when no PAD token."""
        tok = MockTokenizer()
        assert tok.pad_token_id is None

    def test_pad_token_id_present(self):
        """Returns PAD token ID when set."""
        tok = MockTokenizer(pad_token_id=400)
        assert tok.pad_token_id == 400

    def test_pad_token_none_when_id_none(self):
        """Returns None for pad_token when pad_token_id is None."""
        tok = MockTokenizer()
        assert tok.pad_token is None

    def test_pad_token_returns_string(self):
        """Returns PAD token string when available."""
        tok = MockTokenizer(pad_token_id=400, token_map={400: "<|pad|>"})
        assert tok.pad_token == "<|pad|>"


class TestSpecialIds:
    """Tests for special_ids property."""

    def test_empty_when_no_special_tokens(self):
        """Returns empty frozenset when no special tokens."""
        tok = MockTokenizer()
        assert tok.special_ids == frozenset()

    def test_includes_eos_tokens(self):
        """Includes EOS tokens in special_ids."""
        tok = MockTokenizer(eos_tokens=(100, 101))
        assert 100 in tok.special_ids
        assert 101 in tok.special_ids

    def test_includes_bos_token(self):
        """Includes BOS token in special_ids."""
        tok = MockTokenizer(bos_token_id=200)
        assert 200 in tok.special_ids

    def test_includes_unk_token(self):
        """Includes UNK token in special_ids."""
        tok = MockTokenizer(unk_token_id=300)
        assert 300 in tok.special_ids

    def test_includes_pad_token(self):
        """Includes PAD token in special_ids."""
        tok = MockTokenizer(pad_token_id=400)
        assert 400 in tok.special_ids

    def test_all_special_tokens(self):
        """Includes all special tokens."""
        tok = MockTokenizer(
            eos_tokens=(100, 101), bos_token_id=200, unk_token_id=300, pad_token_id=400
        )
        expected = frozenset([100, 101, 200, 300, 400])
        assert tok.special_ids == expected

    def test_returns_frozenset(self):
        """Returns frozenset (immutable)."""
        tok = MockTokenizer(eos_tokens=(100,))
        assert isinstance(tok.special_ids, frozenset)


class TestIsSpecialId:
    """Tests for is_special_id method."""

    def test_regular_token_not_special(self):
        """Regular token returns False."""
        tok = MockTokenizer(eos_tokens=(100,))
        assert tok.is_special_id(999) is False

    def test_eos_token_is_special(self):
        """EOS token returns True."""
        tok = MockTokenizer(eos_tokens=(100, 101))
        assert tok.is_special_id(100) is True
        assert tok.is_special_id(101) is True

    def test_bos_token_is_special(self):
        """BOS token returns True."""
        tok = MockTokenizer(bos_token_id=200)
        assert tok.is_special_id(200) is True

    def test_unk_token_is_special(self):
        """UNK token returns True."""
        tok = MockTokenizer(unk_token_id=300)
        assert tok.is_special_id(300) is True

    def test_pad_token_is_special(self):
        """PAD token returns True."""
        tok = MockTokenizer(pad_token_id=400)
        assert tok.is_special_id(400) is True

    def test_no_special_tokens(self):
        """Returns False when no special tokens defined."""
        tok = MockTokenizer()
        assert tok.is_special_id(100) is False


class TestPrimaryEosTokenId:
    """Tests for primary_eos_token_id method."""

    def test_returns_none_when_no_eos_tokens(self):
        """Returns None when no EOS tokens."""
        tok = MockTokenizer()
        assert tok.primary_eos_token_id() is None

    def test_returns_first_eos_token(self):
        """Returns first EOS token when available."""
        tok = MockTokenizer(eos_tokens=(100, 101, 102))
        assert tok.primary_eos_token_id() == 100

    def test_single_eos_token(self):
        """Returns single EOS token when only one."""
        tok = MockTokenizer(eos_tokens=(100,))
        assert tok.primary_eos_token_id() == 100
