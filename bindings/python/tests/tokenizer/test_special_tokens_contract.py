"""
Special Tokens Contract Tests.

Tests the special token API contract:
- eos_token_ids: immutable tuple, ordered, deduplicated
- bos_token_id: singular int or None
- is_special_id(): single source of truth for special token detection
- primary_eos_token_id(): returns eos_token_ids[0] for insertion
- special_ids: frozenset of all special token IDs
- decode skip_special_tokens: uses ID lookup, not stored masks
"""

import pytest

# =============================================================================
# Test A: eos_token_ids is Immutable Tuple
# =============================================================================


class TestEosTokenIdsImmutable:
    """Test that eos_token_ids returns an immutable tuple."""

    @pytest.mark.requires_model
    def test_eos_token_ids_is_tuple(self, tokenizer):
        """eos_token_ids returns a tuple, not a list."""
        eos = tokenizer.eos_token_ids
        assert isinstance(eos, tuple), f"Expected tuple, got {type(eos).__name__}"

    @pytest.mark.requires_model
    def test_eos_token_ids_immutable(self, tokenizer):
        """eos_token_ids tuple cannot be modified."""
        eos = tokenizer.eos_token_ids
        if len(eos) > 0:
            with pytest.raises(TypeError):
                eos[0] = 0  # type: ignore[index]

    @pytest.mark.requires_model
    def test_eos_token_ids_contains_integers(self, tokenizer):
        """All eos_token_ids elements are integers."""
        for eos_id in tokenizer.eos_token_ids:
            assert isinstance(eos_id, int), f"Expected int, got {type(eos_id).__name__}"

    @pytest.mark.requires_model
    def test_eos_token_ids_deduplicated(self, tokenizer):
        """eos_token_ids contains no duplicates."""
        eos = tokenizer.eos_token_ids
        assert len(eos) == len(set(eos)), "eos_token_ids contains duplicates"


# =============================================================================
# Test B: bos_token_id is Singular
# =============================================================================


class TestBosTokenIdSingular:
    """Test that bos_token_id is singular int or None."""

    @pytest.mark.requires_model
    def test_bos_token_id_type(self, tokenizer):
        """bos_token_id is int or None."""
        bos = tokenizer.bos_token_id
        assert bos is None or isinstance(bos, int), f"Expected int|None, got {type(bos).__name__}"

    @pytest.mark.requires_model
    def test_bos_token_not_list(self, tokenizer):
        """bos_token_id is not a list or tuple."""
        bos = tokenizer.bos_token_id
        assert not isinstance(bos, (list, tuple)), "bos_token_id should be singular, not a sequence"


# =============================================================================
# Test C: is_special_id() is Single Source of Truth
# =============================================================================


class TestIsSpecialId:
    """Test is_special_id() as single source of truth."""

    @pytest.mark.requires_model
    def test_is_special_id_exists(self, tokenizer):
        """Tokenizer has is_special_id() method."""
        assert hasattr(tokenizer, "is_special_id")
        assert callable(tokenizer.is_special_id)

    @pytest.mark.requires_model
    def test_is_special_id_returns_bool(self, tokenizer):
        """is_special_id() returns boolean."""
        # Test with a valid token ID (0 is usually valid)
        result = tokenizer.is_special_id(0)
        assert isinstance(result, bool)

    @pytest.mark.requires_model
    def test_is_special_id_detects_eos(self, tokenizer):
        """is_special_id() returns True for all EOS tokens."""
        for eos_id in tokenizer.eos_token_ids:
            assert tokenizer.is_special_id(eos_id), (
                f"is_special_id({eos_id}) should be True for EOS"
            )

    @pytest.mark.requires_model
    def test_is_special_id_detects_bos(self, tokenizer):
        """is_special_id() returns True for BOS token."""
        bos = tokenizer.bos_token_id
        if bos is not None:
            assert tokenizer.is_special_id(bos), f"is_special_id({bos}) should be True for BOS"

    @pytest.mark.requires_model
    def test_is_special_id_detects_unk(self, tokenizer):
        """is_special_id() returns True for UNK token."""
        unk = tokenizer.unk_token_id
        if unk is not None:
            assert tokenizer.is_special_id(unk), f"is_special_id({unk}) should be True for UNK"

    @pytest.mark.requires_model
    def test_is_special_id_detects_pad(self, tokenizer):
        """is_special_id() returns True for PAD token."""
        pad = tokenizer.pad_token_id
        if pad is not None:
            assert tokenizer.is_special_id(pad), f"is_special_id({pad}) should be True for PAD"

    @pytest.mark.requires_model
    def test_is_special_id_regular_token(self, tokenizer):
        """is_special_id() returns False for regular tokens."""
        # Encode a simple word without special tokens
        tokens = tokenizer.encode("hello", special_tokens=False)
        if len(tokens) > 0:
            # At least one token from "hello" should not be special
            regular_found = False
            for tok_id in tokens.tolist():
                if not tokenizer.is_special_id(tok_id):
                    regular_found = True
                    break
            assert regular_found, "Expected at least one non-special token in 'hello'"


# =============================================================================
# Test D: primary_eos_token_id() Returns First EOS
# =============================================================================


class TestPrimaryEosTokenId:
    """Test primary_eos_token_id() returns first EOS for insertion."""

    @pytest.mark.requires_model
    def test_primary_eos_token_id_exists(self, tokenizer):
        """Tokenizer has primary_eos_token_id() method."""
        assert hasattr(tokenizer, "primary_eos_token_id")
        assert callable(tokenizer.primary_eos_token_id)

    @pytest.mark.requires_model
    def test_primary_eos_is_first(self, tokenizer):
        """primary_eos_token_id() returns eos_token_ids[0]."""
        eos_ids = tokenizer.eos_token_ids
        primary = tokenizer.primary_eos_token_id()
        if len(eos_ids) > 0:
            assert primary == eos_ids[0], "primary_eos_token_id() should be first EOS"
        else:
            assert primary is None, "primary_eos_token_id() should be None when no EOS"

    @pytest.mark.requires_model
    def test_primary_eos_returns_int_or_none(self, tokenizer):
        """primary_eos_token_id() returns int or None."""
        primary = tokenizer.primary_eos_token_id()
        assert primary is None or isinstance(primary, int)


# =============================================================================
# Test E: special_ids Property
# =============================================================================


class TestSpecialIds:
    """Test special_ids property returns frozenset of all special IDs."""

    @pytest.mark.requires_model
    def test_special_ids_exists(self, tokenizer):
        """Tokenizer has special_ids property."""
        assert hasattr(tokenizer, "special_ids")

    @pytest.mark.requires_model
    def test_special_ids_is_frozenset(self, tokenizer):
        """special_ids returns a frozenset."""
        ids = tokenizer.special_ids
        assert isinstance(ids, frozenset), f"Expected frozenset, got {type(ids).__name__}"

    @pytest.mark.requires_model
    def test_special_ids_contains_eos(self, tokenizer):
        """special_ids contains all EOS tokens."""
        ids = tokenizer.special_ids
        for eos_id in tokenizer.eos_token_ids:
            assert eos_id in ids, f"EOS {eos_id} should be in special_ids"

    @pytest.mark.requires_model
    def test_special_ids_contains_bos(self, tokenizer):
        """special_ids contains BOS token."""
        bos = tokenizer.bos_token_id
        if bos is not None:
            assert bos in tokenizer.special_ids, f"BOS {bos} should be in special_ids"

    @pytest.mark.requires_model
    def test_special_ids_contains_unk(self, tokenizer):
        """special_ids contains UNK token."""
        unk = tokenizer.unk_token_id
        if unk is not None:
            assert unk in tokenizer.special_ids, f"UNK {unk} should be in special_ids"

    @pytest.mark.requires_model
    def test_special_ids_contains_pad(self, tokenizer):
        """special_ids contains PAD token."""
        pad = tokenizer.pad_token_id
        if pad is not None:
            assert pad in tokenizer.special_ids, f"PAD {pad} should be in special_ids"

    @pytest.mark.requires_model
    def test_special_ids_consistent_with_is_special_id(self, tokenizer):
        """special_ids membership matches is_special_id()."""
        ids = tokenizer.special_ids
        for token_id in ids:
            assert tokenizer.is_special_id(token_id), (
                f"Token {token_id} in special_ids but is_special_id() returns False"
            )


# =============================================================================
# Test F: Decode Uses ID Lookup (Not Stored Masks)
# =============================================================================


class TestDecodeSkipSpecialTokens:
    """Test that decode skip_special_tokens uses ID lookup."""

    @pytest.mark.requires_model
    def test_decode_skip_special_removes_bos(self, tokenizer):
        """Decode with skip_special_tokens=True removes BOS."""
        bos = tokenizer.bos_token_id
        if bos is None:
            pytest.skip("Model has no BOS token")

        # Create tokens with BOS prepended
        text = "Hello"
        tokens = tokenizer.encode(text, special_tokens=False)
        tokens_with_bos = [bos] + tokens.tolist()

        # Decode should skip BOS
        decoded = tokenizer.decode(tokens_with_bos, skip_special_tokens=True)
        bos_str = tokenizer.bos_token
        if bos_str:
            assert bos_str not in decoded, f"BOS '{bos_str}' should be skipped"

    @pytest.mark.requires_model
    def test_decode_skip_special_removes_eos(self, tokenizer):
        """Decode with skip_special_tokens=True removes EOS."""
        eos_ids = tokenizer.eos_token_ids
        if not eos_ids:
            pytest.skip("Model has no EOS tokens")

        # Create tokens with EOS appended
        text = "Hello"
        tokens = tokenizer.encode(text, special_tokens=False)
        tokens_with_eos = tokens.tolist() + [eos_ids[0]]

        # Decode should skip EOS
        decoded = tokenizer.decode(tokens_with_eos, skip_special_tokens=True)
        eos_str = tokenizer.id_to_token(eos_ids[0])
        if eos_str:
            assert eos_str not in decoded, f"EOS '{eos_str}' should be skipped"

    @pytest.mark.requires_model
    def test_decode_keep_special_includes_tokens(self, tokenizer):
        """Decode with skip_special_tokens=False includes special tokens."""
        eos_ids = tokenizer.eos_token_ids
        if not eos_ids:
            pytest.skip("Model has no EOS tokens")

        # Create tokens with EOS appended
        text = "Hello"
        tokens = tokenizer.encode(text, special_tokens=False)
        tokens_with_eos = tokens.tolist() + [eos_ids[0]]

        # Decode should include EOS
        decoded = tokenizer.decode(tokens_with_eos, skip_special_tokens=False)
        eos_str = tokenizer.id_to_token(eos_ids[0])
        if eos_str:
            assert eos_str in decoded, f"EOS '{eos_str}' should be included"

    @pytest.mark.requires_model
    def test_decode_skip_manually_inserted_special(self, tokenizer):
        """Decode skips special tokens even when manually inserted."""
        # This tests that detection is by ID, not by encoding history
        eos_ids = tokenizer.eos_token_ids
        if not eos_ids:
            pytest.skip("Model has no EOS tokens")

        # Manually create token list with EOS in the middle
        tokens = tokenizer.encode("Hello", special_tokens=False)
        token_list = tokens.tolist()
        if len(token_list) >= 2:
            # Insert EOS in the middle
            middle = len(token_list) // 2
            token_list.insert(middle, eos_ids[0])

            decoded = tokenizer.decode(token_list, skip_special_tokens=True)
            eos_str = tokenizer.id_to_token(eos_ids[0])
            if eos_str:
                assert eos_str not in decoded, (
                    f"Manually inserted EOS '{eos_str}' should still be skipped"
                )


# =============================================================================
# Test G: Consistency Between Properties and Methods
# =============================================================================


class TestSpecialTokenConsistency:
    """Test consistency between various special token APIs."""

    @pytest.mark.requires_model
    def test_eos_tokens_subset_of_ids(self, tokenizer):
        """eos_tokens list length <= eos_token_ids (some may not decode)."""
        ids = tokenizer.eos_token_ids
        tokens = tokenizer.eos_tokens
        # eos_tokens filters out IDs that can't be decoded to strings
        assert len(tokens) <= len(ids), (
            f"eos_tokens ({len(tokens)}) should not exceed eos_token_ids ({len(ids)})"
        )

    @pytest.mark.requires_model
    def test_bos_token_matches_id(self, tokenizer):
        """bos_token is the string for bos_token_id."""
        bos_id = tokenizer.bos_token_id
        bos_str = tokenizer.bos_token

        if bos_id is None:
            assert bos_str is None, "bos_token should be None when bos_token_id is None"
        else:
            # bos_token_id may reference a token outside the vocab (e.g., tiny
            # models with truncated vocabs). In that case both bos_token and
            # id_to_token return None, which is consistent.
            resolved = tokenizer.id_to_token(bos_id)
            assert bos_str == resolved

    @pytest.mark.requires_model
    def test_pad_token_matches_id(self, tokenizer):
        """pad_token is the string for pad_token_id."""
        pad_id = tokenizer.pad_token_id
        pad_str = tokenizer.pad_token

        if pad_id is None:
            assert pad_str is None, "pad_token should be None when pad_token_id is None"
        else:
            # pad_token may be None if id_to_token() returns None (some models)
            expected = tokenizer.id_to_token(pad_id)
            assert pad_str == expected, f"pad_token should match id_to_token({pad_id})"

    @pytest.mark.requires_model
    def test_unk_token_matches_id(self, tokenizer):
        """unk_token is the string for unk_token_id."""
        unk_id = tokenizer.unk_token_id
        unk_str = tokenizer.unk_token

        if unk_id is None:
            assert unk_str is None, "unk_token should be None when unk_token_id is None"
        else:
            assert unk_str is not None, "unk_token should not be None when unk_token_id exists"
            assert unk_str == tokenizer.id_to_token(unk_id)
