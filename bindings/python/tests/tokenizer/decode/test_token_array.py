"""
TokenArray interface tests.

Tests for the TokenArray class returned by Tokenizer.encode().
"""

import pytest


class TestTokenArrayBasic:
    """Basic TokenArray functionality tests."""

    @pytest.mark.requires_model
    def test_token_array_type(self, tokenizer, talu):
        """encode() returns TokenArray type."""
        tokens = tokenizer.encode("Hello")

        # Should be TokenArray (or subclass)
        assert type(tokens).__name__ == "TokenArray"

    @pytest.mark.requires_model
    def test_token_array_length(self, tokenizer):
        """TokenArray reports correct length."""
        text = "Hello World"
        tokens = tokenizer.encode(text)

        assert len(tokens) > 0
        assert len(tokens) == tokens._num_tokens

    @pytest.mark.requires_model
    def test_token_array_bool(self, tokenizer):
        """TokenArray is truthy when non-empty."""
        tokens = tokenizer.encode("Hello")
        assert bool(tokens) == (len(tokens) > 0)


class TestTokenArrayIndexing:
    """Tests for TokenArray indexing."""

    @pytest.mark.requires_model
    def test_positive_indexing(self, tokenizer):
        """TokenArray supports positive indexing."""
        tokens = tokenizer.encode("Hello World")

        first = tokens[0]
        assert isinstance(first, int)

        if len(tokens) > 1:
            second = tokens[1]
            assert isinstance(second, int)

    @pytest.mark.requires_model
    def test_negative_indexing(self, tokenizer):
        """TokenArray supports negative indexing."""
        tokens = tokenizer.encode("Hello World")

        last = tokens[-1]
        assert isinstance(last, int)

        if len(tokens) > 1:
            second_last = tokens[-2]
            assert isinstance(second_last, int)

    @pytest.mark.requires_model
    def test_index_out_of_bounds(self, tokenizer):
        """Out of bounds index raises IndexError."""
        tokens = tokenizer.encode("Hello")

        with pytest.raises(IndexError):
            _ = tokens[len(tokens)]

        with pytest.raises(IndexError):
            _ = tokens[-len(tokens) - 1]

    @pytest.mark.requires_model
    def test_index_consistency(self, tokenizer):
        """Indexing is consistent with tolist()."""
        tokens = tokenizer.encode("Hello World Test")
        token_list = tokens.tolist()

        for i in range(len(tokens)):
            assert tokens[i] == token_list[i]


class TestTokenArrayToList:
    """Tests for TokenArray.tolist()."""

    @pytest.mark.requires_model
    def test_tolist_returns_list(self, tokenizer):
        """tolist() returns Python list."""
        tokens = tokenizer.encode("Hello World")
        token_list = tokens.tolist()

        assert isinstance(token_list, list)

    @pytest.mark.requires_model
    def test_tolist_correct_length(self, tokenizer):
        """tolist() returns correct length."""
        tokens = tokenizer.encode("Hello World")
        token_list = tokens.tolist()

        assert len(token_list) == len(tokens)

    @pytest.mark.requires_model
    def test_tolist_all_ints(self, tokenizer):
        """tolist() returns all integers."""
        tokens = tokenizer.encode("Hello World")
        token_list = tokens.tolist()

        assert all(isinstance(t, int) for t in token_list)

    @pytest.mark.requires_model
    def test_tolist_values_match_indexing(self, tokenizer):
        """tolist() values match individual indexing."""
        tokens = tokenizer.encode("Hello World")
        token_list = tokens.tolist()

        for i, t in enumerate(token_list):
            assert tokens[i] == t


class TestTokenArrayIteration:
    """Tests for TokenArray iteration."""

    @pytest.mark.requires_model
    def test_iteration(self, tokenizer):
        """TokenArray is iterable."""
        tokens = tokenizer.encode("Hello World")

        iterated = list(tokens)
        assert len(iterated) == len(tokens)

    @pytest.mark.requires_model
    def test_iteration_matches_tolist(self, tokenizer):
        """Iteration matches tolist()."""
        tokens = tokenizer.encode("Hello World")

        iterated = list(tokens)
        as_list = tokens.tolist()

        assert iterated == as_list


class TestTokenArrayRepr:
    """Tests for TokenArray string representation."""

    @pytest.mark.requires_model
    def test_repr(self, tokenizer):
        """TokenArray has informative repr."""
        tokens = tokenizer.encode("Hello")
        repr_str = repr(tokens)

        assert "TokenArray" in repr_str
        assert "len=" in repr_str

    @pytest.mark.requires_model
    def test_str(self, tokenizer):
        """TokenArray has string representation."""
        tokens = tokenizer.encode("Hello")
        str_repr = str(tokens)

        # Should have some useful representation
        assert len(str_repr) > 0


class TestTokenArrayMemory:
    """Tests for TokenArray memory behavior."""

    @pytest.mark.requires_model
    def test_token_array_persists(self, tokenizer):
        """TokenArray data persists after encode."""
        tokens = tokenizer.encode("Hello World")

        # Access data multiple times
        for _ in range(10):
            _ = tokens[0]
            _ = tokens.tolist()

        # Should still work
        assert len(tokens) > 0

    @pytest.mark.requires_model
    def test_multiple_token_arrays(self, tokenizer):
        """Multiple TokenArrays can coexist."""
        tokens1 = tokenizer.encode("Hello")
        tokens2 = tokenizer.encode("World")
        tokens3 = tokenizer.encode("Test")

        # All should be valid
        assert len(tokens1) > 0
        assert len(tokens2) > 0
        assert len(tokens3) > 0

        # Values should be independent
        list1 = tokens1.tolist()
        list2 = tokens2.tolist()
        list3 = tokens3.tolist()

        assert list1 != list2 or list2 != list3


class TestTokenArrayConcatenation:
    """Tests for TokenArray concatenation (__add__, __radd__)."""

    @pytest.mark.requires_model
    def test_add_token_arrays(self, tokenizer):
        """TokenArray + TokenArray returns TokenArray."""
        tokens1 = tokenizer.encode("Hello")
        tokens2 = tokenizer.encode("World")

        combined = tokens1 + tokens2
        assert type(combined).__name__ == "TokenArray"
        assert len(combined) == len(tokens1) + len(tokens2)

    @pytest.mark.requires_model
    def test_add_token_array_list(self, tokenizer):
        """TokenArray + list returns TokenArray."""
        tokens = tokenizer.encode("Hello")

        combined = tokens + [1, 2, 3]
        assert type(combined).__name__ == "TokenArray"
        assert len(combined) == len(tokens) + 3

    @pytest.mark.requires_model
    def test_radd_list_token_array(self, tokenizer):
        """list + TokenArray returns TokenArray."""
        tokens = tokenizer.encode("World")

        combined = [100, 200] + tokens
        assert type(combined).__name__ == "TokenArray"
        assert len(combined) == 2 + len(tokens)

    @pytest.mark.requires_model
    def test_add_preserves_values(self, tokenizer):
        """Concatenation preserves token values."""
        tokens1 = tokenizer.encode("Hello")
        tokens2 = tokenizer.encode("World")

        combined = tokens1 + tokens2
        expected = tokens1.tolist() + tokens2.tolist()
        assert combined.tolist() == expected

    @pytest.mark.requires_model
    def test_add_with_list_preserves_values(self, tokenizer):
        """TokenArray + list preserves values."""
        tokens = tokenizer.encode("Hello")
        extra = [1, 2, 3]

        combined = tokens + extra
        expected = tokens.tolist() + extra
        assert combined.tolist() == expected

    @pytest.mark.requires_model
    def test_radd_preserves_values(self, tokenizer):
        """list + TokenArray preserves values."""
        tokens = tokenizer.encode("World")
        prefix = [100, 200]

        combined = prefix + tokens
        expected = prefix + tokens.tolist()
        assert combined.tolist() == expected

    @pytest.mark.requires_model
    def test_add_empty_list(self, tokenizer):
        """TokenArray + [] returns copy."""
        tokens = tokenizer.encode("Hello")

        combined = tokens + []
        assert combined.tolist() == tokens.tolist()
        assert combined is not tokens  # New array

    @pytest.mark.requires_model
    def test_radd_empty_list(self, tokenizer):
        """[] + TokenArray returns copy."""
        tokens = tokenizer.encode("Hello")

        combined = [] + tokens
        assert combined.tolist() == tokens.tolist()
        assert combined is not tokens  # New array

    @pytest.mark.requires_model
    def test_chained_add(self, tokenizer):
        """Multiple concatenations work."""
        t1 = tokenizer.encode("a")
        t2 = tokenizer.encode("b")
        t3 = tokenizer.encode("c")

        combined = t1 + t2 + t3
        expected = t1.tolist() + t2.tolist() + t3.tolist()
        assert combined.tolist() == expected

    @pytest.mark.requires_model
    def test_add_invalid_type(self, tokenizer):
        """TokenArray + invalid type returns NotImplemented."""
        tokens = tokenizer.encode("Hello")

        # Should raise TypeError (not crash)
        with pytest.raises(TypeError):
            _ = tokens + "string"

        with pytest.raises(TypeError):
            _ = tokens + 42


class TestTokenArrayLifecycle:
    """Tests for TokenArray resource lifecycle (alloc->free cycle).

    These tests verify that TokenArray properly manages native memory
    and handles edge cases like GC, multiple references, and scope exit.
    """

    @pytest.mark.requires_model
    def test_gc_reclaims_token_array(self, tokenizer):
        """TokenArray is properly collected by garbage collector.

        Contract: After del and gc.collect(), the native memory should be freed.
        We can't directly verify memory is freed, but we verify no crash occurs.
        """
        import gc

        # Create token arrays in a loop
        for _ in range(100):
            tokens = tokenizer.encode("Hello World Test String " * 10)
            _ = tokens.tolist()  # Access data
            del tokens

        gc.collect()
        # If we get here without crash, lifecycle is working

    @pytest.mark.requires_model
    def test_token_array_outlives_tokenizer_reference(self, talu, tokenizer):
        """TokenArray remains valid even if tokenizer reference is dropped.

        TokenArray holds its own reference to ensure data validity.
        """
        import gc

        tokens = tokenizer.encode("Hello World")
        first_token = tokens[0]  # Capture value before dropping tokenizer

        # Remove our reference to tokenizer
        del tokenizer
        gc.collect()

        # TokenArray should still be valid
        assert tokens[0] == first_token
        assert len(tokens.tolist()) > 0

    @pytest.mark.requires_model
    def test_token_array_scope_exit(self, tokenizer):
        """TokenArray is properly cleaned up when scope exits.

        Contract: TokenArray created inside a function should be cleanly
        freed when the function returns.
        """
        import gc

        def create_and_discard():
            tokens = tokenizer.encode("Temporary data " * 100)
            return tokens.tolist()

        # Create and discard many times
        for _ in range(50):
            result = create_and_discard()
            assert len(result) > 0

        gc.collect()
        # No crash = success


class TestTokenArrayListCompatibility:
    """Tests verifying TokenArray behaves identically to a Python list.

    This test suite ensures TokenArray implements the Sequence Protocol perfectly,
    so users don't realize it isn't a list until they check the type.
    """

    @pytest.mark.requires_model
    def test_slicing_basic(self, tokenizer):
        """tokens[:5] returns a TokenArray slice."""
        tokens = tokenizer.encode("Hello world test string here")
        token_list = tokens.tolist()

        # Basic slicing
        sliced = tokens[:3]
        assert type(sliced).__name__ == "TokenArray"
        assert sliced.tolist() == token_list[:3]

    @pytest.mark.requires_model
    def test_slicing_with_start_stop(self, tokenizer):
        """tokens[1:3] works correctly."""
        tokens = tokenizer.encode("Hello world test")
        token_list = tokens.tolist()

        if len(tokens) >= 3:
            sliced = tokens[1:3]
            assert sliced.tolist() == token_list[1:3]

    @pytest.mark.requires_model
    def test_slicing_negative_indices(self, tokenizer):
        """tokens[-3:] works correctly."""
        tokens = tokenizer.encode("Hello world test string")
        token_list = tokens.tolist()

        sliced = tokens[-3:]
        assert sliced.tolist() == token_list[-3:]

    @pytest.mark.requires_model
    def test_slicing_with_step(self, tokenizer):
        """tokens[::2] works correctly."""
        tokens = tokenizer.encode("Hello world test string")
        token_list = tokens.tolist()

        sliced = tokens[::2]
        assert sliced.tolist() == token_list[::2]

    @pytest.mark.requires_model
    def test_slicing_empty_result(self, tokenizer):
        """tokens[5:5] returns empty TokenArray."""
        tokens = tokenizer.encode("Hello world")

        sliced = tokens[5:5]
        assert len(sliced) == 0
        assert sliced.tolist() == []

    @pytest.mark.requires_model
    def test_containment_present(self, tokenizer):
        """987 in tokens works when token is present."""
        tokens = tokenizer.encode("Hello world")
        token_list = tokens.tolist()

        for token_id in token_list:
            assert token_id in tokens

    @pytest.mark.requires_model
    def test_containment_absent(self, tokenizer):
        """999999 in tokens returns False for absent token."""
        tokens = tokenizer.encode("Hello")
        assert 999999999 not in tokens

    @pytest.mark.requires_model
    def test_equality_with_list(self, tokenizer):
        """tokens == [1, 2, 3] works correctly."""
        tokens = tokenizer.encode("Hello")
        token_list = tokens.tolist()

        assert tokens == token_list
        assert token_list == tokens.tolist()

    @pytest.mark.requires_model
    def test_equality_different_length(self, tokenizer):
        """tokens == [1, 2] returns False for different lengths."""
        tokens = tokenizer.encode("Hello world")

        assert tokens != [1, 2]
        assert tokens != []

    @pytest.mark.requires_model
    def test_multiplication(self, tokenizer):
        """tokens * 2 repeats the sequence."""
        tokens = tokenizer.encode("Hi", special_tokens=False)
        token_list = tokens.tolist()

        repeated = tokens * 2
        assert type(repeated).__name__ == "TokenArray"
        assert repeated.tolist() == token_list * 2

    @pytest.mark.requires_model
    def test_right_multiplication(self, tokenizer):
        """2 * tokens repeats the sequence."""
        tokens = tokenizer.encode("Hi", special_tokens=False)
        token_list = tokens.tolist()

        repeated = 2 * tokens
        assert type(repeated).__name__ == "TokenArray"
        assert repeated.tolist() == token_list * 2

    @pytest.mark.requires_model
    def test_multiplication_by_zero(self, tokenizer):
        """tokens * 0 returns empty TokenArray."""
        tokens = tokenizer.encode("Hello")

        empty = tokens * 0
        assert len(empty) == 0

    @pytest.mark.requires_model
    def test_index_found(self, tokenizer):
        """tokens.index(value) returns correct index."""
        tokens = tokenizer.encode("Hello world")
        token_list = tokens.tolist()

        if len(token_list) > 0:
            first_token = token_list[0]
            assert tokens.index(first_token) == 0

    @pytest.mark.requires_model
    def test_index_not_found(self, tokenizer):
        """tokens.index(absent) raises ValueError."""
        tokens = tokenizer.encode("Hello")

        with pytest.raises(ValueError):
            tokens.index(999999999)

    @pytest.mark.requires_model
    def test_index_with_start_stop(self, tokenizer):
        """tokens.index(value, start, stop) works correctly."""
        tokens = tokenizer.encode("Hello world test")
        token_list = tokens.tolist()

        if len(token_list) >= 3:
            # Find in a range
            target = token_list[1]
            assert tokens.index(target, 0, 3) == token_list.index(target, 0, 3)

    @pytest.mark.requires_model
    def test_count(self, tokenizer):
        """tokens.count(value) returns correct count."""
        tokens = tokenizer.encode("Hello")
        token_list = tokens.tolist()

        if len(token_list) > 0:
            first_token = token_list[0]
            assert tokens.count(first_token) == token_list.count(first_token)

    @pytest.mark.requires_model
    def test_count_absent(self, tokenizer):
        """tokens.count(absent) returns 0."""
        tokens = tokenizer.encode("Hello")

        assert tokens.count(999999999) == 0

    @pytest.mark.requires_model
    def test_concatenation_with_list(self, tokenizer):
        """tokens + [1, 2] works correctly."""
        tokens = tokenizer.encode("Hello")
        token_list = tokens.tolist()

        combined = tokens + [100, 200]
        assert combined.tolist() == token_list + [100, 200]

    @pytest.mark.requires_model
    def test_left_concatenation_with_list(self, tokenizer):
        """[1, 2] + tokens works correctly."""
        tokens = tokenizer.encode("Hello")
        token_list = tokens.tolist()

        combined = [100, 200] + tokens
        assert combined.tolist() == [100, 200] + token_list

    @pytest.mark.requires_model
    def test_iteration_matches_list(self, tokenizer):
        """Iteration yields same values as list."""
        tokens = tokenizer.encode("Hello world")
        token_list = tokens.tolist()

        iterated = list(tokens)
        assert iterated == token_list

    @pytest.mark.requires_model
    def test_len_matches_list(self, tokenizer):
        """len(tokens) matches len(tolist())."""
        tokens = tokenizer.encode("Hello world")

        assert len(tokens) == len(tokens.tolist())

    @pytest.mark.requires_model
    def test_bool_matches_list(self, tokenizer):
        """bool(tokens) matches bool(tolist())."""
        tokens = tokenizer.encode("Hello")

        assert bool(tokens) == bool(tokens.tolist())

    @pytest.mark.requires_model
    def test_reversed_iteration(self, tokenizer):
        """reversed(tokens) works correctly."""
        tokens = tokenizer.encode("Hello world")
        token_list = tokens.tolist()

        # TokenArray should support reversed via __iter__ + __len__ + __getitem__
        rev = list(reversed(tokens))
        assert rev == list(reversed(token_list))


class TestTokenArrayDLPack:
    """Tests for DLPack protocol (__dlpack__, __dlpack_device__).

    Pure Python tests only - torch/numpy validation tests are in tests/reference/.

    TokenArray uses refcounted SharedBuffer semantics:
    - Multiple exports are allowed (each increases refcount)
    - TokenArray remains valid after export (shared, not moved)
    - All exported tensors and the TokenArray share the same memory
    - Memory is freed when the last reference is released
    """

    @pytest.mark.requires_model
    def test_has_dlpack_protocol(self, tokenizer):
        """TokenArray implements DLPack protocol."""
        tokens = tokenizer.encode("Hello")
        assert hasattr(tokens, "__dlpack__")
        assert hasattr(tokens, "__dlpack_device__")

    @pytest.mark.requires_model
    def test_dlpack_device_returns_cpu(self, tokenizer):
        """__dlpack_device__ returns CPU device tuple."""
        tokens = tokenizer.encode("Hello")
        device = tokens.__dlpack_device__()
        assert device == (1, 0)  # kDLCPU = 1, device_id = 0

    @pytest.mark.requires_model
    def test_dlpack_returns_capsule(self, tokenizer):
        """__dlpack__() returns a PyCapsule."""
        tokens = tokenizer.encode("Hello world", special_tokens=False)

        capsule = tokens.__dlpack__()

        # PyCapsule type check
        assert type(capsule).__name__ == "PyCapsule"

    @pytest.mark.requires_model
    def test_dlpack_shared_semantics(self, tokenizer):
        """After __dlpack__, TokenArray is still valid (shared, not moved)."""
        tokens = tokenizer.encode("Hello")
        original_values = tokens.tolist()

        _ = tokens.__dlpack__()

        # TokenArray should still be accessible
        assert tokens.tolist() == original_values
        assert len(tokens) > 0

    @pytest.mark.requires_model
    def test_dlpack_multiple_exports(self, tokenizer):
        """Can call __dlpack__ multiple times (refcounted)."""
        tokens = tokenizer.encode("Hello world")
        expected = tokens.tolist()

        # Export multiple times - should not raise
        c1 = tokens.__dlpack__()
        c2 = tokens.__dlpack__()
        c3 = tokens.__dlpack__()

        # All should be capsules
        assert type(c1).__name__ == "PyCapsule"
        assert type(c2).__name__ == "PyCapsule"
        assert type(c3).__name__ == "PyCapsule"

        # TokenArray should still work
        assert tokens.tolist() == expected

    @pytest.mark.requires_model
    def test_dlpack_empty_raises(self, tokenizer):
        """Cannot export empty TokenArray via DLPack."""
        # Create empty by encoding empty string (no special tokens)
        tokens = tokenizer.encode("", special_tokens=False)

        if len(tokens) == 0:
            import talu

            with pytest.raises(talu.InteropError, match="empty"):
                _ = tokens.__dlpack__()

    @pytest.mark.requires_model
    def test_dlpack_tokenarray_valid_after_multiple_exports(self, tokenizer):
        """TokenArray remains fully functional after multiple exports."""
        tokens = tokenizer.encode("Hello world test")
        original = tokens.tolist()

        # Export multiple times
        for _ in range(5):
            _ = tokens.__dlpack__()

        # All operations should still work
        assert tokens.tolist() == original
        assert len(tokens) == len(original)
        assert tokens[0] == original[0]
        assert tokens[-1] == original[-1]
        assert list(tokens) == original
