"""
Tests for talu.tokenizer.batch - BatchEncoding and TokenArrayView.

Tests the batch tokenization container, including:
- List-like interface (indexing, iteration, len)
- Dict-like interface (keys, string indexing)
- TokenArrayView slice and sequence operations
- DLPack export for input_ids and attention_mask
- Padding configuration and to_list() conversion
"""

import pytest

from talu.exceptions import InteropError, StateError, ValidationError
from talu.tokenizer.batch import BatchEncoding
from talu.tokenizer.token_array import TokenArrayView

# =============================================================================
# BatchEncoding - List-like Interface
# =============================================================================


class TestBatchEncodingListInterface:
    """Tests for BatchEncoding list-like operations."""

    def test_len_returns_num_sequences(self, tokenizer):
        """len() returns number of encoded sequences."""
        batch = tokenizer.encode(["Hello", "World", "Test"])

        assert len(batch) == 3

    def test_single_sequence_len(self, tokenizer):
        """Single sequence batch has len 1."""
        batch = tokenizer.encode(["Hello world"])

        assert len(batch) == 1

    def test_empty_list_len(self, tokenizer):
        """Empty list produces len 0 batch."""
        batch = tokenizer.encode([])

        assert len(batch) == 0

    def test_indexing_returns_token_array_view(self, tokenizer):
        """Positive indexing returns TokenArrayView."""
        batch = tokenizer.encode(["Hello", "World"])

        view = batch[0]

        assert isinstance(view, TokenArrayView)

    def test_negative_indexing(self, tokenizer):
        """Negative indexing works like lists."""
        batch = tokenizer.encode(["First", "Second", "Third"])

        last = batch[-1]

        assert isinstance(last, TokenArrayView)
        # Different sequences should have different tokens (unless by chance)
        assert list(last) == list(batch[2])

    def test_index_out_of_range(self, tokenizer):
        """Out of range index raises IndexError."""
        batch = tokenizer.encode(["Hello"])

        with pytest.raises(IndexError, match="out of range"):
            batch[5]

    def test_iteration(self, tokenizer):
        """Iteration yields TokenArrayViews."""
        batch = tokenizer.encode(["Hello", "World", "Test"])

        views = list(batch)

        assert len(views) == 3
        assert all(isinstance(v, TokenArrayView) for v in views)


# =============================================================================
# BatchEncoding - Dict-like Interface
# =============================================================================


class TestBatchEncodingDictInterface:
    """Tests for BatchEncoding dict-like operations."""

    def test_keys(self, tokenizer):
        """keys() returns input_ids and attention_mask."""
        batch = tokenizer.encode(["Hello"])

        keys = batch.keys()

        assert "input_ids" in keys
        assert "attention_mask" in keys

    def test_string_key_input_ids(self, tokenizer):
        """batch["input_ids"] returns DLPack-compatible accessor."""
        batch = tokenizer.encode(["Hello"])

        accessor = batch["input_ids"]

        assert hasattr(accessor, "__dlpack__")
        assert hasattr(accessor, "__dlpack_device__")

    def test_string_key_attention_mask(self, tokenizer):
        """batch["attention_mask"] returns DLPack-compatible accessor."""
        batch = tokenizer.encode(["Hello"])

        accessor = batch["attention_mask"]

        assert hasattr(accessor, "__dlpack__")
        assert hasattr(accessor, "__dlpack_device__")

    def test_invalid_key_raises_key_error(self, tokenizer):
        """Invalid string key raises KeyError."""
        batch = tokenizer.encode(["Hello"])

        with pytest.raises(KeyError, match="Unknown key"):
            batch["invalid_key"]

    def test_contains(self, tokenizer):
        """__contains__ checks for valid keys."""
        batch = tokenizer.encode(["Hello"])

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "other" not in batch


# =============================================================================
# TokenArrayView Tests
# =============================================================================


class TestTokenArrayView:
    """Tests for TokenArrayView sequence operations."""

    def test_len(self, tokenizer):
        """len() returns token count."""
        batch = tokenizer.encode(["Hello"])
        view = batch[0]

        length = len(view)

        assert length > 0

    def test_getitem_single(self, tokenizer):
        """Single index returns token ID."""
        batch = tokenizer.encode(["Hello"])
        view = batch[0]

        token = view[0]

        assert isinstance(token, int)

    def test_getitem_negative(self, tokenizer):
        """Negative index works."""
        batch = tokenizer.encode(["Hello"])
        view = batch[0]

        last_token = view[-1]

        assert isinstance(last_token, int)
        assert last_token == view[len(view) - 1]

    def test_getitem_slice(self, tokenizer):
        """Slice returns list of tokens."""
        batch = tokenizer.encode(["Hello world this is a test"])
        view = batch[0]

        slice_result = view[:3]

        assert isinstance(slice_result, list)
        assert len(slice_result) <= 3

    def test_index_out_of_range(self, tokenizer):
        """Out of range index raises IndexError."""
        batch = tokenizer.encode(["Hello"])
        view = batch[0]

        with pytest.raises(IndexError):
            view[1000]

    def test_iteration(self, tokenizer):
        """Iteration yields token IDs."""
        batch = tokenizer.encode(["Hello"])
        view = batch[0]

        tokens = list(view)

        assert len(tokens) == len(view)
        assert all(isinstance(t, int) for t in tokens)

    def test_contains(self, tokenizer):
        """__contains__ checks for token ID."""
        batch = tokenizer.encode(["Hello"])
        view = batch[0]

        # First token should be in the view
        first_token = view[0]
        assert first_token in view

        # Very large token ID should not be in the view
        assert 999999999 not in view

    def test_tolist(self, tokenizer):
        """tolist() returns Python list."""
        batch = tokenizer.encode(["Hello"])
        view = batch[0]

        result = view.tolist()

        assert isinstance(result, list)
        assert result == list(view)

    def test_equality_with_list(self, tokenizer):
        """TokenArrayView equals list with same tokens."""
        batch = tokenizer.encode(["Hello"])
        view = batch[0]

        token_list = view.tolist()

        assert view == token_list

    def test_count(self, tokenizer):
        """count() returns occurrences of token."""
        batch = tokenizer.encode(["Hello"])
        view = batch[0]

        first_token = view[0]
        count = view.count(first_token)

        assert count >= 1

    def test_index_method(self, tokenizer):
        """index() returns first occurrence."""
        batch = tokenizer.encode(["Hello"])
        view = batch[0]

        first_token = view[0]
        idx = view.index(first_token)

        assert idx == 0

    def test_index_not_found(self, tokenizer):
        """index() raises ValueError if not found."""
        batch = tokenizer.encode(["Hello"])
        view = batch[0]

        with pytest.raises(ValueError, match="not in TokenArrayView"):
            view.index(999999999)

    def test_repr(self, tokenizer):
        """repr() is informative."""
        batch = tokenizer.encode(["Hello"])
        view = batch[0]

        repr_str = repr(view)

        assert "TokenArrayView" in repr_str
        assert "len=" in repr_str


# =============================================================================
# BatchEncoding Properties and Methods
# =============================================================================


class TestBatchEncodingProperties:
    """Tests for BatchEncoding properties and methods."""

    def test_total_tokens(self, tokenizer):
        """total_tokens returns sum of all tokens."""
        batch = tokenizer.encode(["Hello", "World"])

        total = batch.total_tokens

        assert total >= 2  # At least one token per sequence

    def test_lengths(self, tokenizer):
        """lengths() returns list of sequence lengths."""
        batch = tokenizer.encode(["Hi", "Hello world"])

        lengths = batch.lengths()

        assert isinstance(lengths, list)
        assert len(lengths) == 2

    def test_max_length(self, tokenizer):
        """max_length() returns longest sequence length."""
        batch = tokenizer.encode(["Hi", "Hello world this is longer"])

        max_len = batch.max_length()

        lengths = batch.lengths()
        assert max_len == max(lengths)

    def test_padding_side_default(self, tokenizer):
        """padding_side defaults to 'left'."""
        batch = tokenizer.encode(["Hello"])

        assert batch.padding_side == "left"

    def test_padding_side_setter_valid(self, tokenizer):
        """padding_side can be set to 'left' or 'right'."""
        batch = tokenizer.encode(["Hello"])

        batch.padding_side = "right"
        assert batch.padding_side == "right"

        batch.padding_side = "left"
        assert batch.padding_side == "left"

    def test_padding_side_setter_invalid(self, tokenizer):
        """Invalid padding_side raises ValidationError."""
        batch = tokenizer.encode(["Hello"])

        with pytest.raises(ValidationError, match="must be 'left' or 'right'"):
            batch.padding_side = "center"

    def test_repr(self, tokenizer):
        """repr() shows num_sequences and total_tokens."""
        batch = tokenizer.encode(["Hello", "World"])

        repr_str = repr(batch)

        assert "BatchEncoding" in repr_str
        assert "num_sequences=2" in repr_str
        assert "total_tokens=" in repr_str


# =============================================================================
# DLPack Export Tests
# =============================================================================


class TestDLPackExport:
    """Tests for DLPack tensor export."""

    def test_input_ids_accessor_dlpack_device(self, tokenizer):
        """input_ids accessor.__dlpack_device__ returns CPU."""
        batch = tokenizer.encode(["Hello"])
        accessor = batch.input_ids

        device = accessor.__dlpack_device__()

        assert device == (1, 0)  # kDLCPU = 1

    def test_attention_mask_accessor_dlpack_device(self, tokenizer):
        """attention_mask accessor.__dlpack_device__ returns CPU."""
        batch = tokenizer.encode(["Hello"])
        accessor = batch.attention_mask

        device = accessor.__dlpack_device__()

        assert device == (1, 0)  # kDLCPU = 1

    def test_batch_dlpack_raises_interop_error(self, tokenizer):
        """Calling __dlpack__ on batch itself raises InteropError."""
        batch = tokenizer.encode(["Hello"])

        with pytest.raises(InteropError, match="contains multiple tensors"):
            batch.__dlpack__()

    def test_empty_batch_dlpack_raises(self, tokenizer):
        """Empty batch cannot export DLPack."""
        batch = tokenizer.encode([])

        with pytest.raises(InteropError, match="empty"):
            batch.input_ids.__dlpack__()

    def test_accessor_repr(self, tokenizer):
        """Accessor repr is informative."""
        batch = tokenizer.encode(["Hello", "World"])

        ids_repr = repr(batch.input_ids)
        mask_repr = repr(batch.attention_mask)

        assert "InputIdsAccessor" in ids_repr
        assert "num_sequences=2" in ids_repr
        assert "AttentionMaskAccessor" in mask_repr


# =============================================================================
# to_list() Tests
# =============================================================================


class TestToList:
    """Tests for BatchEncoding.to_list() conversion."""

    def test_to_list_returns_dict(self, tokenizer):
        """to_list() returns dict with input_ids."""
        batch = tokenizer.encode(["Hello", "World"])

        result = batch.to_list()

        assert isinstance(result, dict)
        assert "input_ids" in result

    def test_to_list_includes_attention_mask(self, tokenizer):
        """to_list() includes attention_mask by default."""
        batch = tokenizer.encode(["Hello", "World"])

        result = batch.to_list()

        assert "attention_mask" in result

    def test_to_list_without_mask(self, tokenizer):
        """to_list() can exclude attention_mask."""
        batch = tokenizer.encode(["Hello", "World"])

        result = batch.to_list(return_attention_mask=False)

        assert "attention_mask" not in result

    def test_to_list_padded_shape(self, tokenizer):
        """to_list() produces rectangular 2D structure."""
        batch = tokenizer.encode(["Hi", "Hello world this is longer"])

        result = batch.to_list()
        input_ids = result["input_ids"]

        # All rows should have same length (padded)
        lengths = [len(row) for row in input_ids]
        assert len(set(lengths)) == 1

    def test_to_list_empty_batch(self, tokenizer):
        """Empty batch returns empty lists."""
        batch = tokenizer.encode([])

        result = batch.to_list()

        assert result["input_ids"] == []
        assert result["attention_mask"] == []

    def test_to_list_invalid_padding_side(self, tokenizer):
        """Invalid padding_side raises ValidationError."""
        batch = tokenizer.encode(["Hello"])

        with pytest.raises(ValidationError, match="must be 'left' or 'right'"):
            batch.to_list(padding_side="invalid")


# Note: to_numpy() tests are in tests/reference/tokenizer/test_batch_numpy.py
# because they require numpy (external dependency)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestBatchEncodingEdgeCases:
    """Edge case tests for stability and robustness."""

    def test_encode_empty_string(self, tokenizer):
        """Encoding empty string works."""
        batch = tokenizer.encode([""])

        assert len(batch) == 1
        # Empty string may produce tokens (e.g., BOS) or not
        assert isinstance(batch[0].tolist(), list)

    def test_encode_whitespace_only(self, tokenizer):
        """Encoding whitespace-only strings works."""
        batch = tokenizer.encode(["   ", "\t\n", " "])

        assert len(batch) == 3
        for view in batch:
            assert isinstance(view.tolist(), list)

    def test_encode_mixed_empty_and_content(self, tokenizer):
        """Mixed empty and content strings work."""
        batch = tokenizer.encode(["Hello", "", "World", ""])

        assert len(batch) == 4
        # Content strings should have tokens
        assert len(batch[0]) > 0
        assert len(batch[2]) > 0

    def test_encode_long_input(self, tokenizer):
        """Long input string works (within tokenizer limits)."""
        long_text = "hello " * 1000
        batch = tokenizer.encode([long_text])

        assert len(batch) == 1
        assert batch.total_tokens > 0 or len(batch[0]) > 0

    def test_encode_unicode_edge_cases(self, tokenizer):
        """Unicode edge cases work."""
        unicode_cases = [
            "\u0000",  # Null character
            "\u200b",  # Zero-width space
            "\uffff",  # Noncharacter
            "🎉" * 100,  # Many emojis
            "日本語テスト",  # Japanese
            "مرحبا العالم",  # Arabic
        ]
        batch = tokenizer.encode(unicode_cases)

        assert len(batch) == len(unicode_cases)

    def test_repeated_encode_no_leak(self, tokenizer):
        """Repeated encoding doesn't leak memory."""
        for _ in range(100):
            batch = tokenizer.encode(["Hello", "World", "Test"])
            assert len(batch) == 3
            # Let batch go out of scope to trigger cleanup

    def test_repeated_to_list_no_leak(self, tokenizer):
        """Repeated to_list() calls don't leak memory."""
        batch = tokenizer.encode(["Hello", "World"])
        for _ in range(50):
            result = batch.to_list()
            assert "input_ids" in result

    def test_batch_with_padding_right(self, tokenizer):
        """Padding right works correctly."""
        batch = tokenizer.encode(["Hi", "Hello world this is longer"])
        batch.padding_side = "right"

        result = batch.to_list(padding_side="right")
        input_ids = result["input_ids"]

        # All rows should have same length
        assert len({len(row) for row in input_ids}) == 1

    def test_batch_truncation(self, tokenizer):
        """Truncation option works."""
        batch = tokenizer.encode(["Hello world this is a longer test"])

        result = batch.to_list(max_length=5, truncation=True)
        input_ids = result["input_ids"]

        # Should be truncated to max_length
        assert all(len(row) <= 5 for row in input_ids)


class TestTokenArrayViewEdgeCases:
    """Edge case tests for TokenArrayView."""

    def test_view_equality_with_another_view(self, tokenizer):
        """TokenArrayView equality with another view."""
        batch = tokenizer.encode(["Hello", "Hello"])

        view1 = batch[0]
        view2 = batch[1]

        # Same input should produce equal views
        assert view1 == view2

    def test_view_equality_different_lengths(self, tokenizer):
        """Views of different lengths are not equal."""
        batch = tokenizer.encode(["Hi", "Hello world longer"])

        view1 = batch[0]
        view2 = batch[1]

        assert view1 != view2

    def test_view_index_with_start_stop(self, tokenizer):
        """index() with start/stop bounds."""
        batch = tokenizer.encode(["Hello world test hello world"])
        view = batch[0]

        if len(view) > 3:
            first_token = view[0]
            # Search starting from index 1
            try:
                idx = view.index(first_token, start=1)
                assert idx >= 1
            except ValueError:
                # Token not found after index 1 is valid
                pass

    def test_view_slice_with_step(self, tokenizer):
        """Slice with step works."""
        batch = tokenizer.encode(["Hello world this is a test"])
        view = batch[0]

        if len(view) >= 4:
            sliced = view[::2]  # Every other token
            assert len(sliced) == (len(view) + 1) // 2


class TestBatchResourceManagement:
    """Tests for batch resource management and cleanup."""

    def test_batch_del_is_safe(self, tokenizer):
        """Explicitly deleting batch is safe."""
        batch = tokenizer.encode(["Hello", "World"])

        del batch
        # Should not crash

    def test_view_outlives_batch_reference(self, tokenizer):
        """TokenArrayView holds reference to parent batch."""
        import gc

        batch = tokenizer.encode(["Hello", "World"])
        view = batch[0]
        tokens_copy = view.tolist()

        # Batch still accessible via view's parent reference
        assert view._parent is batch

        # Force collection
        gc.collect()

        # View should still work (parent keeps batch alive)
        assert view.tolist() == tokens_copy

    def test_accessor_repr_after_iteration(self, tokenizer):
        """Accessors work after iterating batch."""
        batch = tokenizer.encode(["Hello", "World"])

        # Iterate through batch
        _ = list(batch)

        # Accessors should still work
        input_ids_accessor = batch.input_ids
        mask_accessor = batch.attention_mask

        assert "InputIdsAccessor" in repr(input_ids_accessor)
        assert "AttentionMaskAccessor" in repr(mask_accessor)

    def test_empty_batch_iteration(self, tokenizer):
        """Iterating empty batch works."""
        batch = tokenizer.encode([])

        views = list(batch)
        assert views == []

    def test_empty_batch_lengths(self, tokenizer):
        """Empty batch lengths() returns empty list."""
        batch = tokenizer.encode([])

        lengths = batch.lengths()
        assert lengths == []

    def test_empty_batch_max_length(self, tokenizer):
        """Empty batch max_length() returns 0."""
        batch = tokenizer.encode([])

        max_len = batch.max_length()
        assert max_len == 0

    def test_uninitialized_batch_getitem_raises(self):
        """Accessing uninitialized batch raises IndexError or StateError.

        An uninitialized batch has num_sequences=0, so IndexError is raised
        before the StateError check for pointers.
        """
        # Create batch without proper initialization
        batch = BatchEncoding()

        # Should raise IndexError because num_sequences=0
        with pytest.raises(IndexError, match="out of range"):
            batch[0]

    def test_partially_initialized_batch_raises_state_error(self):
        """Batch with sequences but no pointers raises StateError."""
        # Create batch with num_sequences > 0 but no pointers
        batch = BatchEncoding(num_sequences=1)

        with pytest.raises(StateError, match="not initialized"):
            batch[0]
