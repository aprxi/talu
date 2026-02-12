"""
Additional tests for talu/tokenizer/token_array.py coverage.

Targets uncovered edge cases in TokenArray.
"""

import pytest

from talu.exceptions import StateError
from talu.tokenizer.token_array import TokenArray, TokenOffset

# =============================================================================
# TokenOffset Tests
# =============================================================================


class TestTokenOffset:
    """Tests for TokenOffset class."""

    def test_construction(self):
        """TokenOffset stores start and end."""
        offset = TokenOffset(10, 20)
        assert offset.start == 10
        assert offset.end == 20

    def test_repr(self):
        """TokenOffset has tuple-like repr."""
        offset = TokenOffset(5, 15)
        assert repr(offset) == "(5, 15)"

    def test_eq_with_offset(self):
        """TokenOffset equals another TokenOffset."""
        a = TokenOffset(1, 2)
        b = TokenOffset(1, 2)
        c = TokenOffset(1, 3)
        assert a == b
        assert a != c

    def test_eq_with_tuple(self):
        """TokenOffset equals tuple."""
        offset = TokenOffset(1, 2)
        assert offset == (1, 2)
        assert offset != (1, 3)

    def test_eq_not_implemented(self):
        """TokenOffset returns NotImplemented for other types."""
        offset = TokenOffset(1, 2)
        assert offset.__eq__("not a tuple") is NotImplemented
        assert offset.__eq__([1, 2]) is NotImplemented

    def test_iter_unpacking(self):
        """TokenOffset supports tuple unpacking."""
        offset = TokenOffset(10, 20)
        start, end = offset
        assert start == 10
        assert end == 20

    def test_slice_string(self):
        """TokenOffset.slice() works with string."""
        text = "Hello World"
        offset = TokenOffset(0, 5)
        assert offset.slice(text) == "Hello"

    def test_slice_bytes(self):
        """TokenOffset.slice() works with bytes."""
        text = b"Hello World"
        offset = TokenOffset(6, 11)
        assert offset.slice(text) == "World"

    def test_slice_with_errors_replace(self):
        """TokenOffset.slice() handles decode errors."""
        # Invalid UTF-8 sequence
        text = b"Hello \xff\xfe World"
        offset = TokenOffset(6, 10)
        result = offset.slice(text, errors="replace")
        assert "\ufffd" in result  # Replacement character


# =============================================================================
# TokenArray Construction Tests
# =============================================================================


class TestTokenArrayConstruction:
    """Tests for TokenArray construction."""

    def test_from_list_empty(self):
        """TokenArray._from_list handles empty list."""
        arr = TokenArray._from_list([])
        assert len(arr) == 0
        assert arr.tolist() == []

    def test_from_list_with_tokens(self):
        """TokenArray._from_list creates array from list."""
        arr = TokenArray._from_list([1, 2, 3])
        assert len(arr) == 3
        assert arr.tolist() == [1, 2, 3]


# =============================================================================
# TokenArray Access Tests
# =============================================================================


class TestTokenArrayAccess:
    """Tests for TokenArray access methods."""

    def test_getitem_positive_index(self):
        """__getitem__ with positive index."""
        arr = TokenArray._from_list([10, 20, 30])
        assert arr[0] == 10
        assert arr[1] == 20
        assert arr[2] == 30

    def test_getitem_negative_index(self):
        """__getitem__ with negative index."""
        arr = TokenArray._from_list([10, 20, 30])
        assert arr[-1] == 30
        assert arr[-2] == 20
        assert arr[-3] == 10

    def test_getitem_out_of_range(self):
        """__getitem__ raises IndexError for out of range."""
        arr = TokenArray._from_list([1, 2, 3])
        with pytest.raises(IndexError):
            _ = arr[10]
        with pytest.raises(IndexError):
            _ = arr[-10]

    def test_getitem_slice(self):
        """__getitem__ with slice returns TokenArray."""
        arr = TokenArray._from_list([1, 2, 3, 4, 5])
        sliced = arr[1:4]
        assert isinstance(sliced, TokenArray)
        assert sliced.tolist() == [2, 3, 4]

    def test_getitem_slice_step(self):
        """__getitem__ with slice and step."""
        arr = TokenArray._from_list([0, 1, 2, 3, 4, 5])
        sliced = arr[::2]
        assert sliced.tolist() == [0, 2, 4]

    def test_getitem_null_ptr_raises(self):
        """__getitem__ raises StateError for null ptr."""
        arr = TokenArray(None, 0)
        with pytest.raises(StateError):
            _ = arr[0]


# =============================================================================
# TokenArray Iterator Tests
# =============================================================================


class TestTokenArrayIterator:
    """Tests for TokenArray iteration."""

    def test_iter_basic(self):
        """__iter__ yields all tokens."""
        arr = TokenArray._from_list([1, 2, 3])
        assert list(arr) == [1, 2, 3]

    def test_iter_empty(self):
        """__iter__ on empty array yields nothing."""
        arr = TokenArray._from_list([])
        assert list(arr) == []

    def test_iter_null_ptr(self):
        """__iter__ on null ptr yields nothing."""
        arr = TokenArray(None, 0)
        assert list(arr) == []


# =============================================================================
# TokenArray tolist Tests
# =============================================================================


class TestTokenArrayTolist:
    """Tests for TokenArray.tolist()."""

    def test_tolist_basic(self):
        """tolist() returns Python list."""
        arr = TokenArray._from_list([1, 2, 3])
        result = arr.tolist()
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_tolist_null_ptr(self):
        """tolist() returns empty list for null ptr."""
        arr = TokenArray(None, 0)
        assert arr.tolist() == []


# =============================================================================
# TokenArray Contains Tests
# =============================================================================


class TestTokenArrayContains:
    """Tests for TokenArray.__contains__()."""

    def test_contains_present(self):
        """__contains__ returns True for present value."""
        arr = TokenArray._from_list([1, 2, 3])
        assert 2 in arr

    def test_contains_absent(self):
        """__contains__ returns False for absent value."""
        arr = TokenArray._from_list([1, 2, 3])
        assert 99 not in arr

    def test_contains_null_ptr(self):
        """__contains__ returns False for null ptr."""
        arr = TokenArray(None, 0)
        assert 1 not in arr


# =============================================================================
# TokenArray index Tests
# =============================================================================


class TestTokenArrayIndex:
    """Tests for TokenArray.index()."""

    def test_index_found(self):
        """index() returns index of value."""
        arr = TokenArray._from_list([10, 20, 30])
        assert arr.index(20) == 1

    def test_index_with_start(self):
        """index() respects start parameter."""
        arr = TokenArray._from_list([1, 2, 1, 2])
        assert arr.index(1, 1) == 2

    def test_index_with_stop(self):
        """index() respects stop parameter."""
        arr = TokenArray._from_list([1, 2, 3])
        with pytest.raises(ValueError):
            arr.index(3, 0, 2)  # Stop before value

    def test_index_not_found(self):
        """index() raises ValueError when not found."""
        arr = TokenArray._from_list([1, 2, 3])
        with pytest.raises(ValueError):
            arr.index(99)

    def test_index_null_ptr(self):
        """index() raises ValueError for null ptr."""
        arr = TokenArray(None, 0)
        with pytest.raises(ValueError):
            arr.index(1)


# =============================================================================
# TokenArray count Tests
# =============================================================================


class TestTokenArrayCount:
    """Tests for TokenArray.count()."""

    def test_count_present(self):
        """count() returns occurrences."""
        arr = TokenArray._from_list([1, 2, 1, 3, 1])
        assert arr.count(1) == 3

    def test_count_absent(self):
        """count() returns 0 for absent value."""
        arr = TokenArray._from_list([1, 2, 3])
        assert arr.count(99) == 0

    def test_count_null_ptr(self):
        """count() returns 0 for null ptr."""
        arr = TokenArray(None, 0)
        assert arr.count(1) == 0


# =============================================================================
# TokenArray Multiplication Tests
# =============================================================================


class TestTokenArrayMultiplication:
    """Tests for TokenArray.__mul__ and __rmul__."""

    def test_mul_basic(self):
        """__mul__ repeats array."""
        arr = TokenArray._from_list([1, 2])
        result = arr * 3
        assert result.tolist() == [1, 2, 1, 2, 1, 2]

    def test_mul_zero(self):
        """__mul__ with 0 returns empty array."""
        arr = TokenArray._from_list([1, 2])
        result = arr * 0
        assert result.tolist() == []

    def test_mul_negative(self):
        """__mul__ with negative returns empty array."""
        arr = TokenArray._from_list([1, 2])
        result = arr * -1
        assert result.tolist() == []

    def test_mul_one(self):
        """__mul__ with 1 returns copy."""
        arr = TokenArray._from_list([1, 2])
        result = arr * 1
        assert result.tolist() == [1, 2]

    def test_mul_not_implemented(self):
        """__mul__ returns NotImplemented for non-int."""
        arr = TokenArray._from_list([1, 2])
        assert arr.__mul__("x") is NotImplemented

    def test_rmul_basic(self):
        """__rmul__ repeats array."""
        arr = TokenArray._from_list([1, 2])
        result = 2 * arr
        assert result.tolist() == [1, 2, 1, 2]


# =============================================================================
# TokenArray Equality Tests
# =============================================================================


class TestTokenArrayEquality:
    """Tests for TokenArray.__eq__."""

    def test_eq_token_array(self):
        """__eq__ compares with TokenArray."""
        a = TokenArray._from_list([1, 2, 3])
        b = TokenArray._from_list([1, 2, 3])
        c = TokenArray._from_list([1, 2, 4])
        assert a == b
        assert a != c

    def test_eq_list(self):
        """__eq__ compares with list."""
        arr = TokenArray._from_list([1, 2, 3])
        assert arr == [1, 2, 3]
        assert arr != [1, 2, 4]

    def test_eq_different_length(self):
        """__eq__ handles different lengths."""
        arr = TokenArray._from_list([1, 2, 3])
        assert arr != [1, 2]
        assert arr != TokenArray._from_list([1, 2])

    def test_eq_not_implemented(self):
        """__eq__ returns NotImplemented for other types."""
        arr = TokenArray._from_list([1, 2, 3])
        assert arr.__eq__("not a list") is NotImplemented


# =============================================================================
# TokenArray Repr Tests
# =============================================================================


class TestTokenArrayRepr:
    """Tests for TokenArray.__repr__."""

    def test_repr_short(self):
        """__repr__ shows all tokens for short arrays."""
        arr = TokenArray._from_list([1, 2, 3])
        repr_str = repr(arr)
        assert "TokenArray" in repr_str
        assert "[1, 2, 3]" in repr_str

    def test_repr_long(self):
        """__repr__ truncates long arrays."""
        arr = TokenArray._from_list(list(range(20)))
        repr_str = repr(arr)
        assert "..." in repr_str
        assert "len=20" in repr_str

    def test_repr_null_ptr(self):
        """__repr__ handles null ptr."""
        arr = TokenArray(None, 0)
        repr_str = repr(arr)
        assert "len=0" in repr_str
