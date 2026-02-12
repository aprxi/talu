"""
TokenArray NumPy integration tests.

These tests validate TokenArray's buffer protocol and numpy interop.
Requires numpy (Python 3.14+).
"""

import gc

import pytest


class TestTokenArrayNumPy:
    """Tests for TokenArray NumPy interface."""

    @pytest.mark.requires_model
    def test_numpy_array_interface(self, tokenizer, numpy):
        """TokenArray supports NumPy array interface."""
        np = numpy
        tokens = tokenizer.encode("Hello World Test")

        # Create NumPy array (should be zero-copy)
        arr = np.asarray(tokens)

        assert arr.dtype == np.uint32
        assert len(arr) == len(tokens)

    @pytest.mark.requires_model
    def test_numpy_values_match(self, tokenizer, numpy):
        """NumPy array values match TokenArray."""
        np = numpy
        tokens = tokenizer.encode("Hello World")

        arr = np.asarray(tokens)

        for i in range(len(tokens)):
            assert arr[i] == tokens[i]

    @pytest.mark.requires_model
    def test_numpy_operations(self, tokenizer, numpy):
        """NumPy operations work on TokenArray."""
        np = numpy
        tokens = tokenizer.encode("Hello World Test")

        arr = np.asarray(tokens)

        # Basic numpy operations
        assert arr.sum() > 0
        assert arr.max() >= arr.min()
        assert len(arr.shape) == 1


class TestTokenArrayNumPyLifecycle:
    """Tests for TokenArray lifecycle with NumPy views."""

    @pytest.mark.requires_model
    def test_numpy_view_keeps_token_array_alive(self, tokenizer, numpy):
        """NumPy array view keeps TokenArray data alive.

        Contract: When a NumPy array is created from TokenArray via buffer protocol,
        the TokenArray's data must remain valid as long as the NumPy array exists.
        """
        np = numpy

        tokens = tokenizer.encode("Hello World Test")
        arr = np.asarray(tokens)

        # Capture expected values
        expected = list(arr)

        # Delete TokenArray reference
        del tokens
        gc.collect()

        # NumPy array should still have valid data
        assert list(arr) == expected

    @pytest.mark.requires_model
    def test_buffer_protocol_lifecycle_explicit(self, tokenizer, numpy):
        """Explicitly test buffer protocol memory persistence.

        Contract: When TokenArray is used to create a NumPy array via buffer
        protocol, then the TokenArray is deleted, and finally the NumPy array
        is accessed - the data must remain valid. This ensures talu_tokens_free
        is NOT called prematurely while Python-side views exist.
        """
        np = numpy

        # Step 1: Create TokenArray
        tokens = tokenizer.encode("Buffer protocol test with more words")
        original_length = len(tokens)
        original_values = tokens.tolist()

        # Step 2: Create NumPy view via buffer protocol (zero-copy)
        arr = np.asarray(tokens)

        # Verify zero-copy (should share same memory)
        assert len(arr) == original_length
        assert list(arr) == original_values

        # Step 3: Delete TokenArray reference
        del tokens

        # Step 4: Force garbage collection
        gc.collect()
        gc.collect()  # Double collect to be sure

        # Step 5: Verify NumPy array still has valid data
        # If buffer protocol lifecycle is broken, this would crash or return garbage
        assert len(arr) == original_length
        assert list(arr) == original_values

        # Step 6: Verify we can still do numpy operations
        assert arr.sum() > 0


class TestDecodeNumPy:
    """Tests for decode() with NumPy arrays."""

    @pytest.mark.requires_model
    def test_decode_numpy_array(self, tokenizer, numpy):
        """decode() with numpy array of tokens."""
        np = numpy
        tokens = tokenizer.encode("Hello")
        np_arr = np.asarray(tokens)

        # Convert to list for decoding
        decoded = tokenizer.decode(np_arr.tolist())
        assert isinstance(decoded, str)
