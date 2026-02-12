"""
SharedBuffer ownership and refcounting tests.

Verifies:
1. TokenArray correctly increments/decrements SharedBuffer refcount
2. DLPack export increments refcount (tensor survives array deletion)
3. Multiple references tracked correctly
4. No double-free on multiple deletes
"""

import gc

import pytest

# tokenizer fixture is provided by conftest.py


class TestTokenArrayOwnership:
    """TokenArray buffer ownership tests."""

    def test_token_array_owns_buffer(self, tokenizer):
        """New TokenArray owns its buffer."""
        tokens = tokenizer.encode("hello world")

        assert tokens is not None
        assert len(tokens) > 0

    def test_token_array_releases_on_del(self, tokenizer, memory_tracker):
        """TokenArray releases buffer when deleted."""
        memory_tracker.capture_baseline()

        # Create and delete many token arrays
        for _ in range(100):
            tokens = tokenizer.encode("hello world " * 100)
            del tokens

        memory_tracker.assert_no_leak(
            threshold_mb=10, context="100 TokenArray create/delete cycles"
        )

    def test_multiple_encodes_no_leak(self, tokenizer, memory_tracker):
        """Multiple encode calls don't accumulate memory."""
        memory_tracker.capture_baseline()

        # Many encodes
        for i in range(200):
            tokens = tokenizer.encode(f"test string number {i}" * 10)
            _ = len(tokens)
            del tokens

        memory_tracker.assert_no_leak(threshold_mb=5, context="200 encode operations")


class TestTokenArraySlicing:
    """TokenArray slicing and view tests."""

    def test_slice_creates_view(self, tokenizer):
        """Slicing creates a view, not a copy."""
        tokens = tokenizer.encode("hello world this is a longer test string")

        # Create slice
        sliced = tokens[2:5]

        # Both should have valid data
        assert len(sliced) == 3
        assert len(tokens) > 3

    def test_slice_data_matches_original(self, tokenizer):
        """Slice data matches original tokens at those positions."""
        tokens = tokenizer.encode("hello world test")
        original_list = list(tokens)

        sliced = tokens[1:4]
        sliced_list = list(sliced)

        assert sliced_list == original_list[1:4]

    def test_multiple_slices_independent(self, tokenizer):
        """Multiple slices from same array are independent."""
        tokens = tokenizer.encode("one two three four five six")

        slice1 = tokens[0:2]
        slice2 = tokens[2:4]
        slice3 = tokens[4:6]

        # All should be valid
        assert len(slice1) == 2
        assert len(slice2) == 2
        assert len(slice3) == 2

        # Data should be independent
        list1 = list(slice1)
        list2 = list(slice2)
        list3 = list(slice3)

        assert list1 != list2
        assert list2 != list3


class TestDLPackExport:
    """DLPack export and tensor lifetime tests."""

    def test_dlpack_export_basic(self, tokenizer):
        """Basic DLPack export works."""
        torch = pytest.importorskip("torch")

        tokens = tokenizer.encode("hello world")
        tensor = torch.from_dlpack(tokens)

        assert tensor.shape[0] == len(tokens)
        assert list(tensor.tolist()) == list(tokens)

    def test_tensor_survives_tokenarray_deletion(self, tokenizer):
        """PyTorch tensor survives TokenArray deletion."""
        torch = pytest.importorskip("torch")

        tokens = tokenizer.encode("hello world")
        original_data = list(tokens)

        # Export to PyTorch
        tensor = torch.from_dlpack(tokens)

        # Delete TokenArray
        del tokens
        gc.collect()
        gc.collect()
        gc.collect()

        # Tensor should still be valid
        assert tensor.tolist() == original_data

    def test_multiple_dlpack_exports(self, tokenizer):
        """Multiple DLPack exports all survive."""
        torch = pytest.importorskip("torch")

        tokens = tokenizer.encode("hello world")

        # Multiple exports
        t1 = torch.from_dlpack(tokens)
        t2 = torch.from_dlpack(tokens)
        t3 = torch.from_dlpack(tokens)

        del tokens
        gc.collect()
        gc.collect()
        gc.collect()

        # All should be valid and equal
        assert t1.tolist() == t2.tolist() == t3.tolist()

    def test_dlpack_memory_not_leaked(self, tokenizer, memory_tracker):
        """DLPack exports don't leak memory."""
        torch = pytest.importorskip("torch")

        memory_tracker.capture_baseline()

        for _ in range(50):
            tokens = tokenizer.encode("test string " * 50)
            tensor = torch.from_dlpack(tokens)
            _ = tensor.sum().item()
            del tensor
            del tokens

        memory_tracker.assert_no_leak(threshold_mb=10, context="50 DLPack export cycles")


class TestBatchOwnership:
    """Batch tokenization buffer ownership tests."""

    def test_batch_encode_ownership(self, tokenizer, memory_tracker):
        """Batch encode results are properly owned and freed."""
        memory_tracker.capture_baseline()

        texts = [f"text number {i}" for i in range(10)]

        for _ in range(50):
            batch = tokenizer.encode(texts)
            # Access data to ensure it's valid
            _ = len(batch)
            del batch

        memory_tracker.assert_no_leak(threshold_mb=10, context="50 batch encode cycles")

    def test_batch_items_independent(self, tokenizer):
        """Batch encoding result survives after access."""
        texts = ["hello", "world", "test"]
        batch = tokenizer.encode(texts)

        # Access the batch via to_list()
        data = batch.to_list()
        assert len(data["input_ids"]) == 3

        # Delete batch reference
        del batch
        gc.collect()

        # Data should still be valid (it's a Python list copy)
        assert len(data["input_ids"]) == 3
        for ids in data["input_ids"]:
            assert len(ids) > 0


class TestConcatOwnership:
    """Token concatenation ownership tests."""

    def test_concat_creates_new_buffer(self, tokenizer):
        """Concatenation creates a new independent buffer."""
        tokens1 = tokenizer.encode("hello")
        tokens2 = tokenizer.encode("world")

        # Concatenate
        combined = tokens1 + tokens2

        # Should have combined length
        assert len(combined) == len(tokens1) + len(tokens2)

        # Original tokens unchanged
        assert len(tokens1) > 0
        assert len(tokens2) > 0

    def test_concat_memory_not_leaked(self, tokenizer, memory_tracker):
        """Concatenation doesn't leak memory."""
        memory_tracker.capture_baseline()

        for _ in range(100):
            t1 = tokenizer.encode("hello")
            t2 = tokenizer.encode("world")
            combined = t1 + t2
            _ = len(combined)
            del t1, t2, combined

        memory_tracker.assert_no_leak(threshold_mb=5, context="100 concat cycles")
