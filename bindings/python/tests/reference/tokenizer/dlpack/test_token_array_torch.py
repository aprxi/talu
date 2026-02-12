"""
TokenArray DLPack torch integration tests.

These tests validate TokenArray's DLPack export with PyTorch:
- True zero-copy (pointer equality)
- Tensor survives TokenArray deletion (refcount safety)
- Multiple exports are safe
- Refcount correctness
"""

import gc

import pytest

torch = pytest.importorskip("torch")


class TestTokenArrayDLPackZeroCopy:
    """Tests proving TokenArray DLPack export is true zero-copy."""

    @pytest.mark.requires_model
    def test_dlpack_values_match(self, tokenizer):
        """torch.from_dlpack(tokens) values equal tokens.tolist()."""
        tokens = tokenizer.encode("Hello world this is a test")
        expected = tokens.tolist()

        tensor = torch.from_dlpack(tokens)

        assert tensor.tolist() == expected

    @pytest.mark.requires_model
    def test_dlpack_dtype_uint32(self, tokenizer):
        """DLPack export has uint32 dtype (torch interprets as int32 unsigned)."""
        tokens = tokenizer.encode("Hello world")

        tensor = torch.from_dlpack(tokens)

        # PyTorch doesn't have native uint32, but DLPack maps to 32-bit
        assert tensor.dtype == torch.uint32

    @pytest.mark.requires_model
    def test_dlpack_shape_1d(self, tokenizer):
        """DLPack export is 1D tensor with correct length."""
        tokens = tokenizer.encode("Hello world test")

        tensor = torch.from_dlpack(tokens)

        assert tensor.dim() == 1
        assert tensor.shape[0] == len(tokens)

    @pytest.mark.requires_model
    def test_dlpack_pointer_equality(self, tokenizer):
        """torch tensor data_ptr equals TokenArray buffer address (zero-copy proof)."""
        tokens = tokenizer.encode("Hello world pointer test")

        tensor = torch.from_dlpack(tokens)

        # Get TokenArray's underlying buffer address via __array_interface__
        array_interface = tokens.__array_interface__
        token_array_ptr = array_interface["data"][0]

        # Get torch tensor's data pointer
        tensor_ptr = tensor.data_ptr()

        assert tensor_ptr == token_array_ptr, (
            f"Pointers differ: tensor={tensor_ptr:#x}, tokens={token_array_ptr:#x}. "
            "DLPack export is not zero-copy!"
        )


class TestTokenArrayDLPackLifetime:
    """Tests for TokenArray DLPack lifetime safety (refcount correctness)."""

    @pytest.mark.requires_model
    def test_tensor_survives_tokenarray_deletion(self, tokenizer):
        """Tensor remains valid after TokenArray is deleted.

        This proves the refcounted SharedBuffer keeps memory alive.
        """
        tokens = tokenizer.encode("Lifetime test with some words")
        expected = tokens.tolist()

        tensor = torch.from_dlpack(tokens)

        # Delete TokenArray reference
        del tokens
        gc.collect()
        gc.collect()

        # Tensor should still have valid data
        assert tensor.tolist() == expected

    @pytest.mark.requires_model
    def test_multiple_exports_safe(self, tokenizer):
        """Multiple DLPack exports from same TokenArray are safe."""
        tokens = tokenizer.encode("Multiple exports test")
        expected = tokens.tolist()

        t1 = torch.from_dlpack(tokens)
        t2 = torch.from_dlpack(tokens)
        t3 = torch.from_dlpack(tokens)

        # All tensors should have same data
        assert t1.tolist() == expected
        assert t2.tolist() == expected
        assert t3.tolist() == expected

        # All tensors should share same memory (same pointer)
        assert t1.data_ptr() == t2.data_ptr() == t3.data_ptr()

    @pytest.mark.requires_model
    def test_tensor_survives_after_multiple_exports_and_deletions(self, tokenizer):
        """Tensor survives complex lifecycle with multiple exports/deletions."""
        tokens = tokenizer.encode("Complex lifecycle test here")
        expected = tokens.tolist()

        t1 = torch.from_dlpack(tokens)
        t2 = torch.from_dlpack(tokens)

        # Delete original TokenArray
        del tokens
        gc.collect()

        # Delete one tensor
        del t1
        gc.collect()

        # Remaining tensor should still be valid
        assert t2.tolist() == expected

    @pytest.mark.requires_model
    def test_all_references_can_be_deleted_safely(self, tokenizer):
        """Deleting all references (TokenArray + tensors) doesn't crash."""
        tokens = tokenizer.encode("Safe deletion test")

        t1 = torch.from_dlpack(tokens)
        t2 = torch.from_dlpack(tokens)

        del tokens
        del t1
        del t2
        gc.collect()
        gc.collect()

        # If we get here without crashing, the test passes
        assert True


class TestTokenArrayDLPackRefcount:
    """Tests for observable refcount behavior."""

    @pytest.mark.requires_model
    def test_refcount_increments_on_export(self, tokenizer):
        """Refcount increases with each DLPack export."""
        from talu._bindings import get_lib

        lib = get_lib()
        tokens = tokenizer.encode("Refcount test")

        if not tokens._buffer_handle:
            pytest.skip("TokenArray has no buffer handle")

        # Initial refcount should be 1
        initial_refcount = lib.talu_buffer_get_refcount(tokens._buffer_handle)
        assert initial_refcount >= 1

        # Export to torch
        t1 = torch.from_dlpack(tokens)
        refcount_after_1 = lib.talu_buffer_get_refcount(tokens._buffer_handle)
        assert refcount_after_1 == initial_refcount + 1

        # Second export
        t2 = torch.from_dlpack(tokens)
        refcount_after_2 = lib.talu_buffer_get_refcount(tokens._buffer_handle)
        assert refcount_after_2 == initial_refcount + 2

        # Clean up to avoid interfering with other tests
        del t1, t2

    @pytest.mark.requires_model
    def test_refcount_decrements_on_tensor_deletion(self, tokenizer):
        """Refcount decreases when tensors are deleted."""
        from talu._bindings import get_lib

        lib = get_lib()
        tokens = tokenizer.encode("Refcount decrement test")

        if not tokens._buffer_handle:
            pytest.skip("TokenArray has no buffer handle")

        initial_refcount = lib.talu_buffer_get_refcount(tokens._buffer_handle)

        t1 = torch.from_dlpack(tokens)
        t2 = torch.from_dlpack(tokens)

        refcount_with_tensors = lib.talu_buffer_get_refcount(tokens._buffer_handle)
        assert refcount_with_tensors == initial_refcount + 2

        # Delete one tensor
        del t1
        gc.collect()

        refcount_after_del = lib.talu_buffer_get_refcount(tokens._buffer_handle)
        assert refcount_after_del == initial_refcount + 1

        # Delete second tensor
        del t2
        gc.collect()

        final_refcount = lib.talu_buffer_get_refcount(tokens._buffer_handle)
        assert final_refcount == initial_refcount


class TestTokenArrayDLPackMutation:
    """Tests demonstrating shared storage mutation semantics.

    NOTE: PyTorch has limited uint32 support (no arithmetic ops).
    We use direct memory writes via numpy to demonstrate mutation.
    """

    @pytest.mark.requires_model
    def test_tensor_mutation_visible_in_tokenarray(self, tokenizer):
        """Mutating the tensor affects TokenArray (shared storage).

        This documents the actual behavior - mutation IS visible.
        The API treats tokens as immutable, but the storage is shared.
        """
        pytest.importorskip("numpy")

        tokens = tokenizer.encode("Mutation test")
        original_first = tokens[0]

        tensor = torch.from_dlpack(tokens)

        # Use numpy for mutation (PyTorch uint32 arithmetic not supported)
        np_view = tensor.numpy()
        np_view[0] = np_view[0] + 1

        # Change should be visible in TokenArray
        assert tokens[0] == original_first + 1

        # Restore original value to not affect other tests
        np_view[0] = original_first

    @pytest.mark.requires_model
    def test_mutation_visible_across_exports(self, tokenizer):
        """Mutation in one tensor is visible in all exports (same buffer)."""
        pytest.importorskip("numpy")

        tokens = tokenizer.encode("Multi mutation test")
        original_first = tokens[0]

        t1 = torch.from_dlpack(tokens)
        t2 = torch.from_dlpack(tokens)

        # Mutate via numpy (PyTorch uint32 arithmetic not supported)
        np_view = t1.numpy()
        np_view[0] = np_view[0] + 100

        # Visible in t2 and tokens
        assert t2[0].item() == original_first + 100
        assert tokens[0] == original_first + 100

        # Restore
        np_view[0] = original_first
