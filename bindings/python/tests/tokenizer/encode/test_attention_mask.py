"""
Attention mask API tests.

Pure Python tests for the batch.attention_mask property and DLPack accessors:
- DLPack accessor protocol
- Dictionary protocol on BatchEncoding
- __call__ behavior

NOTE: torch/numpy integration tests are in tests/reference/tokenizer/.
"""

import pytest

import talu
from talu.tokenizer.batch import BatchEncoding

# =============================================================================
# Attention Mask DLPack Accessor Tests
# =============================================================================


class TestAttentionMaskAccessor:
    """Tests for the attention_mask DLPack accessor."""

    @pytest.mark.requires_model
    def test_attention_mask_property_exists(self, tokenizer):
        """BatchEncoding has attention_mask property."""
        batch = tokenizer.encode(["Hello", "World"])
        assert hasattr(batch, "attention_mask")

    @pytest.mark.requires_model
    def test_attention_mask_accessor_repr(self, tokenizer):
        """AttentionMaskAccessor has meaningful repr."""
        batch = tokenizer.encode(["Hello", "World"])
        accessor = batch.attention_mask
        assert "AttentionMaskAccessor" in repr(accessor)
        assert "num_sequences=2" in repr(accessor)

    @pytest.mark.requires_model
    def test_attention_mask_has_dlpack(self, tokenizer):
        """AttentionMaskAccessor implements __dlpack__ and __dlpack_device__."""
        batch = tokenizer.encode(["Hello", "World"])
        accessor = batch.attention_mask
        assert hasattr(accessor, "__dlpack__")
        assert hasattr(accessor, "__dlpack_device__")

    @pytest.mark.requires_model
    def test_attention_mask_device_is_cpu(self, tokenizer):
        """AttentionMaskAccessor reports CPU device."""
        batch = tokenizer.encode(["Hello", "World"])
        accessor = batch.attention_mask
        device = accessor.__dlpack_device__()
        assert device == (1, 0)  # (kDLCPU, device_id=0)

    @pytest.mark.requires_model
    def test_attention_mask_dlpack_returns_capsule(self, tokenizer):
        """AttentionMaskAccessor.__dlpack__() returns PyCapsule."""
        batch = tokenizer.encode(["Hello", "World"])
        accessor = batch.attention_mask
        capsule = accessor.__dlpack__()
        assert type(capsule).__name__ == "PyCapsule"


# =============================================================================
# Input IDs DLPack Accessor Tests
# =============================================================================


class TestInputIdsAccessor:
    """Tests for the input_ids DLPack accessor."""

    @pytest.mark.requires_model
    def test_input_ids_property_exists(self, tokenizer):
        """BatchEncoding has input_ids property."""
        batch = tokenizer.encode(["Hello", "World"])
        assert hasattr(batch, "input_ids")

    @pytest.mark.requires_model
    def test_input_ids_accessor_repr(self, tokenizer):
        """InputIdsAccessor has meaningful repr."""
        batch = tokenizer.encode(["Hello", "World"])
        accessor = batch.input_ids
        assert "InputIdsAccessor" in repr(accessor)
        assert "num_sequences=2" in repr(accessor)

    @pytest.mark.requires_model
    def test_input_ids_has_dlpack(self, tokenizer):
        """InputIdsAccessor implements __dlpack__ and __dlpack_device__."""
        batch = tokenizer.encode(["Hello", "World"])
        accessor = batch.input_ids
        assert hasattr(accessor, "__dlpack__")
        assert hasattr(accessor, "__dlpack_device__")

    @pytest.mark.requires_model
    def test_input_ids_device_is_cpu(self, tokenizer):
        """InputIdsAccessor reports CPU device."""
        batch = tokenizer.encode(["Hello", "World"])
        accessor = batch.input_ids
        device = accessor.__dlpack_device__()
        assert device == (1, 0)  # (kDLCPU, device_id=0)

    @pytest.mark.requires_model
    def test_input_ids_dlpack_returns_capsule(self, tokenizer):
        """InputIdsAccessor.__dlpack__() returns PyCapsule."""
        batch = tokenizer.encode(["Hello", "World"])
        accessor = batch.input_ids
        capsule = accessor.__dlpack__()
        assert type(capsule).__name__ == "PyCapsule"


# =============================================================================
# BatchEncoding DLPack Raises TypeError Tests
# =============================================================================


class TestBatchEncodingDLPackRaisesTypeError:
    """Tests that BatchEncoding.__dlpack__() raises TypeError.

    BatchEncoding contains multiple tensors (input_ids, attention_mask).
    Calling __dlpack__() directly is ambiguous and should raise TypeError
    to guide users to use explicit accessors.
    """

    @pytest.mark.requires_model
    def test_dlpack_raises_typeerror(self, tokenizer):
        """BatchEncoding.__dlpack__() raises TypeError."""
        batch = tokenizer.encode(["Hello", "World"])
        with pytest.raises(TypeError, match="multiple tensors"):
            batch.__dlpack__()

    @pytest.mark.requires_model
    def test_dlpack_error_message_helpful(self, tokenizer):
        """Error message guides users to use accessors."""
        batch = tokenizer.encode(["Hello", "World"])
        try:
            batch.__dlpack__()
            pytest.fail("Should have raised TypeError")
        except TypeError as e:
            msg = str(e)
            assert "input_ids" in msg
            assert "attention_mask" in msg
            assert "batch.input_ids" in msg or "batch[" in msg

    @pytest.mark.requires_model
    def test_dlpack_device_still_works(self, tokenizer):
        """__dlpack_device__() still returns valid device tuple."""
        batch = tokenizer.encode(["Hello", "World"])
        device = batch.__dlpack_device__()
        assert device == (1, 0)  # kDLCPU = 1, device_id = 0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestBatchEncodingErrors:
    """Tests for error handling in BatchEncoding DLPack export."""

    @pytest.mark.requires_model
    def test_empty_batch_input_ids_raises(self, tokenizer):
        """Empty batch raises InteropError on input_ids export."""
        batch = tokenizer.encode([])
        with pytest.raises(talu.InteropError, match="empty"):
            batch.input_ids.__dlpack__()

    @pytest.mark.requires_model
    def test_empty_batch_attention_mask_raises(self, tokenizer):
        """Empty batch raises InteropError on attention_mask export."""
        batch = tokenizer.encode([])
        with pytest.raises(talu.InteropError, match="empty"):
            batch.attention_mask.__dlpack__()


# =============================================================================
# Dictionary Protocol Tests
# =============================================================================


class TestBatchEncodingDictProtocol:
    """Tests for dict-like access on BatchEncoding."""

    @pytest.mark.requires_model
    def test_getitem_input_ids(self, tokenizer):
        """batch['input_ids'] returns DLPack-compatible accessor."""
        batch = tokenizer.encode(["Hello", "World"])
        result = batch["input_ids"]
        assert hasattr(result, "__dlpack__")
        assert hasattr(result, "__dlpack_device__")

    @pytest.mark.requires_model
    def test_getitem_attention_mask(self, tokenizer):
        """batch['attention_mask'] returns DLPack-compatible accessor."""
        batch = tokenizer.encode(["Hello", "World"])
        result = batch["attention_mask"]
        assert hasattr(result, "__dlpack__")
        assert hasattr(result, "__dlpack_device__")

    @pytest.mark.requires_model
    def test_getitem_invalid_key(self, tokenizer):
        """Invalid key raises KeyError."""
        batch = tokenizer.encode(["Hello", "World"])
        with pytest.raises(KeyError) as exc_info:
            _ = batch["invalid_key"]
        assert "invalid_key" in str(exc_info.value)

    @pytest.mark.requires_model
    def test_getitem_integer_still_works(self, tokenizer):
        """Integer indexing still works for sequence access."""
        batch = tokenizer.encode(["Hello", "World"])
        first = batch[0]
        assert len(first) > 0  # Has tokens

    @pytest.mark.requires_model
    def test_contains_input_ids(self, tokenizer):
        """'input_ids' in batch returns True."""
        batch = tokenizer.encode(["Hello", "World"])
        assert "input_ids" in batch

    @pytest.mark.requires_model
    def test_contains_attention_mask(self, tokenizer):
        """'attention_mask' in batch returns True."""
        batch = tokenizer.encode(["Hello", "World"])
        assert "attention_mask" in batch

    @pytest.mark.requires_model
    def test_contains_invalid_key(self, tokenizer):
        """Invalid key not in batch."""
        batch = tokenizer.encode(["Hello", "World"])
        assert "invalid_key" not in batch

    @pytest.mark.requires_model
    def test_keys(self, tokenizer):
        """batch.keys() returns ['input_ids', 'attention_mask']."""
        batch = tokenizer.encode(["Hello", "World"])
        assert batch.keys() == ["input_ids", "attention_mask"]


# =============================================================================
# __call__ Tests
# =============================================================================


class TestTokenizerCallBehavior:
    """Tests for tokenizer() returning BatchEncoding."""

    @pytest.mark.requires_model
    def test_call_returns_batch_encoding(self, tokenizer):
        """tokenizer() returns BatchEncoding, not dict."""
        batch = tokenizer(["Hello", "World"])
        assert isinstance(batch, BatchEncoding)

    @pytest.mark.requires_model
    def test_call_single_string_returns_batch(self, tokenizer):
        """Single string is wrapped in BatchEncoding with 1 sequence."""
        batch = tokenizer("Hello")
        assert isinstance(batch, BatchEncoding)
        assert len(batch) == 1

    @pytest.mark.requires_model
    def test_call_dict_access_works(self, tokenizer):
        """Dict-like access works on __call__ result."""
        batch = tokenizer(["Hello", "World"])
        assert "input_ids" in batch
        assert "attention_mask" in batch

    @pytest.mark.requires_model
    def test_call_inherits_tokenizer_config(self, tokenizer):
        """__call__ result inherits tokenizer padding_side."""
        batch = tokenizer(["Hello", "World"])
        assert batch.padding_side == tokenizer.padding_side

    @pytest.mark.requires_model
    def test_call_special_tokens_false(self, tokenizer):
        """special_tokens=False is respected."""
        batch_with = tokenizer("Hello", special_tokens=True)
        batch_without = tokenizer("Hello", special_tokens=False)

        # Without special tokens should have fewer tokens
        assert len(batch_without[0]) <= len(batch_with[0])

    @pytest.mark.requires_model
    def test_call_input_ids_is_accessor(self, tokenizer):
        """tokenizer()['input_ids'] returns DLPack-compatible accessor."""
        output = tokenizer("Hello world")
        assert hasattr(output["input_ids"], "__dlpack__")

    @pytest.mark.requires_model
    def test_call_attention_mask_is_accessor(self, tokenizer):
        """tokenizer()['attention_mask'] returns DLPack-compatible accessor."""
        output = tokenizer("Hello world")
        assert hasattr(output["attention_mask"], "__dlpack__")
