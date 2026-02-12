"""
BatchEncoding DLPack torch integration tests.

These tests validate BatchEncoding accessor DLPack export with PyTorch:
- input_ids accessor exports correct 2D padded tensor
- attention_mask accessor exports correct mask aligned with padding
- Left vs right padding alignment
- Repeated exports are stable

NOTE: BatchEncoding.__dlpack__() raises TypeError (ambiguous).
Use explicit accessors: batch.input_ids and batch.attention_mask.

NOTE: Unlike TokenArray, BatchEncoding DLPack export ALLOCATES new
padded buffers (not zero-copy). This is intentional - the internal
CSR storage must be materialized to dense padded format.
"""

import gc

import pytest

torch = pytest.importorskip("torch")


class TestBatchEncodingInputIdsDLPack:
    """Tests for batch.input_ids DLPack export."""

    @pytest.mark.requires_model
    def test_input_ids_shape_2d(self, tokenizer):
        """input_ids exports as 2D tensor [batch_size, padded_length]."""
        batch = tokenizer.encode(["Hello", "Hello world test"])

        tensor = torch.from_dlpack(batch.input_ids)

        assert tensor.dim() == 2
        assert tensor.shape[0] == 2  # batch_size

    @pytest.mark.requires_model
    def test_input_ids_padded_length_is_max(self, tokenizer):
        """Padded length equals longest sequence."""
        texts = ["Hi", "Hello world this is longer"]
        batch = tokenizer.encode(texts)

        tensor = torch.from_dlpack(batch.input_ids)

        # Get individual lengths
        lengths = [len(tokenizer.encode(t)) for t in texts]
        expected_padded_len = max(lengths)

        assert tensor.shape[1] == expected_padded_len

    @pytest.mark.requires_model
    def test_input_ids_values_match_sequences(self, tokenizer):
        """Token values in tensor match encoded sequences."""
        texts = ["Hello", "World"]
        batch = tokenizer.encode(texts)

        tensor = torch.from_dlpack(batch.input_ids)

        # Check each sequence
        for i, text in enumerate(texts):
            expected_tokens = tokenizer.encode(text).tolist()
            row = tensor[i].tolist()

            # Tokens should appear (with possible padding)
            # For right padding: tokens at start
            # For left padding: tokens at end
            if batch.padding_side == "right":
                assert row[: len(expected_tokens)] == expected_tokens
            else:
                assert row[-len(expected_tokens) :] == expected_tokens

    @pytest.mark.requires_model
    def test_input_ids_dtype(self, tokenizer):
        """input_ids tensor has uint32 dtype."""
        batch = tokenizer.encode(["Hello", "World"])

        tensor = torch.from_dlpack(batch.input_ids)

        assert tensor.dtype == torch.uint32


class TestBatchEncodingAttentionMaskDLPack:
    """Tests for batch.attention_mask DLPack export."""

    @pytest.mark.requires_model
    def test_attention_mask_shape_matches_input_ids(self, tokenizer):
        """attention_mask shape equals input_ids shape."""
        batch = tokenizer.encode(["Hello", "Hello world test"])

        input_ids = torch.from_dlpack(batch.input_ids)
        mask = torch.from_dlpack(batch.attention_mask)

        assert mask.shape == input_ids.shape

    @pytest.mark.requires_model
    def test_attention_mask_values_binary(self, tokenizer):
        """attention_mask contains only 0 and 1."""
        batch = tokenizer.encode(["Hi", "Hello world test longer"])

        mask = torch.from_dlpack(batch.attention_mask)

        unique_values = torch.unique(mask).tolist()
        assert set(unique_values).issubset({0, 1})

    @pytest.mark.requires_model
    def test_attention_mask_dtype(self, tokenizer):
        """attention_mask tensor has int32 dtype."""
        batch = tokenizer.encode(["Hello", "World"])

        mask = torch.from_dlpack(batch.attention_mask)

        assert mask.dtype == torch.int32

    @pytest.mark.requires_model
    def test_mask_zeros_align_with_padding(self, tokenizer):
        """Mask zeros correspond to pad positions."""
        texts = ["Hi", "Hello world this is much longer"]
        batch = tokenizer.encode(texts)

        # Convert to int64 for PyTorch indexing (uint32 indexing not supported)
        input_ids = torch.from_dlpack(batch.input_ids).to(torch.int64)
        mask = torch.from_dlpack(batch.attention_mask)

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            # Use EOS as pad fallback (common pattern)
            pad_id = tokenizer.primary_eos_token_id()

        # Where mask is 0, input_ids should be pad_id
        pad_positions = mask == 0
        if pad_positions.any():
            pad_values = input_ids[pad_positions]
            assert all(v == pad_id for v in pad_values.tolist()), (
                f"Pad positions have non-pad tokens. pad_id={pad_id}, found={pad_values.unique().tolist()}"
            )


class TestBatchEncodingPaddingAlignment:
    """Tests for left vs right padding alignment."""

    @pytest.mark.requires_model
    def test_right_padding_tokens_at_start(self, tokenizer):
        """With right padding, real tokens are at start of each row."""
        tokenizer.padding_side = "right"
        texts = ["Hi", "Hello world longer text here"]
        batch = tokenizer.encode(texts)

        mask = torch.from_dlpack(batch.attention_mask)

        # For right padding, first row should have 1s at start, 0s at end
        short_mask = mask[0].tolist()
        short_len = len(tokenizer.encode(texts[0]))

        # First short_len positions should be 1
        assert all(v == 1 for v in short_mask[:short_len])
        # Remaining should be 0 (if any)
        if len(short_mask) > short_len:
            assert all(v == 0 for v in short_mask[short_len:])

    @pytest.mark.requires_model
    def test_left_padding_tokens_at_end(self, tokenizer):
        """With left padding, real tokens are at end of each row."""
        tokenizer.padding_side = "left"
        texts = ["Hi", "Hello world longer text here"]
        batch = tokenizer.encode(texts)

        mask = torch.from_dlpack(batch.attention_mask)

        # For left padding, first row should have 0s at start, 1s at end
        short_mask = mask[0].tolist()
        short_len = len(tokenizer.encode(texts[0]))
        pad_len = len(short_mask) - short_len

        # First pad_len positions should be 0
        if pad_len > 0:
            assert all(v == 0 for v in short_mask[:pad_len])
        # Last short_len positions should be 1
        assert all(v == 1 for v in short_mask[pad_len:])

        # Reset to default
        tokenizer.padding_side = "right"


class TestBatchEncodingRepeatedExports:
    """Tests for repeated DLPack exports."""

    @pytest.mark.requires_model
    def test_repeated_input_ids_exports_consistent(self, tokenizer):
        """Multiple input_ids exports produce same values."""
        batch = tokenizer.encode(["Hello", "World test"])

        t1 = torch.from_dlpack(batch.input_ids)
        t2 = torch.from_dlpack(batch.input_ids)
        t3 = torch.from_dlpack(batch.input_ids)

        assert torch.equal(t1, t2)
        assert torch.equal(t2, t3)

    @pytest.mark.requires_model
    def test_repeated_mask_exports_consistent(self, tokenizer):
        """Multiple attention_mask exports produce same values."""
        batch = tokenizer.encode(["Hello", "World test"])

        m1 = torch.from_dlpack(batch.attention_mask)
        m2 = torch.from_dlpack(batch.attention_mask)
        m3 = torch.from_dlpack(batch.attention_mask)

        assert torch.equal(m1, m2)
        assert torch.equal(m2, m3)

    @pytest.mark.requires_model
    def test_batch_not_mutated_by_export(self, tokenizer):
        """DLPack export doesn't mutate the batch."""
        batch = tokenizer.encode(["Hello", "World"])

        # Capture state before
        len_before = len(batch)
        seq0_before = batch[0].tolist()

        # Export multiple times
        _ = torch.from_dlpack(batch.input_ids)
        _ = torch.from_dlpack(batch.attention_mask)
        _ = torch.from_dlpack(batch.input_ids)

        # State should be unchanged
        assert len(batch) == len_before
        assert batch[0].tolist() == seq0_before


class TestBatchEncodingDLPackRaisesTypeError:
    """Tests that BatchEncoding.__dlpack__() correctly raises TypeError."""

    @pytest.mark.requires_model
    def test_batch_dlpack_raises_typeerror(self, tokenizer):
        """BatchEncoding.__dlpack__() raises TypeError (ambiguous)."""
        batch = tokenizer.encode(["Hello", "World"])

        with pytest.raises(TypeError, match="multiple tensors"):
            torch.from_dlpack(batch)

    @pytest.mark.requires_model
    def test_error_message_guides_to_accessors(self, tokenizer):
        """Error message mentions input_ids and attention_mask."""
        batch = tokenizer.encode(["Hello", "World"])

        try:
            batch.__dlpack__()
            pytest.fail("Should have raised TypeError")
        except TypeError as e:
            msg = str(e)
            assert "input_ids" in msg
            assert "attention_mask" in msg


class TestBatchEncodingDLPackIsNotZeroCopy:
    """Tests documenting that BatchEncoding export allocates (not zero-copy).

    This is intentional: internal CSR storage must be materialized to dense.
    """

    @pytest.mark.requires_model
    def test_repeated_exports_different_pointers(self, tokenizer):
        """Each export allocates a new buffer (different data_ptr)."""
        batch = tokenizer.encode(["Hello", "World test longer"])

        t1 = torch.from_dlpack(batch.input_ids)
        t2 = torch.from_dlpack(batch.input_ids)

        # Pointers should be DIFFERENT (new allocation each time)
        # This is expected behavior - not a bug
        assert t1.data_ptr() != t2.data_ptr(), (
            "Expected different pointers (new allocation each export). "
            "If same pointer, implementation may have changed to caching."
        )

    @pytest.mark.requires_model
    def test_tensor_survives_batch_deletion(self, tokenizer):
        """Exported tensor remains valid after batch deletion.

        Even though not zero-copy, the exported tensor owns its memory.
        """
        batch = tokenizer.encode(["Hello", "World"])
        expected = torch.from_dlpack(batch.input_ids).clone()

        tensor = torch.from_dlpack(batch.input_ids)

        del batch
        gc.collect()

        # Tensor should still be valid (owns its memory)
        assert torch.equal(tensor, expected)
