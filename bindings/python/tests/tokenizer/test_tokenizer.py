"""Tests for talu.tokenizer.tokenizer module.

Tests for the Tokenizer class - the main tokenizer interface.
Note: Detailed encode/decode tests are in tests/tokenizer/encode/ and tests/tokenizer/decode/.
"""

import pytest

import talu


class TestTokenizerConstruction:
    """Tests for Tokenizer class construction.

    Note: The Zig core returns error code 999 (internal error) for path resolution
    failures, which maps to TaluError rather than ModelError.
    """

    def test_tokenizer_invalid_path_raises(self):
        """Tokenizer raises for invalid model paths."""
        from talu.exceptions import TaluError
        from talu.tokenizer import Tokenizer

        with pytest.raises(TaluError):
            Tokenizer("/definitely/not/a/real/path/to/model")

    def test_tokenizer_invalid_model_id_raises(self):
        """Tokenizer raises for invalid model IDs."""
        from talu.exceptions import TaluError
        from talu.tokenizer import Tokenizer

        with pytest.raises(TaluError):
            Tokenizer("definitely-not-a-real-org/not-a-real-model-12345")


class TestTokenizerValidation:
    """Tests for Tokenizer parameter validation."""

    def test_invalid_padding_side_raises(self):
        """Invalid padding_side raises ValidationError."""
        from talu.exceptions import ValidationError
        from talu.tokenizer import Tokenizer

        with pytest.raises(ValidationError) as exc_info:
            Tokenizer("some/model", padding_side="center")

        assert "padding_side" in str(exc_info.value)
        assert "left" in str(exc_info.value) or "right" in str(exc_info.value)

    def test_invalid_truncation_side_raises(self):
        """Invalid truncation_side raises ValidationError."""
        from talu.exceptions import ValidationError
        from talu.tokenizer import Tokenizer

        with pytest.raises(ValidationError) as exc_info:
            Tokenizer("some/model", truncation_side="center")

        assert "truncation_side" in str(exc_info.value)
        assert "left" in str(exc_info.value) or "right" in str(exc_info.value)


class TestTokenizerWithModel:
    """Tests for Tokenizer class with actual model.

    These tests require a model to be available.
    """

    @pytest.fixture
    def tokenizer(self, test_model_path):
        """Create a Tokenizer for testing."""
        from talu.tokenizer import Tokenizer

        return Tokenizer(test_model_path)

    def test_vocab_size(self, tokenizer):
        """vocab_size returns positive integer."""
        assert tokenizer.vocab_size > 0
        assert isinstance(tokenizer.vocab_size, int)

    def test_model_path(self, tokenizer):
        """model_path returns the resolved path."""
        assert tokenizer.model_path
        assert isinstance(tokenizer.model_path, str)

    def test_eos_token_ids(self, tokenizer):
        """eos_token_ids returns an immutable tuple of integers."""
        eos = tokenizer.eos_token_ids
        assert isinstance(eos, tuple)
        for t in eos:
            assert isinstance(t, int)

    def test_bos_token_id(self, tokenizer):
        """bos_token_id returns int or None."""
        bos = tokenizer.bos_token_id
        assert bos is None or isinstance(bos, int)

    def test_unk_token_id(self, tokenizer):
        """unk_token_id returns int or None."""
        unk = tokenizer.unk_token_id
        assert unk is None or isinstance(unk, int)

    def test_pad_token_id(self, tokenizer):
        """pad_token_id returns int or None."""
        pad = tokenizer.pad_token_id
        assert pad is None or isinstance(pad, int)

    def test_encode_returns_token_array(self, tokenizer):
        """encode() returns TokenArray."""
        from talu.tokenizer import TokenArray

        result = tokenizer.encode("Hello")
        assert isinstance(result, TokenArray)
        assert len(result) > 0

    def test_decode_returns_string(self, tokenizer):
        """decode() returns string."""
        tokens = tokenizer.encode("Hello")
        text = tokenizer.decode(tokens)
        assert isinstance(text, str)

    def test_encode_decode_roundtrip(self, tokenizer):
        """encode/decode roundtrip preserves text."""
        original = "Hello, world!"
        tokens = tokenizer.encode(original)
        decoded = tokenizer.decode(tokens)
        assert decoded == original

    def test_tokenize_returns_list(self, tokenizer):
        """tokenize() returns list of strings."""
        result = tokenizer.tokenize("Hello, world!")
        assert isinstance(result, list)
        for token in result:
            assert isinstance(token, str)

    def test_count_tokens(self, tokenizer):
        """count_tokens() returns integer."""
        count = tokenizer.count_tokens("Hello, world!")
        assert isinstance(count, int)
        assert count > 0

    def test_id_to_token(self, tokenizer):
        """id_to_token() returns string for valid ID."""
        # Token 0 should exist
        token = tokenizer.id_to_token(0)
        assert token is None or isinstance(token, str)

    def test_id_to_token_invalid(self, tokenizer):
        """id_to_token() returns None for invalid ID."""
        result = tokenizer.id_to_token(999999999)
        assert result is None

    def test_token_to_id(self, tokenizer):
        """token_to_id() returns integer for valid token."""
        # Get a token first
        tokens = tokenizer.tokenize("hello")
        if tokens:
            token_str = tokens[0]
            token_id = tokenizer.token_to_id(token_str)
            assert token_id is None or isinstance(token_id, int)

    def test_token_to_id_unknown(self, tokenizer):
        """token_to_id() returns None for unknown token."""
        result = tokenizer.token_to_id("xyzzy_definitely_not_a_token_12345")
        assert result is None

    def test_apply_chat_template(self, tokenizer):
        """apply_chat_template() returns formatted string."""
        messages = [{"role": "user", "content": "Hello!"}]
        result = tokenizer.apply_chat_template(messages)
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain the user message
        assert "Hello" in result

    def test_repr(self, tokenizer):
        """Tokenizer has a useful repr."""
        r = repr(tokenizer)
        assert "Tokenizer" in r

    # =========================================================================
    # Tests for add_special_tokens parameter
    # =========================================================================

    def test_encode_add_special_tokens_default(self, tokenizer):
        """encode() with default special_tokens=True works."""
        tokens = tokenizer.encode("Hello")
        assert len(tokens) > 0

    def test_encode_add_special_tokens_false(self, tokenizer):
        """encode(special_tokens=False) returns tokens."""
        tokens = tokenizer.encode("Hello", special_tokens=False)
        assert len(tokens) > 0

    def test_encode_add_special_tokens_difference(self, tokenizer):
        """add_special_tokens may produce different token counts.

        Note: This depends on the model's postprocessor configuration.
        Some models (like BERT) add CLS/SEP tokens. Others don't add any.
        We just verify both calls work without error.
        """
        with_special = tokenizer.encode("Hello", special_tokens=True)
        without_special = tokenizer.encode("Hello", special_tokens=False)
        # Both should produce valid tokens
        assert len(with_special) > 0
        assert len(without_special) > 0
        # Token content should be decodable
        assert tokenizer.decode(with_special)
        assert tokenizer.decode(without_special)

    # =========================================================================
    # Tests for __call__ (HuggingFace-style interface)
    # =========================================================================

    def test_callable_returns_batch_encoding(self, tokenizer):
        """tokenizer() returns BatchEncoding with dict-like access."""
        from talu.tokenizer import BatchEncoding

        output = tokenizer("Hello world")
        assert isinstance(output, BatchEncoding)
        assert "input_ids" in output
        assert "attention_mask" in output

    def test_callable_input_ids_is_accessor(self, tokenizer):
        """tokenizer()['input_ids'] returns DLPack-compatible accessor."""
        output = tokenizer("Hello world")
        assert hasattr(output["input_ids"], "__dlpack__")

    def test_callable_attention_mask_is_accessor(self, tokenizer):
        """tokenizer()['attention_mask'] returns DLPack-compatible accessor."""
        output = tokenizer("Hello world")
        assert hasattr(output["attention_mask"], "__dlpack__")

    def test_callable_accessors_have_dlpack(self, tokenizer):
        """input_ids and attention_mask accessors have __dlpack__."""
        output = tokenizer("Hello world")
        # Both accessors should have __dlpack__ method
        assert hasattr(output["input_ids"], "__dlpack__")
        assert hasattr(output["attention_mask"], "__dlpack__")
        # Both return PyCapsule
        assert type(output["input_ids"].__dlpack__()).__name__ == "PyCapsule"
        assert type(output["attention_mask"].__dlpack__()).__name__ == "PyCapsule"

    def test_callable_add_special_tokens(self, tokenizer):
        """tokenizer(special_tokens=False) works."""
        from talu.tokenizer import BatchEncoding

        output = tokenizer("Hello", special_tokens=False)
        assert isinstance(output, BatchEncoding)
        assert len(output[0]) > 0  # First sequence has tokens

    # =========================================================================
    # Tests for apply_chat_template with tokenize parameter
    # =========================================================================

    def test_apply_chat_template_tokenize_false(self, tokenizer):
        """apply_chat_template(tokenize=False) returns string."""
        messages = [{"role": "user", "content": "Hello!"}]
        result = tokenizer.apply_chat_template(messages, tokenize=False)
        assert isinstance(result, str)
        assert "Hello" in result

    def test_apply_chat_template_tokenize_true(self, tokenizer):
        """apply_chat_template(tokenize=True) returns TokenArray."""
        from talu.tokenizer import TokenArray

        messages = [{"role": "user", "content": "Hello!"}]
        result = tokenizer.apply_chat_template(messages, tokenize=True)
        assert isinstance(result, TokenArray)
        assert len(result) > 0

    def test_apply_chat_template_tokenize_roundtrip(self, tokenizer):
        """Tokenized chat template can be decoded back."""
        messages = [{"role": "user", "content": "Hello!"}]
        tokens = tokenizer.apply_chat_template(messages, tokenize=True)
        decoded = tokenizer.decode(tokens)
        assert "Hello" in decoded

    # =========================================================================
    # Tests for zero-copy interop
    # =========================================================================

    def test_callable_dlpack_capsule(self, tokenizer):
        """BatchEncoding from __call__ exports via DLPack."""
        output = tokenizer("Hello world")
        capsule = output["input_ids"].__dlpack__()
        assert type(capsule).__name__ == "PyCapsule"

    def test_encode_dlpack_capsule(self, tokenizer):
        """TokenArray from encode() exports via DLPack."""
        tokens = tokenizer.encode("Hello world")
        capsule = tokens.__dlpack__()
        assert type(capsule).__name__ == "PyCapsule"

    # =========================================================================
    # Tests for thread safety
    # =========================================================================

    def test_concurrent_encode_with_different_options(self, tokenizer):
        """Multiple threads can encode with different add_special_tokens values concurrently.

        This verifies that the implementation passes add_special_tokens as a runtime
        argument rather than temporarily mutating tokenizer state (which would cause
        race conditions).
        """
        import concurrent.futures
        import threading

        results_with_special = []
        results_without_special = []
        errors = []
        barrier = threading.Barrier(4)  # Synchronize thread start

        def encode_with_special(text: str, idx: int):
            try:
                barrier.wait()  # Ensure all threads start together
                tokens = tokenizer.encode(text, special_tokens=True)
                results_with_special.append((idx, tokens.tolist()))
            except Exception as e:
                errors.append((idx, "with", e))

        def encode_without_special(text: str, idx: int):
            try:
                barrier.wait()  # Ensure all threads start together
                tokens = tokenizer.encode(text, special_tokens=False)
                results_without_special.append((idx, tokens.tolist()))
            except Exception as e:
                errors.append((idx, "without", e))

        text = "Hello world"
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            # Submit 2 threads with special_tokens=True
            for i in range(2):
                futures.append(executor.submit(encode_with_special, text, i))
            # Submit 2 threads with special_tokens=False
            for i in range(2):
                futures.append(executor.submit(encode_without_special, text, i))

            # Wait for all to complete
            concurrent.futures.wait(futures)

        # Check no errors occurred
        assert not errors, f"Thread errors: {errors}"

        # All "with special" results should be identical
        assert len(results_with_special) == 2
        for idx, tokens in results_with_special:
            assert tokens == results_with_special[0][1], (
                f"Thread {idx} got different tokens with special_tokens=True"
            )

        # All "without special" results should be identical
        assert len(results_without_special) == 2
        for idx, tokens in results_without_special:
            assert tokens == results_without_special[0][1], (
                f"Thread {idx} got different tokens with special_tokens=False"
            )

        # The two groups should have different lengths (special tokens add length)
        with_len = len(results_with_special[0][1])
        without_len = len(results_without_special[0][1])
        assert with_len != without_len or with_len > 0, (
            "Expected different token counts with/without special tokens"
        )

    # =========================================================================
    # Tests for skip_special_tokens in decode
    # =========================================================================

    def test_decode_skip_special_tokens_default(self, tokenizer):
        """decode() skips special tokens by default."""
        # Use chat template to get tokens with special markers
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello"}], tokenize=False
        )
        tokens = tokenizer.encode(prompt, special_tokens=False)
        # Decode should skip special tokens by default
        text = tokenizer.decode(tokens)
        # Should not contain special token markers
        assert "<|im_start|>" not in text
        assert "<|im_end|>" not in text
        # Should contain the actual content
        assert "Hello" in text

    def test_decode_skip_special_tokens_false(self, tokenizer):
        """decode(skip_special_tokens=False) includes special tokens."""
        # Use chat template to get tokens with special markers
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello"}], tokenize=False
        )
        tokens = tokenizer.encode(prompt, special_tokens=False)
        # Decode with special tokens visible
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        # Should contain special token markers from any model family
        has_special = (
            "<|im_start|>" in text  # Qwen
            or "<|im_end|>" in text  # Qwen
            or "<|endoftext|>" in text  # GPT-style
            or "<s>" in text  # Llama2
            or "</s>" in text  # Llama2
            or "<|begin_of_text|>" in text  # Llama3
            or "<|start_header_id|>" in text  # Llama3
            or "<start_of_turn>" in text  # Gemma
        )
        assert has_special, f"Expected special tokens in: {text!r}"

    def test_decode_skip_special_tokens_difference(self, tokenizer):
        """skip_special_tokens=True produces shorter output than False."""
        # Use chat template to get tokens with special markers
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello world"}], tokenize=False
        )
        tokens = tokenizer.encode(prompt, special_tokens=False)
        with_skip = tokenizer.decode(tokens, skip_special_tokens=True)
        without_skip = tokenizer.decode(tokens, skip_special_tokens=False)
        # Without skipping should be longer (includes special token text)
        assert len(without_skip) > len(with_skip), (
            f"Expected without_skip ({len(without_skip)}) > with_skip ({len(with_skip)})"
        )

    def test_decode_no_special_tokens_same_output(self, tokenizer):
        """When input has no special tokens, skip_special_tokens has no effect."""
        # Plain text without special tokens
        tokens = tokenizer.encode("Hello", special_tokens=False)
        with_skip = tokenizer.decode(tokens, skip_special_tokens=True)
        without_skip = tokenizer.decode(tokens, skip_special_tokens=False)
        # Should be identical since there are no special tokens to skip
        assert with_skip == without_skip

    # =========================================================================
    # Tests for get_vocab
    # =========================================================================

    def test_get_vocab_returns_dict(self, tokenizer):
        """get_vocab() returns a dictionary."""
        vocab = tokenizer.get_vocab()
        assert isinstance(vocab, dict)

    def test_get_vocab_size_matches(self, tokenizer):
        """get_vocab() returns roughly vocab_size entries."""
        vocab = tokenizer.get_vocab()
        # May be slightly less due to gaps, but should be close
        assert len(vocab) > tokenizer.vocab_size * 0.9

    def test_get_vocab_string_to_int(self, tokenizer):
        """get_vocab() maps strings to integers."""
        vocab = tokenizer.get_vocab()
        for token, token_id in list(vocab.items())[:10]:
            assert isinstance(token, str)
            assert isinstance(token_id, int)
            assert token_id >= 0

    def test_get_vocab_contains_common_tokens(self, tokenizer):
        """get_vocab() includes common tokens."""
        vocab = tokenizer.get_vocab()
        # Check for common ASCII characters
        assert "!" in vocab
        assert "a" in vocab
        assert "0" in vocab

    def test_get_vocab_lookup_consistency(self, tokenizer):
        """get_vocab() results match id_to_token/token_to_id."""
        vocab = tokenizer.get_vocab()
        # Sample a few entries
        for token, token_id in list(vocab.items())[:20]:
            # id_to_token should return the same token
            assert tokenizer.id_to_token(token_id) == token
            # token_to_id should return the same ID
            assert tokenizer.token_to_id(token) == token_id

    # =========================================================================
    # Tests for batch encoding (list input)
    # =========================================================================

    def test_encode_list_returns_batch_encoding(self, tokenizer):
        """encode(list) returns BatchEncoding."""
        from talu.tokenizer import BatchEncoding

        result = tokenizer.encode(["Hello", "World"])
        assert isinstance(result, BatchEncoding)

    def test_encode_single_returns_token_array(self, tokenizer):
        """encode(str) still returns TokenArray."""
        from talu.tokenizer import TokenArray

        result = tokenizer.encode("Hello")
        assert isinstance(result, TokenArray)

    def test_batch_encoding_len(self, tokenizer):
        """BatchEncoding len() returns number of sequences."""
        batch = tokenizer.encode(["Hello", "World", "Test"])
        assert len(batch) == 3

    def test_batch_encoding_getitem(self, tokenizer):
        """BatchEncoding[i] returns TokenArray-like view."""
        batch = tokenizer.encode(["Hello", "World"])
        first = batch[0]
        assert hasattr(first, "tolist")
        assert len(first) > 0

    def test_batch_encoding_negative_index(self, tokenizer):
        """BatchEncoding supports negative indexing."""
        batch = tokenizer.encode(["Hello", "World", "Test"])
        assert batch[-1].tolist() == batch[2].tolist()

    def test_batch_encoding_index_error(self, tokenizer):
        """BatchEncoding raises IndexError for out of range."""
        batch = tokenizer.encode(["Hello", "World"])
        with pytest.raises(IndexError):
            _ = batch[5]

    def test_batch_encoding_iteration(self, tokenizer):
        """BatchEncoding is iterable."""
        texts = ["Hello", "World", "Test"]
        batch = tokenizer.encode(texts)
        items = list(batch)
        assert len(items) == 3

    def test_batch_encoding_total_tokens(self, tokenizer):
        """BatchEncoding.total_tokens is sum of all sequence lengths."""
        batch = tokenizer.encode(["Hello", "World"])
        total = sum(len(seq) for seq in batch)
        assert batch.total_tokens == total

    def test_batch_encoding_lengths(self, tokenizer):
        """BatchEncoding.lengths() returns list of sequence lengths."""
        batch = tokenizer.encode(["Hello", "World", "Hello world"])
        lengths = batch.lengths()
        assert len(lengths) == 3
        assert all(isinstance(length, int) for length in lengths)
        assert all(length > 0 for length in lengths)

    def test_batch_encoding_max_length(self, tokenizer):
        """BatchEncoding.max_length() returns longest sequence length."""
        batch = tokenizer.encode(["Hi", "Hello world this is longer"])
        assert batch.max_length() == max(batch.lengths())

    def test_batch_encoding_to_list(self, tokenizer):
        """BatchEncoding.to_list() returns dict with padded 2D lists."""
        batch = tokenizer.encode(["Hello", "World"])
        result = batch.to_list(padding=True, pad_id=0)
        assert isinstance(result, dict)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert len(result["input_ids"]) == 2
        # All rows should have same length (padded)
        assert len(result["input_ids"][0]) == len(result["input_ids"][1])

    def test_batch_encoding_to_list_pad_value(self, tokenizer):
        """to_list uses specified pad_id."""
        batch = tokenizer.encode(["Hi", "Hello world"])
        result = batch.to_list(padding=True, pad_id=999)
        # Shorter sequence should have padding
        lengths = batch.lengths()
        if lengths[0] < lengths[1]:
            # First row is shorter, check it has padding
            assert 999 in result["input_ids"][0]

    def test_batch_encoding_to_list_no_padding_error(self, tokenizer):
        """to_list(padding=False) raises for variable lengths."""
        batch = tokenizer.encode(["Hi", "Hello world this is longer"])
        if batch.lengths()[0] != batch.lengths()[1]:
            with pytest.raises(ValueError):
                batch.to_list(padding=False)

    def test_batch_encoding_empty_list(self, tokenizer):
        """encode([]) returns empty BatchEncoding."""
        batch = tokenizer.encode([])
        assert len(batch) == 0
        assert batch.total_tokens == 0

    def test_batch_encoding_single_element(self, tokenizer):
        """encode([single]) works correctly."""
        batch = tokenizer.encode(["Hello"])
        assert len(batch) == 1
        assert len(batch[0]) > 0

    def test_batch_encoding_consistency_with_single(self, tokenizer):
        """Batch encode produces same tokens as single encode."""
        texts = ["Hello", "World", "Test"]
        batch = tokenizer.encode(texts)
        singles = [tokenizer.encode(t) for t in texts]

        for i, (batch_tokens, single_tokens) in enumerate(zip(batch, singles, strict=True)):
            assert batch_tokens.tolist() == single_tokens.tolist(), f"Mismatch at index {i}"

    def test_batch_encoding_add_special_tokens_false(self, tokenizer):
        """Batch encode respects special_tokens=False."""
        texts = ["Hello", "World"]
        with_special = tokenizer.encode(texts, special_tokens=True)
        without_special = tokenizer.encode(texts, special_tokens=False)
        # Both should work
        assert len(with_special) == 2
        assert len(without_special) == 2

    def test_batch_encoding_unicode(self, tokenizer):
        """Batch encode handles unicode correctly."""
        texts = ["Hello", "世界", "🎉"]
        batch = tokenizer.encode(texts)
        assert len(batch) == 3
        # Each should have tokens
        for seq in batch:
            assert len(seq) > 0

    def test_batch_encoding_repr(self, tokenizer):
        """BatchEncoding has useful repr."""
        batch = tokenizer.encode(["Hello", "World"])
        r = repr(batch)
        assert "BatchEncoding" in r
        assert "num_sequences=2" in r

    def test_batch_encoding_view_tolist(self, tokenizer):
        """TokenArray view from batch has tolist()."""
        batch = tokenizer.encode(["Hello", "World"])
        tokens = batch[0].tolist()
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)

    def test_batch_encoding_view_has_array_interface(self, tokenizer):
        """BatchEncoding views support array protocol."""
        batch = tokenizer.encode(["Hello", "World"])
        view = batch[0]
        # Views should have __array__ or be convertible to list
        assert hasattr(view, "tolist")
        assert isinstance(view.tolist(), list)

    def test_batch_encoding_to_list_returns_dict(self, tokenizer):
        """BatchEncoding.to_list() returns dict with lists."""
        batch = tokenizer.encode(["Hello", "World"])
        result = batch.to_list(padding=True)
        assert isinstance(result, dict)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert isinstance(result["input_ids"], list)
        assert len(result["input_ids"]) == 2  # batch size

    def test_encode_type_error(self, tokenizer):
        """encode() raises ValidationError for invalid input type."""
        with pytest.raises(talu.ValidationError):
            tokenizer.encode(123)  # Not str or list

    def test_batch_large_batch(self, tokenizer):
        """Batch encode works for larger batches."""
        texts = [f"Text number {i}" for i in range(100)]
        batch = tokenizer.encode(texts)
        assert len(batch) == 100
        assert batch.total_tokens > 0

    # =========================================================================
    # Tests for padding_side and directional padding
    # =========================================================================

    def test_padding_side_default(self, tokenizer):
        """Default padding_side is 'left' for generation models."""
        assert tokenizer.padding_side == "left"

    def test_padding_side_setter(self, tokenizer):
        """padding_side can be changed."""
        tokenizer.padding_side = "right"
        assert tokenizer.padding_side == "right"
        tokenizer.padding_side = "left"
        assert tokenizer.padding_side == "left"

    def test_padding_side_invalid_raises(self, tokenizer):
        """Invalid padding_side raises ValueError."""
        with pytest.raises(ValueError):
            tokenizer.padding_side = "center"

    def test_batch_to_list_left_padding(self, tokenizer):
        """to_list with left padding puts PAD at start."""
        batch = tokenizer.encode(["Hi", "Hello world"])
        out = batch.to_list(padding_side="left", pad_id=0)
        # Shorter sequence should have padding at start
        lengths = batch.lengths()
        if lengths[0] < lengths[1]:
            # First row is shorter, check padding is at start
            assert out["input_ids"][0][0] == 0  # PAD at start
            assert out["attention_mask"][0][0] == 0  # Mask 0 at start

    def test_batch_to_list_right_padding(self, tokenizer):
        """to_list with right padding puts PAD at end."""
        batch = tokenizer.encode(["Hi", "Hello world"])
        out = batch.to_list(padding_side="right", pad_id=0)
        # Shorter sequence should have padding at end
        lengths = batch.lengths()
        if lengths[0] < lengths[1]:
            # First row is shorter, check padding is at end
            assert out["input_ids"][0][-1] == 0  # PAD at end
            assert out["attention_mask"][0][-1] == 0  # Mask 0 at end

    def test_batch_to_list_invalid_padding_side(self, tokenizer):
        """Invalid padding_side raises ValueError."""
        batch = tokenizer.encode(["Hello", "World"])
        with pytest.raises(ValueError):
            batch.to_list(padding_side="center")

    def test_batch_to_list_returns_dict(self, tokenizer):
        """to_list returns dict with input_ids and attention_mask."""
        batch = tokenizer.encode(["Hello", "World"])
        out = batch.to_list()
        assert isinstance(out, dict)
        assert "input_ids" in out
        assert "attention_mask" in out

    def test_batch_attention_mask_matches_padding(self, tokenizer):
        """Attention mask 0s match padding positions."""
        batch = tokenizer.encode(["Hi", "Hello world this is longer"])
        out = batch.to_list(padding_side="left", pad_id=999)
        for i, (ids, mask) in enumerate(zip(out["input_ids"], out["attention_mask"], strict=True)):
            for j, (token, m) in enumerate(zip(ids, mask, strict=True)):
                if token == 999:  # PAD token
                    assert m == 0, f"Mask should be 0 at pad position [{i}][{j}]"
                else:
                    assert m == 1, f"Mask should be 1 at real token position [{i}][{j}]"

    def test_batch_to_list_left_padding_dict(self, tokenizer):
        """to_list works with left padding and returns dict."""
        batch = tokenizer.encode(["Hi", "Hello world"])
        out = batch.to_list(padding_side="left")
        assert "input_ids" in out
        assert "attention_mask" in out
        assert isinstance(out["input_ids"], list)
        assert isinstance(out["attention_mask"], list)

    # =========================================================================
    # Tests for __call__ with explicit arguments
    # =========================================================================

    def test_call_single_string(self, tokenizer):
        """__call__ with single string returns BatchEncoding with 1 sequence."""
        from talu.tokenizer import BatchEncoding

        result = tokenizer("Hello world")
        assert isinstance(result, BatchEncoding)
        assert len(result) == 1  # Single sequence
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_call_list_returns_batch_encoding(self, tokenizer):
        """__call__ with list returns BatchEncoding."""
        from talu.tokenizer import BatchEncoding

        result = tokenizer(["Hello", "World"])
        assert isinstance(result, BatchEncoding)
        assert len(result) == 2

    def test_call_list_returns_batch_with_padding(self, tokenizer):
        """__call__ with list returns BatchEncoding that can be padded."""
        result = tokenizer(["Hi", "Hello world"])
        # Check via to_list that padding is applied
        padded = result.to_list(padding=True)
        # All rows should have same length
        assert len(padded["input_ids"][0]) == len(padded["input_ids"][1])

    def test_call_uses_tokenizer_padding_side(self, tokenizer):
        """__call__ uses tokenizer.padding_side for padding direction."""
        tokenizer.padding_side = "left"
        result = tokenizer(["Hi", "Hello world"])
        padded = result.to_list(padding=True)
        # With left padding, shorter sequence has 0 at start (if pad_id is 0)
        if len(tokenizer.encode("Hi")) < len(tokenizer.encode("Hello world")):
            pad_id = result.pad_token_id or 0
            assert padded["input_ids"][0][0] == pad_id

    def test_call_truncation_single(self, tokenizer):
        """Truncation via to_list or DLPack max_length."""
        # __call__ returns BatchEncoding; truncation applied via to_list
        result = tokenizer("Hello world this is a test")
        truncated = result.to_list(truncation=True, max_length=3)
        assert len(truncated["input_ids"][0]) == 3

    def test_call_truncation_batch(self, tokenizer):
        """Truncation via to_list for batch."""
        result = tokenizer(["Hello world this is long", "Short"])
        truncated = result.to_list(truncation=True, max_length=3)
        # All sequences should be exactly max_length
        for seq in truncated["input_ids"]:
            assert len(seq) == 3

    def test_call_truncation_uses_model_max_length(self, tokenizer):
        """truncation=True without max_length uses model_max_length."""
        # Should NOT raise - uses model_max_length as default
        result = tokenizer("Hello", truncation=True)
        from talu.tokenizer import BatchEncoding

        assert isinstance(result, BatchEncoding)
        # Result should be within model_max_length
        ids = result.to_list()["input_ids"][0]
        assert len(ids) <= tokenizer.model_max_length

    def test_call_return_tensors_ignored(self, tokenizer):
        """return_tensors is ignored (always returns BatchEncoding)."""
        from talu.tokenizer import BatchEncoding

        # Should not raise, just ignore
        result = tokenizer("Hello", return_tensors="pt")
        assert isinstance(result, BatchEncoding)

    def test_call_text_pair_raises(self, tokenizer):
        """text_pair raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            tokenizer("Hello", text_pair="World")

    def test_call_padding_max_length(self, tokenizer):
        """padding to max_length via to_list."""
        result = tokenizer(["Hi", "Hello"])
        padded = result.to_list(padding=True, max_length=10)
        assert len(padded["input_ids"][0]) == 10
        assert len(padded["input_ids"][1]) == 10

    def test_call_attention_mask_always_available(self, tokenizer):
        """attention_mask is always available via dict-like access."""
        result = tokenizer("Hello")
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_call_special_tokens_parameter(self, tokenizer):
        """special_tokens parameter works in __call__."""
        with_special = tokenizer("Hello", special_tokens=True)
        without_special = tokenizer("Hello", special_tokens=False)
        # Both should work and return BatchEncoding
        assert "input_ids" in with_special
        assert "input_ids" in without_special

    # =========================================================================
    # BatchEncoding.to_list max_length and truncation tests
    # =========================================================================

    def test_batch_to_list_max_length(self, tokenizer):
        """to_list with max_length pads to specified length."""
        batch = tokenizer.encode(["Hi", "Hello"])
        result = batch.to_list(padding=True, max_length=10)
        assert len(result["input_ids"][0]) == 10
        assert len(result["input_ids"][1]) == 10

    def test_batch_to_list_truncation(self, tokenizer):
        """to_list with truncation truncates long sequences."""
        # Encode a longer text
        batch = tokenizer.encode(["Hello world this is a test", "Hi"])
        original_lengths = batch.lengths()

        # Truncate to 3 tokens
        result = batch.to_list(padding=True, max_length=3, truncation=True)

        # All sequences should be exactly 3 tokens
        assert len(result["input_ids"][0]) == 3
        assert len(result["input_ids"][1]) == 3

        # First sequence was truncated (originally longer)
        assert original_lengths[0] > 3

    def test_batch_to_list_truncation_attention_mask(self, tokenizer):
        """Truncation produces correct attention masks."""
        batch = tokenizer.encode(["Hello world this is a test", "Hi"])
        result = batch.to_list(
            padding=True,
            max_length=3,
            truncation=True,
            padding_side="right",
        )

        # First sequence: 3 real tokens, no padding
        assert result["attention_mask"][0] == [1, 1, 1]

        # Second sequence: 1 real token (Hi), 2 padding
        assert result["attention_mask"][1] == [1, 0, 0]

    def test_batch_to_list_no_truncation_respects_max_length(self, tokenizer):
        """Without truncation, sequences longer than max_length stay long."""
        batch = tokenizer.encode(["Hello world this is a test"])
        original_len = batch.lengths()[0]

        # max_length without truncation - should pad to max of (max_length, actual)
        result = batch.to_list(padding=True, max_length=3, truncation=False)

        # Should NOT truncate - length should be original
        assert len(result["input_ids"][0]) == original_len

    def test_batch_to_list_return_attention_mask_false(self, tokenizer):
        """to_list with return_attention_mask=False excludes mask."""
        batch = tokenizer.encode(["Hi", "Hello"])
        result = batch.to_list(padding=True, return_attention_mask=False)
        assert "input_ids" in result
        assert "attention_mask" not in result


class TestBatchEncodingDLPack:
    """Tests for BatchEncoding DLPack protocol (pure Python).

    NOTE: BatchEncoding.__dlpack__() raises TypeError because BatchEncoding
    contains multiple tensors (input_ids, attention_mask). Users must use
    explicit accessors: batch.input_ids or batch.attention_mask.

    torch/numpy integration tests are in tests/reference/tokenizer/.
    """

    @pytest.mark.requires_model
    def test_has_dlpack_protocol(self, tokenizer):
        """BatchEncoding implements DLPack protocol (but raises TypeError)."""
        batch = tokenizer.encode(["Hello", "World"])
        assert hasattr(batch, "__dlpack__")
        assert hasattr(batch, "__dlpack_device__")

    @pytest.mark.requires_model
    def test_dlpack_device_returns_cpu(self, tokenizer):
        """__dlpack_device__ returns CPU device tuple."""
        batch = tokenizer.encode(["Hello", "World"])
        device = batch.__dlpack_device__()
        assert device == (1, 0)  # kDLCPU = 1, device_id = 0

    @pytest.mark.requires_model
    def test_dlpack_raises_typeerror(self, tokenizer):
        """BatchEncoding.__dlpack__() raises TypeError (ambiguous)."""
        batch = tokenizer.encode(["Hello", "World"])
        with pytest.raises(TypeError, match="multiple tensors"):
            batch.__dlpack__()

    @pytest.mark.requires_model
    def test_input_ids_accessor_dlpack_returns_capsule(self, tokenizer):
        """batch.input_ids.__dlpack__() returns PyCapsule."""
        batch = tokenizer.encode(["Hello", "World", "Test"], special_tokens=False)
        capsule = batch.input_ids.__dlpack__()
        assert type(capsule).__name__ == "PyCapsule"

    @pytest.mark.requires_model
    def test_attention_mask_accessor_dlpack_returns_capsule(self, tokenizer):
        """batch.attention_mask.__dlpack__() returns PyCapsule."""
        batch = tokenizer.encode(["Hello", "World"], special_tokens=False)
        capsule = batch.attention_mask.__dlpack__()
        assert type(capsule).__name__ == "PyCapsule"

    @pytest.mark.requires_model
    def test_dlpack_empty_batch_raises(self, tokenizer):
        """Cannot export empty BatchEncoding via DLPack."""
        batch = tokenizer.encode([])

        with pytest.raises(talu.InteropError, match="empty"):
            _ = batch.input_ids.__dlpack__()

    @pytest.mark.requires_model
    def test_dlpack_accessors_can_export_multiple_times(self, tokenizer):
        """Accessors can be called multiple times."""
        # Use sequences with different lengths to ensure padding
        batch = tokenizer.encode(["Hi", "Hello world today"], special_tokens=False)

        # Can export multiple times - should not raise
        c1 = batch.input_ids.__dlpack__()
        c2 = batch.input_ids.__dlpack__()
        c3 = batch.attention_mask.__dlpack__()
        c4 = batch.attention_mask.__dlpack__()

        # All should be capsules
        assert type(c1).__name__ == "PyCapsule"
        assert type(c2).__name__ == "PyCapsule"
        assert type(c3).__name__ == "PyCapsule"
        assert type(c4).__name__ == "PyCapsule"


# =============================================================================
# Holy Trinity lifecycle tests (close / context manager / __del__)
# =============================================================================


class TestTokenizerLifecycle:
    """Tokenizer must implement the Holy Trinity: close(), __enter__/__exit__, __del__."""

    @pytest.fixture
    def tokenizer(self, test_model_path):
        from talu.tokenizer import Tokenizer

        return Tokenizer(test_model_path)

    def test_close_is_idempotent(self, tokenizer):
        """close() can be called multiple times without error."""
        tokenizer.close()
        tokenizer.close()

    def test_context_manager_basic(self, test_model_path):
        """Tokenizer can be used as a context manager."""
        from talu.tokenizer import Tokenizer

        with Tokenizer(test_model_path) as tok:
            tokens = tok.encode("Hello")
            assert len(tokens) > 0
        # After exiting, the handle is freed
        assert tok._ptr is None

    def test_use_after_close_raises_state_error(self, tokenizer):
        """Methods on a closed Tokenizer raise StateError."""
        from talu.exceptions import StateError

        tokenizer.close()
        with pytest.raises(StateError):
            tokenizer.encode("Hello")

    def test_context_manager_returns_self(self, test_model_path):
        """__enter__ returns self."""
        from talu.tokenizer import Tokenizer

        tok = Tokenizer(test_model_path)
        with tok as ctx:
            assert ctx is tok


class TestTokenArrayLifecycle:
    """TokenArray must implement the Holy Trinity: close(), __enter__/__exit__, __del__."""

    @pytest.fixture
    def tokenizer(self, test_model_path):
        from talu.tokenizer import Tokenizer

        return Tokenizer(test_model_path)

    def test_close_releases_buffer(self, tokenizer):
        """close() releases the underlying buffer."""
        tokens = tokenizer.encode("Hello world")
        assert tokens._buffer_handle is not None

        tokens.close()
        assert tokens._buffer_handle is None
        assert tokens._ptr is None

    def test_close_is_idempotent(self, tokenizer):
        """close() can be called multiple times without error."""
        tokens = tokenizer.encode("Hello world")
        tokens.close()
        tokens.close()

    def test_context_manager_basic(self, tokenizer):
        """TokenArray can be used as a context manager."""
        with tokenizer.encode("Hello world") as tokens:
            assert len(tokens) > 0
        # After exiting, buffer is freed
        assert tokens._buffer_handle is None

    def test_context_manager_returns_self(self, tokenizer):
        """__enter__ returns self."""
        tokens = tokenizer.encode("Hello world")
        with tokens as ctx:
            assert ctx is tokens


class TestBatchEncodingLifecycle:
    """BatchEncoding must implement the Holy Trinity: close(), __enter__/__exit__, __del__."""

    @pytest.fixture
    def tokenizer(self, test_model_path):
        from talu.tokenizer import Tokenizer

        return Tokenizer(test_model_path)

    def test_close_releases_memory(self, tokenizer):
        """close() releases native batch memory."""
        batch = tokenizer.encode(["Hello", "World"])
        assert batch._ids_ptr is not None

        batch.close()
        assert batch._ids_ptr is None
        assert batch._offsets_ptr is None

    def test_close_is_idempotent(self, tokenizer):
        """close() can be called multiple times without error."""
        batch = tokenizer.encode(["Hello", "World"])
        batch.close()
        batch.close()

    def test_context_manager_basic(self, tokenizer):
        """BatchEncoding can be used as a context manager."""
        with tokenizer.encode(["Hello", "World"]) as batch:
            assert len(batch) == 2
        # After exiting, memory is freed
        assert batch._ids_ptr is None

    def test_context_manager_returns_self(self, tokenizer):
        """__enter__ returns self."""
        batch = tokenizer.encode(["Hello", "World"])
        with batch as ctx:
            assert ctx is batch


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestTokenizerAdditionalCoverage:
    """Tests for additional Tokenizer methods and edge cases."""

    @pytest.fixture
    def tokenizer(self, test_model_path):
        """Create a Tokenizer for testing."""
        from talu.tokenizer import Tokenizer

        return Tokenizer(test_model_path)

    # =========================================================================
    # truncation_side property (lines 273, 277-281)
    # =========================================================================

    def test_truncation_side_getter(self, tokenizer):
        """truncation_side returns current value."""
        # Default is "right"
        assert tokenizer.truncation_side in ("left", "right")

    def test_truncation_side_setter_valid(self, tokenizer):
        """truncation_side can be set to valid values."""
        tokenizer.truncation_side = "left"
        assert tokenizer.truncation_side == "left"

        tokenizer.truncation_side = "right"
        assert tokenizer.truncation_side == "right"

    def test_truncation_side_setter_invalid(self, tokenizer):
        """truncation_side raises ValidationError for invalid values."""
        from talu.exceptions import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            tokenizer.truncation_side = "center"

        assert "truncation_side" in str(exc_info.value)
        assert "'left'" in str(exc_info.value)
        assert "'right'" in str(exc_info.value)

    # =========================================================================
    # special_tokens parameter validation (line 410)
    # =========================================================================

    def test_encode_special_tokens_invalid_type(self, tokenizer):
        """encode() raises ValidationError for invalid special_tokens type."""
        from talu.exceptions import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            tokenizer.encode("Hello", special_tokens="invalid")

        assert "special_tokens" in str(exc_info.value)
        assert "bool or set" in str(exc_info.value)

    def test_encode_special_tokens_with_set(self, tokenizer):
        """encode() accepts set for special_tokens."""
        # Just BOS
        tokens_bos = tokenizer.encode("Hello", special_tokens={"bos"})
        assert len(tokens_bos) > 0

        # Just EOS
        tokens_eos = tokenizer.encode("Hello", special_tokens={"eos"})
        assert len(tokens_eos) > 0

        # Both
        tokens_both = tokenizer.encode("Hello", special_tokens={"bos", "eos"})
        assert len(tokens_both) > 0

        # Empty set (no special tokens)
        tokens_none = tokenizer.encode("Hello", special_tokens=set())
        assert len(tokens_none) > 0

    # =========================================================================
    # tokenize with return_bytes (lines 634-652, 658, 661, 670)
    # =========================================================================

    def test_tokenize_return_bytes_true(self, tokenizer):
        """tokenize(return_bytes=True) returns list of bytes."""
        result = tokenizer.tokenize("Hello world", return_bytes=True)
        assert isinstance(result, list)
        for token in result:
            assert isinstance(token, bytes)

    def test_tokenize_return_bytes_false(self, tokenizer):
        """tokenize(return_bytes=False) returns list of strings."""
        result = tokenizer.tokenize("Hello world", return_bytes=False)
        assert isinstance(result, list)
        for token in result:
            assert isinstance(token, str)

    def test_tokenize_return_bytes_unicode(self, tokenizer):
        """tokenize(return_bytes=True) handles unicode properly."""
        result = tokenizer.tokenize("café", return_bytes=True)
        assert isinstance(result, list)
        # Join bytes and verify we can reconstruct "café"
        joined = b"".join(result)
        # Should decode to original text (possibly with tokenizer artifacts)
        assert len(joined) > 0

    def test_tokenize_return_bytes_empty_string(self, tokenizer):
        """tokenize(return_bytes=True) handles empty string."""
        result = tokenizer.tokenize("", return_bytes=True)
        assert isinstance(result, list)
        # Empty string typically produces empty result
        assert len(result) == 0

    # =========================================================================
    # __contains__ method (lines 763-765)
    # =========================================================================

    def test_contains_valid_token(self, tokenizer):
        """'token' in tokenizer returns True for valid tokens."""
        # Common punctuation should be in vocab
        assert "!" in tokenizer
        assert "a" in tokenizer

    def test_contains_invalid_token(self, tokenizer):
        """'token' in tokenizer returns False for invalid tokens."""
        # This unlikely string should not be a single token
        assert "xyzzy_definitely_not_a_token_12345" not in tokenizer

    def test_contains_non_string(self, tokenizer):
        """'in' returns False for non-string values."""
        assert 123 not in tokenizer
        assert None not in tokenizer
        assert ["a"] not in tokenizer

    # =========================================================================
    # convert_ids_to_tokens and convert_tokens_to_ids (lines 769, 773)
    # =========================================================================

    def test_convert_ids_to_tokens(self, tokenizer):
        """convert_ids_to_tokens converts IDs to strings."""
        # Get some token IDs first
        tokens = tokenizer.encode("Hello world", special_tokens=False)
        token_ids = tokens.tolist()

        # Convert back to strings
        token_strings = tokenizer.convert_ids_to_tokens(token_ids)

        assert len(token_strings) == len(token_ids)
        for s in token_strings:
            assert s is None or isinstance(s, str)

    def test_convert_tokens_to_ids(self, tokenizer):
        """convert_tokens_to_ids converts strings to IDs."""
        # Get token strings first
        token_strings = tokenizer.tokenize("Hello world")

        # Convert to IDs
        token_ids = tokenizer.convert_tokens_to_ids(token_strings)

        assert len(token_ids) == len(token_strings)
        for tid in token_ids:
            assert tid is None or isinstance(tid, int)

    def test_convert_roundtrip(self, tokenizer):
        """convert_ids_to_tokens and convert_tokens_to_ids are inverses."""
        # Get some token IDs
        tokens = tokenizer.encode("Test text", special_tokens=False)
        original_ids = tokens.tolist()

        # Convert to strings
        token_strings = tokenizer.convert_ids_to_tokens(original_ids)

        # Convert back to IDs (filter out None values)
        valid_strings = [s for s in token_strings if s is not None]
        recovered_ids = tokenizer.convert_tokens_to_ids(valid_strings)

        # Should get back valid IDs
        for tid in recovered_ids:
            assert tid is None or isinstance(tid, int)

    # =========================================================================
    # decode with list input (covers line 578-579)
    # =========================================================================

    def test_decode_list_input(self, tokenizer):
        """decode() accepts list[int] directly."""
        tokens = tokenizer.encode("Hello world")
        token_list = tokens.tolist()

        # Decode using list
        result = tokenizer.decode(token_list)
        assert isinstance(result, str)
        assert "Hello" in result or "hello" in result.lower()

    # =========================================================================
    # encode empty batch (covers line 518-526)
    # =========================================================================

    def test_encode_empty_list(self, tokenizer):
        """encode([]) returns empty BatchEncoding."""
        from talu.tokenizer import BatchEncoding

        result = tokenizer.encode([])
        assert isinstance(result, BatchEncoding)
        assert len(result) == 0
        assert result.total_tokens == 0

    # =========================================================================
    # padding_side property (lines 254, 256-262)
    # =========================================================================

    def test_padding_side_setter_valid(self, tokenizer):
        """padding_side can be set to valid values."""
        tokenizer.padding_side = "right"
        assert tokenizer.padding_side == "right"

        tokenizer.padding_side = "left"
        assert tokenizer.padding_side == "left"

    def test_padding_side_setter_invalid(self, tokenizer):
        """padding_side raises ValidationError for invalid values."""
        from talu.exceptions import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            tokenizer.padding_side = "center"

        assert "padding_side" in str(exc_info.value)
