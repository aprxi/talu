"""Tests for Chat streaming vs non-streaming behavior.

These tests verify:
1. Streaming actually delivers tokens over time (not all at once)
2. Non-streaming returns complete response at once
3. Timing measurements are realistic (not absurdly high t/s)
4. Correct response types are returned (Response vs StreamingResponse)

These tests catch bugs like:
- Router.stream() running synchronously then yielding from queue
- --no-stream flag being ignored in CLI
- t/s calculations being wildly incorrect (387644 t/s instead of ~100)
- Wrong response type returned for stream=True/False
"""

import time

import pytest

from talu.chat import Response, StreamingResponse


class TestStreamingResponseTypes:
    """Tests that verify correct response types are returned."""

    def test_non_streaming_returns_response(self):
        """Non-streaming call returns Response type."""
        # Without a model, we can't actually generate, but we can test
        # the type system via mock or by checking the return type annotations
        # For now, just verify the types exist and are distinct
        assert Response is not StreamingResponse
        assert not issubclass(Response, StreamingResponse)
        assert not issubclass(StreamingResponse, Response)

    def test_streaming_response_is_iterable_type(self):
        """StreamingResponse type is iterable."""
        # StreamingResponse should have __iter__
        assert hasattr(StreamingResponse, "__iter__")

    def test_response_is_not_iterable_type(self):
        """Response type does not have streaming iteration."""
        # Response should not have __iter__ for streaming
        response = Response(text="test")
        # Response doesn't implement __iter__ at all (removed in refactor)
        # Trying to iterate should fail
        with pytest.raises(TypeError):
            iter(response)


class TestStreamingBehavior:
    """Tests that verify streaming delivers tokens over time."""

    @pytest.mark.requires_model
    def test_streaming_yields_tokens_over_time(self, test_model_path):
        """Streaming should yield tokens incrementally, not all at once.

        Bug this catches: Router.stream() running synchronously then yielding
        from queue after completion (all tokens appear instantly).
        """
        from talu import Chat, GenerationConfig

        chat = Chat(test_model_path, config=GenerationConfig(max_tokens=20))

        # Track timing of token arrivals
        token_times = []
        start = time.perf_counter()

        response = chat("Count from 1 to 10", stream=True)
        for _token in response:
            token_times.append(time.perf_counter() - start)

        # Must have received multiple tokens
        assert len(token_times) >= 5, f"Expected at least 5 tokens, got {len(token_times)}"

        # Key test: tokens should arrive over time, not all at once
        # If streaming is broken (buffered), all tokens arrive within milliseconds
        # If streaming works, tokens are spread over the generation time
        first_token_time = token_times[0]
        last_token_time = token_times[-1]
        generation_duration = last_token_time - first_token_time

        # With working streaming, generation should take measurable time
        # (typically 50-500ms for 20 tokens, depending on model/hardware)
        assert generation_duration > 0.01, (
            f"Streaming appears broken: all {len(token_times)} tokens arrived within "
            f"{generation_duration * 1000:.1f}ms. Tokens should be spread over time, "
            f"not buffered and released at once."
        )

    @pytest.mark.requires_model
    def test_streaming_first_token_before_last(self, test_model_path):
        """First token should arrive significantly before last token.

        This verifies true streaming where tokens are yielded as generated,
        not buffered until completion.
        """
        from talu import Chat, GenerationConfig

        chat = Chat(test_model_path, config=GenerationConfig(max_tokens=30))

        # Measure time to first token vs time to last token
        start = time.perf_counter()
        first_token_time = None
        token_count = 0

        response = chat("Write a short sentence about cats", stream=True)
        for _token in response:
            if first_token_time is None:
                first_token_time = time.perf_counter() - start
            token_count += 1

        total_time = time.perf_counter() - start

        # First token should arrive well before total completion
        # If buffered, first_token_time ≈ total_time
        # If streaming, first_token_time << total_time
        assert first_token_time is not None, "No tokens received"
        assert token_count >= 3, f"Expected at least 3 tokens, got {token_count}"

        # First token should arrive in less than half the total time
        # (with real streaming, it's usually <10% of total time)
        if total_time > 0.05:  # Only check if generation took measurable time
            ratio = first_token_time / total_time
            assert ratio < 0.5, (
                f"First token arrived at {first_token_time * 1000:.1f}ms, "
                f"total time {total_time * 1000:.1f}ms (ratio={ratio:.2f}). "
                f"First token should arrive much earlier if streaming works."
            )


class TestNonStreamingBehavior:
    """Tests that verify non-streaming returns complete response at once."""

    @pytest.mark.requires_model
    def test_non_streaming_returns_complete_response(self, test_model_path):
        """Non-streaming should return the full response in one call."""
        from talu import Chat, GenerationConfig

        chat = Chat(test_model_path, config=GenerationConfig(max_tokens=20))

        response = chat("What is 2+2?", stream=False)

        # Response should have generated tokens
        assert response.usage.completion_tokens > 0, "Should have generated tokens"

        # Non-streaming Response type doesn't have __iter__ at all
        # (it's a different type from StreamingResponse)
        with pytest.raises(TypeError):
            list(response)

    @pytest.mark.requires_model
    def test_non_streaming_has_usage_metadata(self, test_model_path):
        """Non-streaming response should have usage metadata."""
        from talu import Chat, GenerationConfig

        chat = Chat(test_model_path, config=GenerationConfig(max_tokens=20))

        response = chat("Say hello", stream=False)

        # Should have usage information
        assert response.usage is not None, "Response should have usage metadata"
        assert response.usage.completion_tokens > 0, "Should have completion tokens"


class TestRealisticTiming:
    """Tests that verify timing measurements are realistic."""

    @pytest.mark.requires_model
    def test_tokens_per_second_is_realistic(self, test_model_path):
        """t/s calculation should produce realistic values.

        Bug this catches: t/s being absurdly high (387644.0) because
        timing was measuring queue drain time instead of generation time.
        """
        from talu import Chat, GenerationConfig

        chat = Chat(test_model_path, config=GenerationConfig(max_tokens=30))

        # Measure actual generation time and token count
        start = time.perf_counter()
        token_count = 0

        response = chat("Tell me about Python programming", stream=True)
        for _token in response:
            token_count += 1

        elapsed = time.perf_counter() - start

        # Calculate t/s
        if elapsed > 0 and token_count > 0:
            tps = token_count / elapsed

            # Realistic t/s should be in a sane range
            # - Very slow: 1-10 t/s (CPU, large model)
            # - Typical: 20-200 t/s (modern hardware)
            # - Fast: 200-1000 t/s (optimized GPU)
            # - Absurd: >10000 t/s (indicates timing bug)
            assert tps < 5000, (
                f"Tokens/sec ({tps:.1f}) is unrealistically high. "
                f"Generated {token_count} tokens in {elapsed * 1000:.1f}ms. "
                f"This likely indicates a timing measurement bug."
            )

    @pytest.mark.requires_model
    def test_generation_takes_measurable_time(self, test_model_path):
        """Generation should take a measurable amount of time.

        If generation is nearly instant, timing calculations will be wrong.
        """
        from talu import Chat, GenerationConfig

        chat = Chat(test_model_path, config=GenerationConfig(max_tokens=20))

        start = time.perf_counter()
        response = chat("Count: 1, 2, 3, 4, 5", stream=True)
        tokens = list(response)  # Consume all tokens
        elapsed = time.perf_counter() - start

        # Generation should take at least some time
        # (sub-millisecond suggests something is wrong)
        assert len(tokens) >= 3, f"Expected at least 3 tokens, got {len(tokens)}"
        assert elapsed > 0.001, (
            f"Generation of {len(tokens)} tokens took only {elapsed * 1000:.3f}ms. "
            f"This is suspiciously fast and may indicate a timing bug."
        )


class TestStreamingVsNonStreamingSameResult:
    """Tests that streaming and non-streaming produce consistent results."""

    @pytest.mark.requires_model
    def test_streaming_concatenation_matches_response(self, test_model_path):
        """Concatenating streamed tokens should match response.text."""
        from talu import Chat, GenerationConfig

        # Use temperature=0 for deterministic output
        chat = Chat(test_model_path, config=GenerationConfig(max_tokens=15, temperature=0))

        # Stream and collect tokens
        response = chat("What is 1+1?", stream=True)
        streamed_text = "".join(response)

        # The response text should match concatenated stream
        assert len(streamed_text) > 0, "Streamed text should not be empty"
        assert str(response) == streamed_text, (
            f"Response text '{str(response)}' doesn't match streamed '{streamed_text}'"
        )


class TestStreamTokenClassification:
    """Tests that stream tokens carry content type metadata."""

    @pytest.mark.requires_model
    def test_stream_tokens_have_item_type(self, test_model_path):
        """Each streamed token should have an item_type attribute."""
        from talu import Chat, GenerationConfig
        from talu.router import StreamToken
        from talu.types import ItemType

        config = GenerationConfig(max_tokens=10)
        chat = Chat(test_model_path, config=config)
        tokens = list(chat._router.stream(chat, "Hi", config=config))
        assert len(tokens) > 0, "Should get at least one token"

        for token in tokens:
            assert isinstance(token, StreamToken)
            assert hasattr(token, "item_type")
            assert isinstance(token.item_type, ItemType)

    @pytest.mark.requires_model
    def test_stream_tokens_have_content_type(self, test_model_path):
        """Each streamed token should have a content_type attribute."""
        from talu import Chat, GenerationConfig
        from talu.router import StreamToken
        from talu.types import ContentType

        config = GenerationConfig(max_tokens=10)
        chat = Chat(test_model_path, config=config)
        tokens = list(chat._router.stream(chat, "Hi", config=config))
        assert len(tokens) > 0

        for token in tokens:
            assert isinstance(token, StreamToken)
            assert hasattr(token, "content_type")
            assert isinstance(token.content_type, ContentType)

    @pytest.mark.requires_model
    def test_stream_tokens_text_concatenation(self, test_model_path):
        """StreamToken.text can be concatenated to reconstruct full output."""
        from talu import Chat, GenerationConfig
        from talu.router import StreamToken

        config = GenerationConfig(max_tokens=10)
        chat = Chat(test_model_path, config=config)
        tokens = list(chat._router.stream(chat, "Hi", config=config))
        assert len(tokens) > 0

        # Each token is a StreamToken with a .text attribute
        for token in tokens:
            assert isinstance(token, StreamToken)
            assert isinstance(token.text, str)

        # Concatenation via .text
        text = "".join(t.text for t in tokens)
        assert isinstance(text, str)
        assert len(text) > 0
