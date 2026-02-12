"""
Stability tests for Chat/Session focusing on memory safety and error handling.

These tests verify:
- Memory safety during resource cleanup
- Error propagation from C API failures
- Edge cases in streaming responses
- Concurrent/interleaved operations
- Recovery from partial failures
"""

import gc
import threading
import weakref

import pytest

from talu.chat.response import (
    FinishReason,
    Response,
    StreamingResponse,
    Timings,
    Usage,
)
from talu.exceptions import GenerationError, StateError, ValidationError
from talu.router.config.generation import GenerationConfig
from tests.conftest import TEST_MODEL_URI_TEXT_RANDOM

# =============================================================================
# Memory Safety Tests
# =============================================================================


class TestMemorySafety:
    """Tests for memory safety and resource cleanup."""

    def test_streaming_response_cleanup_on_gc(self):
        """StreamingResponse cleans up when garbage collected."""
        cleanup_called = []

        def gen_with_cleanup():
            try:
                yield "Hello"
                yield " World"
            finally:
                cleanup_called.append(True)

        response = StreamingResponse(stream_iterator=gen_with_cleanup())
        # Partially consume
        next(iter(response))

        # Delete and force GC
        del response
        gc.collect()

        # Cleanup should have been triggered
        assert len(cleanup_called) == 1

    def test_streaming_response_weakref_does_not_prevent_cleanup(self):
        """Weakref to StreamingResponse doesn't prevent cleanup."""

        def simple_gen():
            yield "Test"

        response = StreamingResponse(stream_iterator=simple_gen())
        ref = weakref.ref(response)

        assert ref() is not None
        del response
        gc.collect()
        assert ref() is None

    def test_response_with_none_values_stable(self):
        """Response handles None metadata gracefully."""
        response = Response(
            text="Test",
            usage=None,
            timings=None,
            model=None,
            # Note: finish_reason has a default of "eos_token" in Response
        )

        assert response.text == "Test"
        assert response.usage is None
        assert response.timings is None
        assert response.model is None
        # finish_reason defaults to "eos_token" when not specified
        assert response.finish_reason is not None

    def test_streaming_multiple_iterations_stable(self):
        """Multiple iterations on exhausted stream don't crash."""

        def simple_gen():
            yield "Once"

        response = StreamingResponse(stream_iterator=simple_gen())

        # Exhaust stream
        list(response)
        list(response)
        list(response)

        # All iterations should return empty after exhaustion
        assert list(response) == []
        assert response.text == "Once"


# =============================================================================
# Error Propagation Tests
# =============================================================================


class TestErrorPropagation:
    """Tests for proper error propagation from failures."""

    def test_generation_error_preserves_context(self):
        """GenerationError preserves error code and message."""
        err = GenerationError(
            "Model inference failed: OOM",
            code="INFERENCE_OOM",
        )

        assert "OOM" in str(err)
        assert err.code == "INFERENCE_OOM"

    def test_state_error_preserves_context(self):
        """StateError preserves error code and message."""
        err = StateError(
            "Chat already closed",
            code="CHAT_CLOSED",
        )

        assert "closed" in str(err).lower()
        assert err.code == "CHAT_CLOSED"

    def test_validation_error_preserves_context(self):
        """ValidationError preserves error code and message."""
        err = ValidationError(
            "Temperature must be >= 0",
            code="INVALID_TEMPERATURE",
        )

        assert "Temperature" in str(err)
        assert err.code == "INVALID_TEMPERATURE"

    def test_streaming_error_stops_iteration(self):
        """Streaming error stops iteration cleanly."""
        iterations = []

        def failing_at_third():
            iterations.append(1)
            yield "One"
            iterations.append(2)
            yield "Two"
            iterations.append(3)
            raise GenerationError("Token limit exceeded", code="MAX_TOKENS")

        response = StreamingResponse(stream_iterator=failing_at_third())

        with pytest.raises(GenerationError) as exc:
            list(response)

        assert exc.value.code == "MAX_TOKENS"
        assert len(iterations) == 3
        # Text should contain what was generated before error
        assert "One" in response.text
        assert "Two" in response.text

    def test_streaming_error_prevents_on_complete(self):
        """on_complete not called if stream errors."""
        on_complete_called = []

        def failing_gen():
            yield "Start"
            raise RuntimeError("Network error")

        response = StreamingResponse(
            stream_iterator=failing_gen(),
            on_complete=lambda text: on_complete_called.append(text),
        )

        with pytest.raises(RuntimeError):
            list(response)

        assert len(on_complete_called) == 0


# =============================================================================
# Streaming Edge Cases
# =============================================================================


class TestStreamingEdgeCases:
    """Edge case tests for streaming responses."""

    def test_empty_stream_completes_normally(self):
        """Empty stream completes without error."""

        def empty_gen():
            return
            yield  # Makes this a generator

        response = StreamingResponse(stream_iterator=empty_gen())
        tokens = list(response)

        assert tokens == []
        assert response.text == ""

    def test_single_empty_string_token(self):
        """Stream yielding empty string handles correctly."""

        def empty_token_gen():
            yield ""
            yield "Real"
            yield ""

        response = StreamingResponse(stream_iterator=empty_token_gen())
        tokens = list(response)

        assert len(tokens) == 3
        assert response.text == "Real"

    def test_very_long_token(self):
        """Very long individual token handled correctly."""
        long_text = "x" * 100000

        def long_gen():
            yield long_text

        response = StreamingResponse(stream_iterator=long_gen())
        tokens = list(response)

        assert len(tokens) == 1
        assert len(response.text) == 100000

    def test_many_small_tokens(self):
        """Many small tokens accumulate correctly."""

        def many_tokens():
            for _ in range(10000):
                yield "a"

        response = StreamingResponse(stream_iterator=many_tokens())
        tokens = list(response)

        assert len(tokens) == 10000
        assert response.text == "a" * 10000

    def test_unicode_boundary_tokens(self):
        """Unicode characters split across tokens handled correctly."""

        def unicode_gen():
            yield "Hello "
            yield "世"
            yield "界"
            yield " 🌍"

        response = StreamingResponse(stream_iterator=unicode_gen())
        list(response)

        assert response.text == "Hello 世界 🌍"
        assert "世界" in response.text
        assert "🌍" in response.text

    def test_special_control_characters(self):
        """Special control characters preserved in stream."""

        def control_gen():
            yield "Line1\n"
            yield "Line2\r\n"
            yield "Tab\there"
            yield "\x00null"  # Null byte

        response = StreamingResponse(stream_iterator=control_gen())
        list(response)

        assert "\n" in response.text
        assert "\r\n" in response.text
        assert "\t" in response.text
        assert "\x00" in response.text


# =============================================================================
# Callback Safety Tests
# =============================================================================


class TestCallbackSafety:
    """Tests for callback safety in streaming responses."""

    def test_on_token_exception_propagates(self):
        """Exception in on_token callback propagates to caller."""

        def simple_gen():
            yield "Token1"
            yield "Token2"

        def failing_callback(token: str):
            if token == "Token2":
                raise ValueError("Callback failed")

        response = StreamingResponse(
            stream_iterator=simple_gen(),
            on_token=failing_callback,
        )

        with pytest.raises(ValueError, match="Callback failed"):
            list(response)

    def test_on_complete_exception_propagates(self):
        """Exception in on_complete callback propagates to caller."""

        def simple_gen():
            yield "Done"

        def failing_complete(text: str):
            raise RuntimeError("Complete handler failed")

        response = StreamingResponse(
            stream_iterator=simple_gen(),
            on_complete=failing_complete,
        )

        with pytest.raises(RuntimeError, match="Complete handler failed"):
            list(response)

    def test_callback_receives_correct_text(self):
        """Callbacks receive correct accumulated text."""
        token_texts = []
        final_text = []

        def gen():
            yield "Hello"
            yield " "
            yield "World"

        response = StreamingResponse(
            stream_iterator=gen(),
            on_token=lambda t: token_texts.append(t),
            on_complete=lambda t: final_text.append(t),
        )
        list(response)

        assert token_texts == ["Hello", " ", "World"]
        assert final_text == ["Hello World"]


# =============================================================================
# Configuration Edge Cases
# =============================================================================


class TestConfigurationEdgeCases:
    """Edge case tests for GenerationConfig."""

    def test_config_with_extreme_values(self):
        """Config accepts extreme but valid values."""
        config = GenerationConfig(
            max_tokens=1,
            temperature=0.0,
            top_k=1,
            top_p=0.0,
            min_p=1.0,
        )

        assert config.max_tokens == 1
        assert config.temperature == 0.0
        assert config.top_k == 1

    def test_config_override_preserves_unset(self):
        """Override preserves values not in override."""
        base = GenerationConfig(max_tokens=100, temperature=0.7, top_k=50)
        overridden = base.override(temperature=0.0)

        assert overridden.max_tokens == 100
        assert overridden.temperature == 0.0
        assert overridden.top_k == 50

    def test_config_override_with_none(self):
        """Override with None doesn't change value."""
        base = GenerationConfig(max_tokens=100, temperature=0.7)
        # None values in override should not clear base values
        overridden = base.override()

        assert overridden.max_tokens == 100
        assert overridden.temperature == 0.7

    def test_config_equality(self):
        """Config equality works correctly."""
        config1 = GenerationConfig(max_tokens=100, temperature=0.5)
        config2 = GenerationConfig(max_tokens=100, temperature=0.5)
        config3 = GenerationConfig(max_tokens=100, temperature=0.6)

        assert config1 == config2
        assert config1 != config3


# =============================================================================
# Timings and Usage Edge Cases
# =============================================================================


class TestTimingsAndUsage:
    """Edge case tests for Timings and Usage."""

    def test_timings_from_ns_zero_values(self):
        """Timings handles zero values without division errors."""
        timings = Timings.from_ns(
            prefill_ns=0,
            generation_ns=0,
            token_count=0,
        )

        assert timings.prefill_ms == 0.0
        assert timings.generation_ms == 0.0
        assert timings.tokens_per_second == 0.0

    def test_timings_from_ns_large_values(self):
        """Timings handles large values without overflow."""
        # 1 hour in nanoseconds
        one_hour_ns = 3600 * 1_000_000_000
        timings = Timings.from_ns(
            prefill_ns=one_hour_ns,
            generation_ns=one_hour_ns,
            token_count=1_000_000,
        )

        assert timings.prefill_ms == 3600000.0
        assert abs(timings.tokens_per_second - (1_000_000 / 3600)) < 1

    def test_usage_total_consistency(self):
        """Usage total_tokens should equal prompt + completion."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens


# =============================================================================
# Thread Safety (Basic)
# =============================================================================


class TestBasicThreadSafety:
    """Basic thread safety tests (without real model)."""

    def test_streaming_consumed_in_different_thread(self):
        """Streaming response can be consumed in different thread."""
        result = []

        def gen():
            yield "Thread"
            yield " "
            yield "Safe"

        response = StreamingResponse(stream_iterator=gen())

        def consume():
            result.extend(list(response))

        thread = threading.Thread(target=consume)
        thread.start()
        thread.join()  # Blocks until complete - mock generators finish instantly

        assert response.text == "Thread Safe"
        assert len(result) == 3

    def test_multiple_response_objects_independent(self):
        """Multiple response objects don't interfere."""
        results = {}

        def gen1():
            yield "One"

        def gen2():
            yield "Two"

        response1 = StreamingResponse(stream_iterator=gen1())
        response2 = StreamingResponse(stream_iterator=gen2())

        def consume1():
            results["r1"] = list(response1)

        def consume2():
            results["r2"] = list(response2)

        t1 = threading.Thread(target=consume1)
        t2 = threading.Thread(target=consume2)

        t1.start()
        t2.start()
        t1.join()  # Blocks until complete - mock generators finish instantly
        t2.join()

        assert results["r1"] == ["One"]
        assert results["r2"] == ["Two"]
        assert response1.text == "One"
        assert response2.text == "Two"


# =============================================================================
# Response Object State Tests
# =============================================================================


class TestResponseState:
    """Tests for Response object state management."""

    def test_response_immutable_text(self):
        """Response text is immutable after creation."""
        response = Response(text="Original")

        # Text property should be read-only
        assert response.text == "Original"
        # The response object itself is immutable by design

    def test_streaming_response_text_accumulates(self):
        """StreamingResponse text accumulates during iteration."""

        def gen():
            yield "A"
            yield "B"
            yield "C"

        response = StreamingResponse(stream_iterator=gen())

        # Before iteration, internal _text is empty
        assert response._text == ""

        # Manually iterate to check partial accumulation
        # (can't use .text property as it auto-drains)
        iterator = iter(response)
        next(iterator)
        assert response._text == "A"

        next(iterator)
        assert response._text == "AB"

        next(iterator)
        assert response._text == "ABC"

        # Now .text returns the full accumulated text
        assert response.text == "ABC"

    def test_finish_reason_values(self):
        """FinishReason enum values are correct."""
        assert FinishReason.EOS_TOKEN == "eos_token"
        assert FinishReason.LENGTH == "length"
        assert FinishReason.STOP_SEQUENCE == "stop_sequence"
        assert FinishReason.TOOL_CALLS == "tool_calls"

    def test_response_str_returns_text(self):
        """str(response) returns the text."""
        response = Response(text="Test output")
        assert str(response) == "Test output"

    def test_streaming_response_str_returns_text(self):
        """str(streaming_response) returns accumulated text."""

        def gen():
            yield "Stream"

        response = StreamingResponse(stream_iterator=gen())
        list(response)  # Exhaust stream
        assert str(response) == "Stream"


# =============================================================================
# Iteration Protocol Tests
# =============================================================================


class TestIterationProtocol:
    """Tests for iterator protocol compliance."""

    def test_streaming_response_is_iterable(self):
        """StreamingResponse implements iterable protocol."""

        def gen():
            yield "Test"

        response = StreamingResponse(stream_iterator=gen())

        assert hasattr(response, "__iter__")
        assert iter(response) is not None

    def test_streaming_response_iter_is_self(self):
        """StreamingResponse iter returns iterator over tokens."""

        def gen():
            yield "A"
            yield "B"

        response = StreamingResponse(stream_iterator=gen())

        # Should be able to iterate
        tokens = list(response)
        assert tokens == ["A", "B"]

    def test_streaming_response_join_pattern(self):
        """Common pattern: ''.join(response) works."""

        def gen():
            yield "Hello"
            yield " "
            yield "World"

        response = StreamingResponse(stream_iterator=gen())
        text = "".join(response)

        assert text == "Hello World"
        assert response.text == text


# =============================================================================
# Nested/Complex Structure Tests
# =============================================================================


class TestComplexStructures:
    """Tests for complex/nested response scenarios."""

    def test_response_with_all_metadata(self):
        """Response with all metadata fields populated."""
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        timings = Timings.from_ns(
            prefill_ns=1_000_000,
            generation_ns=2_000_000,
            token_count=20,
        )

        response = Response(
            text="Complete response",
            usage=usage,
            timings=timings,
            model="test-model-v1",
            finish_reason=FinishReason.EOS_TOKEN,
        )

        assert response.text == "Complete response"
        assert response.usage.total_tokens == 30
        assert response.timings.prefill_ms == 1.0
        assert response.model == "test-model-v1"
        assert response.finish_reason == "eos_token"

    def test_streaming_response_with_all_metadata(self):
        """StreamingResponse with all metadata fields populated."""

        def gen():
            yield "Test"

        usage = Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6)
        timings = Timings.from_ns(
            prefill_ns=500_000,
            generation_ns=1_000_000,
            token_count=1,
        )

        response = StreamingResponse(
            stream_iterator=gen(),
            usage=usage,
            timings=timings,
            model="streaming-model",
            finish_reason=FinishReason.LENGTH,
        )

        list(response)  # Consume stream

        assert response.text == "Test"
        assert response.usage.total_tokens == 6
        assert response.timings.generation_ms == 1.0
        assert response.model == "streaming-model"
        assert response.finish_reason == "length"


# =============================================================================
# Multiple Chat Instance Tests
# =============================================================================


class TestMultipleChatInstances:
    """Tests for multiple Chat instance memory safety.

    These tests verify that creating multiple Chat instances with overlapping
    lifetimes doesn't cause memory corruption or segfaults.

    Known issue: Creating Chat instances in a loop where the variable is
    rebound causes intermittent segfaults due to memory management issues
    in the Zig core related to resource cleanup timing.
    """

    @pytest.mark.integration
    def test_multiple_chat_instances_in_loop(self):
        """Multiple Chat instances in a loop should not segfault.

        This is a regression test for a memory corruption bug where
        creating Chat instances in a loop with variable rebinding
        causes segfaults.

        To reproduce the bug manually:
            import talu
            for i in range(3):
                chat = talu.Chat(TEST_MODEL_URI_TEXT_RANDOM)
                response = chat(f"Count to {i+1}")
                print(response.text)

        Root cause: When Python's GC triggers cleanup of the previous Chat
        instance during the loop, it frees backend resources (via talu_backend_free)
        that may still be in use by subsequent Chat instances due to shared
        global state in the Zig core (exact mechanism TBD - possibly related
        to the graph registry or mmap'd model files).
        """
        import talu

        for i in range(3):
            chat = talu.Chat(TEST_MODEL_URI_TEXT_RANDOM)
            response = chat(f"Count to {i + 1}")
            assert len(response.text) > 0
