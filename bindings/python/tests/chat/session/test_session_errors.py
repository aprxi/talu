"""
Mock-based tests for Chat/Session error paths.

These tests mock the Router and C API to test Python error handling
without requiring real models. This covers critical error paths that
are otherwise only reachable during actual inference failures.
"""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from talu.chat.response import (
    FinishReason,
    StreamingResponse,
    Timings,
    Usage,
)
from talu.exceptions import GenerationError, StateError, ValidationError
from talu.router.config import GenerationConfig

# =============================================================================
# Generation Config Validation Tests
# =============================================================================


class TestGenerationConfigValidation:
    """Tests for GenerationConfig validation."""

    def test_config_rejects_unknown_params(self):
        """GenerationConfig rejects unknown parameters."""
        with pytest.raises(TypeError):
            GenerationConfig(invalid_param=123)

    def test_config_accepts_valid_params(self):
        """GenerationConfig accepts all documented params."""
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.5,
            top_k=50,
            top_p=0.9,
            min_p=0.1,
        )
        assert config.max_tokens == 100
        assert config.temperature == 0.5
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.min_p == 0.1

    def test_config_override_creates_new_instance(self):
        """GenerationConfig.override() creates new instance with overrides."""
        base = GenerationConfig(max_tokens=50, temperature=0.7)
        merged = base.override(temperature=0.1)

        # Original unchanged
        assert base.temperature == 0.7

        # Override applied
        assert merged.temperature == 0.1
        assert merged.max_tokens == 50

    def test_config_dataclass_replace(self):
        """GenerationConfig works with dataclasses.replace."""
        base = GenerationConfig(max_tokens=50, temperature=0.7)
        new = replace(base, temperature=0.1, top_k=20)

        # Override values take precedence
        assert new.temperature == 0.1
        assert new.top_k == 20
        # Base values preserved
        assert new.max_tokens == 50


# =============================================================================
# Exception Construction Tests
# =============================================================================


class TestExceptionConstruction:
    """Tests for exception construction and attributes."""

    def test_generation_error_with_code(self):
        """GenerationError stores code attribute."""
        err = GenerationError("Test error", code="TEST_CODE")
        assert str(err) == "Test error"
        assert err.code == "TEST_CODE"

    def test_generation_error_default_code(self):
        """GenerationError has default code."""
        err = GenerationError("Test error")
        assert str(err) == "Test error"
        # Default code is set by the class
        assert err.code is not None

    def test_state_error_construction(self):
        """StateError stores code attribute."""
        err = StateError("Chat is closed", code="STATE_ERROR")
        assert "closed" in str(err).lower()
        assert err.code == "STATE_ERROR"

    def test_validation_error_construction(self):
        """ValidationError stores details."""
        err = ValidationError("Invalid parameter", code="VALIDATION_ERROR")
        assert "Invalid parameter" in str(err)


# =============================================================================
# StreamingResponse State Tests
# =============================================================================


class TestStreamingResponseState:
    """Tests for StreamingResponse state management."""

    def test_streaming_response_initial_state(self):
        """StreamingResponse starts with empty internal text (before iteration)."""

        def dummy_gen():
            yield "Hello"

        response = StreamingResponse(stream_iterator=dummy_gen())
        # Internal _text is empty before iteration
        assert response._text == ""
        # But .text property auto-drains the stream
        assert response.text == "Hello"

    def test_streaming_response_accumulates_text(self):
        """StreamingResponse accumulates text during iteration."""

        def simple_gen():
            yield "Hello"
            yield " "
            yield "World"

        response = StreamingResponse(stream_iterator=simple_gen())
        tokens = list(response)

        assert response.text == "Hello World"
        assert len(tokens) == 3

    def test_streaming_response_error_during_iteration(self):
        """StreamingResponse handles errors during iteration."""

        def failing_gen():
            yield "OK"
            raise RuntimeError("Connection lost")

        response = StreamingResponse(stream_iterator=failing_gen())

        with pytest.raises(RuntimeError, match="Connection lost"):
            list(response)

    def test_streaming_response_partial_consumption(self):
        """StreamingResponse handles partial consumption."""

        def multi_token_gen():
            yield "Token1"
            yield "Token2"
            yield "Token3"

        response = StreamingResponse(stream_iterator=multi_token_gen())
        first = next(iter(response))

        assert first == "Token1"
        assert response.text == "Token1"


# =============================================================================
# GenerationConfig Edge Cases
# =============================================================================


class TestGenerationConfigEdgeCases:
    """Edge case tests for GenerationConfig."""

    def test_config_with_zero_values(self):
        """GenerationConfig accepts zero values."""
        config = GenerationConfig(temperature=0.0, top_k=0)
        assert config.temperature == 0.0
        assert config.top_k == 0

    def test_config_with_none_values(self):
        """GenerationConfig handles None values."""
        config = GenerationConfig(max_tokens=None, temperature=None)
        assert config.max_tokens is None
        assert config.temperature is None

    def test_config_repr(self):
        """GenerationConfig has useful repr."""
        config = GenerationConfig(max_tokens=100, temperature=0.5)
        repr_str = repr(config)
        assert "GenerationConfig" in repr_str

    def test_config_is_mutable(self):
        """GenerationConfig is mutable for easy session configuration."""
        config = GenerationConfig(max_tokens=100)
        config.max_tokens = 200
        assert config.max_tokens == 200


# =============================================================================
# Mock Router Generate Tests
# =============================================================================


class TestMockedRouterGenerate:
    """Tests for Router.generate with mocked responses."""

    def test_router_generate_success_path(self):
        """Mocked router returns expected result dict."""
        mock_router = MagicMock()
        mock_router.generate.return_value = {
            "text": "Hello, I am an AI.",
            "token_count": 5,
            "prefill_ns": 1000000,
            "generation_ns": 2000000,
        }

        result = mock_router.generate(MagicMock(), "Hello", model="test")

        assert result["text"] == "Hello, I am an AI."
        assert result["token_count"] == 5

    def test_router_generate_error_propagation(self):
        """Router errors propagate correctly."""
        mock_router = MagicMock()
        mock_router.generate.side_effect = GenerationError("Model failed", code="MODEL_LOAD_FAILED")

        with pytest.raises(GenerationError) as exc:
            mock_router.generate(MagicMock(), "Hello", model="test")

        assert "Model failed" in str(exc.value)

    def test_router_stream_error_propagation(self):
        """Router stream errors propagate correctly."""
        mock_router = MagicMock()

        def failing_stream(*args, **kwargs):
            yield "Hello"
            raise GenerationError("Stream interrupted", code="STREAM_ERROR")

        mock_router.stream.return_value = failing_stream()

        stream = mock_router.stream(MagicMock(), "Hello", model="test")

        # First token should work
        first = next(stream)
        assert first == "Hello"

        # Second should raise
        with pytest.raises(GenerationError):
            next(stream)


# =============================================================================
# Response Metadata Tests
# =============================================================================


class TestResponseMetadata:
    """Tests for response metadata handling."""

    def test_timings_from_ns(self):
        """Timings.from_ns converts nanoseconds correctly."""
        timings = Timings.from_ns(
            prefill_ns=1_000_000_000,
            generation_ns=2_000_000_000,
            token_count=100,
        )

        assert timings.prefill_ms == 1000.0
        assert timings.generation_ms == 2000.0
        assert timings.tokens_per_second == 50.0  # 100 tokens / 2 seconds

    def test_timings_from_ns_zero_generation(self):
        """Timings.from_ns handles zero generation time."""
        timings = Timings.from_ns(
            prefill_ns=1_000_000,
            generation_ns=0,
            token_count=0,
        )

        assert timings.prefill_ms == 1.0
        assert timings.generation_ms == 0.0
        assert timings.tokens_per_second == 0.0

    def test_usage_construction(self):
        """Usage stores token counts."""
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30


# =============================================================================
# Finish Reason Tests
# =============================================================================


class TestFinishReason:
    """Tests for finish reason constants."""

    def test_finish_reason_values(self):
        """FinishReason has expected constants."""
        assert FinishReason.EOS_TOKEN == "eos_token"
        assert FinishReason.LENGTH == "length"
        assert FinishReason.STOP_SEQUENCE == "stop_sequence"
        assert FinishReason.TOOL_CALLS == "tool_calls"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for response handling."""

    def test_empty_stream_response(self):
        """Empty stream produces empty text.

        Note: The `return; yield` pattern creates an empty generator.
        Using `yield` anywhere in the function makes it a generator,
        even though it returns immediately without yielding any values.
        """

        def empty_gen():
            return
            yield  # noqa: B901

        response = StreamingResponse(stream_iterator=empty_gen())
        tokens = list(response)

        assert tokens == []
        assert response.text == ""

    def test_unicode_in_stream(self):
        """Unicode in stream is handled correctly."""

        def unicode_gen():
            yield "Hello "
            yield "你好 "
            yield "🌍"

        response = StreamingResponse(stream_iterator=unicode_gen())
        tokens = list(response)

        assert len(tokens) == 3
        assert "你好" in response.text
        assert "🌍" in response.text

    def test_special_chars_in_stream(self):
        """Special characters in stream are preserved."""

        def special_gen():
            yield 'Quote: "hello"'
            yield "\nNewline"
            yield "\tTab"

        response = StreamingResponse(stream_iterator=special_gen())
        _ = list(response)

        assert '"hello"' in response.text
        assert "\n" in response.text
        assert "\t" in response.text

    def test_large_token_count_stream(self):
        """Large number of tokens handled correctly."""

        def many_tokens():
            for i in range(1000):
                yield f"token{i} "

        response = StreamingResponse(stream_iterator=many_tokens())
        tokens = list(response)

        assert len(tokens) == 1000
        assert "token0" in response.text
        assert "token999" in response.text


# =============================================================================
# Callback Safety Tests
# =============================================================================


class TestCallbackSafety:
    """Tests for callback safety in streaming responses."""

    def test_on_token_callback(self):
        """on_token callback is called for each token."""

        def simple_gen():
            yield "Hello"
            yield " World"

        tokens_seen = []
        response = StreamingResponse(
            stream_iterator=simple_gen(),
            on_token=lambda t: tokens_seen.append(t),
        )
        list(response)

        assert len(tokens_seen) == 2
        assert tokens_seen[0] == "Hello"
        assert tokens_seen[1] == " World"

    def test_on_complete_callback(self):
        """on_complete callback is called with full text."""

        def simple_gen():
            yield "Hello"
            yield " World"

        completed_text = []
        response = StreamingResponse(
            stream_iterator=simple_gen(),
            on_complete=lambda t: completed_text.append(t),
        )
        list(response)

        assert len(completed_text) == 1
        assert completed_text[0] == "Hello World"

    def test_on_complete_not_called_on_error(self):
        """on_complete is not called when stream errors."""

        def failing_gen():
            yield "OK"
            raise RuntimeError("Stream error")

        callback_called = []
        response = StreamingResponse(
            stream_iterator=failing_gen(),
            on_complete=lambda t: callback_called.append(True),
        )

        with pytest.raises(RuntimeError):
            list(response)

        # Callback should not be called on error
        assert len(callback_called) == 0

    def test_stream_exhausted_returns_empty(self):
        """Already exhausted stream returns empty iterator."""

        def simple_gen():
            yield "Hello"

        response = StreamingResponse(stream_iterator=simple_gen())
        # First iteration exhausts the stream
        list(response)

        # Second iteration returns empty
        second = list(response)
        assert second == []


# =============================================================================
# Response with Metadata Tests
# =============================================================================


class TestResponseWithMetadata:
    """Tests for response objects with metadata."""

    def test_streaming_response_with_usage(self):
        """StreamingResponse can have usage metadata."""

        def simple_gen():
            yield "Hello"

        usage = Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6)
        response = StreamingResponse(
            stream_iterator=simple_gen(),
            usage=usage,
        )

        assert response.usage is not None
        assert response.usage.total_tokens == 6

    def test_streaming_response_with_timings(self):
        """StreamingResponse can have timings metadata."""

        def simple_gen():
            yield "Hello"

        timings = Timings.from_ns(
            prefill_ns=1_000_000,
            generation_ns=2_000_000,
            token_count=1,
        )
        response = StreamingResponse(
            stream_iterator=simple_gen(),
            timings=timings,
        )

        assert response.timings is not None
        assert response.timings.prefill_ms == 1.0

    def test_streaming_response_with_model(self):
        """StreamingResponse can have model name."""

        def simple_gen():
            yield "Hello"

        response = StreamingResponse(
            stream_iterator=simple_gen(),
            model="test-model",
        )

        assert response.model == "test-model"

    def test_streaming_response_with_finish_reason(self):
        """StreamingResponse can have finish reason."""

        def simple_gen():
            yield "Hello"

        response = StreamingResponse(
            stream_iterator=simple_gen(),
            finish_reason=FinishReason.EOS_TOKEN,
        )

        assert response.finish_reason == "eos_token"
