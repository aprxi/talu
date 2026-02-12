"""
Mock-based tests for Router error paths.

These tests mock the C API layer to test Python error handling without
requiring real models. This covers critical error paths that are otherwise
only reachable during actual inference failures, ensuring proper exception
mapping and resource cleanup.
"""

import ctypes
from unittest.mock import MagicMock, patch

import pytest

from talu._native import RouterGenerateResult
from talu.exceptions import GenerationError
from talu.router import Router

# =============================================================================
# Fixtures
# =============================================================================


def _create_mock_result(
    text: bytes | None = None,
    token_count: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    prefill_ns: int = 0,
    generation_ns: int = 0,
    error_code: int = 0,
    finish_reason: int = 0,
) -> RouterGenerateResult:
    """Create a RouterGenerateResult with proper c_void_p handling for text.

    The generated struct uses c_void_p for text field, so we need to
    convert bytes to a pointer address.
    """
    result = RouterGenerateResult()
    result.token_count = token_count
    result.prompt_tokens = prompt_tokens
    result.completion_tokens = completion_tokens
    result.prefill_ns = prefill_ns
    result.generation_ns = generation_ns
    result.error_code = error_code
    result.finish_reason = finish_reason
    result.tool_calls = None
    result.tool_call_count = 0

    if text is not None:
        # Create a ctypes buffer and get its address
        text_buf = ctypes.create_string_buffer(text)
        result._text_buffer = text_buf  # Keep reference alive
        result.text = ctypes.addressof(text_buf)
    else:
        result.text = None

    return result


@pytest.fixture
def mock_lib():
    """Create a mock library with controllable return values."""
    lib = MagicMock()
    # Default: success result
    lib.talu_router_generate_with_backend.return_value = _create_mock_result(
        text=b"Hello world",
        token_count=2,
        prefill_ns=1000000,
        generation_ns=2000000,
        error_code=0,
    )
    lib.talu_router_stream_with_backend.return_value = _create_mock_result(
        text=b"",
        token_count=0,
        prefill_ns=0,
        generation_ns=0,
        error_code=0,
    )
    lib.talu_router_result_free.return_value = None
    return lib


@pytest.fixture
def mock_chat():
    """Create a mock Chat with a fake pointer."""
    chat = MagicMock()
    chat._chat_ptr = ctypes.c_void_p(0x12345678)
    return chat


@pytest.fixture
def router_with_mock_lib(mock_lib):
    """Create a Router with mocked native library."""
    router = Router(models=["test-model"])
    router._lib = mock_lib
    # Mock backend handle creation - _get_or_create_backend returns a handle
    router._get_or_create_backend = MagicMock(return_value=ctypes.c_void_p(0xDEADBEEF))
    return router


# =============================================================================
# Generate Error Code Tests
# =============================================================================


class TestGenerateErrorCodes:
    """Tests for Router.generate() error code handling."""

    def test_error_code_invalid_chat_handle(self, router_with_mock_lib, mock_chat, mock_lib):
        """Error code -1 (invalid chat handle) raises GenerationError."""
        mock_lib.talu_router_generate_with_backend.return_value = RouterGenerateResult(
            text=None,
            token_count=0,
            prefill_ns=0,
            generation_ns=0,
            error_code=-1,
        )

        with patch("talu._bindings.get_last_error", return_value=None):
            with pytest.raises(GenerationError) as exc:
                router_with_mock_lib.generate(mock_chat, "Hello", model="test-model")

            assert "Invalid chat handle" in str(exc.value)
            assert exc.value.code == "GENERATION_FAILED"

    def test_error_code_invalid_user_message(self, router_with_mock_lib, mock_chat, mock_lib):
        """Error code -2 (invalid user message) raises GenerationError."""
        mock_lib.talu_router_generate_with_backend.return_value = RouterGenerateResult(
            text=None,
            token_count=0,
            prefill_ns=0,
            generation_ns=0,
            error_code=-2,
        )

        with patch("talu._bindings.get_last_error", return_value=None):
            with pytest.raises(GenerationError) as exc:
                router_with_mock_lib.generate(mock_chat, "Hello", model="test-model")

            assert "Invalid user message" in str(exc.value)

    def test_error_code_invalid_model(self, router_with_mock_lib, mock_chat, mock_lib):
        """Error code -3 (invalid model) raises GenerationError."""
        mock_lib.talu_router_generate_with_backend.return_value = RouterGenerateResult(
            text=None,
            token_count=0,
            prefill_ns=0,
            generation_ns=0,
            error_code=-3,
        )

        with patch("talu._bindings.get_last_error", return_value=None):
            with pytest.raises(GenerationError) as exc:
                router_with_mock_lib.generate(mock_chat, "Hello", model="test-model")

            assert "Invalid model" in str(exc.value)

    def test_error_code_engine_creation_failed(self, router_with_mock_lib, mock_chat, mock_lib):
        """Error code -4 (engine creation failed) raises GenerationError."""
        mock_lib.talu_router_generate_with_backend.return_value = RouterGenerateResult(
            text=None,
            token_count=0,
            prefill_ns=0,
            generation_ns=0,
            error_code=-4,
        )

        with patch("talu._bindings.get_last_error", return_value=None):
            with pytest.raises(GenerationError) as exc:
                router_with_mock_lib.generate(mock_chat, "Hello", model="test-model")

            assert "Engine creation failed" in str(exc.value)

    def test_error_code_add_message_failed(self, router_with_mock_lib, mock_chat, mock_lib):
        """Error code -5 (failed to add user message) raises GenerationError."""
        mock_lib.talu_router_generate_with_backend.return_value = RouterGenerateResult(
            text=None,
            token_count=0,
            prefill_ns=0,
            generation_ns=0,
            error_code=-5,
        )

        with patch("talu._bindings.get_last_error", return_value=None):
            with pytest.raises(GenerationError) as exc:
                router_with_mock_lib.generate(mock_chat, "Hello", model="test-model")

            assert "Failed to add user message" in str(exc.value)

    def test_error_code_generation_failed(self, router_with_mock_lib, mock_chat, mock_lib):
        """Error code -6 (generation failed) raises GenerationError."""
        mock_lib.talu_router_generate_with_backend.return_value = RouterGenerateResult(
            text=None,
            token_count=0,
            prefill_ns=0,
            generation_ns=0,
            error_code=-6,
        )

        with patch("talu._bindings.get_last_error", return_value=None):
            with pytest.raises(GenerationError) as exc:
                router_with_mock_lib.generate(mock_chat, "Hello", model="test-model")

            assert "Generation failed" in str(exc.value)

    def test_error_code_memory_allocation_failed(self, router_with_mock_lib, mock_chat, mock_lib):
        """Error code -8 (memory allocation failed) raises GenerationError."""
        mock_lib.talu_router_generate_with_backend.return_value = RouterGenerateResult(
            text=None,
            token_count=0,
            prefill_ns=0,
            generation_ns=0,
            error_code=-8,
        )

        with patch("talu._bindings.get_last_error", return_value=None):
            with pytest.raises(GenerationError) as exc:
                router_with_mock_lib.generate(mock_chat, "Hello", model="test-model")

            assert "Memory allocation failed" in str(exc.value)

    def test_error_code_external_api_not_supported(self, router_with_mock_lib, mock_chat, mock_lib):
        """Error code -10 (external API not supported) raises GenerationError."""
        mock_lib.talu_router_generate_with_backend.return_value = RouterGenerateResult(
            text=None,
            token_count=0,
            prefill_ns=0,
            generation_ns=0,
            error_code=-10,
        )

        with patch("talu._bindings.get_last_error", return_value=None):
            with pytest.raises(GenerationError) as exc:
                router_with_mock_lib.generate(mock_chat, "Hello", model="test-model")

            assert "External API" in str(exc.value) or "not yet supported" in str(exc.value)

    def test_error_code_unknown(self, router_with_mock_lib, mock_chat, mock_lib):
        """Unknown error code raises GenerationError with code in message."""
        mock_lib.talu_router_generate_with_backend.return_value = RouterGenerateResult(
            text=None,
            token_count=0,
            prefill_ns=0,
            generation_ns=0,
            error_code=-999,
        )

        with patch("talu._bindings.get_last_error", return_value=None):
            with pytest.raises(GenerationError) as exc:
                router_with_mock_lib.generate(mock_chat, "Hello", model="test-model")

            assert "-999" in str(exc.value)

    def test_zig_error_message_takes_precedence(self, router_with_mock_lib, mock_chat, mock_lib):
        """Zig error message is used when available."""
        mock_lib.talu_router_generate_with_backend.return_value = RouterGenerateResult(
            text=None,
            token_count=0,
            prefill_ns=0,
            generation_ns=0,
            error_code=-6,
        )

        with patch(
            "talu._bindings.get_last_error", return_value="Model file corrupted at offset 0x1234"
        ):
            with pytest.raises(GenerationError) as exc:
                router_with_mock_lib.generate(mock_chat, "Hello", model="test-model")

            assert "Model file corrupted" in str(exc.value)


# =============================================================================
# Stream Error Code Tests
# =============================================================================


class TestStreamErrorCodes:
    """Tests for Router.stream() error code handling.

    Note: Router.stream() now uses the iterator API (talu_router_create_iterator,
    talu_router_iterator_next, etc.) instead of the callback-based streaming API.
    """

    def test_stream_iterator_creation_fails(self, router_with_mock_lib, mock_chat, mock_lib):
        """Iterator creation failure raises GenerationError."""
        # Mock iterator creation to fail (return None)
        mock_lib.talu_router_create_iterator.return_value = None

        with patch("talu._bindings.get_last_error", return_value="chat_handle is null"):
            with pytest.raises(GenerationError) as exc:
                list(router_with_mock_lib.stream(mock_chat, "Hello", model="test-model"))

            assert "Router.stream() failed to create iterator" in str(exc.value)
            assert "chat_handle is null" in str(exc.value)

    def test_stream_error_code_generation_failed(self, router_with_mock_lib, mock_chat, mock_lib):
        """Error during iteration raises GenerationError in stream."""
        # Mock iterator to be created successfully
        mock_iterator = ctypes.c_void_p(0xABCDEF)
        mock_lib.talu_router_create_iterator.return_value = mock_iterator

        # First call returns None (end of stream), and has_error returns True
        mock_lib.talu_router_iterator_next.return_value = None
        mock_lib.talu_router_iterator_has_error.return_value = True
        mock_lib.talu_router_iterator_error_code.return_value = -6
        mock_lib.talu_router_iterator_free.return_value = None

        with pytest.raises(GenerationError) as exc:
            list(router_with_mock_lib.stream(mock_chat, "Hello", model="test-model"))

        assert "Router.stream() failed with error code -6" in str(exc.value)

    def test_stream_iterator_creation_fails_with_zig_error(
        self, router_with_mock_lib, mock_chat, mock_lib
    ):
        """Zig error message is used when iterator creation fails."""
        mock_lib.talu_router_create_iterator.return_value = None

        with patch(
            "talu._bindings.get_last_error", return_value="Failed to load model: file not found"
        ):
            with pytest.raises(GenerationError) as exc:
                list(router_with_mock_lib.stream(mock_chat, "Hello", model="test-model"))

            assert "Failed to load model" in str(exc.value)


# =============================================================================
# Success Path Tests (with mocks)
# =============================================================================


class TestSuccessPathsMocked:
    """Tests for successful generation with mocked C API."""

    def test_generate_success_returns_dict(self, router_with_mock_lib, mock_chat, mock_lib):
        """Successful generation returns result dict."""
        mock_lib.talu_router_generate_with_backend.return_value = _create_mock_result(
            text=b"Hello, I am an AI assistant.",
            token_count=7,
            prefill_ns=1000000,
            generation_ns=5000000,
            error_code=0,
        )

        result = router_with_mock_lib.generate(mock_chat, "Hello", model="test-model")

        assert result["text"] == "Hello, I am an AI assistant."
        assert result["token_count"] == 7
        assert result["prefill_ns"] == 1000000
        assert result["generation_ns"] == 5000000

    def test_generate_frees_result(self, router_with_mock_lib, mock_chat, mock_lib):
        """Successful generation calls result_free."""
        mock_lib.talu_router_generate_with_backend.return_value = _create_mock_result(
            text=b"Hello",
            token_count=1,
            prefill_ns=0,
            generation_ns=0,
            error_code=0,
        )

        router_with_mock_lib.generate(mock_chat, "Hello", model="test-model")

        mock_lib.talu_router_result_free.assert_called_once()

    def test_generate_with_empty_text_result(self, router_with_mock_lib, mock_chat, mock_lib):
        """Generation with None text returns empty string."""
        mock_lib.talu_router_generate_with_backend.return_value = RouterGenerateResult(
            text=None,
            token_count=0,
            prefill_ns=0,
            generation_ns=0,
            error_code=0,
        )

        result = router_with_mock_lib.generate(mock_chat, "Hello", model="test-model")

        assert result["text"] == ""


# =============================================================================
# Resource Cleanup Tests
# =============================================================================


class TestResourceCleanupOnError:
    """Tests verifying resources are cleaned up even when errors occur."""

    def test_result_free_called_on_success(self, router_with_mock_lib, mock_chat, mock_lib):
        """talu_router_result_free is called after successful generation."""
        mock_lib.talu_router_generate_with_backend.return_value = _create_mock_result(
            text=b"Hello",
            token_count=1,
            prefill_ns=0,
            generation_ns=0,
            error_code=0,
        )

        router_with_mock_lib.generate(mock_chat, "Hello", model="test-model")

        assert mock_lib.talu_router_result_free.call_count == 1

    def test_repeated_generate_calls_dont_leak(self, router_with_mock_lib, mock_chat, mock_lib):
        """Multiple generate calls each free their result."""
        mock_lib.talu_router_generate_with_backend.return_value = _create_mock_result(
            text=b"Hello",
            token_count=1,
            prefill_ns=0,
            generation_ns=0,
            error_code=0,
        )

        for _ in range(10):
            router_with_mock_lib.generate(mock_chat, "Hello", model="test-model")

        assert mock_lib.talu_router_result_free.call_count == 10


# =============================================================================
# Content Type Warning Tests
# =============================================================================


class TestContentTypeWarnings:
    """Tests for unsupported content type warnings."""

    def test_multimodal_content_warns(self, router_with_mock_lib, mock_chat, mock_lib):
        """Multimodal content emits warning but proceeds."""
        mock_lib.talu_router_generate_with_backend.return_value = _create_mock_result(
            text=b"I see an image",
            token_count=4,
            prefill_ns=0,
            generation_ns=0,
            error_code=0,
        )

        # Multimodal content with image
        content = [
            {"type": "text", "text": "Describe this:"},
            {"type": "image", "data": "base64data", "mime": "image/png"},
        ]

        with pytest.warns(UserWarning, match="not yet supported"):
            router_with_mock_lib.generate(mock_chat, content, model="test-model")

    def test_text_only_no_warning(self, router_with_mock_lib, mock_chat, mock_lib):
        """Text-only content does not warn."""
        mock_lib.talu_router_generate_with_backend.return_value = _create_mock_result(
            text=b"Hello",
            token_count=1,
            prefill_ns=0,
            generation_ns=0,
            error_code=0,
        )

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Should not raise any warnings
            router_with_mock_lib.generate(mock_chat, "Hello", model="test-model")


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for Router with mocked C API."""

    def test_generate_with_unicode_result(self, router_with_mock_lib, mock_chat, mock_lib):
        """Generation handles Unicode in result text."""
        mock_lib.talu_router_generate_with_backend.return_value = _create_mock_result(
            text="Hello 你好 🌍".encode(),
            token_count=5,
            prefill_ns=0,
            generation_ns=0,
            error_code=0,
        )

        result = router_with_mock_lib.generate(mock_chat, "Hello", model="test-model")

        assert "你好" in result["text"]
        assert "🌍" in result["text"]

    def test_generate_with_large_token_count(self, router_with_mock_lib, mock_chat, mock_lib):
        """Generation handles large token counts."""
        mock_lib.talu_router_generate_with_backend.return_value = _create_mock_result(
            text=b"x" * 10000,
            token_count=100000,
            prefill_ns=0,
            generation_ns=0,
            error_code=0,
        )

        result = router_with_mock_lib.generate(mock_chat, "Hello", model="test-model")

        assert result["token_count"] == 100000
        assert len(result["text"]) == 10000
