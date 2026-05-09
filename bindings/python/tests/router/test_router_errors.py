"""Router generation error and batch handoff tests.

Maps to: talu/router/router.py
"""

import ctypes
from unittest.mock import MagicMock

import pytest

from talu.exceptions import GenerationError
from talu.router import Router
from talu.router import router as router_mod


@pytest.fixture
def mock_lib():
    """Create a mock native library."""
    return MagicMock()


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
    router._get_or_create_backend = MagicMock(return_value=ctypes.c_void_p(0xDEADBEEF))
    return router


@pytest.fixture
def batch_helpers(monkeypatch):
    append = MagicMock()
    generate = MagicMock(
        return_value={
            "text": "Hello, I am an AI assistant.",
            "token_count": 7,
            "prompt_tokens": 3,
            "completion_tokens": 7,
            "prefill_ns": 1_000_000,
            "generation_ns": 5_000_000,
            "ttft_ns": 500_000,
            "finish_reason": 0,
            "tool_calls": None,
        }
    )
    monkeypatch.setattr(router_mod._c, "router_append_user_message", append)
    monkeypatch.setattr(router_mod._c, "router_generate_batch_final", generate)
    return append, generate


@pytest.fixture
def stream_helpers(monkeypatch):
    append = MagicMock()
    stream = MagicMock(
        return_value={
            "prompt_tokens": 3,
            "completion_tokens": 7,
            "prefill_ns": 1_000_000,
            "generation_ns": 5_000_000,
            "ttft_ns": 500_000,
            "finish_reason": 0,
        }
    )
    monkeypatch.setattr(router_mod._c, "router_append_user_message", append)
    monkeypatch.setattr(router_mod._c, "router_generate_batch_streaming", stream)
    return append, stream


def test_generate_success_returns_batch_result_dict(router_with_mock_lib, mock_chat, batch_helpers):
    """Successful non-stream generation is backed by the batch helper."""
    append, generate = batch_helpers

    result = router_with_mock_lib.generate(mock_chat, "Hello", model="test-model")

    append.assert_called_once_with(router_with_mock_lib._lib, mock_chat._chat_ptr, "Hello")
    generate.assert_called_once()
    assert result["text"] == "Hello, I am an AI assistant."
    assert result["token_count"] == 7
    assert result["prompt_tokens"] == 3
    assert result["completion_tokens"] == 7
    assert result["prefill_ns"] == 1_000_000
    assert result["generation_ns"] == 5_000_000
    assert result["ttft_ns"] == 500_000
    assert result["finish_reason"] == "stop"


def test_generate_text_parts_are_appended_as_one_user_message(
    router_with_mock_lib, mock_chat, batch_helpers
):
    """Text-only content lists remain supported by final-only batch generation."""
    append, _generate = batch_helpers
    content = [
        {"type": "text", "text": "Hello"},
        {"type": "input_text", "text": " world"},
    ]

    result = router_with_mock_lib.generate(mock_chat, content, model="test-model")

    append.assert_called_once_with(router_with_mock_lib._lib, mock_chat._chat_ptr, "Hello world")
    assert result["text"]


def test_generate_non_text_content_fails_before_batch_submit(
    router_with_mock_lib, mock_chat, batch_helpers
):
    """Final-only local generation has no direct multimodal fallback."""
    append, generate = batch_helpers
    content = [
        {"type": "text", "text": "Describe this:"},
        {"type": "image", "data": "base64data", "mime": "image/png"},
    ]

    with pytest.raises(GenerationError, match="multimodal|unsupported content"):
        router_with_mock_lib.generate(mock_chat, content, model="test-model")

    append.assert_not_called()
    generate.assert_not_called()


def test_generate_batch_error_propagates(router_with_mock_lib, mock_chat, batch_helpers):
    """Batch helper failures surface as GenerationError."""
    _append, generate = batch_helpers
    generate.side_effect = GenerationError("Router.generate() failed: batch submit failed")

    with pytest.raises(GenerationError, match="batch submit failed"):
        router_with_mock_lib.generate(mock_chat, "Hello", model="test-model")


def test_stream_success_uses_batch_helper(router_with_mock_lib, mock_chat, stream_helpers):
    """Streaming generation is backed by the batch helper."""
    append, stream = stream_helpers

    def run_stream(_lib, _chat_ptr, _backend, _config, callback):
        callback("Hi", 0, 0, 1, 123)
        return {
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "prefill_ns": 1_000_000,
            "generation_ns": 5_000_000,
            "ttft_ns": 500_000,
            "finish_reason": 0,
        }

    stream.side_effect = run_stream

    tokens = list(router_with_mock_lib.stream(mock_chat, "Hello", model="test-model"))

    append.assert_called_once_with(
        router_with_mock_lib._lib,
        mock_chat._chat_ptr,
        "Hello",
        context="Router.stream()",
    )
    stream.assert_called_once()
    assert [token.text for token in tokens] == ["Hi"]


def test_stream_generation_batch_error_propagates(
    router_with_mock_lib, mock_chat, stream_helpers
):
    """Batch helper failures surface from streaming generation."""
    append, stream = stream_helpers
    stream.side_effect = GenerationError("Router.stream() failed: batch submit failed")

    with pytest.raises(GenerationError, match="batch submit failed"):
        list(router_with_mock_lib.stream(mock_chat, "Hello", model="test-model"))

    append.assert_called_once_with(
        router_with_mock_lib._lib,
        mock_chat._chat_ptr,
        "Hello",
        context="Router.stream()",
    )


def test_stream_generation_error_with_zig_message(
    router_with_mock_lib, mock_chat, stream_helpers
):
    """Zig error message is surfaced when streaming generation fails."""
    _append, stream = stream_helpers
    stream.side_effect = GenerationError(
        "Router.stream() failed: Failed to load model: file not found"
    )

    with pytest.raises(GenerationError, match="Failed to load model"):
        list(router_with_mock_lib.stream(mock_chat, "Hello", model="test-model"))
