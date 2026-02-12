"""Tests for talu.chat.session module.

Session-level tests covering Chat initialization, configuration, and lifecycle.
More detailed tests are in test_chat.py, test_lifecycle.py, and test_robustness.py.
"""

from typing import get_overloads

from talu.chat import (
    AsyncChat,
    AsyncResponse,
    AsyncStreamingResponse,
    Chat,
    Response,
    StreamingResponse,
)


class TestChatSession:
    """Tests for Chat session management."""

    def test_create_empty_chat(self):
        """Create chat with no arguments."""
        chat = Chat()
        assert len(chat.items) == 0

    def test_create_with_system(self):
        """Create chat with system message."""
        chat = Chat(system="You are helpful.")
        assert len(chat.items) == 1
        assert chat.items.system == "You are helpful."

    def test_chat_repr(self):
        """Chat has informative repr."""
        chat = Chat()
        repr_str = repr(chat)
        assert "Chat" in repr_str

    def test_chat_str(self):
        """Chat has string representation."""
        chat = Chat(system="Test")
        str_str = str(chat)
        assert isinstance(str_str, str)


class TestChatTypeOverloads:
    """Tests for Chat.__call__ and send() type overloads.

    API design (sync Chat):
    - __call__(msg) -> StreamingResponse (default stream=True)
    - __call__(msg, stream=False) -> Response
    - send(msg) -> Response (default stream=False)
    - send(msg, stream=True) -> StreamingResponse

    For async operations, use AsyncChat instead:
    - AsyncChat.__call__(msg) -> AsyncStreamingResponse (default stream=True)
    - AsyncChat.send(msg) -> AsyncResponse (default stream=False)
    """

    def test_call_has_two_overloads(self):
        """Chat.__call__ has 2 type overloads (stream True/False)."""
        overloads = get_overloads(Chat.__call__)
        assert len(overloads) == 2, f"Expected 2 overloads, got {len(overloads)}"

    def test_send_has_two_overloads(self):
        """Chat.send has 2 type overloads (stream True/False)."""
        overloads = get_overloads(Chat.send)
        assert len(overloads) == 2, f"Expected 2 overloads, got {len(overloads)}"

    def test_overloads_cover_streaming_variants(self):
        """Overloads cover streaming/non-streaming for each method."""
        import inspect

        # Check __call__ overloads
        call_overloads = get_overloads(Chat.__call__)
        call_returns = [str(inspect.signature(ovl).return_annotation) for ovl in call_overloads]
        assert any("Response" in r and "Streaming" not in r for r in call_returns), (
            f"Missing Response return type in __call__, got: {call_returns}"
        )
        assert any("StreamingResponse" in r for r in call_returns), (
            f"Missing StreamingResponse return type in __call__, got: {call_returns}"
        )

    def test_response_types_are_distinct(self):
        """Response types are distinct classes."""
        # Sync types should be different
        assert Response is not StreamingResponse
        assert Response is not AsyncStreamingResponse
        assert StreamingResponse is not AsyncStreamingResponse

        # Async types should be different
        assert AsyncResponse is not AsyncStreamingResponse
        assert AsyncResponse is not Response

        # Sync and Async response types should be different
        assert Response is not AsyncResponse
        assert StreamingResponse is not AsyncStreamingResponse


class TestAsyncChatTypeOverloads:
    """Tests for AsyncChat.__call__ and send() type overloads."""

    def test_async_call_has_two_overloads(self):
        """AsyncChat.__call__ has 2 type overloads (stream True/False)."""
        overloads = get_overloads(AsyncChat.__call__)
        assert len(overloads) == 2, f"Expected 2 overloads, got {len(overloads)}"

    def test_async_send_has_two_overloads(self):
        """AsyncChat.send has 2 type overloads (stream True/False)."""
        overloads = get_overloads(AsyncChat.send)
        assert len(overloads) == 2, f"Expected 2 overloads, got {len(overloads)}"


class TestChatMethodSignatures:
    """Tests that method signatures are correct for new API design."""

    def test_call_accepts_stream_parameter(self):
        """Chat.__call__ accepts stream parameter."""
        import inspect

        sig = inspect.signature(Chat.__call__)
        params = list(sig.parameters.keys())
        assert "stream" in params

    def test_call_no_async_parameter(self):
        """Chat.__call__ no longer has async_ parameter (use AsyncChat instead)."""
        import inspect

        sig = inspect.signature(Chat.__call__)
        params = list(sig.parameters.keys())
        assert "async_" not in params, "async_ parameter removed - use AsyncChat instead"

    def test_send_accepts_stream_parameter(self):
        """Chat.send accepts stream parameter."""
        import inspect

        sig = inspect.signature(Chat.send)
        params = list(sig.parameters.keys())
        assert "stream" in params

    def test_send_no_async_parameter(self):
        """Chat.send no longer has async_ parameter (use AsyncChat instead)."""
        import inspect

        sig = inspect.signature(Chat.send)
        params = list(sig.parameters.keys())
        assert "async_" not in params, "async_ parameter removed - use AsyncChat instead"

    def test_chat_does_not_have_send_async(self):
        """Chat no longer has send_async method (use AsyncChat instead)."""
        assert not hasattr(Chat, "send_async"), (
            "Chat should not have send_async - use AsyncChat instead"
        )


# =============================================================================
# Generation Helper Tests
# =============================================================================


class TestExtractJsonFromResponse:
    """Tests for extract_json_from_response helper."""

    def test_no_thinking_block(self):
        """Returns input unchanged when no thinking block present."""
        from talu.chat._generate import extract_json_from_response

        text = '{"value": 42}'
        assert extract_json_from_response(text) == '{"value": 42}'

    def test_strips_think_block(self):
        """Strips </think> block and returns JSON after it."""
        from talu.chat._generate import extract_json_from_response

        text = '<think>Let me think...</think>{"value": 42}'
        result = extract_json_from_response(text)
        assert result == '{"value": 42}'

    def test_strips_pipe_think_block(self):
        """Strips <|/think|> block (alternate format) and returns JSON after it."""
        from talu.chat._generate import extract_json_from_response

        text = '<|think|>Reasoning here<|/think|>{"answer": "hello"}'
        result = extract_json_from_response(text)
        assert result == '{"answer": "hello"}'

    def test_uses_last_think_end(self):
        """Uses last occurrence of </think> when multiple present."""
        from talu.chat._generate import extract_json_from_response

        text = '<think>First</think>middle<think>Second</think>{"final": true}'
        result = extract_json_from_response(text)
        assert result == '{"final": true}'

    def test_strips_whitespace_after_think(self):
        """Strips whitespace between </think> and JSON."""
        from talu.chat._generate import extract_json_from_response

        text = '<think>Thinking</think>\n\n  {"value": 1}'
        result = extract_json_from_response(text)
        assert result == '{"value": 1}'

    def test_empty_after_think(self):
        """Handles case where nothing follows </think>."""
        from talu.chat._generate import extract_json_from_response

        text = "<think>Thinking only</think>"
        result = extract_json_from_response(text)
        assert result == ""

    def test_empty_input(self):
        """Handles empty input."""
        from talu.chat._generate import extract_json_from_response

        assert extract_json_from_response("") == ""


# Note: Comprehensive Chat tests are in:
# - test_chat.py: Core functionality
# - test_lifecycle.py: Ownership and lifecycle
# - test_robustness.py: Memory, threading, stress tests
# - test_messages.py: Message access patterns
