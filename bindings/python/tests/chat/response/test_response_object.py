"""Tests for talu.chat.response module."""

from unittest.mock import Mock

import pytest

from talu import Chat
from talu.chat.response import (
    AsyncStreamingResponse,
    Response,
    StreamingResponse,
    Timings,
    Usage,
)
from talu.chat.response.types import _ResponseBase
from talu.chat.tools import ToolCall, ToolState
from talu.exceptions import StateError


class TestResponse:
    """Tests for Response class."""

    def test_response_text(self):
        """Response stores text."""
        response = Response(text="Hello!")
        assert response.text == "Hello!"

    def test_response_str(self):
        """Response behaves like string."""
        response = Response(text="Hello!")
        assert str(response) == "Hello!"

    def test_response_finish_reason(self):
        """Response has finish_reason."""
        response = Response(text="Hi", finish_reason="eos_token")
        assert response.finish_reason == "eos_token"

    def test_response_is_not_iterable(self):
        """Response (non-streaming) should not be iterable."""
        response = Response(text="Hello!")
        # Response doesn't have __iter__, so it's not iterable
        assert not hasattr(response, "__iter__") or not callable(
            getattr(response, "__iter__", None)
        )

    def test_response_repr(self):
        """Response has informative repr."""
        response = Response(text="Hello world!", model="test-model")
        repr_str = repr(response)
        assert "Response" in repr_str
        assert "Hello world!" in repr_str
        assert "test-model" in repr_str

    def test_response_contains(self):
        """Response supports 'in' operator."""
        response = Response(text="Hello world!")
        assert "Hello" in response
        assert "world" in response
        assert "foo" not in response

    def test_response_equality_with_string(self):
        """Response can be compared to strings."""
        response = Response(text="Hello!")
        assert response == "Hello!"
        assert response != "Goodbye!"

    def test_response_equality_with_response(self):
        """Response can be compared to other Response."""
        r1 = Response(text="Hello!")
        r2 = Response(text="Hello!")
        r3 = Response(text="Goodbye!")
        assert r1 == r2
        assert r1 != r3

    def test_response_len(self):
        """Response supports len()."""
        response = Response(text="Hello!")
        assert len(response) == 6

    def test_response_concatenation(self):
        """Response can be concatenated with strings."""
        response = Response(text="Hello")
        assert response + " world" == "Hello world"
        assert "Say: " + response == "Say: Hello"

    def test_response_string_methods(self):
        """Response delegates string methods."""
        response = Response(text="Hello World!")
        assert response.lower() == "hello world!"
        assert response.upper() == "HELLO WORLD!"
        assert response.strip() == "Hello World!"
        assert response.split() == ["Hello", "World!"]
        assert response.startswith("Hello")
        assert response.endswith("!")
        assert response.replace("World", "Python") == "Hello Python!"


class TestStreamingResponse:
    """Tests for StreamingResponse class."""

    def test_streaming_response_is_iterable(self):
        """StreamingResponse is iterable."""
        tokens = ["Hello", " ", "world", "!"]
        response = StreamingResponse(stream_iterator=iter(tokens))

        collected = list(response)
        assert collected == tokens

    def test_streaming_response_accumulates_text(self):
        """StreamingResponse accumulates text during iteration."""
        tokens = ["Hello", " ", "world", "!"]
        response = StreamingResponse(stream_iterator=iter(tokens))

        # Before iteration, _text is empty (internal state)
        assert response._text == ""

        # Iterate
        for _ in response:
            pass

        # After iteration, text is accumulated
        assert response.text == "Hello world!"

    def test_streaming_response_text_auto_drains(self):
        """StreamingResponse.text auto-drains stream if not consumed."""
        tokens = ["Hello", " ", "world", "!"]
        response = StreamingResponse(stream_iterator=iter(tokens))

        # Accessing .text should auto-drain the stream
        assert response.text == "Hello world!"
        assert response._stream_exhausted is True

    def test_streaming_response_callback(self):
        """StreamingResponse calls on_token callback."""
        tokens = ["Hello", " ", "world"]
        received = []

        response = StreamingResponse(
            stream_iterator=iter(tokens),
            on_token=lambda t: received.append(t),
        )

        list(response)  # Drain iterator

        assert received == tokens

    def test_streaming_response_on_complete(self):
        """StreamingResponse calls on_complete after iteration."""
        tokens = ["Hello", " ", "world"]
        completed = []

        response = StreamingResponse(
            stream_iterator=iter(tokens),
            on_complete=lambda text: completed.append(text),
        )

        list(response)  # Drain iterator

        assert completed == ["Hello world"]

    def test_streaming_response_exhausted(self):
        """StreamingResponse can only be iterated once."""
        tokens = ["Hello", " ", "world"]
        response = StreamingResponse(stream_iterator=iter(tokens))

        # First iteration
        first = list(response)
        assert first == tokens

        # Second iteration returns empty
        second = list(response)
        assert second == []

    def test_streaming_response_repr(self):
        """StreamingResponse has informative repr."""
        response = StreamingResponse(stream_iterator=iter(["Hello"]))
        repr_str = repr(response)
        assert "StreamingResponse" in repr_str
        assert "pending" in repr_str

        list(response)  # Exhaust

        repr_str = repr(response)
        assert "exhausted" in repr_str

    def test_streaming_response_string_like(self):
        """StreamingResponse has string-like behavior."""
        response = StreamingResponse(stream_iterator=iter(["Hello", " ", "world!"]))
        list(response)  # Must iterate first to accumulate text

        # String-like operations work on accumulated text
        assert str(response) == "Hello world!"
        assert "world" in response
        assert len(response) == 12
        assert response.lower() == "hello world!"


class TestAsyncStreamingResponse:
    """Tests for AsyncStreamingResponse class."""

    @pytest.mark.asyncio
    async def test_async_streaming_response_is_async_iterable(self):
        """AsyncStreamingResponse is async-iterable."""

        async def async_gen():
            for token in ["Hello", " ", "world", "!"]:
                yield token

        response = AsyncStreamingResponse(async_stream_iterator=async_gen())

        collected = []
        async for token in response:
            collected.append(token)

        assert collected == ["Hello", " ", "world", "!"]

    @pytest.mark.asyncio
    async def test_async_streaming_response_accumulates_text(self):
        """AsyncStreamingResponse accumulates text during iteration."""

        async def async_gen():
            for token in ["Hello", " ", "world", "!"]:
                yield token

        response = AsyncStreamingResponse(async_stream_iterator=async_gen())

        # Before iteration
        assert response.text == ""

        async for _ in response:
            pass

        # After iteration
        assert response.text == "Hello world!"

    @pytest.mark.asyncio
    async def test_async_streaming_response_callback(self):
        """AsyncStreamingResponse calls on_token callback."""
        received = []

        async def async_gen():
            for token in ["Hello", " ", "world"]:
                yield token

        response = AsyncStreamingResponse(
            async_stream_iterator=async_gen(),
            on_token=lambda t: received.append(t),
        )

        async for _ in response:
            pass

        assert received == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_async_streaming_response_on_complete(self):
        """AsyncStreamingResponse calls on_complete after iteration."""
        completed = []

        async def async_gen():
            for token in ["Hello", " ", "world"]:
                yield token

        response = AsyncStreamingResponse(
            async_stream_iterator=async_gen(),
            on_complete=lambda text: completed.append(text),
        )

        async for _ in response:
            pass

        assert completed == ["Hello world"]

    @pytest.mark.asyncio
    async def test_async_streaming_response_exhausted(self):
        """AsyncStreamingResponse can only be iterated once."""

        async def async_gen():
            for token in ["Hello", " ", "world"]:
                yield token

        response = AsyncStreamingResponse(async_stream_iterator=async_gen())

        # First iteration
        first = []
        async for token in response:
            first.append(token)
        assert first == ["Hello", " ", "world"]

        # Second iteration returns empty
        second = []
        async for token in response:
            second.append(token)
        assert second == []

    def test_async_streaming_response_repr(self):
        """AsyncStreamingResponse has informative repr."""

        async def async_gen():
            yield "Hello"

        response = AsyncStreamingResponse(async_stream_iterator=async_gen())
        repr_str = repr(response)
        assert "AsyncStreamingResponse" in repr_str
        assert "pending" in repr_str


class TestResponseTypeHierarchy:
    """Tests for response type hierarchy."""

    def test_response_inherits_from_base(self):
        """Response inherits from _ResponseBase."""
        assert issubclass(Response, _ResponseBase)

    def test_streaming_response_inherits_from_base(self):
        """StreamingResponse inherits from _ResponseBase."""
        assert issubclass(StreamingResponse, _ResponseBase)

    def test_async_streaming_response_inherits_from_base(self):
        """AsyncStreamingResponse inherits from _ResponseBase."""
        assert issubclass(AsyncStreamingResponse, _ResponseBase)

    def test_response_not_streaming(self):
        """Response is not StreamingResponse."""
        assert not issubclass(Response, StreamingResponse)

    def test_streaming_not_response(self):
        """StreamingResponse is not Response."""
        assert not issubclass(StreamingResponse, Response)

    def test_all_share_common_properties(self):
        """All response types share common properties."""
        response = Response(text="test", model="m1")
        streaming = StreamingResponse(stream_iterator=iter([]), model="m2")

        # Both have same properties from base
        assert hasattr(response, "text")
        assert hasattr(response, "model")
        assert hasattr(response, "usage")
        assert hasattr(response, "timings")
        assert hasattr(response, "finish_reason")

        assert hasattr(streaming, "text")
        assert hasattr(streaming, "model")
        assert hasattr(streaming, "usage")
        assert hasattr(streaming, "timings")
        assert hasattr(streaming, "finish_reason")


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_tool_call_create(self):
        """Create ToolCall via factory method."""
        tc = ToolCall.create(
            id="call_abc",
            name="search",
            arguments='{"query": "test"}',
        )
        assert tc.id == "call_abc"
        assert tc.name == "search"
        assert tc.arguments == '{"query": "test"}'

    def test_tool_call_type_is_function(self):
        """ToolCall type is always 'function'."""
        tc = ToolCall.create(id="x", name="y", arguments="{}")
        assert tc.type == "function"


class TestToolState:
    """Tests for ToolState dataclass."""

    def test_tool_state_pending(self):
        """Create pending tool state."""
        state = ToolState(
            status="pending",
            title="Fetching...",
        )
        assert state.status == "pending"
        assert state.output is None
        assert state.error is None

    def test_tool_state_completed(self):
        """Tool state with output."""
        state = ToolState(
            status="completed",
            title="Done",
            output="Success",
        )
        assert state.status == "completed"
        assert state.output == "Success"

    def test_tool_state_error(self):
        """Tool state with error."""
        state = ToolState(
            status="error",
            title="Failed",
            error="Connection timeout",
        )
        assert state.status == "error"
        assert state.error == "Connection timeout"


class TestTimings:
    """Tests for Timings dataclass."""

    def test_timings_from_ns_basic(self):
        """Timings.from_ns converts nanoseconds to milliseconds."""
        timings = Timings.from_ns(
            prefill_ns=100_000_000,  # 100ms
            generation_ns=500_000_000,  # 500ms
            token_count=50,
        )
        assert abs(timings.prefill_ms - 100.0) < 0.001
        assert abs(timings.generation_ms - 500.0) < 0.001

    def test_timings_tokens_per_second(self):
        """Timings calculates tokens per second correctly."""
        timings = Timings.from_ns(
            prefill_ns=100_000_000,
            generation_ns=1_000_000_000,  # 1 second
            token_count=100,
        )
        # 100 tokens / 1 second = 100 tokens/sec
        assert abs(timings.tokens_per_second - 100.0) < 0.001

    def test_timings_zero_generation_time(self):
        """Timings handles zero generation time gracefully."""
        timings = Timings.from_ns(
            prefill_ns=100_000_000,
            generation_ns=0,
            token_count=10,
        )
        assert timings.tokens_per_second == 0.0

    def test_timings_zero_tokens(self):
        """Timings handles zero tokens."""
        timings = Timings.from_ns(
            prefill_ns=100_000_000,
            generation_ns=500_000_000,
            token_count=0,
        )
        assert timings.tokens_per_second == 0.0

    def test_timings_small_values(self):
        """Timings handles small nanosecond values."""
        timings = Timings.from_ns(
            prefill_ns=1_000,  # 0.001ms
            generation_ns=2_000,  # 0.002ms
            token_count=1,
        )
        assert abs(timings.prefill_ms - 0.001) < 0.0001
        assert abs(timings.generation_ms - 0.002) < 0.0001

    def test_response_with_timings(self):
        """Response stores timings."""
        timings = Timings.from_ns(100_000_000, 200_000_000, 20)
        response = Response(text="Hello!", timings=timings)
        assert response.timings is not None
        assert abs(response.timings.prefill_ms - 100.0) < 0.001
        assert abs(response.timings.generation_ms - 200.0) < 0.001

    def test_response_timings_default_none(self):
        """Response timings defaults to None."""
        response = Response(text="Hello!")
        assert response.timings is None


class TestResponseMsgIndex:
    """Tests for Response message index tracking."""

    def test_msg_index_computed_from_chat(self):
        """Response computes _msg_index from chat messages length."""
        chat = Chat(system="You are helpful.")
        # Chat starts with 1 message (system)
        # After calling chat(), it will have 3 messages (system, user, assistant)
        response = Response(text="Hello!", chat=chat)
        # With 1 message (system), last index is 0
        assert response._msg_index == 0

    def test_msg_index_explicit(self):
        """Response accepts explicit _msg_index."""
        response = Response(text="Hello!", _msg_index=5)
        assert response._msg_index == 5

    def test_msg_index_no_chat(self):
        """Response without chat has _msg_index -1."""
        response = Response(text="Hello!")
        assert response._msg_index == -1


class TestResponseAutoFork:
    """Tests for Response.append() auto-fork behavior."""

    @pytest.mark.requires_model
    def test_linear_append_uses_same_chat(self, test_model_path):
        """Linear append (at tip) continues on same chat."""
        chat = Chat(test_model_path)

        r1 = chat.send("Hello", max_tokens=3)
        original_chat = r1.chat

        # Append at tip - should use same chat
        r2 = r1.append("Continue", max_tokens=3)

        assert r2.chat is original_chat
        # Chat should have 4 messages: system? + user + assistant + user + assistant
        # or just user + assistant + user + assistant if no system
        assert len(chat.items) >= 4

    @pytest.mark.requires_model
    def test_branching_append_forks_chat(self, test_model_path):
        """Branching append (past tip) auto-forks."""
        chat = Chat(test_model_path)

        r1 = chat.send("Idea 1", max_tokens=3)
        original_len = len(chat.items)

        # Continue the chat past r1
        r2 = r1.append("Critique it", max_tokens=3)
        assert len(chat.items) > original_len

        # Now append to r1 again - should fork
        r3 = r1.append("Expand on it", max_tokens=3)

        # r3 should be on a different chat
        assert r3.chat is not chat
        assert r3.chat is not r2.chat

        # Original chat should be unaffected
        # (still has the critique branch)
        assert len(chat.items) > original_len

        # Forked chat should have branched from r1
        # (should have: user "Idea 1", assistant response, user "Expand", assistant response)
        # The exact count depends on whether there's a system message

    @pytest.mark.requires_model
    def test_multiple_branches_from_same_response(self, test_model_path):
        """Can create multiple branches from same response."""
        chat = Chat(test_model_path)

        r1 = chat.send("Base idea", max_tokens=3)

        # Create first branch
        branch1 = r1.append("Direction A", max_tokens=3)
        # Create second branch (both from r1)
        branch2 = r1.append("Direction B", max_tokens=3)
        # Create third branch
        branch3 = r1.append("Direction C", max_tokens=3)

        # All branches should be on different chats
        chats = {branch1.chat, branch2.chat, branch3.chat}
        assert len(chats) == 3  # All unique

        # Original chat still has first branch (Direction A)
        assert branch1.chat is chat

    def test_truncate_to_removes_messages(self):
        """_truncate_to removes messages after specified index."""
        chat = Chat(system="System")
        # Manually add messages to test truncation
        # (In real usage, messages come from generation)

        # For this test, we'll verify the method exists and works
        # by checking the fork + truncate pattern
        initial_len = len(chat.items)  # Should be 1 (system)
        assert initial_len == 1

        # Fork and verify
        forked = chat.fork()
        assert len(forked.items) == 1

        # Truncate to index 0 should keep just first message
        forked._truncate_to(0)
        assert len(forked.items) == 1


class TestResponseTokensProperty:
    """Tests for Response.tokens property (line 393)."""

    def test_tokens_property_returns_empty_list_by_default(self):
        """Response.tokens returns empty list when no tokens provided."""
        response = Response(text="Hello!")
        assert response.tokens == []

    def test_tokens_property_returns_provided_tokens(self):
        """Response.tokens returns the token IDs that were set."""
        response = Response(text="Hello!", tokens=[1, 2, 3, 4])
        assert response.tokens == [1, 2, 3, 4]

    def test_tokens_property_is_list(self):
        """Response.tokens is always a list."""
        response = Response(text="Hello!", tokens=[100, 200])
        assert isinstance(response.tokens, list)


class TestResponseContentProperty:
    """Tests for Response.content property caching (line 457)."""

    def test_content_property_returns_output_text(self):
        """Response.content returns OutputText for text responses."""
        from talu.types import ContentType, OutputText

        response = Response(text="Hello world!")
        content = response.content
        assert len(content) == 1
        assert isinstance(content[0], OutputText)
        assert content[0].text == "Hello world!"
        assert content[0].type == ContentType.OUTPUT_TEXT

    def test_content_property_caching_returns_same_list(self):
        """Response.content returns cached content on second access."""
        from talu.types import OutputText

        # Set up cached content at construction time via _content parameter
        cached = [OutputText(text="cached")]
        response = Response(text="Hello!", _content=cached)

        # Should return cached content
        assert response.content is cached

    def test_content_property_empty_text(self):
        """Response.content works with empty text."""
        from talu.types import OutputText

        response = Response(text="")
        content = response.content
        assert len(content) == 1
        assert isinstance(content[0], OutputText)
        assert content[0].text == ""


class TestResponseToDict:
    """Tests for Response.to_dict() with optional fields (lines 595-631)."""

    def test_to_dict_basic(self):
        """to_dict() includes text and finish_reason."""
        response = Response(text="Hello!", finish_reason="eos_token")
        result = response.to_dict()
        assert result["text"] == "Hello!"
        assert result["finish_reason"] == "eos_token"

    def test_to_dict_with_model(self):
        """to_dict() includes model when present."""
        response = Response(text="Hi", model="test-model")
        result = response.to_dict()
        assert result["model"] == "test-model"

    def test_to_dict_without_model(self):
        """to_dict() omits model when not present."""
        response = Response(text="Hi")
        result = response.to_dict()
        assert "model" not in result

    def test_to_dict_with_usage(self):
        """to_dict() includes usage statistics when present."""
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        response = Response(text="Hi", usage=usage)
        result = response.to_dict()
        assert "usage" in result
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 20
        assert result["usage"]["total_tokens"] == 30

    def test_to_dict_without_usage(self):
        """to_dict() omits usage when not present."""
        response = Response(text="Hi")
        result = response.to_dict()
        assert "usage" not in result

    def test_to_dict_with_timings(self):
        """to_dict() includes timings when present."""
        timings = Timings(prefill_ms=100.0, generation_ms=500.0, tokens_per_second=50.0)
        response = Response(text="Hi", timings=timings)
        result = response.to_dict()
        assert "timings" in result
        assert result["timings"]["prefill_ms"] == 100.0
        assert result["timings"]["generation_ms"] == 500.0
        assert result["timings"]["total_ms"] == 600.0
        assert result["timings"]["tokens_per_second"] == 50.0

    def test_to_dict_without_timings(self):
        """to_dict() omits timings when not present."""
        response = Response(text="Hi")
        result = response.to_dict()
        assert "timings" not in result

    def test_to_dict_with_tool_calls(self):
        """to_dict() includes tool_calls when present."""
        tool_call = ToolCall.create(
            id="call_123",
            name="search",
            arguments='{"query": "python"}',
        )
        response = Response(text="", tool_calls=[tool_call])
        result = response.to_dict()
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_123"
        assert result["tool_calls"][0]["type"] == "function"
        assert result["tool_calls"][0]["function"]["name"] == "search"
        assert result["tool_calls"][0]["function"]["arguments"] == '{"query": "python"}'

    def test_to_dict_without_tool_calls(self):
        """to_dict() omits tool_calls when not present."""
        response = Response(text="Hi")
        result = response.to_dict()
        assert "tool_calls" not in result

    def test_to_dict_with_all_fields(self):
        """to_dict() works with all optional fields present."""
        usage = Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        timings = Timings(prefill_ms=50.0, generation_ms=100.0, tokens_per_second=100.0)
        tool_call = ToolCall.create(id="tc", name="fn", arguments="{}")
        response = Response(
            text="Output",
            finish_reason="tool_calls",
            model="my-model",
            usage=usage,
            timings=timings,
            tool_calls=[tool_call],
        )
        result = response.to_dict()
        assert result["text"] == "Output"
        assert result["finish_reason"] == "tool_calls"
        assert result["model"] == "my-model"
        assert "usage" in result
        assert "timings" in result
        assert "tool_calls" in result


class TestStreamingResponsePrompt:
    """Tests for StreamingResponse.prompt property (lines 1101, 1109)."""

    def test_prompt_returns_explicit_value(self):
        """StreamingResponse.prompt returns explicitly set value."""
        response = StreamingResponse(stream_iterator=iter([]))
        response._prompt = "explicit prompt"
        assert response.prompt == "explicit prompt"

    def test_prompt_lazy_capture_after_exhausted(self):
        """StreamingResponse.prompt captures from chat after stream exhausted."""
        # Create mock chat
        mock_chat = Mock()
        mock_chat.preview_prompt.return_value = "rendered prompt"
        mock_chat.items = []  # For msg_index calculation

        response = StreamingResponse(
            stream_iterator=iter(["hello"]),
            chat=mock_chat,
        )
        # Exhaust the stream
        list(response)

        # Now prompt should be captured from chat
        assert response.prompt == "rendered prompt"
        mock_chat.preview_prompt.assert_called_with(add_generation_prompt=False)

    def test_prompt_returns_none_if_not_exhausted(self):
        """StreamingResponse.prompt returns None if stream not exhausted."""
        mock_chat = Mock()
        mock_chat.items = []  # Provide items for len() calculation
        response = StreamingResponse(
            stream_iterator=iter(["hello"]),
            chat=mock_chat,
        )
        # Don't exhaust the stream
        assert response.prompt is None

    def test_prompt_returns_none_if_no_chat(self):
        """StreamingResponse.prompt returns None without chat."""
        response = StreamingResponse(stream_iterator=iter(["hello"]))
        list(response)  # Exhaust
        assert response.prompt is None

    def test_prompt_handles_state_error_gracefully(self):
        """StreamingResponse.prompt handles StateError from chat.preview_prompt."""
        mock_chat = Mock()
        mock_chat.preview_prompt.side_effect = StateError("closed", code="STATE_CLOSED")
        mock_chat.items = []

        response = StreamingResponse(
            stream_iterator=iter(["hello"]),
            chat=mock_chat,
        )
        list(response)

        # Should return None instead of raising
        assert response.prompt is None


class TestStreamingResponseAppend:
    """Tests for StreamingResponse.append() (line 1296)."""

    def test_append_raises_without_chat(self):
        """append() raises StateError when no chat associated."""
        response = StreamingResponse(stream_iterator=iter([]))
        list(response)

        with pytest.raises(StateError) as exc_info:
            response.append("Continue")
        assert "Cannot append to a one-shot response" in str(exc_info.value)
        assert exc_info.value.code == "STATE_NO_CHAT"

    def test_append_linear_continues_on_same_chat(self):
        """append() at tip continues on same chat."""
        mock_chat = Mock()
        mock_chat.items = [Mock(), Mock()]  # 2 items, last index = 1
        mock_chat.send.return_value = StreamingResponse(stream_iterator=iter([]))

        response = StreamingResponse(
            stream_iterator=iter(["token"]),
            chat=mock_chat,
        )
        # msg_index is computed from len(chat.items) - 1 = 1 at construction
        list(response)

        result = response.append("Continue")
        mock_chat.send.assert_called_once_with("Continue", stream=True)
        assert result is mock_chat.send.return_value

    def test_append_branching_forks_chat(self):
        """append() past tip auto-forks chat."""
        mock_forked_chat = Mock()
        mock_forked_chat.send.return_value = StreamingResponse(stream_iterator=iter([]))

        # Start with 2 items, msg_index will be 1
        mock_chat = Mock()
        mock_chat.items = [Mock(), Mock()]

        response = StreamingResponse(
            stream_iterator=iter(["token"]),
            chat=mock_chat,
        )
        # Now add more items to simulate conversation moving past this response
        mock_chat.items = [Mock(), Mock(), Mock(), Mock()]  # 4 items, last index = 3
        mock_chat._fork_at.return_value = mock_forked_chat

        list(response)

        result = response.append("Branch")
        mock_chat._fork_at.assert_called_once_with(1)
        mock_forked_chat.send.assert_called_once_with("Branch", stream=True)
        assert result is mock_forked_chat.send.return_value


class TestAsyncResponseAppend:
    """Tests for AsyncResponse.append() auto-fork (lines 986-987)."""

    @pytest.mark.asyncio
    async def test_async_append_raises_without_chat(self):
        """AsyncResponse.append() raises StateError when no chat."""
        from talu.chat.response import AsyncResponse

        response = AsyncResponse(text="Hello!")
        # No chat attached, msg_index will be -1

        with pytest.raises(StateError) as exc_info:
            await response.append("Continue")
        assert "no associated AsyncChat" in str(exc_info.value)
        assert exc_info.value.code == "STATE_NO_CHAT"

    @pytest.mark.asyncio
    async def test_async_append_linear_continues_on_same_chat(self):
        """AsyncResponse.append() at tip continues on same chat."""
        from unittest.mock import AsyncMock

        from talu.chat.response import AsyncResponse

        mock_chat = Mock()
        mock_chat.items = [Mock(), Mock()]  # 2 items, msg_index = 1
        mock_response = AsyncResponse(text="reply")
        # AsyncChat.send is async, so use AsyncMock
        mock_chat.send = AsyncMock(return_value=mock_response)

        response = AsyncResponse(
            text="Hello!",
            chat=mock_chat,
        )
        # msg_index is computed from len(chat.items) - 1 = 1 at construction
        response._stream_mode = False

        result = await response.append("Continue")
        mock_chat.send.assert_called_once_with("Continue", stream=False)
        assert result is mock_response

    @pytest.mark.asyncio
    async def test_async_append_branching_forks_chat(self):
        """AsyncResponse.append() past tip auto-forks chat."""
        from unittest.mock import AsyncMock

        from talu.chat.response import AsyncResponse

        mock_forked_chat = Mock()
        mock_forked_response = AsyncResponse(text="forked reply")
        # AsyncChat.send is async, so use AsyncMock
        mock_forked_chat.send = AsyncMock(return_value=mock_forked_response)

        # Start with 2 items, msg_index will be 1
        mock_chat = Mock()
        mock_chat.items = [Mock(), Mock()]

        response = AsyncResponse(
            text="Hello!",
            chat=mock_chat,
        )
        # Now add more items to simulate conversation moving past this response
        mock_chat.items = [Mock(), Mock(), Mock(), Mock()]  # 4 items, last index = 3
        mock_chat._fork_at.return_value = mock_forked_chat
        response._stream_mode = False

        result = await response.append("Branch")
        mock_chat._fork_at.assert_called_once_with(1)
        mock_forked_chat.send.assert_called_once_with("Branch", stream=False)
        assert result is mock_forked_response


class TestAsyncStreamingResponsePrompt:
    """Tests for AsyncStreamingResponse.prompt (lines 1433, 1441)."""

    def test_async_prompt_returns_explicit_value(self):
        """AsyncStreamingResponse.prompt returns explicitly set value."""

        async def empty_gen():
            return
            yield  # noqa: RET503

        response = AsyncStreamingResponse(async_stream_iterator=empty_gen())
        response._prompt = "explicit async prompt"
        assert response.prompt == "explicit async prompt"

    @pytest.mark.asyncio
    async def test_async_prompt_lazy_capture_after_exhausted(self):
        """AsyncStreamingResponse.prompt captures from chat after exhausted."""
        mock_chat = Mock()
        mock_chat.preview_prompt.return_value = "rendered async prompt"
        mock_chat.items = []

        async def token_gen():
            yield "hello"

        response = AsyncStreamingResponse(
            async_stream_iterator=token_gen(),
            chat=mock_chat,
        )
        # Exhaust the stream
        async for _ in response:
            pass

        assert response.prompt == "rendered async prompt"
        mock_chat.preview_prompt.assert_called_with(add_generation_prompt=False)

    @pytest.mark.asyncio
    async def test_async_prompt_returns_none_if_not_exhausted(self):
        """AsyncStreamingResponse.prompt returns None if stream not exhausted."""
        mock_chat = Mock()
        mock_chat.items = []  # Provide items for len() calculation

        async def token_gen():
            yield "hello"

        response = AsyncStreamingResponse(
            async_stream_iterator=token_gen(),
            chat=mock_chat,
        )
        # Don't exhaust
        assert response.prompt is None

    @pytest.mark.asyncio
    async def test_async_prompt_handles_error_gracefully(self):
        """AsyncStreamingResponse.prompt handles errors from preview_prompt."""
        from talu.exceptions import StateError

        mock_chat = Mock()
        mock_chat.preview_prompt.side_effect = StateError("no engine", code="STATE_NO_ENGINE")
        mock_chat.items = []

        async def token_gen():
            yield "hello"

        response = AsyncStreamingResponse(
            async_stream_iterator=token_gen(),
            chat=mock_chat,
        )
        async for _ in response:
            pass

        # Should return None instead of raising
        assert response.prompt is None


class TestAsyncStreamingResponseIterationErrorHandling:
    """Tests for AsyncStreamingResponse iteration error handling (lines 1499, 1510-1512)."""

    @pytest.mark.asyncio
    async def test_async_iteration_handles_exception(self):
        """AsyncStreamingResponse handles exceptions during iteration."""

        async def failing_gen():
            yield "hello"
            raise ValueError("generation failed")

        response = AsyncStreamingResponse(async_stream_iterator=failing_gen())

        with pytest.raises(ValueError, match="generation failed"):
            async for _ in response:
                pass

        # Stream should be marked exhausted even after error
        assert response._stream_exhausted is True

    @pytest.mark.asyncio
    async def test_async_iteration_on_complete_not_called_on_error(self):
        """on_complete callback is not called when iteration errors."""
        completed = []

        async def failing_gen():
            yield "hello"
            raise ValueError("failed")

        response = AsyncStreamingResponse(
            async_stream_iterator=failing_gen(),
            on_complete=lambda text: completed.append(text),
        )

        with pytest.raises(ValueError):
            async for _ in response:
                pass

        # on_complete should NOT be called when there's an error
        assert completed == []


class TestAsyncStreamingResponseAppend:
    """Tests for AsyncStreamingResponse.append() auto-fork (lines 1557-1565)."""

    @pytest.mark.asyncio
    async def test_async_streaming_append_raises_without_chat(self):
        """AsyncStreamingResponse.append() raises StateError when no chat."""

        async def token_gen():
            yield "hello"

        response = AsyncStreamingResponse(async_stream_iterator=token_gen())
        async for _ in response:
            pass

        with pytest.raises(StateError) as exc_info:
            await response.append("Continue")
        assert "no associated AsyncChat" in str(exc_info.value)
        assert exc_info.value.code == "STATE_NO_CHAT"

    @pytest.mark.asyncio
    async def test_async_streaming_append_linear_continues_on_same_chat(self):
        """AsyncStreamingResponse.append() at tip continues on same chat."""
        from unittest.mock import AsyncMock

        async def token_gen():
            yield "hello"

        async def append_gen():
            yield "reply"

        mock_append = AsyncStreamingResponse(async_stream_iterator=append_gen())

        mock_chat = Mock()
        mock_chat.items = [Mock(), Mock()]  # 2 items, msg_index = 1
        # AsyncChat.send is async, so use AsyncMock
        mock_chat.send = AsyncMock(return_value=mock_append)

        response = AsyncStreamingResponse(
            async_stream_iterator=token_gen(),
            chat=mock_chat,
        )
        # msg_index is computed from len(chat.items) - 1 = 1 at construction
        async for _ in response:
            pass

        result = await response.append("Continue")
        mock_chat.send.assert_called_once_with("Continue", stream=True)
        assert result is mock_append

    @pytest.mark.asyncio
    async def test_async_streaming_append_branching_forks_chat(self):
        """AsyncStreamingResponse.append() past tip auto-forks chat."""
        from unittest.mock import AsyncMock

        async def token_gen():
            yield "hello"

        async def append_gen():
            yield "forked reply"

        mock_forked_response = AsyncStreamingResponse(async_stream_iterator=append_gen())

        mock_forked_chat = Mock()
        # AsyncChat.send is async, so use AsyncMock
        mock_forked_chat.send = AsyncMock(return_value=mock_forked_response)

        # Start with 2 items, msg_index will be 1
        mock_chat = Mock()
        mock_chat.items = [Mock(), Mock()]

        response = AsyncStreamingResponse(
            async_stream_iterator=token_gen(),
            chat=mock_chat,
        )
        # Now add more items to simulate conversation moving past this response
        mock_chat.items = [Mock(), Mock(), Mock(), Mock()]  # 4 items, last index = 3
        mock_chat._fork_at.return_value = mock_forked_chat

        async for _ in response:
            pass

        result = await response.append("Branch")
        mock_chat._fork_at.assert_called_once_with(1)
        mock_forked_chat.send.assert_called_once_with("Branch", stream=True)
        assert result is mock_forked_response
