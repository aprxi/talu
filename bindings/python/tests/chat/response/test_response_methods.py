"""
Additional tests for talu/chat/response.py coverage.

Targets uncovered edge cases, error paths, and internal methods.
"""

from dataclasses import dataclass

import pytest

from talu.chat.response import (
    AsyncResponse,
    AsyncStreamingResponse,
    FinishReason,
    Response,
    StreamingResponse,
    TokenLogprob,
    Usage,
)
from talu.exceptions import IncompleteJSONError, SchemaValidationError, StateError

# =============================================================================
# Usage Dataclass Tests
# =============================================================================


class TestUsage:
    """Tests for Usage dataclass."""

    def test_usage_creation(self):
        """Usage can be created with all fields."""
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30


# =============================================================================
# TokenLogprob Tests
# =============================================================================


class TestTokenLogprob:
    """Tests for TokenLogprob dataclass."""

    def test_token_logprob_basic(self):
        """TokenLogprob stores basic fields."""
        lp = TokenLogprob(token=100, token_str="hello", logprob=-1.5)
        assert lp.token == 100
        assert lp.token_str == "hello"
        assert lp.logprob == -1.5
        assert lp.top_logprobs is None

    def test_token_logprob_with_top_logprobs(self):
        """TokenLogprob stores top_logprobs."""
        top = [(101, "world", -2.0), (102, "there", -2.5)]
        lp = TokenLogprob(token=100, token_str="hello", logprob=-1.5, top_logprobs=top)
        assert lp.top_logprobs == top


# =============================================================================
# FinishReason Tests
# =============================================================================


class TestFinishReason:
    """Tests for FinishReason constants."""

    def test_finish_reason_constants(self):
        """FinishReason has expected constants."""
        assert FinishReason.EOS_TOKEN == "eos_token"
        assert FinishReason.LENGTH == "length"
        assert FinishReason.STOP_SEQUENCE == "stop_sequence"
        assert FinishReason.TOOL_CALLS == "tool_calls"


# =============================================================================
# Response.parsed Tests
# =============================================================================


class TestResponseParsed:
    """Tests for Response.parsed property."""

    def test_parsed_none_when_no_format(self):
        """parsed returns None when no response_format."""
        response = Response(text='{"key": "value"}')
        assert response.parsed is None

    def test_parsed_returns_dict_for_dict_format(self):
        """parsed returns dict for dict response_format."""
        response = Response(
            text='{"key": "value"}',
            _response_format={"type": "object"},
        )
        assert response.parsed == {"key": "value"}

    def test_parsed_hydrates_dataclass(self):
        """parsed hydrates to dataclass."""

        @dataclass
        class Answer:
            value: int

        response = Response(text='{"value": 42}', _response_format=Answer)
        result = response.parsed
        assert isinstance(result, Answer)
        assert result.value == 42

    def test_parsed_raises_schema_validation_error(self):
        """parsed raises SchemaValidationError on invalid data."""

        @dataclass
        class Answer:
            value: int

        # JSON is valid but doesn't match schema (missing field)
        response = Response(text='{"wrong_field": "x"}', _response_format=Answer)
        with pytest.raises(SchemaValidationError):
            _ = response.parsed

    def test_parsed_incomplete_json_error(self):
        """parsed raises IncompleteJSONError when finish_reason is length."""
        response = Response(
            text='{"key": "va',  # Incomplete JSON
            finish_reason="length",
            _response_format={"type": "object"},
        )
        with pytest.raises(IncompleteJSONError):
            _ = response.parsed


# =============================================================================
# Response._schema_overhead Tests
# =============================================================================


class TestSchemaOverhead:
    """Tests for Response._schema_overhead property."""

    def test__schema_overhead_from_metadata(self):
        """_schema_overhead returns metadata.schema_tokens."""
        from talu.chat.response import ResponseMetadata

        metadata = ResponseMetadata(finish_reason="eos_token", schema_tokens=50)
        response = Response(text="Hello", metadata=metadata)
        assert response._schema_overhead == 50

    def test__schema_overhead_default_zero(self):
        """_schema_overhead defaults to 0."""
        response = Response(text="Hello")
        assert response._schema_overhead == 0


# =============================================================================
# Response.append Tests
# =============================================================================


class TestResponseAppend:
    """Tests for Response.append() method."""

    def test_append_without_chat_raises(self):
        """append() without chat raises StateError."""
        response = Response(text="Hello")
        with pytest.raises(StateError, match="one-shot"):
            response.append("Continue")


# =============================================================================
# Response Repr Tests
# =============================================================================


class TestResponseRepr:
    """Tests for Response __repr__."""

    def test_repr_with_usage(self):
        """Repr includes token count from usage."""
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        response = Response(text="Hello", model="test", usage=usage)
        repr_str = repr(response)
        assert "tokens=30" in repr_str

    def test_repr_truncates_long_text(self):
        """Repr truncates text longer than 50 chars."""
        long_text = "x" * 100
        response = Response(text=long_text)
        repr_str = repr(response)
        assert "..." in repr_str
        assert len(repr_str) < len(long_text) + 50


# =============================================================================
# Response Hash Tests
# =============================================================================


class TestResponseHash:
    """Tests for Response __hash__."""

    def test_response_hashable(self):
        """Response can be hashed."""
        response = Response(text="Hello")
        h = hash(response)
        assert isinstance(h, int)

    def test_equal_responses_same_hash(self):
        """Equal responses have same hash."""
        r1 = Response(text="Hello")
        r2 = Response(text="Hello")
        assert hash(r1) == hash(r2)

    def test_responses_usable_in_set(self):
        """Responses can be used in sets."""
        r1 = Response(text="Hello")
        r2 = Response(text="Hello")
        r3 = Response(text="World")
        s = {r1, r2, r3}
        assert len(s) == 2


# =============================================================================
# Response NotImplemented Equality Tests
# =============================================================================


class TestResponseEquality:
    """Tests for Response equality edge cases."""

    def test_equality_with_non_string_non_response(self):
        """Response returns NotImplemented for non-string/non-response."""
        response = Response(text="Hello")
        assert response.__eq__(42) is NotImplemented
        assert response.__eq__([1, 2, 3]) is NotImplemented


# =============================================================================
# submit_tool_result Tests
# =============================================================================


class TestSubmitToolResult:
    """Tests for submit_tool_result method."""

    def test_submit_tool_result_requires_chat(self):
        """submit_tool_result requires an attached Chat."""
        response = Response(text="Hello")
        with pytest.raises(StateError, match="no Chat session"):
            response.submit_tool_result("call_123", {"result": "ok"})

    def test_submit_tool_result_serializes_and_continues(self):
        """submit_tool_result serializes and calls Chat continuation."""

        class DummyChat:
            def __init__(self):
                self.appended = []
                self.continue_args = None
                self.items = []

            def _append_function_call_output(self, call_id: str, content: str) -> None:
                self.appended.append((call_id, content))

            def _continue_generation(self, *, tools_registry=None):
                self.continue_args = tools_registry
                return Response(text="continued")

        chat = DummyChat()
        response = Response(text="Hello", chat=chat)
        response._tool_registry = {"tool": lambda: "ok"}

        next_resp = response.submit_tool_result("call_123", {"result": "ok"})

        assert next_resp.text == "continued"
        assert chat.appended == [("call_123", '{"result": "ok"}')]
        assert chat.continue_args == response._tool_registry


# =============================================================================
# AsyncResponse Tests
# =============================================================================


class TestAsyncResponseParsed:
    """Tests for AsyncResponse.parsed property."""

    def test_parsed_none_when_no_format(self):
        """parsed returns None when no response_format."""
        response = AsyncResponse(text='{"key": "value"}')
        assert response.parsed is None

    def test_parsed_returns_dict_for_dict_format(self):
        """parsed returns dict for dict response_format."""
        response = AsyncResponse(
            text='{"key": "value"}',
            _response_format={"type": "object"},
        )
        assert response.parsed == {"key": "value"}

    def test_parsed_hydrates_dataclass(self):
        """parsed hydrates to dataclass."""

        @dataclass
        class Answer:
            value: int

        response = AsyncResponse(text='{"value": 42}', _response_format=Answer)
        result = response.parsed
        assert isinstance(result, Answer)
        assert result.value == 42

    def test_parsed_raises_incomplete_json_error(self):
        """parsed raises IncompleteJSONError when finish_reason is length."""
        response = AsyncResponse(
            text='{"key": "va',  # Incomplete JSON
            finish_reason="length",
            _response_format={"type": "object"},
        )
        with pytest.raises(IncompleteJSONError):
            _ = response.parsed

    def test__schema_overhead_from_metadata(self):
        """_schema_overhead returns metadata.schema_tokens."""
        from talu.chat.response import ResponseMetadata

        metadata = ResponseMetadata(finish_reason="eos_token", schema_tokens=75)
        response = AsyncResponse(text="Hello", metadata=metadata)
        assert response._schema_overhead == 75


class TestAsyncSubmitToolResult:
    """Tests for AsyncResponse.submit_tool_result."""

    @pytest.mark.asyncio
    async def test_submit_tool_result_serializes_and_continues(self):
        """submit_tool_result serializes and calls AsyncChat continuation."""

        class DummyAsyncChat:
            def __init__(self):
                self.appended = []
                self.continue_args = None
                self.items = []

            def _append_function_call_output(self, call_id: str, content: str) -> None:
                self.appended.append((call_id, content))

            async def _continue_generation(self, *, tools_registry=None):
                self.continue_args = tools_registry
                return AsyncResponse(text="continued")

        chat = DummyAsyncChat()
        response = AsyncResponse(text="Hello", chat=chat)
        response._tool_registry = {"tool": lambda: "ok"}

        next_resp = await response.submit_tool_result("call_123", {"result": "ok"})

        assert next_resp.text == "continued"
        assert chat.appended == [("call_123", '{"result": "ok"}')]
        assert chat.continue_args == response._tool_registry


class TestAsyncResponseAppend:
    """Tests for AsyncResponse.append() method."""

    @pytest.mark.asyncio
    async def test_append_without_chat_raises(self):
        """append() without chat raises StateError."""
        response = AsyncResponse(text="Hello")
        with pytest.raises(StateError, match="no associated AsyncChat"):
            await response.append("Continue")


class TestAsyncResponseRepr:
    """Tests for AsyncResponse __repr__."""

    def test_repr_basic(self):
        """AsyncResponse has informative repr."""
        response = AsyncResponse(text="Hello", model="test")
        repr_str = repr(response)
        assert "AsyncResponse" in repr_str
        assert "Hello" in repr_str

    def test_repr_with_usage(self):
        """Repr includes token count from usage."""
        usage = Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        response = AsyncResponse(text="Hi", model="m", usage=usage)
        repr_str = repr(response)
        assert "tokens=15" in repr_str


# =============================================================================
# StreamingResponse.append Tests
# =============================================================================


class TestStreamingResponseAppend:
    """Tests for StreamingResponse.append() method."""

    def test_append_without_chat_raises(self):
        """append() without chat raises StateError."""
        response = StreamingResponse(stream_iterator=iter([]))
        with pytest.raises(StateError, match="one-shot"):
            response.append("Continue")


# =============================================================================
# AsyncStreamingResponse.append Tests
# =============================================================================


class TestAsyncStreamingResponseAppend:
    """Tests for AsyncStreamingResponse.append() method."""

    @pytest.mark.asyncio
    async def test_append_without_chat_raises(self):
        """append() without chat raises StateError."""

        async def empty_gen():
            return
            yield  # Make it a generator

        response = AsyncStreamingResponse(async_stream_iterator=empty_gen())
        with pytest.raises(StateError, match="no associated AsyncChat"):
            await response.append("Continue")


# =============================================================================
# Response Properties Tests
# =============================================================================


class TestResponseProperties:
    """Tests for Response property accessors."""

    def test_logprobs_property(self):
        """logprobs property returns stored value."""
        lps = [TokenLogprob(token=1, token_str="hi", logprob=-0.5)]
        response = Response(text="hi", logprobs=lps)
        assert response.logprobs == lps

    def test_logprobs_default_none(self):
        """logprobs defaults to None."""
        response = Response(text="hi")
        assert response.logprobs is None

    def test_tool_calls_property(self):
        """tool_calls property returns stored value."""
        from talu.chat.tools import ToolCall

        calls = [ToolCall.create(id="1", name="search", arguments="{}")]
        response = Response(text="hi", tool_calls=calls)
        assert response.tool_calls == calls

    def test_tool_calls_default_none(self):
        """tool_calls defaults to None."""
        response = Response(text="hi")
        assert response.tool_calls is None

    def test_chat_property(self):
        """chat property returns stored chat."""
        from talu import Chat

        chat = Chat(system="Test")
        response = Response(text="hi", chat=chat)
        assert response.chat is chat

    def test_chat_default_none(self):
        """chat defaults to None."""
        response = Response(text="hi")
        assert response.chat is None
