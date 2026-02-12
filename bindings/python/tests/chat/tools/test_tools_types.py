"""
Tests for talu.chat.tools module.

Tests for tool calling types: ToolCall, ToolCallFunction, ToolState, ToolResult.
"""

import json

from talu.chat.tools import (
    ToolCall,
    ToolCallFunction,
    ToolResult,
    ToolState,
    ToolStatus,
)


class TestToolCallFunction:
    """Tests for ToolCallFunction dataclass."""

    def test_creation(self):
        """ToolCallFunction can be created with name and arguments."""
        func = ToolCallFunction(name="get_weather", arguments='{"city": "Paris"}')

        assert func.name == "get_weather"
        assert func.arguments == '{"city": "Paris"}'

    def test_arguments_parsed_returns_dict(self):
        """arguments_parsed() parses JSON arguments to dict."""
        func = ToolCallFunction(name="search", arguments='{"query": "python", "limit": 10}')

        parsed = func.arguments_parsed()

        assert isinstance(parsed, dict)
        assert parsed == {"query": "python", "limit": 10}

    def test_arguments_parsed_empty_string(self):
        """arguments_parsed() returns empty dict for empty string."""
        func = ToolCallFunction(name="ping", arguments="")

        assert func.arguments_parsed() == {}

    def test_arguments_parsed_complex_json(self):
        """arguments_parsed() handles nested JSON."""
        args = json.dumps({"filters": {"min": 0, "max": 100}, "options": ["a", "b"]})
        func = ToolCallFunction(name="filter", arguments=args)

        parsed = func.arguments_parsed()

        assert parsed["filters"]["min"] == 0
        assert parsed["options"] == ["a", "b"]

    def test_arguments_parsed_invalid_json_returns_empty(self):
        """arguments_parsed() returns empty dict on invalid JSON."""
        func = ToolCallFunction(name="bad", arguments='{"incomplete": ')

        assert func.arguments_parsed() == {}


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_creation(self):
        """ToolCall can be created with id, type, and function."""
        func = ToolCallFunction(name="get_time", arguments="{}")
        tool = ToolCall(id="call_123", type="function", function=func)

        assert tool.id == "call_123"
        assert tool.type == "function"
        assert tool.function.name == "get_time"

    def test_create_classmethod(self):
        """ToolCall.create() provides convenient construction."""
        tool = ToolCall.create(id="call_456", name="search", arguments='{"q": "test"}')

        assert tool.id == "call_456"
        assert tool.type == "function"
        assert tool.function.name == "search"
        assert tool.function.arguments == '{"q": "test"}'

    def test_name_property(self):
        """name property provides convenience access to function.name."""
        tool = ToolCall.create(id="call_789", name="get_weather", arguments="{}")

        assert tool.name == "get_weather"
        assert tool.name == tool.function.name

    def test_arguments_property(self):
        """arguments property provides convenience access to function.arguments."""
        tool = ToolCall.create(id="call_abc", name="test", arguments='{"key": "value"}')

        assert tool.arguments == '{"key": "value"}'
        assert tool.arguments == tool.function.arguments


class TestToolStatus:
    """Tests for ToolStatus constants."""

    def test_status_constants(self):
        """ToolStatus defines expected status constants."""
        assert ToolStatus.PENDING == "pending"
        assert ToolStatus.RUNNING == "running"
        assert ToolStatus.COMPLETED == "completed"
        assert ToolStatus.ERROR == "error"


class TestToolState:
    """Tests for ToolState dataclass."""

    def test_minimal_creation(self):
        """ToolState can be created with just status."""
        state = ToolState(status="running")

        assert state.status == "running"
        assert state.input is None
        assert state.title is None
        assert state.output is None
        assert state.error is None

    def test_full_creation(self):
        """ToolState can be created with all fields."""
        state = ToolState(
            status="completed",
            input={"query": "python"},
            title="Search completed",
            output="Found 10 results",
            error=None,
            metadata={"source": "web"},
            time_start=1000.0,
            time_end=1001.5,
        )

        assert state.status == "completed"
        assert state.input == {"query": "python"}
        assert state.title == "Search completed"
        assert state.output == "Found 10 results"
        assert state.metadata == {"source": "web"}
        assert state.time_start == 1000.0
        assert state.time_end == 1001.5

    def test_error_state(self):
        """ToolState can represent error state."""
        state = ToolState(
            status="error",
            input={"url": "invalid"},
            error="Connection timeout",
        )

        assert state.status == "error"
        assert state.error == "Connection timeout"


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_creation(self):
        """ToolResult can be created with tool_call_id and content."""
        result = ToolResult(tool_call_id="call_123", content="The weather is sunny.")

        assert result.tool_call_id == "call_123"
        assert result.content == "The weather is sunny."
        assert result.is_error is False

    def test_error_result(self):
        """ToolResult can represent an error."""
        result = ToolResult(
            tool_call_id="call_456",
            content="Tool execution failed: timeout",
            is_error=True,
        )

        assert result.is_error is True
        assert "failed" in result.content

    def test_to_message(self):
        """to_message() returns OpenAI message format."""
        result = ToolResult(tool_call_id="call_789", content="Success")

        msg = result.to_message()

        assert msg == {
            "role": "tool",
            "tool_call_id": "call_789",
            "content": "Success",
        }

    def test_to_message_format(self):
        """to_message() returns dict with correct keys."""
        result = ToolResult(tool_call_id="test", content="output")

        msg = result.to_message()

        assert "role" in msg
        assert "tool_call_id" in msg
        assert "content" in msg
        assert msg["role"] == "tool"
