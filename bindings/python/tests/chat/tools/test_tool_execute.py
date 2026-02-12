"""Tests for ToolCall execution helpers."""

from __future__ import annotations

import pytest

from talu.chat.tools import ToolCall, ToolExecutionError


def test_tool_call_execute_sync() -> None:
    """ToolCall.execute invokes bound function with parsed args."""

    def multiply(x: int, y: int) -> int:
        return x * y

    call = ToolCall.create("call_1", "multiply", '{"x": 6, "y": 7}')
    call._func = multiply
    assert call.execute() == 42


def test_tool_call_execute_missing_func() -> None:
    """ToolCall.execute raises when no function is mapped."""
    call = ToolCall.create("call_1", "missing", "{}")
    with pytest.raises(ToolExecutionError):
        call.execute()


@pytest.mark.asyncio
async def test_tool_call_execute_async() -> None:
    """execute_async awaits coroutine functions."""

    async def add(x: int, y: int) -> int:
        return x + y

    call = ToolCall.create("call_1", "add", '{"x": 2, "y": 3}')
    call._func = add
    assert await call.execute_async() == 5
