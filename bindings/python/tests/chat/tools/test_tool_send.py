"""Tests for Chat.send tool integration."""

from __future__ import annotations

import json

import pytest

from talu import AsyncChat, Chat
from talu.chat.response import AsyncResponse, Response
from talu.chat.tools import ToolCall, tool


def test_send_with_tools_injects_registry(monkeypatch) -> None:
    """send(tools=...) sets tools_json and binds tool call functions."""

    class DummyPolicy:
        def __init__(self, _json_bytes: bytes) -> None:
            self.attached = False

        def attach(self, _chat_ptr) -> None:
            self.attached = True

        def close(self) -> None:
            pass

    monkeypatch.setattr("talu.chat._policy._PolicyHandle", DummyPolicy)

    chat = Chat(system="You are helpful.")

    @tool
    def add(x: int, y: int) -> int:
        """Add numbers.

        Args:
            x: First number
            y: Second number
        """

        return x + y

    def fake_generate(
        self, message, config=None, stream=False, on_token=None, response_format=None
    ):
        assert message == "hello"
        assert config is not None
        assert config.tools_json is not None
        tools_payload = json.loads(config.tools_json)
        assert tools_payload[0]["function"]["name"] == "add"

        call = ToolCall.create("call_1", "add", '{"x": 1, "y": 2}')
        return Response(
            text="",
            finish_reason="tool_calls",
            tool_calls=[call],
            chat=chat,
        )

    monkeypatch.setattr(chat, "_generate_sync", fake_generate.__get__(chat, Chat))

    try:
        response = chat.send("hello", tools=[add], stream=False)

        assert response.tool_calls is not None
        assert response.tool_calls[0]._func is add._tool_func  # type: ignore[attr-defined]
        assert response._tool_registry is not None
        assert response._tool_registry["add"] is add._tool_func  # type: ignore[attr-defined]
    finally:
        chat.close()


@pytest.mark.asyncio
async def test_async_send_with_tools_injects_registry(monkeypatch) -> None:
    """AsyncChat.send(tools=...) sets tools_json and binds tool call functions."""

    class DummyPolicy:
        def __init__(self, _json_bytes: bytes) -> None:
            self.attached = False

        def attach(self, _chat_ptr) -> None:
            self.attached = True

        def close(self) -> None:
            pass

    monkeypatch.setattr("talu.chat._policy._PolicyHandle", DummyPolicy)

    chat = AsyncChat(system="You are helpful.")

    @tool
    async def add(x: int, y: int) -> int:
        """Add numbers.

        Args:
            x: First number
            y: Second number
        """

        return x + y

    async def fake_generate(
        self, message, config=None, stream=False, on_token=None, response_format=None
    ):
        assert message == "hello"
        assert config is not None
        assert config.tools_json is not None
        tools_payload = json.loads(config.tools_json)
        assert tools_payload[0]["function"]["name"] == "add"

        call = ToolCall.create("call_1", "add", '{"x": 1, "y": 2}')
        return AsyncResponse(
            text="",
            finish_reason="tool_calls",
            tool_calls=[call],
            chat=chat,
        )

    monkeypatch.setattr(chat, "_generate_async", fake_generate.__get__(chat, AsyncChat))

    try:
        response = await chat.send("hello", tools=[add], stream=False)

        assert response.tool_calls is not None
        assert response.tool_calls[0]._func is add._tool_func  # type: ignore[attr-defined]
        assert response._tool_registry is not None
        assert response._tool_registry["add"] is add._tool_func  # type: ignore[attr-defined]
    finally:
        await chat.close()
