"""Maps to: talu/chat/_bindings.py."""

from __future__ import annotations

import ctypes

from talu._native import CGenerateResult, CToolCallRef
from talu.router import _bindings as bindings


def test_router_result_extract_tool_calls() -> None:
    calls = (CToolCallRef * 2)()

    calls[0].item_index = 1
    calls[0].call_id = b"call_1"
    calls[0].name = b"search"
    calls[0].arguments = b'{"query":"zig"}'

    calls[1].item_index = 2
    calls[1].call_id = b"call_2"
    calls[1].name = b"get_weather"
    calls[1].arguments = b'{"location":"Paris"}'

    result = CGenerateResult()
    result.tool_calls = ctypes.cast(calls, ctypes.POINTER(CToolCallRef))
    result.tool_call_count = 2

    extracted = bindings.router_result_extract_tool_calls(result)

    assert extracted == [
        {"id": "call_1", "name": "search", "arguments": '{"query":"zig"}'},
        {"id": "call_2", "name": "get_weather", "arguments": '{"location":"Paris"}'},
    ]


def test_router_result_extract_tool_calls_empty() -> None:
    result = CGenerateResult()
    result.tool_calls = None
    result.tool_call_count = 0

    assert bindings.router_result_extract_tool_calls(result) is None
