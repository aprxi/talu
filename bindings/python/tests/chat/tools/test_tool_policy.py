"""Tests for tool policy helper."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from talu.chat._policy import _PolicyHandle, build_tool_policy


def test_build_tool_policy() -> None:
    """build_tool_policy produces default-deny with allow statements."""
    policy_bytes = build_tool_policy(["get_weather", "search"])
    policy = json.loads(policy_bytes)

    assert policy["default"] == "deny"
    assert policy["statements"] == [
        {"effect": "allow", "action": "tool:get_weather"},
        {"effect": "allow", "action": "tool:search"},
    ]


def test_policy_handle_close_idempotent(monkeypatch) -> None:
    """_PolicyHandle.close is idempotent and frees once."""
    mock_lib = MagicMock()
    handle = object()
    policy_create = MagicMock(return_value=handle)
    policy_free = MagicMock()
    chat_set_policy = MagicMock()
    get_chat_lib = MagicMock(return_value=mock_lib)

    monkeypatch.setattr("talu.chat._policy._c.policy_create", policy_create)
    monkeypatch.setattr("talu.chat._policy._c.policy_free", policy_free)
    monkeypatch.setattr("talu.chat._policy._c.chat_set_policy", chat_set_policy)
    monkeypatch.setattr("talu.chat._policy._c.get_chat_lib", get_chat_lib)

    policy = _PolicyHandle(b"{}")
    policy.attach(object())
    policy.close()
    policy.close()

    assert policy_free.call_count == 1
    assert chat_set_policy.call_count == 1
