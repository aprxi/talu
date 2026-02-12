"""Tests for talu.chat._policy module.

Tests for the internal policy wrapper used for tool calling.
Uses mocking to test without requiring model inference.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from talu.chat._policy import _PolicyHandle, build_tool_policy
from talu.exceptions import StateError


class TestBuildToolPolicy:
    """Tests for build_tool_policy function."""

    def test_empty_tool_names(self):
        """Empty tool names produces deny-all policy."""
        result = build_tool_policy([])

        policy = json.loads(result)
        assert policy["default"] == "deny"
        assert policy["statements"] == []

    def test_single_tool(self):
        """Single tool produces correct policy."""
        result = build_tool_policy(["calculator"])

        policy = json.loads(result)
        assert policy["default"] == "deny"
        assert len(policy["statements"]) == 1
        assert policy["statements"][0]["effect"] == "allow"
        assert policy["statements"][0]["action"] == "tool:calculator"

    def test_multiple_tools(self):
        """Multiple tools produce correct policy."""
        result = build_tool_policy(["web_search", "code_exec", "file_read"])

        policy = json.loads(result)
        assert policy["default"] == "deny"
        assert len(policy["statements"]) == 3

        actions = [s["action"] for s in policy["statements"]]
        assert "tool:web_search" in actions
        assert "tool:code_exec" in actions
        assert "tool:file_read" in actions

    def test_returns_bytes(self):
        """Returns bytes (JSON encoded as UTF-8)."""
        result = build_tool_policy(["test"])

        assert isinstance(result, bytes)
        # Should be valid JSON when decoded
        json.loads(result.decode("utf-8"))

    def test_accepts_iterator(self):
        """Accepts any iterable of tool names."""
        # Test with generator
        result = build_tool_policy(name for name in ["a", "b", "c"])

        policy = json.loads(result)
        assert len(policy["statements"]) == 3

    def test_tool_names_with_special_chars(self):
        """Tool names with special characters are preserved."""
        result = build_tool_policy(["my_tool", "my-tool", "my.tool"])

        policy = json.loads(result)
        actions = [s["action"] for s in policy["statements"]]
        assert "tool:my_tool" in actions
        assert "tool:my-tool" in actions
        assert "tool:my.tool" in actions


class TestPolicyHandle:
    """Tests for _PolicyHandle class."""

    def test_construction(self):
        """PolicyHandle can be constructed."""
        mock_lib = MagicMock()
        mock_handle = MagicMock()

        with patch("talu.chat._policy._c.get_chat_lib", return_value=mock_lib):
            with patch("talu.chat._policy._c.policy_create", return_value=mock_handle):
                policy = _PolicyHandle(b'{"default": "deny", "statements": []}')

                assert policy._handle is mock_handle
                assert policy._closed is False

    def test_close_is_idempotent(self):
        """Calling close() multiple times is safe."""
        mock_lib = MagicMock()
        mock_handle = MagicMock()

        with patch("talu.chat._policy._c.get_chat_lib", return_value=mock_lib):
            with patch("talu.chat._policy._c.policy_create", return_value=mock_handle):
                with patch("talu.chat._policy._c.policy_free") as mock_free:
                    policy = _PolicyHandle(b"{}")

                    policy.close()
                    policy.close()  # Second call should be safe
                    policy.close()  # Third call should be safe

                    # free should only be called once
                    assert mock_free.call_count == 1

    def test_context_manager_calls_close(self):
        """Context manager calls close on exit."""
        mock_lib = MagicMock()
        mock_handle = MagicMock()

        with patch("talu.chat._policy._c.get_chat_lib", return_value=mock_lib):
            with patch("talu.chat._policy._c.policy_create", return_value=mock_handle):
                with patch("talu.chat._policy._c.policy_free") as mock_free:
                    with _PolicyHandle(b"{}") as policy:
                        assert policy._closed is False

                    assert policy._closed is True
                    mock_free.assert_called_once()

    def test_context_manager_returns_self(self):
        """Context manager __enter__ returns self."""
        mock_lib = MagicMock()
        mock_handle = MagicMock()

        with patch("talu.chat._policy._c.get_chat_lib", return_value=mock_lib):
            with patch("talu.chat._policy._c.policy_create", return_value=mock_handle):
                with patch("talu.chat._policy._c.policy_free"):
                    policy = _PolicyHandle(b"{}")

                    with policy as p:
                        assert p is policy

    def test_attach_raises_state_error_when_closed(self):
        """attach() raises StateError when handle is closed."""
        mock_lib = MagicMock()
        mock_handle = MagicMock()

        with patch("talu.chat._policy._c.get_chat_lib", return_value=mock_lib):
            with patch("talu.chat._policy._c.policy_create", return_value=mock_handle):
                with patch("talu.chat._policy._c.policy_free"):
                    policy = _PolicyHandle(b"{}")
                    policy.close()

                    with pytest.raises(StateError) as exc_info:
                        policy.attach(MagicMock())

                    assert "closed" in str(exc_info.value).lower()
                    assert exc_info.value.code == "STATE_INVALID_POLICY"

    def test_attach_raises_state_error_when_handle_is_none(self):
        """attach() raises StateError when handle is None."""
        mock_lib = MagicMock()

        with patch("talu.chat._policy._c.get_chat_lib", return_value=mock_lib):
            with patch("talu.chat._policy._c.policy_create", return_value=None):
                with patch("talu.chat._policy._c.policy_free"):
                    policy = _PolicyHandle(b"{}")
                    # Handle is None from construction
                    policy._closed = True

                    with pytest.raises(StateError):
                        policy.attach(MagicMock())

    def test_attach_calls_chat_set_policy(self):
        """attach() calls chat_set_policy with correct arguments."""
        mock_lib = MagicMock()
        mock_handle = MagicMock()
        mock_chat_ptr = MagicMock()

        with patch("talu.chat._policy._c.get_chat_lib", return_value=mock_lib):
            with patch("talu.chat._policy._c.policy_create", return_value=mock_handle):
                with patch("talu.chat._policy._c.chat_set_policy") as mock_set:
                    policy = _PolicyHandle(b"{}")
                    policy.attach(mock_chat_ptr)

                    mock_set.assert_called_once_with(mock_lib, mock_chat_ptr, mock_handle)

    def test_del_calls_close(self):
        """__del__ calls close() safely."""
        mock_lib = MagicMock()
        mock_handle = MagicMock()

        with patch("talu.chat._policy._c.get_chat_lib", return_value=mock_lib):
            with patch("talu.chat._policy._c.policy_create", return_value=mock_handle):
                with patch("talu.chat._policy._c.policy_free") as mock_free:
                    policy = _PolicyHandle(b"{}")
                    # Simulate garbage collection
                    policy.__del__()

                    mock_free.assert_called_once()

    def test_del_suppresses_exceptions(self):
        """__del__ suppresses exceptions during cleanup."""
        mock_lib = MagicMock()
        mock_handle = MagicMock()

        with patch("talu.chat._policy._c.get_chat_lib", return_value=mock_lib):
            with patch("talu.chat._policy._c.policy_create", return_value=mock_handle):
                with patch(
                    "talu.chat._policy._c.policy_free", side_effect=RuntimeError("cleanup error")
                ):
                    policy = _PolicyHandle(b"{}")
                    # Should not raise
                    policy.__del__()

    def test_close_sets_handle_to_none(self):
        """close() sets _handle to None."""
        mock_lib = MagicMock()
        mock_handle = MagicMock()

        with patch("talu.chat._policy._c.get_chat_lib", return_value=mock_lib):
            with patch("talu.chat._policy._c.policy_create", return_value=mock_handle):
                with patch("talu.chat._policy._c.policy_free"):
                    policy = _PolicyHandle(b"{}")

                    assert policy._handle is not None
                    policy.close()
                    assert policy._handle is None
