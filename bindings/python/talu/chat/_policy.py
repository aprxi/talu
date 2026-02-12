"""Internal policy wrapper for tool calling."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from . import _bindings as _c


class _PolicyHandle:
    """Internal wrapper for a Core Policy handle. Not public API."""

    def __init__(self, json_bytes: bytes) -> None:
        self._lib = _c.get_chat_lib()
        self._handle = _c.policy_create(self._lib, json_bytes)
        self._closed = False

    def attach(self, chat_ptr: Any) -> None:
        """Attach policy to a chat handle."""
        if self._closed or self._handle is None:
            from ..exceptions import StateError

            raise StateError(
                "Policy handle is closed.",
                code="STATE_INVALID_POLICY",
            )
        _c.chat_set_policy(self._lib, chat_ptr, self._handle)

    def close(self) -> None:
        """Free policy handle (idempotent)."""
        if self._closed:
            return
        _c.policy_free(self._lib, self._handle)
        self._handle = None
        self._closed = True

    def __enter__(self) -> _PolicyHandle:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def build_tool_policy(tool_names: Iterable[str]) -> bytes:
    """
    Build a minimal policy JSON that allows all named tools.

    Args:
        tool_names: Iterable of tool names to allow.

    Returns
    -------
        JSON bytes for _PolicyHandle.
    """
    statements = [{"effect": "allow", "action": f"tool:{name}"} for name in tool_names]
    policy = {"default": "deny", "statements": statements}
    return json.dumps(policy).encode("utf-8")
