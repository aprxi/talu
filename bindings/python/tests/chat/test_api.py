"""Smoke tests for chat convenience API module.

Maps to: talu/chat/api.py
"""

from talu.chat.api import ask, raw_complete, stream


def test_chat_api_exports() -> None:
    assert callable(ask)
    assert callable(raw_complete)
    assert callable(stream)

