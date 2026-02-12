"""
Regression tests for hidden message append.

These tests verify that Chat.append() with hidden=True works correctly.
This is a regression test for a segfault that occurred when
talu_responses_append_message_hidden was missing ctypes argtypes,
causing pointer sign extension on 64-bit systems.
"""

from talu import Chat


class TestAppendHiddenRegression:
    """Regression tests for the hidden append bug."""

    def test_append_hidden_does_not_corrupt_pointer(self):
        """Appending with hidden=True should work correctly.

        This is a regression test for the segfault that occurred when
        talu_responses_append_message_hidden was missing argtypes.
        """
        chat = Chat(session_id="test")
        chat.append("user", "visible message")
        chat.append("user", "hidden message", hidden=True)
        assert len(chat.items) == 2

    def test_multiple_hidden_appends(self):
        """Multiple hidden appends should work correctly."""
        chat = Chat(session_id="test")
        for i in range(10):
            chat.append("user", f"visible {i}")
            chat.append("assistant", f"response {i}", hidden=True)
        assert len(chat.items) == 20

    def test_insert_hidden_works(self):
        """Inserting with hidden=True should work correctly."""
        chat = Chat(session_id="test")
        chat.append("user", "first message")
        chat.append("user", "third message")
        chat.insert(1, "assistant", "second message (hidden)", hidden=True)
        assert len(chat.items) == 3
