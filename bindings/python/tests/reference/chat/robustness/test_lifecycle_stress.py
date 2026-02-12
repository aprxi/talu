"""
Lifecycle tests for Chat and Conversation memory safety.

These tests verify that Chat objects maintain valid memory throughout their lifecycle,
especially after operations that may reallocate or replace internal structures.

Key scenarios tested:
1. Pointer validity after creation
2. Pointer validity after append operations
3. Pointer validity after setSystem (which may replace Conversation)
4. Magic number validation for detecting use-after-free
5. Stress tests with many allocations and deallocations
"""

import gc

import pytest

from talu import Chat
from talu.chat._bindings import get_chat_lib


class TestConversationPointerValidity:
    """Test that _conversation_ptr remains valid throughout Chat lifecycle."""

    def test_fresh_chat_has_valid_pointer(self):
        """A newly created Chat should have a valid conversation pointer."""
        chat = Chat()
        lib = get_chat_lib()

        # Pointer should not be None
        assert chat._conversation_ptr is not None

        # Chat handle should be valid (len should work)
        chat_len = lib.talu_chat_len(chat._chat_ptr)
        assert chat_len == 0

        # Conversation pointer should match what Chat reports
        fresh_ptr = lib.talu_chat_get_conversation(chat._chat_ptr)
        assert fresh_ptr == chat._conversation_ptr

        chat.close()

    def test_pointer_valid_after_append(self):
        """Conversation pointer should remain valid after append."""
        chat = Chat()
        lib = get_chat_lib()

        initial_ptr = chat._conversation_ptr

        chat.append("user", "Hello")
        assert len(chat.items) == 1

        # Pointer should still be the same (no reallocation on append)
        assert chat._conversation_ptr == initial_ptr

        # Fresh lookup should match
        fresh_ptr = lib.talu_chat_get_conversation(chat._chat_ptr)
        assert fresh_ptr == chat._conversation_ptr

        chat.close()

    def test_pointer_valid_after_multiple_appends(self):
        """Conversation pointer should remain valid after many appends."""
        chat = Chat()
        lib = get_chat_lib()

        for i in range(100):
            chat.append("user", f"Message {i}")
            chat.append("assistant", f"Response {i}")

        assert len(chat.items) == 200

        # Fresh lookup should match cached pointer
        fresh_ptr = lib.talu_chat_get_conversation(chat._chat_ptr)
        assert fresh_ptr == chat._conversation_ptr

        chat.close()

    def test_pointer_refreshed_after_set_system(self):
        """setSystem may replace Conversation; pointer should be refreshed."""
        chat = Chat()
        lib = get_chat_lib()

        # Add some messages first
        chat.append("user", "Hello")
        chat.append("assistant", "Hi")

        # Set system prompt (may replace Conversation internally)
        chat.system = "You are a helpful assistant."

        # Cached pointer should be refreshed
        fresh_ptr = lib.talu_chat_get_conversation(chat._chat_ptr)
        assert fresh_ptr == chat._conversation_ptr

        # Chat should still work
        assert len(chat.items) == 3  # system + user + assistant

        chat.close()

    def test_pointer_valid_with_hidden_messages(self):
        """Hidden messages should not corrupt the conversation pointer."""
        chat = Chat()
        lib = get_chat_lib()

        # Mix of hidden and visible messages
        chat.append("user", "visible 1")
        chat.append("user", "hidden 1", hidden=True)
        chat.append("user", "visible 2")
        chat.append("user", "hidden 2", hidden=True)

        assert len(chat.items) == 4

        # Pointer should still be valid
        fresh_ptr = lib.talu_chat_get_conversation(chat._chat_ptr)
        assert fresh_ptr == chat._conversation_ptr

        chat.close()


class TestMagicNumberValidation:
    """Test magic number validation for detecting memory corruption."""

    def test_valid_conversation_passes_validation(self):
        """A valid conversation should pass magic number validation."""
        chat = Chat()
        lib = get_chat_lib()

        # Setup validation function
        import ctypes

        lib.talu_responses_validate.argtypes = [ctypes.c_void_p]
        lib.talu_responses_validate.restype = ctypes.c_int32

        is_valid = lib.talu_responses_validate(chat._conversation_ptr)
        assert is_valid == 1, "Fresh conversation should be valid"

        chat.close()

    def test_validation_after_operations(self):
        """Validation should pass after various operations."""
        chat = Chat(system="Test system")
        lib = get_chat_lib()

        import ctypes

        lib.talu_responses_validate.argtypes = [ctypes.c_void_p]
        lib.talu_responses_validate.restype = ctypes.c_int32

        # After creation with system
        assert lib.talu_responses_validate(chat._conversation_ptr) == 1

        # After append
        chat.append("user", "Hello")
        assert lib.talu_responses_validate(chat._conversation_ptr) == 1

        # After hidden append
        chat.append("user", "Hidden", hidden=True)
        assert lib.talu_responses_validate(chat._conversation_ptr) == 1

        # After clear
        chat.clear()
        assert lib.talu_responses_validate(chat._conversation_ptr) == 1

        chat.close()


class TestStressLifecycle:
    """Stress tests for Chat lifecycle."""

    def test_many_chat_creations(self):
        """Creating and closing many Chat objects should not cause issues."""
        for i in range(100):
            chat = Chat()
            chat.append("user", f"Message {i}")
            chat.close()
            del chat

        # Force garbage collection
        gc.collect()

        # Create one more to verify no corruption
        chat = Chat()
        chat.append("user", "Final message")
        assert len(chat.items) == 1
        chat.close()

    def test_many_chat_creations_with_gc(self):
        """Creating many Chats with explicit GC should not cause issues."""
        for batch in range(10):
            chats = []
            for i in range(10):
                chat = Chat()
                chat.append("user", f"Batch {batch} Message {i}")
                chats.append(chat)

            # Close all chats
            for chat in chats:
                chat.close()

            # Clear references and GC
            chats.clear()
            gc.collect()

        # Verify no corruption
        final_chat = Chat()
        final_chat.append("user", "Final")
        final_chat.append("user", "Hidden", hidden=True)
        assert len(final_chat.items) == 2
        final_chat.close()


class TestClosedChatErrors:
    """Test that accessing properties on a closed Chat raises StateError."""

    def test_items_property_after_close(self):
        """Accessing items property after close raises StateError."""
        from talu.exceptions import StateError

        chat = Chat()
        chat.close()

        with pytest.raises(StateError, match="closed"):
            _ = chat.items

    def test_system_property_after_close(self):
        """Accessing system property after close raises StateError."""
        from talu.exceptions import StateError

        chat = Chat(system="Test")
        chat.close()

        with pytest.raises(StateError, match="closed"):
            _ = chat.system

    def test_system_setter_after_close(self):
        """Setting system property after close raises StateError."""
        from talu.exceptions import StateError

        chat = Chat()
        chat.close()

        with pytest.raises(StateError, match="closed"):
            chat.system = "New system"

    def test_append_after_close(self):
        """Appending after close raises StateError."""
        from talu.exceptions import StateError

        chat = Chat()
        chat.close()

        with pytest.raises(StateError, match="closed"):
            chat.append("user", "Hello")

    def test_close_is_idempotent(self):
        """Calling close multiple times is safe."""
        chat = Chat()
        chat.close()
        chat.close()  # Should not raise
        chat.close()  # Should not raise

    def test_error_code_is_chat_closed(self):
        """StateError has CHAT_CLOSED code."""
        from talu.exceptions import StateError

        chat = Chat()
        chat.close()

        with pytest.raises(StateError) as exc_info:
            _ = chat.items

        # Verify the error code
        assert exc_info.value.code in ("CHAT_CLOSED", "STATE_CLOSED")
