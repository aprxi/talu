"""
Tests for ctypes argtypes/restype configuration.

These tests ensure that all C API functions have proper type signatures configured.
Missing argtypes can cause subtle pointer corruption on 64-bit systems, leading to
segfaults or invalid handle errors.

Issue: talu_responses_append_message_hidden was missing argtypes, causing sign
extension of 32-bit pointer values on 64-bit systems, resulting in invalid pointers.
"""

import ctypes

from talu.chat._bindings import get_chat_lib


class TestResponsesBindings:
    """Test that responses C API functions have proper type signatures."""

    def test_append_message_has_argtypes(self):
        """talu_responses_append_message must have argtypes configured."""
        lib = get_chat_lib()
        assert hasattr(lib.talu_responses_append_message, "argtypes")
        assert lib.talu_responses_append_message.argtypes is not None
        assert len(lib.talu_responses_append_message.argtypes) == 4

    def test_append_message_hidden_has_argtypes(self):
        """talu_responses_append_message_hidden must have argtypes configured.

        This was the root cause of the segfault issue - missing argtypes caused
        pointer sign extension on 64-bit systems.
        """
        lib = get_chat_lib()
        assert hasattr(lib.talu_responses_append_message_hidden, "argtypes")
        assert lib.talu_responses_append_message_hidden.argtypes is not None
        assert len(lib.talu_responses_append_message_hidden.argtypes) == 5

    def test_append_function_call_has_argtypes(self):
        """talu_responses_append_function_call must have argtypes configured."""
        lib = get_chat_lib()
        assert hasattr(lib.talu_responses_append_function_call, "argtypes")
        assert lib.talu_responses_append_function_call.argtypes is not None
        assert len(lib.talu_responses_append_function_call.argtypes) == 5

    def test_append_function_call_output_has_argtypes(self):
        """talu_responses_append_function_call_output must have argtypes configured."""
        lib = get_chat_lib()
        assert hasattr(lib.talu_responses_append_function_call_output, "argtypes")
        assert lib.talu_responses_append_function_call_output.argtypes is not None
        assert len(lib.talu_responses_append_function_call_output.argtypes) == 4

    def test_insert_message_has_argtypes(self):
        """talu_responses_insert_message must have argtypes configured."""
        lib = get_chat_lib()
        assert hasattr(lib.talu_responses_insert_message, "argtypes")
        assert lib.talu_responses_insert_message.argtypes is not None
        assert len(lib.talu_responses_insert_message.argtypes) == 5

    def test_insert_message_hidden_has_argtypes(self):
        """talu_responses_insert_message_hidden must have argtypes configured."""
        lib = get_chat_lib()
        assert hasattr(lib.talu_responses_insert_message_hidden, "argtypes")
        assert lib.talu_responses_insert_message_hidden.argtypes is not None
        assert len(lib.talu_responses_insert_message_hidden.argtypes) == 6


class TestChatBindings:
    """Test that chat C API functions have proper type signatures."""

    def test_chat_get_conversation_has_argtypes(self):
        """talu_chat_get_conversation must have argtypes configured."""
        lib = get_chat_lib()
        assert hasattr(lib.talu_chat_get_conversation, "argtypes")
        assert lib.talu_chat_get_conversation.argtypes is not None
        assert lib.talu_chat_get_conversation.restype == ctypes.c_void_p

    def test_chat_create_has_argtypes(self):
        """talu_chat_create must have argtypes configured."""
        lib = get_chat_lib()
        assert hasattr(lib.talu_chat_create, "argtypes")
        assert lib.talu_chat_create.argtypes is not None
        assert lib.talu_chat_create.restype == ctypes.c_void_p
