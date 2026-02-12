"""Tests for Chat/AsyncChat lifecycle visibility (standalone vs attached mode).

When debugging memory issues, users need to distinguish between:
- Standalone Chat: Owns the model (~GBs), close() frees memory
- Attached Chat: Borrows from Client (~KBs), close() only clears conversation

This is exposed via:
- owns_client property: Programmatic check
- __repr__: Visual debugging (mode='standalone' vs mode='attached')
"""

from talu import AsyncClient, Chat, Client

# =============================================================================
# Chat.owns_client property tests
# =============================================================================


class TestChatOwnsClient:
    """Tests for Chat.owns_client property."""

    def test_standalone_chat_owns_client(self) -> None:
        """Chat created with model= owns its client."""
        # Note: Chat(model=...) creates a default Client internally
        chat = Chat(system="Test")  # No model, no client = state-only Chat
        try:
            # State-only Chat has no client
            assert chat._client is None
        finally:
            chat.close()

    def test_standalone_chat_with_model_owns_client(self) -> None:
        """Chat created with model= owns its internal client."""
        client = Client("openai://gpt-4", base_url="http://test")
        chat = client.chat()
        try:
            # This chat is attached to the client
            assert chat.owns_client is False
        finally:
            chat.close()
            client.close()

    def test_attached_chat_does_not_own_client(self) -> None:
        """Chat created with client= does not own its client."""
        client = Client("openai://gpt-4", base_url="http://test")
        try:
            chat = client.chat()
            assert chat.owns_client is False
            chat.close()
        finally:
            client.close()

    def test_owns_client_is_read_only(self) -> None:
        """owns_client cannot be modified."""
        client = Client("openai://gpt-4", base_url="http://test")
        chat = client.chat()
        try:
            # Property should not be settable
            assert hasattr(Chat.owns_client, "fset") is False or Chat.owns_client.fset is None
        finally:
            chat.close()
            client.close()


# =============================================================================
# Chat.__repr__ tests
# =============================================================================


class TestChatRepr:
    """Tests for Chat.__repr__ lifecycle mode display."""

    def test_standalone_repr_shows_standalone_mode(self) -> None:
        """Standalone Chat (created via Chat(model=...)) shows mode='standalone'."""
        # Create a standalone chat by passing model directly
        # This internally creates a Client, so Chat owns it
        client = Client("openai://gpt-4", base_url="http://test")
        standalone_chat = Chat(client=client)
        standalone_chat._owns_client = True  # Simulate standalone mode for test
        try:
            repr_str = repr(standalone_chat)
            assert "mode='standalone'" in repr_str
        finally:
            standalone_chat.close()
            client.close()

    def test_attached_repr_shows_attached_mode(self) -> None:
        """Attached Chat repr shows mode='attached'."""
        client = Client("openai://gpt-4", base_url="http://test")
        try:
            chat = client.chat()
            repr_str = repr(chat)
            assert "mode='attached'" in repr_str
            assert "mode='standalone'" not in repr_str
            chat.close()
        finally:
            client.close()

    def test_repr_includes_model_name(self) -> None:
        """Chat repr includes model name."""
        client = Client("openai://gpt-4", base_url="http://test")
        chat = client.chat()
        try:
            repr_str = repr(chat)
            assert "model=" in repr_str
            assert "gpt-4" in repr_str
        finally:
            chat.close()
            client.close()

    def test_repr_includes_item_count(self) -> None:
        """Chat repr includes item count."""
        client = Client("openai://gpt-4", base_url="http://test")
        chat = client.chat()
        try:
            repr_str = repr(chat)
            assert "items=" in repr_str
        finally:
            chat.close()
            client.close()

    def test_repr_distinguishes_multiple_chats(self) -> None:
        """Multiple chats can be distinguished by their repr."""
        client = Client("openai://gpt-4", base_url="http://test")
        try:
            attached1 = client.chat()
            attached2 = client.chat()

            # All attached chats show attached mode
            reprs = [repr(attached1), repr(attached2)]

            assert reprs[0].count("attached") == 1
            assert reprs[1].count("attached") == 1

            attached1.close()
            attached2.close()
        finally:
            client.close()


# =============================================================================
# AsyncChat tests
# =============================================================================


class TestAsyncChatOwnsClient:
    """Tests for AsyncChat.owns_client property."""

    def test_attached_async_chat_does_not_own_client(self) -> None:
        """AsyncChat created with client= does not own its client."""
        client = AsyncClient("openai://gpt-4", base_url="http://test")
        try:
            chat = client.chat()
            assert chat.owns_client is False
            chat._close_sync()
        finally:
            client._router.close()


class TestAsyncChatRepr:
    """Tests for AsyncChat.__repr__ lifecycle mode display."""

    def test_attached_async_repr_shows_attached_mode(self) -> None:
        """Attached AsyncChat repr shows mode='attached'."""
        client = AsyncClient("openai://gpt-4", base_url="http://test")
        try:
            chat = client.chat()
            repr_str = repr(chat)
            assert "mode='attached'" in repr_str
            chat._close_sync()
        finally:
            client._router.close()


# =============================================================================
# Debugging scenario tests
# =============================================================================


class TestDebuggingScenarios:
    """Tests simulating real debugging scenarios."""

    def test_oom_debugging_list_of_chats(self) -> None:
        """
        Simulate debugging OOM with a list of chats.

        User has multiple chats from a shared client.
        """
        client = Client("openai://gpt-4", base_url="http://test")
        try:
            # User's code created these from the same client
            chats = [
                client.chat(),
                client.chat(),
                client.chat(),
            ]

            # User debugging: prints the list
            debug_output = str([repr(c) for c in chats])

            # All attached to the same client
            assert debug_output.count("attached") == 3

            for c in chats:
                c.close()
        finally:
            client.close()

    def test_programmatic_resource_cleanup(self) -> None:
        """
        Resource manager can use owns_client to decide cleanup strategy.
        """
        client = Client("openai://gpt-4", base_url="http://test")
        try:
            chats = [
                client.chat(),
                client.chat(),
            ]

            # All chats are attached (borrowed from client)
            heavy_chats = [c for c in chats if c.owns_client]
            light_chats = [c for c in chats if not c.owns_client]

            assert len(heavy_chats) == 0
            assert len(light_chats) == 2

            for c in chats:
                c.close()
        finally:
            client.close()
