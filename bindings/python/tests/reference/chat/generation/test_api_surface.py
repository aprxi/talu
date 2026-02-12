"""
Tests for the user-facing Chat API.

These tests verify the API surface as documented in examples/chat/
They test the expected behavior, not implementation.

The API has two user paths:
1. Casual: Chat("model") - loads model, single user
2. Power:  Client("model") + client.chat() - shared model, multi user
"""

import pytest

from talu import Chat, Client, CompletionOptions, GenerationConfig
from talu.chat import Response
from talu.types import MessageItem

# =============================================================================
# Casual User Path: Chat("model")
# =============================================================================


class TestChatCasualPath:
    """Tests for Chat("model") - the casual user path."""

    def test_chat_with_model_creates_session(self, test_model_path):
        """Chat("model") creates a working chat session."""
        chat = Chat(test_model_path)
        assert chat is not None
        # Should have a client internally
        assert chat._client is not None

    def test_chat_callable_generates_response(self, test_model_path):
        """chat.send("message") returns a Response."""
        chat = Chat(test_model_path)
        response = chat.send("Hello", max_tokens=3)

        assert isinstance(response, Response)
        assert isinstance(str(response), str)
        # With thinking models, 3 tokens may produce only reasoning content
        # (stripped from .text). Check tokens were generated, not text length.
        assert response.usage.completion_tokens > 0

    def test_chat_callable_updates_history(self, test_model_path):
        """chat.send("message") adds user and assistant messages to history."""
        chat = Chat(test_model_path)
        chat.send("Hello", max_tokens=3)

        # Should have user + generation output (may include reasoning items)
        assert len(chat.items) >= 2
        messages = [item for item in chat.items if isinstance(item, MessageItem)]
        roles = [m.role.name.lower() for m in messages]
        assert "user" in roles

    def test_chat_with_system_prompt(self, test_model_path):
        """Chat("model", system="...") sets system prompt."""
        chat = Chat(test_model_path, system="You are a pirate.")

        assert len(chat.items) == 1
        assert chat.items[0].role.name.lower() == "system"
        assert chat.items[0].text == "You are a pirate."

    def test_response_append_continues_conversation(self, test_model_path):
        """response.append("message") continues the conversation."""
        chat = Chat(test_model_path)
        response1 = chat.send("Hello", max_tokens=3)
        response2 = response1.append("How are you?", max_tokens=3)

        assert isinstance(response2, Response)
        # Two rounds of interaction: at least 2 user messages + generation output
        assert len(chat.items) >= 4
        user_msgs = [i for i in chat.items if isinstance(i, MessageItem) and i.role.name == "USER"]
        assert len(user_msgs) == 2

    def test_response_append_chain(self, test_model_path):
        """response.append().append() chains work."""
        chat = Chat(test_model_path)
        response = chat.send("Hi", max_tokens=3)
        response = response.append("One", max_tokens=3)
        response = response.append("Two", max_tokens=3)

        # Three rounds: at least 3 user messages + generation output per round
        assert len(chat.items) >= 6
        user_msgs = [i for i in chat.items if isinstance(i, MessageItem) and i.role.name == "USER"]
        assert len(user_msgs) == 3

    def test_response_has_chat_reference(self, test_model_path):
        """response.chat returns the Chat instance."""
        chat = Chat(test_model_path)
        response = chat.send("Hello", max_tokens=3)

        assert response.chat is chat


# =============================================================================
# Power User Path: Client("model")
# =============================================================================


class TestClientPowerPath:
    """Tests for Client("model") - the power user path."""

    def test_client_creation(self, test_model_path):
        """Client("model") creates a client."""
        client = Client(test_model_path)
        assert client is not None
        assert test_model_path in client.models or any(test_model_path in m for m in client.models)
        client.close()

    def test_client_context_manager(self, test_model_path):
        """Client works as context manager."""
        with Client(test_model_path) as client:
            assert client is not None
        # Should be closed after context

    def test_client_chat_creates_chat(self, test_model_path):
        """client.chat() creates a Chat instance."""
        with Client(test_model_path) as client:
            chat = client.chat()
            assert isinstance(chat, Chat)

    def test_client_chat_with_system(self, test_model_path):
        """client.chat(system="...") sets system prompt."""
        with Client(test_model_path) as client:
            chat = client.chat(system="You are helpful.")
            assert chat.items[0].role.name.lower() == "system"
            assert chat.items[0].text == "You are helpful."

    def test_multiple_chats_share_client(self, test_model_path):
        """Multiple chats from same client share the model."""
        with Client(test_model_path) as client:
            alice = client.chat(system="You are Alice.")
            bob = client.chat(system="You are Bob.")

            # Both should reference same client
            assert alice._client is client
            assert bob._client is client

            # Both can generate independently
            r1 = alice.send("Hi", max_tokens=3)
            r2 = bob.send("Hi", max_tokens=3)

            assert isinstance(r1, Response)
            assert isinstance(r2, Response)

    def test_client_ask(self, test_model_path):
        """client.ask() does stateless completion."""
        with Client(test_model_path) as client:
            response = client.ask("Hello", max_tokens=3)
            assert isinstance(response, Response)

    def test_client_raw_complete(self, test_model_path):
        """client.raw_complete() does raw completion without chat template."""
        with Client(test_model_path) as client:
            response = client.raw_complete("The sky is", max_tokens=5)
            assert isinstance(response, Response)
            assert response.usage.completion_tokens > 0

    def test_client_raw_complete_with_system(self, test_model_path):
        """client.raw_complete() accepts system prompt (same signature as complete())."""
        with Client(test_model_path) as client:
            response = client.raw_complete("Hello", system="You are helpful.", max_tokens=5)
            assert isinstance(response, Response)
            assert response.usage.completion_tokens > 0

    def test_client_raw_complete_with_completion_opts(self, test_model_path):
        """client.raw_complete() accepts CompletionOptions for raw-only features."""
        # Note: echo_prompt and other CompletionOptions features are not yet
        # implemented in the backend. This test verifies the API accepts the
        # parameter without error when using default values.
        with Client(test_model_path) as client:
            opts = CompletionOptions()  # All defaults (no backend params set)
            response = client.raw_complete("Continue: ", completion_opts=opts, max_tokens=5)
            assert isinstance(response, Response)
            assert response.usage.completion_tokens > 0


# =============================================================================
# Streaming
# =============================================================================


class TestStreaming:
    """Tests for streaming generation."""

    def test_stream_true_returns_iterable(self, test_model_path):
        """chat("msg") returns StreamingResponse (streaming is default)."""
        chat = Chat(test_model_path)
        response = chat("Hello", max_tokens=5)  # stream=True is default

        chunks = []
        for chunk in response:
            assert isinstance(chunk, str)
            chunks.append(chunk)

        assert len(chunks) >= 1

    def test_stream_response_has_text_after_iteration(self, test_model_path):
        """After streaming, response.text contains full text."""
        chat = Chat(test_model_path)
        response = chat("Hello", max_tokens=5)

        collected = []
        for chunk in response:
            collected.append(chunk)

        # Full text should match collected chunks
        assert str(response) == "".join(collected)

    def test_stream_updates_history(self, test_model_path):
        """Streaming updates chat history after completion."""
        chat = Chat(test_model_path)
        response = chat("Hello", max_tokens=5)

        # Consume the stream
        for _ in response:
            pass

        # History should be updated with at least user + generation output
        assert len(chat.items) >= 2

    def test_streaming_response_append_inherits_stream(self, test_model_path):
        """StreamingResponse.append() returns StreamingResponse (inherits mode)."""
        chat = Chat(test_model_path)
        # Streaming response
        response1 = chat("Hello", max_tokens=3)
        for _ in response1:
            pass  # consume stream

        # append() inherits streaming mode
        response2 = response1.append("More", max_tokens=3)

        chunks = list(response2)
        assert len(chunks) >= 1


# =============================================================================
# Async
# =============================================================================


class TestAsync:
    """Tests for async generation using AsyncChat."""

    @pytest.mark.asyncio
    async def test_async_generation(self, test_model_path):
        """AsyncChat.send("msg") returns awaitable AsyncResponse."""
        from talu import AsyncChat
        from talu.chat import AsyncResponse

        chat = AsyncChat(test_model_path)
        response = await chat.send("Hello", max_tokens=3)

        assert isinstance(response, AsyncResponse)
        assert response.usage.completion_tokens > 0

    @pytest.mark.asyncio
    async def test_async_append(self, test_model_path):
        """response.append("msg") on async response returns awaitable."""
        from talu import AsyncChat
        from talu.chat import AsyncResponse

        chat = AsyncChat(test_model_path)
        response1 = await chat.send("Hello", max_tokens=3)
        # append() on AsyncResponse is async, so must await
        response2 = await response1.append("More", max_tokens=3)

        assert isinstance(response2, AsyncResponse)

    @pytest.mark.asyncio
    async def test_async_streaming(self, test_model_path):
        """AsyncChat("msg", stream=True) returns AsyncStreamingResponse."""
        from talu import AsyncChat
        from talu.chat import AsyncStreamingResponse

        chat = AsyncChat(test_model_path)
        response = await chat.send("Hello", stream=True, max_tokens=5)

        assert isinstance(response, AsyncStreamingResponse)
        chunks = []
        async for chunk in response:
            assert isinstance(chunk, str)
            chunks.append(chunk)

        assert len(chunks) >= 1


# =============================================================================
# Fork
# =============================================================================


class TestFork:
    """Tests for chat.fork() - branching conversations."""

    def test_fork_creates_independent_chat(self, test_model_path):
        """chat.fork() creates independent copy."""
        chat = Chat(test_model_path, system="You are helpful.")
        chat.send("Hello", max_tokens=3)

        forked = chat.fork()

        # Same history length
        assert len(forked.items) == len(chat.items)
        # Same content
        for i in range(len(chat.items)):
            assert forked.items[i].text == chat.items[i].text

    def test_fork_is_independent(self, test_model_path):
        """Changes to forked chat don't affect original."""
        chat = Chat(test_model_path)
        chat.send("Hello", max_tokens=3)
        original_len = len(chat.items)

        forked = chat.fork()
        forked.send("Another message", max_tokens=3)

        # Original unchanged
        assert len(chat.items) == original_len
        # Forked has more
        assert len(forked.items) > original_len

    def test_fork_via_response_chat(self, test_model_path):
        """response.chat.fork() is the recommended pattern."""
        chat = Chat(test_model_path)
        response = chat.send("Hello", max_tokens=3)

        forked = response.chat.fork()

        assert isinstance(forked, Chat)
        assert len(forked.items) == len(chat.items)

    def test_multiple_forks_are_independent(self, test_model_path):
        """Multiple forks from same point are independent."""
        chat = Chat(test_model_path)
        response = chat.send("I have ingredients", max_tokens=3)

        asian = response.chat.fork()
        italian = response.chat.fork()

        asian.send("Asian recipe", max_tokens=3)
        italian.send("Italian recipe", max_tokens=3)

        # Each fork diverged independently
        assert len(asian.items) == len(italian.items)
        # But their content differs
        assert (
            asian.items.last.text != italian.items.last.text
            or asian.items[-2].text != italian.items[-2].text
        )


# =============================================================================
# Response
# =============================================================================


class TestResponse:
    """Tests for Response object."""

    def test_response_str(self, test_model_path):
        """str(response) returns the generated text."""
        chat = Chat(test_model_path)
        response = chat.send("Hello", max_tokens=3)

        text = str(response)
        assert isinstance(text, str)
        assert text == response.text

    def test_response_usage(self, test_model_path):
        """response.usage contains token counts."""
        chat = Chat(test_model_path)
        response = chat.send("Hello", max_tokens=3)

        assert response.usage is not None
        # Note: prompt_tokens is 0 until Router C API returns it
        assert response.usage.prompt_tokens >= 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens >= response.usage.completion_tokens

    def test_response_finish_reason(self, test_model_path):
        """response.finish_reason indicates why generation stopped."""
        chat = Chat(test_model_path)
        response = chat.send("Hello", max_tokens=3)

        # Should be 'length' (hit max_tokens) or 'stop' (hit EOS)
        assert response.finish_reason in ("length", "stop", "max_tokens")

    def test_response_model(self, test_model_path):
        """response.model indicates which model was used."""
        chat = Chat(test_model_path)
        response = chat.send("Hello", max_tokens=3)

        assert response.model is not None
        assert isinstance(response.model, str)


# =============================================================================
# Messages (Read-Only)
# =============================================================================


class TestMessages:
    """Tests for chat.messages - read-only message access."""

    def test_messages_is_list_like(self):
        """messages supports list operations."""
        chat = Chat(system="Test")

        # Length
        assert len(chat.items) == 1

        # Indexing
        assert chat.items[0].role.name.lower() == "system"

        # Negative indexing
        assert chat.items[-1].role.name.lower() == "system"

        # Iteration
        roles = [item.role.name.lower() for item in chat.items]
        assert roles == ["system"]

    def test_messages_slicing(self):
        """items supports slicing."""
        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "User1"},
                    {"role": "assistant", "content": "Asst1"},
                    {"role": "user", "content": "User2"},
                ]
            }
        )

        # Slice
        subset = chat.items[1:3]
        assert len(subset) == 2
        assert subset[0].role.name.lower() == "user"
        assert subset[1].role.name.lower() == "assistant"

    def test_items_iteration(self):
        """items supports iteration and typed access."""
        chat = Chat(system="Test")

        items_list = list(chat.items)
        assert len(items_list) == 1
        assert items_list[0].role.name.lower() == "system"
        assert items_list[0].text == "Test"

    def test_messages_system_property(self):
        """messages.system returns system prompt content."""
        chat = Chat(system="You are helpful.")
        assert chat.items.system == "You are helpful."

    def test_messages_last_property(self):
        """messages.last returns last message."""
        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Hello"},
                ]
            }
        )

        assert chat.items.last.role.name.lower() == "user"
        assert chat.items.last.text == "Hello"


# =============================================================================
# Config Overrides
# =============================================================================


class TestConfigOverrides:
    """Tests for per-call config overrides."""

    def test_kwargs_override_default_config(self, test_model_path):
        """Kwargs override the default config."""
        chat = Chat(test_model_path, config=GenerationConfig(max_tokens=100))
        response = chat.send("Hello", max_tokens=3)

        # Should have used max_tokens=3, not 100
        assert response.usage.completion_tokens <= 5  # Allow some tolerance

    def test_config_param_overrides_default(self, test_model_path):
        """config= parameter overrides default config."""
        chat = Chat(test_model_path, config=GenerationConfig(max_tokens=100))
        override = GenerationConfig(max_tokens=3)
        response = chat.send("Hello", config=override)

        assert response.usage.completion_tokens <= 5

    def test_kwargs_override_config_param(self, test_model_path):
        """Kwargs take precedence over config= parameter."""
        chat = Chat(test_model_path)
        override = GenerationConfig(max_tokens=100)
        response = chat.send("Hello", config=override, max_tokens=3)

        # max_tokens=3 from kwargs should win
        assert response.usage.completion_tokens <= 5


# =============================================================================
# Persistence / Serialization
# =============================================================================


class TestPersistence:
    """Tests for saving and restoring chats."""

    def test_to_dict_roundtrip(self, test_model_path):
        """to_dict() -> from_dict() preserves message state."""
        chat = Chat(test_model_path, system="You are helpful.")
        chat.send("Hello", max_tokens=3)

        data = chat.to_dict()
        restored = Chat.from_dict(data, model=test_model_path)

        # Roundtrip preserves messages (reasoning items are not in completions format)
        orig_msgs = [i for i in chat.items if isinstance(i, MessageItem)]
        rest_msgs = [i for i in restored.items if isinstance(i, MessageItem)]
        assert len(rest_msgs) == len(orig_msgs)
        for orig, rest in zip(orig_msgs, rest_msgs, strict=True):
            assert rest.role == orig.role
            assert rest.text == orig.text

    def test_from_dict_with_messages_only(self, test_model_path):
        """from_dict() works with just messages."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        chat = Chat.from_dict({"messages": messages}, model=test_model_path)

        assert len(chat.items) == 3
        assert chat.items[0].text == "You are helpful."

    def test_restored_chat_can_continue(self, test_model_path):
        """Restored chat can continue the conversation."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        chat = Chat.from_dict({"messages": messages}, model=test_model_path)
        response = chat.send("How are you?", max_tokens=3)

        assert isinstance(response, Response)
        # original 2 + new user + generation output (may include reasoning items)
        assert len(chat.items) >= 4
