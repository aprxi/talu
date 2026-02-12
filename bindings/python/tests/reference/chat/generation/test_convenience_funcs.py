"""
Tests for module-level convenience functions in talu.chat.api.

These functions wrap Client internally, providing the simplest API for casual users.
Tests validate:
- One-shot: ask() (PRIMARY - auto-cleans resources, optional system prompt)
- Streaming: stream()
- Raw completion: raw_complete() (POWER USER - no chat template)

Note: Async convenience functions (achat, complete_async, stream_async) were
intentionally removed. Async users should use AsyncClient/AsyncChat directly.
See talu.chat.api docstring for rationale.
"""

import pytest

import talu
from talu import Chat, CompletionOptions, GenerationConfig, Tokenizer
from talu.chat import Response
from talu.exceptions import StateError

# =============================================================================
# Raw Completion: raw_complete() - Power User Function
# =============================================================================


class TestRawComplete:
    """Tests for talu.raw_complete() - raw completion without chat templates."""

    def test_raw_complete_returns_response(self, test_model_path):
        """raw_complete() returns a Response object."""
        response = talu.raw_complete(test_model_path, "Hello", max_tokens=3)

        assert isinstance(response, Response)
        assert response.usage.completion_tokens > 0

    def test_raw_complete_with_config(self, test_model_path):
        """raw_complete() accepts GenerationConfig."""
        config = GenerationConfig(max_tokens=5, temperature=0.1, seed=42)
        response = talu.raw_complete(test_model_path, "Count: 1", config=config)

        assert isinstance(response, Response)
        assert response.usage.completion_tokens <= 7  # Allow slight tolerance

    def test_raw_complete_kwargs_override_config(self, test_model_path):
        """Kwargs override config values."""
        config = GenerationConfig(max_tokens=100)
        response = talu.raw_complete(test_model_path, "Hello", config=config, max_tokens=3)

        # max_tokens=3 from kwargs should take precedence
        assert response.usage.completion_tokens <= 5

    def test_raw_complete_response_has_usage(self, test_model_path):
        """Response includes usage stats."""
        response = talu.raw_complete(test_model_path, "Hello", max_tokens=3)

        assert response.usage is not None
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens >= response.usage.completion_tokens

    def test_raw_complete_with_system(self, test_model_path):
        """raw_complete() accepts system prompt (same signature as ask())."""
        response = talu.raw_complete(
            test_model_path,
            "Hello",
            system="You are helpful.",
            max_tokens=5,
        )

        assert isinstance(response, Response)
        assert response.usage.completion_tokens > 0

    def test_raw_complete_with_completion_opts(self, test_model_path):
        """raw_complete() accepts CompletionOptions for raw-only features."""
        # Note: echo_prompt and other CompletionOptions features are not yet
        # implemented in the backend. This test verifies the API accepts the
        # parameter without error when using default values.
        opts = CompletionOptions()  # All defaults (no backend params set)
        response = talu.raw_complete(
            test_model_path,
            "Continue: ",
            completion_opts=opts,
            max_tokens=5,
        )

        assert isinstance(response, Response)
        assert response.usage.completion_tokens > 0

    def test_raw_complete_uses_raw_prompt_tokens(self, test_model_path):
        """raw_complete() does not apply chat templates to prompt tokens."""
        prompt = "Hello world"
        tokenizer = Tokenizer(test_model_path)
        token_ids = tokenizer.encode(prompt, special_tokens=False)

        response = talu.raw_complete(test_model_path, prompt, max_tokens=3)

        assert response.usage is not None
        # Backend may prepend a BOS token; allow +1 tolerance.
        # The key contract: no chat template tokens are added (which would add many more).
        assert response.usage.prompt_tokens in (len(token_ids), len(token_ids) + 1)

    def test_completion_options_class_exists(self):
        """CompletionOptions can be imported from talu."""
        # Verify the class is importable and has expected attributes
        opts = CompletionOptions()
        assert opts.token_ids is None
        assert opts.continue_from_token_id is None
        assert opts.echo_prompt is False

        # Verify it can be instantiated with all parameters
        opts2 = CompletionOptions(
            token_ids=[1, 2, 3],
            continue_from_token_id=42,
            echo_prompt=True,
        )
        assert opts2.token_ids == [1, 2, 3]
        assert opts2.continue_from_token_id == 42
        assert opts2.echo_prompt is True


# =============================================================================
# Streaming: stream()
# =============================================================================


class TestStream:
    """Tests for talu.stream() - streaming text completion."""

    def test_stream_yields_chunks(self, test_model_path):
        """stream() yields string chunks."""
        chunks = list(talu.stream(test_model_path, "Hello", max_tokens=5))

        assert len(chunks) >= 1
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_stream_accumulates_text(self, test_model_path):
        """Accumulated chunks form complete text."""
        chunks = []
        for chunk in talu.stream(test_model_path, "Say hi", max_tokens=5):
            chunks.append(chunk)

        full_text = "".join(chunks)
        assert len(full_text) > 0

    def test_stream_with_config(self, test_model_path):
        """stream() accepts GenerationConfig."""
        config = GenerationConfig(max_tokens=3, temperature=0.1, seed=42)
        chunks = list(talu.stream(test_model_path, "Hello", config=config))

        assert len(chunks) >= 1


# =============================================================================
# One-shot: ask()
# =============================================================================


class TestAsk:
    """Tests for talu.ask() - one-shot question-answer with auto-cleanup."""

    def test_ask_returns_response(self, test_model_path):
        """ask() returns a Response object."""
        response = talu.ask(test_model_path, "Hello", max_tokens=3)

        assert isinstance(response, Response)
        assert response.usage.completion_tokens > 0

    def test_ask_with_system_prompt(self, test_model_path):
        """ask() accepts system prompt."""
        response = talu.ask(
            test_model_path,
            "Who are you?",
            system="You are a helpful assistant named Bob.",
            max_tokens=10,
        )

        assert isinstance(response, Response)
        assert response.usage.completion_tokens > 0

    def test_ask_with_config(self, test_model_path):
        """ask() accepts GenerationConfig."""
        config = GenerationConfig(max_tokens=5, seed=42)
        response = talu.ask(test_model_path, "Hello", config=config)

        assert isinstance(response, Response)

    def test_ask_response_is_detached(self, test_model_path):
        """ask() returns a detached Response that cannot be replied to."""
        response = talu.ask(test_model_path, "Hello", max_tokens=3)

        # Response should be detached (no associated Chat)
        assert response._chat is None

        # Attempting to append should raise StateError
        with pytest.raises(StateError, match="one-shot"):
            response.append("More")

    def test_ask_safe_in_loop(self, test_model_path):
        """ask() is safe in loops (no resource leak)."""
        # This would leak resources with the old talu.chat() API
        results = []
        for question in ["Hello", "Hi"]:
            response = talu.ask(test_model_path, question, max_tokens=2)
            results.append(str(response))

        assert len(results) == 2


# =============================================================================
# Multi-turn: Chat class
# =============================================================================


class TestChatClass:
    """Tests for talu.Chat() - multi-turn conversations."""

    def test_chat_class_basic(self, test_model_path):
        """Chat() can be created and generate responses."""
        chat = Chat(model=test_model_path)

        assert isinstance(chat, Chat)

    def test_chat_with_system(self, test_model_path):
        """Chat() with system message."""
        chat = Chat(model=test_model_path, system="You are helpful.")

        assert len(chat.items) == 1
        assert chat.items[0].role.name.lower() == "system"
        assert chat.items[0].text == "You are helpful."

    def test_chat_multi_turn_can_generate(self, test_model_path):
        """Chat can generate responses."""
        chat = Chat(model=test_model_path, system="Be brief.")

        response = chat.send("Hello", max_tokens=3)

        assert isinstance(response, Response)
        assert len(chat.items) >= 3  # system + user + assistant

    def test_chat_append_works(self, test_model_path):
        """Response.append() works for Chat-attached responses."""
        chat = Chat(model=test_model_path)

        response1 = chat.send("My name is Alice", max_tokens=5)
        response2 = response1.append("What did I just tell you?", max_tokens=10)

        # Should have 4 messages: 2 user + 2 assistant
        assert len(chat.items) == 4
        assert isinstance(response2, Response)


# =============================================================================
# Async API Design Note
# =============================================================================
#
# Async convenience functions (complete_async, stream_async, achat) were
# intentionally removed from the public API.
#
# Async users should use AsyncClient or AsyncChat directly:
#
#     from talu import AsyncClient
#     async with AsyncClient("model") as client:
#         response = await client.complete("Hello")
#
# Or:
#     from talu import AsyncChat
#     chat = AsyncChat("model")
#     response = await chat.send("Hello")
#
# See talu.chat.api docstring for full rationale.
