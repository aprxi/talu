"""
Unit tests for module-level convenience functions in talu.chat.api.

These tests mock the Client to avoid model loading, focusing on:
- Correct delegation to Client methods
- Resource cleanup (context manager usage)
- Response detachment for ask()
"""

from unittest.mock import MagicMock, patch

import talu
from talu.router import CompletionOptions, GenerationConfig

# The Client is imported inside each function via `from talu.client import Client`
# So we need to patch at the source module: talu.client.Client
CLIENT_PATCH_PATH = "talu.client.Client"


class TestAsk:
    """Tests for talu.ask() - one-shot question-answer with auto-cleanup."""

    def test_ask_delegates_to_client_chat_send(self):
        """ask() creates a Client, calls chat().send(), and returns response."""
        mock_response = MagicMock()
        mock_response._chat = MagicMock()
        mock_response._msg_index = 0

        mock_chat = MagicMock()
        mock_chat.send.return_value = mock_response

        mock_client = MagicMock()
        mock_client.chat.return_value = mock_chat
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)

        with patch(CLIENT_PATCH_PATH, return_value=mock_client):
            response = talu.ask("test-model", "Hello", max_tokens=10)

        # Verify Client was created with correct model
        mock_client.chat.assert_called_once()
        # Verify send was called with correct args
        mock_chat.send.assert_called_once_with("Hello", stream=False, max_tokens=10)
        # Verify response is detached
        assert response._chat is None
        assert response._msg_index == -1

    def test_ask_with_system_prompt(self):
        """ask() passes system prompt to client.chat()."""
        mock_response = MagicMock()
        mock_response._chat = MagicMock()
        mock_response._msg_index = 0

        mock_chat = MagicMock()
        mock_chat.send.return_value = mock_response

        mock_client = MagicMock()
        mock_client.chat.return_value = mock_chat
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)

        with patch(CLIENT_PATCH_PATH, return_value=mock_client):
            talu.ask("test-model", "Hello", system="Be helpful")

        mock_client.chat.assert_called_once()
        call_kwargs = mock_client.chat.call_args.kwargs
        assert call_kwargs.get("system") == "Be helpful"

    def test_ask_with_config(self):
        """ask() passes GenerationConfig to client.chat()."""
        mock_response = MagicMock()
        mock_response._chat = MagicMock()
        mock_response._msg_index = 0

        mock_chat = MagicMock()
        mock_chat.send.return_value = mock_response

        mock_client = MagicMock()
        mock_client.chat.return_value = mock_chat
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)

        config = GenerationConfig(max_tokens=50, temperature=0.5)

        with patch(CLIENT_PATCH_PATH, return_value=mock_client):
            talu.ask("test-model", "Hello", config=config)

        call_kwargs = mock_client.chat.call_args.kwargs
        assert call_kwargs.get("config") == config

    def test_ask_uses_context_manager(self):
        """ask() uses context manager to ensure cleanup."""
        mock_response = MagicMock()
        mock_response._chat = MagicMock()
        mock_response._msg_index = 0

        mock_chat = MagicMock()
        mock_chat.send.return_value = mock_response

        mock_client = MagicMock()
        mock_client.chat.return_value = mock_chat
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)

        with patch(CLIENT_PATCH_PATH, return_value=mock_client):
            talu.ask("test-model", "Hello")

        # Verify context manager was used
        mock_client.__enter__.assert_called_once()
        mock_client.__exit__.assert_called_once()


class TestStream:
    """Tests for talu.stream() - streaming text completion."""

    def test_stream_delegates_to_client_stream(self):
        """stream() creates a Client and yields from client.stream()."""
        mock_client = MagicMock()
        mock_client.stream.return_value = iter(["Hello", " world"])
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)

        with patch(CLIENT_PATCH_PATH, return_value=mock_client):
            chunks = list(talu.stream("test-model", "Hi"))

        assert chunks == ["Hello", " world"]
        mock_client.stream.assert_called_once()

    def test_stream_with_config(self):
        """stream() passes GenerationConfig to client.stream()."""
        mock_client = MagicMock()
        mock_client.stream.return_value = iter(["chunk"])
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)

        config = GenerationConfig(max_tokens=10)

        with patch(CLIENT_PATCH_PATH, return_value=mock_client):
            list(talu.stream("test-model", "Hi", config=config))

        call_kwargs = mock_client.stream.call_args.kwargs
        assert call_kwargs.get("config") == config

    def test_stream_with_kwargs(self):
        """stream() passes kwargs to client.stream()."""
        mock_client = MagicMock()
        mock_client.stream.return_value = iter(["chunk"])
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)

        with patch(CLIENT_PATCH_PATH, return_value=mock_client):
            list(talu.stream("test-model", "Hi", temperature=0.5, max_tokens=20))

        call_kwargs = mock_client.stream.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.5
        assert call_kwargs.get("max_tokens") == 20

    def test_stream_uses_context_manager(self):
        """stream() uses context manager to ensure cleanup."""
        mock_client = MagicMock()
        mock_client.stream.return_value = iter(["chunk"])
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)

        with patch(CLIENT_PATCH_PATH, return_value=mock_client):
            # Must consume the generator to trigger cleanup
            list(talu.stream("test-model", "Hi"))

        mock_client.__enter__.assert_called_once()
        mock_client.__exit__.assert_called_once()


class TestRawComplete:
    """Tests for talu.raw_complete() - raw completion without chat templates."""

    def test_raw_complete_delegates_to_client(self):
        """raw_complete() creates a Client and calls client.raw_complete()."""
        mock_response = MagicMock()

        mock_client = MagicMock()
        mock_client.raw_complete.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)

        with patch(CLIENT_PATCH_PATH, return_value=mock_client):
            response = talu.raw_complete("test-model", "The sky is")

        assert response == mock_response
        mock_client.raw_complete.assert_called_once()
        # Verify prompt was passed
        call_args = mock_client.raw_complete.call_args
        assert call_args.args[0] == "The sky is"

    def test_raw_complete_with_system(self):
        """raw_complete() passes system prompt."""
        mock_response = MagicMock()

        mock_client = MagicMock()
        mock_client.raw_complete.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)

        with patch(CLIENT_PATCH_PATH, return_value=mock_client):
            talu.raw_complete("test-model", "Hello", system="Be brief")

        call_kwargs = mock_client.raw_complete.call_args.kwargs
        assert call_kwargs.get("system") == "Be brief"

    def test_raw_complete_with_config(self):
        """raw_complete() passes GenerationConfig."""
        mock_response = MagicMock()

        mock_client = MagicMock()
        mock_client.raw_complete.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)

        config = GenerationConfig(max_tokens=50)

        with patch(CLIENT_PATCH_PATH, return_value=mock_client):
            talu.raw_complete("test-model", "Hello", config=config)

        call_kwargs = mock_client.raw_complete.call_args.kwargs
        assert call_kwargs.get("config") == config

    def test_raw_complete_with_completion_opts(self):
        """raw_complete() passes CompletionOptions."""
        mock_response = MagicMock()

        mock_client = MagicMock()
        mock_client.raw_complete.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)

        opts = CompletionOptions(echo_prompt=True)

        with patch(CLIENT_PATCH_PATH, return_value=mock_client):
            talu.raw_complete("test-model", "Hello", completion_opts=opts)

        call_kwargs = mock_client.raw_complete.call_args.kwargs
        assert call_kwargs.get("completion_opts") == opts

    def test_raw_complete_with_kwargs(self):
        """raw_complete() passes kwargs."""
        mock_response = MagicMock()

        mock_client = MagicMock()
        mock_client.raw_complete.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)

        with patch(CLIENT_PATCH_PATH, return_value=mock_client):
            talu.raw_complete("test-model", "Hello", temperature=0.7, max_tokens=100)

        call_kwargs = mock_client.raw_complete.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.7
        assert call_kwargs.get("max_tokens") == 100

    def test_raw_complete_uses_context_manager(self):
        """raw_complete() uses context manager to ensure cleanup."""
        mock_response = MagicMock()

        mock_client = MagicMock()
        mock_client.raw_complete.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)

        with patch(CLIENT_PATCH_PATH, return_value=mock_client):
            talu.raw_complete("test-model", "Hello")

        mock_client.__enter__.assert_called_once()
        mock_client.__exit__.assert_called_once()
