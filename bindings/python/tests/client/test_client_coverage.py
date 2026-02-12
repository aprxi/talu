"""
Additional tests for talu/chat/client.py coverage.

Targets uncovered edge cases, error paths, and internal methods.
"""

import pytest

from talu import AsyncClient, Client
from talu.exceptions import StateError, ValidationError

# =============================================================================
# Client Construction Tests
# =============================================================================


class TestClientConstruction:
    """Tests for Client construction paths."""

    def test_single_string_model(self):
        """Client accepts single string model."""
        client = Client("test-model")
        assert client.models == ["test-model"]
        assert client.default_model == "test-model"
        client.close()

    def test_list_of_models(self):
        """Client accepts list of models."""
        client = Client(["model1", "model2"])
        assert client.models == ["model1", "model2"]
        assert client.default_model == "model1"
        client.close()

    def test_empty_list_raises_validation_error(self):
        """Client with empty list raises ValidationError."""
        with pytest.raises(ValidationError, match="At least one model"):
            Client([])


class TestAsyncClientConstruction:
    """Tests for AsyncClient construction paths."""

    def test_single_string_model(self):
        """AsyncClient accepts single string model."""
        client = AsyncClient("test-model")
        assert client.models == ["test-model"]
        assert client.default_model == "test-model"
        # Sync close for test cleanup
        client._router.close()
        client._closed = True

    def test_empty_list_raises_validation_error(self):
        """AsyncClient with empty list raises ValidationError."""
        with pytest.raises(ValidationError, match="At least one model"):
            AsyncClient([])


# =============================================================================
# Default Model Property Tests
# =============================================================================


class TestDefaultModelProperty:
    """Tests for default_model getter/setter."""

    def test_get_default_model(self):
        """default_model returns first model."""
        client = Client(["model1", "model2"])
        assert client.default_model == "model1"
        client.close()

    def test_set_default_model(self):
        """default_model setter updates default."""
        client = Client(["model1", "model2"])
        client.default_model = "model2"
        assert client.default_model == "model2"
        client.close()

    def test_router_property(self):
        """router property returns Router instance."""
        client = Client("test-model")
        assert client.router is not None
        assert client.router is client._router
        client.close()


class TestAsyncDefaultModelProperty:
    """Tests for AsyncClient default_model getter/setter."""

    def test_set_default_model(self):
        """default_model setter updates default."""
        client = AsyncClient(["model1", "model2"])
        client.default_model = "model2"
        assert client.default_model == "model2"
        client._router.close()
        client._closed = True

    def test_router_property(self):
        """router property returns Router instance."""
        client = AsyncClient("test-model")
        assert client.router is not None
        assert client.router is client._router
        client._router.close()
        client._closed = True


# =============================================================================
# Closed State Tests
# =============================================================================


class TestClientClosedState:
    """Tests for Client closed state error handling."""

    def test_ask_after_close_raises(self):
        """ask() on closed client raises StateError."""
        client = Client("test-model")
        client.close()
        with pytest.raises(StateError, match="has been closed"):
            client.ask("Hello")

    def test_chat_after_close_raises(self):
        """chat() on closed client raises StateError."""
        client = Client("test-model")
        client.close()
        with pytest.raises(StateError, match="has been closed"):
            client.chat()

    def test_stream_after_close_raises(self):
        """stream() on closed client raises StateError."""
        client = Client("test-model")
        client.close()
        with pytest.raises(StateError, match="has been closed"):
            list(client.stream("Hello"))

    def test_embed_after_close_raises(self):
        """embed() on closed client raises StateError."""
        client = Client("test-model")
        client.close()
        with pytest.raises(StateError, match="has been closed"):
            client.embed("Hello")

    def test_embed_batch_after_close_raises(self):
        """embed_batch() on closed client raises StateError."""
        client = Client("test-model")
        client.close()
        with pytest.raises(StateError, match="has been closed"):
            client.embed_batch(["Hello", "World"])

    def test_embedding_dim_after_close_raises(self):
        """embedding_dim() on closed client raises StateError."""
        client = Client("test-model")
        client.close()
        with pytest.raises(StateError, match="has been closed"):
            client.embedding_dim()

    def test_capabilities_after_close_raises(self):
        """capabilities() on closed client raises StateError."""
        client = Client("test-model")
        client.close()
        with pytest.raises(StateError, match="has been closed"):
            client.capabilities()

    def test_close_idempotent(self):
        """close() can be called multiple times safely."""
        client = Client("test-model")
        client.close()
        client.close()  # Should not raise
        assert client._closed is True


class TestAsyncClientClosedState:
    """Tests for AsyncClient closed state error handling."""

    def test_check_closed_raises(self):
        """_check_closed() raises on closed client."""
        client = AsyncClient("test-model")
        client._router.close()
        client._closed = True
        with pytest.raises(StateError, match="has been closed"):
            client._check_closed()

    @pytest.mark.asyncio
    async def test_ask_after_close_raises(self):
        """ask() on closed client raises StateError."""
        client = AsyncClient("test-model")
        await client.close()
        with pytest.raises(StateError, match="has been closed"):
            await client.ask("Hello")

    def test_chat_after_close_raises(self):
        """chat() on closed client raises StateError."""
        client = AsyncClient("test-model")
        client._router.close()
        client._closed = True
        with pytest.raises(StateError, match="has been closed"):
            client.chat()

    def test_embed_after_close_raises(self):
        """embed() on closed client raises StateError."""
        client = AsyncClient("test-model")
        client._router.close()
        client._closed = True
        with pytest.raises(StateError, match="has been closed"):
            client.embed("Hello")

    def test_embed_batch_after_close_raises(self):
        """embed_batch() on closed client raises StateError."""
        client = AsyncClient("test-model")
        client._router.close()
        client._closed = True
        with pytest.raises(StateError, match="has been closed"):
            client.embed_batch(["Hello"])

    def test_embedding_dim_after_close_raises(self):
        """embedding_dim() on closed client raises StateError."""
        client = AsyncClient("test-model")
        client._router.close()
        client._closed = True
        with pytest.raises(StateError, match="has been closed"):
            client.embedding_dim()

    def test_capabilities_after_close_raises(self):
        """capabilities() on closed client raises StateError."""
        client = AsyncClient("test-model")
        client._router.close()
        client._closed = True
        with pytest.raises(StateError, match="has been closed"):
            client.capabilities()

    @pytest.mark.asyncio
    async def test_stream_after_close_raises(self):
        """stream() on closed client raises StateError."""
        client = AsyncClient("test-model")
        await client.close()
        with pytest.raises(StateError, match="has been closed"):
            async for _ in client.stream("Hello"):
                pass

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        """close() can be called multiple times safely."""
        client = AsyncClient("test-model")
        await client.close()
        await client.close()  # Should not raise
        assert client._closed is True


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestClientContextManager:
    """Tests for Client context manager protocol."""

    def test_context_manager_returns_self(self):
        """__enter__ returns the client instance."""
        client = Client("test-model")
        with client as c:
            assert c is client
        assert client._closed is True

    def test_context_manager_closes_on_exit(self):
        """__exit__ closes the client."""
        client = Client("test-model")
        with client:
            assert client._closed is False
        assert client._closed is True

    def test_context_manager_closes_on_exception(self):
        """__exit__ closes client even on exception."""
        client = Client("test-model")
        try:
            with client:
                raise RuntimeError("Test error")
        except RuntimeError:
            pass
        assert client._closed is True


class TestAsyncClientContextManager:
    """Tests for AsyncClient async context manager protocol."""

    @pytest.mark.asyncio
    async def test_context_manager_returns_self(self):
        """__aenter__ returns the client instance."""
        client = AsyncClient("test-model")
        async with client as c:
            assert c is client
        assert client._closed is True

    @pytest.mark.asyncio
    async def test_context_manager_closes_on_exit(self):
        """__aexit__ closes the client."""
        client = AsyncClient("test-model")
        async with client:
            assert client._closed is False
        assert client._closed is True


# =============================================================================
# Chat Creation with Messages Tests
# =============================================================================


class TestChatWithMessages:
    """Tests for chat() with messages parameter."""

    def test_chat_with_messages_raises_not_implemented(self):
        """chat(messages=...) raises NotImplementedError."""
        client = Client("test-model")
        try:
            with pytest.raises(NotImplementedError, match="Loading initial messages"):
                client.chat(messages=[{"role": "user", "content": "Hi"}])
        finally:
            client.close()


class TestAsyncChatWithMessages:
    """Tests for AsyncClient.chat() with messages parameter."""

    def test_chat_with_messages_raises_not_implemented(self):
        """chat(messages=...) raises NotImplementedError."""
        client = AsyncClient("test-model")
        try:
            with pytest.raises(NotImplementedError, match="Loading initial messages"):
                client.chat(messages=[{"role": "user", "content": "Hi"}])
        finally:
            client._router.close()
            client._closed = True


# =============================================================================
# Repr Tests
# =============================================================================


class TestClientRepr:
    """Tests for Client __repr__."""

    def test_repr_open_client(self):
        """Repr shows open status."""
        client = Client("test-model")
        repr_str = repr(client)
        assert "Client" in repr_str
        assert "test-model" in repr_str
        assert "open" in repr_str
        client.close()

    def test_repr_closed_client(self):
        """Repr shows closed status."""
        client = Client("test-model")
        client.close()
        repr_str = repr(client)
        assert "closed" in repr_str


class TestAsyncClientRepr:
    """Tests for AsyncClient __repr__."""

    def test_repr_open_client(self):
        """Repr shows open status."""
        client = AsyncClient("test-model")
        repr_str = repr(client)
        assert "AsyncClient" in repr_str
        assert "test-model" in repr_str
        assert "open" in repr_str
        client._router.close()
        client._closed = True

    def test_repr_closed_client(self):
        """Repr shows closed status."""
        client = AsyncClient("test-model")
        client._router.close()
        client._closed = True
        repr_str = repr(client)
        assert "closed" in repr_str


# =============================================================================
# Capabilities Tests (require real model)
# =============================================================================


@pytest.mark.requires_model
class TestCapabilities:
    """Tests for Client.capabilities()."""

    def test_capabilities_returns_capabilities_object(self, test_model_path):
        """capabilities() returns Capabilities object."""
        from talu.router import Capabilities

        client = Client(test_model_path)
        try:
            caps = client.capabilities()
            assert isinstance(caps, Capabilities)
        finally:
            client.close()

    def test_capabilities_with_explicit_model(self, test_model_path):
        """capabilities(model=...) queries specific model."""
        from talu.router import Capabilities

        client = Client(test_model_path)
        try:
            caps = client.capabilities(model=test_model_path)
            assert isinstance(caps, Capabilities)
        finally:
            client.close()


@pytest.mark.requires_model
class TestAsyncCapabilities:
    """Tests for AsyncClient.capabilities()."""

    def test_capabilities_returns_capabilities_object(self, test_model_path):
        """capabilities() returns Capabilities object."""
        from talu.router import Capabilities

        client = AsyncClient(test_model_path)
        try:
            caps = client.capabilities()
            assert isinstance(caps, Capabilities)
        finally:
            client._router.close()
            client._closed = True
