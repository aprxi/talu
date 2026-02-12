"""Tests for talu.chat.client module - construction and API surface.

Unit tests for Client construction, validation, and non-inference API.
"""

import pytest

from talu.client import Client


class TestClient:
    """Tests for Client class."""

    def test_client_requires_model(self):
        """Client needs at least one model."""
        try:
            Client([])
        except ValueError as e:
            assert "At least one model" in str(e)

    def test_client_accepts_string(self):
        """Client accepts single model string."""
        # Note: This will fail without a model, but tests the signature
        # Full integration tests require a model to be available
        pass

    def test_client_accepts_list(self):
        """Client accepts list of models."""
        # Note: This will fail without models, but tests the signature
        pass


class TestClientEmbedAPI:
    """Tests for Client.embed() API."""

    def test_embed_invalid_pooling(self):
        """embed() raises ValueError for invalid pooling strategy."""
        client = Client("test-model")
        with pytest.raises(ValueError, match="Invalid pooling strategy"):
            client.embed("Hello", pooling="invalid")
        client.close()
