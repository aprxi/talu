"""Integration tests for Client embedding requiring a real model.

Tests that require TEST_MODEL_URI_TEXT for Client.embed() and embedding_dim().
"""

import pytest

from talu.client import Client


@pytest.mark.requires_model
class TestClientEmbedIntegration:
    """Integration tests for Client.embed() requiring a real model."""

    def test_embed_basic(self, test_model_path):
        """embed() returns a valid embedding vector."""
        client = Client(test_model_path)
        embedding = client.embed("Hello, world!")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
        client.close()

    def test_embed_with_pooling(self, test_model_path):
        """embed() accepts pooling parameter."""
        client = Client(test_model_path)
        embedding = client.embed("Hello", pooling="mean")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        client.close()

    def test_embed_with_normalize(self, test_model_path):
        """embed() accepts normalize parameter."""
        import math

        client = Client(test_model_path)
        embedding = client.embed("Hello", normalize=True)
        norm = math.sqrt(sum(x * x for x in embedding))
        assert abs(norm - 1.0) < 0.01
        client.close()


@pytest.mark.requires_model
class TestClientEmbeddingDim:
    """Tests for Client.embedding_dim() requiring a real model."""

    def test_embedding_dim_returns_int(self, test_model_path):
        """embedding_dim() returns positive integer."""
        client = Client(test_model_path)
        dim = client.embedding_dim()
        assert isinstance(dim, int)
        assert dim > 0
        client.close()

    def test_embedding_dim_matches_embed_output(self, test_model_path):
        """embedding_dim() matches actual embed() output dimension."""
        client = Client(test_model_path)
        dim = client.embedding_dim()
        embedding = client.embed("Hello")
        assert len(embedding) == dim
        client.close()
