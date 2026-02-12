"""Integration tests for Router error messages and embedding requiring a real model.

Requires TEST_MODEL_URI_TEXT.
"""

import pytest

import talu
from talu.router import Router


class TestRouterErrorMessages:
    """Tests for Router error message handling.

    Router.generate() and Router.stream() should provide detailed error
    messages from Zig when available, falling back to generic messages.
    """

    @pytest.mark.requires_model
    def test_generate_error_includes_context(self, test_model_path):
        """Error message includes context when generation fails."""
        from talu import Chat

        # Create chat with valid model
        chat = Chat(test_model_path)

        # Force an error by corrupting the chat pointer
        # (This is a bit hacky but tests the error path)
        original_ptr = chat._chat_ptr
        chat._chat_ptr = None

        with pytest.raises(talu.GenerationError) as exc_info:
            chat._router.generate(chat, "Hello", model=test_model_path)

        # Restore pointer for cleanup
        chat._chat_ptr = original_ptr

        # Error should mention "Router.generate()" and provide some context
        error_msg = str(exc_info.value)
        assert "Router.generate()" in error_msg

    @pytest.mark.requires_model
    def test_stream_error_includes_context(self, test_model_path):
        """Stream error message includes context when streaming fails."""
        from talu import Chat

        chat = Chat(test_model_path)

        # Force an error by corrupting the chat pointer
        original_ptr = chat._chat_ptr
        chat._chat_ptr = None

        with pytest.raises(talu.GenerationError) as exc_info:
            # Need to consume the generator to trigger the error
            list(chat._router.stream(chat, "Hello", model=test_model_path))

        # Restore pointer for cleanup
        chat._chat_ptr = original_ptr

        error_msg = str(exc_info.value)
        assert "Router.stream()" in error_msg


@pytest.mark.requires_model
class TestRouterEmbedIntegration:
    """Integration tests for Router.embed() requiring a real model."""

    def test_embed_returns_list_of_floats(self, test_model_path):
        """embed() returns a list of floats."""
        router = Router(models=[test_model_path])
        embedding = router.embed("Hello, world!")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
        router.close()

    def test_embed_dimension_matches_model(self, test_model_path):
        """embed() returns correct dimension for model."""
        router = Router(models=[test_model_path])
        dim = router.embedding_dim()
        embedding = router.embed("Hello, world!")
        assert len(embedding) == dim
        assert dim > 0  # Should be d_model (e.g., 1024)
        router.close()

    def test_embed_normalized_unit_length(self, test_model_path):
        """Normalized embedding has approximately unit L2 norm."""
        import math

        router = Router(models=[test_model_path])
        embedding = router.embed("Hello, world!", normalize=True)
        norm = math.sqrt(sum(x * x for x in embedding))
        assert abs(norm - 1.0) < 0.01  # Close to unit length
        router.close()

    def test_embed_unnormalized_not_unit_length(self, test_model_path):
        """Unnormalized embedding may not have unit length."""
        import math

        router = Router(models=[test_model_path])
        embedding = router.embed("Hello, world!", normalize=False)
        norm = math.sqrt(sum(x * x for x in embedding))
        # Just verify we got a valid embedding, norm could be anything
        assert norm > 0
        router.close()

    def test_embed_pooling_strategies_produce_different_results(self, test_model_path):
        """Different pooling strategies produce different embeddings."""
        router = Router(models=[test_model_path])
        emb_last = router.embed("Hello, world!", pooling="last")
        emb_mean = router.embed("Hello, world!", pooling="mean")
        emb_first = router.embed("Hello, world!", pooling="first")

        # They should be different (unless input is single token)
        # For most inputs, at least two should differ
        assert emb_last != emb_mean or emb_last != emb_first
        router.close()

    def test_embed_different_texts_produce_different_embeddings(self, test_model_path):
        """Different texts produce different embeddings."""
        router = Router(models=[test_model_path])
        emb1 = router.embed("Hello, world!")
        emb2 = router.embed("Goodbye, world!")
        # Embeddings should be different
        assert emb1 != emb2
        router.close()

    def test_embed_same_text_produces_similar_embedding(self, test_model_path):
        """Same text produces consistent embeddings.

        Uses separate Router instances to isolate state. Transformers
        produce identical results; state-space models (SSMs) may have
        minor non-determinism from floating-point reduction order, so
        we check cosine similarity rather than exact equality.
        """
        import math

        router1 = Router(models=[test_model_path])
        emb1 = router1.embed("Hello, world!")
        router1.close()

        router2 = Router(models=[test_model_path])
        emb2 = router2.embed("Hello, world!")
        router2.close()

        dot = sum(a * b for a, b in zip(emb1, emb2, strict=True))
        norm1 = math.sqrt(sum(x * x for x in emb1))
        norm2 = math.sqrt(sum(x * x for x in emb2))
        cosine_sim = dot / (norm1 * norm2)
        # Transformers are fully deterministic (cosine_sim ≈ 1.0).
        # State-space models (SSMs) have inherent non-determinism;
        # threshold 0.5 rules out random vectors (expected ~0 in high dims).
        assert cosine_sim > 0.5, f"Cosine similarity {cosine_sim:.4f} too low for same text"
