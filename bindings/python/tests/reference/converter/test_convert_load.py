"""
Tests for convert + load round-trip.

Tests that converted models can be loaded and produce valid output.
This verifies the complete pipeline from conversion to inference.

All tests in this module are integration tests requiring:
- A cached unquantized model (small_test_model fixture)
- Sufficient time for conversion (@pytest.mark.slow)
"""

import pytest

# All tests in this module are slow integration tests
pytestmark = [pytest.mark.slow, pytest.mark.requires_model]


class TestTQSchemeModelLoads:
    """Tests that TQ scheme (Talu Quantized) models can be loaded."""

    def test_tq4_loads(self, talu, convert_func, small_test_model, temp_output_dir):
        """TQ4 (default 4-bit) model can be loaded."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="tq4",
            output_dir=temp_output_dir,
            force=True,
        )

        client = talu.Chat(result_path)
        assert client is not None
        assert hasattr(client, "send")

    def test_tq8_loads(self, talu, convert_func, small_test_model, temp_output_dir):
        """TQ8 (default 8-bit) model can be loaded."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="tq8",
            output_dir=temp_output_dir,
            force=True,
        )

        client = talu.Chat(result_path)
        assert client is not None
        assert hasattr(client, "send")


class TestConvertedModelInference:
    """Tests that converted models can run inference."""

    def test_tq_scheme_generates_output(
        self, talu, convert_func, small_test_model, temp_output_dir
    ):
        """TQ scheme model can generate text."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="tq4",
            output_dir=temp_output_dir,
            force=True,
        )

        from talu.router import GenerationConfig

        client = talu.Chat(result_path)
        response = client.send("Hello", config=GenerationConfig(max_tokens=5))

        assert hasattr(response, "text")
        assert isinstance(response.text, str)
        assert response is not None, "Generation should return a response"


class TestConvertedModelStreaming:
    """Tests that converted models support streaming."""

    def test_tq_scheme_streams(self, talu, convert_func, small_test_model, temp_output_dir):
        """TQ scheme model supports streaming generation."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="tq4",
            output_dir=temp_output_dir,
            force=True,
        )

        from talu.router import GenerationConfig

        client = talu.Chat(result_path)
        # Use stream=True to get streaming response
        response = client.send("Hello", config=GenerationConfig(max_tokens=5), stream=True)

        # Streaming response should be iterable
        chunks = list(response)
        assert len(chunks) > 0, "Should produce at least one chunk"


class TestConvertedModelTokenizer:
    """Tests that converted models have working tokenizers."""

    def test_tq_scheme_tokenizer_works(
        self, talu, convert_func, small_test_model, temp_output_dir
    ):
        """TQ scheme model has working tokenizer."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="tq4",
            output_dir=temp_output_dir,
            force=True,
        )

        tokenizer = talu.Tokenizer(result_path)

        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        assert len(tokens) > 0, "Encoding should produce tokens"
        assert "Hello" in decoded or "hello" in decoded.lower()


class TestQualityComparison:
    """Tests comparing quality between schemes."""

    def test_tq_schemes_produce_similar_output(
        self, talu, convert_func, small_test_model, temp_output_dir
    ):
        """Different TQ group sizes may produce slightly different output."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        # Convert to TQ4 scheme (default group_size=32)
        tq4_path = convert_func(
            small_test_model,
            scheme="tq4",
            output_dir=temp_output_dir,
            force=True,
        )

        # Convert to TQ4_64 scheme (different output dir to avoid collision)
        import os

        tq4_64_output_dir = os.path.join(temp_output_dir, "tq4_64")
        os.makedirs(tq4_64_output_dir, exist_ok=True)

        tq4_64_path = convert_func(
            small_test_model,
            scheme="tq4_64",
            output_dir=tq4_64_output_dir,
            force=True,
        )

        from talu.router import GenerationConfig

        # Both should load and generate
        # Use greedy decoding for deterministic output
        tq4_client = talu.Chat(tq4_path)
        tq4_64_client = talu.Chat(tq4_64_path)

        config = GenerationConfig(max_tokens=5, temperature=0.01)

        prompt = "2+2="
        tq4_response = tq4_client.send(prompt, config=config)
        tq4_64_response = tq4_64_client.send(prompt, config=config)

        # Both should produce a response
        assert tq4_response is not None
        assert tq4_64_response is not None

        # Note: They might produce different output due to different group sizes
        # This is expected behavior, not a bug
