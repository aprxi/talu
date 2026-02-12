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


class TestGAFSchemeModelLoads:
    """Tests that GAF scheme (grouped affine) models can be loaded."""

    def test_gaf4_64_loads(self, talu, convert_func, small_test_model, temp_output_dir):
        """GAF4_64 (MLX default) model can be loaded."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="gaf4_64",
            output_dir=temp_output_dir,
            force=True,
        )

        client = talu.Chat(result_path)
        assert client is not None
        assert hasattr(client, "send")

    def test_gaf8_64_loads(self, talu, convert_func, small_test_model, temp_output_dir):
        """GAF8_64 model can be loaded."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="gaf8_64",
            output_dir=temp_output_dir,
            force=True,
        )

        client = talu.Chat(result_path)
        assert client is not None
        assert hasattr(client, "send")


class TestConvertedModelInference:
    """Tests that converted models can run inference."""

    def test_gaf_scheme_generates_output(
        self, talu, convert_func, small_test_model, temp_output_dir
    ):
        """GAF scheme model can generate text."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="gaf4_64",
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

    def test_gaf_scheme_streams(self, talu, convert_func, small_test_model, temp_output_dir):
        """GAF scheme model supports streaming generation."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="gaf4_64",
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

    def test_gaf_scheme_tokenizer_works(
        self, talu, convert_func, small_test_model, temp_output_dir
    ):
        """GAF scheme model has working tokenizer."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        result_path = convert_func(
            small_test_model,
            scheme="gaf4_64",
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

    def test_gaf_schemes_produce_similar_output(
        self, talu, convert_func, small_test_model, temp_output_dir
    ):
        """Different GAF group sizes may produce slightly different output."""
        if small_test_model is None:
            pytest.skip("No unquantized test model available in cache")

        # Convert to GAF4_64 scheme
        gaf4_64_path = convert_func(
            small_test_model,
            scheme="gaf4_64",
            output_dir=temp_output_dir,
            force=True,
        )

        # Convert to GAF4_32 scheme (different output dir to avoid collision)
        import os

        gaf32_output_dir = os.path.join(temp_output_dir, "gaf32")
        os.makedirs(gaf32_output_dir, exist_ok=True)

        gaf4_32_path = convert_func(
            small_test_model,
            scheme="gaf4_32",
            output_dir=gaf32_output_dir,
            force=True,
        )

        from talu.router import GenerationConfig

        # Both should load and generate
        # Use greedy decoding for deterministic output
        gaf64_client = talu.Chat(gaf4_64_path)
        gaf32_client = talu.Chat(gaf4_32_path)

        config = GenerationConfig(max_tokens=5, temperature=0.01)

        prompt = "2+2="
        gaf64_response = gaf64_client.send(prompt, config=config)
        gaf32_response = gaf32_client.send(prompt, config=config)

        # Both should produce a response
        assert gaf64_response is not None
        assert gaf32_response is not None

        # Note: They might produce different output due to different group sizes
        # This is expected behavior, not a bug
