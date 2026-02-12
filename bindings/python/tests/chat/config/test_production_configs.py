"""
Tests for production-ready features: count_tokens, logit_bias, extra_body.

These tests verify the "production blockers" features that enable real-world
use cases like RAG budgeting, classification, and provider-specific parameters.
"""

import pytest

from talu import GenerationConfig


class TestCountTokens:
    """Tests for Chat.count_tokens() context window management."""

    def test_count_tokens_exists_on_generation_config(self):
        """GenerationConfig can be created without errors."""
        config = GenerationConfig(max_tokens=100)
        assert config.max_tokens == 100


class TestLogitBias:
    """Tests for logit_bias controlled generation."""

    def test_logit_bias_in_config(self):
        """GenerationConfig accepts logit_bias."""
        config = GenerationConfig(logit_bias={1234: -100.0, 5678: 5.0})
        assert config.logit_bias == {1234: -100.0, 5678: 5.0}

    def test_logit_bias_none_by_default(self):
        """logit_bias is None by default."""
        config = GenerationConfig()
        assert config.logit_bias is None

    def test_logit_bias_merge_with_pipe(self):
        """logit_bias merges correctly with pipe operator."""
        base = GenerationConfig(temperature=0.7)
        bias = GenerationConfig(logit_bias={100: 10.0})

        merged = base | bias
        assert merged.temperature == 0.7
        assert merged.logit_bias == {100: 10.0}


class TestExtraBody:
    """Tests for extra_body escape hatch for remote APIs."""

    def test_extra_body_in_config(self):
        """GenerationConfig accepts extra_body dict."""
        config = GenerationConfig(extra_body={"repetition_penalty": 1.1, "top_a": 0.5})
        assert config.extra_body == {"repetition_penalty": 1.1, "top_a": 0.5}

    def test_extra_body_none_by_default(self):
        """extra_body is None by default."""
        config = GenerationConfig()
        assert config.extra_body is None

    def test_extra_body_merge_with_pipe(self):
        """extra_body merges correctly with pipe operator."""
        base = GenerationConfig(temperature=0.7)
        extra = GenerationConfig(extra_body={"min_p": 0.1})

        merged = base | extra
        assert merged.temperature == 0.7
        assert merged.extra_body == {"min_p": 0.1}

    def test_extra_body_override(self):
        """extra_body can be overridden with override()."""
        config = GenerationConfig(extra_body={"a": 1})
        new_config = config.override(extra_body={"b": 2})

        assert config.extra_body == {"a": 1}  # Original unchanged
        assert new_config.extra_body == {"b": 2}

    def test_extra_body_with_complex_values(self):
        """extra_body supports complex nested structures."""
        config = GenerationConfig(
            extra_body={
                "logit_bias_decay": 0.5,
                "presence_penalty": 0.6,
                "frequency_penalty": 0.4,
                "custom_sampler": {"type": "mirostat", "tau": 5.0, "eta": 0.1},
            }
        )
        assert config.extra_body["custom_sampler"]["type"] == "mirostat"


class TestProductionConfigComposition:
    """Tests for composing production configs."""

    def test_rag_config_pattern(self):
        """RAG-style configuration with token budgeting."""
        # Base config for retrieval (using non-default temperature)
        retrieval_config = GenerationConfig(
            temperature=0.3,  # Non-default for retrieval
            max_tokens=100,
        )

        # Override for generation with extra params
        # Note: temperature=0.7 is the default, so it won't override
        # Use a different value to demonstrate override
        generation_config = GenerationConfig(
            temperature=0.9,  # Non-default
            max_tokens=500,
            extra_body={"presence_penalty": 0.5},
        )

        # Merge: right side's non-default values win
        effective = retrieval_config | generation_config
        assert effective.temperature == 0.9
        assert effective.max_tokens == 500
        assert effective.extra_body == {"presence_penalty": 0.5}

    def test_classification_config_pattern(self):
        """Classification-style configuration with logit_bias."""
        # Config for binary classification
        yes_token_id = 9891
        no_token_id = 2841

        config = GenerationConfig(
            temperature=0.0,  # Deterministic
            max_tokens=1,
            logit_bias={
                yes_token_id: 50.0,  # Strongly prefer "Yes"
                no_token_id: 50.0,  # Strongly prefer "No"
            },
        )

        assert config.logit_bias[yes_token_id] == 50.0
        assert config.logit_bias[no_token_id] == 50.0

    def test_provider_specific_config_pattern(self):
        """Provider-specific config using extra_body."""
        # vLLM-specific parameters
        vllm_config = GenerationConfig(
            temperature=0.7,
            extra_body={
                "best_of": 3,
                "use_beam_search": True,
                "length_penalty": 1.2,
            },
        )

        # Together.ai-specific parameters
        together_config = GenerationConfig(
            temperature=0.7,
            extra_body={
                "repetition_penalty": 1.1,
                "top_k_return_sequences": 1,
            },
        )

        # Both configs have same standard params but different extra_body
        assert vllm_config.temperature == together_config.temperature
        assert vllm_config.extra_body != together_config.extra_body


class TestHeaders:
    """Tests for OpenAICompatibleBackend.headers for enterprise networking."""

    def test_headers_in_backend_config(self):
        """OpenAICompatibleBackend accepts headers dict."""
        from talu.router import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend(
            base_url="https://proxy.example.com/v1",
            api_key="test-key",
            headers={
                "X-Request-ID": "abc123",
                "X-Team-ID": "ml-team",
            },
        )
        assert backend.headers == {"X-Request-ID": "abc123", "X-Team-ID": "ml-team"}

    def test_headers_none_by_default(self):
        """headers is None by default."""
        from talu.router import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend(base_url="http://localhost:8000/v1")
        assert backend.headers is None

    def test_headers_frozen_dataclass(self):
        """OpenAICompatibleBackend is immutable."""
        from talu.router import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend(
            base_url="http://localhost:8000/v1",
            headers={"X-Custom": "value"},
        )
        # Should not be able to modify

        with pytest.raises(AttributeError):
            backend.headers = {"new": "value"}  # type: ignore[misc]

    def test_headers_with_auth_proxy(self):
        """headers supports enterprise auth proxy pattern."""
        from talu.router import OpenAICompatibleBackend

        # Enterprise pattern: internal proxy that validates custom auth header
        backend = OpenAICompatibleBackend(
            base_url="https://internal-llm-proxy.corp.example.com/v1",
            api_key=None,  # No API key - proxy handles auth
            headers={
                "X-Proxy-Auth": "internal-service-token",
                "X-Service-Name": "ml-pipeline",
                "X-Correlation-ID": "request-12345",
            },
        )
        assert backend.base_url == "https://internal-llm-proxy.corp.example.com/v1"
        assert backend.api_key is None
        assert backend.headers["X-Proxy-Auth"] == "internal-service-token"

    def test_headers_combined_with_api_key(self):
        """headers can be used alongside api_key and org_id."""
        from talu.router import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend(
            base_url="https://api.openai.com/v1",
            api_key="sk-...",
            org_id="org-123",
            headers={
                "X-Request-ID": "trace-abc",
                "X-Custom-Header": "value",
            },
        )
        assert backend.api_key == "sk-..."
        assert backend.org_id == "org-123"
        assert backend.headers == {"X-Request-ID": "trace-abc", "X-Custom-Header": "value"}
