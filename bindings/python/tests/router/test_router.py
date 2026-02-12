"""Tests for Router - the Python/Zig boundary.

Router.submit() is where Python hands off to Zig. These tests verify:
1. Router configuration (models, endpoints)
2. Model resolution and validation
3. The expected interface for generate/stream
4. C API bindings for RouterGenerateConfig

Tests that require actual inference are in tests/reference/chat/.
"""

import pytest

import talu
from talu.router import ModelSpec, ModelTarget, OpenAICompatibleBackend, Router
from talu.router._bindings import CLogitBiasEntry, RouterGenerateConfig


class TestRouterConstruction:
    """Tests for Router construction."""

    def test_construct_with_string_list(self):
        """Router accepts list of model strings."""
        router = Router(models=["model-a", "model-b"])
        assert router.models == ["model-a", "model-b"]
        router.close()

    def test_construct_with_model_targets(self):
        """Router accepts list of ModelTarget objects."""
        targets = [
            ModelTarget(model="model-a", endpoint="http://a.local"),
            ModelTarget(model="model-b"),
        ]
        router = Router(models=targets)
        assert router.models == ["model-a", "model-b"]
        assert router.get_endpoint("model-a") == "http://a.local"
        assert router.get_endpoint("model-b") is None
        router.close()

    def test_construct_mixed(self):
        """Router accepts mixed strings and ModelTargets."""
        router = Router(
            models=[
                "model-a",
                ModelTarget(model="model-b", endpoint="http://b.local"),
            ],
        )
        assert set(router.models) == {"model-a", "model-b"}
        router.close()

    def test_construct_empty_raises(self):
        """Router requires at least one model."""
        with pytest.raises(ValueError, match="At least one model"):
            Router(models=[])

    def test_first_model_is_default(self):
        """First model becomes default."""
        router = Router(models=["first", "second", "third"])
        assert router.default_model == "first"
        router.close()

    def test_explicit_default_model(self):
        """Can specify explicit default model."""
        router = Router(models=["a", "b", "c"], default_model="b")
        assert router.default_model == "b"
        router.close()

    def test_invalid_default_model_raises(self):
        """Invalid default model raises ValueError."""
        with pytest.raises(ValueError, match="not in models"):
            Router(models=["a", "b"], default_model="c")


class TestRouterWithModelSpec:
    """Tests for Router construction with ModelSpec."""

    def test_construct_with_model_spec(self):
        """Router accepts ModelSpec objects."""
        spec = ModelSpec(ref="my-model")
        router = Router(models=[spec])
        assert router.models == ["my-model"]
        router.close()

    def test_construct_with_openai_backend(self):
        """Router accepts ModelSpec with OpenAICompatibleBackend."""
        spec = ModelSpec(
            ref="gpt-4o",
            backend=OpenAICompatibleBackend(api_key="sk-test"),
        )
        router = Router(models=[spec])
        assert router.models == ["gpt-4o"]
        router.close()

    def test_construct_mixed_strings_and_specs(self):
        """Router accepts mixed strings and ModelSpec."""
        router = Router(
            models=[
                "model-a",
                ModelSpec(ref="model-b"),
                ModelSpec(
                    ref="gpt-4o",
                    backend=OpenAICompatibleBackend(api_key="sk-test"),
                ),
            ],
        )
        assert set(router.models) == {"model-a", "model-b", "gpt-4o"}
        router.close()

    def test_model_spec_ref_used_as_model_name(self):
        """ModelSpec.ref is used as the model identifier."""
        spec = ModelSpec(ref="custom-name")
        router = Router(models=[spec])
        assert router.default_model == "custom-name"
        router.close()

    def test_multiple_openai_specs_different_endpoints(self):
        """Router can have multiple OpenAI specs with different base_urls."""
        router = Router(
            models=[
                ModelSpec(
                    ref="openai-model",
                    backend=OpenAICompatibleBackend(api_key="sk-openai"),
                ),
                ModelSpec(
                    ref="local-server",
                    backend=OpenAICompatibleBackend(
                        base_url="http://localhost:8000/v1",
                    ),
                ),
            ],
        )
        assert set(router.models) == {"openai-model", "local-server"}
        router.close()


class TestRouterDefaultModel:
    """Tests for default model property."""

    def test_get_default_model(self):
        """default_model property returns current default."""
        router = Router(models=["model-a", "model-b"])
        assert router.default_model == "model-a"
        router.close()

    def test_set_default_model(self):
        """Can change default model."""
        router = Router(models=["model-a", "model-b"])
        router.default_model = "model-b"
        assert router.default_model == "model-b"
        router.close()

    def test_set_invalid_default_raises(self):
        """Setting invalid default raises ValueError."""
        router = Router(models=["model-a", "model-b"])
        with pytest.raises(ValueError, match="not available"):
            router.default_model = "model-c"
        router.close()


class TestRouterEndpoints:
    """Tests for endpoint configuration."""

    def test_get_endpoint_default(self):
        """get_endpoint returns None for no custom endpoint."""
        router = Router(models=["model-a"])
        assert router.get_endpoint("model-a") is None
        router.close()

    def test_get_endpoint_configured(self):
        """get_endpoint returns custom endpoint when set."""
        router = Router(models=[ModelTarget("model-a", "http://custom")])
        assert router.get_endpoint("model-a") == "http://custom"
        router.close()

    def test_set_endpoint(self):
        """Can set custom endpoint after construction."""
        router = Router(models=["model-a"])
        router.set_endpoint("model-a", "http://custom")
        assert router.get_endpoint("model-a") == "http://custom"
        router.close()

    def test_set_endpoint_invalid_model_raises(self):
        """Setting endpoint for unknown model raises."""
        router = Router(models=["model-a"])
        with pytest.raises(ValueError, match="not available"):
            router.set_endpoint("model-b", "http://custom")
        router.close()

    def test_clear_endpoint(self):
        """Can clear custom endpoint by setting to None."""
        router = Router(models=[ModelTarget("model-a", "http://custom")])
        router.set_endpoint("model-a", None)
        assert router.get_endpoint("model-a") is None
        router.close()


class TestRouterModelResolution:
    """Tests for model resolution logic."""

    def test_resolve_explicit_model(self):
        """_resolve_model returns explicit model when provided."""
        router = Router(models=["model-a", "model-b"])
        assert router._resolve_model("model-b") == "model-b"
        router.close()

    def test_resolve_none_uses_default(self):
        """_resolve_model returns default when None."""
        router = Router(models=["model-a", "model-b"])
        assert router._resolve_model(None) == "model-a"
        router.close()

    def test_resolve_invalid_raises(self):
        """_resolve_model raises for unknown model."""
        router = Router(models=["model-a"])
        with pytest.raises(ValueError, match="not available"):
            router._resolve_model("unknown-model")
        router.close()

    def test_resolve_after_close_raises(self):
        """_resolve_model raises if router is closed."""
        router = Router(models=["model-a"])
        router.close()
        with pytest.raises(talu.StateError, match="closed"):
            router._resolve_model("model-a")


class TestRouterLifecycle:
    """Tests for Router lifecycle."""

    def test_close_is_idempotent(self):
        """close() can be called multiple times."""
        router = Router(models=["model-a"])
        router.close()
        router.close()  # Should not raise

    def test_models_empty_after_close(self):
        """models list is empty after close."""
        router = Router(models=["model-a", "model-b"])
        router.close()
        assert router.models == []

    def test_repr_shows_status(self):
        """repr shows open/closed status."""
        router = Router(models=["model-a"])
        assert "open" in repr(router)
        router.close()
        assert "closed" in repr(router)


class TestModelTarget:
    """Tests for ModelTarget dataclass."""

    def test_model_target_model_only(self):
        """ModelTarget with just model name."""
        target = ModelTarget(model="test-model")
        assert target.model == "test-model"
        assert target.endpoint is None

    def test_model_target_with_endpoint(self):
        """ModelTarget with custom endpoint."""
        target = ModelTarget(model="test-model", endpoint="http://custom")
        assert target.model == "test-model"
        assert target.endpoint == "http://custom"


class TestRouterExternalApiDetection:
    """Tests for external API detection using :: namespace separator."""

    def test_external_api_double_colon(self):
        """Double colon (::) indicates external API."""
        router = Router.__new__(Router)
        assert router._is_external_api("openai::gpt-4o") is True
        assert router._is_external_api("anthropic::claude-3-sonnet") is True
        assert router._is_external_api("bedrock::model-id") is True
        assert router._is_external_api("vllm::Foo/Bar-0B") is True
        assert router._is_external_api("ollama::llama3") is True

    def test_native_backend_not_external(self):
        """native:: is the native backend, NOT external API."""
        router = Router.__new__(Router)
        assert router._is_external_api("native::Foo/Bar-0B") is False
        assert router._is_external_api("native::./my-model") is False

    def test_bare_models_not_external(self):
        """Bare model IDs (implicit native::) are not external API."""
        router = Router.__new__(Router)
        # Bare model IDs
        assert router._is_external_api("Foo/Bar-0B") is False
        assert router._is_external_api("meta-llama/Llama-2-7b") is False
        # Local paths
        assert router._is_external_api("/path/to/model") is False
        assert router._is_external_api("./my-model") is False

    def test_single_colon_not_external(self):
        """Single colon is NOT external API."""
        router = Router.__new__(Router)
        # Single colon is NOT external API
        assert router._is_external_api("openai:gpt-4") is False
        assert router._is_external_api("org/model") is False

    def test_external_api_skips_path_validation(self):
        """External API models skip path validation during Router construction."""
        # This should NOT raise because Router no longer validates paths at construction
        router = Router(models=["openai::gpt-4o"])
        assert "openai::gpt-4o" in router.models
        router.close()

    def test_native_prefix_accepted(self):
        """native:: models are accepted (validation happens at generation time)."""
        # Router no longer validates paths at construction time
        # Path validation happens in Zig when generate/stream is called
        router = Router(models=["native::/nonexistent/path"])
        assert "native::/nonexistent/path" in router.models
        router.close()


class TestRouterGenerateConfig:
    """Tests for RouterGenerateConfig C API bindings."""

    def test_config_defaults(self):
        """RouterGenerateConfig has correct defaults."""
        config = RouterGenerateConfig()
        assert config.max_tokens == 0
        assert config.temperature == -1.0
        assert config.top_k == 0
        assert config.top_p == -1.0
        assert config.min_p == -1.0
        assert config.repetition_penalty == 0.0
        # ctypes pointers are falsy when null, check count instead
        assert config.stop_sequence_count == 0
        assert config.logit_bias_count == 0
        assert config.raw_output == 0

    def test_config_with_values(self):
        """RouterGenerateConfig accepts all parameters."""
        config = RouterGenerateConfig(
            max_tokens=100,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            min_p=0.05,
            repetition_penalty=1.1,
            raw_output=True,
        )
        assert config.max_tokens == 100
        assert abs(config.temperature - 0.7) < 0.001
        assert config.top_k == 50
        assert abs(config.top_p - 0.9) < 0.001
        assert abs(config.min_p - 0.05) < 0.001
        assert abs(config.repetition_penalty - 1.1) < 0.001
        assert config.raw_output == 1


class TestRouterGenerateConfigStopSequences:
    """Tests for stop_sequences in RouterGenerateConfig."""

    def test_stop_sequences_none(self):
        """stop_sequences defaults to empty (count=0)."""
        config = RouterGenerateConfig()
        assert config.stop_sequence_count == 0

    def test_stop_sequences_single(self):
        """stop_sequences with single string."""
        config = RouterGenerateConfig(stop_sequences=["END"])
        assert config.stop_sequence_count == 1
        # The pointer should be valid (not None)
        assert config.stop_sequences is not None

    def test_stop_sequences_multiple(self):
        """stop_sequences with multiple strings."""
        config = RouterGenerateConfig(stop_sequences=["END", "User:", "```"])
        assert config.stop_sequence_count == 3
        assert config.stop_sequences is not None

    def test_stop_sequences_empty_list(self):
        """stop_sequences with empty list."""
        config = RouterGenerateConfig(stop_sequences=[])
        assert config.stop_sequence_count == 0

    def test_stop_sequences_memory_kept_alive(self):
        """stop_sequences keeps memory references alive."""
        config = RouterGenerateConfig(stop_sequences=["test1", "test2"])
        # Internal storage should keep references
        assert hasattr(config, "_stop_sequence_strs")
        assert len(config._stop_sequence_strs) == 2


class TestRouterGenerateConfigLogitBias:
    """Tests for logit_bias in RouterGenerateConfig."""

    def test_logit_bias_none(self):
        """logit_bias defaults to empty (count=0)."""
        config = RouterGenerateConfig()
        assert config.logit_bias_count == 0

    def test_logit_bias_single_entry(self):
        """logit_bias with single entry."""
        config = RouterGenerateConfig(logit_bias={1234: -100.0})
        assert config.logit_bias_count == 1
        assert config.logit_bias is not None

    def test_logit_bias_multiple_entries(self):
        """logit_bias with multiple entries."""
        config = RouterGenerateConfig(logit_bias={100: -100.0, 200: 5.0, 300: 0.0})
        assert config.logit_bias_count == 3
        assert config.logit_bias is not None

    def test_logit_bias_empty_dict(self):
        """logit_bias with empty dict."""
        config = RouterGenerateConfig(logit_bias={})
        assert config.logit_bias_count == 0

    def test_logit_bias_memory_kept_alive(self):
        """logit_bias keeps memory references alive."""
        config = RouterGenerateConfig(logit_bias={100: -100.0, 200: 5.0})
        # Internal storage should keep references
        assert hasattr(config, "_logit_bias_entries")
        assert len(config._logit_bias_entries) == 2

    def test_logit_bias_values_preserved(self):
        """logit_bias values are correctly stored in entries."""
        config = RouterGenerateConfig(logit_bias={1234: -100.0, 5678: 5.5})
        # Check the internal entries have correct values
        entries = {e.token_id: e.bias for e in config._logit_bias_entries}
        assert 1234 in entries
        assert 5678 in entries
        assert abs(entries[1234] - (-100.0)) < 0.001
        assert abs(entries[5678] - 5.5) < 0.001


class TestCLogitBiasEntry:
    """Tests for CLogitBiasEntry struct."""

    def test_create_entry(self):
        """CLogitBiasEntry can be created."""
        entry = CLogitBiasEntry(token_id=1234, bias=-100.0)
        assert entry.token_id == 1234
        assert abs(entry.bias - (-100.0)) < 0.001

    def test_entry_positive_bias(self):
        """CLogitBiasEntry accepts positive bias."""
        entry = CLogitBiasEntry(token_id=42, bias=10.0)
        assert entry.token_id == 42
        assert abs(entry.bias - 10.0) < 0.001

    def test_entry_zero_bias(self):
        """CLogitBiasEntry accepts zero bias."""
        entry = CLogitBiasEntry(token_id=99, bias=0.0)
        assert entry.token_id == 99
        assert entry.bias == 0.0


# =============================================================================
# Embedding Tests
# =============================================================================


class TestRouterEmbedAPI:
    """Tests for Router.embed() API."""

    def test_embed_invalid_pooling(self):
        """embed() raises ValueError for invalid pooling strategy."""
        router = Router(models=["test-model"])
        with pytest.raises(ValueError, match="Invalid pooling strategy"):
            router.embed("Hello", pooling="invalid")
        router.close()

    def test_embed_valid_pooling_values(self):
        """embed() accepts all valid pooling strategies."""
        router = Router(models=["test-model"])
        # These should not raise ValueError (they'll fail on model load)
        for pooling in ["last", "mean", "first"]:
            with pytest.raises(talu.GenerationError):
                # Will fail because model doesn't exist, but pooling is valid
                router.embed("Hello", pooling=pooling)
        router.close()


class TestRouterEmbeddingDim:
    """Tests for Router.embedding_dim() API."""

    def test_embedding_dim_nonexistent_model(self):
        """embedding_dim() raises GenerationError for non-existent model."""
        from talu.exceptions import GenerationError

        router = Router(models=["nonexistent-model"])
        # Should raise since model can't be loaded
        with pytest.raises(GenerationError) as exc_info:
            router.embedding_dim()
        assert "nonexistent-model" in str(exc_info.value)
        router.close()


# =============================================================================
# Router Error Handling Tests
# =============================================================================


class TestRouterErrorHandling:
    """Tests for Router error handling and fallback paths."""

    def test_get_endpoint_unknown_model(self):
        """get_endpoint returns None for unknown model (no error)."""
        router = Router(models=["model-a"])
        # get_endpoint with unknown model returns None (doesn't raise)
        endpoint = router.get_endpoint("unknown-model")
        assert endpoint is None
        router.close()

    def test_models_property_safe_after_close(self):
        """models property returns empty list after close."""
        router = Router(models=["model-a", "model-b"])
        router.close()
        assert router.models == []

    def test_default_model_persists_after_close(self):
        """default_model property preserves last value after close."""
        router = Router(models=["model-a"])
        router.close()
        # The default model string is preserved even after close
        assert router.default_model == "model-a"

    def test_generate_after_close_raises(self):
        """generate() after close raises StateError."""
        from talu import Chat

        router = Router(models=["test-model"])
        chat = Chat()
        router.close()

        with pytest.raises(talu.StateError, match="closed"):
            router.generate(chat, "Hello")

    def test_stream_after_close_raises(self):
        """stream() after close raises StateError."""
        from talu import Chat

        router = Router(models=["test-model"])
        chat = Chat()
        router.close()

        with pytest.raises(talu.StateError, match="closed"):
            list(router.stream(chat, "Hello"))

    def test_submit_chat_none_raises(self):
        """submit() with None chat raises ValidationError."""
        router = Router(models=["test-model"])

        with pytest.raises(talu.ValidationError, match="Chat instance"):
            router.submit(None, "Hello")

        router.close()

    def test_generate_model_not_found_raises(self):
        """generate() with unknown model raises ValidationError."""
        from talu import Chat

        router = Router(models=["model-a"])
        chat = Chat()

        with pytest.raises(talu.ValidationError, match="not available"):
            router.generate(chat, "Hello", model="unknown-model")

        router.close()


class TestRouterResourceCleanup:
    """Tests for Router resource cleanup and lifecycle."""

    def test_repeated_close_safe(self):
        """close() can be called multiple times safely."""
        router = Router(models=["model-a"])
        for _ in range(5):
            router.close()
        # Should not raise

    def test_del_without_close(self):
        """Router can be garbage collected without explicit close."""
        import gc

        router = Router(models=["model-a"])

        del router
        gc.collect()
        # Should not crash

    def test_router_creation_with_duplicate_models(self):
        """Router handles duplicate model names (last wins)."""
        router = Router(
            models=[
                ModelTarget(model="model-a", endpoint="http://first"),
                ModelTarget(model="model-a", endpoint="http://second"),
            ]
        )

        # Last endpoint wins for duplicate model name
        assert router.get_endpoint("model-a") == "http://second"
        router.close()

    def test_router_repr_after_close(self):
        """Router repr shows closed status."""
        router = Router(models=["model-a"])
        assert "open" in repr(router)
        router.close()
        assert "closed" in repr(router)


# =============================================================================
# Content Parts Building Tests
# =============================================================================


class TestBuildContentParts:
    """Tests for build_router_content_parts() Open Responses format handling."""

    def test_input_text_format(self):
        """build_router_content_parts handles input_text (Open Responses format)."""
        from talu.router._bindings import build_router_content_parts

        parts = [{"type": "input_text", "text": "Hello, world!"}]
        c_parts, data_refs = build_router_content_parts(parts)

        assert len(c_parts) == 1
        assert c_parts[0].content_type == 0  # input_text
        assert b"Hello, world!" in data_refs

    def test_legacy_text_format(self):
        """build_router_content_parts handles legacy text format."""
        from talu.router._bindings import build_router_content_parts

        parts = [{"type": "text", "text": "Hello, world!"}]
        c_parts, data_refs = build_router_content_parts(parts)

        assert len(c_parts) == 1
        assert c_parts[0].content_type == 0  # text maps to 0
        assert b"Hello, world!" in data_refs

    def test_input_image_format(self):
        """build_router_content_parts handles input_image with image_url (Open Responses)."""
        from talu.router._bindings import build_router_content_parts

        data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAE="
        parts = [{"type": "input_image", "image_url": data_uri, "detail": "auto"}]
        c_parts, data_refs = build_router_content_parts(parts)

        assert len(c_parts) == 1
        assert c_parts[0].content_type == 1  # input_image
        assert data_uri.encode("utf-8") in data_refs

    def test_legacy_image_format(self):
        """build_router_content_parts handles legacy image format with data key."""
        from talu.router._bindings import build_router_content_parts

        raw_data = "base64encodeddata"
        parts = [{"type": "image", "data": raw_data, "mime": "image/png"}]
        c_parts, data_refs = build_router_content_parts(parts)

        assert len(c_parts) == 1
        assert c_parts[0].content_type == 1  # image maps to 1
        assert raw_data.encode("utf-8") in data_refs

    def test_input_audio_format(self):
        """build_router_content_parts handles input_audio with audio_data (Open Responses)."""
        from talu.router._bindings import build_router_content_parts

        data_uri = "data:audio/wav;base64,UklGRiQAAABXQVZF="
        parts = [{"type": "input_audio", "audio_data": data_uri}]
        c_parts, data_refs = build_router_content_parts(parts)

        assert len(c_parts) == 1
        assert c_parts[0].content_type == 2  # input_audio
        assert data_uri.encode("utf-8") in data_refs

    def test_legacy_audio_format(self):
        """build_router_content_parts handles legacy audio format with data key."""
        from talu.router._bindings import build_router_content_parts

        raw_data = "base64encodedaudio"
        parts = [{"type": "audio", "data": raw_data}]
        c_parts, data_refs = build_router_content_parts(parts)

        assert len(c_parts) == 1
        assert c_parts[0].content_type == 2  # audio maps to 2
        assert raw_data.encode("utf-8") in data_refs

    def test_input_video_format(self):
        """build_router_content_parts handles input_video with video_url (Open Responses)."""
        from talu.router._bindings import build_router_content_parts

        data_uri = "data:video/mp4;base64,AAAAIGZ0eXBpc29t="
        parts = [{"type": "input_video", "video_url": data_uri}]
        c_parts, data_refs = build_router_content_parts(parts)

        assert len(c_parts) == 1
        assert c_parts[0].content_type == 3  # input_video
        assert data_uri.encode("utf-8") in data_refs

    def test_legacy_video_format(self):
        """build_router_content_parts handles legacy video format with data key."""
        from talu.router._bindings import build_router_content_parts

        raw_data = "base64encodedvideo"
        parts = [{"type": "video", "data": raw_data}]
        c_parts, data_refs = build_router_content_parts(parts)

        assert len(c_parts) == 1
        assert c_parts[0].content_type == 3  # video maps to 3
        assert raw_data.encode("utf-8") in data_refs

    def test_mixed_content_types(self):
        """build_router_content_parts handles mixed content types."""
        from talu.router._bindings import build_router_content_parts

        parts = [
            {"type": "input_text", "text": "Describe this image:"},
            {"type": "input_image", "image_url": "data:image/png;base64,abc123"},
        ]
        c_parts, data_refs = build_router_content_parts(parts)

        assert len(c_parts) == 2
        assert c_parts[0].content_type == 0  # input_text
        assert c_parts[1].content_type == 1  # input_image

    def test_mime_type_passthrough(self):
        """build_router_content_parts passes through explicit MIME type."""
        from talu.router._bindings import build_router_content_parts

        parts = [
            {"type": "input_image", "image_url": "data:image/jpeg;base64,abc", "mime": "image/jpeg"}
        ]
        c_parts, data_refs = build_router_content_parts(parts)

        assert len(c_parts) == 1
        # MIME bytes should be in data_refs
        assert b"image/jpeg" in data_refs

    def test_unknown_type_defaults_to_text(self):
        """build_router_content_parts defaults unknown types to text (0)."""
        from talu.router._bindings import build_router_content_parts

        parts = [{"type": "unknown_type", "text": "fallback text"}]
        c_parts, data_refs = build_router_content_parts(parts)

        assert len(c_parts) == 1
        assert c_parts[0].content_type == 0  # defaults to text

    def test_empty_parts_list(self):
        """build_router_content_parts handles empty parts list."""
        from talu.router._bindings import build_router_content_parts

        parts = []
        c_parts, data_refs = build_router_content_parts(parts)

        assert len(c_parts) == 0
        assert len(data_refs) == 0
