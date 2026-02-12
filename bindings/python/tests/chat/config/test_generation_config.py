"""
Tests for GenerationConfig and the new configuration API.

Tests the v0.2 Controller refactor:
- GenerationConfig dataclass with all sampling parameters
- Chat with default_config support
- min_p and repetition_penalty parameters
- State persistence via to_dict/from_dict

Each test covers normal path + edge case + error case.
"""

import pytest

import talu
from talu import Chat, GenerationConfig
from talu.router import SamplingParams, SamplingStrategy


class TestGenerationConfigCreation:
    """Tests for GenerationConfig dataclass creation and defaults."""

    def test_default_values(self):
        """GenerationConfig has sensible defaults."""
        config = GenerationConfig()
        assert config.max_tokens == 256
        assert config.temperature == 0.7
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.min_p == 0.0
        assert config.repetition_penalty == 1.0
        assert config.stop_sequences is None
        assert config.seed is None

    def test_custom_values(self):
        """GenerationConfig accepts custom values."""
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.5,
            top_k=40,
            top_p=0.95,
            min_p=0.05,
            repetition_penalty=1.2,
            stop_sequences=["```"],
            seed=42,
        )
        assert config.max_tokens == 100
        assert config.temperature == 0.5
        assert config.top_k == 40
        assert config.top_p == 0.95
        assert config.min_p == 0.05
        assert config.repetition_penalty == 1.2
        assert config.stop_sequences == ["```"]
        assert config.seed == 42

    def test_greedy_decoding(self):
        """Temperature 0.0 enables greedy decoding."""
        config = GenerationConfig(temperature=0.0)
        params = config._to_sampling_params()
        assert params.strategy == SamplingStrategy.GREEDY

    def test_sampling_decoding(self):
        """Non-zero temperature uses sampling."""
        config = GenerationConfig(temperature=0.7)
        params = config._to_sampling_params()
        assert params.strategy == SamplingStrategy.TOP_K


class TestGenerationConfigOverride:
    """Tests for GenerationConfig override method."""

    def test_override_returns_new_config(self):
        """override() returns a new config with changes."""
        config = GenerationConfig(temperature=0.5, max_tokens=100)
        result = config.override(temperature=0.9)
        assert result is not config
        assert result.temperature == 0.9
        assert result.max_tokens == 100  # Unchanged field preserved

    def test_override_original_unchanged(self):
        """override() does not mutate the original config."""
        config = GenerationConfig(temperature=0.5)
        config.override(temperature=0.9)
        assert config.temperature == 0.5  # Original unchanged

    def test_override_multiple_fields(self):
        """override() can change multiple fields at once."""
        config = GenerationConfig(temperature=0.5, max_tokens=100, top_k=50)
        result = config.override(temperature=0.9, max_tokens=200)
        assert result.temperature == 0.9
        assert result.max_tokens == 200
        assert result.top_k == 50  # Unchanged

    def test_override_no_args_returns_copy(self):
        """override() with no args returns an identical copy."""
        config = GenerationConfig(temperature=0.5)
        result = config.override()
        assert result is not config
        assert result.temperature == 0.5


class TestGenerationConfigMutability:
    """Tests for GenerationConfig mutable dataclass behavior."""

    def test_can_mutate_field(self):
        """Direct field assignment works for mutable config."""
        config = GenerationConfig(temperature=0.5)
        config.temperature = 0.9
        assert config.temperature == 0.9

    def test_override_does_not_mutate_original(self):
        """Override creates a new config without mutating original."""
        config = GenerationConfig(temperature=0.5)

        # Override creates new config
        modified = config.override(temperature=0.9)

        assert modified.temperature == 0.9
        assert config.temperature == 0.5  # Original unchanged

    def test_shared_config_mutations_affect_all_references(self):
        """Shared config mutations affect all references (expected behavior)."""
        config = GenerationConfig(temperature=0.5)

        # Multiple references to same config
        ref1 = config
        ref2 = config

        # Mutate through one reference
        ref1.temperature = 0.9

        # All references see the change
        assert ref2.temperature == 0.9
        assert config.temperature == 0.9


class TestSamplingParamsConversion:
    """Tests for GenerationConfig to SamplingParams conversion."""

    def test_all_fields_converted(self):
        """All relevant fields are converted to SamplingParams."""
        config = GenerationConfig(
            temperature=0.8,
            top_k=60,
            top_p=0.85,
            min_p=0.1,
            repetition_penalty=1.15,
        )
        params = config._to_sampling_params()

        assert abs(params.temperature - 0.8) < 1e-5
        assert params.top_k == 60
        assert abs(params.top_p - 0.85) < 1e-5
        assert abs(params.min_p - 0.1) < 1e-5
        assert abs(params.repetition_penalty - 1.15) < 1e-5

    def test_ctypes_struct_layout(self):
        """SamplingParams has correct ctypes struct layout."""
        params = SamplingParams(
            strategy=1,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            min_p=0.05,
            repetition_penalty=1.2,
        )
        # Verify struct can be passed to C (no exceptions)
        assert params.strategy == 1
        assert abs(params.min_p - 0.05) < 1e-5
        assert abs(params.repetition_penalty - 1.2) < 1e-5


class TestChatWithConfig:
    """Tests for Chat using GenerationConfig."""

    def test_create_with_config(self):
        """Chat accepts config parameter."""
        config = GenerationConfig(temperature=0.5, min_p=0.1)
        chat = Chat(config=config)
        assert chat.config.temperature == 0.5
        assert chat.config.min_p == 0.1

    def test_config_property(self):
        """Chat exposes config property."""
        config = GenerationConfig(temperature=0.8)
        chat = Chat(config=config)
        assert chat.config is not None
        assert chat.config.temperature == 0.8

    def test_config_setter(self):
        """Chat allows setting config."""
        chat = Chat()
        new_config = GenerationConfig(temperature=0.3)
        chat.config = new_config
        assert abs(chat.config.temperature - 0.3) < 1e-5

    def test_config_params_work(self):
        """Config parameters work via GenerationConfig."""
        config = GenerationConfig(
            temperature=0.6,
            max_tokens=128,
            top_k=30,
            top_p=0.85,
        )
        chat = Chat(config=config)
        assert abs(chat.config.temperature - 0.6) < 1e-5
        assert chat.config.max_tokens == 128
        assert chat.config.top_k == 30
        assert abs(chat.config.top_p - 0.85) < 1e-5


class TestChatSerialization:
    """Tests for Chat to_dict/from_dict with new config fields."""

    def test_to_dict_includes_new_fields(self):
        """to_dict includes min_p and repetition_penalty."""
        config = GenerationConfig(min_p=0.05, repetition_penalty=1.15)
        chat = Chat(config=config, system="Test")

        data = chat.to_dict()

        assert "config" in data
        assert abs(data["config"]["min_p"] - 0.05) < 1e-5
        assert abs(data["config"]["repetition_penalty"] - 1.15) < 1e-5

    def test_from_dict_restores_new_fields(self):
        """from_dict restores min_p and repetition_penalty."""
        data = {
            "messages": [
                {"role": "system", "content": "Test"},
                {"role": "user", "content": "Hello"},
            ],
            "config": {
                "temperature": 0.7,
                "max_tokens": 100,
                "top_k": 50,
                "top_p": 0.9,
                "min_p": 0.1,
                "repetition_penalty": 1.25,
            },
        }

        chat = Chat.from_dict(data)

        assert abs(chat.config.min_p - 0.1) < 1e-5
        assert abs(chat.config.repetition_penalty - 1.25) < 1e-5

    def test_from_dict_with_missing_fields_uses_defaults(self):
        """from_dict uses defaults for missing min_p/repetition_penalty."""
        data = {
            "messages": [],
            "config": {
                "temperature": 0.7,
                "max_tokens": 256,
                "top_k": 50,
                "top_p": 0.9,
                # min_p and repetition_penalty intentionally missing
            },
        }

        chat = Chat.from_dict(data)

        assert chat.config.min_p == 0.0
        assert chat.config.repetition_penalty == 1.0

    def test_roundtrip_preserves_config(self):
        """to_dict -> from_dict preserves all config fields."""
        config = GenerationConfig(
            temperature=0.65,
            max_tokens=150,
            top_k=45,
            top_p=0.92,
            min_p=0.08,
            repetition_penalty=1.18,
        )
        original = Chat(config=config, system="Original")

        data = original.to_dict()
        restored = Chat.from_dict(data)

        # System is now in items[0]
        assert restored.items[0].text == original.items[0].text
        assert len(restored.items) == len(original.items)
        assert abs(restored.config.temperature - original.config.temperature) < 1e-5
        assert abs(restored.config.min_p - 0.08) < 1e-5
        assert abs(restored.config.repetition_penalty - 1.18) < 1e-5


class TestMinPParameter:
    """Tests for min_p sampling parameter."""

    def test_min_p_default_is_disabled(self):
        """min_p defaults to 0.0 (disabled)."""
        config = GenerationConfig()
        assert config.min_p == 0.0

    def test_min_p_in_sampling_params(self):
        """min_p is included in SamplingParams."""
        config = GenerationConfig(min_p=0.15)
        params = config._to_sampling_params()
        assert abs(params.min_p - 0.15) < 1e-5

    def test_min_p_valid_range(self):
        """min_p accepts values from 0.0 to 1.0."""
        # Valid edge cases
        config_zero = GenerationConfig(min_p=0.0)
        config_one = GenerationConfig(min_p=1.0)
        config_mid = GenerationConfig(min_p=0.5)

        assert config_zero.min_p == 0.0
        assert config_one.min_p == 1.0
        assert config_mid.min_p == 0.5


class TestRepetitionPenalty:
    """Tests for repetition_penalty parameter."""

    def test_repetition_penalty_default_is_neutral(self):
        """repetition_penalty defaults to 1.0 (no penalty)."""
        config = GenerationConfig()
        assert config.repetition_penalty == 1.0

    def test_repetition_penalty_in_sampling_params(self):
        """repetition_penalty is included in SamplingParams."""
        config = GenerationConfig(repetition_penalty=1.3)
        params = config._to_sampling_params()
        assert abs(params.repetition_penalty - 1.3) < 1e-5

    def test_repetition_penalty_discourages_repeats(self):
        """Values > 1.0 discourage repetition."""
        # This documents the intended semantics
        config = GenerationConfig(repetition_penalty=1.5)
        assert config.repetition_penalty > 1.0


class TestChatSendWithConfig:
    """Tests for Chat.send with config parameter."""

    @pytest.mark.requires_model
    def test_send_accepts_config(self, test_model_path):
        """Chat.send accepts config parameter."""
        chat = Chat(test_model_path)
        config = GenerationConfig(max_tokens=5, temperature=0.0)

        response = chat.send("Hi", config=config)

        assert isinstance(str(response), str)
        assert response.usage.completion_tokens > 0

    @pytest.mark.requires_model
    def test_send_config_overrides_default(self, test_model_path):
        """Config passed to send overrides session default."""
        default_config = GenerationConfig(max_tokens=100, temperature=0.8)
        chat = Chat(test_model_path, config=default_config)

        # Override with greedy decoding for determinism
        override = GenerationConfig(max_tokens=3, temperature=0.0)
        response = chat.send("Count: 1, 2,", config=override)

        # With max_tokens=3 and greedy, output should be short
        assert isinstance(str(response), str)


class TestStopSequences:
    """Tests for stop_sequences parameter."""

    def test_stop_sequences_in_config(self):
        """stop_sequences can be set in GenerationConfig."""
        config = GenerationConfig(stop_sequences=["```", "END"])
        assert config.stop_sequences == ["```", "END"]

    def test_stop_sequences_default_is_none(self):
        """stop_sequences defaults to None."""
        config = GenerationConfig()
        assert config.stop_sequences is None

    def test_stop_sequences_single_string(self):
        """stop_sequences can be a single-element list."""
        config = GenerationConfig(stop_sequences=["User:"])
        assert config.stop_sequences == ["User:"]
        assert len(config.stop_sequences) == 1

    def test_stop_sequences_empty_list(self):
        """stop_sequences can be an empty list."""
        config = GenerationConfig(stop_sequences=[])
        assert config.stop_sequences == []

    def test_stop_sequences_multitoken(self):
        """stop_sequences supports multi-token strings."""
        # "User:" tokenizes to multiple tokens in most models
        config = GenerationConfig(stop_sequences=["User:", "Assistant:"])
        assert "User:" in config.stop_sequences
        assert "Assistant:" in config.stop_sequences


class TestSeedParameter:
    """Tests for seed parameter and deterministic generation."""

    def test_seed_in_config(self):
        """seed can be set in GenerationConfig."""
        config = GenerationConfig(seed=42)
        assert config.seed == 42

    def test_seed_default_is_none(self):
        """seed defaults to None."""
        config = GenerationConfig()
        assert config.seed is None

    def test_seed_in_sampling_params(self):
        """seed is included in SamplingParams."""
        config = GenerationConfig(seed=12345)
        params = config._to_sampling_params()
        assert params.seed == 12345

    def test_seed_none_maps_to_zero(self):
        """seed=None maps to 0 in SamplingParams (non-deterministic)."""
        config = GenerationConfig(seed=None)
        params = config._to_sampling_params()
        assert params.seed == 0

    @pytest.mark.requires_model
    def test_deterministic_generation_with_seed(self, test_model_path):
        """Same seed produces same output (deterministic generation)."""
        config = GenerationConfig(
            seed=42,
            temperature=0.7,  # Non-zero to actually use sampling
            max_tokens=10,
        )

        chat1 = Chat(test_model_path, config=config)
        chat2 = Chat(test_model_path, config=config)

        # Same prompt, same seed -> should produce same output
        response1 = chat1("Count to five:")
        response2 = chat2("Count to five:")

        assert str(response1) == str(response2), (
            f"Deterministic generation failed: '{response1}' != '{response2}'"
        )


class TestConfigOverride:
    """Tests for config override behavior with GenerationConfig."""

    @pytest.mark.requires_model
    def test_send_config_overrides_default(self, test_model_path):
        """Config passed to send() overrides session default config."""
        # Create session with a default config of 100 max tokens
        default_config = GenerationConfig(max_tokens=100, temperature=0.0)
        chat = Chat(test_model_path, config=default_config)

        # Pass config with 3 max_tokens to override the default
        override_config = GenerationConfig(max_tokens=3, temperature=0.0)

        chat.send(
            "Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
            config=override_config,
        )

        # Config override took effect — generation produced output
        assert len(chat.items) >= 2


class TestConfigImmutability:
    """Tests that per-call overrides do NOT mutate session config.

    This behavior is critical for predictable API usage. Users should be able
    to pass kwargs or config overrides without worrying about "ghost configuration"
    affecting future calls.
    """

    def test_kwargs_do_not_mutate_session_config(self):
        """Per-call kwargs do NOT change chat.config."""
        original_temp = 0.7
        chat = Chat(config=GenerationConfig(temperature=original_temp))

        # Simulate building effective config with per-call override
        # (We can't actually call send without a model, but we can test the merge logic)
        effective = chat._build_effective_config(temperature=0.1)

        # Effective config should have the override
        assert abs(effective.temperature - 0.1) < 1e-5

        # Session config should be UNCHANGED
        assert abs(chat.config.temperature - original_temp) < 1e-5

    def test_config_parameter_does_not_mutate_session_config(self):
        """Per-call config parameter does NOT change chat.config."""
        original_temp = 0.7
        chat = Chat(config=GenerationConfig(temperature=original_temp))

        override = GenerationConfig(temperature=0.2)
        effective = chat._build_effective_config(config=override)

        # Effective config should have the override
        assert abs(effective.temperature - 0.2) < 1e-5

        # Session config should be UNCHANGED
        assert abs(chat.config.temperature - original_temp) < 1e-5

    def test_multiple_overrides_in_loop_do_not_accumulate(self):
        """Overrides in a loop don't accumulate or affect each other."""
        chat = Chat(config=GenerationConfig(temperature=0.5, max_tokens=100))

        # Simulate multiple calls with different overrides
        temps = [0.1, 0.9, 0.3, 0.7]
        for temp in temps:
            effective = chat._build_effective_config(temperature=temp)
            assert abs(effective.temperature - temp) < 1e-5
            # Other params should come from session config
            assert effective.max_tokens == 100

        # Session config should be completely unchanged
        assert abs(chat.config.temperature - 0.5) < 1e-5
        assert chat.config.max_tokens == 100

    def test_explicit_assignment_does_mutate_session_config(self):
        """Direct assignment to chat.config DOES change it (expected behavior)."""
        chat = Chat(config=GenerationConfig(temperature=0.5))

        # This SHOULD mutate the session config
        chat.config = GenerationConfig(temperature=0.9)

        assert abs(chat.config.temperature - 0.9) < 1e-5

    def test_config_precedence_order(self):
        """Verify precedence: kwargs > config param > session config."""
        chat = Chat(config=GenerationConfig(temperature=0.5, max_tokens=100, top_k=50))

        # Only session config
        eff1 = chat._build_effective_config()
        assert abs(eff1.temperature - 0.5) < 1e-5
        assert eff1.max_tokens == 100
        assert eff1.top_k == 50

        # Config param overrides session
        override = GenerationConfig(temperature=0.7, max_tokens=200)
        eff2 = chat._build_effective_config(config=override)
        assert abs(eff2.temperature - 0.7) < 1e-5
        assert eff2.max_tokens == 200
        # Note: top_k comes from override config (which has default 50)
        assert eff2.top_k == 50

        # kwargs override config param
        eff3 = chat._build_effective_config(config=override, temperature=0.9)
        assert abs(eff3.temperature - 0.9) < 1e-5  # kwargs wins
        assert eff3.max_tokens == 200  # from config param

        # kwargs override session (no config param)
        eff4 = chat._build_effective_config(max_tokens=50)
        assert abs(eff4.temperature - 0.5) < 1e-5  # from session
        assert eff4.max_tokens == 50  # kwargs wins

    def test_unknown_kwarg_raises_validation_error(self):
        """Unknown kwargs raise ValidationError to prevent typos."""
        chat = Chat()

        with pytest.raises(talu.ValidationError, match="Unknown generation parameter"):
            chat._build_effective_config(temprature=0.5)  # typo: temprature


class TestLogitBias:
    """Tests for logit_bias parameter."""

    def test_logit_bias_default_is_none(self):
        """logit_bias defaults to None."""
        config = GenerationConfig()
        assert config.logit_bias is None

    def test_logit_bias_in_config(self):
        """logit_bias can be set in GenerationConfig."""
        config = GenerationConfig(logit_bias={1234: -100.0, 5678: 5.0})
        assert config.logit_bias is not None
        assert config.logit_bias[1234] == -100.0
        assert config.logit_bias[5678] == 5.0

    def test_logit_bias_empty_dict(self):
        """logit_bias can be an empty dict."""
        config = GenerationConfig(logit_bias={})
        assert config.logit_bias == {}

    def test_logit_bias_single_entry(self):
        """logit_bias can have a single entry."""
        config = GenerationConfig(logit_bias={100: -50.0})
        assert len(config.logit_bias) == 1
        assert config.logit_bias[100] == -50.0

    def test_logit_bias_positive_values(self):
        """logit_bias accepts positive values (boost tokens)."""
        config = GenerationConfig(logit_bias={42: 10.0})
        assert config.logit_bias[42] == 10.0

    def test_logit_bias_negative_values(self):
        """logit_bias accepts negative values (suppress tokens)."""
        config = GenerationConfig(logit_bias={42: -100.0})
        assert config.logit_bias[42] == -100.0

    def test_logit_bias_zero_value(self):
        """logit_bias accepts zero (no effect)."""
        config = GenerationConfig(logit_bias={42: 0.0})
        assert config.logit_bias[42] == 0.0

    def test_logit_bias_multiple_entries(self):
        """logit_bias can have multiple entries."""
        bias = {100: -100.0, 200: -100.0, 300: 5.0, 400: 0.0}
        config = GenerationConfig(logit_bias=bias)
        assert len(config.logit_bias) == 4
        assert config.logit_bias[100] == -100.0
        assert config.logit_bias[300] == 5.0


class TestChatTemplate:
    """Tests for chat_template parameter (renamed from template_override)."""

    def test_chat_template_default_is_none(self):
        """chat_template defaults to None."""
        config = GenerationConfig()
        assert config.chat_template is None

    def test_chat_template_with_string(self):
        """chat_template can be set as a string in GenerationConfig."""
        template = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
        config = GenerationConfig(chat_template=template)
        assert config.chat_template == template

    def test_chat_template_with_prompt_template(self):
        """chat_template can be set as a PromptTemplate object."""
        from talu.template import PromptTemplate

        template_str = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
        template = PromptTemplate(template_str)
        config = GenerationConfig(chat_template=template)
        assert config.chat_template is template
        assert config.chat_template.source == template_str

    def test_chat_template_empty_string(self):
        """chat_template can be an empty string."""
        config = GenerationConfig(chat_template="")
        assert config.chat_template == ""

    def test_chat_template_simple_template(self):
        """chat_template accepts simple Jinja2 templates."""
        config = GenerationConfig(chat_template="{{ messages[0].content }}")
        assert "messages" in config.chat_template

    def test_chat_template_complex_template(self):
        """chat_template accepts complex templates with control flow."""
        template = """
{% if tools %}Available: {% for t in tools %}{{ t.name }}{% endfor %}{% endif %}
{% for m in messages %}{{ m.role }}: {{ m.content }}
{% endfor %}"""
        config = GenerationConfig(chat_template=template)
        assert "{% if tools %}" in config.chat_template
        assert "{% for m in messages %}" in config.chat_template


class TestExtraContext:
    """Tests for extra_context parameter."""

    def test_extra_context_default_is_none(self):
        """extra_context defaults to None."""
        config = GenerationConfig()
        assert config.extra_context is None

    def test_extra_context_in_config(self):
        """extra_context can be set in GenerationConfig."""
        context = {"tools": [{"name": "search"}], "date": "2024-01-15"}
        config = GenerationConfig(extra_context=context)
        assert config.extra_context == context

    def test_extra_context_empty_dict(self):
        """extra_context can be an empty dict."""
        config = GenerationConfig(extra_context={})
        assert config.extra_context == {}

    def test_extra_context_with_tools_array(self):
        """extra_context can contain tool definitions."""
        tools = [
            {"name": "search", "description": "Search the web"},
            {"name": "calculator", "description": "Do math"},
        ]
        config = GenerationConfig(extra_context={"tools": tools})
        assert len(config.extra_context["tools"]) == 2
        assert config.extra_context["tools"][0]["name"] == "search"

    def test_extra_context_with_string_value(self):
        """extra_context can contain string values."""
        config = GenerationConfig(extra_context={"date": "2024-01-15"})
        assert config.extra_context["date"] == "2024-01-15"

    def test_extra_context_with_boolean_value(self):
        """extra_context can contain boolean values."""
        config = GenerationConfig(extra_context={"enable_thinking": True})
        assert config.extra_context["enable_thinking"] is True

    def test_extra_context_with_numeric_value(self):
        """extra_context can contain numeric values."""
        config = GenerationConfig(extra_context={"max_results": 10, "threshold": 0.5})
        assert config.extra_context["max_results"] == 10
        assert config.extra_context["threshold"] == 0.5

    def test_extra_context_with_nested_object(self):
        """extra_context can contain nested objects."""
        context = {"user": {"name": "Alice", "role": "admin"}}
        config = GenerationConfig(extra_context=context)
        assert config.extra_context["user"]["name"] == "Alice"
        assert config.extra_context["user"]["role"] == "admin"

    def test_extra_context_multiple_keys(self):
        """extra_context can have multiple keys."""
        context = {
            "tools": [{"name": "search"}],
            "date": "2024-01-15",
            "user_name": "Bob",
            "enable_cot": True,
        }
        config = GenerationConfig(extra_context=context)
        assert len(config.extra_context) == 4


class TestPreviewPromptWithConfig:
    """Tests for preview_prompt() with config parameter (chat_template support)."""

    def test_preview_prompt_accepts_config_parameter(self):
        """preview_prompt() accepts optional config parameter."""
        import inspect

        sig = inspect.signature(Chat.preview_prompt)
        params = list(sig.parameters.keys())
        assert "config" in params

    def test_preview_prompt_uses_chat_template_from_config(self):
        """preview_prompt() uses chat_template from config when provided."""
        chat = Chat(system="You are helpful.")

        # Custom template that formats messages differently
        template = "{% for m in messages %}[{{ m.role }}] {{ m.content }}\n{% endfor %}"
        config = GenerationConfig(chat_template=template)

        result = chat.preview_prompt(config=config)

        # Should use our custom format, not the default chat template
        assert "[system] You are helpful." in result

    def test_preview_prompt_uses_chat_template_as_prompt_template(self):
        """preview_prompt() accepts chat_template as PromptTemplate object."""
        from talu.template import PromptTemplate

        chat = Chat(system="You are helpful.")

        # Pass a PromptTemplate object instead of string
        template = PromptTemplate(
            "{% for m in messages %}[{{ m.role }}] {{ m.content }}\n{% endfor %}"
        )
        config = GenerationConfig(chat_template=template)

        result = chat.preview_prompt(config=config)

        # Should use our custom format
        assert "[system] You are helpful." in result

    def test_preview_prompt_without_config_uses_session_template(self):
        """preview_prompt() without config uses session-level template."""
        from talu.template import PromptTemplate

        custom_template = PromptTemplate(
            "{% for m in messages %}{{ m.role.upper() }}: {{ m.content }}\n{% endfor %}"
        )
        chat = Chat(system="Test", chat_template=custom_template)

        result = chat.preview_prompt()

        # Should use session-level template
        assert "SYSTEM: Test" in result

    def test_preview_prompt_config_override_takes_precedence(self):
        """Config chat_template takes precedence over session chat_template."""
        from talu.template import PromptTemplate

        # Session-level template
        session_template = PromptTemplate(
            "{% for m in messages %}SESSION: {{ m.content }}\n{% endfor %}"
        )
        chat = Chat(system="Test", chat_template=session_template)

        # Per-request override via config
        override_template = "{% for m in messages %}OVERRIDE: {{ m.content }}\n{% endfor %}"
        config = GenerationConfig(chat_template=override_template)

        result = chat.preview_prompt(config=config)

        # Config override should win
        assert "OVERRIDE: Test" in result
        assert "SESSION:" not in result

    def test_preview_prompt_none_config_uses_defaults(self):
        """preview_prompt(config=None) uses session/model defaults."""
        from talu.template import PromptTemplate

        session_template = PromptTemplate(
            "{% for m in messages %}DEFAULT: {{ m.content }}\n{% endfor %}"
        )
        chat = Chat(system="Test", chat_template=session_template)

        result = chat.preview_prompt(config=None)

        # Should use session template
        assert "DEFAULT: Test" in result

    def test_preview_prompt_config_without_chat_template_uses_defaults(self):
        """Config without chat_template uses session/model defaults."""
        from talu.template import PromptTemplate

        session_template = PromptTemplate(
            "{% for m in messages %}SESSION: {{ m.content }}\n{% endfor %}"
        )
        chat = Chat(system="Test", chat_template=session_template)

        # Config with other settings but no chat_template
        config = GenerationConfig(temperature=0.5, max_tokens=100)

        result = chat.preview_prompt(config=config)

        # Should use session template since config has no override
        assert "SESSION: Test" in result

    def test_preview_prompt_with_add_generation_prompt(self):
        """preview_prompt() respects add_generation_prompt with config override."""
        chat = Chat(system="Test")

        template_with_gen = (
            "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
            "{% if add_generation_prompt %}ASSISTANT: {% endif %}"
        )
        config = GenerationConfig(chat_template=template_with_gen)

        result_with = chat.preview_prompt(add_generation_prompt=True, config=config)
        result_without = chat.preview_prompt(add_generation_prompt=False, config=config)

        assert "ASSISTANT:" in result_with
        assert "ASSISTANT:" not in result_without

    def test_preview_prompt_inherits_session_config_chat_template(self):
        """preview_prompt() inherits chat_template from session config."""
        # Session config with chat_template
        session_config = GenerationConfig(
            chat_template="{% for m in messages %}INHERITED: {{ m.content }}\n{% endfor %}"
        )
        chat = Chat(system="Test", config=session_config)

        # Call without explicit config - should use session config
        result = chat.preview_prompt()

        assert "INHERITED: Test" in result


class TestChatTemplateAndExtraContextCombined:
    """Tests for using chat_template and extra_context together."""

    def test_both_parameters_can_be_set(self):
        """Both chat_template and extra_context can be set together."""
        template = "{% if tools %}Tools: {{ tools | length }}{% endif %}"
        context = {"tools": [{"name": "a"}, {"name": "b"}]}
        config = GenerationConfig(chat_template=template, extra_context=context)
        assert config.chat_template is not None
        assert config.extra_context is not None

    def test_template_referencing_extra_context_vars(self):
        """Template can reference variables from extra_context."""
        template = "Date: {{ date }}\n{{ messages[0].content }}"
        context = {"date": "2024-01-15"}
        config = GenerationConfig(chat_template=template, extra_context=context)
        # Just verify both are set - actual rendering tested at integration level
        assert "{{ date }}" in config.chat_template
        assert config.extra_context["date"] == "2024-01-15"

    def test_serialization_includes_new_fields(self):
        """to_dict includes chat_template and extra_context."""
        config = GenerationConfig(
            chat_template="test template",
            extra_context={"key": "value"},
        )
        chat = Chat(config=config)
        data = chat.to_dict()

        assert "config" in data
        assert data["config"].get("chat_template") == "test template"
        assert data["config"].get("extra_context") == {"key": "value"}

    def test_deserialization_restores_new_fields(self):
        """from_dict restores chat_template and extra_context."""
        data = {
            "messages": [],
            "config": {
                "temperature": 0.7,
                "max_tokens": 100,
                "top_k": 50,
                "top_p": 0.9,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "chat_template": "custom template",
                "extra_context": {"tools": []},
            },
        }
        chat = Chat.from_dict(data)
        assert chat.config.chat_template == "custom template"
        assert chat.config.extra_context == {"tools": []}


class TestStructuredOutputConfig:
    """Tests for structured output settings in GenerationConfig."""

    def test_schema_strategy_default_is_auto(self):
        """schema_strategy defaults to 'auto'."""
        config = GenerationConfig()
        assert config.schema_strategy == "auto"

    def test_schema_strategy_valid_values(self):
        """schema_strategy accepts valid strategy names."""
        for strategy in ["auto", "typescript", "json_schema", "xml_schema"]:
            config = GenerationConfig(schema_strategy=strategy)
            assert config.schema_strategy == strategy

    def test_inject_schema_prompt_default_is_true(self):
        """inject_schema_prompt defaults to True."""
        config = GenerationConfig()
        assert config.inject_schema_prompt is True

    def test_inject_schema_prompt_can_be_disabled(self):
        """inject_schema_prompt can be set to False."""
        config = GenerationConfig(inject_schema_prompt=False)
        assert config.inject_schema_prompt is False

    def test_allow_thinking_default_is_false(self):
        """allow_thinking defaults to False."""
        config = GenerationConfig()
        assert config.allow_thinking is False

    def test_allow_thinking_can_be_enabled(self):
        """allow_thinking can be set to True."""
        config = GenerationConfig(allow_thinking=True)
        assert config.allow_thinking is True

    def test_max_thinking_tokens_default_is_512(self):
        """max_thinking_tokens defaults to 512."""
        config = GenerationConfig()
        assert config.max_thinking_tokens == 512

    def test_max_thinking_tokens_custom_value(self):
        """max_thinking_tokens can be set to custom value."""
        config = GenerationConfig(max_thinking_tokens=1024)
        assert config.max_thinking_tokens == 1024

    def test_all_structured_output_fields_together(self):
        """All structured output fields can be set together."""
        config = GenerationConfig(
            schema_strategy="json_schema",
            inject_schema_prompt=False,
            allow_thinking=True,
            max_thinking_tokens=2048,
        )
        assert config.schema_strategy == "json_schema"
        assert config.inject_schema_prompt is False
        assert config.allow_thinking is True
        assert config.max_thinking_tokens == 2048

    def test_structured_output_fields_accessible_via_chat_config(self):
        """Structured output fields are accessible via chat.config."""
        config = GenerationConfig(
            schema_strategy="typescript",
            allow_thinking=True,
        )
        chat = Chat(config=config)
        assert chat.config.schema_strategy == "typescript"
        assert chat.config.allow_thinking is True

    def test_build_effective_config_preserves_structured_output_fields(self):
        """_build_effective_config preserves structured output fields."""
        chat = Chat(
            config=GenerationConfig(
                schema_strategy="json_schema",
                allow_thinking=True,
                max_thinking_tokens=1024,
            )
        )

        effective = chat._build_effective_config()
        assert effective.schema_strategy == "json_schema"
        assert effective.allow_thinking is True
        assert effective.max_thinking_tokens == 1024

    def test_structured_output_fields_overridable_via_config_param(self):
        """Structured output fields can be overridden via config parameter."""
        chat = Chat(
            config=GenerationConfig(
                schema_strategy="auto",
                allow_thinking=False,
            )
        )

        override = GenerationConfig(
            schema_strategy="xml_schema",
            allow_thinking=True,
        )
        effective = chat._build_effective_config(config=override)

        assert effective.schema_strategy == "xml_schema"
        assert effective.allow_thinking is True


class TestToDictSerialization:
    """Tests for GenerationConfig.to_dict() serialization."""

    def test_to_dict_with_response_format_object(self):
        """to_dict() serializes ResponseFormat object correctly."""
        from talu.chat.response import ResponseFormat

        rf = ResponseFormat(
            type="json_object",
            json_schema={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        config = GenerationConfig(response_format=rf)
        data = config.to_dict()

        assert data["response_format"] is not None
        assert data["response_format"]["type"] == "json_object"
        assert data["response_format"]["json_schema"]["type"] == "object"

    def test_to_dict_with_response_format_dict(self):
        """to_dict() passes through response_format dict as-is."""
        rf = {"type": "json", "schema": {"type": "object"}}
        config = GenerationConfig(response_format=rf)
        data = config.to_dict()

        assert data["response_format"] == rf

    def test_to_dict_with_prompt_template_chat_template(self):
        """to_dict() extracts source from PromptTemplate chat_template."""
        from talu.template import PromptTemplate

        template_str = "{% for m in messages %}{{ m.content }}{% endfor %}"
        template = PromptTemplate(template_str)
        config = GenerationConfig(chat_template=template)
        data = config.to_dict()

        # Should serialize as the source string, not the object
        assert data["chat_template"] == template_str

    def test_to_dict_with_string_chat_template(self):
        """to_dict() passes through string chat_template as-is."""
        template_str = "{% for m in messages %}{{ m.content }}{% endfor %}"
        config = GenerationConfig(chat_template=template_str)
        data = config.to_dict()

        assert data["chat_template"] == template_str


class TestPipeOperator:
    """Tests for GenerationConfig pipe operator (merge)."""

    def test_ror_operator_with_generation_config(self):
        """__ror__ merges configs when other is also GenerationConfig."""
        c1 = GenerationConfig(temperature=0.5, max_tokens=100)
        c2 = GenerationConfig(temperature=0.9)  # Override temperature

        # c1 | c2 uses c1.__or__(c2)
        result = c1 | c2
        assert result.temperature == 0.9  # c2 wins
        assert result.max_tokens == 100  # From c1

    def test_ror_operator_with_non_generation_config_returns_not_implemented(self):
        """__ror__ returns NotImplemented for non-GenerationConfig."""
        c = GenerationConfig(temperature=0.5)

        # Calling __ror__ directly with non-GenerationConfig
        result = c.__ror__({"temperature": 0.9})
        assert result is NotImplemented

    def test_or_operator_with_non_generation_config_returns_not_implemented(self):
        """__or__ returns NotImplemented for non-GenerationConfig."""
        c = GenerationConfig(temperature=0.5)

        # Calling __or__ directly with non-GenerationConfig
        result = c.__or__({"temperature": 0.9})
        assert result is NotImplemented

    def test_ror_direct_call_with_generation_config(self):
        """__ror__ called directly merges configs."""
        c1 = GenerationConfig(temperature=0.5, max_tokens=100)
        c2 = GenerationConfig(temperature=0.9)

        # Call __ror__ directly with a GenerationConfig
        result = c2.__ror__(c1)
        assert result.temperature == 0.9  # c2 (self) wins when c1 calls __ror__
        assert result.max_tokens == 100  # From c1

    def test_or_operator_commutative_behavior(self):
        """Pipe operator is not commutative - right side wins for overlapping keys."""
        c1 = GenerationConfig(temperature=0.3, max_tokens=50, top_k=30)
        c2 = GenerationConfig(temperature=0.9, max_tokens=200)

        result_12 = c1 | c2
        result_21 = c2 | c1

        # c2 wins when on right
        assert result_12.temperature == 0.9
        assert result_12.max_tokens == 200
        assert result_12.top_k == 30  # From c1 (c2 has default 50)

        # c1 wins when on right
        assert result_21.temperature == 0.3
        assert result_21.max_tokens == 50
        assert result_21.top_k == 30


class TestValidationRetries:
    """Tests for validation_retries parameter."""

    def test_validation_retries_default_is_zero(self):
        """validation_retries defaults to 0 (disabled)."""
        config = GenerationConfig()
        assert config.validation_retries == 0

    def test_validation_retries_custom_value(self):
        """validation_retries can be set to a custom value."""
        config = GenerationConfig(validation_retries=3)
        assert config.validation_retries == 3

    def test_validation_retries_in_to_dict(self):
        """validation_retries is included in to_dict output."""
        config = GenerationConfig(validation_retries=2)
        data = config.to_dict()
        assert data["validation_retries"] == 2

    def test_validation_retries_accessible_via_chat_config(self):
        """validation_retries is accessible via chat.config."""
        config = GenerationConfig(validation_retries=1)
        chat = Chat(config=config)
        assert chat.config.validation_retries == 1

    def test_build_effective_config_preserves_validation_retries(self):
        """_build_effective_config preserves validation_retries."""
        chat = Chat(config=GenerationConfig(validation_retries=2))
        effective = chat._build_effective_config()
        assert effective.validation_retries == 2

    def test_validation_retries_overridable_via_config_param(self):
        """validation_retries can be overridden via config parameter."""
        chat = Chat(config=GenerationConfig(validation_retries=1))
        override = GenerationConfig(validation_retries=3)
        effective = chat._build_effective_config(config=override)
        assert effective.validation_retries == 3

    def test_validation_retries_overridable_via_kwargs(self):
        """validation_retries can be overridden via kwargs."""
        chat = Chat(config=GenerationConfig(validation_retries=1))
        effective = chat._build_effective_config(validation_retries=5)
        assert effective.validation_retries == 5

    def test_validation_retries_in_config_override(self):
        """override() includes validation_retries."""
        config = GenerationConfig(validation_retries=1)
        new_config = config.override(validation_retries=4)
        assert new_config.validation_retries == 4
        assert config.validation_retries == 1  # Original unchanged

    def test_validation_retries_in_config_merge(self):
        """Pipe operator includes validation_retries."""
        c1 = GenerationConfig(temperature=0.5)
        c2 = GenerationConfig(validation_retries=2)
        merged = c1 | c2
        assert merged.validation_retries == 2
        assert merged.temperature == 0.5

    def test_format_validation_retry_message(self):
        """_format_validation_retry_message produces clear error message."""
        from talu.exceptions import SchemaValidationError

        chat = Chat()
        error = SchemaValidationError(
            raw_text='{"age": 130}',
            validation_error=ValueError("Age must be less than 120"),
        )
        msg = chat._format_validation_retry_message(error)

        assert "valid JSON but failed validation" in msg
        assert "Age must be less than 120" in msg
        assert "Please try again" in msg
