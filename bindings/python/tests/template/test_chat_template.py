"""
Tests for PromptTemplate chat functionality.

Tests for PromptTemplate's chat format presets.
"""

import json

import pytest

from talu.template import PromptTemplate


class TestChatTemplateBasics:
    """Basic tests for PromptTemplate chat presets."""

    def test_chatml_preset_exists(self):
        """ChatML preset is available."""
        t = PromptTemplate.from_preset("chatml")
        assert isinstance(t, PromptTemplate)
        assert "<|im_start|>" in t.source

    def test_llama2_preset_exists(self):
        """Llama-2 preset is available."""
        t = PromptTemplate.from_preset("llama2")
        assert isinstance(t, PromptTemplate)
        assert "[INST]" in t.source

    def test_alpaca_preset_exists(self):
        """Alpaca preset is available."""
        t = PromptTemplate.from_preset("alpaca")
        assert isinstance(t, PromptTemplate)
        assert "### Instruction:" in t.source

    def test_vicuna_preset_exists(self):
        """Vicuna preset is available."""
        t = PromptTemplate.from_preset("vicuna")
        assert isinstance(t, PromptTemplate)
        assert "USER:" in t.source

    def test_zephyr_preset_exists(self):
        """Zephyr preset is available."""
        t = PromptTemplate.from_preset("zephyr")
        assert isinstance(t, PromptTemplate)

    def test_all_presets_return_prompttemplate(self):
        """Every preset returns a PromptTemplate instance."""
        for name in ["chatml", "llama2", "alpaca", "vicuna", "zephyr"]:
            t = PromptTemplate.from_preset(name)
            assert isinstance(t, PromptTemplate), f"{name} did not return PromptTemplate"

    def test_unknown_preset_raises_valueerror(self):
        """Unknown preset name raises ValueError with available names."""
        with pytest.raises(ValueError, match="Unknown preset"):
            PromptTemplate.from_preset("nonexistent")


class TestChatMLFormat:
    """Tests for ChatML format preset."""

    @pytest.fixture
    def chatml(self):
        """Get ChatML template."""
        return PromptTemplate.from_preset("chatml")

    def test_single_user_message(self, chatml):
        """Format single user message."""
        result = chatml.apply([{"role": "user", "content": "Hello!"}])
        assert "<|im_start|>user" in result
        assert "Hello!" in result
        assert "<|im_end|>" in result
        assert "<|im_start|>assistant" in result  # generation prompt

    def test_system_and_user(self, chatml):
        """Format system + user messages."""
        result = chatml.apply(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi!"},
            ]
        )
        assert "<|im_start|>system" in result
        assert "You are helpful." in result
        assert "<|im_start|>user" in result
        assert "Hi!" in result

    def test_multi_turn_conversation(self, chatml):
        """Format multi-turn conversation."""
        result = chatml.apply(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ]
        )
        assert result.count("<|im_start|>user") == 2
        assert result.count("<|im_start|>assistant") == 2  # 1 in history + 1 generation
        assert "Hello" in result
        assert "Hi there!" in result
        assert "How are you?" in result

    def test_no_generation_prompt(self, chatml):
        """Disable generation prompt for training data."""
        result = chatml.apply(
            [{"role": "user", "content": "Test"}],
            add_generation_prompt=False,
        )
        # Should not end with assistant marker
        assert not result.rstrip().endswith("<|im_start|>assistant")


class TestLlama2Format:
    """Tests for Llama-2 format preset."""

    @pytest.fixture
    def llama2(self):
        """Get Llama-2 template."""
        return PromptTemplate.from_preset("llama2")

    def test_with_bos_token(self, llama2):
        """Include BOS token when provided."""
        result = llama2.apply(
            [{"role": "user", "content": "Hello"}],
            bos_token="<s>",
        )
        assert result.startswith("<s>")

    def test_system_in_sys_tags(self, llama2):
        """System message uses <<SYS>> tags."""
        result = llama2.apply(
            [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"},
            ]
        )
        assert "<<SYS>>" in result
        assert "<</SYS>>" in result
        assert "Be helpful." in result

    def test_user_in_inst_tags(self, llama2):
        """User message uses [INST] tags."""
        result = llama2.apply([{"role": "user", "content": "Hello"}])
        assert "[INST]" in result
        assert "[/INST]" in result
        assert "Hello" in result


class TestAlpacaFormat:
    """Tests for Alpaca format preset."""

    @pytest.fixture
    def alpaca(self):
        """Get Alpaca template."""
        return PromptTemplate.from_preset("alpaca")

    def test_instruction_format(self, alpaca):
        """Use ### Instruction: format."""
        result = alpaca.apply([{"role": "user", "content": "What is 2+2?"}])
        assert "### Instruction:" in result
        assert "What is 2+2?" in result
        assert "### Response:" in result

    def test_system_as_context(self, alpaca):
        """System message appears before instruction."""
        result = alpaca.apply(
            [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "What is 2+2?"},
            ]
        )
        # System should come before instruction marker
        sys_pos = result.find("You are a math tutor.")
        inst_pos = result.find("### Instruction:")
        assert sys_pos < inst_pos


class TestApplyMethod:
    """Tests for PromptTemplate.apply() method."""

    def test_apply_is_shorthand_for_call(self):
        """Apply method is equivalent to __call__."""
        t = PromptTemplate.from_preset("chatml")
        messages = [{"role": "user", "content": "Test"}]

        via_apply = t.apply(messages, add_generation_prompt=True)
        via_call = t(messages=messages, add_generation_prompt=True)

        assert via_apply == via_call

    def test_apply_accepts_extra_kwargs(self):
        """Apply passes through extra kwargs."""
        # Create template that uses custom variable
        t = PromptTemplate("{{ prefix }}{% for m in messages %}{{ m.content }}{% endfor %}")
        result = t.apply(
            [{"role": "user", "content": "Hi"}],
            prefix="START: ",
        )
        assert result == "START: Hi"


class TestFromChatTemplate:
    """Tests for PromptTemplate.from_chat_template() class method."""

    @pytest.fixture
    def model_dir(self, tmp_path):
        """Create mock model directory with ChatML template."""
        model_path = tmp_path / "chat-model"
        model_path.mkdir()

        config = {
            "chat_template": (
                "{%- for message in messages %}"
                "<|{{ message.role }}|>{{ message.content }}<|end|>"
                "{%- endfor %}"
            ),
        }
        (model_path / "tokenizer_config.json").write_text(json.dumps(config))
        return model_path

    def test_from_chat_template_returns_prompttemplate(self, model_dir):
        """Return PromptTemplate instance."""
        t = PromptTemplate.from_chat_template(str(model_dir))
        assert isinstance(t, PromptTemplate)

    def test_from_chat_template_can_apply(self, model_dir):
        """Loaded template can use apply() method."""
        t = PromptTemplate.from_chat_template(str(model_dir))
        result = t.apply([{"role": "user", "content": "Hello"}])
        assert "<|user|>Hello<|end|>" in result


class TestPromptTemplateFeatures:
    """Tests for PromptTemplate unified features."""

    def test_source_property(self):
        """Source property works."""
        t = PromptTemplate.from_preset("chatml")
        assert isinstance(t.source, str)
        assert len(t.source) > 0

    def test_strict_mode(self):
        """Strict mode works."""
        t = PromptTemplate("{{ messages }}", strict=True)
        assert t.strict is True

    def test_input_variables(self):
        """Input variables detection works."""
        t = PromptTemplate.from_preset("chatml")
        # ChatML uses 'messages' and 'add_generation_prompt'
        # May not detect all due to regex limitations, but should get some
        assert isinstance(t.input_variables, set)

    def test_partial_application(self):
        """Partial application works."""
        t = PromptTemplate("{{ system }}{% for m in messages %}{{ m.content }}{% endfor %}")
        t2 = t.partial(system="System: ")
        result = t2(messages=[{"content": "Hi"}])
        assert "System: " in result

    def test_supports_system_role_chatml(self):
        """ChatML supports system role (uses message.role)."""
        t = PromptTemplate.from_preset("chatml")
        assert t.supports_system_role is True

    def test_supports_system_role_llama2(self):
        """Llama2 supports system role (has <<SYS>>)."""
        t = PromptTemplate.from_preset("llama2")
        assert t.supports_system_role is True

    def test_supports_system_role_explicit(self):
        """Template with explicit 'system' check supports system role."""
        t = PromptTemplate("{% if message.role == 'system' %}SYS{% endif %}")
        assert t.supports_system_role is True

    def test_supports_system_role_none(self):
        """Template without role handling doesn't support system role."""
        t = PromptTemplate("{{ messages[0].content }}")
        assert t.supports_system_role is False

    def test_supports_system_role_loop_variable(self):
        """Template using any .role pattern supports system role."""
        t = PromptTemplate("{% for m in messages %}{{ m.role }}: {{ m.content }}{% endfor %}")
        # Uses .role so detected as supporting roles
        assert t.supports_system_role is True

    def test_supports_system_role_zephyr(self):
        """Zephyr uses message.role so supports system."""
        t = PromptTemplate.from_preset("zephyr")
        assert t.supports_system_role is True

    def test_supports_tools_basic_presets(self):
        """Basic presets don't have tool support."""
        for name in ["chatml", "llama2", "alpaca", "vicuna", "zephyr"]:
            t = PromptTemplate.from_preset(name)
            assert t.supports_tools is False, f"{name} should not support tools"

    def test_supports_tools_with_tools_variable(self):
        """Template with tools variable supports tools."""
        t = PromptTemplate("{% if tools %}Tools: {{ tools }}{% endif %}")
        assert t.supports_tools is True

    def test_supports_tools_with_functions(self):
        """Template with functions variable supports tools."""
        t = PromptTemplate("{% for f in functions %}{{ f.name }}{% endfor %}")
        assert t.supports_tools is True

    def test_supports_tools_with_tool_call(self):
        """Template with tool_call handling supports tools."""
        t = PromptTemplate("{% if message.tool_calls %}TOOL{% endif %}")
        assert t.supports_tools is True

    def test_supports_tools_none(self):
        """Template without tool handling doesn't support tools."""
        t = PromptTemplate("{{ messages[0].content }}")
        assert t.supports_tools is False


class TestEdgeCases:
    """Edge case tests for PromptTemplate chat presets."""

    def test_empty_messages(self):
        """Handle empty message list."""
        t = PromptTemplate.from_preset("chatml")
        result = t.apply([])
        # Should still have generation prompt
        assert "<|im_start|>assistant" in result

    def test_unicode_content(self):
        """Handle unicode in messages."""
        t = PromptTemplate.from_preset("chatml")
        result = t.apply(
            [
                {"role": "user", "content": "こんにちは"},
                {"role": "assistant", "content": "你好"},
            ]
        )
        assert "こんにちは" in result
        assert "你好" in result

    def test_multiline_content(self):
        """Handle multiline message content."""
        t = PromptTemplate.from_preset("chatml")
        result = t.apply(
            [
                {"role": "user", "content": "Line 1\nLine 2\nLine 3"},
            ]
        )
        assert "Line 1\nLine 2\nLine 3" in result

    def test_special_characters_in_content(self):
        """Handle special characters in content."""
        t = PromptTemplate.from_preset("chatml")
        result = t.apply(
            [
                {"role": "user", "content": "Test <tag> & \"quotes\" 'apostrophe'"},
            ]
        )
        assert "<tag>" in result
        assert "&" in result
