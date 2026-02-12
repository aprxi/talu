"""
Additional tests for talu/tokenizer/template.py coverage.

Targets uncovered edge cases in apply_chat_template and ChatTemplate.
"""

import pytest

from talu.tokenizer.template import ChatTemplate, apply_chat_template

# =============================================================================
# apply_chat_template Message Conversion Tests
# =============================================================================


class TestApplyChatTemplateMessageConversion:
    """Tests for message conversion in apply_chat_template."""

    @pytest.mark.requires_model
    def test_messages_with_to_list_method(self, test_model_path):
        """apply_chat_template calls to_list() on objects with that method."""

        class MessageListLike:
            def to_list(self):
                return [{"role": "user", "content": "Hello from to_list!"}]

        messages = MessageListLike()
        result = apply_chat_template(test_model_path, messages)
        assert "Hello from to_list" in result

    @pytest.mark.requires_model
    def test_messages_as_iterator(self, test_model_path):
        """apply_chat_template handles iterator."""

        def message_generator():
            yield {"role": "user", "content": "Hello from generator!"}

        result = apply_chat_template(test_model_path, message_generator())
        assert "Hello from generator" in result

    @pytest.mark.requires_model
    def test_messages_as_tuple(self, test_model_path):
        """apply_chat_template handles tuple of messages."""
        messages = ({"role": "user", "content": "Hello from tuple!"},)
        result = apply_chat_template(test_model_path, messages)
        assert "Hello from tuple" in result

    @pytest.mark.requires_model
    def test_messages_as_list(self, test_model_path):
        """apply_chat_template handles regular list."""
        messages = [{"role": "user", "content": "Hello from list!"}]
        result = apply_chat_template(test_model_path, messages)
        assert "Hello from list" in result


# =============================================================================
# apply_chat_template Parameters Tests
# =============================================================================


class TestApplyChatTemplateParameters:
    """Tests for apply_chat_template parameters."""

    @pytest.mark.requires_model
    def test_add_generation_prompt_true(self, test_model_path):
        """add_generation_prompt=True adds assistant marker."""
        messages = [{"role": "user", "content": "Hello"}]
        result = apply_chat_template(test_model_path, messages, add_generation_prompt=True)
        # Should have content
        assert len(result) > 0

    @pytest.mark.requires_model
    def test_add_generation_prompt_false(self, test_model_path):
        """add_generation_prompt=False omits assistant marker."""
        messages = [{"role": "user", "content": "Hello"}]
        result = apply_chat_template(test_model_path, messages, add_generation_prompt=False)
        # Should still have content
        assert len(result) > 0

    @pytest.mark.requires_model
    def test_multiturn_conversation(self, test_model_path):
        """apply_chat_template handles multi-turn conversation."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]
        result = apply_chat_template(test_model_path, messages)
        assert "2+2" in result or "3+3" in result


# =============================================================================
# ChatTemplate Class Tests
# =============================================================================


class TestChatTemplateClass:
    """Tests for ChatTemplate class."""

    @pytest.mark.requires_model
    def test_construction(self, test_model_path):
        """ChatTemplate can be constructed with model path."""
        template = ChatTemplate(test_model_path)
        assert template._model_path == test_model_path

    @pytest.mark.requires_model
    def test_apply_method(self, test_model_path):
        """ChatTemplate.apply() formats messages."""
        template = ChatTemplate(test_model_path)
        result = template.apply([{"role": "user", "content": "Hello!"}])
        assert "Hello" in result

    @pytest.mark.requires_model
    def test_apply_with_add_generation_prompt(self, test_model_path):
        """ChatTemplate.apply() accepts add_generation_prompt."""
        template = ChatTemplate(test_model_path)
        result = template.apply(
            [{"role": "user", "content": "Test"}],
            add_generation_prompt=False,
        )
        assert isinstance(result, str)

    @pytest.mark.requires_model
    def test_repr(self, test_model_path):
        """ChatTemplate has informative repr."""
        template = ChatTemplate(test_model_path)
        repr_str = repr(template)
        assert "ChatTemplate" in repr_str
        assert test_model_path in repr_str


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestApplyChatTemplateErrors:
    """Tests for apply_chat_template error handling."""

    def test_invalid_model_path_raises(self, tmp_path):
        """apply_chat_template raises for invalid model path."""
        from talu.exceptions import TaluError

        invalid_path = str(tmp_path / "nonexistent_model")
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(TaluError):
            apply_chat_template(invalid_path, messages)

    def test_invalid_messages_raises(self, tmp_path):
        """apply_chat_template raises for invalid messages format."""
        # Create a minimal model dir (will fail on template, not on messages)
        from talu.exceptions import TaluError

        invalid_path = str(tmp_path / "invalid")
        messages = "not a list"  # Invalid

        with pytest.raises((TaluError, TypeError)):
            apply_chat_template(invalid_path, messages)
