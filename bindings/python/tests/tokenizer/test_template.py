"""Smoke tests for tokenizer chat-template module.

Maps to: talu/tokenizer/template.py
"""

from talu.tokenizer.template import ChatTemplate, apply_chat_template


def test_template_module_exports() -> None:
    assert callable(apply_chat_template)
    assert ChatTemplate.__name__ == "ChatTemplate"

