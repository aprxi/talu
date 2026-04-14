"""Smoke tests for template prompt module.

Maps to: talu/template/prompt.py
"""

from talu.template.prompt import PromptTemplate


def test_prompt_template_constructs() -> None:
    template = PromptTemplate("Hello {{ name }}")
    assert template(name="World") == "Hello World"

