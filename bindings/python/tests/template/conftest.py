"""
Template test fixtures.

Pytest fixtures and test data for PromptTemplate tests.
"""

import pytest


@pytest.fixture(scope="session")
def Template():
    """
    Get the PromptTemplate class from talu.

    Fails with clear message if PromptTemplate is not yet implemented.
    """
    try:
        from talu import PromptTemplate

        return PromptTemplate
    except ImportError:
        pytest.fail(
            "talu.PromptTemplate not found.\n\n"
            "Implementation required:\n"
            "  1. Add PromptTemplate class to talu/__init__.py\n"
            "  2. Implement PromptTemplate(template_string) constructor\n"
            "  3. Implement __call__(**variables) -> str\n"
            "  4. Implement .render(**variables) -> str"
        )


@pytest.fixture(scope="session")
def LenientTemplate():
    """
    Factory for lenient (strict=False) templates.

    Use this for tests that specifically test undefined variable handling
    in lenient mode (Jinja2-compatible behavior).

    Example:
        def test_undefined_renders_empty(LenientTemplate):
            t = LenientTemplate("{{ x }}")
            assert t() == ""  # No error in lenient mode
    """
    from functools import partial

    from talu import PromptTemplate

    return partial(PromptTemplate, strict=False)


@pytest.fixture
def simple_template(Template):
    """A simple template with one variable."""
    return Template("Hello {{ name }}!")


@pytest.fixture
def multi_var_template(Template):
    """A template with multiple variables."""
    return Template("{{ greeting }}, {{ name }}! You are {{ age }} years old.")


# =============================================================================
# Test Data (shared across test modules)
# =============================================================================

# Basic variable substitution cases
BASIC_CASES = [
    ("{{ x }}", {"x": "hello"}, "hello"),
    ("{{ name }}", {"name": "World"}, "World"),
    ("Hello {{ name }}!", {"name": "Alice"}, "Hello Alice!"),
    ("{{ a }} and {{ b }}", {"a": "one", "b": "two"}, "one and two"),
    ("No variables here", {}, "No variables here"),
]

# Number cases
NUMBER_CASES = [
    ("{{ x }}", {"x": 42}, "42"),
    ("{{ x }}", {"x": 3.14}, "3.14"),
    ("{{ x }}", {"x": 0}, "0"),
    ("{{ x }}", {"x": -1}, "-1"),
    ("Sum: {{ a + b }}", {"a": 1, "b": 2}, "Sum: 3"),
]

# Control flow cases
IF_CASES = [
    ("{% if x %}yes{% endif %}", {"x": True}, "yes"),
    ("{% if x %}yes{% endif %}", {"x": False}, ""),
    ("{% if x %}yes{% else %}no{% endif %}", {"x": True}, "yes"),
    ("{% if x %}yes{% else %}no{% endif %}", {"x": False}, "no"),
    ("{% if x > 5 %}big{% else %}small{% endif %}", {"x": 10}, "big"),
    ("{% if x > 5 %}big{% else %}small{% endif %}", {"x": 3}, "small"),
]

FOR_CASES = [
    ("{% for i in items %}{{ i }}{% endfor %}", {"items": [1, 2, 3]}, "123"),
    ("{% for i in items %}{{ i }} {% endfor %}", {"items": ["a", "b"]}, "a b "),
    (
        "{% for k, v in d.items() %}{{ k }}={{ v }} {% endfor %}",
        {"d": {"a": 1, "b": 2}},
        "a=1 b=2 ",
    ),
]

# Filter cases
FILTER_CASES = [
    ("{{ x | upper }}", {"x": "hello"}, "HELLO"),
    ("{{ x | lower }}", {"x": "HELLO"}, "hello"),
    ("{{ x | trim }}", {"x": "  hello  "}, "hello"),
    ("{{ x | length }}", {"x": [1, 2, 3]}, "3"),
    ("{{ x | first }}", {"x": [1, 2, 3]}, "1"),
    ("{{ x | last }}", {"x": [1, 2, 3]}, "3"),
    ("{{ x | join(', ') }}", {"x": ["a", "b", "c"]}, "a, b, c"),
    ("{{ x | default('N/A') }}", {"x": None}, "N/A"),
    ("{{ x | default('N/A') }}", {"x": "value"}, "value"),
]

# Operator cases
OPERATOR_CASES = [
    ("{{ a + b }}", {"a": 1, "b": 2}, "3"),
    ("{{ a - b }}", {"a": 5, "b": 3}, "2"),
    ("{{ a * b }}", {"a": 3, "b": 4}, "12"),
    ("{{ a / b }}", {"a": 10, "b": 2}, "5.0"),
    ("{{ a // b }}", {"a": 7, "b": 2}, "3"),
    ("{{ a ** b }}", {"a": 2, "b": 3}, "8"),
    ("{{ a % b }}", {"a": 7, "b": 3}, "1"),
    ("{{ a ~ b }}", {"a": "hello", "b": "world"}, "helloworld"),
    ("{{ 'x' in items }}", {"items": ["x", "y"]}, "True"),
    ("{{ 'z' not in items }}", {"items": ["x", "y"]}, "True"),
]

# AI pattern templates
RAG_TEMPLATE = """
Based on the following context:
{% for doc in documents %}
---
Source: {{ doc.source }}
{{ doc.content }}
{% endfor %}
---

Question: {{ question }}
"""

FEW_SHOT_TEMPLATE = """
{% for example in examples %}
Input: {{ example.input }}
Output: {{ example.output }}

{% endfor %}
Input: {{ query }}
Output:
"""

AGENT_TEMPLATE = """
You have access to the following tools:
{% for tool in tools %}
## {{ tool.name }}
{{ tool.description }}
Parameters: {{ tool.params }}

{% endfor %}

Task: {{ instruction }}
"""

STRUCTURED_OUTPUT_TEMPLATE = """
Extract the following fields from the text:
{% for field in schema %}
- {{ field.name }} ({{ field.type }}): {{ field.description }}
{% endfor %}

Text: {{ text }}

Return valid JSON.
"""
