"""Tests for talu.template.template module.

Tests for the Template class - Jinja2-compatible prompt templating.
Note: Detailed tests for specific features are in separate test files:
- test_basic.py: Basic variable substitution
- test_control_flow.py: if/for/set/macro
- test_filters.py: Jinja2 filters
- test_functions.py: Built-in functions
- test_operators.py: Operators
- test_errors.py: Error handling
- test_edge_cases.py: Edge cases
"""

import pytest

from talu.exceptions import TemplateSyntaxError
from talu.template import PromptTemplate


class TestTemplateConstruction:
    """Tests for Template construction."""

    def test_template_from_string(self):
        """Template can be created from a string."""
        t = PromptTemplate("Hello {{ name }}!")
        assert t is not None

    def test_template_source_property(self):
        """Template.source returns the template string."""
        source = "Hello {{ name }}!"
        t = PromptTemplate(source)
        assert t.source == source


class TestTemplateRendering:
    """Tests for Template rendering."""

    def test_render_simple(self):
        """render() substitutes variables."""
        t = PromptTemplate("Hello {{ name }}!")
        result = t.render(name="World")
        assert result == "Hello World!"

    def test_format_simple(self):
        """format() substitutes variables (like str.format)."""
        t = PromptTemplate("Hello {{ name }}!")
        result = t.render(name="World")
        assert result == "Hello World!"

    def test_callable(self):
        """Template is callable."""
        t = PromptTemplate("Hello {{ name }}!")
        result = t(name="World")
        assert result == "Hello World!"

    def test_render_multiple_vars(self):
        """render() handles multiple variables."""
        t = PromptTemplate("{{ greeting }} {{ name }}!")
        result = t.render(greeting="Hello", name="World")
        assert result == "Hello World!"

    def test_render_missing_var(self):
        """render() with missing variable renders as empty in lenient mode."""
        t = PromptTemplate("Hello {{ name }}!", strict=False)
        result = t.render()
        assert result == "Hello !"

    def test_render_with_default_filter(self):
        """render() uses default filter for missing variables."""
        t = PromptTemplate("Hello {{ name | default('World') }}!")
        result = t.render()
        assert result == "Hello World!"


class TestTemplateSyntaxErrors:
    """Tests for template syntax error handling."""

    def test_unclosed_variable(self):
        """Unclosed variable raises TemplateSyntaxError."""
        with pytest.raises(TemplateSyntaxError):
            PromptTemplate("Hello {{ name")

    def test_unclosed_block(self):
        """Unclosed block raises TemplateSyntaxError."""
        with pytest.raises(TemplateSyntaxError):
            PromptTemplate("{% if true %}")


class TestTemplateControlFlow:
    """Basic tests for Template control flow."""

    def test_if_true(self):
        """{% if %} renders when condition is true."""
        t = PromptTemplate("{% if show %}Hello{% endif %}")
        result = t.render(show=True)
        assert result == "Hello"

    def test_if_false(self):
        """{% if %} skips when condition is false."""
        t = PromptTemplate("{% if show %}Hello{% endif %}")
        result = t.render(show=False)
        assert result == ""

    def test_for_loop(self):
        """{% for %} iterates over items."""
        t = PromptTemplate("{% for i in items %}{{ i }}{% endfor %}")
        result = t.render(items=[1, 2, 3])
        assert result == "123"


class TestTemplateFilters:
    """Basic tests for Template filters."""

    def test_upper(self):
        """| upper filter works."""
        t = PromptTemplate("{{ name | upper }}")
        result = t.render(name="hello")
        assert result == "HELLO"

    def test_lower(self):
        """| lower filter works."""
        t = PromptTemplate("{{ name | lower }}")
        result = t.render(name="HELLO")
        assert result == "hello"

    def test_length(self):
        """| length filter works."""
        t = PromptTemplate("{{ items | length }}")
        result = t.render(items=[1, 2, 3])
        assert result == "3"


class TestTemplateRepr:
    """Tests for Template repr."""

    def test_repr(self):
        """Template has a useful repr."""
        t = PromptTemplate("Hello {{ name }}!")
        r = repr(t)
        assert "Template" in r
