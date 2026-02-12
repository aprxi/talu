"""
Basic Template functionality tests.

Tests for Template construction, calling, and variable substitution.
"""

import pytest

from tests.template.conftest import BASIC_CASES, NUMBER_CASES


class TestTemplateConstruction:
    """Tests for Template class construction."""

    def test_template_import(self, Template):
        """Template is importable from talu."""
        assert Template is not None

    def test_template_construction(self, Template):
        """Template can be constructed with a string."""
        t = Template("Hello {{ name }}!")
        assert t is not None

    def test_template_empty_string(self, Template):
        """Template accepts empty string."""
        t = Template("")
        assert t is not None

    def test_template_no_variables(self, Template):
        """Template accepts string without variables."""
        t = Template("Hello World!")
        assert t is not None


class TestTemplateCallable:
    """Tests for Template callable syntax: t(var=value)."""

    def test_callable_simple(self, Template):
        """Template is callable with keyword arguments."""
        t = Template("Hello {{ name }}!")
        result = t(name="World")
        assert result == "Hello World!"

    def test_callable_multiple_vars(self, Template):
        """Template callable with multiple variables."""
        t = Template("{{ a }} + {{ b }} = {{ c }}")
        result = t(a=1, b=2, c=3)
        assert result == "1 + 2 = 3"

    def test_callable_no_vars_needed(self, Template):
        """Template callable when no variables in template."""
        t = Template("Static text")
        result = t()
        assert result == "Static text"

    def test_callable_extra_vars_ignored(self, Template):
        """Extra variables are ignored."""
        t = Template("Hello {{ name }}!")
        result = t(name="World", extra="ignored")
        assert result == "Hello World!"

    @pytest.mark.parametrize("template,variables,expected", BASIC_CASES)
    def test_callable_basic_cases(self, Template, template, variables, expected):
        """Basic substitution cases work with callable syntax."""
        t = Template(template)
        result = t(**variables)
        assert result == expected


class TestTemplateRender:
    """Tests for Template .render() method."""

    def test_render_simple(self, Template):
        """Template.render() works for Jinja2 familiarity."""
        t = Template("Hello {{ name }}!")
        result = t.render(name="World")
        assert result == "Hello World!"

    def test_render_multiple_vars(self, Template):
        """Template.render() with multiple variables."""
        t = Template("{{ a }} and {{ b }}")
        result = t.render(a="one", b="two")
        assert result == "one and two"

    @pytest.mark.parametrize("template,variables,expected", BASIC_CASES)
    def test_render_basic_cases(self, Template, template, variables, expected):
        """Basic substitution cases work with .render()."""
        t = Template(template)
        result = t.render(**variables)
        assert result == expected


class TestTemplateEquivalence:
    """Tests that callable and .render() produce same results."""

    def test_callable_render_equivalent(self, Template):
        """Callable and .render() produce same result."""
        t = Template("Hello {{ name }}!")

        result_callable = t(name="World")
        result_render = t.render(name="World")

        assert result_callable == result_render == "Hello World!"

    def test_equivalence_complex_template(self, Template):
        """Equivalence holds for complex templates."""
        t = Template("{% for i in items %}{{ i }}{% endfor %}")
        vars = {"items": [1, 2, 3]}

        result_callable = t(**vars)
        result_render = t.render(**vars)

        assert result_callable == result_render == "123"


class TestTemplateNumbers:
    """Tests for numeric variable handling."""

    @pytest.mark.parametrize("template,variables,expected", NUMBER_CASES)
    def test_number_substitution(self, Template, template, variables, expected):
        """Numbers are correctly converted to strings."""
        t = Template(template)
        result = t(**variables)
        assert result == expected


class TestTemplateReuse:
    """Tests for template reuse (compile once, render many)."""

    def test_reuse_same_template(self, Template):
        """Same template can be rendered multiple times."""
        t = Template("Hello {{ name }}!")

        assert t(name="Alice") == "Hello Alice!"
        assert t(name="Bob") == "Hello Bob!"
        assert t(name="Charlie") == "Hello Charlie!"

    def test_reuse_different_values(self, Template):
        """Template produces correct output for different values."""
        t = Template("{{ x }} * {{ y }} = {{ x * y }}")

        assert t(x=2, y=3) == "2 * 3 = 6"
        assert t(x=5, y=5) == "5 * 5 = 25"
        assert t(x=0, y=100) == "0 * 100 = 0"


class TestTemplateDataTypes:
    """Tests for various data type handling."""

    def test_string_variable(self, Template):
        """String variables work."""
        t = Template("{{ x }}")
        assert t(x="hello") == "hello"

    def test_int_variable(self, Template):
        """Integer variables work."""
        t = Template("{{ x }}")
        assert t(x=42) == "42"

    def test_float_variable(self, Template):
        """Float variables work."""
        t = Template("{{ x }}")
        assert t(x=3.14) == "3.14"

    def test_bool_variable(self, Template):
        """Boolean variables work."""
        t = Template("{{ x }}")
        assert t(x=True) == "True"
        assert t(x=False) == "False"

    def test_none_variable(self, Template):
        """None variable renders as empty or 'None'."""
        t = Template("{{ x }}")
        result = t(x=None)
        assert result in ["", "None"]  # Implementation dependent

    def test_list_variable(self, Template):
        """List variables can be iterated."""
        t = Template("{% for i in x %}{{ i }}{% endfor %}")
        assert t(x=[1, 2, 3]) == "123"

    def test_dict_variable(self, Template):
        """Dict variables can be accessed."""
        t = Template("{{ x.a }} and {{ x.b }}")
        assert t(x={"a": 1, "b": 2}) == "1 and 2"

    def test_nested_dict(self, Template):
        """Nested dict access works."""
        t = Template("{{ user.address.city }}")
        data = {"user": {"address": {"city": "Paris"}}}
        assert t(**data) == "Paris"

    def test_list_indexing(self, Template):
        """List indexing works."""
        t = Template("{{ items[0] }} and {{ items[2] }}")
        assert t(items=["a", "b", "c"]) == "a and c"


class TestTemplateStr:
    """Tests for string representation."""

    def test_str_returns_string(self, Template):
        """Template result is a string."""
        t = Template("Hello {{ name }}!")
        result = t(name="World")
        assert isinstance(result, str)

    def test_template_repr(self, Template):
        """Template has informative repr."""
        t = Template("Hello {{ name }}!")
        repr_str = repr(t)
        assert "Template" in repr_str


class TestWhitespaceHandling:
    """Tests for whitespace trimming rules."""

    def test_whitespace_preserved_by_default(self, Template):
        """Whitespace in template is preserved by default."""
        t = Template("  {{ x }}  ")
        result = t(x="test")
        assert result == "  test  "

    def test_whitespace_control_minus(self, Template):
        """Minus sign trims whitespace before/after blocks."""
        t = Template("  {%- if True %}\ntest\n{%- endif %}")
        result = t()
        # Leading whitespace should be trimmed due to -%
        assert not result.startswith("  ")

    def test_leading_newline_preserved(self, Template):
        """Leading newlines are preserved without trimming."""
        t = Template("\n{{ x }}")
        result = t(x="test")
        assert result.startswith("\n")

    def test_trailing_newline_preserved(self, Template):
        """Trailing newlines are preserved."""
        t = Template("{{ x }}\n")
        result = t(x="test")
        assert result.endswith("\n")

    def test_whitespace_only_template(self, Template):
        """Whitespace-only template works."""
        t = Template("   \n   ")
        result = t()
        assert result == "   \n   "

    def test_tabs_preserved(self, Template):
        """Tab characters are preserved."""
        t = Template("\t{{ x }}\t")
        result = t(x="test")
        assert result == "\ttest\t"

    def test_mixed_whitespace(self, Template):
        """Mixed whitespace (spaces, tabs, newlines) preserved."""
        t = Template(" \t\n{{ x }}\n\t ")
        result = t(x="test")
        assert result == " \t\ntest\n\t "


class TestTemplateFromFile:
    """Tests for Template.from_file() class method."""

    def test_from_file_basic(self, Template, tmp_path):
        """Template.from_file() loads a template from disk."""
        template_file = tmp_path / "test.j2"
        template_file.write_text("Hello {{ name }}!")

        t = Template.from_file(str(template_file))
        result = t(name="World")
        assert result == "Hello World!"

    def test_from_file_complex(self, Template, tmp_path):
        """Template.from_file() handles complex templates."""
        template_file = tmp_path / "complex.j2"
        template_file.write_text("""
{% for item in items %}
- {{ item }}
{% endfor %}
""")
        t = Template.from_file(str(template_file))
        result = t(items=["one", "two", "three"])
        assert "- one" in result
        assert "- two" in result
        assert "- three" in result

    def test_from_file_not_found(self, Template):
        """Template.from_file() raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            Template.from_file("/nonexistent/path/template.j2")

    def test_from_file_unicode(self, Template, tmp_path):
        """Template.from_file() handles unicode content."""
        template_file = tmp_path / "unicode.j2"
        template_file.write_text("Hello {{ name }}! 你好 世界")

        t = Template.from_file(str(template_file))
        result = t(name="用户")
        assert "用户" in result
        assert "你好 世界" in result

    def test_from_file_reusable(self, Template, tmp_path):
        """Template loaded from file can be reused."""
        template_file = tmp_path / "reuse.j2"
        template_file.write_text("{{ x }} + {{ y }} = {{ x + y }}")

        t = Template.from_file(str(template_file))
        assert t(x=1, y=2) == "1 + 2 = 3"
        assert t(x=10, y=20) == "10 + 20 = 30"
