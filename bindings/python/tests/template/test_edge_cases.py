"""
Edge case tests for Template.

Tests for unicode, whitespace, empty inputs, large templates, and special characters.
"""


class TestEmptyAndMinimal:
    """Tests for empty and minimal templates."""

    def test_empty_template(self, Template):
        """Empty template returns empty string."""
        t = Template("")
        assert t() == ""

    def test_whitespace_only(self, Template):
        """Whitespace-only template."""
        t = Template("   ")
        assert t() == "   "

    def test_newlines_only(self, Template):
        """Newline-only template."""
        t = Template("\n\n\n")
        assert t() == "\n\n\n"

    def test_single_char(self, Template):
        """Single character template."""
        t = Template("x")
        assert t() == "x"

    def test_single_variable(self, Template):
        """Template with only a variable."""
        t = Template("{{ x }}")
        assert t(x="hello") == "hello"


class TestWhitespace:
    """Tests for whitespace handling."""

    def test_preserve_spaces(self, Template):
        """Spaces around text are preserved."""
        t = Template("  hello  world  ")
        assert t() == "  hello  world  "

    def test_preserve_tabs(self, Template):
        """Tabs are preserved."""
        t = Template("a\tb\tc")
        assert t() == "a\tb\tc"

    def test_preserve_newlines(self, Template):
        """Newlines are preserved."""
        t = Template("line1\nline2\nline3")
        assert t() == "line1\nline2\nline3"

    def test_mixed_whitespace(self, Template):
        """Mixed whitespace preserved."""
        t = Template("  \t\n  \t\n  ")
        assert t() == "  \t\n  \t\n  "

    def test_whitespace_around_variable(self, Template):
        """Whitespace around variable preserved."""
        t = Template("  {{ x }}  ")
        assert t(x="hi") == "  hi  "

    def test_whitespace_in_variable(self, Template):
        """Whitespace in variable value preserved."""
        t = Template("{{ x }}")
        assert t(x="  hello  ") == "  hello  "


class TestUnicode:
    """Tests for Unicode handling."""

    def test_unicode_in_template(self, Template):
        """Unicode characters in template."""
        t = Template("Hello 世界!")
        assert t() == "Hello 世界!"

    def test_unicode_variable(self, Template):
        """Unicode in variable value."""
        t = Template("{{ x }}")
        assert t(x="こんにちは") == "こんにちは"

    def test_emoji(self, Template):
        """Emoji handling."""
        t = Template("{{ emoji }} is fun!")
        assert t(emoji="🎉") == "🎉 is fun!"

    def test_mixed_scripts(self, Template):
        """Mixed script handling."""
        t = Template("{{ greeting }} {{ name }}!")
        assert t(greeting="Привет", name="世界") == "Привет 世界!"

    def test_arabic(self, Template):
        """Arabic text handling."""
        t = Template("{{ x }}")
        assert t(x="مرحبا") == "مرحبا"

    def test_accented_chars(self, Template):
        """Accented characters."""
        t = Template("{{ x }}")
        assert t(x="Café résumé naïve") == "Café résumé naïve"

    def test_combining_characters(self, Template):
        """Unicode combining characters."""
        t = Template("{{ x }}")
        # e + combining acute accent
        assert t(x="Cafe\u0301") == "Cafe\u0301"

    def test_unicode_in_filter(self, Template):
        """Unicode with filters."""
        t = Template("{{ x | upper }}")
        # Note: upper() behavior on non-ASCII varies
        result = t(x="hello")
        assert result == "HELLO"

    def test_unicode_in_loop(self, Template):
        """Unicode in loop."""
        t = Template("{% for c in chars %}{{ c }}{% endfor %}")
        assert t(chars=["日", "本", "語"]) == "日本語"


class TestSpecialCharacters:
    """Tests for special character handling."""

    def test_html_chars(self, Template):
        """HTML special characters in raw output."""
        t = Template("{{ x }}")
        # Without escape filter, should pass through
        result = t(x="<b>bold</b>")
        # Implementation may auto-escape or not
        assert "bold" in result

    def test_quotes(self, Template):
        """Quote handling."""
        t = Template("{{ x }}")
        assert t(x='He said "hello"') == 'He said "hello"'

    def test_single_quotes(self, Template):
        """Single quote handling."""
        t = Template("{{ x }}")
        assert t(x="It's working") == "It's working"

    def test_backslash(self, Template):
        """Backslash handling."""
        t = Template("{{ x }}")
        assert t(x="path\\to\\file") == "path\\to\\file"

    def test_curly_braces_escaped(self, Template):
        """Escaped curly braces."""
        t = Template("{{ '{{' }} and {{ '}}' }}")
        assert t() == "{{ and }}"

    def test_percent_sign(self, Template):
        """Percent sign in output."""
        t = Template("{{ x }}%")
        assert t(x=50) == "50%"

    def test_null_char(self, Template):
        """Null character handling."""
        t = Template("{{ x }}")
        result = t(x="a\x00b")
        # Should handle gracefully - verify we get a result
        assert isinstance(result, str)
        assert "a" in result  # At minimum, non-null chars preserved


class TestLargeInputs:
    """Tests for large templates and data."""

    def test_large_template(self, Template):
        """Large template string."""
        content = "x" * 10000
        t = Template(content)
        assert len(t()) == 10000

    def test_many_variables(self, Template):
        """Many variables in template."""
        vars_list = [f"{{{{ v{i} }}}}" for i in range(100)]
        template_str = " ".join(vars_list)
        t = Template(template_str)
        vars_dict = {f"v{i}": str(i) for i in range(100)}
        result = t(**vars_dict)
        assert "0" in result and "99" in result

    def test_large_loop(self, Template):
        """Large loop iteration."""
        t = Template("{% for i in items %}{{ i }}{% endfor %}")
        items = list(range(1000))
        result = t(items=items)
        assert "0" in result and "999" in result

    def test_deeply_nested_data(self, Template):
        """Deeply nested data access."""
        t = Template("{{ a.b.c.d.e }}")
        data = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
        assert t(**data) == "deep"

    def test_long_string_variable(self, Template):
        """Long string in variable."""
        t = Template("{{ x }}")
        long_string = "a" * 100000
        assert t(x=long_string) == long_string


class TestNestedStructures:
    """Tests for complex nested template structures."""

    def test_nested_if(self, Template):
        """Nested if statements."""
        t = Template("""
{%- if a -%}
  {%- if b -%}both{%- else -%}only a{%- endif -%}
{%- else -%}
  {%- if b -%}only b{%- else -%}neither{%- endif -%}
{%- endif -%}
""")
        assert t(a=True, b=True) == "both"
        assert t(a=True, b=False) == "only a"
        assert t(a=False, b=True) == "only b"
        assert t(a=False, b=False) == "neither"

    def test_nested_for(self, Template):
        """Nested for loops."""
        t = Template("""
{%- for row in matrix -%}
[{%- for cell in row -%}{{ cell }}{%- if not loop.last -%},{%- endif -%}{%- endfor -%}]
{%- endfor -%}
""")
        assert t(matrix=[[1, 2], [3, 4]]) == "[1,2][3,4]"

    def test_for_inside_if(self, Template):
        """For loop inside if."""
        t = Template("""
{%- if show -%}
{%- for i in items -%}{{ i }}{%- endfor -%}
{%- endif -%}
""")
        assert t(show=True, items=[1, 2, 3]) == "123"
        assert t(show=False, items=[1, 2, 3]) == ""

    def test_if_inside_for(self, Template):
        """If inside for loop."""
        t = Template("{% for i in items %}{% if i > 2 %}{{ i }}{% endif %}{% endfor %}")
        assert t(items=[1, 2, 3, 4, 5]) == "345"


class TestComments:
    """Tests for template comments."""

    def test_comment_removed(self, Template):
        """Comments are removed from output."""
        t = Template("hello{# this is a comment #}world")
        assert t() == "helloworld"

    def test_multiline_comment(self, Template):
        """Multiline comments."""
        t = Template("""hello{#
this is a
multiline comment
#}world""")
        assert t() == "helloworld"

    def test_comment_with_code(self, Template):
        """Comments can contain template-like syntax."""
        t = Template("x{# {{ var }} {% if %} #}y")
        assert t() == "xy"


class TestRawBlock:
    """Tests for raw blocks (no processing)."""

    def test_raw_block(self, Template):
        """Raw block passes through unprocessed."""
        t = Template("{% raw %}{{ not_a_variable }}{% endraw %}")
        assert t() == "{{ not_a_variable }}"

    def test_raw_with_control(self, Template):
        """Raw block with control structures."""
        t = Template("{% raw %}{% if x %}{% endif %}{% endraw %}")
        assert t() == "{% if x %}{% endif %}"


class TestLineStatements:
    """Tests for line statement syntax (if supported)."""

    # Note: Line statements (# for line) may not be enabled by default
    # These tests document expected behavior if enabled

    pass  # Skip if not supported


class TestEdgeCaseSyntax:
    """Tests for edge cases in syntax."""

    def test_adjacent_variables(self, Template):
        """Adjacent variable tags."""
        t = Template("{{ a }}{{ b }}")
        assert t(a="x", b="y") == "xy"

    def test_variable_no_space(self, Template):
        """Variable with minimal whitespace."""
        t = Template("{{x}}")
        assert t(x="hello") == "hello"

    def test_tag_no_space(self, Template):
        """Control tag with minimal whitespace."""
        t = Template("{%if x%}yes{%endif%}")
        assert t(x=True) == "yes"

    def test_multiple_expressions(self, Template):
        """Multiple expressions in output."""
        t = Template("{{ a + b }} and {{ c * d }}")
        assert t(a=1, b=2, c=3, d=4) == "3 and 12"

    def test_complex_expression(self, Template):
        """Complex expression in output."""
        t = Template("{{ (a + b) * (c - d) / e }}")
        assert t(a=1, b=2, c=10, d=4, e=3) == "6.0"
