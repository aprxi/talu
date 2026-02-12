"""
Tests for custom Python filters in PromptTemplate.

Tests the ability to register Python functions as template filters
that execute via callback during template rendering.
"""

import pytest

import talu


class TestBasicCustomFilters:
    """Basic custom filter functionality tests."""

    def test_simple_string_filter(self, Template):
        """Simple string filter transforms input."""
        t = Template("{{ name | shout }}")
        t.register_filter("shout", lambda s: s.upper() + "!!!")
        assert t(name="hello") == "HELLO!!!"

    def test_filter_with_single_arg(self, Template):
        """Filter with one argument works."""
        t = Template("{{ x | add(5) }}")
        t.register_filter("add", lambda x, n: x + n)
        assert t(x=10) == "15"

    def test_filter_with_multiple_args(self, Template):
        """Filter with multiple arguments works."""
        t = Template("{{ value | clamp(0, 100) }}")
        t.register_filter("clamp", lambda v, lo, hi: max(lo, min(hi, v)))
        assert t(value=-50) == "0"
        assert t(value=150) == "100"
        assert t(value=50) == "50"

    def test_chained_filters(self, Template):
        """Multiple custom filters can be chained."""
        t = Template("{{ x | double | add(1) }}")
        t.register_filter("double", lambda x: x * 2)
        t.register_filter("add", lambda x, n: x + n)
        assert t(x=5) == "11"  # (5*2)+1 = 11

    def test_method_chaining(self, Template):
        """register_filter returns self for chaining."""
        t = (
            Template("{{ x | triple | squared }}")
            .register_filter("triple", lambda x: x * 3)
            .register_filter("squared", lambda x: x**2)
        )
        assert t(x=2) == "36"  # (2*3)^2 = 36


class TestBuiltinOverride:
    """Tests for overriding built-in filters with custom ones."""

    def test_override_upper(self, Template):
        """Custom filter can override built-in filter."""
        t = Template("{{ name | upper }}")
        t.register_filter("upper", lambda s: "CUSTOM: " + s.upper())
        assert t(name="hello") == "CUSTOM: HELLO"

    def test_override_lower(self, Template):
        """Custom lower filter takes precedence."""
        t = Template("{{ name | lower }}")
        t.register_filter("lower", lambda s: s.lower() + " (custom)")
        assert t(name="HELLO") == "hello (custom)"


class TestComplexReturnTypes:
    """Tests for filters returning complex types."""

    def test_filter_returning_list(self, Template):
        """Filter can return list for further processing."""
        t = Template("{{ text | split_words | join(', ') }}")
        t.register_filter("split_words", lambda s: s.split())
        assert t(text="hello world foo") == "hello, world, foo"

    def test_filter_returning_dict(self, Template):
        """Filter can return dict."""
        t = Template("{{ data | wrap | tojson }}")
        t.register_filter("wrap", lambda d: {"wrapped": d})
        result = t(data={"key": "value"})
        assert '"wrapped"' in result
        assert '"key"' in result

    def test_filter_returning_number(self, Template):
        """Filter can return number."""
        t = Template("{{ items | count_positive }}")
        t.register_filter("count_positive", lambda lst: sum(1 for x in lst if x > 0))
        assert t(items=[1, -2, 3, -4, 5]) == "3"


class TestWithPartialApplication:
    """Tests for custom filters with partial() templates."""

    def test_filters_preserved_in_partial(self, Template):
        """Custom filters are preserved when using partial()."""
        t = Template("{{ greeting | emphasize }} {{ name }}")
        t.register_filter("emphasize", lambda s: s + "!")
        partial = t.partial(greeting="Hello")
        assert partial(name="World") == "Hello! World"

    def test_register_on_partial(self, Template):
        """Can register filters on partial template."""
        t = Template("{{ x | transform }} {{ y | transform }}")
        partial = t.partial(x=1)
        partial.register_filter("transform", lambda v: v * 10)
        assert partial(y=2) == "10 20"


class TestInputTypes:
    """Tests for various input types to custom filters."""

    def test_string_input(self, Template):
        """String input to filter."""
        t = Template("{{ x | reverse_str }}")
        t.register_filter("reverse_str", lambda s: s[::-1])
        assert t(x="hello") == "olleh"

    def test_integer_input(self, Template):
        """Integer input to filter."""
        t = Template("{{ x | square }}")
        t.register_filter("square", lambda n: n * n)
        assert t(x=5) == "25"

    def test_float_input(self, Template):
        """Float input to filter."""
        t = Template("{{ x | half }}")
        t.register_filter("half", lambda n: n / 2)
        assert t(x=10.0) == "5.0"

    def test_list_input(self, Template):
        """List input to filter."""
        t = Template("{{ items | double_all | join(', ') }}")
        t.register_filter("double_all", lambda lst: [x * 2 for x in lst])
        assert t(items=[1, 2, 3]) == "2, 4, 6"

    def test_dict_input(self, Template):
        """Dict input to filter."""
        t = Template("{{ data | get_keys | sort | join(', ') }}")
        t.register_filter("get_keys", lambda d: list(d.keys()))
        assert t(data={"a": 1, "b": 2}) == "a, b"

    def test_none_input(self, Template):
        """None input to filter."""
        t = Template("{{ x | or_default }}")
        t.register_filter("or_default", lambda v: v if v is not None else "N/A")
        assert t(x=None) == "N/A"
        assert t(x="value") == "value"

    def test_bool_input(self, Template):
        """Boolean input to filter."""
        t = Template("{{ x | negate }}")
        t.register_filter("negate", lambda b: not b)
        assert t(x=True) == "False"
        assert t(x=False) == "True"


class TestMixedBuiltinAndCustom:
    """Tests for using built-in and custom filters together."""

    def test_custom_then_builtin(self, Template):
        """Custom filter followed by built-in."""
        t = Template("{{ name | prefix('Hello ') | upper }}")
        t.register_filter("prefix", lambda s, p: p + s)
        assert t(name="world") == "HELLO WORLD"

    def test_builtin_then_custom(self, Template):
        """Built-in filter followed by custom."""
        t = Template("{{ name | upper | append('!') }}")
        t.register_filter("append", lambda s, suffix: s + suffix)
        assert t(name="hello") == "HELLO!"

    def test_interleaved_filters(self, Template):
        """Interleaved built-in and custom filters."""
        t = Template("{{ x | trim | double_str | upper }}")
        t.register_filter("double_str", lambda s: s + s)
        assert t(x="  hi  ") == "HIHI"


class TestErrorHandling:
    """Tests for error handling in custom filters."""

    def test_non_callable_raises_error(self, Template):
        """Registering non-callable raises ValidationError."""
        t = Template("{{ x }}")
        with pytest.raises(talu.ValidationError, match="callable"):
            t.register_filter("bad", "not a function")

    def test_filter_error_propagates(self, Template):
        """Errors in filter function propagate to render."""

        def bad_filter(x):
            raise ValueError("intentional error")

        t = Template("{{ x | bad_filter }}")
        t.register_filter("bad_filter", bad_filter)
        with pytest.raises(talu.TemplateError):
            t(x="test")


class TestRealWorldUseCases:
    """Real-world use case tests for custom filters."""

    def test_markdown_bold(self, Template):
        """Custom filter to make text bold in markdown."""
        t = Template("{{ title | bold }}")
        t.register_filter("bold", lambda s: f"**{s}**")
        assert t(title="Important") == "**Important**"

    def test_currency_format(self, Template):
        """Custom filter to format currency."""
        t = Template("Price: {{ amount | currency }}")
        t.register_filter("currency", lambda n: f"${n:.2f}")
        assert t(amount=42.5) == "Price: $42.50"

    def test_truncate_with_ellipsis(self, Template):
        """Custom truncate filter with ellipsis."""
        t = Template("{{ text | smart_truncate(10) }}")

        def smart_truncate(s, max_len):
            if len(s) <= max_len:
                return s
            return s[: max_len - 3] + "..."

        t.register_filter("smart_truncate", smart_truncate)
        assert t(text="Hello World!") == "Hello W..."
        assert t(text="Short") == "Short"

    def test_pluralize(self, Template):
        """Custom pluralize filter."""
        t = Template("{{ count }} {{ count | pluralize('item', 'items') }}")

        def pluralize(n, singular, plural):
            return singular if n == 1 else plural

        t.register_filter("pluralize", pluralize)
        assert t(count=1) == "1 item"
        assert t(count=5) == "5 items"

    def test_chat_message_role_format(self, Template):
        """Format chat message roles."""
        t = Template(
            "{% for msg in messages %}[{{ msg.role | role_emoji }}] {{ msg.content }}\n{% endfor %}"
        )
        t.register_filter(
            "role_emoji",
            lambda r: {
                "user": "\U0001f464",
                "assistant": "\U0001f916",
                "system": "\u2699\ufe0f",
            }.get(r, r),
        )
        result = t(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        )
        assert "\U0001f464" in result
        assert "\U0001f916" in result
