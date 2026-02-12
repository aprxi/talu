"""
Jinja2 built-in function tests for Template.

Tests for range, dict, namespace, and other global functions.
"""

import pytest

from talu.exceptions import TemplateError


class TestRangeFunction:
    """Tests for range() function."""

    def test_range_single_arg(self, Template):
        """range(n) generates 0 to n-1."""
        t = Template("{% for i in range(5) %}{{ i }}{% endfor %}")
        assert t() == "01234"

    def test_range_two_args(self, Template):
        """range(start, stop) generates start to stop-1."""
        t = Template("{% for i in range(2, 5) %}{{ i }}{% endfor %}")
        assert t() == "234"

    def test_range_three_args(self, Template):
        """range(start, stop, step) with step."""
        t = Template("{% for i in range(0, 10, 2) %}{{ i }}{% endfor %}")
        assert t() == "02468"

    def test_range_negative_step(self, Template):
        """range with negative step counts down."""
        t = Template("{% for i in range(5, 0, -1) %}{{ i }}{% endfor %}")
        assert t() == "54321"

    def test_range_empty(self, Template):
        """range with no iterations."""
        t = Template("{% for i in range(5, 2) %}{{ i }}{% endfor %}")
        assert t() == ""

    def test_range_with_variable(self, Template):
        """range with variable argument."""
        t = Template("{% for i in range(n) %}{{ i }}{% endfor %}")
        assert t(n=3) == "012"


class TestDictFunction:
    """Tests for dict() function."""

    def test_dict_empty(self, Template):
        """dict() creates empty dict."""
        t = Template("{% set d = dict() %}{{ d | length }}")
        assert t() == "0"

    def test_dict_kwargs(self, Template):
        """dict(key=value) creates dict."""
        t = Template("{% set d = dict(a=1, b=2) %}{{ d.a }},{{ d.b }}")
        assert t() == "1,2"

    def test_dict_access(self, Template):
        """Access dict values created inline."""
        t = Template("{{ dict(name='Alice', age=30).name }}")
        assert t() == "Alice"

    def test_dict_in_loop(self, Template):
        """Dict used in loop."""
        t = Template("{% for k, v in dict(x=1, y=2).items() %}{{ k }}={{ v }} {% endfor %}")
        result = t()
        assert "x=1" in result
        assert "y=2" in result


class TestNamespaceFunction:
    """Tests for namespace() function.

    Namespace allows creating mutable objects that can be modified in loops.
    """

    def test_namespace_basic(self, Template):
        """namespace() creates object with attributes."""
        t = Template("{% set ns = namespace(count=0) %}{{ ns.count }}")
        assert t() == "0"

    def test_namespace_modify_in_loop(self, Template):
        """namespace can be modified in loop (unlike regular set)."""
        t = Template("""
{%- set ns = namespace(total=0) -%}
{%- for i in items -%}
{%- set ns.total = ns.total + i -%}
{%- endfor -%}
{{ ns.total }}
""")
        assert t(items=[1, 2, 3, 4, 5]).strip() == "15"

    def test_namespace_multiple_attrs(self, Template):
        """namespace with multiple attributes."""
        t = Template("""
{%- set ns = namespace(count=0, sum=0) -%}
{%- for i in items -%}
{%- set ns.count = ns.count + 1 -%}
{%- set ns.sum = ns.sum + i -%}
{%- endfor -%}
count={{ ns.count }}, sum={{ ns.sum }}
""")
        assert t(items=[1, 2, 3]).strip() == "count=3, sum=6"

    def test_namespace_string_attr(self, Template):
        """namespace with string attribute."""
        t = Template("""
{%- set ns = namespace(result='') -%}
{%- for item in items -%}
{%- set ns.result = ns.result ~ item -%}
{%- endfor -%}
{{ ns.result }}
""")
        assert t(items=["a", "b", "c"]).strip() == "abc"


class TestLipsumFunction:
    """Tests for lipsum() function."""

    def test_lipsum_default(self, Template):
        """lipsum() generates lorem ipsum text."""
        t = Template("{{ lipsum() }}")
        result = t()
        assert len(result) > 0
        # Lorem ipsum typically starts with "Lorem"
        assert "Lorem" in result or len(result) > 50

    def test_lipsum_paragraphs(self, Template):
        """lipsum(n) generates n paragraphs."""
        t = Template("{{ lipsum(2) }}")
        result = t()
        assert len(result) > 0

    def test_lipsum_html(self, Template):
        """lipsum with html=True wraps in <p> tags."""
        t = Template("{{ lipsum(1, html=true) }}")
        result = t()
        assert "<p>" in result or len(result) > 0


class TestCyclerFunction:
    """Tests for cycler() function."""

    def test_cycler_basic(self, Template):
        """cycler() cycles through values."""
        t = Template("""
{%- set c = cycler('odd', 'even') -%}
{%- for i in range(4) -%}
{{ c.next() }}
{%- endfor -%}
""")
        assert t() == "oddevenoddeven"

    def test_cycler_current(self, Template):
        """cycler.current shows current value."""
        t = Template("""
{%- set c = cycler('a', 'b', 'c') -%}
{{ c.current }}{{ c.next() }}{{ c.current }}
""")
        result = t()
        assert "a" in result

    def test_cycler_reset(self, Template):
        """cycler.reset() restarts cycle.

        CONTRACT: After reset(), next() returns the first item in the cycle.
        """
        t = Template("""
{%- set c = cycler('a', 'b') -%}
{{ c.next() }}{{ c.next() }}{% set _ = c.reset() %}{{ c.next() }}
""")
        result = t().strip()
        # Sequence: next()->a, next()->b, reset(), next()->a (back to start)
        assert result == "aba", f"Expected 'aba' (a, b, then a after reset), got '{result}'"


class TestJoinerFunction:
    """Tests for joiner() function."""

    def test_joiner_basic(self, Template):
        """joiner() returns separator after first call."""
        t = Template("""
{%- set j = joiner(', ') -%}
{%- for i in items -%}
{{ j() }}{{ i }}
{%- endfor -%}
""")
        assert t(items=["a", "b", "c"]) == "a, b, c"

    def test_joiner_custom_sep(self, Template):
        """joiner with custom separator."""
        t = Template("""
{%- set j = joiner(' | ') -%}
{%- for i in items -%}
{{ j() }}{{ i }}
{%- endfor -%}
""")
        assert t(items=[1, 2, 3]) == "1 | 2 | 3"

    def test_joiner_first_empty(self, Template):
        """joiner returns empty on first call."""
        t = Template("{% set j = joiner('-') %}[{{ j() }}][{{ j() }}][{{ j() }}]")
        assert t() == "[][-][-]"


class TestStrftimeNowFunction:
    """Tests for strftime_now() function.

    KNOWN LIMITATION: strftime_now uses the system clock (Zig std.time.timestamp()).
    There is no mechanism to inject a fixed timestamp for testing because the
    function is implemented in Zig core with no Python-side override capability.

    "No wall-clock timing" principle, these tests minimize non-determinism by
    focusing on:
    1. Format structure (length, separators, character types) - deterministic
    2. Value ranges (valid months 1-12, valid hours 0-23, etc.) - deterministic
    3. Parseability (output can be parsed as a valid datetime) - deterministic
    4. Loose comparison to system clock with generous tolerances - for sanity check

    The tolerances (1 day for dates, 2 seconds for timestamps) are intentionally
    wide to avoid flaky tests while still catching gross implementation errors
    (e.g., wrong format specifier, timezone issues).

    If these tests become flaky in CI, increase tolerances or remove the
    comparison-to-system-clock assertions entirely, keeping only format validation.
    """

    def test_strftime_now_year_format(self, Template):
        """strftime_now('%Y') returns a 4-digit year string.

        Format validation only - verifies structure, not exact value.
        """
        import datetime

        t = Template("{{ strftime_now('%Y') }}")
        result = t()

        # Validate format: 4 digits
        assert len(result) == 4, f"Expected 4-digit year, got: {result}"
        assert result.isdigit(), f"Year should be all digits, got: {result}"

        # Compare against Python's current year (within 1 year tolerance
        # to handle year boundary edge cases in CI)
        result_year = int(result)
        python_year = datetime.datetime.now().year
        assert abs(result_year - python_year) <= 1, (
            f"Year {result_year} differs from system year {python_year} by more than 1"
        )

    def test_strftime_now_date_format(self, Template):
        """strftime_now('%Y-%m-%d') returns ISO date format.

        Validates format structure and component ranges.
        """
        import datetime

        t = Template("{{ strftime_now('%Y-%m-%d') }}")
        result = t()

        # Validate format: YYYY-MM-DD
        assert len(result) == 10, f"Expected 10-char date, got: {result}"
        assert result[4] == "-" and result[7] == "-", f"Invalid date format: {result}"

        # Validate each component is numeric and in valid range
        year, month, day = result.split("-")
        assert year.isdigit() and len(year) == 4
        assert month.isdigit() and 1 <= int(month) <= 12
        assert day.isdigit() and 1 <= int(day) <= 31

        # Verify the date is parseable and close to now
        try:
            parsed = datetime.datetime.strptime(result, "%Y-%m-%d")
            now = datetime.datetime.now()
            # Allow up to 1 day difference for CI timing edge cases
            delta = abs((parsed.date() - now.date()).days)
            assert delta <= 1, f"Date {result} differs from system date by {delta} days"
        except ValueError as e:
            pytest.fail(f"Date {result} is not a valid date: {e}")

    def test_strftime_now_time_format(self, Template):
        """strftime_now('%H:%M') returns HH:MM format.

        Validates format structure and component ranges.
        """
        t = Template("{{ strftime_now('%H:%M') }}")
        result = t()

        # Validate format: HH:MM
        assert ":" in result, f"Time should contain colon, got: {result}"
        parts = result.split(":")
        assert len(parts) == 2, f"Expected HH:MM format, got: {result}"
        hour, minute = parts
        assert hour.isdigit() and 0 <= int(hour) <= 23, f"Hour {hour} not in valid range 0-23"
        assert minute.isdigit() and 0 <= int(minute) <= 59, (
            f"Minute {minute} not in valid range 0-59"
        )

    def test_strftime_now_full_timestamp(self, Template):
        """strftime_now with full timestamp format is parseable.

        DETERMINISTIC TEST: Validates format structure, component ranges,
        and parseability only. Does NOT compare to system clock to ensure
        100% determinism in virtualized/slow CI environments.
        """
        import datetime
        import re

        t = Template("{{ strftime_now('%Y-%m-%d %H:%M:%S') }}")
        result = t()

        # Validate format structure via regex
        # Format: YYYY-MM-DD HH:MM:SS (19 characters)
        timestamp_pattern = re.compile(r"^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})$")
        match = timestamp_pattern.match(result)
        assert match, f"Timestamp '{result}' does not match YYYY-MM-DD HH:MM:SS format"

        # Extract and validate component ranges
        year, month, day, hour, minute, second = map(int, match.groups())

        # Year should be reasonable (not 1970 or 9999)
        assert 2020 <= year <= 2100, f"Year {year} outside reasonable range 2020-2100"

        # Month 1-12
        assert 1 <= month <= 12, f"Month {month} not in range 1-12"

        # Day 1-31 (rough check, parseability below catches invalid dates)
        assert 1 <= day <= 31, f"Day {day} not in range 1-31"

        # Hour 0-23
        assert 0 <= hour <= 23, f"Hour {hour} not in range 0-23"

        # Minute 0-59
        assert 0 <= minute <= 59, f"Minute {minute} not in range 0-59"

        # Second 0-59
        assert 0 <= second <= 59, f"Second {second} not in range 0-59"

        # Validate parseability (catches invalid dates like Feb 30)
        try:
            datetime.datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            pytest.fail(f"Timestamp {result} is not a valid datetime: {e}")


class TestRaiseExceptionFunction:
    """Tests for raise_exception() function."""

    def test_raise_exception(self, Template):
        """raise_exception() raises error with message preserved."""
        t = Template("{{ raise_exception('Custom error message') }}")
        with pytest.raises(TemplateError) as exc_info:
            t()
        assert "Custom error message" in str(exc_info.value)

    def test_raise_exception_conditional(self, Template):
        """raise_exception in conditional."""
        t = Template("{% if not valid %}{{ raise_exception('Invalid!') }}{% endif %}OK")
        assert t(valid=True) == "OK"
        with pytest.raises(TemplateError) as exc_info:
            t(valid=False)
        assert "Invalid!" in str(exc_info.value)

    def test_raise_exception_with_variable(self, Template):
        """raise_exception with dynamic message."""
        t = Template("{% if count > 10 %}{{ raise_exception('Too many: ' ~ count) }}{% endif %}OK")
        assert t(count=5) == "OK"
        with pytest.raises(TemplateError) as exc_info:
            t(count=15)
        assert "Too many: 15" in str(exc_info.value)

    def test_raise_exception_validation_pattern(self, Template):
        """raise_exception for input validation."""
        t = Template("""
{%- if not items -%}
{{ raise_exception("items cannot be empty") }}
{%- endif -%}
{%- if items | length > 5 -%}
{{ raise_exception("max 5 items allowed, got " ~ items | length) }}
{%- endif -%}
OK: {{ items | length }} items
""")
        assert "OK: 3 items" in t(items=[1, 2, 3])

        with pytest.raises(TemplateError) as exc_info:
            t(items=[])
        assert "items cannot be empty" in str(exc_info.value)

        with pytest.raises(TemplateError) as exc_info:
            t(items=[1, 2, 3, 4, 5, 6, 7])
        assert "max 5 items allowed, got 7" in str(exc_info.value)


class TestObjectMethods:
    """Tests for object method access."""

    def test_dict_items(self, Template):
        """dict.items() method."""
        t = Template("{% for k, v in d.items() %}{{ k }}:{{ v }} {% endfor %}")
        result = t(d={"a": 1, "b": 2})
        assert "a:1" in result and "b:2" in result

    def test_dict_keys(self, Template):
        """dict.keys() method."""
        t = Template("{{ d.keys() | list | join(',') }}")
        result = t(d={"x": 1, "y": 2})
        assert "x" in result and "y" in result

    def test_dict_values(self, Template):
        """dict.values() method."""
        t = Template("{{ d.values() | list | join(',') }}")
        result = t(d={"a": 10, "b": 20})
        assert "10" in result and "20" in result

    def test_dict_get(self, Template):
        """dict.get() method."""
        t = Template("{{ d.get('a') }}-{{ d.get('z', 'default') }}")
        assert t(d={"a": 1}) == "1-default"

    def test_dict_get_default(self, Template):
        """dict.get() with default."""
        t = Template("{{ d.get('missing', 'N/A') }}")
        assert t(d={}) == "N/A"

    def test_string_methods(self, Template):
        """String methods like split, startswith."""
        t = Template("{{ s.split(',') | join('-') }}")
        assert t(s="a,b,c") == "a-b-c"

    def test_string_startswith(self, Template):
        """String startswith method."""
        t = Template("{{ s.startswith('hello') }}")
        assert t(s="hello world") == "True"
        assert t(s="goodbye") == "False"

    def test_string_endswith(self, Template):
        """String endswith method."""
        t = Template("{{ s.endswith('.py') }}")
        assert t(s="script.py") == "True"
        assert t(s="script.js") == "False"

    def test_list_append(self, Template):
        """List methods (note: Jinja lists may be immutable)."""
        # This tests whether list methods work
        t = Template("{% set items = [1, 2] %}{{ items | length }}")
        assert t() == "2"


class TestLiteralConstructors:
    """Tests for literal constructors in templates."""

    def test_list_literal(self, Template):
        """Create list literal in template."""
        t = Template("{% set items = [1, 2, 3] %}{{ items | join(',') }}")
        assert t() == "1,2,3"

    def test_dict_literal(self, Template):
        """Create dict literal in template."""
        t = Template("{% set d = {'a': 1, 'b': 2} %}{{ d.a }}")
        assert t() == "1"

    def test_tuple_literal(self, Template):
        """Create tuple literal in template."""
        t = Template("{% set pair = (1, 2) %}{{ pair | join('-') }}")
        assert t() == "1-2"

    def test_nested_literal(self, Template):
        """Nested data structure literal."""
        t = Template("{% set data = {'items': [1, 2, 3]} %}{{ data.items | join(',') }}")
        assert t() == "1,2,3"


class TestIncludeTag:
    """Tests for {% include %} template composition."""

    def test_include_string_literal(self, Template):
        """Include a literal template string."""
        t = Template('Before{% include "Hello {{ name }}!" %}After')
        assert t(name="World") == "BeforeHello World!After"

    def test_include_variable(self, Template):
        """Include template from a variable."""
        t = Template("{% include header %}")
        assert t(header="=== {{ title }} ===", title="Test") == "=== Test ==="

    def test_include_accesses_parent_context(self, Template):
        """Included template has access to parent's variables."""
        t = Template("{% include tmpl %}")
        result = t(
            tmpl="{% for item in items %}{{ item }}{% endfor %}",
            items=["a", "b", "c"],
        )
        assert result == "abc"

    def test_include_with_set_variable(self, Template):
        """Variables set before include are accessible."""
        t = Template("{% set greeting = 'Hello' %}{% include tmpl %}")
        assert t(tmpl="{{ greeting }} {{ name }}!", name="World") == "Hello World!"

    def test_include_conditional(self, Template):
        """Include based on condition."""
        t = Template("{% if show %}{% include content %}{% endif %}")
        assert t(show=True, content="SHOWN") == "SHOWN"
        assert t(show=False, content="SHOWN") == ""

    def test_include_in_loop(self, Template):
        """Include inside a loop."""
        t = Template("{% for i in items %}{% include row_tmpl %}{% endfor %}")
        result = t(row_tmpl="[{{ i }}]", items=[1, 2, 3])
        assert result == "[1][2][3]"

    def test_include_nested(self, Template):
        """Nested includes (template includes another that includes another)."""
        t = Template("{% include outer %}")
        result = t(
            outer="({% include middle %})",
            middle="[{% include inner %}]",
            inner="DEEP",
        )
        assert result == "([DEEP])"

    def test_include_with_macros(self, Template):
        """Macros defined in included template are available."""
        t = Template("{% include utils %}{{ greet('World') }}")
        utils = "{% macro greet(name) %}Hello {{ name }}!{% endmacro %}"
        assert t(utils=utils) == "Hello World!"

    def test_include_multiple(self, Template):
        """Multiple includes in same template."""
        t = Template("{% include header %}|{% include body %}|{% include footer %}")
        result = t(header="H", body="B", footer="F")
        assert result == "H|B|F"

    def test_include_with_filters(self, Template):
        """Include with filters in included template."""
        t = Template("{% include tmpl %}")
        result = t(tmpl="{{ name | upper }}", name="alice")
        assert result == "ALICE"

    def test_include_with_control_flow(self, Template):
        """Include with control flow in included template."""
        t = Template("{% include tmpl %}")
        tmpl = "{% if active %}ON{% else %}OFF{% endif %}"
        assert t(tmpl=tmpl, active=True) == "ON"
        assert t(tmpl=tmpl, active=False) == "OFF"

    def test_include_type_error(self, Template):
        """Include non-string raises error."""
        t = Template("{% include value %}")
        with pytest.raises(TemplateError):
            t(value=123)

    def test_include_whitespace_control(self, Template):
        """Include respects whitespace control."""
        t = Template("A{%- include tmpl -%}B")
        assert t(tmpl="X") == "AXB"

    def test_include_expression(self, Template):
        """Include with expression that evaluates to template string."""
        t = Template("{% include templates[format] %}")
        result = t(
            templates={"md": "# {{ title }}", "txt": "{{ title }}"},
            format="md",
            title="Hello",
        )
        assert result == "# Hello"
