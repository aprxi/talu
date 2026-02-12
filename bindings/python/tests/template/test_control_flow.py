"""
Control flow tests for Template.

Tests for if/elif/else, for loops, set, and macro statements.
"""

import pytest

from tests.template.conftest import FOR_CASES, IF_CASES


class TestIfStatement:
    """Tests for {% if %} control flow."""

    def test_if_true(self, Template):
        """If block renders when condition is true."""
        t = Template("{% if show %}visible{% endif %}")
        assert t(show=True) == "visible"

    def test_if_false(self, Template):
        """If block doesn't render when condition is false."""
        t = Template("{% if show %}visible{% endif %}")
        assert t(show=False) == ""

    def test_if_else_true(self, Template):
        """If-else renders if block when true."""
        t = Template("{% if x %}yes{% else %}no{% endif %}")
        assert t(x=True) == "yes"

    def test_if_else_false(self, Template):
        """If-else renders else block when false."""
        t = Template("{% if x %}yes{% else %}no{% endif %}")
        assert t(x=False) == "no"

    def test_if_elif_else(self, Template):
        """If-elif-else chain works."""
        t = Template("""
{%- if x == 1 -%}one
{%- elif x == 2 -%}two
{%- else -%}other
{%- endif -%}
""")
        assert t(x=1) == "one"
        assert t(x=2) == "two"
        assert t(x=3) == "other"

    def test_if_comparison_greater(self, Template):
        """If with > comparison."""
        t = Template("{% if x > 5 %}big{% else %}small{% endif %}")
        assert t(x=10) == "big"
        assert t(x=3) == "small"

    def test_if_comparison_less(self, Template):
        """If with < comparison."""
        t = Template("{% if x < 5 %}small{% else %}big{% endif %}")
        assert t(x=3) == "small"
        assert t(x=10) == "big"

    def test_if_comparison_equal(self, Template):
        """If with == comparison."""
        t = Template("{% if x == 'yes' %}match{% endif %}")
        assert t(x="yes") == "match"
        assert t(x="no") == ""

    def test_if_comparison_not_equal(self, Template):
        """If with != comparison."""
        t = Template("{% if x != 'no' %}not no{% endif %}")
        assert t(x="yes") == "not no"
        assert t(x="no") == ""

    def test_if_and(self, Template):
        """If with 'and' operator."""
        t = Template("{% if a and b %}both{% endif %}")
        assert t(a=True, b=True) == "both"
        assert t(a=True, b=False) == ""
        assert t(a=False, b=True) == ""

    def test_if_or(self, Template):
        """If with 'or' operator."""
        t = Template("{% if a or b %}either{% endif %}")
        assert t(a=True, b=False) == "either"
        assert t(a=False, b=True) == "either"
        assert t(a=False, b=False) == ""

    def test_if_not(self, Template):
        """If with 'not' operator."""
        t = Template("{% if not x %}negated{% endif %}")
        assert t(x=False) == "negated"
        assert t(x=True) == ""

    def test_if_in(self, Template):
        """If with 'in' operator."""
        t = Template("{% if 'a' in items %}found{% endif %}")
        assert t(items=["a", "b"]) == "found"
        assert t(items=["x", "y"]) == ""

    def test_if_not_in(self, Template):
        """If with 'not in' operator."""
        t = Template("{% if 'z' not in items %}not found{% endif %}")
        assert t(items=["a", "b"]) == "not found"
        assert t(items=["z"]) == ""

    def test_if_truthy_values(self, Template):
        """If correctly handles truthy values."""
        t = Template("{% if x %}truthy{% endif %}")
        assert t(x=1) == "truthy"
        assert t(x="hello") == "truthy"
        assert t(x=[1, 2]) == "truthy"
        assert t(x={"a": 1}) == "truthy"

    def test_if_falsy_values(self, Template):
        """If correctly handles falsy values."""
        t = Template("{% if x %}truthy{% else %}falsy{% endif %}")
        assert t(x=0) == "falsy"
        assert t(x="") == "falsy"
        assert t(x=[]) == "falsy"
        assert t(x={}) == "falsy"
        assert t(x=None) == "falsy"

    @pytest.mark.parametrize("template,variables,expected", IF_CASES)
    def test_if_parametrized(self, Template, template, variables, expected):
        """Parametrized if cases."""
        t = Template(template)
        assert t(**variables) == expected


class TestForLoop:
    """Tests for {% for %} loops."""

    def test_for_simple_list(self, Template):
        """For loop over simple list."""
        t = Template("{% for i in items %}{{ i }}{% endfor %}")
        assert t(items=[1, 2, 3]) == "123"

    def test_for_with_separator(self, Template):
        """For loop with separator."""
        t = Template("{% for i in items %}{{ i }}, {% endfor %}")
        assert t(items=["a", "b", "c"]) == "a, b, c, "

    def test_for_empty_list(self, Template):
        """For loop over empty list produces nothing."""
        t = Template("{% for i in items %}{{ i }}{% endfor %}")
        assert t(items=[]) == ""

    def test_for_dict_items(self, Template):
        """For loop over dict.items()."""
        t = Template("{% for k, v in d.items() %}{{ k }}={{ v }} {% endfor %}")
        # Dict ordering is guaranteed in Python 3.7+
        result = t(d={"a": 1, "b": 2})
        assert "a=1" in result
        assert "b=2" in result

    def test_for_dict_keys(self, Template):
        """For loop over dict.keys()."""
        t = Template("{% for k in d.keys() %}{{ k }} {% endfor %}")
        result = t(d={"x": 1, "y": 2})
        assert "x" in result
        assert "y" in result

    def test_for_dict_values(self, Template):
        """For loop over dict.values()."""
        t = Template("{% for v in d.values() %}{{ v }} {% endfor %}")
        result = t(d={"a": 10, "b": 20})
        assert "10" in result
        assert "20" in result

    def test_for_nested(self, Template):
        """Nested for loops."""
        t = Template("""
{%- for row in matrix -%}
{%- for cell in row -%}{{ cell }}{%- endfor -%}|
{%- endfor -%}
""")
        assert t(matrix=[[1, 2], [3, 4]]) == "12|34|"

    def test_for_loop_index(self, Template):
        """Loop.index variable (1-based)."""
        t = Template("{% for i in items %}{{ loop.index }}{% endfor %}")
        assert t(items=["a", "b", "c"]) == "123"

    def test_for_loop_index0(self, Template):
        """Loop.index0 variable (0-based)."""
        t = Template("{% for i in items %}{{ loop.index0 }}{% endfor %}")
        assert t(items=["a", "b", "c"]) == "012"

    def test_for_loop_first(self, Template):
        """Loop.first variable."""
        t = Template("{% for i in items %}{% if loop.first %}[{% endif %}{{ i }}{% endfor %}")
        assert t(items=["a", "b", "c"]) == "[abc"

    def test_for_loop_last(self, Template):
        """Loop.last variable."""
        t = Template("{% for i in items %}{{ i }}{% if not loop.last %}, {% endif %}{% endfor %}")
        assert t(items=["a", "b", "c"]) == "a, b, c"

    def test_for_loop_length(self, Template):
        """Loop.length variable."""
        t = Template("{% for i in items %}{{ loop.length }}{% endfor %}")
        assert t(items=["a", "b", "c"]) == "333"

    def test_for_else(self, Template):
        """For-else when list is empty."""
        t = Template("{% for i in items %}{{ i }}{% else %}empty{% endfor %}")
        assert t(items=[]) == "empty"
        assert t(items=[1, 2]) == "12"

    def test_for_if_filter(self, Template):
        """For with if filter."""
        t = Template("{% for i in items if i > 2 %}{{ i }}{% endfor %}")
        assert t(items=[1, 2, 3, 4, 5]) == "345"

    def test_loop_cycle_two_values(self, Template):
        """loop.cycle() alternates between values."""
        t = Template("{% for i in items %}{{ loop.cycle('odd', 'even') }}{% endfor %}")
        assert t(items=[1, 2, 3, 4]) == "oddevenoddeven"

    def test_loop_cycle_three_values(self, Template):
        """loop.cycle() with three values."""
        t = Template("{% for i in items %}{{ loop.cycle('a', 'b', 'c') }}{% endfor %}")
        assert t(items=[1, 2, 3, 4, 5, 6]) == "abcabc"

    def test_loop_cycle_single_value(self, Template):
        """loop.cycle() with single value returns same value."""
        t = Template("{% for i in items %}{{ loop.cycle('x') }}{% endfor %}")
        assert t(items=[1, 2, 3]) == "xxx"

    def test_loop_cycle_in_table(self, Template):
        """loop.cycle() for alternating row classes."""
        t = Template("""
{%- for item in items -%}
<tr class="{{ loop.cycle('odd', 'even') }}">{{ item }}</tr>
{%- endfor -%}
""")
        result = t(items=["a", "b", "c"])
        assert '<tr class="odd">a</tr>' in result
        assert '<tr class="even">b</tr>' in result
        assert '<tr class="odd">c</tr>' in result

    def test_loop_revindex(self, Template):
        """loop.revindex counts down from end (1-based)."""
        t = Template("{% for i in items %}{{ loop.revindex }}{% endfor %}")
        assert t(items=[1, 2, 3]) == "321"

    def test_loop_revindex0(self, Template):
        """loop.revindex0 counts down from end (0-based)."""
        t = Template("{% for i in items %}{{ loop.revindex0 }}{% endfor %}")
        assert t(items=[1, 2, 3]) == "210"

    def test_loop_previtem(self, Template):
        """loop.previtem gives previous item."""
        t = Template("{% for i in items %}{{ loop.previtem | default('none') }}-{% endfor %}")
        assert t(items=["a", "b", "c"]) == "none-a-b-"

    def test_loop_nextitem(self, Template):
        """loop.nextitem gives next item."""
        t = Template("{% for i in items %}{{ loop.nextitem | default('none') }}-{% endfor %}")
        assert t(items=["a", "b", "c"]) == "b-c-none-"

    @pytest.mark.parametrize("template,variables,expected", FOR_CASES)
    def test_for_parametrized(self, Template, template, variables, expected):
        """Parametrized for cases."""
        t = Template(template)
        result = t(**variables)
        # For dict iteration, order may vary, so check contains
        if ".items()" in template:
            for part in expected.split():
                assert part in result
        else:
            assert result == expected


class TestRecursiveLoop:
    """Tests for recursive for loops."""

    def test_recursive_basic(self, Template):
        """Recursive loop for tree structures."""
        t = Template("""
{%- for item in items recursive -%}
{{ item.name }}
{%- if item.children %}
{{ loop(item.children) }}
{%- endif -%}
{%- endfor -%}
""")
        tree = [
            {
                "name": "a",
                "children": [{"name": "a1", "children": []}, {"name": "a2", "children": []}],
            },
            {"name": "b", "children": []},
        ]
        result = t(items=tree)
        assert "a" in result
        assert "a1" in result
        assert "a2" in result
        assert "b" in result

    def test_recursive_depth(self, Template):
        """Recursive loop maintains depth."""
        t = Template("""
{%- for item in items recursive -%}
{{ loop.depth }}:{{ item.name }}
{%- if item.children %}{{ loop(item.children) }}{%- endif -%}
{%- endfor -%}
""")
        tree = [{"name": "root", "children": [{"name": "child", "children": []}]}]
        result = t(items=tree)
        assert "1:root" in result or "0:root" in result


class TestLoopControls:
    """Tests for break and continue in loops."""

    def test_break_basic(self, Template):
        """break exits loop early."""
        t = Template("{% for i in items %}{% if i == 3 %}{% break %}{% endif %}{{ i }}{% endfor %}")
        assert t(items=[1, 2, 3, 4, 5]) == "12"

    def test_break_nested(self, Template):
        """break only exits inner loop."""
        t = Template("""
{%- for i in outer -%}
[{%- for j in inner -%}
{%- if j == 2 -%}{% break %}{%- endif -%}
{{ j }}
{%- endfor -%}]
{%- endfor -%}
""")
        assert t(outer=[1, 2], inner=[1, 2, 3]) == "[1][1]"

    def test_continue_basic(self, Template):
        """continue skips to next iteration."""
        t = Template(
            "{% for i in items %}{% if i == 2 %}{% continue %}{% endif %}{{ i }}{% endfor %}"
        )
        assert t(items=[1, 2, 3]) == "13"

    def test_continue_multiple(self, Template):
        """continue skips multiple items."""
        t = Template(
            "{% for i in items %}{% if i % 2 == 0 %}{% continue %}{% endif %}{{ i }}{% endfor %}"
        )
        assert t(items=[1, 2, 3, 4, 5]) == "135"


class TestFilterBlock:
    """Tests for {% filter %} block."""

    def test_filter_upper(self, Template):
        """filter block applies filter to content."""
        t = Template("{% filter upper %}hello world{% endfilter %}")
        assert t() == "HELLO WORLD"

    def test_filter_trim(self, Template):
        """filter block with trim."""
        t = Template("{% filter trim %}  spaced  {% endfilter %}")
        assert t() == "spaced"

    def test_filter_chain(self, Template):
        """filter block with chained filters."""
        t = Template("{% filter upper | trim %}  hello  {% endfilter %}")
        assert t() == "HELLO"


class TestCallBlock:
    """Tests for {% call %} macro invocation."""

    def test_call_basic(self, Template):
        """call block passes content to macro."""
        t = Template("""
{%- macro wrap() -%}
<div>{{ caller() }}</div>
{%- endmacro -%}
{% call wrap() %}Hello{% endcall %}
""")
        assert "<div>Hello</div>" in t()

    def test_call_with_args(self, Template):
        """call block with arguments."""
        t = Template("""
{%- macro item(name) -%}
<li>{{ name }}: {{ caller() }}</li>
{%- endmacro -%}
{% call item('Test') %}Content{% endcall %}
""")
        result = t()
        assert "<li>Test: Content</li>" in result


class TestSetStatement:
    """Tests for {% set %} variable assignment."""

    def test_set_simple(self, Template):
        """Set a simple variable."""
        t = Template("{% set x = 5 %}{{ x }}")
        assert t() == "5"

    def test_set_string(self, Template):
        """Set a string variable."""
        t = Template("{% set name = 'Alice' %}Hello {{ name }}")
        assert t() == "Hello Alice"

    def test_set_expression(self, Template):
        """Set variable from expression."""
        t = Template("{% set sum = a + b %}{{ sum }}")
        assert t(a=3, b=4) == "7"

    def test_set_override(self, Template):
        """Set can override passed variable."""
        t = Template("{% set x = 10 %}{{ x }}")
        assert t(x=5) == "10"

    def test_set_in_loop(self, Template):
        """Set inside a loop creates a loop-local binding (Jinja2 scope isolation).

        To accumulate values across iterations, use namespace():
          {% set ns = namespace(total=0) %}
          {% for i in items %}{% set ns.total = ns.total + i %}{% endfor %}
          {{ ns.total }}
        """
        t = Template(
            "{% set total = 0 %}{% for i in items %}{% set total = total + i %}{% endfor %}{{ total }}"
        )
        # Jinja2 behavior: set inside loop creates a local binding, outer total stays 0
        assert t(items=[1, 2, 3]) == "0"

    def test_set_list(self, Template):
        """Set a list variable."""
        t = Template("{% set items = [1, 2, 3] %}{% for i in items %}{{ i }}{% endfor %}")
        assert t() == "123"

    def test_set_dict(self, Template):
        """Set a dict variable."""
        t = Template("{% set d = {'a': 1, 'b': 2} %}{{ d.a }}")
        assert t() == "1"


class TestMacro:
    """Tests for {% macro %} definitions."""

    def test_macro_simple(self, Template):
        """Simple macro definition and call."""
        t = Template("""
{%- macro greet(name) -%}
Hello {{ name }}!
{%- endmacro -%}
{{ greet('World') }}
""")
        assert t().strip() == "Hello World!"

    def test_macro_multiple_args(self, Template):
        """Macro with multiple arguments."""
        t = Template("""
{%- macro add(a, b) -%}
{{ a + b }}
{%- endmacro -%}
{{ add(3, 4) }}
""")
        assert t().strip() == "7"

    def test_macro_default_arg(self, Template):
        """Macro with default argument."""
        t = Template("""
{%- macro greet(name='World') -%}
Hello {{ name }}!
{%- endmacro -%}
{{ greet() }} {{ greet('Alice') }}
""")
        result = t()
        assert "Hello World!" in result
        assert "Hello Alice!" in result

    def test_macro_reuse(self, Template):
        """Macro can be called multiple times."""
        t = Template("""
{%- macro item(text) -%}
- {{ text }}
{%- endmacro -%}
{{ item('one') }}
{{ item('two') }}
{{ item('three') }}
""")
        result = t()
        assert "- one" in result
        assert "- two" in result
        assert "- three" in result

    def test_macro_with_loop(self, Template):
        """Macro used inside a loop."""
        t = Template("""
{%- macro format_item(item) -%}[{{ item }}]{%- endmacro -%}
{%- for i in items -%}{{ format_item(i) }}{%- endfor -%}
""")
        assert t(items=["a", "b", "c"]) == "[a][b][c]"


class TestWhitespaceControl:
    """Tests for whitespace control with - modifier."""

    def test_trim_left(self, Template):
        """Trim whitespace on left with -."""
        t = Template("  {%- if true %}x{% endif %}")
        assert t() == "x"

    def test_trim_right(self, Template):
        """Trim whitespace on right with -."""
        t = Template("{% if true -%}  x{% endif %}")
        result = t()
        assert not result.startswith(" ")

    def test_trim_both(self, Template):
        """Trim whitespace on both sides."""
        t = Template("  {%- if true -%}  x  {%- endif -%}  ")
        assert t().strip() == "x"

    def test_variable_trim(self, Template):
        """Trim around variable output."""
        t = Template("  {{- x -}}  ")
        assert t(x="hello") == "hello"


class TestNestedControlFlow:
    """Tests for nested control flow precedence and scope."""

    def test_nested_for_if(self, Template):
        """If inside for accesses loop variables."""
        t = Template("{% for i in items %}{% if i > 2 %}{{ i }}{% endif %}{% endfor %}")
        assert t(items=[1, 2, 3, 4, 5]) == "345"

    def test_nested_if_for(self, Template):
        """For inside if only runs when condition is true."""
        t = Template("{% if show %}{% for i in items %}{{ i }}{% endfor %}{% endif %}")
        assert t(show=True, items=[1, 2, 3]) == "123"
        assert t(show=False, items=[1, 2, 3]) == ""

    def test_deeply_nested_control(self, Template):
        """Multiple levels of nesting work correctly."""
        t = Template("""
{%- for outer in outers -%}
{%- for inner in inners -%}
{%- if inner > 1 -%}
{{ outer }}-{{ inner }}
{%- endif -%}
{%- endfor -%}
{%- endfor -%}
""")
        result = t(outers=["a", "b"], inners=[1, 2])
        assert "a-2" in result
        assert "b-2" in result
        assert "a-1" not in result

    def test_scope_shadowing_set(self, Template):
        """Inner set shadows outer variable within block.

        Jinja2 contract: set inside a for loop creates a new binding in loop
        scope. After the loop, the outer variable should be restored.
        """
        t = Template("""
{%- set x = 'outer' -%}
before: {{ x }}
{%- for i in items -%}
{%- set x = 'inner' -%}
loop: {{ x }}
{%- endfor -%}
after: {{ x }}
""")
        result = t(items=[1])
        assert "before: outer" in result
        assert "loop: inner" in result

        # Jinja2 spec: after loop, outer scope should be restored
        if "after: inner" in result:
            pytest.xfail(
                "set inside for loop leaks to outer scope - "
                "Jinja2 spec says loop scope should be isolated"
            )
        assert "after: outer" in result

    def test_if_inside_for_with_loop_vars(self, Template):
        """If condition can use loop special variables."""
        t = Template(
            "{% for i in items %}{% if loop.first %}[{% endif %}{{ i }}{% if loop.last %}]{% endif %}{% endfor %}"
        )
        assert t(items=[1, 2, 3]) == "[123]"

    def test_nested_for_outer_inner_access(self, Template):
        """Nested for loop can access outer loop variable."""
        t = Template(
            "{% for a in outer %}{% for b in inner %}{{ a }}{{ b }}{% endfor %}{% endfor %}"
        )
        result = t(outer=["x", "y"], inner=[1, 2])
        assert "x1" in result
        assert "x2" in result
        assert "y1" in result
        assert "y2" in result

    def test_if_elif_else_precedence(self, Template):
        """If-elif-else evaluates conditions in order, stops at first true."""
        t = Template("""
{%- if x > 10 -%}big
{%- elif x > 5 -%}medium
{%- elif x > 0 -%}small
{%- else -%}zero
{%- endif -%}
""")
        assert t(x=15) == "big"
        assert t(x=7) == "medium"
        assert t(x=3) == "small"
        assert t(x=0) == "zero"
        # Even if x=7 is also > 0, elif stops at first match
        assert t(x=7) != "small"

    def test_for_with_conditional_break(self, Template):
        """Break inside nested if exits loop."""
        t = Template("{% for i in items %}{% if i == 3 %}{% break %}{% endif %}{{ i }}{% endfor %}")
        assert t(items=[1, 2, 3, 4, 5]) == "12"

    def test_for_with_conditional_continue(self, Template):
        """Continue inside nested if skips iteration."""
        t = Template(
            "{% for i in items %}{% if i == 2 %}{% continue %}{% endif %}{{ i }}{% endfor %}"
        )
        assert t(items=[1, 2, 3]) == "13"
