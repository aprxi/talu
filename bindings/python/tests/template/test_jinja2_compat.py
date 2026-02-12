"""
Jinja2 Compatibility Tests.

Tests that talu's Zig template engine produces identical output to Python's Jinja2.
Each test covers a specific Jinja2 feature found in real HuggingFace chat templates.

Feature coverage is based on analysis of 339+ real model templates.
Run: python scripts/fetch_chat_templates.py --analyze

To add new tests:
1. Run the fetch script to update feature analysis
2. Find features without test coverage
3. Add a test method for each missing feature
"""

import pytest

jinja2 = pytest.importorskip("jinja2")

from talu import PromptTemplate  # noqa: E402


def render_jinja2(template_str: str, **variables) -> str:
    """Render template using Python's Jinja2 with HF-compatible settings."""
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(jinja2.TemplateError(msg))
    # Add strftime_now for templates that use it
    import datetime

    env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
    return env.from_string(template_str).render(**variables)


def render_talu(template_str: str, strict: bool = False, **variables) -> str:
    """Render template using talu's Zig engine.

    Args:
        template_str: The Jinja2 template string to test
        strict: If True, undefined variables raise errors. Default False for
               Jinja2 compatibility (lenient mode matches Jinja2 default behavior).
        **variables: Template variables
    """
    return PromptTemplate(template_str, strict=strict)(**variables)


def assert_compat(template_str: str, normalize_quotes: bool = False, **variables):
    """Assert both engines produce identical output.

    Uses lenient mode (strict=False) to match Jinja2's default undefined behavior.

    Args:
        template_str: The Jinja2 template string to test
        normalize_quotes: If True, normalize single quotes to double quotes before comparison.
                         Use this for tests that output arrays/objects where Python uses
                         single quotes but JSON uses double quotes.
        **variables: Template variables
    """
    jinja_result = render_jinja2(template_str, **variables)
    talu_result = render_talu(template_str, strict=False, **variables)

    if normalize_quotes:
        # Normalize Python-style single quotes to JSON-style double quotes
        jinja_result = jinja_result.replace("'", '"')

    assert talu_result == jinja_result, (
        f"Output mismatch:\n  Jinja2: {jinja_result!r}\n  Talu:   {talu_result!r}"
    )


# =============================================================================
# Test Data
# =============================================================================

SIMPLE_MSG = [{"role": "user", "content": "Hello!"}]
MULTI_TURN = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "And 3+3?"},
]
WITH_SYSTEM = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"},
]


# =============================================================================
# Control Flow (99% of templates)
# =============================================================================


class TestControlFlow:
    """Test control flow statements: for, if, elif, else."""

    def test_for_loop(self):
        """for_loop: 99% of templates."""
        template = "{% for x in items %}{{ x }}{% endfor %}"
        assert_compat(template, items=["a", "b", "c"])

    def test_for_loop_empty(self):
        """for loop with empty list."""
        template = "{% for x in items %}{{ x }}{% endfor %}"
        assert_compat(template, items=[])

    def test_if_statement(self):
        """if_statement: 99% of templates."""
        template = "{% if x %}yes{% endif %}"
        assert_compat(template, x=True)
        assert_compat(template, x=False)

    def test_if_else(self):
        """else: 16% of templates."""
        template = "{% if x %}yes{% else %}no{% endif %}"
        assert_compat(template, x=True)
        assert_compat(template, x=False)

    def test_elif(self):
        """elif: 20% of templates."""
        template = "{% if x == 1 %}one{% elif x == 2 %}two{% else %}other{% endif %}"
        assert_compat(template, x=1)
        assert_compat(template, x=2)
        assert_compat(template, x=3)

    def test_nested_if_in_for(self):
        """Nested if inside for loop."""
        template = "{% for m in msgs %}{% if m.role == 'user' %}U{% else %}A{% endif %}{% endfor %}"
        assert_compat(template, msgs=MULTI_TURN)

    def test_for_if_filter(self):
        """for_if_filter: 2% of templates - inline if in for."""
        template = "{% for x in items if x > 1 %}{{ x }}{% endfor %}"
        assert_compat(template, items=[1, 2, 3])


# =============================================================================
# Loop Variables (38% of templates)
# =============================================================================


class TestLoopVariables:
    """Test loop.* variables used in for loops."""

    def test_loop_index(self):
        """loop_index: 12% of templates (1-based)."""
        template = "{% for x in items %}{{ loop.index }}{% endfor %}"
        assert_compat(template, items=["a", "b", "c"])

    def test_loop_index0(self):
        """loop_index0: 38% of templates (0-based)."""
        template = "{% for x in items %}{{ loop.index0 }}{% endfor %}"
        assert_compat(template, items=["a", "b", "c"])

    def test_loop_first(self):
        """loop_first: 28% of templates."""
        template = "{% for x in items %}{% if loop.first %}F{% endif %}{{ x }}{% endfor %}"
        assert_compat(template, items=["a", "b", "c"])

    def test_loop_last(self):
        """loop_last: 38% of templates."""
        template = "{% for x in items %}{{ x }}{% if loop.last %}L{% endif %}{% endfor %}"
        assert_compat(template, items=["a", "b", "c"])

    def test_loop_length(self):
        """loop.length - total items."""
        template = "{% for x in items %}{{ loop.length }}{% endfor %}"
        assert_compat(template, items=["a", "b", "c"])

    def test_loop_revindex(self):
        """loop.revindex - reverse index (1-based)."""
        template = "{% for x in items %}{{ loop.revindex }}{% endfor %}"
        assert_compat(template, items=["a", "b", "c"])

    def test_loop_revindex0(self):
        """loop.revindex0 - reverse index (0-based)."""
        template = "{% for x in items %}{{ loop.revindex0 }}{% endfor %}"
        assert_compat(template, items=["a", "b", "c"])

    def test_loop_previtem(self):
        """loop_previtem: 2% of templates."""
        template = (
            "{% for x in items %}{% if loop.previtem %}{{ loop.previtem }}{% endif %}{% endfor %}"
        )
        assert_compat(template, items=["a", "b", "c"])

    def test_loop_nextitem(self):
        """loop_nextitem: 2% of templates."""
        template = (
            "{% for x in items %}{% if loop.nextitem %}{{ loop.nextitem }}{% endif %}{% endfor %}"
        )
        assert_compat(template, items=["a", "b", "c"])


# =============================================================================
# Assignment (71% of templates)
# =============================================================================


class TestAssignment:
    """Test set statement and namespace."""

    def test_set_statement(self):
        """set_statement: 71% of templates."""
        template = "{% set x = 'hello' %}{{ x }}"
        assert_compat(template)

    def test_set_multiple(self):
        """Multiple set statements."""
        template = "{% set a = 1 %}{% set b = 2 %}{{ a + b }}"
        assert_compat(template)

    def test_set_from_expression(self):
        """Set with expression."""
        template = "{% set total = items | length %}{{ total }}"
        assert_compat(template, items=[1, 2, 3])

    def test_namespace(self):
        """func_namespace: 24% of templates."""
        template = "{% set ns = namespace(count=0) %}{% for x in items %}{% set ns.count = ns.count + 1 %}{% endfor %}{{ ns.count }}"
        assert_compat(template, items=["a", "b", "c"])

    def test_namespace_boolean(self):
        """Namespace with boolean tracking."""
        template = """{% set ns = namespace(found=false) %}{% for m in msgs %}{% if m.role == 'system' %}{% set ns.found = true %}{% endif %}{% endfor %}{{ ns.found }}"""
        assert_compat(template, msgs=WITH_SYSTEM)

    def test_namespace_created_inside_for_loop(self):
        """Namespace created inside a for loop (Granite 4.0 Hybrid pattern).

        Regression test: namespace created inside a for loop must be accessible
        for modification via local scopes. The Granite 4.0 Hybrid chat template
        uses this pattern extensively.
        """
        template = """{%- for message in messages %}
{%- set content = namespace(val='') %}
{%- if message.content is string %}
{%- set content.val = message.content %}
{%- endif %}
{{ message.role }}: {{ content.val }}
{% endfor %}"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        assert_compat(template, messages=messages)


# =============================================================================
# Operators (66% of templates)
# =============================================================================


class TestOperators:
    """Test arithmetic, comparison, and logical operators."""

    def test_op_add(self):
        """op_add: 66% of templates."""
        template = "{{ a + b }}"
        assert_compat(template, a=1, b=2)
        # String concatenation
        template2 = "{{ 'Hello' + ' ' + 'World' }}"
        assert_compat(template2)

    def test_op_sub(self):
        """op_sub: 15% of templates."""
        template = "{{ a - b }}"
        assert_compat(template, a=5, b=3)

    def test_op_mul(self):
        """op_mul: <1% of templates."""
        template = "{{ a * b }}"
        assert_compat(template, a=3, b=4)

    def test_op_mod(self):
        """op_mod: 5% of templates."""
        template = "{{ a % b }}"
        assert_compat(template, a=7, b=3)

    def test_floor_div(self):
        """floor_div_operator: 1% of templates."""
        template = "{{ a // b }}"
        assert_compat(template, a=7, b=3)

    def test_op_neg(self):
        """op_neg: 10% of templates."""
        template = "{{ -x }}"
        assert_compat(template, x=5)

    def test_op_and(self):
        """op_and: 42% of templates."""
        template = "{% if a and b %}yes{% endif %}"
        assert_compat(template, a=True, b=True)
        assert_compat(template, a=True, b=False)

    def test_op_or(self):
        """op_or: 28% of templates."""
        template = "{% if a or b %}yes{% endif %}"
        assert_compat(template, a=False, b=True)
        assert_compat(template, a=False, b=False)

    def test_op_not(self):
        """op_not: 42% of templates."""
        template = "{% if not x %}yes{% endif %}"
        assert_compat(template, x=False)
        assert_compat(template, x=True)

    def test_comparison_eq(self):
        """comparison: 58% of templates - equality."""
        template = "{% if x == 'user' %}yes{% endif %}"
        assert_compat(template, x="user")
        assert_compat(template, x="other")

    def test_comparison_neq(self):
        """Not equal comparison."""
        template = "{% if x != 'user' %}yes{% endif %}"
        assert_compat(template, x="other")

    def test_comparison_lt_gt(self):
        """Less than / greater than."""
        template = "{% if x > 5 %}big{% elif x < 3 %}small{% else %}medium{% endif %}"
        assert_compat(template, x=10)
        assert_compat(template, x=1)
        assert_compat(template, x=4)

    def test_comparison_lte_gte(self):
        """Less/greater than or equal."""
        template = "{% if x >= 5 %}yes{% endif %}{% if y <= 3 %}yes{% endif %}"
        assert_compat(template, x=5, y=3)

    def test_in_operator(self):
        """in_operator: 33% of templates."""
        template = "{% if 'a' in items %}yes{% endif %}"
        assert_compat(template, items=["a", "b"])
        assert_compat(template, items=["c", "d"])

    def test_not_in_operator(self):
        """not_in_operator: <1% of templates."""
        template = "{% if 'x' not in items %}yes{% endif %}"
        assert_compat(template, items=["a", "b"])

    def test_concat_operator(self):
        """op_concat / concat_operator: 2% of templates - tilde operator."""
        template = "{{ 'Hello' ~ ' ' ~ name }}"
        assert_compat(template, name="World")


# =============================================================================
# Access Patterns (66% of templates)
# =============================================================================


class TestAccessPatterns:
    """Test attribute access, bracket access, slicing."""

    def test_getitem_bracket(self):
        """getitem/bracket_access: 66% of templates."""
        template = "{{ msg['role'] }}: {{ msg['content'] }}"
        assert_compat(template, msg={"role": "user", "content": "Hi"})

    def test_getattr_dot(self):
        """Dot notation access."""
        template = "{{ msg.role }}: {{ msg.content }}"
        assert_compat(template, msg={"role": "user", "content": "Hi"})

    def test_slice(self):
        """slice: 24% of templates."""
        # Use join to avoid repr differences (single vs double quotes)
        template = "{{ items[1:] | join(',') }}"
        assert_compat(template, items=["a", "b", "c"])

    def test_slice_start(self):
        """Slice with start only."""
        template = "{{ items[1:] | join(',') }}"
        assert_compat(template, items=["a", "b", "c"])

    def test_slice_end(self):
        """Slice with end only."""
        template = "{{ items[:2] | join(',') }}"
        assert_compat(template, items=["a", "b", "c"])

    def test_slice_both(self):
        """Slice with start and end."""
        template = "{{ items[1:3] | join(',') }}"
        assert_compat(template, items=["a", "b", "c", "d"])

    def test_negative_index(self):
        """negative_index: 11% of templates."""
        template = "{{ items[-1] }}"
        assert_compat(template, items=["a", "b", "c"])

    def test_negative_index_slice(self):
        """Negative index in slice."""
        template = "{{ items[:-1] | join(',') }}"
        assert_compat(template, items=["a", "b", "c"])


# =============================================================================
# Tests (is X) - 48% of templates
# =============================================================================


class TestTests:
    """Test Jinja2 'is' tests."""

    def test_is_defined(self):
        """test_defined: 48% of templates."""
        template = "{% if x is defined %}yes{% else %}no{% endif %}"
        assert_compat(template, x="value")
        assert_compat(template)  # x not passed

    def test_is_undefined(self):
        """is undefined test."""
        template = "{% if x is undefined %}yes{% else %}no{% endif %}"
        assert_compat(template)
        assert_compat(template, x="value")

    def test_is_none(self):
        """test_none: 19% of templates."""
        template = "{% if x is none %}yes{% else %}no{% endif %}"
        assert_compat(template, x=None)
        assert_compat(template, x="value")

    def test_is_not_none(self):
        """test_not_none: <1% of templates."""
        # Note: 'is not none' syntax
        template = "{% if x is not none %}yes{% else %}no{% endif %}"
        assert_compat(template, x="value")
        assert_compat(template, x=None)

    def test_is_string(self):
        """test_string: 20% of templates."""
        template = "{% if x is string %}yes{% else %}no{% endif %}"
        assert_compat(template, x="hello")
        assert_compat(template, x=123)

    def test_is_mapping(self):
        """test_mapping: 9% of templates."""
        template = "{% if x is mapping %}yes{% else %}no{% endif %}"
        assert_compat(template, x={"a": 1})
        assert_compat(template, x=[1, 2])

    def test_is_iterable(self):
        """test_iterable: 10% of templates."""
        template = "{% if x is iterable %}yes{% else %}no{% endif %}"
        assert_compat(template, x=[1, 2])
        assert_compat(template, x="abc")
        assert_compat(template, x=123)

    def test_is_sequence(self):
        """test_sequence: 2% of templates."""
        template = "{% if x is sequence %}yes{% else %}no{% endif %}"
        assert_compat(template, x=[1, 2])
        assert_compat(template, x="abc")

    def test_is_false(self):
        """test_false: 5% of templates."""
        template = "{% if x is false %}yes{% else %}no{% endif %}"
        assert_compat(template, x=False)
        assert_compat(template, x=True)

    def test_is_true(self):
        """is true test."""
        template = "{% if x is true %}yes{% else %}no{% endif %}"
        assert_compat(template, x=True)
        assert_compat(template, x=False)

    def test_is_equalto(self):
        """test_equalto: <1% of templates - used with selectattr."""
        template = "{% if x is equalto(5) %}yes{% else %}no{% endif %}"
        assert_compat(template, x=5)
        assert_compat(template, x=3)


# =============================================================================
# Filters (35% use tojson, 20% use length, etc.)
# =============================================================================


class TestFilters:
    """Test Jinja2 filters."""

    def test_filter_length(self):
        """filter_length: 20% of templates."""
        template = "{{ items | length }}"
        assert_compat(template, items=[1, 2, 3])
        assert_compat(template, items="hello")

    def test_filter_tojson(self):
        """filter_tojson: 35% of templates."""
        template = "{{ data | tojson }}"
        # Use simple types to avoid ordering issues
        assert_compat(template, data=[1, 2, 3])
        assert_compat(template, data="hello")
        assert_compat(template, data={"key": "value"})

    def test_filter_trim(self):
        """filter_trim: 18% of templates."""
        template = "{{ text | trim }}"
        assert_compat(template, text="  hello  ")

    def test_filter_join(self):
        """filter_join: 5% of templates."""
        template = "{{ items | join(', ') }}"
        assert_compat(template, items=["a", "b", "c"])

    def test_filter_first(self):
        """filter_first: 3% of templates."""
        template = "{{ items | first }}"
        assert_compat(template, items=["a", "b", "c"])

    def test_filter_last(self):
        """filter_last: 3% of templates."""
        template = "{{ items | last }}"
        assert_compat(template, items=["a", "b", "c"])

    def test_filter_list(self):
        """filter_list: 2% of templates."""
        template = "{{ items | list | length }}"
        assert_compat(template, items=["a", "b"])

    def test_filter_string(self):
        """filter_string: 5% of templates."""
        template = "{{ x | string }}"
        assert_compat(template, x=123)

    def test_filter_default(self):
        """filter_default: 1% of templates."""
        template = "{{ x | default('fallback') }}"
        assert_compat(template, x="value")
        # Note: Jinja2 and talu may differ on None handling

    def test_filter_items(self):
        """filter_items: 5% of templates."""
        # Use single-key dict to avoid ordering differences
        template = "{% for k, v in data | items %}{{ k }}={{ v }}{% endfor %}"
        assert_compat(template, data={"key": "value"})

    def test_filter_selectattr(self):
        """filter_selectattr: 2% of templates."""
        template = "{{ msgs | selectattr('role', 'equalto', 'user') | list | length }}"
        assert_compat(template, msgs=MULTI_TURN)

    def test_filter_reject(self):
        """filter_reject: 3% of templates."""
        template = "{{ items | reject('equalto', 2) | list | join(',') }}"
        assert_compat(template, items=[1, 2, 3])

    def test_filter_map(self):
        """filter_map: <1% of templates."""
        # map with attribute extraction (more common in real templates)
        template = "{{ items | map(attribute='name') | list | join(',') }}"
        assert_compat(template, items=[{"name": "A"}, {"name": "B"}])

    def test_filter_replace(self):
        """filter_replace: <1% of templates."""
        template = "{{ text | replace('old', 'new') }}"
        assert_compat(template, text="old value old")

    def test_filter_unique(self):
        """filter_unique: <1% of templates."""
        template = "{{ items | unique | list | join(',') }}"
        assert_compat(template, items=["a", "b", "a", "c"])

    def test_filter_safe(self):
        """filter_safe: 2% of templates."""
        template = "{{ html | safe }}"
        assert_compat(template, html="<b>bold</b>")

    def test_filter_upper(self):
        """filter_upper: <1% of templates."""
        template = "{{ text | upper }}"
        assert_compat(template, text="hello")

    def test_filter_lower(self):
        """filter_lower."""
        template = "{{ text | lower }}"
        assert_compat(template, text="HELLO")

    def test_filter_capitalize(self):
        """filter_capitalize: <1% of templates."""
        template = "{{ text | capitalize }}"
        assert_compat(template, text="hello world")

    def test_filter_title(self):
        """filter_title: <1% of templates."""
        template = "{{ text | title }}"
        assert_compat(template, text="hello world")


# =============================================================================
# String Methods (14% use split)
# =============================================================================


class TestStringMethods:
    """Test string method calls."""

    def test_method_split(self):
        """method_split: 14% of templates."""
        template = "{{ text.split(',') | join('-') }}"
        assert_compat(template, text="a,b,c")

    def test_method_strip(self):
        """method_strip: 10% of templates."""
        template = "{{ text.strip() }}"
        assert_compat(template, text="  hello  ")

    def test_method_lstrip(self):
        """method_lstrip: 8% of templates."""
        template = "{{ text.lstrip() }}"
        assert_compat(template, text="  hello  ")

    def test_method_rstrip(self):
        """method_rstrip: 7% of templates."""
        template = "{{ text.rstrip() }}"
        assert_compat(template, text="  hello  ")

    def test_method_startswith(self):
        """method_startswith: 4% of templates."""
        template = "{% if text.startswith('hello') %}yes{% endif %}"
        assert_compat(template, text="hello world")
        assert_compat(template, text="world hello")

    def test_method_endswith(self):
        """method_endswith: 4% of templates."""
        template = "{% if text.endswith('world') %}yes{% endif %}"
        assert_compat(template, text="hello world")
        assert_compat(template, text="world hello")

    def test_method_replace(self):
        """method_replace: <1% of templates."""
        template = "{{ text.replace('old', 'new') }}"
        assert_compat(template, text="old value")

    def test_method_upper(self):
        """String upper method."""
        template = "{{ text.upper() }}"
        assert_compat(template, text="hello")

    def test_method_lower(self):
        """String lower method."""
        template = "{{ text.lower() }}"
        assert_compat(template, text="HELLO")

    def test_method_title(self):
        """method_title: <1% of templates."""
        template = "{{ text.title() }}"
        assert_compat(template, text="hello world")


# =============================================================================
# Dict/Object Methods (2% of templates)
# =============================================================================


class TestDictMethods:
    """Test dictionary method calls."""

    def test_method_items(self):
        """method_items: 2% of templates."""
        # Use single-key dict to avoid ordering differences
        template = "{% for k, v in data.items() %}{{ k }}={{ v }}{% endfor %}"
        assert_compat(template, data={"key": "value"})

    def test_method_keys(self):
        """Dict keys method."""
        # Use single-key dict to avoid ordering differences
        template = "{{ data.keys() | list | join(',') }}"
        assert_compat(template, data={"key": 1})

    def test_method_values(self):
        """Dict values method."""
        # Use single-key dict to avoid ordering differences
        template = "{{ data.values() | list | join(',') }}"
        assert_compat(template, data={"key": "value"})

    def test_method_get(self):
        """method_get: <1% of templates."""
        template = "{{ data.get('key', 'default') }}"
        assert_compat(template, data={"key": "value"})
        assert_compat(template, data={})


# =============================================================================
# Whitespace Control (37% of templates)
# =============================================================================


class TestWhitespaceControl:
    """Test whitespace control with - modifier."""

    def test_whitespace_control_block(self):
        """whitespace_control_block: 37% of templates."""
        template = "{%- for x in items -%}{{ x }}{%- endfor -%}"
        assert_compat(template, items=["a", "b"])

    def test_whitespace_control_output(self):
        """whitespace_control_output: 30% of templates."""
        template = "Hello {{- name -}} World"
        assert_compat(template, name="X")

    def test_mixed_whitespace_control(self):
        """Mixed whitespace control."""
        template = """
{%- for m in msgs %}
{{ m.content }}
{%- endfor %}"""
        assert_compat(template, msgs=SIMPLE_MSG)


# =============================================================================
# Literals (8% of templates)
# =============================================================================


class TestLiterals:
    """Test literal expressions."""

    def test_tuple_literal(self):
        """tuple_literal: 8% of templates."""
        template = "{% for x in (1, 2, 3) %}{{ x }}{% endfor %}"
        assert_compat(template)

    def test_list_literal(self):
        """list_literal: 4% of templates."""
        template = "{% for x in [1, 2, 3] %}{{ x }}{% endfor %}"
        assert_compat(template)

    def test_dict_literal(self):
        """dict_literal: 1% of templates."""
        template = "{% set d = {'a': 1} %}{{ d.a }}"
        assert_compat(template)

    def test_string_literal(self):
        """String literals with quotes."""
        template = "{{ 'single' }} {{ \"double\" }}"
        assert_compat(template)


# =============================================================================
# Ternary / Conditional Expression (10% of templates)
# =============================================================================


class TestTernary:
    """Test ternary/conditional expressions."""

    def test_ternary(self):
        """ternary: 10% of templates."""
        template = "{{ 'yes' if x else 'no' }}"
        assert_compat(template, x=True)
        assert_compat(template, x=False)

    def test_ternary_in_string(self):
        """Ternary inside string concatenation."""
        template = "{{ 'Value: ' + ('yes' if x else 'no') }}"
        assert_compat(template, x=True)


# =============================================================================
# Macros (5% of templates)
# =============================================================================


class TestMacros:
    """Test macro definitions and calls."""

    def test_macro_simple(self):
        """macro: 5% of templates."""
        template = "{% macro greet(name) %}Hello {{ name }}{% endmacro %}{{ greet('World') }}"
        assert_compat(template)

    def test_macro_with_default(self):
        """Macro with default argument."""
        template = "{% macro greet(name='Guest') %}Hello {{ name }}{% endmacro %}{{ greet() }}"
        assert_compat(template)


# =============================================================================
# Functions (24% use namespace, 19% use raise_exception)
# =============================================================================


class TestFunctions:
    """Test built-in functions."""

    def test_func_range(self):
        """func_range: <1% of templates."""
        template = "{% for i in range(3) %}{{ i }}{% endfor %}"
        assert_compat(template)

    def test_func_range_with_start(self):
        """Range with start and end."""
        template = "{% for i in range(1, 4) %}{{ i }}{% endfor %}"
        assert_compat(template)

    # raise_exception is tested implicitly - it causes errors which we handle


# =============================================================================
# Real Template Patterns
# =============================================================================


class TestRealTemplatePatterns:
    """Test patterns from real HuggingFace chat templates."""

    def test_chatml_format(self):
        """ChatML format (Qwen, OpenHermes, etc.)."""
        template = """{%- for message in messages %}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{% endif %}"""
        assert_compat(template, messages=SIMPLE_MSG, add_generation_prompt=True)
        assert_compat(template, messages=MULTI_TURN, add_generation_prompt=False)

    def test_zephyr_format(self):
        """Zephyr template pattern."""
        template = """{% for message in messages %}
{% if message['role'] == 'user' %}
<|user|>{{ message['content'] }}{{ eos_token }}
{% elif message['role'] == 'assistant' %}
<|assistant|>{{ message['content'] }}{{ eos_token }}
{% endif %}
{% if loop.last and add_generation_prompt %}
<|assistant|>
{% endif %}
{% endfor %}"""
        assert_compat(template, messages=MULTI_TURN, eos_token="</s>", add_generation_prompt=True)

    def test_system_message_extraction(self):
        """Pattern: extract system message then loop rest."""
        template = """{%- if messages[0].role == 'system' %}
System: {{ messages[0].content }}
{%- set loop_messages = messages[1:] %}
{%- else %}
{%- set loop_messages = messages %}
{%- endif %}
{%- for m in loop_messages %}
{{ m.role }}: {{ m.content }}
{% endfor %}"""
        assert_compat(template, messages=WITH_SYSTEM)

    def test_namespace_system_tracking(self):
        """Pattern: track if system message exists."""
        template = """{%- set ns = namespace(has_system=false) %}
{%- for m in messages %}
{%- if m.role == 'system' %}{% set ns.has_system = true %}{% endif %}
{%- endfor %}
Has system: {{ ns.has_system }}"""
        assert_compat(template, messages=WITH_SYSTEM)
        assert_compat(template, messages=SIMPLE_MSG)

    def test_tool_message_pattern(self):
        """Pattern: handle tool-related messages."""
        template = """{%- for m in messages %}
{%- if m.role == 'tool' %}
Tool result: {{ m.content }}
{%- elif m.role == 'assistant' and m.tool_calls is defined %}
Tool call
{%- else %}
{{ m.role }}: {{ m.content }}
{%- endif %}
{% endfor %}"""
        msgs = [
            {"role": "user", "content": "search"},
            {"role": "assistant", "content": "result", "tool_calls": [{"name": "search"}]},
        ]
        assert_compat(template, messages=msgs)

    def test_generation_prompt_pattern(self):
        """generation_prompt: 85% of templates."""
        template = """{% for m in messages %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}Assistant:{% endif %}"""
        assert_compat(template, messages=SIMPLE_MSG, add_generation_prompt=True)
        assert_compat(template, messages=SIMPLE_MSG, add_generation_prompt=False)

    def test_bos_eos_tokens(self):
        """bos_token/eos_token: 25%/7% of templates."""
        template = "{{ bos_token }}{% for m in msgs %}{{ m.content }}{% endfor %}{{ eos_token }}"
        assert_compat(template, msgs=SIMPLE_MSG, bos_token="<s>", eos_token="</s>")


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and potential compatibility issues."""

    def test_empty_list(self):
        """Empty list iteration."""
        template = "{% for x in items %}{{ x }}{% endfor %}"
        assert_compat(template, items=[])

    def test_empty_string(self):
        """Empty string handling."""
        template = "{{ text }}"
        assert_compat(template, text="")

    def test_unicode_content(self):
        """Unicode in content."""
        template = "{{ text }}"
        assert_compat(template, text="Hello 世界 🌍")

    def test_special_characters(self):
        """Special characters in content."""
        template = "{{ text }}"
        assert_compat(template, text="<tag> & \"quotes\" 'apostrophe'")

    def test_multiline_content(self):
        """Multiline string content."""
        template = "{{ text }}"
        assert_compat(template, text="Line 1\nLine 2\nLine 3")

    def test_nested_dicts(self):
        """Nested dictionary access."""
        template = "{{ data.inner.value }}"
        assert_compat(template, data={"inner": {"value": "deep"}})

    def test_nested_lists(self):
        """Nested list access."""
        template = "{{ items[0][1] }}"
        assert_compat(template, items=[["a", "b"], ["c", "d"]])

    def test_boolean_values(self):
        """Boolean true/false values."""
        template = "{% if x %}{{ x }}{% endif %}"
        assert_compat(template, x=True)
        assert_compat(template, x=False)

    def test_numeric_values(self):
        """Numeric values."""
        template = "{{ x }} {{ y }}"
        assert_compat(template, x=42, y=3.14)

    def test_none_value(self):
        """None/null value."""
        template = "{% if x is none %}null{% else %}{{ x }}{% endif %}"
        assert_compat(template, x=None)
        assert_compat(template, x="value")

    def test_complex_expression(self):
        """Complex nested expression."""
        # Note: Use explicit parens around ternary to avoid precedence issues
        # Jinja2 parses: (a + b) * c if x else d - e as ((a+b)*c) if x else (d-e)
        # Some engines may parse differently without explicit grouping
        template = "{{ ((a + b) * c) if x else (d - e) }}"
        assert_compat(template, a=1, b=2, c=3, d=10, e=5, x=True)
        assert_compat(template, a=1, b=2, c=3, d=10, e=5, x=False)


# =============================================================================
# Known Limitations (tests that document current gaps)
# =============================================================================


class TestKnownLimitations:
    """Tests documenting known differences from Jinja2.

    These tests use pytest.xfail to document behaviors that differ from Jinja2.
    When the underlying issue is fixed, the test will start passing and
    pytest will notify us to update the test.
    """

    def test_bracket_access_undefined_key(self):
        """Bracket access to missing dict key should be treated as undefined.

        In Jinja2, `item['missing_key'] is defined` returns False when the key
        doesn't exist. Talu now handles this correctly.
        """
        template = "{% if item['tools'] is defined %}yes{% else %}no{% endif %}"
        # Key exists
        assert_compat(template, item={"tools": []})
        # Key missing - both should return 'no'
        assert_compat(template, item={})

    def test_closing_braces_in_string(self):
        """Closing braces }} inside a string should not end the expression.

        The lexer must properly handle }} inside string literals.
        This affects Mistral chat templates.
        """
        template = '{{ "}}" }}'
        assert_compat(template)

    def test_ternary_precedence_with_arithmetic(self):
        """Ternary if/else should have lower precedence than arithmetic.

        In Jinja2: `a + b if x else c` parses as `(a + b) if x else c`
        """
        template = "{{ a + b if x else c }}"
        # When x=False, Jinja2 returns c (3)
        assert_compat(template, a=1, b=2, c=3, x=False)

    def test_ternary_without_else(self):
        """Ternary if without else clause.

        In Jinja2: `x if cond` outputs x when true, empty string when false.
        """
        template = "{{ 'yes' if condition }}"
        assert_compat(template, condition=True)
        assert_compat(template, condition=False)

    def test_keyword_as_variable_name(self):
        """Keywords can be used as variable names in for loops.

        In Jinja2, most keywords (like 'call', 'filter', 'if') are context-sensitive
        and can be used as variable names. This is used in nvidia/NVIDIA-Nemotron templates.
        """
        # 'call' is used as a variable name in nvidia template
        template = "{% for call in items %}{{ call }}{% endfor %}"
        assert_compat(template, items=[1, 2, 3])

        # 'if' and 'for' as variable names
        template2 = "{% for if in items %}{{ if }}{% endfor %}"
        assert_compat(template2, items=["a", "b"])

        # 'filter' as variable name
        template3 = "{% for filter in items %}{{ filter }}{% endfor %}"
        assert_compat(template3, items=["x", "y"])


# =============================================================================
# HuggingFace Extensions
# =============================================================================


class TestHuggingFaceExtensions:
    """Tests for HuggingFace-specific Jinja2 extensions.

    These extensions are not part of standard Jinja2 but are used by
    HuggingFace chat templates. We test talu directly since Jinja2
    doesn't support these tags.
    """

    def test_generation_block(self):
        """Generation block: {% generation %}...{% endgeneration %}

        Used by MiniMaxAI models to mark generated content.
        Acts as a pass-through block (content is rendered as-is).
        """
        template = "{% generation %}hello{% endgeneration %}"
        result = render_talu(template)
        assert result == "hello"

    def test_generation_block_with_expression(self):
        """Generation block can contain expressions."""
        template = "{% generation %}{{ message }}{% endgeneration %}"
        result = render_talu(template, message="world")
        assert result == "world"

    def test_generation_block_nested(self):
        """Generation block inside a for loop (MiniMaxAI pattern)."""
        template = (
            "{% for item in items %}{% generation %}{{ item }}{% endgeneration %}{% endfor %}"
        )
        result = render_talu(template, items=["a", "b", "c"])
        assert result == "abc"


# =============================================================================
# Implicit String Concatenation
# =============================================================================


class TestImplicitStringConcatenation:
    """Tests for implicit string concatenation (Jinja2/Python style).

    In Jinja2 (like Python), adjacent string literals are implicitly concatenated:
    'hello' 'world' becomes 'helloworld'

    This is used in MiniMaxAI templates for escaping quotes:
    {''name'': value} means {'name': value}
    """

    def test_adjacent_strings(self):
        """Adjacent string literals should be concatenated."""
        template = "{{ 'hello' 'world' }}"
        assert_compat(template)

    def test_adjacent_strings_with_space(self):
        """Spaces between adjacent strings are allowed."""
        template = "{{ 'a'   'b' }}"
        assert_compat(template)

    def test_escaped_quote_pattern(self):
        """Double single-quote inside string acts as escape (via concatenation).

        This pattern is used in MiniMaxAI/MiniMax-M1-80k template:
        {''name'': <tool-name>} which renders as {'name': <tool-name>}
        """
        template = "{{ '{' 'name' ': value}' }}"
        assert_compat(template)

    def test_mixed_quote_adjacent(self):
        """Mixed single and double quote strings can be adjacent."""
        template = "{{ 'hello' \"world\" }}"
        assert_compat(template)

    def test_multiline_adjacent_strings(self):
        """Adjacent strings work across implicit newlines in strings."""
        # This is the actual pattern from MiniMaxAI
        template = "{{ 'line1\\n' 'line2' }}"
        assert_compat(template)


# =============================================================================
# Missing Features (tests for features not yet implemented)
# =============================================================================


class TestMapFilter:
    """Tests for the map filter.

    The map filter applies a filter or extracts an attribute from each item
    in a sequence. Used by meetkai/functionary templates.
    """

    def test_map_attribute(self):
        """map(attribute='name') extracts attribute from each item."""
        # Use join to avoid list format differences (single vs double quotes)
        template = "{{ items | map(attribute='name') | join(',') }}"
        assert_compat(template, items=[{"name": "a"}, {"name": "b"}])

    def test_map_attribute_join(self):
        """map with join to create comma-separated list."""
        template = "{{ items | map(attribute='name') | join(', ') }}"
        assert_compat(template, items=[{"name": "foo"}, {"name": "bar"}])

    def test_map_filter_upper(self):
        """map('upper') applies filter to each item."""
        template = "{{ items | map('upper') | join(',') }}"
        assert_compat(template, items=["a", "b", "c"])


class TestUniqueFilter:
    """Tests for the unique filter.

    The unique filter removes duplicate values from a sequence.
    Used by meetkai/functionary templates.
    """

    def test_unique_simple(self):
        """unique removes duplicates."""
        template = "{{ items | unique | list }}"
        assert_compat(template, items=[1, 2, 2, 3, 1])

    def test_unique_strings(self):
        """unique works with strings."""
        template = "{{ items | unique | join(',') }}"
        assert_compat(template, items=["a", "b", "a", "c"])

    def test_unique_preserves_order(self):
        """unique preserves first occurrence order (with join)."""
        template = "{{ items | unique | join(',') }}"
        assert_compat(template, items=["c", "a", "b", "a", "c"])

    def test_unique_to_list(self):
        """unique preserves first occurrence order (with list)."""
        template = "{{ items | unique | list }}"
        assert_compat(template, normalize_quotes=True, items=["c", "a", "b", "a", "c"])


class TestTypeCheckTests:
    """Tests for type-checking test expressions.

    These tests check the type of a value: is iterable, is mapping, is string.
    Used by meetkai/functionary templates.
    """

    def test_is_iterable_list(self):
        """is iterable returns true for lists."""
        template = "{{ x is iterable }}"
        assert_compat(template, x=[1, 2, 3])

    def test_is_iterable_string(self):
        """is iterable returns true for strings (they are iterable)."""
        template = "{{ x is iterable }}"
        assert_compat(template, x="hello")

    def test_is_iterable_int(self):
        """is iterable returns false for integers."""
        template = "{{ x is iterable }}"
        assert_compat(template, x=42)

    def test_is_mapping_dict(self):
        """is mapping returns true for dicts."""
        template = "{{ x is mapping }}"
        assert_compat(template, x={"a": 1})

    def test_is_mapping_list(self):
        """is mapping returns false for lists."""
        template = "{{ x is mapping }}"
        assert_compat(template, x=[1, 2])

    def test_is_string_true(self):
        """is string returns true for strings."""
        template = "{{ x is string }}"
        assert_compat(template, x="hello")

    def test_is_string_false(self):
        """is string returns false for non-strings."""
        template = "{{ x is string }}"
        assert_compat(template, x=42)


class TestNamespaceAsVariable:
    """Tests for using 'namespace' as a variable name.

    In Jinja2, 'namespace' is only a keyword when followed by '('.
    Otherwise it can be used as a regular variable name.
    Used by meetkai/functionary templates.
    """

    def test_namespace_in_for_loop(self):
        """namespace can be used as a for loop variable."""
        template = "{% for namespace in items %}{{ namespace }}{% endfor %}"
        assert_compat(template, items=["a", "b", "c"])

    def test_namespace_as_set_variable(self):
        """namespace can be set and printed."""
        template = "{% set namespace = 'test' %}{{ namespace }}"
        assert_compat(template)

    def test_namespace_as_macro_argument(self):
        """namespace can be used as a macro argument."""
        template = """
{%- macro greet(namespace) -%}
Hello {{ namespace }}
{%- endmacro -%}
{{ greet('world') }}"""
        assert_compat(template)

    def test_namespace_expression_concat(self):
        """namespace variable can be used in string concatenation."""
        template = "{% set namespace = 'test' %}{{ 'ns:' + namespace }}"
        assert_compat(template)

    def test_namespace_vs_namespace_call(self):
        """namespace() creates a namespace, namespace alone is a variable."""
        template = """
{%- set ns = namespace(x=1) -%}
{%- set namespace = 'var' -%}
ns.x={{ ns.x }}, namespace={{ namespace }}"""
        assert_compat(template)


class TestBracketAccessMissingKey:
    """Tests for bracket access on missing dictionary keys.

    In Jinja2, accessing a missing key via bracket notation returns undefined,
    not an error. This is important for templates that check optional fields.
    Used by meetkai/functionary templates.
    """

    def test_bracket_missing_key_prints_empty(self):
        """Bracket access to missing key should print empty string."""
        template = "{{ d['missing'] }}"
        assert_compat(template, d={})

    def test_bracket_missing_key_in_if(self):
        """Bracket access to missing key in if should be falsy."""
        template = "{% if d['missing'] %}yes{% else %}no{% endif %}"
        assert_compat(template, d={})

    def test_bracket_missing_key_with_or(self):
        """Bracket access to missing key with or should use default."""
        template = "{{ d['missing'] or 'default' }}"
        assert_compat(template, d={})

    def test_bracket_missing_key_multiple_checks(self):
        """Multiple bracket accesses to missing keys in condition."""
        template = "{% if d['a'] or d['b'] or d['c'] %}yes{% else %}no{% endif %}"
        assert_compat(template, d={})

    def test_bracket_existing_key_still_works(self):
        """Bracket access to existing key should still work."""
        template = "{{ d['key'] }}"
        assert_compat(template, d={"key": "value"})

    def test_bracket_missing_nested(self):
        """Bracket access in nested dict context."""
        # Use single item to avoid iteration order differences
        template = "{{ d['a']['optional'] or 'none' }}"
        assert_compat(template, d={"a": {}})


class TestMapFilterWithFilterName:
    """Tests for map filter with filter name argument.

    The map filter can apply another filter to each item: items|map('upper').
    Used by meetkai/functionary templates.
    """

    def test_map_filter_upper(self):
        """map('upper') should uppercase each string."""
        # Use join to avoid list format differences (single vs double quotes)
        template = "{{ items | map('upper') | join(',') }}"
        assert_compat(template, items=["a", "b", "c"])

    def test_map_filter_lower(self):
        """map('lower') should lowercase each string."""
        template = "{{ items | map('lower') | join(',') }}"
        assert_compat(template, items=["A", "B", "C"])

    def test_map_filter_trim(self):
        """map('trim') should trim each string."""
        template = "{{ items | map('trim') | join(',') }}"
        assert_compat(template, items=["  a  ", "  b  "])

    def test_map_filter_string(self):
        """map('string') should convert each item to string."""
        template = "{{ items | map('string') | join(',') }}"
        assert_compat(template, items=[1, 2, 3])


class TestStringConcatenation:
    """Tests for string concatenation operators.

    The + operator concatenates strings, ~ converts to string first.
    Used extensively in meetkai/functionary templates for building output.
    """

    def test_plus_strings(self):
        """+ concatenates two strings."""
        template = "{{ 'hello ' + 'world' }}"
        assert_compat(template)

    def test_plus_multiple(self):
        """+ chains multiple strings."""
        template = "{{ 'a' + 'b' + 'c' + 'd' }}"
        assert_compat(template)

    def test_plus_with_variables(self):
        """+ works with string variables."""
        template = "{{ prefix + middle + suffix }}"
        assert_compat(template, prefix="hello", middle=" ", suffix="world")

    def test_plus_newlines(self):
        """+ preserves newlines in strings."""
        template = "{{ '\\n<|from|>' + role + '\\n<|content|>' + content }}"
        assert_compat(template, role="user", content="Hi")

    def test_tilde_converts_to_string(self):
        """~ converts non-strings before concatenation."""
        template = "{{ 'count: ' ~ num }}"
        assert_compat(template, num=42)

    def test_tilde_with_array(self):
        """~ converts array to string."""
        template = "{{ 'items: ' ~ items }}"
        assert_compat(template, normalize_quotes=True, items=["a", "b"])


class TestSelectattrFilter:
    """Tests for selectattr filter with various tests.

    selectattr filters items by attribute value using test functions.
    Used by meetkai templates to filter tools by type.
    """

    def test_selectattr_defined(self):
        """selectattr with 'defined' test."""
        template = "{{ items | selectattr('a', 'defined') | list }}"
        assert_compat(template, normalize_quotes=True, items=[{"a": 1}, {"b": 2}, {"a": 3}])

    def test_selectattr_equalto(self):
        """selectattr with 'equalto' test."""
        template = "{{ items | selectattr('type', 'equalto', 'function') | list | length }}"
        assert_compat(
            template,
            items=[
                {"type": "function", "name": "a"},
                {"type": "tool", "name": "b"},
                {"type": "function", "name": "c"},
            ],
        )

    def test_selectattr_equalto_string(self):
        """selectattr equalto with string comparison."""
        template = "{{ tools | selectattr('type', 'equalto', 'code_interpreter') | list | length }}"
        assert_compat(
            template,
            tools=[
                {"type": "function"},
                {"type": "code_interpreter"},
            ],
        )

    def test_selectattr_chain(self):
        """selectattr chained with map."""
        template = "{{ items | selectattr('active', 'equalto', true) | map(attribute='name') | join(',') }}"
        assert_compat(
            template,
            items=[
                {"name": "a", "active": True},
                {"name": "b", "active": False},
                {"name": "c", "active": True},
            ],
        )


class TestComplexMacros:
    """Tests for complex macro definitions and calls.

    meetkai templates use deeply nested macros that call each other.
    """

    def test_macro_with_default_args(self):
        """Macro with default argument values."""
        template = """{%- macro greet(name, greeting='Hello') -%}
{{ greeting }}, {{ name }}!
{%- endmacro -%}
{{ greet('World') }}"""
        assert_compat(template)

    def test_macro_calling_macro(self):
        """One macro calls another."""
        template = """{%- macro inner(x) -%}[{{ x }}]{%- endmacro -%}
{%- macro outer(items) -%}
{%- for item in items -%}{{ inner(item) }}{%- endfor -%}
{%- endmacro -%}
{{ outer(['a', 'b', 'c']) }}"""
        assert_compat(template)

    def test_macro_with_conditional(self):
        """Macro with if/else inside."""
        template = """{%- macro convert(val) -%}
{%- if val == 'integer' -%}number{%- else -%}{{ val }}{%- endif -%}
{%- endmacro -%}
{{ convert('integer') }},{{ convert('string') }}"""
        assert_compat(template)

    def test_multiple_macros(self):
        """Multiple macro definitions."""
        template = """{%- macro a() -%}A{%- endmacro -%}
{%- macro b() -%}B{%- endmacro -%}
{%- macro c() -%}C{%- endmacro -%}
{{ a() }}{{ b() }}{{ c() }}"""
        assert_compat(template)


class TestNestedLoops:
    """Tests for nested for loops.

    meetkai templates iterate over messages and their tool_calls.
    """

    def test_nested_for_simple(self):
        """Simple nested for loop."""
        template = """{%- for outer in items -%}
{%- for inner in outer -%}{{ inner }},{%- endfor -%}|
{%- endfor -%}"""
        assert_compat(template, items=[["a", "b"], ["c", "d"]])

    def test_nested_for_with_condition(self):
        """Nested loop with conditional."""
        template = """{%- for msg in messages -%}
{{ msg.role }}:
{%- if msg.tool_calls -%}
{%- for tc in msg.tool_calls -%}
  call:{{ tc.name }}
{%- endfor -%}
{%- endif -%}
{%- endfor -%}"""
        assert_compat(
            template,
            messages=[
                {"role": "user", "tool_calls": None},
                {"role": "assistant", "tool_calls": [{"name": "search"}, {"name": "calc"}]},
            ],
        )

    def test_nested_for_dict_items(self):
        """Nested loop over dict items."""
        # Use sorted() to ensure consistent ordering across implementations
        template = """{%- for key, val in data.items() | sort -%}
{{ key }}:{{ val }},
{%- endfor -%}"""
        assert_compat(template, data={"a": 1, "b": 2})


class TestTojsonFilter:
    """Tests for tojson filter with various options.

    Used for serializing function parameters in tool definitions.
    Note: Dict key ordering may differ between Python and Zig, so we test
    with single-key dicts or arrays to avoid ordering issues.
    """

    def test_tojson_simple(self):
        """tojson on simple value."""
        template = "{{ data | tojson }}"
        assert_compat(template, data={"name": "test"})

    def test_tojson_array(self):
        """tojson on array."""
        template = "{{ data | tojson }}"
        assert_compat(template, data=[1, 2, 3])

    def test_tojson_indent(self):
        """tojson with indent parameter."""
        template = "{{ data | tojson(indent=2) }}"
        assert_compat(template, data={"key": "value"})

    def test_tojson_nested_array(self):
        """tojson on nested structure with array."""
        template = "{{ data | tojson }}"
        assert_compat(template, data={"items": [1, 2, 3]})

    def test_tojson_in_output(self):
        """tojson used in string building."""
        template = "params: {{ params | tojson }}"
        assert_compat(template, params={"type": "object"})


class TestDictKeyChecking:
    """Tests for checking if key exists in dict.

    Used by templates to check for optional fields like 'tool_calls'.
    """

    def test_in_operator_present(self):
        """'in' returns true for existing key."""
        template = "{{ 'a' in data }}"
        assert_compat(template, data={"a": 1, "b": 2})

    def test_in_operator_missing(self):
        """'in' returns false for missing key."""
        template = "{{ 'c' in data }}"
        assert_compat(template, data={"a": 1, "b": 2})

    def test_in_with_conditional(self):
        """'in' used in if statement."""
        template = "{%- if 'tool_calls' in msg and msg['tool_calls'] -%}has_calls{%- else -%}no_calls{%- endif -%}"
        assert_compat(template, msg={"role": "user"})

    def test_in_with_conditional_true(self):
        """'in' with truthy value."""
        template = "{%- if 'tool_calls' in msg and msg['tool_calls'] -%}has_calls{%- else -%}no_calls{%- endif -%}"
        assert_compat(template, msg={"role": "assistant", "tool_calls": [{"name": "test"}]})

    def test_not_in_operator(self):
        """'not in' operator."""
        template = "{{ 'c' not in data }}"
        assert_compat(template, data={"a": 1, "b": 2})


class TestTypeChecksAdvanced:
    """Advanced type checking tests from meetkai templates."""

    def test_is_iterable_and_not_string(self):
        """Combined iterable and not string check."""
        template = "{{ x is iterable and x is not string }}"
        assert_compat(template, x=[1, 2, 3])

    def test_is_iterable_and_not_string_false(self):
        """String is iterable but should fail 'not string'."""
        template = "{{ x is iterable and x is not string }}"
        assert_compat(template, x="hello")

    def test_type_check_in_conditional(self):
        """Type check in if statement."""
        template = """{%- if val is iterable and val is not string -%}
{{ val | join(' | ') }}
{%- else -%}
{{ val }}
{%- endif -%}"""
        assert_compat(template, val=["a", "b", "c"])

    def test_type_check_string_path(self):
        """Type check takes string path."""
        template = """{%- if val is iterable and val is not string -%}
{{ val | join(' | ') }}
{%- else -%}
{{ val }}
{%- endif -%}"""
        assert_compat(template, val="hello")
