"""
Jinja2 operator tests for Template.

Tests for all supported operators: +, -, *, /, in, is, etc.
"""

import pytest

from talu.exceptions import TemplateError
from tests.template.conftest import OPERATOR_CASES


class TestArithmeticOperators:
    """Tests for arithmetic operators."""

    def test_addition(self, Template):
        """Addition operator +."""
        t = Template("{{ a + b }}")
        assert t(a=3, b=4) == "7"

    def test_subtraction(self, Template):
        """Subtraction operator -."""
        t = Template("{{ a - b }}")
        assert t(a=10, b=3) == "7"

    def test_multiplication(self, Template):
        """Multiplication operator *."""
        t = Template("{{ a * b }}")
        assert t(a=3, b=4) == "12"

    def test_division(self, Template):
        """Division operator /."""
        t = Template("{{ a / b }}")
        assert t(a=10, b=2) == "5.0"

    def test_floor_division(self, Template):
        """Floor division operator //."""
        t = Template("{{ a // b }}")
        assert t(a=7, b=2) == "3"

    def test_modulo(self, Template):
        """Modulo operator %."""
        t = Template("{{ a % b }}")
        assert t(a=7, b=3) == "1"

    def test_power(self, Template):
        """Power operator **."""
        t = Template("{{ a ** b }}")
        assert t(a=2, b=3) == "8"

    def test_negative(self, Template):
        """Unary minus."""
        t = Template("{{ -x }}")
        assert t(x=5) == "-5"

    def test_positive(self, Template):
        """Unary plus."""
        t = Template("{{ +x }}")
        assert t(x=5) == "5"


class TestStringOperators:
    """Tests for string operators."""

    def test_concatenation(self, Template):
        """String concatenation with ~."""
        t = Template("{{ a ~ b }}")
        assert t(a="hello", b="world") == "helloworld"

    def test_concatenation_mixed_types(self, Template):
        """String concatenation converts types."""
        t = Template("{{ a ~ b }}")
        assert t(a="count: ", b=42) == "count: 42"

    def test_string_repeat(self, Template):
        """String repeat with *."""
        t = Template("{{ x * 3 }}")
        assert t(x="ab") == "ababab"


class TestComparisonOperators:
    """Tests for comparison operators."""

    def test_equal(self, Template):
        """Equal operator ==."""
        t = Template("{{ a == b }}")
        assert t(a=5, b=5) == "True"
        assert t(a=5, b=3) == "False"

    def test_not_equal(self, Template):
        """Not equal operator !=."""
        t = Template("{{ a != b }}")
        assert t(a=5, b=3) == "True"
        assert t(a=5, b=5) == "False"

    def test_greater_than(self, Template):
        """Greater than operator >."""
        t = Template("{{ a > b }}")
        assert t(a=5, b=3) == "True"
        assert t(a=3, b=5) == "False"

    def test_less_than(self, Template):
        """Less than operator <."""
        t = Template("{{ a < b }}")
        assert t(a=3, b=5) == "True"
        assert t(a=5, b=3) == "False"

    def test_greater_equal(self, Template):
        """Greater or equal operator >=."""
        t = Template("{{ a >= b }}")
        assert t(a=5, b=5) == "True"
        assert t(a=5, b=3) == "True"
        assert t(a=3, b=5) == "False"

    def test_less_equal(self, Template):
        """Less or equal operator <=."""
        t = Template("{{ a <= b }}")
        assert t(a=5, b=5) == "True"
        assert t(a=3, b=5) == "True"
        assert t(a=5, b=3) == "False"


class TestLogicalOperators:
    """Tests for logical operators."""

    def test_and(self, Template):
        """Logical and operator."""
        t = Template("{{ a and b }}")
        assert t(a=True, b=True) == "True"
        assert t(a=True, b=False) == "False"
        assert t(a=False, b=True) == "False"

    def test_or(self, Template):
        """Logical or operator."""
        t = Template("{{ a or b }}")
        assert t(a=True, b=False) == "True"
        assert t(a=False, b=True) == "True"
        assert t(a=False, b=False) == "False"

    def test_not(self, Template):
        """Logical not operator."""
        t = Template("{{ not x }}")
        assert t(x=True) == "False"
        assert t(x=False) == "True"

    def test_and_short_circuit(self, Template):
        """And short-circuits on false."""
        t = Template("{{ false and undefined_var }}")
        assert t() == "False"

    def test_or_short_circuit(self, Template):
        """Or short-circuits on true."""
        t = Template("{{ true or undefined_var }}")
        assert t() == "True"


class TestMembershipOperators:
    """Tests for in/not in operators."""

    def test_in_list(self, Template):
        """In operator with list."""
        t = Template("{{ x in items }}")
        assert t(x="a", items=["a", "b", "c"]) == "True"
        assert t(x="z", items=["a", "b", "c"]) == "False"

    def test_in_string(self, Template):
        """In operator with string."""
        t = Template("{{ x in s }}")
        assert t(x="ell", s="hello") == "True"
        assert t(x="xyz", s="hello") == "False"

    def test_in_dict(self, Template):
        """In operator with dict (checks keys)."""
        t = Template("{{ x in d }}")
        assert t(x="a", d={"a": 1, "b": 2}) == "True"
        assert t(x="c", d={"a": 1, "b": 2}) == "False"

    def test_not_in(self, Template):
        """Not in operator."""
        t = Template("{{ x not in items }}")
        assert t(x="z", items=["a", "b"]) == "True"
        assert t(x="a", items=["a", "b"]) == "False"


class TestIdentityOperators:
    """Tests for is/is not operators."""

    def test_is_none(self, Template):
        """Is none test."""
        t = Template("{{ x is none }}")
        assert t(x=None) == "True"
        assert t(x=0) == "False"

    def test_is_not_none(self, Template):
        """Is not none test."""
        t = Template("{{ x is not none }}")
        assert t(x="hello") == "True"
        assert t(x=None) == "False"

    def test_is_defined(self, Template):
        """Is defined test."""
        t = Template("{{ x is defined }}")
        assert t(x=5) == "True"

    def test_is_undefined(self, Template):
        """Is undefined test."""
        t = Template("{{ x is undefined }}")
        assert t() == "True"
        assert t(x=5) == "False"

    def test_is_true(self, Template):
        """Is true test."""
        t = Template("{{ x is true }}")
        assert t(x=True) == "True"
        assert t(x=False) == "False"
        assert t(x=1) == "False"  # 1 is not True

    def test_is_false(self, Template):
        """Is false test."""
        t = Template("{{ x is false }}")
        assert t(x=False) == "True"
        assert t(x=True) == "False"

    def test_is_odd(self, Template):
        """Is odd test."""
        t = Template("{{ x is odd }}")
        assert t(x=3) == "True"
        assert t(x=4) == "False"

    def test_is_even(self, Template):
        """Is even test."""
        t = Template("{{ x is even }}")
        assert t(x=4) == "True"
        assert t(x=3) == "False"

    def test_is_divisibleby(self, Template):
        """Is divisibleby test."""
        t = Template("{{ x is divisibleby(3) }}")
        assert t(x=9) == "True"
        assert t(x=10) == "False"

    def test_is_string(self, Template):
        """Is string test."""
        t = Template("{{ x is string }}")
        assert t(x="hello") == "True"
        assert t(x=42) == "False"

    def test_is_number(self, Template):
        """Is number test."""
        t = Template("{{ x is number }}")
        assert t(x=42) == "True"
        assert t(x=3.14) == "True"
        assert t(x="42") == "False"

    def test_is_sequence(self, Template):
        """Is sequence test."""
        t = Template("{{ x is sequence }}")
        assert t(x=[1, 2, 3]) == "True"
        assert t(x="hello") == "True"
        assert t(x=42) == "False"

    def test_is_mapping(self, Template):
        """Is mapping test."""
        t = Template("{{ x is mapping }}")
        assert t(x={"a": 1}) == "True"
        assert t(x=[1, 2]) == "False"

    def test_is_iterable(self, Template):
        """Is iterable test."""
        t = Template("{{ x is iterable }}")
        assert t(x=[1, 2, 3]) == "True"
        assert t(x="hello") == "True"
        assert t(x=42) == "False"

    def test_is_callable(self, Template):
        """Is callable test."""
        Template("{{ x is callable }}")
        # Note: In templates, we usually don't pass callables
        # but this tests the template compiles

    def test_is_equalto(self, Template):
        """Is equalto test (used by select/reject filters)."""
        t = Template("{{ x is equalto(5) }}")
        assert t(x=5) == "True"
        assert t(x=3) == "False"

    def test_is_eq(self, Template):
        """Is eq test (alias for equalto)."""
        t = Template("{{ x is eq(5) }}")
        assert t(x=5) == "True"
        assert t(x=3) == "False"

    def test_is_sameas(self, Template):
        """Is sameas test."""
        t = Template("{{ x is sameas(y) }}")
        assert t(x=5, y=5) == "True"
        assert t(x=5, y=6) == "False"


class TestOperatorPrecedence:
    """Tests for operator precedence."""

    def test_mul_before_add(self, Template):
        """Multiplication before addition."""
        t = Template("{{ 2 + 3 * 4 }}")
        assert t() == "14"  # 2 + 12, not 20

    def test_parentheses_override(self, Template):
        """Parentheses override precedence."""
        t = Template("{{ (2 + 3) * 4 }}")
        assert t() == "20"

    def test_power_right_associative(self, Template):
        """Power is right-associative."""
        t = Template("{{ 2 ** 3 ** 2 }}")
        assert t() == "512"  # 2^9, not 8^2

    def test_comparison_chain(self, Template):
        """Comparison operators can be chained."""
        Template("{{ 1 < 2 < 3 }}")
        # Note: Jinja2 doesn't support chaining like Python
        # This tests the template compiles

    def test_complex_expression(self, Template):
        """Complex expression with multiple operators."""
        t = Template("{{ (a + b) * c - d / e }}")
        assert t(a=1, b=2, c=3, d=10, e=2) == "4.0"  # (1+2)*3 - 10/2 = 9 - 5


class TestTernaryOperator:
    """Tests for ternary/conditional expression."""

    def test_ternary_true(self, Template):
        """Ternary expression when true."""
        t = Template("{{ 'yes' if x else 'no' }}")
        assert t(x=True) == "yes"

    def test_ternary_false(self, Template):
        """Ternary expression when false."""
        t = Template("{{ 'yes' if x else 'no' }}")
        assert t(x=False) == "no"

    def test_ternary_with_expression(self, Template):
        """Ternary with expression condition."""
        t = Template("{{ 'big' if x > 10 else 'small' }}")
        assert t(x=15) == "big"
        assert t(x=5) == "small"

    def test_ternary_nested(self, Template):
        """Nested ternary expressions."""
        t = Template("{{ 'a' if x == 1 else ('b' if x == 2 else 'c') }}")
        assert t(x=1) == "a"
        assert t(x=2) == "b"
        assert t(x=3) == "c"


class TestOperatorParametrized:
    """Parametrized operator tests."""

    @pytest.mark.parametrize("template,variables,expected", OPERATOR_CASES)
    def test_operator_cases(self, Template, template, variables, expected):
        """Parametrized operator cases from conftest."""
        t = Template(template)
        assert t(**variables) == expected


class TestDivisionSemantics:
    """Tests for integer vs float division semantics."""

    def test_single_slash_is_float_division(self, Template):
        """Single / produces float result (Python 3 semantics)."""
        t = Template("{{ 7 / 2 }}")
        result = t()
        # Should be 3.5, not 3
        assert result == "3.5" or float(result) == 3.5

    def test_double_slash_is_integer_division(self, Template):
        """Double // produces integer result."""
        t = Template("{{ 7 // 2 }}")
        result = t()
        assert result == "3"

    def test_float_division_with_integers(self, Template):
        """Float division even when divisible."""
        t = Template("{{ 6 / 2 }}")
        result = t()
        # Result could be "3.0" or "3" depending on implementation
        assert float(result) == 3.0

    def test_integer_division_truncates_toward_zero(self, Template):
        """Integer division truncates toward negative infinity (Python floor)."""
        t = Template("{{ -7 // 2 }}")
        result = t()
        # Python floor division: -7 // 2 = -4 (toward -inf)
        assert result == "-4"

    def test_float_division_with_floats(self, Template):
        """Float division with float operands."""
        t = Template("{{ 7.0 / 2.0 }}")
        result = t()
        assert float(result) == 3.5

    def test_integer_division_with_floats(self, Template):
        """Integer division with float operands produces float."""
        t = Template("{{ 7.5 // 2.0 }}")
        result = t()
        # 7.5 // 2.0 = 3.0 (floor)
        assert float(result) == 3.0

    def test_modulo_with_division(self, Template):
        """Modulo works correctly with division semantics."""
        t = Template("{{ 7 % 3 }}")
        assert t() == "1"

    def test_modulo_with_negative(self, Template):
        """Modulo with negative follows Python semantics."""
        t = Template("{{ -7 % 3 }}")
        result = t()
        # Python: -7 % 3 = 2 (sign follows divisor)
        assert result == "2"

    def test_division_by_variable(self, Template):
        """Division works with variable operands."""
        t = Template("{{ a / b }}")
        result = t(a=10, b=4)
        assert float(result) == 2.5

    def test_integer_division_by_variable(self, Template):
        """Integer division works with variable operands."""
        t = Template("{{ a // b }}")
        result = t(a=10, b=4)
        assert result == "2"

    def test_chained_division(self, Template):
        """Chained division evaluates left-to-right."""
        t = Template("{{ 24 / 4 / 2 }}")
        result = t()
        # (24 / 4) / 2 = 6 / 2 = 3.0
        assert float(result) == 3.0


class TestArithmeticErrors:
    """Tests for arithmetic error handling.

    These tests ensure that division by zero and modulo by zero
    raise appropriate errors rather than crashing or producing undefined behavior.
    """

    def test_division_by_zero_literal(self, Template):
        """Division by zero literal raises error."""
        t = Template("{{ 1 / 0 }}")
        with pytest.raises(TemplateError):
            t()

    def test_division_by_zero_variable(self, Template):
        """Division by zero via variable raises error."""
        t = Template("{{ x / y }}")
        with pytest.raises(TemplateError):
            t(x=10, y=0)

    def test_floor_division_by_zero_literal(self, Template):
        """Floor division by zero literal raises error."""
        t = Template("{{ 7 // 0 }}")
        with pytest.raises(TemplateError):
            t()

    def test_floor_division_by_zero_variable(self, Template):
        """Floor division by zero via variable raises error."""
        t = Template("{{ a // b }}")
        with pytest.raises(TemplateError):
            t(a=7, b=0)

    def test_modulo_by_zero_literal(self, Template):
        """Modulo by zero literal raises error."""
        t = Template("{{ 7 % 0 }}")
        with pytest.raises(TemplateError):
            t()

    def test_modulo_by_zero_variable(self, Template):
        """Modulo by zero via variable raises error."""
        t = Template("{{ x % y }}")
        with pytest.raises(TemplateError):
            t(x=7, y=0)

    def test_division_by_zero_in_expression(self, Template):
        """Division by zero in complex expression raises error."""
        t = Template("{{ (a + b) / (c - c) }}")
        with pytest.raises(TemplateError):
            t(a=1, b=2, c=5)

    def test_division_by_zero_float(self, Template):
        """Division by zero with float zero raises error."""
        t = Template("{{ 1.0 / 0.0 }}")
        with pytest.raises(TemplateError):
            t()

    def test_modulo_by_zero_negative(self, Template):
        """Modulo by zero with negative dividend raises error."""
        t = Template("{{ -7 % 0 }}")
        with pytest.raises(TemplateError):
            t()

    def test_chained_division_with_zero(self, Template):
        """Chained division ending in zero raises error."""
        t = Template("{{ 10 / 2 / 0 }}")
        with pytest.raises(TemplateError):
            t()

    def test_division_by_zero_conditional_not_taken(self, Template):
        """Division by zero in untaken branch doesn't raise (short-circuit)."""
        t = Template("{{ 'safe' if true else (1 / 0) }}")
        # The false branch should not be evaluated
        assert t() == "safe"

    def test_modulo_by_zero_conditional_not_taken(self, Template):
        """Modulo by zero in untaken branch doesn't raise (short-circuit)."""
        t = Template("{{ 'safe' if true else (1 % 0) }}")
        assert t() == "safe"
