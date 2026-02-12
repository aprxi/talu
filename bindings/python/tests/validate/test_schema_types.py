"""Test different schema input formats."""

import json

from talu import Validator

# =============================================================================
# Dict Schema Input
# =============================================================================


class TestDictSchema:
    """Test schema input as Python dict."""

    def test_simple_dict(self):
        """Accept simple dict schema."""
        schema = {"type": "string"}
        validator = Validator(schema)
        assert validator.validate('"hello"') is True

    def test_complex_dict(self):
        """Accept complex dict schema."""
        schema = {
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                        "required": ["name"],
                    },
                }
            },
            "required": ["users"],
        }
        validator = Validator(schema)
        assert validator.validate('{"users": [{"name": "Alice", "age": 30}]}') is True

    def test_dict_with_refs(self):
        """Accept dict with $defs references."""
        schema = {
            "$defs": {"name_type": {"type": "string"}},
            "type": "object",
            "properties": {"name": {"$ref": "#/$defs/name_type"}},
            "required": ["name"],
        }
        validator = Validator(schema)
        assert validator.validate('{"name": "Bob"}') is True


# =============================================================================
# String Schema Input
# =============================================================================


class TestStringSchema:
    """Test schema input as JSON string."""

    def test_simple_string(self):
        """Accept simple JSON string schema."""
        schema = '{"type": "integer"}'
        validator = Validator(schema)
        assert validator.validate("42") is True

    def test_complex_string(self):
        """Accept complex JSON string schema."""
        schema = json.dumps(
            {
                "type": "object",
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            }
        )
        validator = Validator(schema)
        assert validator.validate('{"x": 1.5, "y": 2.5}') is True

    def test_minified_string(self):
        """Accept minified JSON string."""
        schema = '{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}'
        validator = Validator(schema)
        assert validator.validate('{"a":1}') is True


# =============================================================================
# JSON Schema Types
# =============================================================================


class TestJsonSchemaTypes:
    """Test various JSON Schema type constraints."""

    def test_type_string(self):
        """Validate string type."""
        validator = Validator({"type": "string"})
        assert validator.validate('"hello"') is True
        assert validator.validate("123") is False
        assert validator.validate("true") is False

    def test_type_integer(self):
        """Validate integer type - strict, no decimals allowed."""
        validator = Validator({"type": "integer"})
        assert validator.validate("42") is True
        assert validator.validate("-5") is True
        assert validator.validate("0") is True
        assert validator.validate("3.14") is False  # Decimal not allowed
        assert validator.validate("1e5") is False  # Exponent not allowed
        assert validator.validate("1.0") is False  # Decimal not allowed
        assert validator.validate('"42"') is False  # String not allowed

    def test_type_number(self):
        """Validate number type (int or float)."""
        validator = Validator({"type": "number"})
        assert validator.validate("42") is True
        assert validator.validate("3.14") is True
        assert validator.validate('"42"') is False

    def test_type_boolean(self):
        """Validate boolean type."""
        validator = Validator({"type": "boolean"})
        assert validator.validate("true") is True
        assert validator.validate("false") is True
        assert validator.validate('"true"') is False

    def test_type_null(self):
        """Validate null type."""
        validator = Validator({"type": "null"})
        assert validator.validate("null") is True
        assert validator.validate('""') is False

    def test_type_array_with_items(self):
        """Validate array type with items schema."""
        validator = Validator({"type": "array", "items": {"type": "integer"}})
        assert validator.validate("[]") is True
        assert validator.validate("[1, 2, 3]") is True
        assert validator.validate('["a"]') is False

    def test_type_object_with_properties(self):
        """Validate object type with properties."""
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        validator = Validator(schema)
        assert validator.validate('{"x": 1}') is True
        assert validator.validate('{"x": "a"}') is False


# =============================================================================
# JSON Schema Constraints
# =============================================================================


class TestJsonSchemaConstraints:
    """Test JSON Schema constraint keywords.

    The grammar engine validates JSON at the **syntactic** level using a
    context-free grammar (CFG). This means:

    SUPPORTED (syntactic - can be expressed in grammar):
    - enum: Finite set of literal values → grammar alternatives
    - const: Single literal value → grammar literal
    - integer min/max: Bounded range → enumerate all valid values as alternatives
      (works for small ranges, configurable via max_exact_span)

    NOT SUPPORTED (semantic - require runtime counting/state):
    - minLength/maxLength: Would need to count characters during parsing
    - minItems/maxItems: Would need to count array elements during parsing
    - number min/max: Infinite float values, cannot enumerate

    To support length/count constraints, the engine would need:
    1. A counting mechanism in the grammar (e.g., bounded repetition {n,m})
    2. Runtime state to track counts during parsing
    3. This adds complexity and may impact performance for streaming validation
    """

    def test_enum(self):
        """Validate enum constraint."""
        validator = Validator({"enum": ["red", "green", "blue"]})
        assert validator.validate('"red"') is True
        assert validator.validate('"yellow"') is False

    def test_const(self):
        """Validate const constraint."""
        validator = Validator({"const": "fixed"})
        assert validator.validate('"fixed"') is True
        assert validator.validate('"other"') is False

    def test_string_minLength(self):
        """Validate string minLength constraint.

        SUPPORTED: Grammar generates bounded string: " char{min,max} "
        where min is the minLength constraint. Required chars are explicit
        in the grammar sequence.
        """
        validator = Validator({"type": "string", "minLength": 3})
        assert validator.validate('"abc"') is True
        assert validator.validate('"ab"') is False

    def test_string_maxLength(self):
        """Validate string maxLength constraint.

        SUPPORTED: Grammar generates bounded string: " char{min,max} "
        where max is the maxLength constraint. Extra chars beyond max
        are not allowed by the grammar.
        """
        validator = Validator({"type": "string", "maxLength": 5})
        assert validator.validate('"hello"') is True
        assert validator.validate('"toolong"') is False

    def test_integer_minimum_maximum(self):
        """Validate integer min/max constraint - SUPPORTED via enumeration.

        SUPPORTED: For bounded integer ranges, the grammar enumerates all
        valid values as alternatives: (1|2|3|4|5) for range 1-5.

        This works for small ranges (configurable via max_exact_span).
        Default limit is 1000 values. Larger ranges fall back to generic
        integer grammar without range checking.
        """
        validator = Validator({"type": "integer", "minimum": 1, "maximum": 5})
        assert validator.validate("0") is False
        assert validator.validate("1") is True
        assert validator.validate("3") is True
        assert validator.validate("5") is True
        assert validator.validate("6") is False
        assert validator.validate("3.5") is False  # Not an integer

    def test_number_minimum(self):
        """Validate number minimum constraint.

        SUPPORTED via semantic validation: Grammar validates JSON number syntax,
        then semantic validator checks value is >= minimum after parsing.
        """
        validator = Validator({"type": "number", "minimum": 0})
        assert validator.validate("0") is True
        assert validator.validate("10") is True
        assert validator.validate("-1") is False

    def test_number_maximum(self):
        """Validate number maximum constraint.

        SUPPORTED via semantic validation: Grammar validates JSON number syntax,
        then semantic validator checks value is <= maximum after parsing.
        """
        validator = Validator({"type": "number", "maximum": 100})
        assert validator.validate("100") is True
        assert validator.validate("50") is True
        assert validator.validate("101") is False

    def test_array_minItems(self):
        """Validate array minItems constraint.

        SUPPORTED: Grammar generates bounded array with required items:
        [ ws item (, item){min-1} (, item)* ws ]
        First item + (min-1) required comma-items, then unbounded optional.
        """
        validator = Validator({"type": "array", "items": {"type": "integer"}, "minItems": 2})
        assert validator.validate("[1, 2]") is True
        assert validator.validate("[1]") is False

    def test_array_maxItems(self):
        """Validate array maxItems constraint.

        SUPPORTED: Grammar generates bounded array with optional items:
        [ ws (item (, item){0,max-1})? ws ]
        First item optional, then up to (max-1) optional comma-items.
        """
        validator = Validator({"type": "array", "items": {"type": "integer"}, "maxItems": 3})
        assert validator.validate("[1, 2, 3]") is True
        assert validator.validate("[1, 2, 3, 4]") is False


# =============================================================================
# Required and Optional Properties
# =============================================================================


class TestRequiredProperties:
    """Test required vs optional properties.

    NOTE: The grammar engine expects properties in definition order.
    Optional properties can be omitted, but when present must follow
    required properties.
    """

    def test_all_required(self):
        """All properties required."""
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }
        validator = Validator(schema)
        assert validator.validate('{"a": 1, "b": 2}') is True
        assert validator.validate('{"a": 1}') is False
        assert validator.validate('{"b": 2}') is False

    def test_no_required(self):
        """No properties required - empty object valid.

        NOTE: When no properties are required, the grammar expects the
        defined properties in order if present, or empty object.
        """
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        }
        validator = Validator(schema)
        assert validator.validate("{}") is True
        # Optional properties can be included (in definition order)
        assert validator.validate('{"a": 1}') is True
        assert validator.validate('{"a": 1, "b": 2}') is True

    def test_partial_required(self):
        """Some properties required."""
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a"],
        }
        validator = Validator(schema)
        assert validator.validate('{"a": 1}') is True
        assert validator.validate('{"a": 1, "b": 2}') is True
        assert validator.validate('{"b": 2}') is False


# =============================================================================
# Additional Properties
# =============================================================================


class TestAdditionalProperties:
    """Test additionalProperties handling.

    SUPPORTED BEHAVIOR:
    - additionalProperties: true (default) - Grammar accepts any valid JSON object,
      semantic validation checks defined properties for type correctness and required.
    - additionalProperties: false - Grammar enforces strict property names, only
      defined properties are accepted.
    - additionalProperties: {schema} - NOT YET SUPPORTED. Would need semantic
      validation to check additional property values against the schema.

    Implementation approach:
    1. When additionalProperties is not explicitly false, use generic JSON grammar
    2. Semantic validator handles: type checking, required properties
    3. When additionalProperties: false, use strict grammar with only defined props
    """

    def test_additional_allowed_by_default(self):
        """Additional properties allowed by default.

        SUPPORTED: Grammar accepts any valid JSON object when additionalProperties
        is not explicitly false. Semantic validation handles type checking for
        defined properties while allowing additional properties.
        """
        schema = {
            "type": "object",
            "properties": {"known": {"type": "integer"}},
            "required": ["known"],
        }
        validator = Validator(schema)
        assert validator.validate('{"known": 1, "unknown": "value"}') is True

    def test_additional_properties_false(self):
        """Additional properties forbidden when explicitly disabled.

        SUPPORTED: The grammar only accepts defined property names, effectively
        enforcing additionalProperties: false. This is the expected behavior
        for structured output - we want to constrain the model to known properties.
        """
        schema = {
            "type": "object",
            "properties": {"known": {"type": "integer"}},
            "additionalProperties": False,
            "required": ["known"],
        }
        validator = Validator(schema)
        assert validator.validate('{"known": 1}') is True
        assert validator.validate('{"known": 1, "unknown": "value"}') is False

    def test_additional_properties_typed(self):
        """Additional properties with type constraint."""
        schema = {
            "type": "object",
            "properties": {"known": {"type": "integer"}},
            "additionalProperties": {"type": "string"},
            "required": ["known"],
        }
        validator = Validator(schema)
        assert validator.validate('{"known": 1, "extra": "ok"}') is True
        assert validator.validate('{"known": 1, "extra": 123}') is False
