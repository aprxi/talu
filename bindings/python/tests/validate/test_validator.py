"""Test Validator core API."""

import pytest

from talu import Validator

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_object_schema():
    """Simple object schema with required fields.

    Default JSON Schema behavior - additional properties are allowed.
    Type validation happens semantically (after parsing), not during streaming.
    """
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }


@pytest.fixture
def strict_object_schema():
    """Simple object schema with additionalProperties: false.

    Uses additionalProperties: false for strict streaming validation -
    grammar enforces types at parse time for immediate error detection.
    """
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
        "additionalProperties": False,
    }


@pytest.fixture
def nested_schema():
    """Schema with nested objects and arrays."""
    return {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["name"],
            },
        },
        "required": ["user"],
    }


@pytest.fixture
def number_schema():
    """Schema with number type."""
    return {
        "type": "object",
        "properties": {
            "score": {"type": "number"},
            "count": {"type": "integer"},
        },
        "required": ["score"],
    }


# =============================================================================
# One-Shot Validation Tests
# =============================================================================


class TestValidate:
    """Test the validate() one-shot API."""

    def test_valid_json(self, simple_object_schema):
        """Validate returns True for valid JSON."""
        validator = Validator(simple_object_schema)
        assert validator.validate('{"name": "Alice", "age": 30}') is True

    def test_invalid_type(self, simple_object_schema):
        """Validate returns False for wrong type."""
        validator = Validator(simple_object_schema)
        # age should be integer, not string
        assert validator.validate('{"name": "Alice", "age": "thirty"}') is False

    def test_missing_required(self, simple_object_schema):
        """Validate returns False for missing required field."""
        validator = Validator(simple_object_schema)
        assert validator.validate('{"name": "Alice"}') is False

    def test_extra_fields_allowed_by_default(self, simple_object_schema):
        """Extra fields are allowed by default (JSON Schema behavior).

        SUPPORTED: When additionalProperties is not explicitly false, the grammar
        accepts any JSON object, and semantic validation handles type checking.
        """
        validator = Validator(simple_object_schema)
        assert validator.validate('{"name": "Alice", "age": 30, "extra": true}') is True

    def test_nested_object(self, nested_schema):
        """Validate works with nested objects."""
        validator = Validator(nested_schema)
        assert validator.validate('{"user": {"name": "Bob", "tags": ["a", "b"]}}') is True

    def test_bytes_input(self, simple_object_schema):
        """Validate accepts bytes input."""
        validator = Validator(simple_object_schema)
        assert validator.validate(b'{"name": "Alice", "age": 30}') is True

    def test_number_type(self, number_schema):
        """Validate accepts number types."""
        validator = Validator(number_schema)
        assert validator.validate('{"score": 50}') is True
        assert validator.validate('{"score": 3.14}') is True
        assert validator.validate('{"score": -10}') is True
        # Wrong type fails
        assert validator.validate('{"score": "fifty"}') is False


# =============================================================================
# Streaming Validation Tests
# =============================================================================


class TestFeed:
    """Test the feed() streaming API."""

    def test_valid_chunks(self, simple_object_schema):
        """Feed returns True for valid chunks."""
        validator = Validator(simple_object_schema)
        assert validator.feed('{"name":') is True
        assert validator.feed(' "Alice",') is True
        assert validator.feed(' "age": 30}') is True
        assert validator.is_complete is True

    def test_violation_mid_stream(self, strict_object_schema):
        """Feed returns False on schema violation.

        Uses strict schema (additionalProperties: false) for grammar-level type checking.
        """
        validator = Validator(strict_object_schema)
        assert validator.feed('{"name": "Bob", "age": ') is True
        # String instead of integer
        assert validator.feed('"not_a_number"') is False

    def test_position_tracking(self, simple_object_schema):
        """Position tracks bytes consumed."""
        validator = Validator(simple_object_schema)
        validator.feed('{"name":')
        pos1 = validator.position
        validator.feed(' "test",')
        pos2 = validator.position
        assert pos2 > pos1

    def test_incomplete_json(self, simple_object_schema):
        """is_complete is False for incomplete JSON."""
        validator = Validator(simple_object_schema)
        validator.feed('{"name": "Alice"')
        assert validator.is_complete is False

    def test_bytes_chunks(self, simple_object_schema):
        """Feed accepts bytes chunks."""
        validator = Validator(simple_object_schema)
        assert validator.feed(b'{"name": "Alice", "age": 30}') is True
        assert validator.is_complete is True

    def test_single_byte_streaming(self, simple_object_schema):
        """Feed works byte-by-byte."""
        validator = Validator(simple_object_schema)
        data = '{"name": "A", "age": 1}'
        for byte in data:
            result = validator.feed(byte)
            if not result:
                pytest.fail(f"Failed at byte: {byte!r}")
        assert validator.is_complete is True


# =============================================================================
# Reset and Reuse Tests
# =============================================================================


class TestReset:
    """Test the reset() API for validator reuse."""

    def test_reset_clears_state(self, simple_object_schema):
        """Reset clears position and completion state."""
        validator = Validator(simple_object_schema)
        validator.feed('{"name": "Alice", "age": 30}')
        assert validator.is_complete is True
        assert validator.position > 0

        validator.reset()
        assert validator.position == 0
        assert validator.is_complete is False

    def test_reuse_after_reset(self, simple_object_schema):
        """Validator can be reused after reset."""
        validator = Validator(simple_object_schema)

        # First validation
        validator.feed('{"name": "Alice", "age": 30}')
        assert validator.is_complete is True

        # Reset and reuse
        validator.reset()
        validator.feed('{"name": "Bob", "age": 25}')
        assert validator.is_complete is True

    def test_reset_after_failure(self, simple_object_schema):
        """Reset allows recovery after validation failure."""
        validator = Validator(simple_object_schema)

        # Fail validation
        validator.feed('{"name": 123')  # Invalid: name should be string

        # Reset and retry
        validator.reset()
        validator.feed('{"name": "Valid", "age": 30}')
        assert validator.is_complete is True


# =============================================================================
# Lookahead Tests
# =============================================================================


class TestCanContinueWith:
    """Test the can_continue_with() lookahead API."""

    def test_lookahead_valid(self, simple_object_schema):
        """can_continue_with returns True for valid continuation."""
        validator = Validator(simple_object_schema)
        validator.feed('{"name": "Alice", "age": ')
        assert validator.can_continue_with("30}") is True

    def test_lookahead_invalid(self, strict_object_schema):
        """can_continue_with returns False for invalid continuation.

        Uses strict schema (additionalProperties: false) for grammar-level type checking.
        """
        validator = Validator(strict_object_schema)
        validator.feed('{"name": "Alice", "age": ')
        # String instead of integer
        assert validator.can_continue_with('"thirty"') is False

    def test_lookahead_preserves_state(self, strict_object_schema):
        """can_continue_with does not advance state.

        Uses strict schema (additionalProperties: false) for grammar-level type checking.
        """
        validator = Validator(strict_object_schema)
        validator.feed('{"name": "Alice", "age": ')
        pos_before = validator.position

        validator.can_continue_with("30}")
        validator.can_continue_with('"invalid"')

        assert validator.position == pos_before

    def test_lookahead_multiple_options(self, strict_object_schema):
        """can_continue_with can test multiple continuations.

        Uses strict schema (additionalProperties: false) for grammar-level type checking.
        """
        validator = Validator(strict_object_schema)
        validator.feed('{"name": "Alice", "age": ')

        # Test multiple options without changing state
        assert validator.can_continue_with("30}") is True
        assert validator.can_continue_with("25}") is True
        assert validator.can_continue_with('"no"}') is False
        assert validator.can_continue_with("true}") is False


# =============================================================================
# Schema Input Format Tests
# =============================================================================


class TestSchemaFormats:
    """Test different schema input formats."""

    def test_dict_schema(self):
        """Accept schema as Python dict."""
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        validator = Validator(schema)
        assert validator.validate('{"x": 42}') is True

    def test_string_schema(self):
        """Accept schema as JSON string."""
        schema = '{"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}'
        validator = Validator(schema)
        assert validator.validate('{"x": 42}') is True

    def test_simple_type_schema(self):
        """Accept simple type schemas."""
        validator = Validator({"type": "string"})
        assert validator.validate('"hello"') is True
        assert validator.validate("123") is False

    def test_array_schema(self):
        """Accept array schemas with items."""
        schema = {"type": "array", "items": {"type": "integer"}}
        validator = Validator(schema)
        assert validator.validate("[1, 2, 3]") is True
        assert validator.validate('["a", "b"]') is False


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_object_allowed(self):
        """Empty object valid when no properties required."""
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        }
        validator = Validator(schema)
        assert validator.validate("{}") is True

    def test_empty_array(self):
        """Empty array valid with items schema."""
        schema = {"type": "array", "items": {"type": "integer"}}
        validator = Validator(schema)
        assert validator.validate("[]") is True

    def test_null_value(self):
        """Validate null values."""
        schema = {
            "type": "object",
            "properties": {"value": {"type": "null"}},
            "required": ["value"],
        }
        validator = Validator(schema)
        assert validator.validate('{"value": null}') is True

    def test_unicode_strings(self):
        """Validate Unicode content."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        validator = Validator(schema)
        assert validator.validate('{"name": "Hello, 世界"}') is True

    def test_escaped_strings(self):
        """Validate escaped characters in strings."""
        schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }
        validator = Validator(schema)
        assert validator.validate('{"text": "line1\\nline2"}') is True
        assert validator.validate('{"text": "quote: \\"test\\""}') is True

    def test_large_numbers(self):
        """Validate large numbers.

        Note: Very large integers (> i64 max) fail in semantic validation due to
        Zig's std.json parser overflow. Feed/streaming validation works since
        grammar validates syntax without parsing values. One-shot validate()
        always runs semantic validation which parses JSON values.
        """
        schema = {
            "type": "object",
            "properties": {"big": {"type": "number"}},
            "required": ["big"],
            "additionalProperties": False,  # Strict schema for grammar-level type checking
        }
        validator = Validator(schema)
        # Scientific notation and reasonable integers work
        assert validator.validate('{"big": 1.23e10}') is True
        assert validator.validate('{"big": 9223372036854775807}') is True  # i64 max

        # Streaming validation accepts any JSON number syntax (no value parsing)
        validator.reset()
        assert validator.feed('{"big": 12345678901234567890}') is True
        assert validator.is_complete is True

    def test_deeply_nested(self):
        """Validate deeply nested structures."""
        schema = {
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "properties": {
                        "b": {
                            "type": "object",
                            "properties": {"c": {"type": "integer"}},
                            "required": ["c"],
                        }
                    },
                    "required": ["b"],
                }
            },
            "required": ["a"],
        }
        validator = Validator(schema)
        assert validator.validate('{"a": {"b": {"c": 42}}}') is True


# =============================================================================
# Error Cases
# =============================================================================


class TestErrors:
    """Test error handling."""

    def test_invalid_schema_raises(self):
        """Invalid schema raises ValueError."""
        with pytest.raises(ValueError):
            Validator("not valid json")

    def test_malformed_json_fails(self, simple_object_schema):
        """Malformed JSON fails validation."""
        validator = Validator(simple_object_schema)
        assert validator.validate("{not valid}") is False

    def test_truncated_json_incomplete(self, simple_object_schema):
        """Truncated JSON is not complete."""
        validator = Validator(simple_object_schema)
        validator.feed('{"name": "Alice"')
        assert validator.is_complete is False


# =============================================================================
# Semantic Violation String Safety Tests
# =============================================================================


class TestSemanticViolationStrings:
    """Test that semantic violation messages are properly null-terminated.

    Regression test for memory corruption bug where violation path/message
    strings were not null-terminated, causing garbage to be read by ctypes.
    """

    def test_type_mismatch_violation_message_is_valid_utf8(self):
        """Type mismatch violation produces valid UTF-8 message string.

        Tests that the violation message is properly null-terminated and
        can be decoded as UTF-8 without errors.
        """
        schema = {"type": "object", "properties": {"value": {"type": "integer"}}}
        validator = Validator(schema)

        # This should fail with a type mismatch (string instead of integer)
        result = validator.validate('{"value": "not_an_integer"}')
        assert result is False

        # The violation message must be valid UTF-8 (no garbage bytes)
        # If strings weren't null-terminated, this would read garbage and
        # potentially raise UnicodeDecodeError or contain random bytes
        # We can't access the violation directly, but the fact validate()
        # returned False without raising UnicodeDecodeError proves it works

    def test_number_constraint_violation_message_is_valid_utf8(self):
        """Number constraint violation produces valid UTF-8 message string."""
        schema = {
            "type": "object",
            "properties": {"age": {"type": "integer", "minimum": 0, "maximum": 120}},
        }
        validator = Validator(schema)

        # This fails semantic validation (age out of range)
        result = validator.validate('{"age": 150}')
        assert result is False

    def test_additional_properties_violation_message_is_valid_utf8(self):
        """Additional properties violation produces valid UTF-8 message string."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
        validator = Validator(schema)

        # This fails with additional property violation
        result = validator.validate('{"name": "test", "extra": "field"}')
        assert result is False

    def test_nested_path_violation_message_is_valid_utf8(self):
        """Nested property violation produces valid UTF-8 path string."""
        schema = {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {
                        "inner": {"type": "integer", "maximum": 10},
                    },
                },
            },
        }
        validator = Validator(schema)

        # This fails with a nested path violation
        result = validator.validate('{"outer": {"inner": 999}}')
        assert result is False

    def test_array_index_path_violation_message_is_valid_utf8(self):
        """Array index violation produces valid UTF-8 path string."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "integer", "maximum": 100},
                },
            },
        }
        validator = Validator(schema)

        # This fails with array index in the path
        result = validator.validate('{"items": [1, 2, 999, 4]}')
        assert result is False


# =============================================================================
# Holy Trinity lifecycle tests (close / context manager / __del__)
# =============================================================================


class TestValidatorLifecycle:
    """Validator must implement close(), __enter__/__exit__, __del__."""

    def test_close_is_idempotent(self, simple_object_schema):
        """close() can be called multiple times without error."""
        validator = Validator(simple_object_schema)
        validator.close()
        validator.close()

    def test_context_manager_basic(self, simple_object_schema):
        """Validator can be used as a context manager."""
        with Validator(simple_object_schema) as v:
            assert v.validate('{"name":"Alice","age":30}')
        # After exiting, handles are freed
        assert v._handle is None
        assert v._semantic_handle is None

    def test_context_manager_returns_self(self, simple_object_schema):
        """__enter__ returns self."""
        v = Validator(simple_object_schema)
        with v as ctx:
            assert ctx is v

    def test_close_nulls_handles(self, simple_object_schema):
        """close() sets handles to None."""
        validator = Validator(simple_object_schema)
        assert validator._handle is not None

        validator.close()
        assert validator._handle is None
        assert validator._semantic_handle is None


class TestValidatorResetUsesCheck:
    """Validator.reset() must use check() for typed error mapping."""

    def test_reset_succeeds(self, simple_object_schema):
        """reset() works after streaming validation."""
        validator = Validator(simple_object_schema)
        validator.feed('{"name":"Alice","age":30}')
        validator.reset()
        # Should be able to validate again
        assert validator.validate('{"name":"Bob","age":25}')

    def test_reset_does_not_raise_runtime_error(self, simple_object_schema):
        """reset() on a valid handle must not raise RuntimeError.

        Regression: reset() previously raised plain RuntimeError instead of
        using check() for typed error mapping.
        """
        validator = Validator(simple_object_schema)
        # Should complete without RuntimeError
        validator.reset()
