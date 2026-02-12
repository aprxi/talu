"""Test strict mode and error reporting for streaming validation."""

import pytest

from talu import Validator
from talu.exceptions import StreamingValidationError

# =============================================================================
# Strict Mode Basics
# =============================================================================


class TestStrictMode:
    """Test strict=True parameter on feed()."""

    @pytest.fixture
    def user_schema(self):
        """Schema for user with name and age.

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

    def test_strict_raises_on_type_violation(self, user_schema):
        """strict=True raises StreamingValidationError on type violation."""
        validator = Validator(user_schema)

        # Valid prefix
        validator.feed('{"name": "Alice", "age": ', strict=True)

        # Type violation: string instead of integer
        with pytest.raises(StreamingValidationError) as exc_info:
            validator.feed('"twenty"', strict=True)

        error = exc_info.value
        assert error.position > 0
        assert error.invalid_byte == b'"'
        assert (
            "digit" in error.expected or "'-'" in error.expected
        )  # integer starts with digit or minus

    def test_strict_false_returns_bool(self, user_schema):
        """strict=False (default) returns False without raising."""
        validator = Validator(user_schema)

        validator.feed('{"name": "Alice", "age": ')
        result = validator.feed('"twenty"', strict=False)

        assert result is False
        # Error is available via property
        assert validator.error is not None

    def test_strict_default_is_false(self, user_schema):
        """Default strict is False for backward compatibility."""
        validator = Validator(user_schema)

        validator.feed('{"name": "Alice", "age": ')
        # This should not raise
        result = validator.feed('"twenty"')

        assert result is False

    def test_strict_valid_data_no_exception(self, user_schema):
        """strict=True doesn't raise on valid data."""
        validator = Validator(user_schema)

        # All valid chunks
        validator.feed('{"name": "Alice", ', strict=True)
        validator.feed('"age": 30}', strict=True)

        assert validator.is_complete is True

    def test_strict_with_bytes_input(self, user_schema):
        """strict=True works with bytes input."""
        validator = Validator(user_schema)

        validator.feed(b'{"name": "Bob", "age": ', strict=True)

        with pytest.raises(StreamingValidationError):
            validator.feed(b'"old"', strict=True)


# =============================================================================
# Error Property
# =============================================================================


class TestErrorProperty:
    """Test validator.error property."""

    @pytest.fixture
    def int_schema(self):
        """Simple integer schema.

        Uses additionalProperties: false for strict streaming validation -
        grammar enforces types at parse time for immediate error detection.
        """
        return {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "additionalProperties": False,
        }

    def test_error_is_none_before_failure(self, int_schema):
        """error property is None before any failure."""
        validator = Validator(int_schema)
        assert validator.error is None

        validator.feed('{"x": ')
        assert validator.error is None

    def test_error_available_after_failure(self, int_schema):
        """error property available after validation failure."""
        validator = Validator(int_schema)

        validator.feed('{"x": ')
        validator.feed('"str"')  # Type violation

        error = validator.error
        assert error is not None
        assert isinstance(error, StreamingValidationError)
        assert error.invalid_byte == b'"'

    def test_error_cleared_on_reset(self, int_schema):
        """error property cleared after reset()."""
        validator = Validator(int_schema)

        validator.feed('{"x": ')
        validator.feed('"str"')
        assert validator.error is not None

        validator.reset()
        assert validator.error is None

    def test_error_contains_position(self, int_schema):
        """error contains correct position."""
        validator = Validator(int_schema)

        validator.feed('{"x": ')  # 6 bytes
        validator.feed('"str"')  # Fails at first byte

        error = validator.error
        assert error is not None
        assert error.position == 6  # Position in stream where failure occurred

    def test_error_contains_context(self, int_schema):
        """error contains context around failure."""
        validator = Validator(int_schema)

        validator.feed('{"x": ')
        validator.feed('"string value"')

        error = validator.error
        assert error is not None
        assert len(error.context) > 0


# =============================================================================
# Error Message Quality
# =============================================================================


class TestErrorMessageQuality:
    """Test that error messages are helpful."""

    def test_expected_describes_valid_bytes(self):
        """expected field describes what bytes would be valid."""
        schema = {
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "additionalProperties": False,
        }
        validator = Validator(schema)

        validator.feed('{"n": ')
        validator.feed("abc")  # Invalid: letters where number expected

        error = validator.error
        assert error is not None
        # Should mention digit or minus for integer start
        assert "digit" in error.expected or "'-'" in error.expected

    def test_expected_for_string_start(self):
        """expected shows quote when string expected."""
        schema = {
            "type": "object",
            "properties": {"s": {"type": "string"}},
            "additionalProperties": False,
        }
        validator = Validator(schema)

        validator.feed('{"s": ')
        validator.feed("123")  # Number where string expected

        error = validator.error
        assert error is not None
        # Should mention quote for string start
        assert '"' in error.expected or "quote" in error.expected.lower()

    def test_expected_for_object_value(self):
        """expected shows valid value starters after colon."""
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": False,
        }
        validator = Validator(schema)

        validator.feed('{"x": ')
        validator.feed("}")  # Invalid: closing brace where value expected

        error = validator.error
        assert error is not None
        # Position should be tracked
        assert error.position > 0


# =============================================================================
# Exception Hierarchy
# =============================================================================


class TestExceptionHierarchy:
    """Test StreamingValidationError exception hierarchy."""

    def test_inherits_from_structured_output_error(self):
        """StreamingValidationError inherits from StructuredOutputError."""
        from talu.exceptions import StructuredOutputError

        schema = {"type": "integer"}
        validator = Validator(schema)

        validator.feed('"str"')

        error = validator.error
        assert isinstance(error, StructuredOutputError)

    def test_inherits_from_talu_error(self):
        """StreamingValidationError inherits from TaluError."""
        from talu.exceptions import TaluError

        schema = {"type": "integer"}
        validator = Validator(schema)

        validator.feed('"str"')

        error = validator.error
        assert isinstance(error, TaluError)

    def test_has_code_attribute(self):
        """Error has code attribute for programmatic handling."""
        schema = {"type": "integer"}
        validator = Validator(schema)

        with pytest.raises(StreamingValidationError) as exc_info:
            validator.feed('"str"', strict=True)

        assert exc_info.value.code == "STREAMING_VALIDATION_FAILED"

    def test_has_details_attribute(self):
        """Error has details dict for structured context."""
        schema = {"type": "integer"}
        validator = Validator(schema)

        with pytest.raises(StreamingValidationError) as exc_info:
            validator.feed('"str"', strict=True)

        details = exc_info.value.details
        assert "position" in details
        assert "invalid_byte" in details
        assert "expected" in details


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_failure_at_first_byte(self):
        """Error when very first byte is invalid."""
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "additionalProperties": False,
        }
        validator = Validator(schema)

        with pytest.raises(StreamingValidationError) as exc_info:
            validator.feed("[", strict=True)  # Array where object expected

        error = exc_info.value
        assert error.position == 0
        assert error.invalid_byte == b"["

    def test_failure_mid_chunk(self):
        """Error position correct when failure is mid-chunk."""
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "additionalProperties": False,
        }
        validator = Validator(schema)

        # Feed a chunk where failure is partway through
        with pytest.raises(StreamingValidationError) as exc_info:
            validator.feed('{"a": "xyz"}', strict=True)

        error = exc_info.value
        # Should fail at the quote character
        assert error.invalid_byte == b'"'

    def test_context_shows_surrounding_bytes(self):
        """Context field shows bytes around the failure."""
        schema = {
            "type": "object",
            "properties": {"val": {"type": "integer"}},
            "additionalProperties": False,
        }
        validator = Validator(schema)

        validator.feed('{"val": ')
        validator.feed('"not a number"')

        error = validator.error
        assert error is not None
        # Context should contain the failing chunk or part of it
        assert len(error.context) > 0

    def test_multiple_failures_tracked_correctly(self):
        """After reset, new failure is tracked correctly."""
        schema = {"type": "integer"}
        validator = Validator(schema)

        # First failure
        validator.feed('"first"')
        first_error = validator.error
        assert first_error is not None
        assert first_error.invalid_byte == b'"'

        # Reset and new failure
        validator.reset()
        validator.feed("[1]")
        second_error = validator.error
        assert second_error is not None
        assert second_error.invalid_byte == b"["
        assert second_error.position == 0  # Fresh start


# =============================================================================
# Valid Bytes Timing (Regression Test)
# =============================================================================


class TestValidBytesTiming:
    """Test that valid_bytes is queried at failure point, not chunk start.

    This is a regression test for a bug where valid_bytes was cached BEFORE
    advancing the engine, resulting in incorrect "expected" values.

    Example: For '{"age": "five"}' with schema {"age": integer}:
    - Engine advances through '{"age": ' successfully
    - Engine stops at '"' (start of "five")
    - At this point, engine expects: digit, minus (for integer)
    - Bug would report: '{' (what was valid at chunk start)
    """

    def test_expected_reflects_failure_point_not_chunk_start(self):
        """Verify expected bytes are from failure point, not chunk start."""
        schema = {
            "type": "object",
            "properties": {"age": {"type": "integer"}},
            "required": ["age"],
            "additionalProperties": False,
        }
        validator = Validator(schema)

        # Feed entire invalid JSON in one chunk
        # Engine will advance through '{"age": ' then fail at '"'
        with pytest.raises(StreamingValidationError) as exc_info:
            validator.feed('{"age": "five"}', strict=True)

        error = exc_info.value
        # At failure point (the '"' before "five"), engine expects integer start
        # Integer can start with: digit (0-9) or minus sign
        assert "digit" in error.expected or "'-'" in error.expected
        # It should NOT expect '{' (which was valid at position 0)
        assert "'{'" not in error.expected

    def test_expected_after_partial_consumption(self):
        """Verify correct expected after engine consumes part of chunk."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["name", "count"],
            "additionalProperties": False,
        }
        validator = Validator(schema)

        # First chunk is valid
        validator.feed('{"name": "test", "count": ', strict=True)

        # Second chunk: engine will try to parse, fail at 'n' of 'null'
        # (null is not valid for integer type)
        with pytest.raises(StreamingValidationError) as exc_info:
            validator.feed("null}", strict=True)

        error = exc_info.value
        # At 'n', engine expected integer (digit or minus)
        assert "digit" in error.expected or "'-'" in error.expected
        # Should NOT show what was valid before this chunk
        assert error.invalid_byte == b"n"

    def test_error_property_also_uses_failure_point(self):
        """Verify error property (non-strict) also gets failure point state."""
        schema = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "additionalProperties": False,
        }
        validator = Validator(schema)

        # Feed chunk that fails partway through
        result = validator.feed('{"value": "wrong"}')
        assert result is False

        error = validator.error
        assert error is not None
        # Should expect integer at the '"' before "wrong"
        assert "digit" in error.expected or "'-'" in error.expected
        assert error.invalid_byte == b'"'


# =============================================================================
# Integration with Streaming Patterns
# =============================================================================


class TestStreamingIntegration:
    """Test strict mode in realistic streaming scenarios."""

    def test_llm_streaming_with_early_abort(self):
        """Simulate LLM streaming with early abort on violation."""
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["answer", "confidence"],
            "additionalProperties": False,
        }
        validator = Validator(schema)

        # Simulate LLM token stream
        tokens = [
            '{"answer": "The answer is ',
            "42",
            '", "confidence": ',
            '"high"',  # Error: string instead of number
        ]

        received = []
        error_at_token = None

        for i, token in enumerate(tokens):
            try:
                validator.feed(token, strict=True)
                received.append(token)
            except StreamingValidationError as e:
                error_at_token = i
                # Early abort - stop processing
                assert "digit" in e.expected or "'-'" in e.expected
                break

        assert error_at_token == 3  # Failed at "high"
        assert len(received) == 3  # Received 3 valid tokens before failure
