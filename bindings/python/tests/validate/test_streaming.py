"""Test streaming validation patterns."""

import pytest

from talu import Validator

# =============================================================================
# Practical Streaming Patterns
# =============================================================================


class TestStreamingPatterns:
    """Test real-world streaming validation patterns."""

    @pytest.fixture
    def api_response_schema(self):
        """Schema for typical API response.

        Uses additionalProperties: false for strict streaming validation -
        grammar enforces types at parse time for immediate error detection.
        """
        return {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
                "sources": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "confidence"],
            "additionalProperties": False,
        }

    def test_valid_streaming_response(self, api_response_schema):
        """Validate a typical streaming API response."""
        validator = Validator(api_response_schema)

        chunks = [
            '{"answer": "',
            "The answer is 42.",
            '", "confidence": ',
            "0.95, ",
            '"sources": ["wiki"]}',
        ]

        for chunk in chunks:
            assert validator.feed(chunk) is True

        assert validator.is_complete is True

    def test_early_abort_on_violation(self, api_response_schema):
        """Detect violation early and abort."""
        validator = Validator(api_response_schema)

        # Valid start
        assert validator.feed('{"answer": "test", "confidence": ') is True

        # Violation: string instead of number
        result = validator.feed('"high"')
        assert result is False

        # Position tells us where it failed
        assert validator.position > 0

    def test_accumulate_and_validate(self, api_response_schema):
        """Pattern: accumulate chunks while validating."""
        validator = Validator(api_response_schema)
        buffer = []

        chunks = ['{"answer": "ok", ', '"confidence": 0.8}']

        for chunk in chunks:
            if not validator.feed(chunk):
                pytest.fail(f"Validation failed at: {''.join(buffer)}")
            buffer.append(chunk)

        assert "".join(buffer) == '{"answer": "ok", "confidence": 0.8}'
        assert validator.is_complete is True


# =============================================================================
# Chunk Size Variations
# =============================================================================


class TestChunkSizes:
    """Test various chunk sizes."""

    @pytest.fixture
    def schema(self):
        """Simple schema for chunk tests."""
        return {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        }

    def test_single_chunk(self, schema):
        """Entire JSON in one chunk."""
        validator = Validator(schema)
        assert validator.feed('{"value": 42}') is True
        assert validator.is_complete is True

    def test_character_by_character(self, schema):
        """One character at a time."""
        validator = Validator(schema)
        data = '{"value": 42}'

        for char in data:
            result = validator.feed(char)
            if not result:
                pytest.fail(f"Failed at character: {char!r}")

        assert validator.is_complete is True

    def test_varying_chunk_sizes(self, schema):
        """Mixed chunk sizes."""
        validator = Validator(schema)

        # Varying sizes
        assert validator.feed("{") is True
        assert validator.feed('"val') is True
        assert validator.feed('ue": ') is True
        assert validator.feed("42}") is True
        assert validator.is_complete is True


# =============================================================================
# Partial JSON States
# =============================================================================


class TestPartialStates:
    """Test validation at various partial states."""

    @pytest.fixture
    def schema(self):
        """Schema for partial state tests."""
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "items": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["name"],
        }

    def test_mid_string(self, schema):
        """Validation mid-string."""
        validator = Validator(schema)
        assert validator.feed('{"name": "hel') is True
        assert validator.is_complete is False
        assert validator.feed('lo"}') is True
        assert validator.is_complete is True

    def test_mid_array(self, schema):
        """Validation mid-array."""
        validator = Validator(schema)
        assert validator.feed('{"name": "x", "items": [1, 2') is True
        assert validator.is_complete is False
        assert validator.feed(", 3]}") is True
        assert validator.is_complete is True

    def test_mid_number(self, schema):
        """Validation mid-number."""
        validator = Validator(schema)
        assert validator.feed('{"name": "x", "items": [12') is True
        assert validator.is_complete is False
        assert validator.feed("34]}") is True
        assert validator.is_complete is True


# =============================================================================
# Whitespace Handling
# =============================================================================


class TestWhitespace:
    """Test whitespace handling in streams."""

    @pytest.fixture
    def schema(self):
        """Simple schema."""
        return {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }

    def test_whitespace_between_tokens(self, schema):
        """Handle whitespace between tokens."""
        validator = Validator(schema)
        assert validator.feed("{  ") is True
        assert validator.feed('"x"  ') is True
        assert validator.feed(":  ") is True
        assert validator.feed("1  ") is True
        assert validator.feed("}") is True
        assert validator.is_complete is True

    def test_pretty_printed_json(self, schema):
        """Handle pretty-printed JSON (whitespace inside)."""
        validator = Validator(schema)
        # Feed the whole pretty-printed JSON
        pretty = '{\n    "x": 42\n}'
        assert validator.feed(pretty) is True
        assert validator.is_complete is True


# =============================================================================
# Multiple Validations
# =============================================================================


class TestMultipleValidations:
    """Test multiple sequential validations."""

    def test_many_validations_same_schema(self):
        """Reuse validator for many documents."""
        schema = {
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        }
        validator = Validator(schema)

        for i in range(100):
            validator.reset()
            assert validator.feed(f'{{"n": {i}}}') is True
            assert validator.is_complete is True

    def test_alternating_valid_invalid(self):
        """Alternate between valid and invalid inputs."""
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        validator = Validator(schema)

        # Valid
        validator.feed('{"x": 1}')
        assert validator.is_complete is True

        # Invalid
        validator.reset()
        validator.feed('{"x": "not int"}')
        # May or may not be complete depending on when violation detected

        # Valid again
        validator.reset()
        validator.feed('{"x": 2}')
        assert validator.is_complete is True
