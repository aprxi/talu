"""Streaming Validation - Validate JSON incrementally as it arrives.

Primary API: talu.Validator
Scope: Single

The key differentiator of talu's validation is streaming support.
Unlike Pydantic or jsonschema which require complete JSON, talu
can validate byte-by-byte, enabling early abort on violations.

Use strict=True for rich error diagnostics (position, invalid byte,
expected bytes, context) via StreamingValidationError.

Related:
    - examples/basics/20_streaming_validation.py
"""

from talu import Validator


# =============================================================================
# Core Streaming API
# =============================================================================

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "value": {"type": "integer"},
    },
    "required": ["name", "value"],
}

validator = Validator(schema)

# feed() returns True if chunk was valid, False on violation
assert validator.feed('{"name":') is True
assert validator.feed(' "test",') is True
assert validator.feed(' "value": 42}') is True

# is_complete tells you if we have valid, complete JSON
assert validator.is_complete is True

# position tells you how many bytes were consumed
print(f"Consumed {validator.position} bytes")


# =============================================================================
# Detecting Violations Mid-Stream
# =============================================================================

validator = Validator(schema)

# Valid so far
assert validator.feed('{"name": "test", "value": ') is True

# Now we get a string instead of integer - violation!
valid = validator.feed('"oops"')
assert valid is False

# position tells us where the violation occurred
print(f"Violation at byte {validator.position}")


# =============================================================================
# Reusing Validators
# =============================================================================

# Validators can be reset and reused
validator = Validator(schema)

# First validation
validator.feed('{"name": "a", "value": 1}')
assert validator.is_complete

# Reset for new validation
validator.reset()
assert validator.position == 0
assert validator.is_complete is False

# Second validation
validator.feed('{"name": "b", "value": 2}')
assert validator.is_complete


# =============================================================================
# Lookahead Without Advancing
# =============================================================================

validator = Validator(schema)
validator.feed('{"name": "test", "value": ')

# can_continue_with() checks validity without advancing state
assert validator.can_continue_with('42') is True   # integer would be valid
assert validator.can_continue_with('"x"') is False  # string would not

# State hasn't changed - we're still at the same position
print(f"Still at position {validator.position}")


# =============================================================================
# Strict Mode: Rich Error Diagnostics
# =============================================================================

from talu.exceptions import StreamingValidationError

validator = Validator(schema)

# strict=True raises StreamingValidationError with rich diagnostics
try:
    validator.feed('{"name": "test", "value": ', strict=True)
    validator.feed('"oops"', strict=True)  # string where int expected
except StreamingValidationError as e:
    print(f"Position: {e.position}")
    print(f"Invalid byte: {e.invalid_byte!r}")
    print(f"Expected: {e.expected}")  # Human-readable description
    print(f"Context: {e.context}")    # Surrounding bytes

# Alternative: check error property after failed feed()
validator = Validator(schema)
validator.feed('{"name": "test", "value": ')
if not validator.feed('"oops"'):
    error = validator.error  # Rich error available
    print(f"Expected: {error.expected}")


# =============================================================================
# Practical: Streaming API Consumer
# =============================================================================

def consume_stream_with_validation(stream, schema, strict=False):
    """
    Consume a streaming response with validation.

    Aborts early if schema violation detected, saving API costs
    and providing fast failure.

    Args:
        stream: Iterable of JSON chunks
        schema: JSON Schema dict
        strict: If True, raises StreamingValidationError with diagnostics
    """
    validator = Validator(schema)
    buffer = []

    for chunk in stream:
        if strict:
            validator.feed(chunk, strict=True)  # Raises with rich error
        elif not validator.feed(chunk):
            # Violation detected - abort and report
            raise ValueError(
                f"Schema violation at byte {validator.position}. "
                f"Received so far: {''.join(buffer)}"
            )
        buffer.append(chunk)

    if not validator.is_complete:
        raise ValueError("Stream ended with incomplete JSON")

    return "".join(buffer)


# Simulate a valid stream
valid_stream = ['{"name":', ' "ok",', ' "value":', ' 100}']
result = consume_stream_with_validation(valid_stream, schema)
print(f"Valid: {result}")

# Simulate an invalid stream
invalid_stream = ['{"name":', ' 123']  # name should be string
try:
    consume_stream_with_validation(invalid_stream, schema)
except ValueError as e:
    print(f"Caught: {e}")

# With strict mode for better debugging
try:
    consume_stream_with_validation(invalid_stream, schema, strict=True)
except StreamingValidationError as e:
    print(f"Rich error: {e}")


"""
Topics covered:
* validate.streaming
* validate.reuse
* validate.strict
* validate.error
"""
