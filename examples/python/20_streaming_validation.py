"""Validate JSON as it streams in.

This example shows:
- Validating complete JSON against a schema (one-shot)
- Streaming syntax validation for early abort on malformed JSON
- Combining streaming syntax checks with final schema validation
"""

from talu import Validator


# =============================================================================
# Define schema as a dict (no dependencies needed)
# =============================================================================

user_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"},
    },
    "required": ["name", "age"],
}


# =============================================================================
# Complete validation (one-shot)
# =============================================================================

print("--- One-shot validation ---")

validator = Validator(user_schema)

print(f"Valid:         {validator.validate('{\"name\": \"Alice\", \"age\": 30}')}")
print(f"Wrong type:    {validator.validate('{\"name\": \"Alice\", \"age\": \"thirty\"}')}")
print(f"Missing field: {validator.validate('{\"name\": \"Alice\"}')}")


# =============================================================================
# Streaming syntax validation
# =============================================================================

# feed() validates JSON syntax as chunks arrive.
# This catches malformed JSON early (missing quotes, bad escapes, etc.)
# but does NOT check schema types — that requires the full document.

print("\n--- Streaming valid JSON ---")

chunks = [
    '{"name": "',
    'Alice",',
    ' "age": ',
    '30}',
]

validator = Validator(user_schema)
for chunk in chunks:
    valid = validator.feed(chunk)
    print(f"  Fed {chunk!r}: valid={valid}")

print(f"  Complete: {validator.is_complete}")


# =============================================================================
# Early abort on malformed JSON
# =============================================================================

print("\n--- Streaming malformed JSON (deliberately bad) ---")

bad_syntax_chunks = [
    '{"name": "',
    'Bob", "age": ',
    '}',  # Syntax error: missing value after colon
]

validator = Validator(user_schema)
for chunk in bad_syntax_chunks:
    valid = validator.feed(chunk)
    print(f"  Fed {chunk!r}: valid={valid}")
    if not valid:
        print(f"  -> Detected syntax error at position {validator.position}")
        break


# =============================================================================
# Practical use: stream then validate
# =============================================================================

print("\n--- Practical: stream + validate helper ---")


def validate_api_stream(chunks, schema):
    """Validate streaming response: syntax during stream, schema at end."""
    validator = Validator(schema)
    buffer = ""

    for chunk in chunks:
        buffer += chunk
        if not validator.feed(chunk):
            raise ValueError(f"Malformed JSON at position {validator.position}")

    if not validator.is_complete:
        raise ValueError("Incomplete JSON response")

    # Full schema validation once the document is complete
    if not validator.validate(buffer):
        raise ValueError(f"Schema validation failed for: {buffer}")

    return buffer


# 1) Valid stream passes both checks
result = validate_api_stream(chunks, user_schema)
print(f"  Valid stream: {result}")

# 2) Malformed JSON caught during streaming
try:
    validate_api_stream(bad_syntax_chunks, user_schema)
except ValueError as e:
    print(f"  Malformed stream (expected): {e}")

# 3) Structurally valid but wrong types — caught by final schema check
wrong_type_chunks = [
    '{"name": "Bob", ',
    '"age": "thirty"}',
]

try:
    validate_api_stream(wrong_type_chunks, user_schema)
except ValueError as e:
    print(f"  Wrong types (expected): {e}")
