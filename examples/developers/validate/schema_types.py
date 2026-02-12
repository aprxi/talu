"""Schema Types - Different ways to specify validation schemas.

Primary API: talu.Validator
Scope: Single

Validator accepts schemas in multiple formats, making it easy to
integrate with existing code regardless of how you define schemas.

Related:
    - examples/basics/20_streaming_validation.py
"""

from talu import Validator


# =============================================================================
# JSON Schema Dict (Zero Dependencies)
# =============================================================================

# The simplest approach - just a Python dict
# No Pydantic, no external libraries needed

dict_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
        "tags": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["name", "age"],
}

validator = Validator(dict_schema)
assert validator.validate('{"name": "Alice", "age": 30}')
assert validator.validate('{"name": "Bob", "age": 25, "tags": ["admin"]}')
assert not validator.validate('{"name": "Eve", "age": -5}')  # age < 0


# =============================================================================
# JSON Schema String
# =============================================================================

# You can also pass a JSON string directly
# Useful when loading schemas from files or APIs

json_string = '''{
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "active": {"type": "boolean"}
    },
    "required": ["id"]
}'''

validator = Validator(json_string)
assert validator.validate('{"id": 1, "active": true}')
assert validator.validate('{"id": 2}')
assert not validator.validate('{"active": true}')  # missing id


# =============================================================================
# Pydantic Models (Optional)
# =============================================================================

# If you have Pydantic, you can pass models directly
# The schema is extracted via model_json_schema()

# Uncomment if Pydantic is available:
#
# from pydantic import BaseModel, Field
#
# class User(BaseModel):
#     name: str
#     age: int = Field(ge=0, le=150)
#     email: str | None = None
#
# validator = Validator(User)
# assert validator.validate('{"name": "Alice", "age": 30}')


# =============================================================================
# Nested Objects
# =============================================================================

nested_schema = {
    "type": "object",
    "properties": {
        "user": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "profile": {
                    "type": "object",
                    "properties": {
                        "bio": {"type": "string"},
                    },
                },
            },
            "required": ["name"],
        },
    },
    "required": ["user"],
}

validator = Validator(nested_schema)
assert validator.validate('{"user": {"name": "Alice", "profile": {"bio": "Developer"}}}')
assert validator.validate('{"user": {"name": "Bob"}}')
assert not validator.validate('{"user": {}}')  # missing name


# =============================================================================
# Arrays
# =============================================================================

array_schema = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                },
                "required": ["id"],
            },
        },
    },
}

validator = Validator(array_schema)
assert validator.validate('{"items": [{"id": 1, "name": "a"}, {"id": 2}]}')
assert not validator.validate('{"items": [{"name": "missing id"}]}')


# =============================================================================
# Enums and Const
# =============================================================================

enum_schema = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["pending", "active", "completed"],
        },
        "version": {
            "const": 1,
        },
    },
    "required": ["status", "version"],
}

validator = Validator(enum_schema)
assert validator.validate('{"status": "active", "version": 1}')
assert not validator.validate('{"status": "unknown", "version": 1}')
assert not validator.validate('{"status": "active", "version": 2}')


# =============================================================================
# Typed Dictionaries (additionalProperties)
# =============================================================================

# For dynamic key-value maps where values must match a type.
# Common use case: LLM extracts entity scores with unknown keys.

scores_schema = {
    "type": "object",
    "additionalProperties": {"type": "number", "minimum": -1, "maximum": 1},
}

validator = Validator(scores_schema)
assert validator.validate('{"food": 0.8, "service": -0.3, "ambiance": 0.5}')
assert not validator.validate('{"food": "great"}')  # string, not number
assert not validator.validate('{"food": 5.0}')  # exceeds maximum

# Can combine with defined properties
config_schema = {
    "type": "object",
    "properties": {
        "version": {"type": "integer"},
    },
    "additionalProperties": {"type": "string"},
    "required": ["version"],
}

validator = Validator(config_schema)
assert validator.validate('{"version": 1, "env": "prod", "region": "us-east"}')
assert not validator.validate('{"version": 1, "env": 123}')  # 123 not string


print("All schema type tests passed!")


"""
Topics covered:
* structured.output
* schema.pydantic
* parsing.typed
"""
