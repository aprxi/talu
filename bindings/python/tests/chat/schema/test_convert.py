"""
Tests for talu.chat.schema.convert - Dataclass/TypedDict/Pydantic to JSON Schema conversion.

This module tests type conversion from Python types to JSON Schema.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TypedDict

import pytest

from talu.exceptions import StructuredOutputError, ValidationError
from talu.router.schema.convert import (
    MAX_SCHEMA_DEPTH,
    AmbiguousUnionWarning,
    dataclass_to_schema,
    normalize_response_format,
    typeddict_to_schema,
)

# =============================================================================
# Test Fixtures - Dataclasses
# =============================================================================


@dataclass
class SimpleResponse:
    """Simple dataclass with basic types."""

    name: str
    age: int


@dataclass
class OptionalFieldResponse:
    """Dataclass with optional/default fields."""

    name: str
    nickname: str = "anonymous"


@dataclass
class ComplexResponse:
    """Dataclass with nested types."""

    items: list[str]
    count: int
    score: float
    enabled: bool


@dataclass
class NestedResponse:
    """Dataclass with nested dataclass."""

    user: SimpleResponse
    active: bool


@dataclass
class ListOfDataclassResponse:
    """Dataclass with list of dataclasses."""

    users: list[SimpleResponse]


class StatusEnum(Enum):
    """Enum for testing."""

    ACTIVE = "active"
    INACTIVE = "inactive"


@dataclass
class EnumFieldResponse:
    """Dataclass with enum field."""

    status: StatusEnum


@dataclass
class DescribedFieldResponse:
    """Dataclass with field descriptions."""

    name: str = field(metadata={"description": "The user's full name"})


# =============================================================================
# Test Fixtures - TypedDicts
# =============================================================================


class SimpleTypedDict(TypedDict):
    """Simple TypedDict with basic types."""

    name: str
    age: int


class PartialTypedDict(TypedDict, total=False):
    """TypedDict with all optional fields."""

    name: str
    age: int


class NestedTypedDict(TypedDict):
    """TypedDict with nested TypedDict."""

    user: SimpleTypedDict
    active: bool


class ListTypedDict(TypedDict):
    """TypedDict with list field."""

    items: list[str]
    tags: list[SimpleTypedDict]


# =============================================================================
# dataclass_to_schema() Tests
# =============================================================================


class TestDataclassToSchema:
    """Tests for dataclass_to_schema()."""

    def test_simple_dataclass(self):
        """Converts simple dataclass with basic types."""
        schema = dataclass_to_schema(SimpleResponse)

        assert schema["type"] == "object"
        assert schema["title"] == "SimpleResponse"
        assert "name" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["age"]["type"] == "integer"
        assert set(schema["required"]) == {"name", "age"}

    def test_optional_fields_not_required(self):
        """Fields with defaults are not required."""
        schema = dataclass_to_schema(OptionalFieldResponse)

        assert "name" in schema["required"]
        assert "nickname" not in schema.get("required", [])

    def test_complex_types(self):
        """Handles list, float, bool types."""
        schema = dataclass_to_schema(ComplexResponse)

        assert schema["properties"]["items"]["type"] == "array"
        assert schema["properties"]["items"]["items"]["type"] == "string"
        assert schema["properties"]["count"]["type"] == "integer"
        assert schema["properties"]["score"]["type"] == "number"
        assert schema["properties"]["enabled"]["type"] == "boolean"

    def test_nested_dataclass(self):
        """Converts nested dataclass to nested object schema."""
        schema = dataclass_to_schema(NestedResponse)

        user_schema = schema["properties"]["user"]
        assert user_schema["type"] == "object"
        assert "name" in user_schema["properties"]

    def test_list_of_dataclass(self):
        """Converts list of dataclasses."""
        schema = dataclass_to_schema(ListOfDataclassResponse)

        users_schema = schema["properties"]["users"]
        assert users_schema["type"] == "array"
        assert users_schema["items"]["type"] == "object"
        assert "name" in users_schema["items"]["properties"]

    def test_enum_field(self):
        """Converts enum to string with enum values."""
        schema = dataclass_to_schema(EnumFieldResponse)

        status_schema = schema["properties"]["status"]
        assert status_schema["type"] == "string"
        assert set(status_schema["enum"]) == {"active", "inactive"}

    def test_field_description_metadata(self):
        """Field metadata description is included in schema."""
        schema = dataclass_to_schema(DescribedFieldResponse)

        name_schema = schema["properties"]["name"]
        assert name_schema["description"] == "The user's full name"

    def test_non_dataclass_raises_valueerror(self):
        """Raises ValueError for non-dataclass input."""
        with pytest.raises(ValueError, match="Target must be a dataclass"):
            dataclass_to_schema(str)

    def test_instance_works_but_title_differs(self):
        """Instance is accepted but title uses repr instead of class name."""
        instance = SimpleResponse(name="test", age=25)
        # dataclasses.is_dataclass returns True for instances too
        schema = dataclass_to_schema(instance)
        # It works because is_dataclass is True for instances
        # But the title is the instance repr, not the class name
        assert "SimpleResponse" in schema["title"]
        # Schema still has correct properties
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]


# =============================================================================
# typeddict_to_schema() Tests
# =============================================================================


class TestTypedDictToSchema:
    """Tests for typeddict_to_schema()."""

    def test_simple_typeddict(self):
        """Converts simple TypedDict with basic types."""
        schema = typeddict_to_schema(SimpleTypedDict)

        assert schema["type"] == "object"
        assert schema["title"] == "SimpleTypedDict"
        assert "name" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["age"]["type"] == "integer"
        assert set(schema["required"]) == {"name", "age"}

    def test_partial_typeddict_no_required(self):
        """TypedDict with total=False has no required fields."""
        schema = typeddict_to_schema(PartialTypedDict)

        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]
        assert "required" not in schema  # No required fields

    def test_nested_typeddict(self):
        """Converts nested TypedDict to nested object schema."""
        schema = typeddict_to_schema(NestedTypedDict)

        user_schema = schema["properties"]["user"]
        assert user_schema["type"] == "object"
        assert "name" in user_schema["properties"]
        assert "age" in user_schema["properties"]

    def test_list_of_typeddict(self):
        """Converts list of TypedDict."""
        schema = typeddict_to_schema(ListTypedDict)

        items_schema = schema["properties"]["items"]
        assert items_schema["type"] == "array"
        assert items_schema["items"]["type"] == "string"

        tags_schema = schema["properties"]["tags"]
        assert tags_schema["type"] == "array"
        assert tags_schema["items"]["type"] == "object"
        assert "name" in tags_schema["items"]["properties"]

    def test_non_typeddict_raises_valueerror(self):
        """Raises ValueError for non-TypedDict input."""
        with pytest.raises(ValueError, match="Target must be a TypedDict"):
            typeddict_to_schema(str)


# =============================================================================
# normalize_response_format() Tests
# =============================================================================


class TestNormalizeResponseFormat:
    """Tests for normalize_response_format()."""

    def test_none_returns_none(self):
        """None input returns None."""
        result = normalize_response_format(None)
        assert result is None

    def test_dict_passthrough(self):
        """Dict schema is passed through unchanged."""
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        result = normalize_response_format(schema)

        assert result == schema

    def test_dataclass_converted(self):
        """Dataclass is converted to JSON Schema."""
        result = normalize_response_format(SimpleResponse)

        assert result["type"] == "object"
        assert "name" in result["properties"]

    def test_typeddict_converted(self):
        """TypedDict is converted to JSON Schema."""
        result = normalize_response_format(SimpleTypedDict)

        assert result["type"] == "object"
        assert result["title"] == "SimpleTypedDict"
        assert "name" in result["properties"]
        assert "age" in result["properties"]

    def test_invalid_type_raises_validation_error(self):
        """Invalid type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            normalize_response_format("invalid")

        assert exc_info.value.code == "INVALID_ARGUMENT"
        assert "response_format" in str(exc_info.value)


# =============================================================================
# Schema Depth Validation Tests
# =============================================================================


class TestSchemaDepthValidation:
    """Tests for schema depth validation."""

    def test_deep_nesting_raises_error(self):
        """Schema exceeding depth limit raises StructuredOutputError."""
        # Create a deeply nested dataclass (exceeds MAX_SCHEMA_DEPTH of 32)
        # We build this dynamically because we can't easily write 33+ nested classes

        # Build a schema with nesting depth > MAX_SCHEMA_DEPTH via $refs
        deeply_nested = {"type": "object", "properties": {}}
        current = deeply_nested
        for i in range(MAX_SCHEMA_DEPTH + 5):
            nested = {"type": "object", "properties": {}}
            current["properties"][f"level{i}"] = nested
            current = nested

        with pytest.raises(StructuredOutputError, match="nesting exceeds"):
            normalize_response_format(deeply_nested)

    def test_missing_ref_raises_error(self):
        """Schema with missing $ref raises StructuredOutputError."""
        schema_with_missing_ref = {
            "type": "object",
            "properties": {"user": {"$ref": "#/$defs/NonExistent"}},
        }

        with pytest.raises(StructuredOutputError, match="not found"):
            normalize_response_format(schema_with_missing_ref)

    def test_valid_ref_resolution(self):
        """Valid $ref is properly resolved."""
        schema_with_ref = {
            "type": "object",
            "properties": {"user": {"$ref": "#/$defs/User"}},
            "$defs": {"User": {"type": "object", "properties": {"name": {"type": "string"}}}},
        }

        result = normalize_response_format(schema_with_ref)
        assert result == schema_with_ref

    def test_circular_ref_handled(self):
        """Circular $refs don't cause infinite recursion."""
        schema_with_circular_ref = {
            "type": "object",
            "properties": {"self": {"$ref": "#/$defs/Node"}},
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "next": {"$ref": "#/$defs/Node"},
                    },
                }
            },
        }

        # Should not hang or crash - circular refs are detected and handled
        result = normalize_response_format(schema_with_circular_ref)
        assert result == schema_with_circular_ref


# =============================================================================
# Type Conversion Edge Cases
# =============================================================================


class TestTypeConversionEdgeCases:
    """Tests for edge cases in type conversion."""

    def test_optional_type(self):
        """Optional[T] is converted (ignoring None union)."""

        @dataclass
        class WithOptional:
            value: str | None

        schema = dataclass_to_schema(WithOptional)
        # Optional[str] should be converted to string type
        assert schema["properties"]["value"]["type"] == "string"

    def test_any_type_defaults_to_string(self):
        """Unknown types default to string."""
        from typing import Any

        @dataclass
        class WithAny:
            data: Any

        schema = dataclass_to_schema(WithAny)
        # Any type should default to string
        assert schema["properties"]["data"]["type"] == "string"

    def test_list_without_type_arg(self):
        """List without type argument uses Any items."""

        @dataclass
        class WithBareList:
            items: list  # No type argument

        schema = dataclass_to_schema(WithBareList)
        # List without args should have string items (default for Any)
        assert schema["properties"]["items"]["type"] == "array"
        assert schema["properties"]["items"]["items"]["type"] == "string"


# =============================================================================
# Ambiguous Union Warning Tests
# =============================================================================


class TestAmbiguousUnionWarning:
    """Tests for ambiguous union detection and warnings."""

    def test_ambiguous_union_emits_warning(self):
        """Union without discriminator emits AmbiguousUnionWarning."""
        # Schema with anyOf but no discriminator
        schema = {
            "anyOf": [
                {"type": "object", "title": "Refund", "properties": {"amount": {"type": "number"}}},
                {
                    "type": "object",
                    "title": "Escalate",
                    "properties": {"reason": {"type": "string"}},
                },
            ]
        }

        with pytest.warns(AmbiguousUnionWarning, match="Union.*Refund.*Escalate"):
            normalize_response_format(schema)

    def test_oneOf_union_emits_warning(self):
        """oneOf union without discriminator emits warning."""
        schema = {
            "oneOf": [
                {"type": "object", "title": "OptionA", "properties": {"x": {"type": "integer"}}},
                {"type": "object", "title": "OptionB", "properties": {"y": {"type": "integer"}}},
            ]
        }

        with pytest.warns(AmbiguousUnionWarning, match="Union.*OptionA.*OptionB"):
            normalize_response_format(schema)

    def test_union_with_discriminator_no_warning(self):
        """Union with discriminator does not emit warning."""
        schema = {
            "anyOf": [
                {"type": "object", "title": "Refund", "properties": {"amount": {"type": "number"}}},
                {
                    "type": "object",
                    "title": "Escalate",
                    "properties": {"reason": {"type": "string"}},
                },
            ],
            "discriminator": {"propertyName": "type"},
        }

        # Should NOT emit warning
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", AmbiguousUnionWarning)
            normalize_response_format(schema)

    def test_union_with_literal_fields_no_warning(self):
        """Union with Literal discriminator fields does not emit warning."""
        # When each member has a const field (Literal), it's not ambiguous
        schema = {
            "anyOf": [
                {
                    "type": "object",
                    "title": "Refund",
                    "properties": {
                        "kind": {"const": "refund"},
                        "amount": {"type": "number"},
                    },
                },
                {
                    "type": "object",
                    "title": "Escalate",
                    "properties": {
                        "kind": {"const": "escalate"},
                        "reason": {"type": "string"},
                    },
                },
            ]
        }

        # Should NOT emit warning because const fields distinguish the types
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", AmbiguousUnionWarning)
            normalize_response_format(schema)

    def test_union_with_enum_single_value_no_warning(self):
        """Union with single-value enum fields does not emit warning."""
        # Single-value enum is another representation of Literal
        schema = {
            "anyOf": [
                {
                    "type": "object",
                    "title": "Cat",
                    "properties": {
                        "pet_type": {"enum": ["cat"]},
                        "meows": {"type": "integer"},
                    },
                },
                {
                    "type": "object",
                    "title": "Dog",
                    "properties": {
                        "pet_type": {"enum": ["dog"]},
                        "barks": {"type": "integer"},
                    },
                },
            ]
        }

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", AmbiguousUnionWarning)
            normalize_response_format(schema)

    def test_non_union_schema_no_warning(self):
        """Simple object schema does not emit warning."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", AmbiguousUnionWarning)
            normalize_response_format(schema)

    def test_single_option_anyOf_no_warning(self):
        """Single-option anyOf does not emit warning."""
        schema = {
            "anyOf": [
                {"type": "object", "title": "OnlyOption", "properties": {"x": {"type": "string"}}}
            ]
        }

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", AmbiguousUnionWarning)
            normalize_response_format(schema)

    def test_warning_message_includes_example(self):
        """Warning message includes the example fix with class name."""
        schema = {
            "anyOf": [
                {"type": "object", "title": "Refund", "properties": {"amount": {"type": "number"}}},
                {
                    "type": "object",
                    "title": "Escalate",
                    "properties": {"reason": {"type": "string"}},
                },
            ]
        }

        with pytest.warns(AmbiguousUnionWarning) as record:
            normalize_response_format(schema)

        assert len(record) == 1
        message = str(record[0].message)
        # Check warning includes example fix
        assert "class Refund(BaseModel):" in message
        assert 'kind: Literal["refund"]' in message

    def test_pydantic_union_without_discriminator_warns(self):
        """Pydantic model with Union but no discriminator emits warning."""
        pytest.importorskip("pydantic")
        from pydantic import BaseModel

        class Refund(BaseModel):
            transaction_id: str
            amount: float

        class Escalate(BaseModel):
            reason: str
            urgency: int

        class Response(BaseModel):
            result: Refund | Escalate

        with pytest.warns(AmbiguousUnionWarning, match="Union.*Refund.*Escalate"):
            normalize_response_format(Response)

    def test_pydantic_union_with_literal_discriminator_no_warning(self):
        """Pydantic model with Literal discriminator does not warn."""
        pytest.importorskip("pydantic")
        from typing import Literal

        from pydantic import BaseModel

        class Refund(BaseModel):
            kind: Literal["refund"] = "refund"
            transaction_id: str
            amount: float

        class Escalate(BaseModel):
            kind: Literal["escalate"] = "escalate"
            reason: str
            urgency: int

        class Response(BaseModel):
            result: Refund | Escalate

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", AmbiguousUnionWarning)
            normalize_response_format(Response)
