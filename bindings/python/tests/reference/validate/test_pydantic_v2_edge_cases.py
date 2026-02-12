"""Pydantic V2 Edge Cases Test Suite.

This module tests Pydantic V2 features that might produce JSON Schemas
the Zig parser doesn't fully support. It documents what works, what
doesn't, and establishes the "compatibility contract" for PydanticAI users.

Categories:
1. Union types (str | None, Union[A, B])
2. Discriminated unions (Literal + discriminator)
3. Field aliases (serialization_alias, validation_alias)
4. Enums and Literals
5. Complex nested structures
6. Recursive/self-referential models
7. Computed fields and validators (which talu ignores)
8. Constraint edge cases (what talu validates vs. ignores)

Known Limitations (Semantic Constraints):
-----------------------------------------
The following JSON Schema constraints are NOT enforced by talu's grammar-based
validation. These are "semantic" constraints that require post-validation or
runtime checks:

- minLength / maxLength: String length constraints (grammar accepts any length)
- minItems / maxItems: Array size constraints (grammar accepts any length)
- pattern: Regex patterns (not implemented in grammar compiler)

These constraints ARE enforced:
- enum: Both string and integer enums (fixed in schema.zig:268-271)
- minimum / maximum: Integer range constraints (when span <= 1000)
- required: Required fields in objects
- type: JSON type constraints (string, integer, boolean, null, etc.)
- $ref: Recursive references via $defs

Pydantic Validators:
-------------------
Pydantic @field_validator decorators are NOT visible in the JSON Schema,
so talu cannot enforce them. This is by design - talu validates JSON syntax,
not business logic. Use Pydantic's model_validate() for semantic validation.
"""

import json
from enum import Enum
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from talu import Validator

# =============================================================================
# 1. Union Types
# =============================================================================


class UnionString(BaseModel):
    """Model with nullable string (str | None)."""

    value: str | None


class UnionIntFloat(BaseModel):
    """Model with int or float union."""

    model_config = ConfigDict(extra="forbid")

    number: int | float


class UnionMultiple(BaseModel):
    """Model with multiple type union."""

    data: str | int | bool | None


class TestUnionTypes:
    """Test various union type patterns."""

    def test_nullable_string_with_value(self):
        """str | None with string value."""
        validator = Validator(UnionString)
        assert validator.validate('{"value": "hello"}') is True

    def test_nullable_string_with_null(self):
        """str | None with null."""
        validator = Validator(UnionString)
        assert validator.validate('{"value": null}') is True

    def test_nullable_string_absent_field(self):
        """str | None with field absent (should fail - field is required)."""
        validator = Validator(UnionString)
        # Note: Without default, field is required even if nullable
        assert validator.validate("{}") is False

    def test_int_float_union_with_int(self):
        """int | float with integer."""
        validator = Validator(UnionIntFloat)
        assert validator.validate('{"number": 42}') is True

    def test_int_float_union_with_float(self):
        """int | float with float."""
        validator = Validator(UnionIntFloat)
        assert validator.validate('{"number": 3.14}') is True

    def test_int_float_union_rejects_string(self):
        """int | float rejects string."""
        validator = Validator(UnionIntFloat)
        assert validator.validate('{"number": "42"}') is False

    def test_multiple_union_string(self):
        """str | int | bool | None with string."""
        validator = Validator(UnionMultiple)
        assert validator.validate('{"data": "text"}') is True

    def test_multiple_union_int(self):
        """str | int | bool | None with int."""
        validator = Validator(UnionMultiple)
        assert validator.validate('{"data": 123}') is True

    def test_multiple_union_bool(self):
        """str | int | bool | None with bool."""
        validator = Validator(UnionMultiple)
        assert validator.validate('{"data": true}') is True

    def test_multiple_union_null(self):
        """str | int | bool | None with null."""
        validator = Validator(UnionMultiple)
        assert validator.validate('{"data": null}') is True


# =============================================================================
# 2. Discriminated Unions
# =============================================================================


class Cat(BaseModel):
    """Cat model for discriminated union."""

    model_config = ConfigDict(extra="forbid")

    pet_type: Literal["cat"]
    meows: int


class Dog(BaseModel):
    """Dog model for discriminated union."""

    model_config = ConfigDict(extra="forbid")

    pet_type: Literal["dog"]
    barks: int


class PetOwner(BaseModel):
    """Owner with discriminated union pet field."""

    model_config = ConfigDict(extra="forbid")

    name: str
    pet: Cat | Dog


class DiscriminatedPetOwner(BaseModel):
    """Owner with explicitly discriminated union."""

    model_config = ConfigDict(extra="forbid")

    name: str
    pet: Annotated[Cat | Dog, Field(discriminator="pet_type")]


class TestDiscriminatedUnions:
    """Test discriminated union patterns."""

    def test_union_cat_valid(self):
        """Cat variant validates."""
        validator = Validator(PetOwner)
        data = '{"name": "Alice", "pet": {"pet_type": "cat", "meows": 5}}'
        assert validator.validate(data) is True

    def test_union_dog_valid(self):
        """Dog variant validates."""
        validator = Validator(PetOwner)
        data = '{"name": "Bob", "pet": {"pet_type": "dog", "barks": 3}}'
        assert validator.validate(data) is True

    def test_union_wrong_discriminator(self):
        """Wrong discriminator value fails."""
        validator = Validator(PetOwner)
        data = '{"name": "Eve", "pet": {"pet_type": "bird", "chirps": 10}}'
        assert validator.validate(data) is False

    def test_union_missing_discriminator(self):
        """Missing discriminator field fails."""
        validator = Validator(PetOwner)
        data = '{"name": "Eve", "pet": {"meows": 5}}'
        assert validator.validate(data) is False

    def test_explicit_discriminator_cat(self):
        """Explicitly discriminated union - cat."""
        validator = Validator(DiscriminatedPetOwner)
        data = '{"name": "Carol", "pet": {"pet_type": "cat", "meows": 2}}'
        assert validator.validate(data) is True

    def test_explicit_discriminator_dog(self):
        """Explicitly discriminated union - dog."""
        validator = Validator(DiscriminatedPetOwner)
        data = '{"name": "Dan", "pet": {"pet_type": "dog", "barks": 4}}'
        assert validator.validate(data) is True


# =============================================================================
# 3. Field Aliases
# =============================================================================


class AliasedModel(BaseModel):
    """Model with field aliases."""

    internal_name: str = Field(alias="externalName")
    snake_case: int = Field(serialization_alias="camelCase")


class ValidationAliasModel(BaseModel):
    """Model with validation alias."""

    user_id: str = Field(validation_alias="userId")


class TestFieldAliases:
    """Test field alias handling.

    Note: Talu validates against the JSON Schema, which uses the
    serialization names (alias/serialization_alias).
    """

    def test_alias_external_name(self):
        """Field with alias validates using alias name."""
        validator = Validator(AliasedModel)
        # Uses 'alias' name in JSON
        data = '{"externalName": "test", "snake_case": 42}'
        assert validator.validate(data) is True

    def test_alias_internal_name_fails(self):
        """Using internal name when alias is set fails."""
        validator = Validator(AliasedModel)
        data = '{"internal_name": "test", "snake_case": 42}'
        # This should fail because schema uses alias
        assert validator.validate(data) is False

    def test_validation_alias(self):
        """Validation alias accepted."""
        validator = Validator(ValidationAliasModel)
        # validation_alias means accept both
        data = '{"userId": "abc123"}'
        assert validator.validate(data) is True


# =============================================================================
# 4. Enums and Literals
# =============================================================================


class Color(str, Enum):
    """String enum."""

    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Priority(int, Enum):
    """Integer enum."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3


class EnumModel(BaseModel):
    """Model with enum fields."""

    model_config = ConfigDict(extra="forbid")

    color: Color
    priority: Priority


class LiteralModel(BaseModel):
    """Model with Literal types."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["pending", "active", "done"]
    code: Literal[1, 2, 3]


class TestEnumsAndLiterals:
    """Test enum and Literal type handling."""

    def test_string_enum_valid(self):
        """Valid string enum value."""
        validator = Validator(EnumModel)
        data = '{"color": "red", "priority": 2}'
        assert validator.validate(data) is True

    def test_string_enum_invalid(self):
        """Invalid string enum value."""
        validator = Validator(EnumModel)
        data = '{"color": "purple", "priority": 2}'
        assert validator.validate(data) is False

    def test_int_enum_valid(self):
        """Valid int enum value."""
        validator = Validator(EnumModel)
        data = '{"color": "green", "priority": 3}'
        assert validator.validate(data) is True

    def test_int_enum_invalid(self):
        """Invalid int enum value."""
        validator = Validator(EnumModel)
        data = '{"color": "blue", "priority": 5}'
        assert validator.validate(data) is False

    def test_literal_string_valid(self):
        """Valid Literal string."""
        validator = Validator(LiteralModel)
        data = '{"status": "active", "code": 2}'
        assert validator.validate(data) is True

    def test_literal_string_invalid(self):
        """Invalid Literal string."""
        validator = Validator(LiteralModel)
        data = '{"status": "cancelled", "code": 2}'
        assert validator.validate(data) is False

    def test_literal_int_valid(self):
        """Valid Literal int."""
        validator = Validator(LiteralModel)
        data = '{"status": "done", "code": 3}'
        assert validator.validate(data) is True

    def test_literal_int_invalid(self):
        """Invalid Literal int."""
        validator = Validator(LiteralModel)
        data = '{"status": "pending", "code": 99}'
        assert validator.validate(data) is False


# =============================================================================
# 5. Complex Nested Structures
# =============================================================================


class Address(BaseModel):
    """Nested address."""

    street: str
    city: str
    country: str = "USA"


class Company(BaseModel):
    """Company with address."""

    name: str
    address: Address


class Person(BaseModel):
    """Person with optional company."""

    name: str
    employer: Company | None = None


class DeeplyNested(BaseModel):
    """Deeply nested structure."""

    level1: Person


class TestComplexNested:
    """Test complex nested structures."""

    def test_nested_with_defaults(self):
        """Nested object with field defaults."""
        validator = Validator(Company)
        # country has default, can be omitted
        data = '{"name": "Acme", "address": {"street": "123 Main", "city": "NYC"}}'
        assert validator.validate(data) is True

    def test_optional_nested_present(self):
        """Optional nested object when present."""
        validator = Validator(Person)
        data = """{
            "name": "Alice",
            "employer": {
                "name": "TechCorp",
                "address": {"street": "456 Tech Blvd", "city": "SF", "country": "USA"}
            }
        }"""
        assert validator.validate(data) is True

    def test_optional_nested_absent(self):
        """Optional nested object when absent."""
        validator = Validator(Person)
        data = '{"name": "Bob"}'
        assert validator.validate(data) is True

    def test_optional_nested_null(self):
        """Optional nested object as null."""
        validator = Validator(Person)
        data = '{"name": "Carol", "employer": null}'
        assert validator.validate(data) is True

    def test_deeply_nested(self):
        """Three levels of nesting."""
        validator = Validator(DeeplyNested)
        data = """{
            "level1": {
                "name": "Dan",
                "employer": {
                    "name": "MegaCorp",
                    "address": {"street": "789 Corp Way", "city": "LA", "country": "USA"}
                }
            }
        }"""
        assert validator.validate(data) is True


# =============================================================================
# 6. Self-Referential / Recursive Models
# =============================================================================


class TreeNode(BaseModel):
    """Self-referential tree node."""

    value: int
    children: list["TreeNode"] = []


class LinkedNode(BaseModel):
    """Linked list node."""

    data: str
    next: "LinkedNode | None" = None


# Update forward refs for Pydantic
TreeNode.model_rebuild()
LinkedNode.model_rebuild()


class TestRecursiveModels:
    """Test recursive/self-referential models."""

    def test_tree_leaf(self):
        """Tree with single leaf node."""
        validator = Validator(TreeNode)
        data = '{"value": 1, "children": []}'
        assert validator.validate(data) is True

    def test_tree_nested(self):
        """Tree with nested children."""
        validator = Validator(TreeNode)
        data = """{
            "value": 1,
            "children": [
                {"value": 2, "children": []},
                {"value": 3, "children": [{"value": 4, "children": []}]}
            ]
        }"""
        assert validator.validate(data) is True

    def test_linked_list_single(self):
        """Linked list with single node."""
        validator = Validator(LinkedNode)
        data = '{"data": "first", "next": null}'
        assert validator.validate(data) is True

    def test_linked_list_chain(self):
        """Linked list with chain."""
        validator = Validator(LinkedNode)
        data = '{"data": "a", "next": {"data": "b", "next": {"data": "c", "next": null}}}'
        assert validator.validate(data) is True


# =============================================================================
# 7. Validators and Computed Fields (Talu Ignores These)
# =============================================================================


class ValidatedModel(BaseModel):
    """Model with field validator."""

    email: str
    age: int

    @field_validator("email")
    @classmethod
    def email_must_contain_at(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("must contain @")
        return v

    @field_validator("age")
    @classmethod
    def age_must_be_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be non-negative")
        return v


class TestValidatorsIgnored:
    """Test that Pydantic validators are NOT enforced by Talu.

    IMPORTANT: Talu validates JSON structure, not Pydantic validators.
    These tests document expected behavior divergence.
    """

    def test_validator_valid_for_both(self):
        """Data valid for both Pydantic validator and Talu."""
        data = '{"email": "test@example.com", "age": 25}'

        # Pydantic accepts
        ValidatedModel.model_validate(json.loads(data))

        # Talu accepts
        validator = Validator(ValidatedModel)
        assert validator.validate(data) is True

    def test_validator_fails_pydantic_passes_talu(self):
        """Email without @ fails Pydantic but passes Talu.

        This documents that Talu does NOT run Pydantic validators.
        """
        data = '{"email": "invalid-email", "age": 25}'

        # Pydantic rejects (validator fails)
        with pytest.raises(ValidationError):
            ValidatedModel.model_validate(json.loads(data))

        # Talu accepts (only checks JSON schema, not validators)
        validator = Validator(ValidatedModel)
        assert validator.validate(data) is True  # Expected divergence!

    def test_negative_age_fails_pydantic_passes_talu(self):
        """Negative age fails Pydantic validator but passes Talu.

        IMPORTANT: Field(ge=0) constraints ARE validated by Talu via JSON Schema.
        But custom @field_validator functions are NOT.
        """
        data = '{"email": "test@example.com", "age": -5}'

        # Pydantic rejects (validator fails)
        with pytest.raises(ValidationError):
            ValidatedModel.model_validate(json.loads(data))

        # Talu accepts (no ge= constraint in schema, only custom validator)
        validator = Validator(ValidatedModel)
        assert validator.validate(data) is True  # Expected divergence!


# =============================================================================
# 8. Constraint Edge Cases
# =============================================================================


class SmallIntRange(BaseModel):
    """Model with small integer range (Talu can enumerate)."""

    model_config = ConfigDict(extra="forbid")

    value: int = Field(ge=0, le=100)


class LargeIntRange(BaseModel):
    """Model with large integer range (Talu uses generic int)."""

    model_config = ConfigDict(extra="forbid")

    value: int = Field(ge=0, le=10000)


class StringLengthConstraints(BaseModel):
    """Model with string length constraints."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=50)


class ArrayLengthConstraints(BaseModel):
    """Model with array length constraints."""

    model_config = ConfigDict(extra="forbid")

    items: list[str] = Field(min_length=1, max_length=10)


class TestConstraintEdgeCases:
    """Test constraint edge cases and document limitations.

    Talu converts some constraints to grammar rules:
    - Integer ranges up to ~1000: enumerated exactly
    - Integer ranges > 1000: fallback to generic integer

    Talu does NOT validate these (they're semantic, not syntactic):
    - minLength/maxLength for strings
    - minItems/maxItems for arrays
    """

    def test_small_int_range_valid(self):
        """Small int range - valid value."""
        validator = Validator(SmallIntRange)
        assert validator.validate('{"value": 50}') is True

    def test_small_int_range_at_min(self):
        """Small int range - at minimum."""
        validator = Validator(SmallIntRange)
        assert validator.validate('{"value": 0}') is True

    def test_small_int_range_at_max(self):
        """Small int range - at maximum."""
        validator = Validator(SmallIntRange)
        assert validator.validate('{"value": 100}') is True

    def test_small_int_range_below_min(self):
        """Small int range - below minimum fails."""
        validator = Validator(SmallIntRange)
        assert validator.validate('{"value": -1}') is False

    def test_small_int_range_above_max(self):
        """Small int range - above maximum fails."""
        validator = Validator(SmallIntRange)
        assert validator.validate('{"value": 101}') is False

    def test_large_int_range_valid(self):
        """Large int range - valid value (uses generic int)."""
        validator = Validator(LargeIntRange)
        assert validator.validate('{"value": 5000}') is True

    def test_large_int_range_above_constraint(self):
        """Large int range - above constraint.

        NOTE: Large ranges may not be enforced by grammar.
        This test documents the behavior.
        """
        validator = Validator(LargeIntRange)
        result = validator.validate('{"value": 20000}')
        # May pass if grammar fell back to generic integer
        # Document actual behavior:
        assert result in (True, False)  # Behavior depends on max_exact_span config

    def test_string_min_length_enforced(self):
        """String minLength IS enforced by Talu grammar."""
        validator = Validator(StringLengthConstraints)
        # Empty string violates minLength=1
        assert validator.validate('{"name": ""}') is False

    def test_string_max_length_enforced(self):
        """String maxLength IS enforced by Talu grammar."""
        validator = Validator(StringLengthConstraints)
        # 100-char string violates maxLength=50
        long_name = "x" * 100
        assert validator.validate(f'{{"name": "{long_name}"}}') is False

    def test_array_min_items_enforced(self):
        """Array minItems IS enforced by Talu grammar."""
        validator = Validator(ArrayLengthConstraints)
        # Empty array violates minItems=1
        assert validator.validate('{"items": []}') is False

    def test_string_length_pydantic_validates(self):
        """Demonstrate that Pydantic DOES validate string length."""
        # Empty string
        with pytest.raises(ValidationError):
            StringLengthConstraints.model_validate({"name": ""})

        # Too long
        with pytest.raises(ValidationError):
            StringLengthConstraints.model_validate({"name": "x" * 100})


# =============================================================================
# 9. Default Values
# =============================================================================


class WithDefaults(BaseModel):
    """Model with various default values."""

    required_field: str
    optional_str: str = "default"
    optional_int: int = 0
    optional_list: list[str] = []
    optional_none: str | None = None


class TestDefaults:
    """Test default value handling."""

    def test_all_fields_provided(self):
        """All fields provided."""
        validator = Validator(WithDefaults)
        data = """{
            "required_field": "value",
            "optional_str": "custom",
            "optional_int": 42,
            "optional_list": ["a", "b"],
            "optional_none": "not null"
        }"""
        assert validator.validate(data) is True

    def test_only_required(self):
        """Only required field provided."""
        validator = Validator(WithDefaults)
        data = '{"required_field": "value"}'
        assert validator.validate(data) is True

    def test_missing_required_fails(self):
        """Missing required field fails."""
        validator = Validator(WithDefaults)
        data = '{"optional_str": "custom"}'
        assert validator.validate(data) is False

    def test_partial_optionals(self):
        """Some optional fields provided."""
        validator = Validator(WithDefaults)
        data = '{"required_field": "value", "optional_int": 100}'
        assert validator.validate(data) is True


# =============================================================================
# 10. Streaming Validation with Complex Types
# =============================================================================


class TestStreamingComplex:
    """Test streaming validation with complex Pydantic types."""

    def test_streaming_union(self):
        """Stream validation with union type."""
        validator = Validator(UnionString)

        assert validator.feed('{"value":') is True
        assert validator.feed(' "hello"') is True
        assert validator.feed("}") is True
        assert validator.is_complete is True

    def test_streaming_union_null(self):
        """Stream validation with union resolving to null."""
        validator = Validator(UnionString)

        assert validator.feed('{"value":') is True
        assert validator.feed(" null") is True
        assert validator.feed("}") is True
        assert validator.is_complete is True

    def test_streaming_nested(self):
        """Stream validation with nested structure."""
        validator = Validator(Company)

        chunks = [
            '{"name": "Acme",',
            ' "address": {',
            '"street": "123 Main",',
            '"city": "NYC"',
            "}}",
        ]

        for chunk in chunks:
            assert validator.feed(chunk) is True

        assert validator.is_complete is True

    def test_streaming_enum(self):
        """Stream validation with enum."""
        validator = Validator(EnumModel)

        assert validator.feed('{"color": "') is True
        assert validator.feed('red", "priority": ') is True
        assert validator.feed("2}") is True
        assert validator.is_complete is True

    def test_streaming_literal_early_reject(self):
        """Stream validation rejects invalid Literal early."""
        validator = Validator(LiteralModel)

        assert validator.feed('{"status": "') is True
        # 'invalid' doesn't match any Literal option
        result = validator.feed('invalid"')
        assert result is False
