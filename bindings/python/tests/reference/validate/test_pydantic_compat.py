"""Test Validator compatibility with Pydantic models.

This module tests that talu.Validator produces the same validation
results as Pydantic for equivalent schemas.
"""

import json

import pytest
from pydantic import BaseModel, ConfigDict, Field

from talu import Validator

# =============================================================================
# Test Models
# =============================================================================


class SimpleUser(BaseModel):
    """Simple user model.

    Uses extra='forbid' for strict streaming validation - grammar enforces
    types at parse time for immediate error detection.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    age: int


class UserWithOptional(BaseModel):
    """User with optional field."""

    name: str
    email: str | None = None


class UserWithConstraints(BaseModel):
    """User with field constraints."""

    name: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=0, le=150)


class NestedAddress(BaseModel):
    """Address model for nesting."""

    street: str
    city: str
    zip_code: str


class UserWithAddress(BaseModel):
    """User with nested address."""

    name: str
    address: NestedAddress


class UserWithTags(BaseModel):
    """User with list field."""

    name: str
    tags: list[str] = []


# =============================================================================
# Basic Pydantic Model Tests
# =============================================================================


class TestPydanticBasic:
    """Test basic Pydantic model validation."""

    def test_simple_model_valid(self):
        """Valid data passes both Pydantic and Validator."""
        data = '{"name": "Alice", "age": 30}'
        data_dict = json.loads(data)

        # Pydantic validates
        user = SimpleUser.model_validate(data_dict)
        assert user.name == "Alice"

        # Validator validates
        validator = Validator(SimpleUser)
        assert validator.validate(data) is True

    def test_simple_model_invalid_type(self):
        """Invalid type fails both Pydantic and Validator."""
        data = '{"name": "Alice", "age": "thirty"}'
        data_dict = json.loads(data)

        # Pydantic rejects
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SimpleUser.model_validate(data_dict)

        # Validator rejects
        validator = Validator(SimpleUser)
        assert validator.validate(data) is False

    def test_simple_model_missing_required(self):
        """Missing required field fails both."""
        data = '{"name": "Alice"}'
        data_dict = json.loads(data)

        # Pydantic rejects
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SimpleUser.model_validate(data_dict)

        # Validator rejects
        validator = Validator(SimpleUser)
        assert validator.validate(data) is False


# =============================================================================
# Optional Fields
# =============================================================================


class TestPydanticOptional:
    """Test optional field handling."""

    def test_optional_present(self):
        """Optional field present passes both."""
        data = '{"name": "Bob", "email": "bob@example.com"}'
        data_dict = json.loads(data)

        user = UserWithOptional.model_validate(data_dict)
        assert user.email == "bob@example.com"

        validator = Validator(UserWithOptional)
        assert validator.validate(data) is True

    def test_optional_absent(self):
        """Optional field absent passes both."""
        data = '{"name": "Bob"}'
        data_dict = json.loads(data)

        user = UserWithOptional.model_validate(data_dict)
        assert user.email is None

        validator = Validator(UserWithOptional)
        assert validator.validate(data) is True

    def test_optional_null(self):
        """Optional field as null passes both."""
        data = '{"name": "Bob", "email": null}'
        data_dict = json.loads(data)

        user = UserWithOptional.model_validate(data_dict)
        assert user.email is None

        validator = Validator(UserWithOptional)
        assert validator.validate(data) is True


# =============================================================================
# Field Constraints
# =============================================================================


class TestPydanticConstraints:
    """Test field constraint validation."""

    def test_constraints_valid(self):
        """Valid constrained values pass both."""
        data = '{"name": "Charlie", "age": 25}'
        data_dict = json.loads(data)

        user = UserWithConstraints.model_validate(data_dict)
        assert user.age == 25

        validator = Validator(UserWithConstraints)
        assert validator.validate(data) is True

    def test_age_too_low(self):
        """Age below minimum fails both."""
        data = '{"name": "Charlie", "age": -5}'
        data_dict = json.loads(data)

        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            UserWithConstraints.model_validate(data_dict)

        validator = Validator(UserWithConstraints)
        assert validator.validate(data) is False

    def test_age_too_high(self):
        """Age above maximum fails both."""
        data = '{"name": "Charlie", "age": 200}'
        data_dict = json.loads(data)

        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            UserWithConstraints.model_validate(data_dict)

        validator = Validator(UserWithConstraints)
        assert validator.validate(data) is False


# =============================================================================
# Nested Models
# =============================================================================


class TestPydanticNested:
    """Test nested model validation."""

    def test_nested_valid(self):
        """Valid nested object passes both."""
        data = """{
            "name": "Dana",
            "address": {
                "street": "123 Main St",
                "city": "Boston",
                "zip_code": "02101"
            }
        }"""
        data_dict = json.loads(data)

        user = UserWithAddress.model_validate(data_dict)
        assert user.address.city == "Boston"

        validator = Validator(UserWithAddress)
        assert validator.validate(data) is True

    def test_nested_missing_field(self):
        """Missing nested field fails both."""
        data = """{
            "name": "Dana",
            "address": {
                "street": "123 Main St",
                "city": "Boston"
            }
        }"""
        data_dict = json.loads(data)

        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            UserWithAddress.model_validate(data_dict)

        validator = Validator(UserWithAddress)
        assert validator.validate(data) is False

    def test_nested_wrong_type(self):
        """Wrong type in nested object fails both."""
        data = '{"name": "Dana", "address": "not an object"}'
        data_dict = json.loads(data)

        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            UserWithAddress.model_validate(data_dict)

        validator = Validator(UserWithAddress)
        assert validator.validate(data) is False


# =============================================================================
# List Fields
# =============================================================================


class TestPydanticLists:
    """Test list field validation."""

    def test_list_valid(self):
        """Valid list passes both."""
        data = '{"name": "Eve", "tags": ["python", "rust"]}'
        data_dict = json.loads(data)

        user = UserWithTags.model_validate(data_dict)
        assert user.tags == ["python", "rust"]

        validator = Validator(UserWithTags)
        assert validator.validate(data) is True

    def test_list_empty(self):
        """Empty list passes both."""
        data = '{"name": "Eve", "tags": []}'
        data_dict = json.loads(data)

        user = UserWithTags.model_validate(data_dict)
        assert user.tags == []

        validator = Validator(UserWithTags)
        assert validator.validate(data) is True

    def test_list_wrong_item_type(self):
        """Wrong item type in list fails both."""
        data = '{"name": "Eve", "tags": [1, 2, 3]}'
        data_dict = json.loads(data)

        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            UserWithTags.model_validate(data_dict)

        validator = Validator(UserWithTags)
        assert validator.validate(data) is False


# =============================================================================
# Streaming with Pydantic Schema
# =============================================================================


class TestPydanticStreaming:
    """Test streaming validation with Pydantic schemas."""

    def test_streaming_pydantic_model(self):
        """Stream validation with Pydantic model works."""
        validator = Validator(SimpleUser)

        chunks = ['{"name": "', "Frank", '", "age": ', "40}"]

        for chunk in chunks:
            assert validator.feed(chunk) is True

        assert validator.is_complete is True

    def test_streaming_early_abort(self):
        """Stream validation detects type violation early."""
        validator = Validator(SimpleUser)

        # Valid start
        assert validator.feed('{"name": "Grace", "age": ') is True

        # Type violation: string instead of int
        result = validator.feed('"not a number"')
        assert result is False
