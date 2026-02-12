"""
Tests for talu.chat.schema.hydrate - JSON dict to dataclass/TypedDict/Pydantic model conversion.

This module tests reconstruction of Python objects from JSON dicts.
"""

from dataclasses import dataclass
from typing import TypedDict

import pytest

from talu.chat.response.hydrate import dict_to_dataclass

# =============================================================================
# Test Fixtures - Dataclasses
# =============================================================================


@dataclass
class SimpleModel:
    """Simple dataclass for testing."""

    name: str
    age: int


@dataclass
class NestedModel:
    """Dataclass with nested dataclass."""

    user: SimpleModel
    active: bool


@dataclass
class ListModel:
    """Dataclass with list of dataclasses."""

    users: list[SimpleModel]


@dataclass
class MixedModel:
    """Dataclass with mixed types."""

    name: str
    tags: list[str]
    score: float


# =============================================================================
# dict_to_dataclass() Tests - Simple Cases
# =============================================================================


class TestDictToDataclassSimple:
    """Tests for simple dataclass reconstruction."""

    def test_simple_dataclass(self):
        """Reconstructs simple dataclass from dict."""
        data = {"name": "Alice", "age": 30}
        result = dict_to_dataclass(SimpleModel, data)

        assert isinstance(result, SimpleModel)
        assert result.name == "Alice"
        assert result.age == 30

    def test_preserves_types(self):
        """Preserves correct Python types."""
        data = {"name": "Bob", "age": 25}
        result = dict_to_dataclass(SimpleModel, data)

        assert isinstance(result.name, str)
        assert isinstance(result.age, int)

    def test_non_dict_passthrough(self):
        """Non-dict values pass through unchanged."""
        result = dict_to_dataclass(SimpleModel, "already a string")
        assert result == "already a string"

        result = dict_to_dataclass(SimpleModel, 42)
        assert result == 42

    def test_non_dataclass_returns_data(self):
        """Non-dataclass target returns data unchanged."""
        data = {"x": 1, "y": 2}
        result = dict_to_dataclass(str, data)

        assert result == data


# =============================================================================
# dict_to_dataclass() Tests - Nested Dataclasses
# =============================================================================


class TestDictToDataclassNested:
    """Tests for nested dataclass reconstruction."""

    def test_nested_dataclass(self):
        """Reconstructs nested dataclass from nested dict."""
        data = {"user": {"name": "Alice", "age": 30}, "active": True}
        result = dict_to_dataclass(NestedModel, data)

        assert isinstance(result, NestedModel)
        assert isinstance(result.user, SimpleModel)
        assert result.user.name == "Alice"
        assert result.user.age == 30
        assert result.active is True

    def test_deeply_nested(self):
        """Handles multiple levels of nesting."""

        @dataclass
        class DeepModel:
            nested: NestedModel

        data = {"nested": {"user": {"name": "Bob", "age": 25}, "active": False}}
        result = dict_to_dataclass(DeepModel, data)

        assert isinstance(result, DeepModel)
        assert isinstance(result.nested, NestedModel)
        assert isinstance(result.nested.user, SimpleModel)
        assert result.nested.user.name == "Bob"


# =============================================================================
# dict_to_dataclass() Tests - List of Dataclasses
# =============================================================================


class TestDictToDataclassList:
    """Tests for list of dataclass reconstruction."""

    def test_list_of_dataclasses(self):
        """Reconstructs list of dataclasses."""
        data = {
            "users": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ]
        }
        result = dict_to_dataclass(ListModel, data)

        assert isinstance(result, ListModel)
        assert len(result.users) == 2
        assert all(isinstance(u, SimpleModel) for u in result.users)
        assert result.users[0].name == "Alice"
        assert result.users[1].name == "Bob"

    def test_empty_list(self):
        """Handles empty list."""
        data = {"users": []}
        result = dict_to_dataclass(ListModel, data)

        assert isinstance(result, ListModel)
        assert result.users == []

    def test_list_of_primitives(self):
        """List of primitives passed through unchanged."""
        data = {"name": "Test", "tags": ["a", "b", "c"], "score": 0.5}
        result = dict_to_dataclass(MixedModel, data)

        assert isinstance(result, MixedModel)
        assert result.tags == ["a", "b", "c"]


# =============================================================================
# dict_to_dataclass() Tests - Edge Cases
# =============================================================================


class TestDictToDataclassEdgeCases:
    """Tests for edge cases in dataclass reconstruction."""

    def test_extra_keys_ignored(self):
        """Extra keys in data are ignored."""
        data = {"name": "Alice", "age": 30, "extra_field": "ignored"}
        result = dict_to_dataclass(SimpleModel, data)

        assert isinstance(result, SimpleModel)
        assert result.name == "Alice"
        assert not hasattr(result, "extra_field")

    def test_missing_required_field_raises(self):
        """Missing required field raises TypeError."""
        data = {"name": "Alice"}  # Missing 'age'

        with pytest.raises(TypeError):
            dict_to_dataclass(SimpleModel, data)

    def test_none_values(self):
        """None values are passed through."""
        from dataclasses import dataclass

        @dataclass
        class OptionalModel:
            name: str | None

        data = {"name": None}
        result = dict_to_dataclass(OptionalModel, data)

        assert result.name is None

    def test_nested_none(self):
        """None for nested dataclass is passed through."""
        from dataclasses import dataclass

        @dataclass
        class OptionalNestedModel:
            user: SimpleModel | None

        data = {"user": None}
        result = dict_to_dataclass(OptionalNestedModel, data)

        # None is passed through (not converted to SimpleModel)
        assert result.user is None


# =============================================================================
# dict_to_dataclass() Tests - TypedDict Support
# =============================================================================


class SimpleTypedDict(TypedDict):
    """Simple TypedDict for testing."""

    name: str
    age: int


class TestDictToDataclassTypedDict:
    """Tests for TypedDict handling."""

    def test_typeddict_returns_dict(self):
        """TypedDict returns the dict as-is (no conversion)."""
        data = {"name": "Alice", "age": 30}
        result = dict_to_dataclass(SimpleTypedDict, data)

        # TypedDict is just a typed dict - returns the same dict
        assert result is data
        assert result == {"name": "Alice", "age": 30}

    def test_typeddict_preserves_dict_type(self):
        """TypedDict result is still a dict."""
        data = {"name": "Bob", "age": 25}
        result = dict_to_dataclass(SimpleTypedDict, data)

        assert isinstance(result, dict)
        assert result["name"] == "Bob"
        assert result["age"] == 25
