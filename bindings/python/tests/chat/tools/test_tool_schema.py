"""Tests for tool schema extraction and type mapping."""

from __future__ import annotations

from typing import Literal

from talu.chat.tools.schema import extract_schema, python_type_to_json


def test_python_type_to_json_literal_enum() -> None:
    """Literal is mapped to enum with correct type."""
    schema = python_type_to_json(Literal["a", "b"])
    assert schema["type"] == "string"
    assert schema["extra"]["enum"] == ["a", "b"]


def test_python_type_to_json_optional() -> None:
    """Optional types are marked optional and mapped to inner type."""
    schema = python_type_to_json(int | None)
    assert schema["type"] == "integer"
    assert schema["optional"] is True


def test_python_type_to_json_list_item_type() -> None:
    """List[T] maps to array with item type."""
    schema = python_type_to_json(list[str])
    assert schema["type"] == "array"
    assert schema["extra"]["items"]["type"] == "string"


def test_extract_schema_optional_not_required() -> None:
    """Optional parameters are not required even without defaults."""

    def foo(name: str | None) -> str:
        """Test.

        Args:
            name: Optional name
        """

        return name or "none"

    schema = extract_schema(foo)
    assert "name" not in schema["parameters"]["required"]


def test_extract_schema_unknown_type_fallback() -> None:
    """Unknown types fall back to string without crashing."""

    class Weird:  # noqa: N801 - test type
        pass

    def bar(value: Weird) -> str:
        """Test.

        Args:
            value: Weird value
        """

        return "ok"

    schema = extract_schema(bar)
    assert schema["parameters"]["properties"]["value"]["type"] == "string"
