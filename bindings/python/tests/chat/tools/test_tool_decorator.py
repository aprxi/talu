"""Tests for @tool decorator and docstring parsing."""

from __future__ import annotations

from talu.chat.tools import tool
from talu.chat.tools.schema import parse_docstring


def test_tool_decorator_sets_metadata() -> None:
    """@tool sets schema and metadata on the wrapped function."""

    @tool
    def greet(name: str, formal: bool = False) -> str:
        """Greet someone.

        Args:
            name: The person's name
            formal: Whether to use a formal greeting
        """
        return f"Hello, {name}!"

    assert greet._is_tool is True  # type: ignore[attr-defined]
    assert greet._tool_func is not None  # type: ignore[attr-defined]
    schema = greet._tool_schema  # type: ignore[attr-defined]
    assert schema["name"] == "greet"
    assert schema["description"] == "Greet someone."
    assert "name" in schema["parameters"]["required"]
    assert "formal" not in schema["parameters"]["required"]
    assert schema["parameters"]["properties"]["name"]["type"] == "string"
    assert schema["parameters"]["properties"]["formal"]["type"] == "boolean"


def test_parse_docstring_args_section() -> None:
    """parse_docstring extracts summary and args descriptions."""
    doc = """
    Do a thing.

    Args:
        alpha: First value.
        beta: Second value
            continued line.
    Returns:
        Nothing.
    """
    parsed = parse_docstring(doc)
    assert parsed["summary"] == "Do a thing."
    assert parsed["args"]["alpha"] == "First value."
    assert parsed["args"]["beta"] == "Second value continued line."
