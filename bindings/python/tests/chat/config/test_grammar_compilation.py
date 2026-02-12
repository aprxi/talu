"""Tests for pre-compiled Grammar handles.

The Grammar class allows pre-compiling JSON schemas into reusable grammar handles.
This enables zero-latency structured output when the same schema is used across
multiple generations.
"""

from dataclasses import dataclass

import pytest

from talu.exceptions import StructuredOutputError
from talu.router import Grammar

# =============================================================================
# Test dataclasses
# =============================================================================


@dataclass
class Person:
    """Simple dataclass for testing."""

    name: str
    age: int


@dataclass
class Answer:
    """Structured output for Q&A."""

    value: int
    reasoning: str


@dataclass
class RecursiveNode:
    """Recursive dataclass (not currently supported)."""

    value: int
    # Note: recursive schemas may not be supported


# =============================================================================
# Grammar instantiation tests
# =============================================================================


class TestGrammarInstantiation:
    """Tests for Grammar class instantiation."""

    def test_create_from_dataclass(self) -> None:
        """Grammar can be created from a dataclass type."""
        grammar = Grammar(Person)
        assert grammar is not None
        assert grammar.response_format is Person

    def test_create_from_dict_schema(self) -> None:
        """Grammar can be created from a dict schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "integer"},
            },
            "required": ["name", "value"],
        }
        grammar = Grammar(schema)
        assert grammar is not None
        assert grammar.response_format == schema

    def test_schema_property(self) -> None:
        """Grammar.schema returns the compiled JSON schema."""
        grammar = Grammar(Person)
        schema = grammar.schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]

    def test_response_format_property(self) -> None:
        """Grammar.response_format returns the original response format."""
        grammar = Grammar(Person)
        assert grammar.response_format is Person

    def test_repr(self) -> None:
        """Grammar has a useful repr."""
        grammar = Grammar(Person)
        assert repr(grammar) == "Grammar(Person)"

    def test_repr_dict(self) -> None:
        """Grammar created from dict has 'dict' in repr."""
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
        grammar = Grammar(schema)
        assert repr(grammar) == "Grammar(dict)"


# =============================================================================
# Grammar error handling tests
# =============================================================================


class TestGrammarErrorHandling:
    """Tests for Grammar error handling."""

    def test_invalid_schema_raises(self) -> None:
        """Invalid schema raises StructuredOutputError at compile time."""
        # Empty object without properties should fail
        invalid_schema = {"type": "invalid_type"}
        with pytest.raises(StructuredOutputError):
            Grammar(invalid_schema)

    def test_none_raises(self) -> None:
        """None as response_format raises StructuredOutputError."""
        with pytest.raises((StructuredOutputError, TypeError)):
            Grammar(None)  # type: ignore[arg-type]


# =============================================================================
# Grammar reuse tests
# =============================================================================


class TestGrammarReuse:
    """Tests for Grammar reuse across multiple generations."""

    def test_grammar_can_be_reused(self) -> None:
        """Same Grammar instance can be passed multiple times."""
        grammar = Grammar(Person)

        # Grammar should be reusable (schema and handle don't change)
        schema1 = grammar.schema
        schema2 = grammar.schema
        assert schema1 == schema2

        # Handle should remain valid
        assert grammar._handle is not None

    def test_multiple_grammars_independent(self) -> None:
        """Multiple Grammar instances are independent."""
        grammar1 = Grammar(Person)
        grammar2 = Grammar(Answer)

        assert grammar1.schema != grammar2.schema
        assert grammar1.response_format != grammar2.response_format

    def test_same_schema_multiple_grammars(self) -> None:
        """Creating multiple grammars from same schema is valid."""
        grammar1 = Grammar(Person)
        grammar2 = Grammar(Person)

        # Both should have same schema
        assert grammar1.schema == grammar2.schema

        # But different handles (different Python objects)
        assert grammar1 is not grammar2


# =============================================================================
# Grammar lifecycle tests
# =============================================================================


class TestGrammarLifecycle:
    """Tests for Grammar lifecycle management."""

    def test_grammar_cleanup_on_delete(self) -> None:
        """Grammar handle is freed when object is deleted."""
        grammar = Grammar(Person)

        # Handle should be non-null
        assert grammar._handle is not None
        assert grammar._handle != 0

        # Delete should not raise
        del grammar

    def test_grammar_survives_scope(self) -> None:
        """Grammar survives being returned from function."""

        def create_grammar() -> Grammar:
            return Grammar(Person)

        grammar = create_grammar()
        assert grammar._handle is not None
        assert grammar._handle != 0
        assert grammar.schema["type"] == "object"


# =============================================================================
# Grammar with complex schemas
# =============================================================================


class TestGrammarComplexSchemas:
    """Tests for Grammar with complex schemas."""

    def test_nested_object(self) -> None:
        """Grammar handles nested objects."""

        @dataclass
        class Address:
            street: str
            city: str

        @dataclass
        class PersonWithAddress:
            name: str
            address: Address

        grammar = Grammar(PersonWithAddress)
        schema = grammar.schema
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        # Nested object should be in schema (either inline or in $defs)
        assert "address" in schema["properties"] or "$defs" in schema

    def test_optional_fields(self) -> None:
        """Grammar handles optional fields."""

        @dataclass
        class OptionalPerson:
            name: str
            nickname: str | None = None

        grammar = Grammar(OptionalPerson)
        schema = grammar.schema
        assert schema["type"] == "object"
        # name should be required, nickname should not be
        assert "name" in schema.get("required", [])

    def test_list_field(self) -> None:
        """Grammar handles list fields."""

        @dataclass
        class Team:
            members: list[str]

        grammar = Grammar(Team)
        schema = grammar.schema
        assert schema["type"] == "object"
        assert "members" in schema["properties"]
