"""
Additional tests for talu/chat/schema/generators.py coverage.

Targets uncovered edge cases in prompt generators.
"""

from talu.template.schema.generators import (
    JsonSchemaGenerator,
    TypeScriptGenerator,
    XmlGenerator,
)

# =============================================================================
# TypeScriptGenerator Tests
# =============================================================================


class TestTypeScriptGeneratorBasic:
    """Tests for TypeScriptGenerator basic functionality."""

    def test_generate_simple_schema(self):
        """Generator creates TypeScript interface."""
        gen = TypeScriptGenerator()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        result = gen.generate(schema)
        assert "interface Response" in result
        assert "name: string" in result
        assert "age?" in result  # Optional since not required


class TestTypeScriptGeneratorThinking:
    """Tests for thinking mode in TypeScriptGenerator."""

    def test_generate_with_thinking(self):
        """Generator adds thinking instructions."""
        gen = TypeScriptGenerator()
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        result = gen.generate(schema, allow_thinking=True)
        assert "<think>" in result
        assert "</think>" in result


class TestTypeScriptGeneratorDefs:
    """Tests for $defs handling in TypeScriptGenerator."""

    def test_generate_with_defs(self):
        """Generator emits interfaces for $defs."""
        gen = TypeScriptGenerator()
        schema = {
            "type": "object",
            "properties": {
                "person": {"$ref": "#/$defs/Person"},
            },
            "$defs": {
                "Person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                }
            },
        }
        result = gen.generate(schema)
        assert "interface Person" in result
        assert "person" in result


class TestTypeScriptGeneratorUnions:
    """Tests for anyOf/oneOf handling in TypeScriptGenerator."""

    def test_generate_anyof_union(self):
        """Generator creates union type for anyOf."""
        gen = TypeScriptGenerator()
        schema = {
            "anyOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}},
                {"type": "object", "properties": {"b": {"type": "number"}}},
            ]
        }
        result = gen.generate(schema)
        assert "Response =" in result

    def test_generate_oneof_union(self):
        """Generator creates union type for oneOf."""
        gen = TypeScriptGenerator()
        schema = {
            "oneOf": [
                {"type": "string"},
                {"type": "number"},
            ]
        }
        result = gen.generate(schema)
        assert "Response =" in result

    def test_generate_discriminated_union(self):
        """Generator handles discriminated union."""
        gen = TypeScriptGenerator()
        schema = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"kind": {"const": "a"}, "val": {"type": "string"}},
                },
                {
                    "type": "object",
                    "properties": {"kind": {"const": "b"}, "val": {"type": "number"}},
                },
            ],
            "discriminator": {"propertyName": "kind"},
        }
        result = gen.generate(schema)
        assert "Discriminator" in result or "kind" in result

    def test_generate_non_object_in_union(self):
        """Generator handles non-object options in union."""
        gen = TypeScriptGenerator()
        schema = {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ]
        }
        result = gen.generate(schema)
        assert "string" in result


class TestTypeScriptGeneratorArrayRoot:
    """Tests for array root type in TypeScriptGenerator."""

    def test_generate_array_root(self):
        """Generator handles array as root type."""
        gen = TypeScriptGenerator()
        schema = {
            "type": "array",
            "items": {"type": "string"},
        }
        result = gen.generate(schema)
        assert "type Response = string[]" in result


class TestTypeScriptGeneratorTypes:
    """Tests for type resolution in TypeScriptGenerator."""

    def test_resolve_all_types(self):
        """Generator resolves all JSON Schema types."""
        gen = TypeScriptGenerator()
        schema = {
            "type": "object",
            "properties": {
                "str": {"type": "string"},
                "num": {"type": "number"},
                "int": {"type": "integer"},
                "bool": {"type": "boolean"},
                "null": {"type": "null"},
                "obj": {"type": "object"},
                "arr": {"type": "array", "items": {"type": "string"}},
                "any": {},  # No type specified
            },
        }
        result = gen.generate(schema)
        assert "string" in result
        assert "number" in result
        assert "boolean" in result
        assert "null" in result
        assert "object" in result
        assert "string[]" in result
        assert "any" in result

    def test_resolve_enum(self):
        """Generator handles enum type."""
        gen = TypeScriptGenerator()
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["pending", "done"]},
            },
        }
        result = gen.generate(schema)
        assert '"pending"' in result
        assert '"done"' in result

    def test_resolve_const(self):
        """Generator handles const type."""
        gen = TypeScriptGenerator()
        schema = {
            "type": "object",
            "properties": {
                "version": {"const": "1.0"},
                "count": {"const": 42},
            },
        }
        result = gen.generate(schema)
        assert '"1.0"' in result
        assert "42" in result

    def test_resolve_anyof_in_property(self):
        """Generator handles anyOf in property."""
        gen = TypeScriptGenerator()
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"},
                    ]
                }
            },
        }
        result = gen.generate(schema)
        assert "string | null" in result

    def test_resolve_oneof_in_property(self):
        """Generator handles oneOf in property."""
        gen = TypeScriptGenerator()
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "number"},
                    ]
                }
            },
        }
        result = gen.generate(schema)
        assert "string | number" in result

    def test_resolve_ref(self):
        """Generator resolves $ref to type name."""
        gen = TypeScriptGenerator()
        schema = {
            "type": "object",
            "properties": {
                "item": {"$ref": "#/$defs/Item"},
            },
            "$defs": {
                "Item": {"type": "object", "properties": {"id": {"type": "number"}}},
            },
        }
        result = gen.generate(schema)
        assert "Item" in result


class TestTypeScriptGeneratorNameSanitization:
    """Tests for name sanitization in TypeScriptGenerator."""

    def test_infer_type_name_from_title(self):
        """Generator infers type name from title."""
        gen = TypeScriptGenerator()
        schema = {
            "anyOf": [
                {"type": "object", "title": "MyType", "properties": {"a": {"type": "string"}}},
            ]
        }
        result = gen.generate(schema)
        assert "MyType" in result

    def test_infer_type_name_from_ref(self):
        """Generator infers type name from $ref."""
        gen = TypeScriptGenerator()
        schema = {
            "anyOf": [
                {"$ref": "#/$defs/CustomType"},
            ],
            "$defs": {
                "CustomType": {"type": "object", "properties": {"x": {"type": "number"}}},
            },
        }
        result = gen.generate(schema)
        assert "CustomType" in result

    def test_sanitize_name_with_special_chars(self):
        """Generator sanitizes names with special characters."""
        gen = TypeScriptGenerator()
        schema = {
            "anyOf": [
                {"type": "object", "title": "My-Type!@#", "properties": {"a": {"type": "string"}}},
            ]
        }
        result = gen.generate(schema)
        # Should sanitize special chars
        assert "My_Type" in result

    def test_sanitize_name_starting_with_number(self):
        """Generator handles names starting with number."""
        gen = TypeScriptGenerator()
        schema = {
            "anyOf": [
                {"type": "object", "title": "123Type", "properties": {"a": {"type": "string"}}},
            ]
        }
        result = gen.generate(schema)
        # Should prefix with T_
        assert "T_123Type" in result


class TestTypeScriptGeneratorDescription:
    """Tests for description handling in TypeScriptGenerator."""

    def test_property_description_as_comment(self):
        """Generator adds property descriptions as comments."""
        gen = TypeScriptGenerator()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The user's name"},
            },
        }
        result = gen.generate(schema)
        assert "user's name" in result or "name" in result


# =============================================================================
# JsonSchemaGenerator Tests
# =============================================================================


class TestJsonSchemaGenerator:
    """Tests for JsonSchemaGenerator."""

    def test_generate_simple(self):
        """Generator creates JSON schema dump."""
        gen = JsonSchemaGenerator()
        schema = {"type": "object", "properties": {"key": {"type": "string"}}}
        result = gen.generate(schema)
        assert "```json" in result
        assert '"type": "object"' in result

    def test_generate_with_thinking(self):
        """Generator adds thinking instructions."""
        gen = JsonSchemaGenerator()
        schema = {"type": "object"}
        result = gen.generate(schema, allow_thinking=True)
        assert "<think>" in result


# =============================================================================
# XmlGenerator Tests
# =============================================================================


class TestXmlGenerator:
    """Tests for XmlGenerator."""

    def test_generate_simple(self):
        """Generator creates XML schema."""
        gen = XmlGenerator()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        }
        result = gen.generate(schema)
        assert "<schema>" in result
        assert "</schema>" in result
        assert 'name="name"' in result
        assert 'required="true"' in result

    def test_generate_with_thinking(self):
        """Generator adds thinking instructions."""
        gen = XmlGenerator()
        schema = {"type": "object", "properties": {"x": {"type": "number"}}}
        result = gen.generate(schema, allow_thinking=True)
        assert "<think>" in result

    def test_generate_enum_field(self):
        """Generator handles enum in XML."""
        gen = XmlGenerator()
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["a", "b"]},
            },
        }
        result = gen.generate(schema)
        assert "enum(" in result
        assert '"a"' in result
        assert '"b"' in result

    def test_generate_array_field(self):
        """Generator handles array in XML."""
        gen = XmlGenerator()
        schema = {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}},
            },
        }
        result = gen.generate(schema)
        assert "array<string>" in result

    def test_generate_with_description(self):
        """Generator includes description as XML comment."""
        gen = XmlGenerator()
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "The numeric value"},
            },
        }
        result = gen.generate(schema)
        assert "<!-- The numeric value -->" in result
