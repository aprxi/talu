"""Tests for ResponseFormat dataclass."""

from talu.chat import ResponseFormat


class TestResponseFormatCreation:
    """Tests for ResponseFormat dataclass creation."""

    def test_default_values(self):
        """ResponseFormat has sensible defaults."""
        rf = ResponseFormat()
        assert rf.type == "text"
        assert rf.json_schema is None

    def test_text_type(self):
        """ResponseFormat with type='text' is the default."""
        rf = ResponseFormat(type="text")
        assert rf.type == "text"

    def test_json_object_type(self):
        """ResponseFormat can be created with json_object type."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        rf = ResponseFormat(type="json_object", json_schema=schema)
        assert rf.type == "json_object"
        assert rf.json_schema == schema


class TestResponseFormatJsonSchema:
    """Tests for ResponseFormat json_schema handling."""

    def test_simple_object_schema(self):
        """ResponseFormat accepts simple object schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        rf = ResponseFormat(type="json_object", json_schema=schema)
        assert rf.json_schema["type"] == "object"
        assert "name" in rf.json_schema["properties"]

    def test_nested_schema(self):
        """ResponseFormat accepts nested schemas."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string", "format": "email"},
                    },
                },
            },
        }
        rf = ResponseFormat(type="json_object", json_schema=schema)
        assert rf.json_schema["properties"]["user"]["type"] == "object"

    def test_array_schema(self):
        """ResponseFormat accepts array schemas."""
        schema = {
            "type": "array",
            "items": {"type": "string"},
        }
        rf = ResponseFormat(type="json_object", json_schema=schema)
        assert rf.json_schema["type"] == "array"


class TestResponseFormatEquality:
    """Tests for ResponseFormat equality comparison."""

    def test_equal_text_formats(self):
        """Two text ResponseFormats are equal."""
        rf1 = ResponseFormat(type="text")
        rf2 = ResponseFormat(type="text")
        assert rf1 == rf2

    def test_equal_json_formats(self):
        """Two json ResponseFormats with same schema are equal."""
        schema = {"type": "string"}
        rf1 = ResponseFormat(type="json_object", json_schema=schema)
        rf2 = ResponseFormat(type="json_object", json_schema=schema)
        assert rf1 == rf2

    def test_unequal_types(self):
        """ResponseFormats with different types are not equal."""
        rf1 = ResponseFormat(type="text")
        rf2 = ResponseFormat(type="json_object")
        assert rf1 != rf2

    def test_unequal_schemas(self):
        """ResponseFormats with different schemas are not equal."""
        rf1 = ResponseFormat(type="json_object", json_schema={"type": "string"})
        rf2 = ResponseFormat(type="json_object", json_schema={"type": "number"})
        assert rf1 != rf2
