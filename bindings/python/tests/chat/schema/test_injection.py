"""Tests for schema -> TypeScript prompt injection."""

from typing import Literal

from pydantic import BaseModel, Field

from talu.template.schema.injection import schema_to_prompt_description


class Weather(BaseModel):
    """Weather response."""

    location: str
    temperature: float
    unit: Literal["celsius", "fahrenheit"]


class Address(BaseModel):
    street: str
    city: str
    zip_code: str = Field(alias="zip")


class User(BaseModel):
    name: str
    email: str
    address: Address
    tags: list[str] = []


class TestBasicTypeScript:
    """Test basic TypeScript interface generation."""

    def test_simple_model(self):
        schema = Weather.model_json_schema()
        result = schema_to_prompt_description(schema)

        assert "interface Response {" in result
        assert "location: string;" in result
        assert "temperature: number;" in result
        assert '"celsius" | "fahrenheit"' in result

    def test_optional_fields(self):
        class OptionalModel(BaseModel):
            required_field: str
            optional_field: str | None = None

        schema = OptionalModel.model_json_schema()
        result = schema_to_prompt_description(schema)

        assert "required_field: string;" in result
        assert "optional_field?: string | null;" in result

    def test_array_types(self):
        class ListModel(BaseModel):
            items: list[str]
            numbers: list[int]

        schema = ListModel.model_json_schema()
        result = schema_to_prompt_description(schema)

        assert "items: string[];" in result
        assert "numbers: number[];" in result


class TestNestedModels:
    """Test $defs handling - flat structure, not deep nesting."""

    def test_nested_model_flat_structure(self):
        """Nested models should produce separate interface definitions."""
        schema = User.model_json_schema()
        result = schema_to_prompt_description(schema)

        assert "interface Address {" in result
        assert "interface User {" in result or "interface Response {" in result

        assert "street: string;" in result
        assert "city: string;" in result

        assert "address: Address;" in result

    def test_deeply_nested_not_inlined(self):
        """Deep nesting should not produce unreadable inline types."""

        class Inner(BaseModel):
            value: int

        class Middle(BaseModel):
            inner: Inner

        class Outer(BaseModel):
            middle: Middle

        schema = Outer.model_json_schema()
        result = schema_to_prompt_description(schema)

        assert "{ ... }" not in result or result.count("{ ... }") == 0

        assert "interface Inner {" in result
        assert "interface Middle {" in result


class TestDescriptions:
    """Test that Field descriptions become comments."""

    def test_field_descriptions(self):
        class Documented(BaseModel):
            name: str = Field(description="The user's full name")
            age: int = Field(description="Age in years")

        schema = Documented.model_json_schema()
        result = schema_to_prompt_description(schema)

        assert "// The user's full name" in result
        assert "// Age in years" in result


class TestRecursiveModels:
    """Test handling of self-referential models."""

    def test_simple_recursion(self):
        """Recursive models should not cause stack overflow."""
        from typing import Optional

        class Node(BaseModel):
            value: int
            next: Optional["Node"] = None

        schema = Node.model_json_schema()
        result = schema_to_prompt_description(schema)

        assert "interface Node {" in result
        assert "next?: Node | null;" in result

    def test_comment_tree_recursion(self):
        """Comment-style recursion with lists."""

        class Comment(BaseModel):
            text: str
            replies: list["Comment"] = []

        schema = Comment.model_json_schema()
        result = schema_to_prompt_description(schema)

        assert "interface Comment {" in result
        assert "replies" in result
        assert "Comment[]" in result


class TestRootUnions:
    """Test handling of root-level union types (anyOf/oneOf)."""

    def test_root_union_schema(self):
        """Root anyOf should generate TypeScript union type."""

        class SuccessResponse(BaseModel):
            status: Literal["success"]
            data: str

        class ErrorResponse(BaseModel):
            status: Literal["error"]
            message: str

        schema = {
            "anyOf": [
                SuccessResponse.model_json_schema(),
                ErrorResponse.model_json_schema(),
            ]
        }

        result = schema_to_prompt_description(schema)

        assert "type Response = " in result or "type I_Response = " in result
        assert (
            "SuccessResponse | ErrorResponse" in result
            or "I_SuccessResponse | I_ErrorResponse" in result
        )

    def test_union_with_discriminator(self):
        """Union types with discriminator field."""
        from typing import Literal

        class Cat(BaseModel):
            type: Literal["cat"]
            meows: bool

        class Dog(BaseModel):
            type: Literal["dog"]
            barks: bool

        schema = {
            "oneOf": [
                Cat.model_json_schema(),
                Dog.model_json_schema(),
            ],
            "discriminator": {"propertyName": "type"},
        }

        result = schema_to_prompt_description(schema)

        assert "Cat" in result or "I_Cat" in result
        assert "Dog" in result or "I_Dog" in result


class TestRootArrays:
    """Test handling of root-level array types (list[Model])."""

    def test_root_array_of_objects(self):
        """Root array should generate TypeScript array type."""
        from pydantic import RootModel

        class User(BaseModel):
            name: str
            email: str

        class UserList(RootModel[list[User]]):
            pass

        schema = UserList.model_json_schema()
        result = schema_to_prompt_description(schema)

        assert "User[]" in result or "I_User[]" in result
        assert "interface User" in result or "interface I_User" in result
        assert "name: string;" in result
        assert "email: string;" in result

    def test_root_array_of_primitives(self):
        """Root array of primitives."""
        from pydantic import RootModel

        class StringList(RootModel[list[str]]):
            pass

        schema = StringList.model_json_schema()
        result = schema_to_prompt_description(schema)

        assert "string[]" in result
        assert "interface Response" not in result or "type Response = string[]" in result

    def test_root_array_grammar_accepts_bracket(self):
        """Grammar should allow [ as start token for root arrays."""
        from pydantic import RootModel

        class Item(BaseModel):
            value: int

        class ItemList(RootModel[list[Item]]):
            pass

        schema = ItemList.model_json_schema()
        assert schema.get("type") == "array"

    def test_nested_arrays(self):
        """Nested arrays (list[list[str]])."""
        from pydantic import RootModel

        class Matrix(RootModel[list[list[int]]]):
            pass

        schema = Matrix.model_json_schema()
        result = schema_to_prompt_description(schema)

        assert "number[][]" in result or "int[][]" in result
