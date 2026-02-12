"""Convert dataclasses and TypedDicts to JSON Schema.

Supported response_format types:
  - dict: JSON Schema passed through as-is
  - dataclass: Converted via dataclass_to_schema()
  - TypedDict: Converted via typeddict_to_schema()
  - Pydantic BaseModel: Converted via model_json_schema()

Performance note (grammar compilation):
  Schema-to-grammar compilation costs 3-9 microseconds for typical schemas
  (simple: ~4μs, nested: ~7μs, complex with unions: ~9μs). This is negligible
  compared to LLM inference (100ms-10s), so implicit caching is not implemented.
  For explicit pre-compilation, use the Grammar class directly.
"""

from __future__ import annotations

import dataclasses
import typing
import warnings
from enum import Enum
from typing import Any, get_args, get_origin, is_typeddict, overload

from ...exceptions import StructuredOutputError, ValidationError


class AmbiguousUnionWarning(UserWarning):
    """Warning emitted when a union type lacks a discriminator field."""

    pass


def _check_ambiguous_union(schema: dict[str, Any]) -> None:
    """Check for ambiguous unions and emit a warning if found.

    A union is considered ambiguous when:
    1. It uses anyOf/oneOf (at root or nested in properties)
    2. It has no discriminator field defined
    3. The union members don't have distinguishing Literal fields

    When detected, emits an AmbiguousUnionWarning with guidance on adding
    a discriminator field.
    """
    defs = schema.get("$defs") or schema.get("definitions") or {}

    def resolve_ref(obj: dict[str, Any]) -> dict[str, Any]:
        """Resolve $ref to actual schema."""
        ref = obj.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            def_name = ref.split("/")[-1]
            return defs.get(def_name, obj)
        return obj

    def get_type_name(obj: dict[str, Any]) -> str:
        """Extract type name from schema object."""
        resolved = resolve_ref(obj)
        return resolved.get("title", "Unknown")

    def has_literal_discriminator(obj: dict[str, Any]) -> bool:
        """Check if object has a Literal field that could serve as discriminator."""
        resolved = resolve_ref(obj)
        properties = resolved.get("properties", {})
        for prop_schema in properties.values():
            # Check for const (Literal with single value)
            if "const" in prop_schema:
                return True
            # Check for enum with single value (another Literal representation)
            enum_vals = prop_schema.get("enum")
            if isinstance(enum_vals, list) and len(enum_vals) == 1:
                return True
        return False

    def check_union(node: dict[str, Any], stacklevel: int) -> None:
        """Check a schema node for ambiguous unions, recursively."""
        if not isinstance(node, dict):
            return

        # Check for union at this level
        union_key = "anyOf" if "anyOf" in node else ("oneOf" if "oneOf" in node else None)
        if union_key is not None and not node.get("discriminator"):
            options = node.get(union_key, [])
            if len(options) >= 2:
                # Check if all members have distinguishing Literal fields
                all_have_discriminator = all(
                    has_literal_discriminator(resolve_ref(opt)) for opt in options
                )
                if not all_have_discriminator:
                    # Collect type names for the warning message
                    type_names = [get_type_name(opt) for opt in options]
                    types_str = ", ".join(type_names)
                    example_class = type_names[0] if type_names else "MyClass"
                    example_value = type_names[0].lower() if type_names else "myclass"

                    warning_msg = (
                        f"Union[{types_str}] members have no discriminator field. "
                        f"For reliable disambiguation, add a discriminator field:\n\n"
                        f"    class {example_class}(BaseModel):\n"
                        f'        kind: Literal["{example_value}"] = "{example_value}"'
                        f"  # Add this line\n"
                        f"        ..."
                    )
                    warnings.warn(warning_msg, AmbiguousUnionWarning, stacklevel=stacklevel)

        # Recursively check properties
        properties = node.get("properties", {})
        for prop_schema in properties.values():
            if isinstance(prop_schema, dict):
                check_union(prop_schema, stacklevel)

        # Check array items
        items = node.get("items")
        if isinstance(items, dict):
            check_union(items, stacklevel)

    # Start checking from root with appropriate stacklevel
    # stacklevel=4: warn -> check_union -> _check_ambiguous_union -> normalize_response_format -> user code
    check_union(schema, stacklevel=5)


MAX_SCHEMA_DEPTH = 32


def _validate_schema_depth(schema: dict[str, Any]) -> None:
    defs = schema.get("$defs") or schema.get("definitions") or {}
    active_refs: set[str] = set()

    def walk(node: Any, depth: int) -> None:
        if depth > MAX_SCHEMA_DEPTH:
            raise StructuredOutputError(
                f"Schema nesting exceeds {MAX_SCHEMA_DEPTH} levels; consider simplifying the model."
            )
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                def_name = ref.split("/")[-1]
                target = defs.get(def_name)
                if target is None:
                    raise StructuredOutputError(f"Schema $ref not found: {ref}")
                if def_name in active_refs:
                    return
                active_refs.add(def_name)
                walk(target, depth + 1)
                active_refs.remove(def_name)
                return

            for key in ("properties", "patternProperties"):
                props = node.get(key)
                if isinstance(props, dict):
                    for value in props.values():
                        walk(value, depth + 1)

            items = node.get("items")
            if isinstance(items, list):
                for item in items:
                    walk(item, depth + 1)
            elif isinstance(items, dict):
                walk(items, depth + 1)

            for key in ("anyOf", "oneOf", "allOf"):
                options = node.get(key)
                if isinstance(options, list):
                    for option in options:
                        walk(option, depth + 1)

        elif isinstance(node, list):
            for item in node:
                walk(item, depth + 1)

    walk(schema, 0)
    for def_schema in defs.values():
        if isinstance(def_schema, dict):
            walk(def_schema, 1)


def _get_type_name(t: type[Any]) -> str:
    return getattr(t, "__name__", str(t))


def _py_type_to_json_schema(t: type | Any) -> dict[str, Any]:
    """Recursively convert a Python type to a JSON schema dict."""
    origin = get_origin(t)
    args = get_args(t)

    if origin is typing.Union and type(None) in args:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _py_type_to_json_schema(non_none_args[0])

    # Handle list types: both bare `list` and parameterized `list[T]`
    # For bare `list`, origin is None but t is list itself
    # For `list[T]`, origin is list
    if origin is list or t is list:
        item_type = args[0] if args else Any
        return {"type": "array", "items": _py_type_to_json_schema(item_type)}

    if dataclasses.is_dataclass(t):
        return dataclass_to_schema(t)

    if is_typeddict(t):
        return typeddict_to_schema(t)

    if isinstance(t, type) and issubclass(t, Enum):
        return {"type": "string", "enum": [e.value for e in t]}

    if t is str:
        return {"type": "string"}
    if t is int:
        return {"type": "integer"}
    if t is float:
        return {"type": "number"}
    if t is bool:
        return {"type": "boolean"}

    return {"type": "string"}


def dataclass_to_schema(model: type | Any) -> dict[str, Any]:
    """Convert a python dataclass to JSON Schema (Draft 2020-12 compatible)."""
    if not dataclasses.is_dataclass(model):
        raise ValueError(f"Target must be a dataclass, got {_get_type_name(model)}")

    properties: dict[str, Any] = {}
    required: list[str] = []

    for field in dataclasses.fields(model):
        field_schema = _py_type_to_json_schema(field.type)
        if "description" in field.metadata:
            field_schema["description"] = field.metadata["description"]
        properties[field.name] = field_schema
        if field.default == dataclasses.MISSING and field.default_factory == dataclasses.MISSING:
            required.append(field.name)

    schema: dict[str, Any] = {
        "type": "object",
        "title": getattr(model, "__name__", str(model)),
        "properties": properties,
    }

    if required:
        schema["required"] = required

    return schema


def typeddict_to_schema(td: type) -> dict[str, Any]:
    """Convert a TypedDict to JSON Schema (Draft 2020-12 compatible)."""
    if not is_typeddict(td):
        raise ValueError(f"Target must be a TypedDict, got {_get_type_name(td)}")

    properties: dict[str, Any] = {}
    for name, type_hint in typing.get_type_hints(td).items():
        properties[name] = _py_type_to_json_schema(type_hint)

    schema: dict[str, Any] = {
        "type": "object",
        "title": getattr(td, "__name__", str(td)),
        "properties": properties,
    }

    required = list(td.__required_keys__)
    if required:
        schema["required"] = required

    return schema


@overload
def normalize_response_format(fmt: None) -> None: ...


@overload
def normalize_response_format(fmt: type[Any] | dict[str, Any]) -> dict[str, Any]: ...


def normalize_response_format(
    fmt: type[Any] | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Normalize response_format to JSON Schema dict.

    Supports:
    - dict: Already a JSON Schema, used as-is
    - dataclass: Converted to JSON Schema via dataclass_to_schema
    - TypedDict: Converted to JSON Schema via typeddict_to_schema
    - Pydantic BaseModel: Converted via model_json_schema() (no pydantic import needed)
    """
    if fmt is None:
        return None

    if isinstance(fmt, dict):
        schema = fmt
        _validate_schema_depth(schema)
        _check_ambiguous_union(schema)
        return schema

    if dataclasses.is_dataclass(fmt):
        schema = dataclass_to_schema(fmt)
        _validate_schema_depth(schema)
        return schema

    if is_typeddict(fmt):
        schema = typeddict_to_schema(fmt)
        _validate_schema_depth(schema)
        return schema

    # Pydantic v2 models have model_json_schema method (detected without importing pydantic)
    model_json_schema = getattr(fmt, "model_json_schema", None)
    if callable(model_json_schema):
        schema = model_json_schema()
        if not isinstance(schema, dict):
            raise ValidationError(
                f"model_json_schema() returned {type(schema).__name__}, expected dict",
                code="INVALID_ARGUMENT",
            )
        _validate_schema_depth(schema)
        _check_ambiguous_union(schema)
        return schema

    raise ValidationError(
        f"Invalid response_format type: {type(fmt)}. "
        "Expected dict (JSON schema), dataclass, TypedDict, or Pydantic BaseModel.",
        code="INVALID_ARGUMENT",
        details={"param": "response_format", "type": type(fmt).__name__},
    )
