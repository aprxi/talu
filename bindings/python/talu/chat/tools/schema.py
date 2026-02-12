"""Schema extraction helpers for tool functions."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

__all__: list[str] = []


def parse_docstring(docstring: str) -> dict[str, Any]:
    r"""
    Parse Google-style docstring into structured data.

    Args:
        docstring: Raw docstring text.

    Returns
    -------
        Mapping with summary and argument descriptions.

    Example:
        >>> parse_docstring(\"\"\"Greet someone.\n\nArgs:\n    name: The name\n\"\"\")\n
        {'summary': 'Greet someone.', 'args': {'name': 'The name'}}
    """
    lines = docstring.strip().split("\n") if docstring else []
    result: dict[str, Any] = {"summary": "", "args": {}}

    # First non-empty line is summary
    for line in lines:
        stripped = line.strip()
        if stripped:
            result["summary"] = stripped
            break

    # Find and parse Args: section
    in_args = False
    current_arg: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped == "Args:":
            in_args = True
            continue
        if stripped in ("Returns:", "Raises:", "Example:", "Examples:"):
            in_args = False
            continue
        if in_args and stripped:
            if ":" in stripped and not stripped.startswith(" "):
                name, desc = stripped.split(":", 1)
                current_arg = name.strip()
                result["args"][current_arg] = desc.strip()
            elif current_arg:
                result["args"][current_arg] += " " + stripped

    return result


def _is_optional(type_hint: Any) -> tuple[bool, Any]:
    origin = get_origin(type_hint)
    if origin is Union:
        args = get_args(type_hint)
        none_types = [arg for arg in args if arg is type(None)]
        if none_types:
            remaining = [arg for arg in args if arg is not type(None)]
            if len(remaining) == 1:
                return True, remaining[0]
    return False, type_hint


def python_type_to_json(type_hint: Any) -> dict[str, Any]:
    """
    Map a Python type hint to JSON Schema.

    Args:
        type_hint: A Python type annotation.

    Returns
    -------
        JSON schema fragment with a required "type" key and optional extras.
    """
    is_optional, inner = _is_optional(type_hint)
    type_hint = inner

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is Literal:
        enum_values = list(args)
        enum_type = "string"
        if enum_values and all(isinstance(v, bool) for v in enum_values):
            enum_type = "boolean"
        elif enum_values and all(isinstance(v, int) for v in enum_values):
            enum_type = "integer"
        elif enum_values and all(isinstance(v, float) for v in enum_values):
            enum_type = "number"
        elif enum_values and all(isinstance(v, str) for v in enum_values):
            enum_type = "string"
        else:
            enum_type = "string"
        return {
            "type": enum_type,
            "extra": {"enum": enum_values},
            "optional": is_optional,
        }

    if origin is list:
        item_type = args[0] if args else str
        item_schema = python_type_to_json(item_type)
        return {
            "type": "array",
            "extra": {"items": {"type": item_schema["type"], **item_schema.get("extra", {})}},
            "optional": is_optional,
        }

    if origin is dict:
        return {"type": "object", "optional": is_optional}

    if type_hint is str:
        return {"type": "string", "optional": is_optional}
    if type_hint is int:
        return {"type": "integer", "optional": is_optional}
    if type_hint is float:
        return {"type": "number", "optional": is_optional}
    if type_hint is bool:
        return {"type": "boolean", "optional": is_optional}
    if type_hint is list:
        return {"type": "array", "optional": is_optional}
    if type_hint is dict:
        return {"type": "object", "optional": is_optional}

    # Unknown or unsupported type -> string fallback
    return {"type": "string", "optional": is_optional}


def extract_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """
    Extract OpenAI-compatible tool schema from a Python function.

    Args:
        func: Function to inspect.

    Returns
    -------
        JSON schema dict for the inner "function" object.
    """
    try:
        hints = get_type_hints(func)
    except (NameError, TypeError):
        hints = getattr(func, "__annotations__", {})
    sig = inspect.signature(func)
    doc = parse_docstring(func.__doc__ or "")

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        type_hint = hints.get(name, str)
        json_type = python_type_to_json(type_hint)

        description = doc.get("args", {}).get(name, "")

        prop: dict[str, Any] = {"type": json_type["type"]}
        prop.update(json_type.get("extra", {}))
        if description:
            prop["description"] = description

        properties[name] = prop

        is_optional = json_type.get("optional", False)
        if param.default is inspect.Parameter.empty and not is_optional:
            required.append(name)

    return {
        "name": func.__name__,
        "description": doc.get("summary", ""),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
