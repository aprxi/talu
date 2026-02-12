"""Tool decorator for registering Python functions as LLM tools."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from .schema import extract_schema

__all__ = ["tool"]


def tool(func: Callable[..., Any]) -> Callable[..., Any]:
    r"""
    Mark a function as an LLM tool.

    Extracts JSON Schema from type hints and docstring at decoration time.
    The decorated function can be passed to ``Chat.send(tools=[...])``.

    Args:
        func: Function to register as a tool.

    Returns
    -------
        Wrapped function with tool metadata attached.

    Raises
    ------
        None.

    Example:
        >>> from talu.chat.tools import tool
        >>>
        >>> @tool
        ... def greet(name: str) -> str:
        ...     \"\"\"Greet someone.
        ...
        ...     Args:
        ...         name: The person's name
        ...     \"\"\"
        ...     return f\"Hello, {name}!\"
    """
    schema = extract_schema(func)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    wrapper._tool_schema = schema  # type: ignore[attr-defined]
    wrapper._tool_func = func  # type: ignore[attr-defined]
    wrapper._is_tool = True  # type: ignore[attr-defined]

    return wrapper
