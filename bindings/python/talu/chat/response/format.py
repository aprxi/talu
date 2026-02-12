"""ResponseFormat - Structured output format specification."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ResponseFormat"]


@dataclass
class ResponseFormat:
    """
    Structured output format specification.

    Used to constrain generation to produce valid JSON matching a schema.

    Attributes
    ----------
        type: The format type ("text" or "json_object").
        json_schema: JSON Schema dict for structured output (when type="json_object").

    Example:
        >>> config = GenerationConfig(
        ...     response_format=ResponseFormat(
        ...         type="json_object",
        ...         json_schema={"type": "object", "properties": {"name": {"type": "string"}}}
        ...     )
        ... )
    """

    type: str = "text"  # "text", "json_object"
    json_schema: dict | None = None
