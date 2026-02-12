"""Streaming events for structured output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "DataEvent",
    "ErrorEvent",
    "ResponseMetadata",
]


@dataclass
class DataEvent:
    """JSON data is being streamed.

    Attributes
    ----------
        snapshot: The current parsed JSON state (complete to the extent parsed).
        delta: Keys that changed since the last event (shallow diff).
            For dicts: contains keys that are new or have different values.
            For lists: returns the entire new list (no element-level diffing).
            None if nothing changed or on first event with a list snapshot.
    """

    snapshot: dict[str, Any] | list[Any]
    delta: dict[str, Any] | list[Any] | None = None


@dataclass
class ErrorEvent:
    """Non-fatal parsing error during streaming."""

    error: str


@dataclass
class ResponseMetadata:
    """Generation metadata and debug information."""

    finish_reason: str
    schema_tokens: int = 0
    schema_injection: str | None = None
    grammar_gbnf: str | None = None
    grammar_trace: list[str] | None = None
    prefill_success: bool | None = None
