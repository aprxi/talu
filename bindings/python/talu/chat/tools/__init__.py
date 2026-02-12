"""
Tool calling utilities for chat sessions.

This package defines the tool call types and the ``@tool`` decorator.
"""

from .decorator import tool
from .types import (
    ToolCall,
    ToolCallFunction,
    ToolExecutionError,
    ToolResult,
    ToolState,
    ToolStatus,
)

__all__ = [
    "ToolCallFunction",
    "ToolCall",
    "ToolExecutionError",
    "ToolStatus",
    "ToolState",
    "ToolResult",
    "tool",
]
