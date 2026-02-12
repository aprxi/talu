"""
Tool calling types for agent applications.

Provides OpenAI-compatible tool call types and OpenCode-style execution state.

OpenAI Format (ToolCall, ToolCallFunction, ToolResult):
    Standard format for tool calls, compatible with OpenWebUI, agent frameworks,
    and existing tool-calling ecosystems.

    >>> if response.tool_calls:
    ...     for tool in response.tool_calls:
    ...         result = execute_tool(tool.name, tool.function.arguments_parsed())
    ...         chat.add_tool_result(tool.id, result)

OpenCode Format (ToolState, ToolStatus):
    Richer state tracking for streaming UIs that need to show tool execution
    progress in real-time.

    >>> state = ToolState(status="running", title="Searching...")
    >>> # ... later ...
    >>> state = ToolState(status="completed", title="Found 10 results", output="...")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ...exceptions import TaluError

__all__ = [
    "ToolCallFunction",
    "ToolCall",
    "ToolExecutionError",
    "ToolStatus",
    "ToolState",
    "ToolResult",
]


class ToolExecutionError(TaluError):
    """
    Error raised when a tool call cannot be executed.

    This covers missing function bindings or execution configuration issues.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "TOOL_EXECUTION_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


@dataclass
class ToolCallFunction:
    """
    Function name and arguments within a tool call.

    Attributes
    ----------
        name: Name of the function to call.
        arguments: JSON string of arguments to pass.
    """

    name: str
    arguments: str

    def arguments_parsed(self) -> dict:
        """Parse arguments as dict. Returns empty dict on parse failure."""
        import json

        try:
            return json.loads(self.arguments) if self.arguments else {}
        except (json.JSONDecodeError, TypeError):
            return {}


@dataclass
class ToolCall:
    """
    Tool call requested by the model.

    Follows the OpenAI tool call format for compatibility with agent
    frameworks and tool-calling workflows.

    Attributes
    ----------
        id: Unique identifier for this tool call.
        type: Always "function" for function calls.
        function: The function details (name and arguments).

    Example:
        >>> if response.tool_calls:
        ...     for tool in response.tool_calls:
        ...         print(f"Call: {tool.function.name}")
        ...         args = tool.function.arguments_parsed()
        ...         result = execute_tool(tool.function.name, args)
    """

    id: str
    type: str  # Always "function" for now
    function: ToolCallFunction
    _func: Callable[..., Any] | None = field(default=None, repr=False)

    @classmethod
    def create(cls, id: str, name: str, arguments: str) -> ToolCall:
        """Create a ToolCall with the given parameters."""
        return cls(
            id=id, type="function", function=ToolCallFunction(name=name, arguments=arguments)
        )

    @property
    def name(self) -> str:
        """Convenience access to function name."""
        return self.function.name

    @property
    def arguments(self) -> str:
        """Convenience access to function arguments."""
        return self.function.arguments

    def execute(self) -> Any:
        """
        Execute the tool call by invoking the mapped Python function.

        Returns
        -------
            The return value of the tool function.

        Raises
        ------
            ToolExecutionError: If no function is mapped to this tool call.
        """
        if self._func is None:
            raise ToolExecutionError(
                f"Tool '{self.name}' has no mapped function. "
                "Pass the tool to chat.send(tools=[...])."
            )
        args = self.function.arguments_parsed()
        return self._func(**args)

    async def execute_async(self) -> Any:
        """
        Execute the tool call asynchronously.

        Awaits coroutine functions directly. Runs sync functions in an
        executor to avoid blocking the event loop.

        Returns
        -------
            The return value of the tool function.

        Raises
        ------
            ToolExecutionError: If no function is mapped to this tool call.
        """
        import asyncio

        if self._func is None:
            raise ToolExecutionError(
                f"Tool '{self.name}' has no mapped function. "
                "Pass the tool to chat.send(tools=[...])."
            )

        func = self._func
        args = self.function.arguments_parsed()
        if asyncio.iscoroutinefunction(func):
            return await func(**args)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(**args))


class ToolStatus:
    """Constants for tool execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ToolState:
    r"""
    Tool execution state for streaming UIs.

    Provides state tracking for live UI updates during tool execution.

    Attributes
    ----------
        status: Current status (pending, running, completed, error).
        title: Human-readable title for UI display.
        input: Parsed input arguments (dict, not JSON string).
        output: Tool result (when completed).
        error: Error message (when error).
        metadata: Additional metadata for UI display.
        time_start: When execution started (Unix timestamp).
        time_end: When execution ended (Unix timestamp).

    Example - Streaming updates:
        >>> # Tool starts
        >>> state = ToolState(status="running", title="Searching...", input={"query": "python"})
        >>>
        >>> # Tool completes
        >>> state = ToolState(
        ...     status="completed",
        ...     title="Found 10 results",
        ...     input={"query": "python"},
        ...     output="1. Python docs\n2. ...",
        ... )
    """

    status: str  # pending, running, completed, error
    input: dict | None = None
    title: str | None = None
    output: str | None = None
    error: str | None = None
    metadata: dict | None = None
    time_start: float | None = None
    time_end: float | None = None


@dataclass
class ToolResult:
    """
    Result of a tool execution.

    Added back to the conversation history so the model can incorporate
    the tool output in its next response.

    Attributes
    ----------
        tool_call_id: ID of the tool call this is responding to.
        content: The tool's output/result.
        is_error: Whether this result represents an error.

    Example:
        >>> # Execute tool and add result
        >>> result = execute_tool(tool.function.name, tool.function.arguments_parsed())
        >>> # Tool results are added to the conversation automatically during generation
    """

    tool_call_id: str
    content: str
    is_error: bool = False

    def to_message(self) -> dict:
        """Convert to OpenAI message format."""
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }
