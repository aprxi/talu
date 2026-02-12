"""Tool streaming - Show tool execution state in real-time.

Primary API: talu.Chat, talu.chat.ToolState
Scope: Single

This example shows the OpenCode-style ToolState for building
rich streaming UIs that show tool progress.

Related:
    - examples/basics/02_streaming.py
"""

import time

import talu
from talu.chat import ToolCall, ToolState, ToolStatus
from talu.chat.tools import tool


def execute_tool_with_state(tool_call: ToolCall, on_state: callable) -> str:
    """Execute a tool with state callbacks for UI updates."""
    args = tool_call.function.arguments_parsed()

    # Notify: starting
    on_state(ToolState(
        status=ToolStatus.RUNNING,
        title=f"Running {tool_call.name}...",
        input=args,
        time_start=time.time(),
    ))

    # Simulate work
    time.sleep(1)

    # Execute
    if tool_call.name == "search":
        result = f"Found 10 results for '{args.get('query', '')}'"
    else:
        result = "Done"

    # Notify: completed
    on_state(ToolState(
        status=ToolStatus.COMPLETED,
        title=f"{tool_call.name} completed",
        input=args,
        output=result,
        time_start=time.time() - 1,
        time_end=time.time(),
    ))

    return result


def render_state(state: ToolState):
    """Render tool state to terminal (like OpenCode UI)."""
    if state.status == ToolStatus.RUNNING:
        print(f"⏳ {state.title}")
    elif state.status == ToolStatus.COMPLETED:
        duration = (state.time_end - state.time_start) if state.time_end else 0
        print(f"✅ {state.title} ({duration:.1f}s)")
        print(f"   Output: {state.output[:50]}...")
    elif state.status == ToolStatus.ERROR:
        print(f"❌ Error: {state.error}")


@tool
def search(query: str) -> str:
    """Search the web.

    Args:
        query: Search query
    """
    return f"Found 10 results for '{query}'"


chat = talu.Chat("Qwen/Qwen3-0.6B")

response = chat.send("Search for Python tutorials", tools=[search], stream=False)

while response.tool_calls:
    for tool_call in response.tool_calls:
        # Execute with live state updates
        result = execute_tool_with_state(tool_call, render_state)

        # Submit tool result and continue generation
        response = response.submit_tool_result(tool_call.id, result)

print(f"\nAssistant: {response}")

"""
Topics covered:
* chat.streaming
* stream.tokens
"""
