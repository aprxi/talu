"""Tool calling - Let the model use tools/functions.

Primary API: talu.Chat
Scope: Single
"""

import talu
from talu.chat.tools import tool


@tool
def search(query: str) -> str:
    """Search the web for information.

    Args:
        query: Search query
    """
    return f"Search results for '{query}': Python is a programming language..."


@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations.

    Args:
        expression: Math expression
    """
    # In production, use a safe eval
    return str(eval(expression))


chat = talu.Chat("Qwen/Qwen3-0.6B", system="You have access to search and calculator tools.")

# Tool calling requires stream=False for the continuation loop.
response = chat.send("What is 25 * 4?", tools=[search, calculator], stream=False)

while response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call.name}")
        print(f"Args: {tool_call.function.arguments_parsed()}")

        # Execute the tool
        result = tool_call.execute()
        print(f"Result: {result}")

        # Submit tool result and continue generation
        response = response.submit_tool_result(tool_call.id, result)

print(f"Final: {response}")

"""
Topics covered:
* chat.session
* chat.send
"""
