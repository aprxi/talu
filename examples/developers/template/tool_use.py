"""
Tool Use - Format function definitions with parameters for tool calling.

Primary API: talu.PromptTemplate, talu.template.PromptTemplate
Scope: Single

Tool calling (function calling) lets LLMs interact with external systems.
The template defines available tools so the model can choose which to invoke.

Related:
- examples/developers/template/function_docs.py
"""

import talu

# Basic tool definitions
tools_prompt = talu.PromptTemplate("""
You have access to these tools:

{% for tool in tools %}
## {{ tool.name }}
{{ tool.description }}
Parameters: {{ tool.params | join(', ') }}

{% endfor %}
Choose the best tool for the user's request. Respond with:
Tool: <name>
Args: <json args>

User: {{ request }}
""")

tools = [
    {"name": "search", "description": "Search the web", "params": ["query"]},
    {"name": "calculate", "description": "Do math", "params": ["expression"]},
    {"name": "weather", "description": "Get weather", "params": ["city"]},
]

print(tools_prompt(tools=tools, request="What's the weather in Tokyo?"))

# Standard function schema
functions_prompt = talu.PromptTemplate("""
# Functions

{% for fn in functions %}
## {{ fn.name }}
{{ fn.description }}

```json
{{ fn.parameters | tojson }}
```

{% endfor %}
Respond with a function call as JSON.

User: {{ query }}
""")

functions = [
    {
        "name": "get_stock_price",
        "description": "Get current stock price",
        "parameters": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
]

print(functions_prompt(functions=functions, query="What's Apple's stock price?"))

# ReAct pattern - Thought/Action/Observation loop
react = talu.PromptTemplate("""
Answer the question using the available tools.

Tools: {{ tools | join(', ') }}

{% for step in steps %}
Thought: {{ step.thought }}
Action: {{ step.action }}
Observation: {{ step.observation }}

{% endfor %}
Question: {{ question }}
Thought:""")

steps = [
    {"thought": "I need to search for this", "action": "search('python creator')", "observation": "Guido van Rossum"},
]

print(react(tools=["search", "calculate"], steps=steps, question="Who created Python and when?"))

"""
Topics covered:
* template.render
* chat.templates
"""
