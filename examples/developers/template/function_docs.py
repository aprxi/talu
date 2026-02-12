"""
Function Documentation - Use parse_functions filter to extract tool metadata.

Primary API: talu.PromptTemplate, talu.template.PromptTemplate
Scope: Single

The parse_functions filter extracts function metadata directly from Python source
code. This enables automatic tool descriptions without manual transcription.

Related:
- examples/developers/template/tool_use.py
"""

import talu

# Your actual tool implementations
TOOLS_SOURCE = '''
def search(query: str, limit: int = 10) -> list:
    """Search the web for information.

    Args:
        query: The search query to execute
        limit: Maximum number of results to return
    """
    pass

def calculate(expression: str) -> float:
    """Evaluate a mathematical expression.

    Args:
        expression: The math expression to evaluate (e.g., "2 + 2 * 3")
    """
    pass

def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city.

    Args:
        city: City name (e.g., "London", "New York")
        units: Temperature units - "celsius" or "fahrenheit"
    """
    pass
'''

# Template for tool descriptions
tool_template = talu.PromptTemplate("""You have access to the following tools:

{% for fn in tools | parse_functions %}
## {{ fn.name }}
{{ fn.description }}

Parameters:
{% for p in fn.parameters %}- {{ p.name }} ({{ p.type }}{% if not p.required %}, optional{% endif %}): {{ p.description | default("No description") }}{% if p.default %} Default: {{ p.default }}{% endif %}

{% endfor %}
{% endfor %}
To use a tool, respond with: tool_name(arg1, arg2, ...)

User request: {{ request }}""")

# Generate the prompt
prompt = tool_template(tools=TOOLS_SOURCE, request="What's the weather in Paris?")
print("=== Generated Tool Prompt ===")
print(prompt)
print()

# You can also access parsed function data directly
data_template = talu.PromptTemplate("{{ source | parse_functions | tojson }}")
import json
functions = json.loads(data_template(source=TOOLS_SOURCE))
print("=== Parsed Function Data ===")
print(f"Found {len(functions)} functions:")
for fn in functions:
    print(f"  - {fn['name']}: {len(fn['parameters'])} parameters")
print()

# Simpler listing
list_template = talu.PromptTemplate("""Tools: {% for fn in src | parse_functions %}{{ fn.name }}{% if not loop.last %}, {% endif %}{% endfor %}""")
print("=== Tool List ===")
print(list_template(src=TOOLS_SOURCE))

"""
Topics covered:
* template.render
* template.control.flow
"""
