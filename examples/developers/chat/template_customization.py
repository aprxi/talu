"""Template Customization - Override chat templates and inject extra context.

Primary API: talu.Chat
Scope: Single

Two powerful features for controlling how messages are formatted:
1. chat_template: Use a custom template instead of the model's built-in one
2. extra_context: Inject additional variables into the template (tools, dates, etc.)

These work at generation time without modifying model files.

Related:
    - examples/basics/10_chat_template.py
"""

import datetime

import talu
from talu.router import GenerationConfig

# =============================================================================
# Extra Context - Inject variables into the model's chat template
# =============================================================================

# Pass tool definitions that some chat templates expect
tools = [
    {"name": "search", "description": "Search the web for information"},
    {"name": "calculator", "description": "Perform mathematical calculations"},
    {"name": "weather", "description": "Get current weather for a location"},
]

config = GenerationConfig(
    extra_context={"tools": tools},
    max_tokens=100,
)

chat = talu.Chat("Qwen/Qwen3-0.6B")
# The model's template can now use {{ tools }} if it supports tool definitions
response = chat("What tools do you have?", config=config)
print("With tools context:", response.text[:100], "...")

# Pass metadata like dates, user info, or feature flags
config = GenerationConfig(
    extra_context={
        "date": "2024-01-15",
        "user_name": "Alice",
        "enable_thinking": True,
    },
    max_tokens=50,
)

# Templates that reference these variables will have access to them
response = chat.send("Hello!", config=config)
print("With metadata:", response.text[:100], "...")


# =============================================================================
# Chat Template Override - Use a completely custom template
# =============================================================================

# Simple custom format
simple_template = """{% for m in messages %}{{ m.role | upper }}: {{ m.content }}
{% endfor %}ASSISTANT:"""

config = GenerationConfig(
    chat_template=simple_template,
    max_tokens=50,
)

chat2 = talu.Chat("Qwen/Qwen3-0.6B")
response = chat2("Hi there!", config=config)
print("Custom format:", response.text[:100], "...")


# =============================================================================
# Combining Both - Custom template with extra context
# =============================================================================

# Template that uses both messages and extra context
tool_template = """{% if tools %}Available tools: {% for t in tools %}{{ t.name }}{% if not loop.last %}, {% endif %}{% endfor %}

{% endif %}{% for m in messages %}{{ m.role }}: {{ m.content }}
{% endfor %}assistant:"""

config = GenerationConfig(
    chat_template=tool_template,
    extra_context={
        "tools": [{"name": "search"}, {"name": "calculate"}],
    },
    max_tokens=50,
)

chat3 = talu.Chat("Qwen/Qwen3-0.6B")
response = chat3("What can you do?", config=config)
print("Combined:", response.text[:100], "...")


# =============================================================================
# Use Cases
# =============================================================================

# 1. Testing different prompt formats without changing model files
formats = [
    "{% for m in messages %}[{{ m.role }}] {{ m.content }}\n{% endfor %}",
    "{% for m in messages %}<{{ m.role }}>\n{{ m.content }}\n</{{ m.role }}>\n{% endfor %}",
]

chat4 = talu.Chat("Qwen/Qwen3-0.6B")
for fmt in formats:
    config = GenerationConfig(chat_template=fmt, max_tokens=30)
    response = chat4.send("Hi", config=config)
    print(f"Format test: {response.text[:50]}...")

# 2. Adding system context dynamically
config = GenerationConfig(
    extra_context={
        "current_date": datetime.date.today().isoformat(),
        "timezone": "UTC",
    },
    max_tokens=50,
)
response = chat4.send("What's today's date?", config=config)
print("Dynamic context:", response.text[:100], "...")

# 3. Feature flags for experimental prompting
config = GenerationConfig(
    extra_context={
        "use_cot": True,  # Chain of thought
        "verbose": False,
    },
    max_tokens=100,
)
# Templates can check: {% if use_cot %}Let me think step by step...{% endif %}


# =============================================================================
# Preview - Debug your template before sending
# =============================================================================

# Use preview_prompt() with config to see exactly what would be sent
debug_template = """{% for m in messages %}[{{ m.role | upper }}] {{ m.content }}
{% endfor %}[ASSISTANT]"""

debug_config = GenerationConfig(chat_template=debug_template)
chat5 = talu.Chat("Qwen/Qwen3-0.6B", system="You are helpful.")

# Preview shows the formatted prompt without actually generating
print("\n=== Template Preview ===")
print(chat5.preview_prompt(config=debug_config))

# Compare with the model's default template
print("\n=== Default Template Preview ===")
print(chat5.preview_prompt())

"""
Topics covered:
* chat.templates
* tokenizer.chat.template
"""
