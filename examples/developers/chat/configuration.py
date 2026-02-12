"""Config - Control temperature, stop sequences, logit bias, and other parameters.

Primary API: talu.Chat, talu.GenerationConfig
Scope: Single

Configuration Precedence (highest to lowest):
1. Per-call kwargs (e.g., temperature=0.9) - affects this call only
2. Per-call config object - affects this call only
3. Session config (chat.config) - default for all calls

IMPORTANT: Per-call overrides do NOT mutate session state. They only affect
that single generation call. To permanently change the default, assign to chat.config.

Related:
    - examples/basics/19_generation_settings.py
"""

import talu
from talu.router import GenerationConfig

# =============================================================================
# Session vs Per-Call Configuration
# =============================================================================

# Create chat with session-level defaults
chat = talu.Chat("Qwen/Qwen3-0.6B", config=GenerationConfig(temperature=0.7))
print(f"Session default temperature: {chat.config.temperature}")

# Offline resolution (requires model already cached)
offline_chat = talu.Chat("Qwen/Qwen3-0.6B", offline=True)
print(offline_chat("Summarize the ocean in five words."))

# Per-call override - DOES NOT change chat.config
response = chat("Write a haiku", temperature=0.9, max_tokens=50)
print(response)
print(f"After per-call override, session temperature: {chat.config.temperature}")  # Still 0.7!

# Reusable config object - also does NOT change chat.config
creative = GenerationConfig(temperature=1.2, max_tokens=200)
response = chat("Write a poem", config=creative)
print(response)
print(f"After config object, session temperature: {chat.config.temperature}")  # Still 0.7!

# Create variations using override() - original config unchanged
precise = creative.override(temperature=0.0)  # Same max_tokens, but greedy
print(f"creative.temperature: {creative.temperature}")  # Still 1.2
print(f"precise.temperature: {precise.temperature}")  # 0.0

# Continue with different settings
response = response.append("Make it shorter", temperature=0.5)
print(response)

# To PERMANENTLY change session config, assign directly:
chat.config = GenerationConfig(temperature=0.3)
print(f"After assignment, session temperature: {chat.config.temperature}")  # Now 0.3

# Stop sequences - stop generation when these strings appear
chat2 = talu.Chat("Qwen/Qwen3-0.6B")
config = GenerationConfig(
    stop_sequences=["User:", "```"],  # Stop at user turn or code block end
    max_tokens=100,
)
response = chat2("Write a short story", config=config)
print(response)

# Logit bias - ban or boost specific tokens
# Use negative values to suppress tokens, positive to boost
# -100 effectively bans a token
chat3 = talu.Chat("Qwen/Qwen3-0.6B")
config = GenerationConfig(
    logit_bias={
        # Token IDs vary by model - these are examples
        # 1234: -100,  # Ban token 1234
        # 5678: 5.0,   # Boost token 5678
    },
    max_tokens=50,
)
response = chat3("Hello!", config=config)
print(response)

# Template customization - override the model's chat template
# or inject extra context variables (tools, dates, etc.)
config = GenerationConfig(
    chat_template="{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}",
    extra_context={"tools": [{"name": "search"}], "date": "2024-01-15"},
    max_tokens=50,
)
chat4 = talu.Chat("Qwen/Qwen3-0.6B")
response = chat4("Hi!", config=config)
print(response)

"""
Topics covered:
* generation.config
* generation.parameters
"""
