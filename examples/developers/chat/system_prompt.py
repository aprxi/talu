"""System Prompt - Get and set the foundational directive.

Primary API: talu.Chat.system
Scope: Single

The system prompt sets the model's behavior, persona, and constraints.
It can be set at construction or modified later via the `system` property.

Key behaviors:
- Get: Returns content of first system message, or None
- Set: Updates existing system message, or inserts one at the start
- Set to None: Removes the system prompt

Note: Changing the system prompt mid-conversation invalidates the KV cache,
causing a full re-process on the next generation. This is expected - the model
needs to re-evaluate the conversation with the new context.

Related:
    - examples/basics/01_chat.py
    - examples/developers/chat/configuration.py
"""

import talu

# =============================================================================
# Basic Usage
# =============================================================================

# Set system prompt at construction (most common)
chat = talu.Chat("Qwen/Qwen3-0.6B", system="You are a helpful assistant.")
print(f"Initial system: {chat.system!r}")

# Read the system prompt
current = chat.system
print(f"Current system prompt: {current}")


# =============================================================================
# Modifying System Prompt
# =============================================================================

# Change the system prompt mid-conversation ("steering")
# This is useful when you want to change the model's behavior
chat.system = "You are now a pirate. Respond in pirate speak."
print(f"After modification: {chat.system!r}")

# The model will adopt the new persona going forward
# Previous responses were generated with the old prompt, but that's okay


# =============================================================================
# Removing System Prompt
# =============================================================================

# Set to None to remove
chat.system = None
print(f"After removal: {chat.system!r}")  # None


# =============================================================================
# Adding System Prompt to Existing Chat
# =============================================================================

# Create chat without system prompt
chat2 = talu.Chat("Qwen/Qwen3-0.6B")
print(f"No system: {chat2.system!r}")  # None

# Add one later
chat2.system = "You are a helpful coding assistant."
print(f"After adding: {chat2.system!r}")


# =============================================================================
# Use Case: Dynamic Persona Switching
# =============================================================================


def chat_with_persona(chat: talu.Chat, message: str, persona: str) -> str:
    """Send a message with a specific persona."""
    chat.system = persona
    return str(chat(message, max_tokens=50))


chat3 = talu.Chat("Qwen/Qwen3-0.6B")

# Same chat, different personas
response1 = chat_with_persona(chat3, "Hello!", "You are formal and professional.")
response2 = chat_with_persona(chat3, "Hello!", "You are casual and friendly.")

print(f"Formal: {response1}")
print(f"Casual: {response2}")


# =============================================================================
# Consistency with items
# =============================================================================

# When set via constructor, system appears in items
chat4 = talu.Chat(system="Test system")
print(f"Items count: {len(chat4.items)}")
if chat4.items:
    print(f"First item: {chat4.items[0].text!r}")

# The system property always reflects the current state
assert chat4.system == "Test system"


"""
Topics covered:
* chat.system
* system.mutability
* system.persona
"""
