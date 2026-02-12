"""History Access - Inspect conversation items.

Primary API: talu.Chat.items
Scope: Single

History is accessed via `chat.items`, which provides a read-only view of
the conversation as typed objects (MessageItem, FunctionCallItem, etc.).
This is the "Responses" API view.

To get the legacy list-of-dicts format (OpenAI compatible), use `chat.to_dict()['messages']`.
"""

import talu
from talu.types import MessageItem, FunctionCallItem

chat = talu.Chat("Qwen/Qwen3-0.6B", system="You are helpful.")
response = chat("Hello!")
response = response.append("How are you?")

# Access items via chat.items
items = chat.items
print(f"System: {items.system}")
print(f"Total: {len(items)} items")

# Last item (usually the assistant response)
last = items.last
if isinstance(last, MessageItem):
    print(f"Last: {last.role.name} - {last.text[:20]}...")

# Iterate over items (Typed objects)
print("\nConversation History:")
for item in items:
    if isinstance(item, MessageItem):
        # Access typed fields: role (enum), text (string)
        print(f"  [{item.role.name}]: {item.text[:30]}...")
    elif isinstance(item, FunctionCallItem):
        print(f"  [TOOL CALL]: {item.name}({item.arguments})")

# Access by index
first = items[0]
if isinstance(first, MessageItem):
    print(f"\nFirst message: {first.text}")

# Slice access
print(f"\nItems 1 and 2: {items[1:3]}")

# Export to legacy format (list of dicts) if needed
legacy_history = chat.to_dict()["messages"]
print(f"\nExported legacy count: {len(legacy_history)}")

# Clear conversation (keeps system prompt)
chat.clear()
print(f"\nAfter clear: {len(items)} items")

# Reset everything (including system prompt)
chat.reset()
print(f"After reset: {len(items)} items")

# =============================================================================
# Appending items to conversation
# =============================================================================

# Method 1: String-based append (simple)
chat.append("user", "Hello!")
chat.append("assistant", "Hi there!")
print(f"\nAfter string append: {len(items)} items")

# Method 2: Object-based append (for advanced use cases)
# Use MessageItem.create() for ergonomic construction
user_item = MessageItem.create("user", "How are you?")
chat.append(user_item)

# With MessageRole enum
from talu.types import MessageRole
assistant_item = MessageItem.create(MessageRole.ASSISTANT, "I'm doing great!")
chat.append(assistant_item)
print(f"After object append: {len(items)} items")

# Object-based append is useful when:
# - Working with items from another chat (e.g., merging conversations)
# - Programmatically constructing messages with typed fields
# - Preserving item metadata from external sources

"""
Topics covered:
* chat.history
* chat.items
* chat.messages
"""
