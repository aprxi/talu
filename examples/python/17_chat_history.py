"""Inspect and manage chat history.

This example shows:
- Accessing stored items
- Restoring chats from saved history
- Clearing and resetting chat state

For more advanced history access with type safety, see:
examples/developers/chat/history_access.py
"""

import json
import os
import sys

import talu
from talu import repository

MODEL_URI = os.environ.get("MODEL_URI", "LiquidAI/LFM2-350M")

if not repository.is_cached(MODEL_URI):
    sys.exit(f"Model '{MODEL_URI}' not found. Run: python examples/python/00_fetch_model.py")

chat = talu.Chat(MODEL_URI, system="Be concise.")
chat("What is 2 + 2?")
chat("What about 3 + 3?")
chat("Give me a quick tip for memorization.")

# Simple inspection: count items
print(f"Items stored: {len(chat.items)}")

# For beginners: export to dicts for easy access
# This is the "easy mode" - works like any list of dicts
history = chat.to_dict()["messages"]
print(f"\nHistory as list: {type(history)} with {len(history)} messages")
print(f"First role: {history[0]['role']}")
print(f"Last role: {history[-1]['role']}")

print("\nLast 2 messages:")
for msg in history[-2:]:
    print(f"- {msg['role']}: {msg['content']}")

# Restore a chat from saved history
restored = talu.Chat.from_dict({"messages": history}, model=MODEL_URI)
print(restored("Continue the conversation in one sentence."))

# Fork the conversation into a new path
forked = restored.fork()
print(forked("Continue with a different tip."))

# Save history to disk
with open("/tmp/talu_18_chat_history_saved.json", "w") as f:
    json.dump(history, f, indent=2)

# Load history back
with open("/tmp/talu_18_chat_history_saved.json") as f:
    loaded = json.load(f)

loaded_chat = talu.Chat.from_dict({"messages": loaded}, model=MODEL_URI)
print(loaded_chat("Summarize the loaded history."))

# Clear user/assistant turns but keep system
chat.clear()
print(f"\nAfter clear: {len(chat.items)} items")

# Reset everything
chat.reset()
print(f"After reset: {len(chat.items)} items")

