"""Save and restore a chat session.

This example shows:
- Exporting a chat session to JSON
- Restoring a session from saved data
- Saving just messages for lightweight persistence
"""

import json
import os
import sys

import talu
from talu import repository

MODEL_URI = os.environ.get("MODEL_URI", "LiquidAI/LFM2-350M")

if not repository.is_cached(MODEL_URI):
    sys.exit(f"Model '{MODEL_URI}' not found. Run: python examples/python/00_fetch_model.py")

chat = talu.Chat(MODEL_URI, system="You are helpful.")
response = chat("Summarize this in one sentence: LLMs can generate text.")
print(response)

# Save session to JSON
chat_data = chat.to_dict()
print(f"Saved keys: {list(chat_data.keys())}")
with open("/tmp/talu_08_save_restore_chat.json", "w") as f:
    json.dump(chat_data, f)

# Restore later
with open("/tmp/talu_08_save_restore_chat.json") as f:
    data = json.load(f)

restored = talu.Chat.from_dict(data, model=MODEL_URI)
print(restored("Add one more detail."))

# Save just messages (lightweight)
# chat.to_dict()["messages"] returns the standard list-of-dicts format
with open("/tmp/talu_08_save_restore_messages.json", "w") as f:
    json.dump(chat.to_dict()["messages"], f)

with open("/tmp/talu_08_save_restore_messages.json") as f:
    messages = json.load(f)

restored_messages = talu.Chat.from_dict({"messages": messages}, model=MODEL_URI)
print(restored_messages("Continue from the saved messages."))

