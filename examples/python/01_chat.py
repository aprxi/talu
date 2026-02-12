"""Chat basics — start a conversation and add a follow-up.

This example shows:
- One-shot usage via `talu.ask(...)`
- Multi-turn chat sessions via `talu.Chat(...)`
- Saving message history to JSON

Note: chat.send() returns a complete Response (non-streaming).
For token-by-token streaming, see 02_streaming.py.
"""

import json
import os
import sys

import talu
from talu import repository

MODEL_URI = os.environ.get("MODEL_URI", "LiquidAI/LFM2-350M")

if not repository.is_cached(MODEL_URI):
    sys.exit(f"Model '{MODEL_URI}' not found. Run: python examples/python/00_fetch_model.py")

# One-liner — blocks until the full response is ready
print(talu.ask(MODEL_URI, "Hello, world!"))

# chat.send() waits for the full response before returning.
# Use this when you want the complete text at once (batch processing,
# scripts, APIs). For real-time token display, see 02_streaming.py.
chat = talu.Chat(MODEL_URI)
response = chat.send("Give me a short greeting.")
print(response)

# Offline mode (requires model already cached)
offline_chat = talu.Chat(MODEL_URI, offline=True)
response = offline_chat.send("Summarize the ocean in five words.")
print(response)

# System prompt
concise = talu.Chat(MODEL_URI, system="Be concise.")
response = concise.send("Explain gravity in one sentence.")
print(response)

# Multi-turn conversation — append continues the same session
response = chat.send("Say hello.")
response = response.append("Make it friendlier.")
print(response)
response = response.append("Add a fun emoji.")
print(response)

# Inspect the stored items
print(f"Stored items: {len(chat.items)}")

# For simple inspection, export to dicts
transcript = chat.to_dict()["messages"]
print(f"Last role: {transcript[-1]['role']}")

# Save the conversation transcript
with open("/tmp/talu_01_chat_transcript.json", "w") as f:
    json.dump(transcript, f, indent=2)

print("Saved transcript to /tmp/talu_01_chat_transcript.json")

