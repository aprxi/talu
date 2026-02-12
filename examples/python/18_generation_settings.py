"""Adjust basic generation settings.

This example shows:
- Temperature and sampling parameters
- Stop sequences and max tokens
- Using GenerationConfig for reusable settings
"""

import os
import sys

import talu
from talu import repository
from talu.router import GenerationConfig

MODEL_URI = os.environ.get("MODEL_URI", "LiquidAI/LFM2-350M")

if not repository.is_cached(MODEL_URI):
    sys.exit(f"Model '{MODEL_URI}' not found. Run: python examples/python/00_fetch_model.py")

chat = talu.Chat(MODEL_URI, config=GenerationConfig(max_tokens=80))

# Sampling controls
print("Low temperature:")
print(chat("Write a short haiku about the sea.", temperature=0.2))

print("\nHigher temperature:")
print(chat("Write a short haiku about the sea.", temperature=1.0))

print("\nLimit response length:")
print(chat("Give two bullet points about the moon.", max_tokens=40))

print("\nStop sequence example:")
print(chat("Say 'hello' then write END.", stop_sequences=["END"]))

print("\nTop-k example:")
print(chat("List three animals.", top_k=40))

print("\nTop-p example:")
print(chat("List three fruits.", top_p=0.9))

print("\nRepetition penalty example:")
print(chat("Repeat the word 'echo' five times.", repetition_penalty=1.2))

# Save a default config for reuse
chat.config = GenerationConfig(max_tokens=60, temperature=0.7, top_p=0.9)
print("\nConfig default:")
print(chat("Give a short fact about Mars."))

# Override per-call without changing the session config
print(chat("Give a short fact about Jupiter.", temperature=0.2))

# Use a reusable config object for a single call
fast = GenerationConfig(max_tokens=30, temperature=0.3)
print(chat("Give a short fact about Venus.", config=fast))

# Create a variation with override() - original unchanged
slower = fast.override(max_tokens=100)
print(f"fast.max_tokens: {fast.max_tokens}")  # Still 30
print(f"slower.max_tokens: {slower.max_tokens}")  # 100

# Use the same config in a multi-turn session
chat_fast = talu.Chat(MODEL_URI, config=fast)
print(chat_fast("Give a short fact about Saturn."))
print(chat_fast("Now add one more detail."))

