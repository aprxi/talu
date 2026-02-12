"""Branch conversations to explore alternatives.

This example shows:
- Appending to continue a conversation
- Forking to explore alternative paths
- Independent histories for each branch
"""

import os
import sys

import talu
from talu import repository

MODEL_URI = os.environ.get("MODEL_URI", "LiquidAI/LFM2-350M")

if not repository.is_cached(MODEL_URI):
    sys.exit(f"Model '{MODEL_URI}' not found. Run: python examples/python/00_fetch_model.py")

chat = talu.Chat(MODEL_URI, config=talu.GenerationConfig(max_tokens=80))

# Start a conversation
r1 = chat("Give me a recipe idea")
print(f"Idea: {r1}")

# Linear append - continues on same chat
r2 = r1.append("Make it vegetarian")
print(f"Vegetarian: {r2}")
print(f"  (same chat: {r2.chat is chat})")

# Branch from r1 - auto-forks since chat moved past r1
r3 = r1.append("Make it quick to prepare")
print(f"Quick: {r3}")
print(f"  (forked: {r3.chat is not chat})")

# Each branch has its own history
print(f"\nOriginal chat: {len(chat.items)} items")
print(f"Quick branch: {len(r3.chat.items)} items")

# Continue each branch independently
r2b = r2.append("Add protein")
print(f"\nVegetarian + protein: {r2b}")
print(f"  (same chat: {r2b.chat is chat})")

r3b = r3.append("Under 15 minutes")
print(f"Quick + 15min: {r3b}")
print(f"  (same as r3's chat: {r3b.chat is r3.chat})")

