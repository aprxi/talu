"""Reuse one model across multiple chats.

This example shows:
- Sharing a Client across multiple Chat sessions
- Independent chat histories with shared model
- One-shot completions via Client
"""

import os
import sys

from talu import Client
from talu import repository
from talu.router import GenerationConfig

MODEL_URI = os.environ.get("MODEL_URI", "LiquidAI/LFM2-350M")

if not repository.is_cached(MODEL_URI):
    sys.exit(f"Model '{MODEL_URI}' not found. Run: python examples/python/00_fetch_model.py")

config = GenerationConfig(max_tokens=80)

with Client(MODEL_URI) as client:
    # Two independent chats sharing the same model
    alice = client.chat(system="You are Alice.", config=config)
    bob = client.chat(system="You are Bob.", config=config)

    print(alice("Say hello."))
    print(bob("Say hello."))

    # One-shot completion
    print(client.ask("Define gravity in one sentence.", max_tokens=80))

    # Each chat keeps its own history
    alice("What is your role?")
    bob("What is your role?")
    print(f"Alice items: {len(alice.items)}")
    print(f"Bob items: {len(bob.items)}")

    # Create a third chat with a different system prompt
    helper = client.chat(system="You are a troubleshooting assistant.", config=config)
    print(helper("List two debugging steps."))

    # Chat histories are independent
    print(f"Helper items: {len(helper.items)}")

    # Reuse the same client for multiple short tasks
    for task in ["Explain caching in one sentence.", "Define concurrency in one sentence."]:
        print(client.ask(task, max_tokens=80))

    # Create a few chats in a loop
    personas = ["planner", "coach"]
    chats = [client.chat(system=f"You are a {persona}.", config=config) for persona in personas]

    for persona, chat in zip(personas, chats):
        print(f"{persona}: {chat('Give one tip for productivity.')}")

