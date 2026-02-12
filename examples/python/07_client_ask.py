"""Use Client for one-shot tasks and shared models.

This example shows:
- One-shot completions via client.ask()
- Sharing models across multiple chats
- When to use Client vs Chat
"""

import os
import sys

from talu import Client
from talu import repository
from talu.router import GenerationConfig

MODEL_URI = os.environ.get("MODEL_URI", "LiquidAI/LFM2-350M")

if not repository.is_cached(MODEL_URI):
    sys.exit(f"Model '{MODEL_URI}' not found. Run: python examples/python/00_fetch_model.py")

with Client(MODEL_URI) as client:
    # One-shot completions (no history)
    text = client.ask("Explain what a database is in one sentence.", max_tokens=80)
    print(text)

    more = client.ask("Now give one real-world example.", max_tokens=80)
    print(more)

    # Multiple one-shot prompts
    for prompt in ["Define recursion.", "What is a REST API?"]:
        print(f"\nQ: {prompt}")
        print(client.ask(prompt, max_tokens=80))

    # Create a chat from the same client (shares the model)
    chat = client.chat(system="Be concise.", config=GenerationConfig(max_tokens=80))
    response = chat("Give a one-line summary of photosynthesis.")
    print(f"\nChat response: {response}")
    print(f"Chat items stored: {len(chat.items)}")

    response = response.append("Make it even shorter.")
    print(response)

    # Another independent chat
    chat2 = client.chat(system="Answer like a tutor.", config=GenerationConfig(max_tokens=80))
    print(chat2("Explain Newton's first law."))
    print(f"Chat2 items stored: {len(chat2.items)}")

    # Configure a chat with GenerationConfig
    config = GenerationConfig(max_tokens=80, temperature=0.5)
    chat3 = client.chat(system="Be practical.", config=config)
    print(chat3("Give two practical tips for learning Python."))

    # Embeddings from the same client
    embedding = client.embed("Hello, embeddings!")
    print(f"\nEmbedding dimension: {len(embedding)}")
    print(f"Model dimension: {client.embedding_dim()}")

    # Batch embeddings for multiple texts
    texts = ["cats and dogs", "birds and fish", "apples and oranges"]
    batch = client.embed_batch(texts)
    print(f"Batch embeddings: {len(batch)} items")

# When to use Client vs Chat (summary)
print("\nUse Client for one-shot tasks and shared models.")
print("Use Chat when you need message history and multi-turn context.")

