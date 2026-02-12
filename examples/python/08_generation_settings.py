"""Control generation behavior with temperature and limits.

This example shows:
- Setting default config at Chat creation
- Overriding parameters per call with kwargs
- Using temperature for randomness control
- Limiting output length with max_tokens
"""

import os
import sys

import talu
from talu import repository

MODEL_URI = os.environ.get("MODEL_URI", "LiquidAI/LFM2-350M")

if not repository.is_cached(MODEL_URI):
    sys.exit(f"Model '{MODEL_URI}' not found. Run: python examples/python/00_fetch_model.py")

# Create a Chat with default config
config = talu.GenerationConfig(temperature=0.7, max_tokens=100)
chat = talu.Chat(MODEL_URI, config=config)

# Use default config
print("Default (temp=0.7, max=100):")
print(chat("Tell me a short joke."))

# Override for this call only
print("\nMore focused (temp=0.1):")
print(chat("What is 2+2?", temperature=0.1))

print("\nMore creative (temp=1.5):")
print(chat("What is a database?", temperature=1.5))

# Limit response length
print("\nShort response (max_tokens=50):")
print(chat("Explain gravity", max_tokens=50))

# Longer response
print("\nLonger response (max_tokens=150):")
print(chat("Explain gravity", max_tokens=150))

