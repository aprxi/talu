"""Stream tokens to stdout.

This example shows:
- Token-by-token streaming with chat() (the default callable)
- Collecting tokens and timing generation
- Using a callback for inline processing
- Streaming follow-ups via append

Note: chat() streams by default — it returns a StreamingResponse that
yields tokens as the model generates them. This is the recommended mode
for interactive use (CLIs, chat UIs) since users see output immediately.

For complete responses without iteration, see 01_chat.py (chat.send()).
Under the hood, the engine always generates token-by-token; chat.send()
simply collects all tokens before returning.
"""

import os
import sys

import talu
from talu import repository

MODEL_URI = os.environ.get("MODEL_URI", "LiquidAI/LFM2-350M")

if not repository.is_cached(MODEL_URI):
    sys.exit(f"Model '{MODEL_URI}' not found. Run: python examples/python/00_fetch_model.py")

chat = talu.Chat(MODEL_URI)

# chat() returns a StreamingResponse — iterate to get tokens as they arrive
response = chat("Tell me a short joke")
for token in response:
    print(token, end="", flush=True)
print()

# One-liner with callback — on_token fires for each generated token
chat("Tell me another joke", on_token=lambda t: print(t, end="", flush=True))
print()

# append() inherits streaming mode — no need to pass stream=True
response = chat("Why are jokes funny?")
response = response.append("Explain in one sentence.")
for token in response:
    print(token, end="", flush=True)
print()

