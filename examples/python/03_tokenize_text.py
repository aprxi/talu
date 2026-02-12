"""Tokenize text — encode, decode, and inspect tokens.

This example shows:
- Encoding text to tokens and decoding back
- Counting tokens in a prompt
- Batch encoding multiple texts
"""

import os
import sys

import talu
from talu import repository

MODEL_URI = os.environ.get("MODEL_URI", "LiquidAI/LFM2-350M")

if not repository.is_cached(MODEL_URI):
    sys.exit(f"Model '{MODEL_URI}' not found. Run: python examples/python/00_fetch_model.py")

tokenizer = talu.Tokenizer(MODEL_URI)

# Encode text to tokens
tokens = tokenizer.encode("Hello, world!")
print(f"Tokens: {tokens.tolist()}")
print(f"Length: {len(tokens)}")

# Decode tokens back to text
text = tokenizer.decode(tokens)
print(f"Decoded: {text}")

prompt = "Explain quantum computing in simple terms."
count = tokenizer.count_tokens(prompt)
print(f"\nPrompt has {count} tokens")

# See how text is tokenized
pieces = tokenizer.tokenize("Hello, world!")
print(f"\nToken pieces: {pieces}")

# Batch encode multiple texts
texts = [
    "Short text.",
    "Another short example.",
    "A longer sentence that will produce more tokens.",
]
batch = tokenizer.encode(texts)
print(f"\nBatch size: {len(batch)}")
print(f"Lengths: {batch.lengths()}")

# Map tokens back to text spans
text = "Hello world"
tokens = tokenizer.encode(text)
print(f"\nOffsets for: '{text}'")
for offset in tokens.offsets:
    print(f"  {offset} -> '{offset.slice(text)}'")

