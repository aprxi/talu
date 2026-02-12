"""Run multiple prompts and save results.

This example shows:
- Processing a batch of prompts
- Saving results to JSON and CSV
- Simple sequential chat batch processing
"""

import csv
import json
import os
import sys

import talu
from talu import repository

MODEL_URI = os.environ.get("MODEL_URI", "LiquidAI/LFM2-350M")

if not repository.is_cached(MODEL_URI):
    sys.exit(f"Model '{MODEL_URI}' not found. Run: python examples/python/00_fetch_model.py")

chat = talu.Chat(MODEL_URI, system="Be concise.")

prompts = [
    "Define gravity in one sentence.",
    "Give a one-line summary of photosynthesis.",
    "What is a database?",
    "Explain caching in one sentence.",
]

results = []
for prompt in prompts:
    response = chat(prompt)
    results.append({"prompt": prompt, "response": str(response)})
    print(f"Q: {prompt}")
    print(f"A: {response}\n")

with open("/tmp/talu_11_batch_chat_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved results to /tmp/talu_11_batch_chat_results.json")

with open("/tmp/talu_11_batch_chat_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["prompt", "response"])
    writer.writeheader()
    writer.writerows(results)

print("Saved results to /tmp/talu_11_batch_chat_results.csv")

