"""Format messages with a chat template.

This example shows:
- Applying chat templates to messages
- Adding generation prompts
- Token counting for formatted prompts
"""

import os
import sys

import talu
from talu import repository

MODEL_URI = os.environ.get("MODEL_URI", "LiquidAI/LFM2-350M")

if not repository.is_cached(MODEL_URI):
    sys.exit(f"Model '{MODEL_URI}' not found. Run: python examples/python/00_fetch_model.py")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short haiku about rain."},
]

tokenizer = talu.Tokenizer(MODEL_URI)

prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
print(prompt)

prompt_no_gen = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
print(f"With generation prompt: {len(prompt)} chars")
print(f"Without generation prompt: {len(prompt_no_gen)} chars")

# Add another turn and reformat
messages.append({"role": "assistant", "content": "Here is a haiku about rain..."})
messages.append({"role": "user", "content": "Now write one about snow."})
print(tokenizer.apply_chat_template(messages, add_generation_prompt=True))

# Send the formatted prompt to chat
chat = talu.Chat(MODEL_URI)
response = chat(prompt)
print(response)

# Inspect token count for the formatted prompt
print(f"Prompt tokens: {tokenizer.count_tokens(prompt)}")

