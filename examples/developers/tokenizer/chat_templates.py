"""Chat Templates - Format conversations for chat models.

Primary API: talu.Tokenizer
Scope: Single

Chat models expect conversations in a specific format with special tokens.
Each model has its own chat template (ChatML, Llama, Vicuna, etc.).
The tokenizer's apply_chat_template() handles this automatically.

Related:
    - examples/basics/10_chat_template.py
"""

import talu

tokenizer = talu.Tokenizer("Qwen/Qwen3-0.6B")

# =============================================================================
# Basic chat formatting
# =============================================================================

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
]

# Format the conversation
formatted = tokenizer.apply_chat_template(messages)
print("Formatted conversation:")
print(formatted)
print()

# =============================================================================
# Multi-turn conversations
# =============================================================================

conversation = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."},
    {"role": "user", "content": "And what is that times 3?"},
]

formatted = tokenizer.apply_chat_template(conversation)
print("Multi-turn conversation:")
print(formatted)
print()

# =============================================================================
# Control generation prompt
# =============================================================================

# add_generation_prompt=True (default): adds the assistant turn marker
# This is what you want when sending to a model for generation
with_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
print("With generation prompt:")
print(repr(with_prompt[-50:]))  # Show the end

# add_generation_prompt=False: no assistant marker
# Use this for training data where you have the full conversation
without_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
print("\nWithout generation prompt:")
print(repr(without_prompt[-50:]))

# =============================================================================
# Direct tokenization
# =============================================================================

# Get tokens directly instead of string
tokens = tokenizer.apply_chat_template(messages, tokenize=True)
print(f"\nDirect tokenization: {len(tokens)} tokens")
print(f"First 10 tokens: {tokens[:10].tolist()}")

# =============================================================================
# Example: Build a chat prompt manually vs using template
# =============================================================================

# Manual (error-prone, model-specific):
manual_prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
"""

# Using apply_chat_template (correct, portable):
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]
auto_prompt = tokenizer.apply_chat_template(messages)

# The template handles model-specific formatting automatically
print("\nManual vs Template comparison:")
print(f"Manual: {len(manual_prompt)} chars")
print(f"Template: {len(auto_prompt)} chars")

# =============================================================================
# When to use what
# =============================================================================

# Use apply_chat_template when:
# - You have a conversation (list of messages)
# - You want portable code that works with different models
# - You're doing chat-style generation

# Use encode() directly when:
# - You have raw text, not a conversation
# - You're doing embeddings or similarity
# - You need fine-grained control over tokenization

# Use Chat (from talu.Chat) when:
# - You want high-level chat with automatic history
# - You don't need direct token access
# See examples/developers/chat/ for more details

"""
Topics covered:
* tokenizer.chat.template
* chat.templates
"""
