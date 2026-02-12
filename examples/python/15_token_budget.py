"""Manage a token budget with truncation.

This example shows:
- Counting tokens for context management
- Truncating text to fit token limits
- Splitting budget across documents
"""

import os
import sys

import talu
from talu import repository

MODEL_URI = os.environ.get("MODEL_URI", "LiquidAI/LFM2-350M")

if not repository.is_cached(MODEL_URI):
    sys.exit(f"Model '{MODEL_URI}' not found. Run: python examples/python/00_fetch_model.py")

tokenizer = talu.Tokenizer(MODEL_URI)

max_context = 4096
reserve_for_answer = 256
available = max_context - reserve_for_answer


def truncate_to_fit(text: str, limit: int, side: str = "right") -> str:
    """Truncate text to fit a token limit."""
    tokens = tokenizer.encode(text, truncation=True, max_length=limit, truncation_side=side)
    return tokenizer.decode(tokens)


# Example: one long block of context
long_text = "Context " * 2000
count = tokenizer.count_tokens(long_text)
print(f"Original tokens: {count}")
print(f"Available tokens: {available}")

if count > available:
    long_text = truncate_to_fit(long_text, available)
    print(f"Truncated tokens: {tokenizer.count_tokens(long_text)}")
else:
    print("No truncation needed")


# Example: split budget across multiple documents
print("\nSplit a budget across documents")

docs = [
    "Document A: " + ("alpha " * 400),
    "Document B: " + ("beta " * 400),
    "Document C: " + ("gamma " * 400),
]

per_doc = available // len(docs)
truncated_docs = [truncate_to_fit(doc, per_doc, side="right") for doc in docs]

context = "\n\n".join(truncated_docs)
print(f"Context tokens: {tokenizer.count_tokens(context)}")


# Example: reserve space for a question
print("\nReserve tokens for a question")

question = "What are the key differences between A and B?"
question_tokens = tokenizer.count_tokens(question)
context_budget = available - question_tokens

context = "Facts: " + ("info " * 1200)
context = truncate_to_fit(context, max(0, context_budget), side="left")

final_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
print(f"Final prompt tokens: {tokenizer.count_tokens(final_prompt)}")

# Use the trimmed prompt with chat
chat = talu.Chat(MODEL_URI, config=talu.GenerationConfig(max_tokens=80))
response = chat(final_prompt)
print(response)

# Keep the most recent context (left truncation)
recent_history = " ".join([f"Turn {i}: info" for i in range(1, 200)])
recent = truncate_to_fit(recent_history, limit=200, side="left")
print(f"\nRecent context tokens: {tokenizer.count_tokens(recent)}")

# Combine recent history with a new question
prompt = f"Recent:\n{recent}\n\nQuestion: Summarize the last turns.\nAnswer:"
print(chat(prompt))

