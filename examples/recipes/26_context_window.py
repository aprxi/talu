"""Manage context window with Chat.count_tokens().

This example shows:
- Checking current token usage with chat.count_tokens()
- Pre-checking if a message will fit with chat.count_tokens(message)
- Using chat.max_context_length to get the model's limit
- Implementing a "will it fit?" workflow
"""

import talu

# Create a chat with a system prompt
chat = talu.Chat("Qwen/Qwen3-0.6B", system="You are a helpful assistant.")

# =============================================================================
# Basic token counting
# =============================================================================

# Check the model's context limit
max_context = chat.max_context_length
print(f"Model max context: {max_context:,} tokens")

# Check current token usage (includes system prompt + chat template overhead)
current = chat.count_tokens()
print(f"Current usage: {current} tokens")

# Pre-check: will a message fit?
message = "What is the capital of France?"
tokens_with_message = chat.count_tokens(message)
print(f"Tokens if we add message: {tokens_with_message}")

# The message is NOT added to history - count_tokens() is non-destructive
print(f"Items in chat (unchanged): {len(chat.items)}")


# =============================================================================
# "Will it fit?" workflow
# =============================================================================

print("\n--- Will It Fit? ---")


def will_it_fit(chat, new_message, reserve_for_response=512):
    """Check if a message will fit in the context window."""
    max_ctx = chat.max_context_length
    if max_ctx is None:
        return True  # No limit specified

    tokens_needed = chat.count_tokens(new_message)
    return tokens_needed + reserve_for_response <= max_ctx


# Test with a short message
short_msg = "Hello!"
print(f"Short message fits: {will_it_fit(chat, short_msg)}")

# Test with a very long message
long_msg = "This is a very long document. " * 10000
print(f"Long message fits: {will_it_fit(chat, long_msg)}")


# =============================================================================
# Reserve space for response
# =============================================================================

print("\n--- Budget Planning ---")

# Common pattern: reserve tokens for the model's response
reserve_for_response = 512
effective_limit = max_context - reserve_for_response
current_usage = chat.count_tokens()
available_for_input = effective_limit - current_usage

print(f"Max context:          {max_context:,}")
print(f"Reserved for output:  {reserve_for_response}")
print(f"Current usage:        {current_usage}")
print(f"Available for input:  {available_for_input:,}")


# =============================================================================
# Practical example: Check before sending document
# =============================================================================

print("\n--- Document Check ---")

# Simulate a document
document = "Important context about the project. " * 100

# Check token count
doc_tokens = chat.count_tokens(document)
print(f"Document tokens: {doc_tokens}")

if will_it_fit(chat, document):
    print("Document fits in context window!")
else:
    print("Document too long - need to truncate or summarize")

"""
Topics covered:

* chat.context
* chat.tokens
* budget.management

Related:

* examples/basics/15_token_budget.py - Lower-level tokenizer counting
* examples/developers/tokenizer/context_management.py - Truncation strategies
"""
