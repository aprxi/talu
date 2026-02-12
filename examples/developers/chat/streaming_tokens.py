"""Streaming Token Metadata - Access per-token information during streaming.

Primary API: talu.chat.Token
Scope: Single

This example demonstrates the Token class, which subclasses str to provide
per-token metadata during streaming while remaining fully compatible with
string operations.

Key points:
- Token is a str subclass - works transparently with print(), join(), etc.
- Access metadata via token.id, token.logprob, token.is_special, token.finish_reason
- Enables confidence visualization, debugging, and advanced streaming UIs

Related:
    - examples/developers/chat/streaming.py (basic streaming)
"""

import talu
from talu.chat import Token


# =============================================================================
# Basic Usage - Token works like a string
# =============================================================================

chat = talu.Chat("Qwen/Qwen3-0.6B")

print("=== Casual Usage (Token as string) ===")
for token in chat("What is 2+2?", stream=True):
    # Token is a str subclass - print works directly
    print(token, end="", flush=True)
print("\n")


# =============================================================================
# Power Usage - Access Token Metadata
# =============================================================================

print("=== Power Usage (Token metadata) ===")
chat2 = talu.Chat("Qwen/Qwen3-0.6B")

tokens_collected = []
for token in chat2("Say hello", stream=True):
    tokens_collected.append(token)

    # Access metadata (when available from the engine)
    print(f"Token: {token!r:20} id={token.id:6} logprob={token.logprob}")

print(f"\nTotal tokens: {len(tokens_collected)}")


# =============================================================================
# Confidence Visualization
# =============================================================================

print("\n=== Confidence Visualization ===")

def format_with_confidence(token: Token) -> str:
    """Format token with ANSI colors based on confidence."""
    if token.logprob is None:
        return str(token)

    # Higher logprob = more confident (closer to 0)
    if token.logprob > -0.5:
        return f"\033[92m{token}\033[0m"  # Green - high confidence
    elif token.logprob > -2.0:
        return f"\033[93m{token}\033[0m"  # Yellow - medium
    else:
        return f"\033[91m{token}\033[0m"  # Red - low confidence

chat3 = talu.Chat("Qwen/Qwen3-0.6B")
for token in chat3("What is the capital of France?", stream=True):
    print(format_with_confidence(token), end="", flush=True)
print("\n")


# =============================================================================
# Token Accumulation
# =============================================================================

print("=== Token Accumulation ===")

chat4 = talu.Chat("Qwen/Qwen3-0.6B")
all_tokens = []

for token in chat4("Count to 3", stream=True):
    all_tokens.append(token)

# Since Token subclasses str, join works directly
full_text = "".join(all_tokens)
print(f"Full response: {full_text}")
print(f"Token count: {len(all_tokens)}")


# =============================================================================
# Finish Reason Detection
# =============================================================================

print("\n=== Finish Reason Detection ===")

chat5 = talu.Chat("Qwen/Qwen3-0.6B")
last_token = None

for token in chat5("Hi", stream=True, max_tokens=5):
    last_token = token
    print(token, end="", flush=True)

print()
if last_token and last_token.finish_reason:
    print(f"Generation stopped: {last_token.finish_reason}")
else:
    print("Generation completed normally")


"""
Topics covered:
* streaming.token
* streaming.metadata
* streaming.confidence
* token.logprob

Related:
* examples/developers/chat/streaming.py
* examples/developers/chat/hooks.py (for TTFT measurement)
"""
