"""Truncation - Handle long texts that exceed context limits.

Primary API: talu.Tokenizer
Scope: Single

When text is too long for a model's context window, you need to truncate.
The truncation_side setting controls which part of the text is kept.

Related:
    - examples/basics/12_token_budget.py
"""

import talu

tokenizer = talu.Tokenizer("Qwen/Qwen3-0.6B")

# =============================================================================
# Basic truncation
# =============================================================================

long_text = "Word " * 1000  # A very long text
print(f"Original length: {tokenizer.count_tokens(long_text)} tokens")

# Truncate to max_length
truncated = tokenizer.encode(long_text, truncation=True, max_length=100)
print(f"Truncated length: {len(truncated)} tokens")

# Without truncation=True, max_length is ignored
not_truncated = tokenizer.encode(long_text, max_length=100)
print(f"Without truncation flag: {len(not_truncated)} tokens")

# =============================================================================
# Truncation side
# =============================================================================

# Example: numbered sentences to show what gets kept
numbered_text = " ".join([f"Sentence {i}." for i in range(1, 21)])
print(f"\nOriginal: {numbered_text[:60]}...{numbered_text[-40:]}")

# Right truncation (default): keep beginning, cut end
# Good for: documents where the start is most important
tokenizer.truncation_side = "right"
right_trunc = tokenizer.encode(numbered_text, truncation=True, max_length=30)
right_decoded = tokenizer.decode(right_trunc)
print(f"\nRight truncation (keep start): {right_decoded}")

# Left truncation: keep end, cut beginning
# Good for: RAG contexts, chat history (recent messages more important)
tokenizer.truncation_side = "left"
left_trunc = tokenizer.encode(numbered_text, truncation=True, max_length=30)
left_decoded = tokenizer.decode(left_trunc)
print(f"Left truncation (keep end): {left_decoded}")

# Override per-call without changing tokenizer default
tokens = tokenizer.encode(
    numbered_text,
    truncation=True,
    max_length=30,
    truncation_side="right",  # Override for this call
)

# =============================================================================
# Truncation with batches
# =============================================================================

texts = [
    "Short text.",
    "Medium length text with a few more words.",
    "Very long text that definitely needs truncation. " * 10,
]

# All sequences truncated to same max_length
batch = tokenizer.encode(texts, truncation=True, max_length=20)
print(f"\nBatch lengths after truncation: {batch.lengths()}")

# =============================================================================
# Real-world RAG example
# =============================================================================


def prepare_rag_context(documents: list[str], max_tokens: int = 2048) -> str:
    """Prepare RAG context with smart truncation."""
    tokenizer.truncation_side = "right"  # Keep beginning of each doc

    truncated_docs = []
    tokens_per_doc = max_tokens // len(documents)

    for doc in documents:
        tokens = tokenizer.encode(doc, truncation=True, max_length=tokens_per_doc)
        truncated_docs.append(tokenizer.decode(tokens))

    return "\n\n".join(truncated_docs)


# Example usage
docs = [
    "First document with important information. " * 50,
    "Second document about another topic. " * 50,
]
context = prepare_rag_context(docs, max_tokens=100)
print(f"\nRAG context ({tokenizer.count_tokens(context)} tokens):")
print(context[:200] + "...")

# =============================================================================
# Context window management
# =============================================================================


def fits_in_context(text: str, max_context: int = 4096, reserve: int = 512) -> bool:
    """Check if text fits in context window with room for response."""
    available = max_context - reserve
    token_count = tokenizer.count_tokens(text)
    return token_count <= available


def truncate_to_fit(text: str, max_context: int = 4096, reserve: int = 512) -> str:
    """Truncate text to fit in context window."""
    available = max_context - reserve
    tokens = tokenizer.encode(text, truncation=True, max_length=available)
    return tokenizer.decode(tokens)


# Example
long_prompt = "Explain this concept: " + "context " * 5000
if not fits_in_context(long_prompt):
    long_prompt = truncate_to_fit(long_prompt)
    print(f"\nTruncated prompt to {tokenizer.count_tokens(long_prompt)} tokens")

"""
Topics covered:
* tokenizer.context
* tokenizer.truncation
"""
