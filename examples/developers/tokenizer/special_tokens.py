"""Special Tokens - Control BOS, EOS, and other special tokens.

Primary API: talu.Tokenizer
Scope: Single

Special tokens are non-text tokens that models use for structure:
- BOS (Beginning of Sequence): Marks start of input
- EOS (End of Sequence): Signals generation should stop
- PAD: Fills space when batching sequences of different lengths
- UNK: Represents unknown/out-of-vocabulary text

Different models handle special tokens differently. Understanding this
is crucial for correct tokenization.

Related:
    - examples/basics/03_tokenize_text.py
"""

import talu

tokenizer = talu.Tokenizer("Qwen/Qwen3-0.6B")

# =============================================================================
# Inspect special tokens
# =============================================================================

print("Special tokens for this model:")
print(f"  BOS token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
print(f"  EOS tokens: {tokenizer.eos_tokens} (ids: {tokenizer.eos_token_ids})")
print(f"  PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
print(f"  UNK token: {tokenizer.unk_token} (id: {tokenizer.unk_token_id})")

# Note: Some models have multiple EOS tokens
# eos_token_ids is an immutable tuple (ordered, deduplicated)
print(f"\nAll EOS tokens: {tokenizer.eos_tokens}")
print(f"All EOS token IDs: {tokenizer.eos_token_ids}")

# For insertion, use primary_eos_token_id() to get the first EOS
print(f"Primary EOS for insertion: {tokenizer.primary_eos_token_id()}")

# =============================================================================
# Control special token insertion
# =============================================================================

text = "Hello, world!"

# Default: add all special tokens (BOS + EOS)
with_special = tokenizer.encode(text, special_tokens=True)
print(f"\nWith special tokens: {with_special.tolist()}")

# Raw tokenization: no special tokens
without_special = tokenizer.encode(text, special_tokens=False)
print(f"Without special tokens: {without_special.tolist()}")

# Fine-grained control with a set
bos_only = tokenizer.encode(text, special_tokens={"bos"})
eos_only = tokenizer.encode(text, special_tokens={"eos"})
both = tokenizer.encode(text, special_tokens={"bos", "eos"})

print(f"BOS only: {bos_only.tolist()}")
print(f"EOS only: {eos_only.tolist()}")
print(f"Both: {both.tolist()}")

# =============================================================================
# When to use which setting
# =============================================================================

# For generation input: usually add BOS, no EOS
# (the model generates until it produces EOS)
generation_input = tokenizer.encode("Tell me a story", special_tokens={"bos"})

# For training data: add both BOS and EOS
training_sample = tokenizer.encode("The answer is 42", special_tokens=True)

# For embedding/similarity: often no special tokens
embedding_input = tokenizer.encode("semantic search query", special_tokens=False)

# For chat: use chat template instead (handles special tokens automatically)
# See chat_templates.py

# =============================================================================
# Decode with special tokens visible
# =============================================================================

tokens = tokenizer.encode("Hello!", special_tokens=True)

# Default: skip special tokens in output
clean = tokenizer.decode(tokens, skip_special_tokens=True)
print(f"\nDecoded (clean): '{clean}'")

# Show special tokens (useful for debugging)
raw = tokenizer.decode(tokens, skip_special_tokens=False)
print(f"Decoded (raw): '{raw}'")

# =============================================================================
# Special token detection (single source of truth)
# =============================================================================

# Use is_special_id() to check if any token is special
sample_tokens = tokenizer.encode("Hello", special_tokens=True)
print("\nChecking token specialness:")
for token_id in sample_tokens.tolist():
    token_str = tokenizer.id_to_token(token_id) or f"<id:{token_id}>"
    is_special = tokenizer.is_special_id(token_id)
    print(f"  {token_id} ({token_str}): {'special' if is_special else 'regular'}")

# Get all special token IDs as a frozen set (immutable, O(1) lookup)
print(f"\nAll special token IDs: {tokenizer.special_ids}")

# =============================================================================
# Model-specific notes
# =============================================================================

# Qwen models: bos_token is null (not used), multiple EOS tokens
# Llama models: bos_token is <|begin_of_text|>
# BERT models: bos=[CLS], eos=[SEP]

# Check if a model uses BOS before relying on it
if tokenizer.bos_token_id is not None:
    print(f"\nThis model uses BOS token: {tokenizer.bos_token}")
else:
    print("\nThis model does not use a BOS token")

# Check EOS tokens (always a tuple, may be empty)
if tokenizer.eos_token_ids:
    print(f"This model uses EOS tokens: {tokenizer.eos_tokens}")
    print(f"Primary EOS for insertion: {tokenizer.primary_eos_token_id()}")
else:
    print("This model has no EOS tokens defined")

"""
Topics covered:
* tokenizer.encode
* tokenizer.decode
"""
