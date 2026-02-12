"""Batch Encoding - Tokenize multiple texts efficiently.

Primary API: talu.Tokenizer
Scope: Single

When processing multiple texts, batch encoding is more efficient than
encoding one at a time. The result is a BatchEncoding object that provides
efficient access to all sequences.

Related:
    - examples/basics/03_tokenize_text.py
"""

import talu

tokenizer = talu.Tokenizer("Qwen/Qwen3-0.6B")

# =============================================================================
# Encode multiple texts at once
# =============================================================================

texts = [
    "Hello, world!",
    "How are you today?",
    "This is a longer sentence with more tokens.",
]

batch = tokenizer.encode(texts)

print(f"Batch size: {len(batch)}")
print(f"Total tokens: {batch.total_tokens}")
print(f"Sequence lengths: {batch.lengths()}")
print(f"Max length: {batch.max_length()}")

# =============================================================================
# Access individual sequences
# =============================================================================

# Index into the batch like a list
first = batch[0]
print(f"\nFirst sequence: {first.tolist()}")
print(f"First length: {len(first)}")

# Iterate over sequences
for i, seq in enumerate(batch):
    print(f"Sequence {i}: {len(seq)} tokens")

# =============================================================================
# Convert to padded format
# =============================================================================

# to_list() returns padded Python lists (useful for debugging)
result = batch.to_list()
print(f"\nPadded input_ids:")
for row in result["input_ids"]:
    print(f"  {row}")

print(f"\nAttention mask:")
for row in result["attention_mask"]:
    print(f"  {row}")

# =============================================================================
# Padding options
# =============================================================================

# Left padding (default) - tokens at end, padding at start
# Good for: generation models (Llama, Qwen, GPT)
left_padded = batch.to_list(padding_side="left")
print(f"\nLeft padded: {left_padded['input_ids'][0][:5]}...")

# Right padding - tokens at start, padding at end
# Good for: encoder models (BERT, classification)
right_padded = batch.to_list(padding_side="right")
print(f"Right padded: {right_padded['input_ids'][0][:5]}...")

# Set padding side on the tokenizer (affects all batches)
tokenizer.padding_side = "right"
new_batch = tokenizer.encode(texts)
print(f"Tokenizer padding_side: {tokenizer.padding_side}")

# =============================================================================
# Callable interface
# =============================================================================

# The tokenizer is callable and returns a batch-like object
output = tokenizer(texts)
print(f"\nCallable interface: {len(output)} sequences")

# Dict-like access to tensors
print(f"Keys: {output.keys()}")
print(f"Has input_ids: {'input_ids' in output}")
print(f"Has attention_mask: {'attention_mask' in output}")

"""
Topics covered:
* tokenizer.batch
* tokenizer.encode
"""
