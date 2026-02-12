"""ML Integration - Use tokenized data with PyTorch, JAX, and NumPy.

Primary API: talu.Tokenizer
Scope: Single

BatchEncoding implements the DLPack protocol for zero-copy transfer to
ML frameworks. This is the most efficient way to get tokenized data
into PyTorch or JAX tensors.

Related:
    - examples/basics/03_tokenize_text.py
"""

import talu

tokenizer = talu.Tokenizer("Qwen/Qwen3-0.6B")

texts = [
    "Hello, world!",
    "How are you today?",
    "This is a longer sentence.",
]

batch = tokenizer.encode(texts)

# =============================================================================
# PyTorch integration (zero-copy via DLPack)
# =============================================================================

try:
    import torch

    # Zero-copy transfer to PyTorch
    input_ids = torch.from_dlpack(batch)
    attention_mask = torch.from_dlpack(batch.attention_mask)

    print("PyTorch tensors:")
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  input_ids dtype: {input_ids.dtype}")
    print(f"  attention_mask shape: {attention_mask.shape}")

    # Dict-style access
    input_ids_alt = torch.from_dlpack(batch["input_ids"])
    mask_alt = torch.from_dlpack(batch["attention_mask"])

    print(f"\nDict-style access works: {input_ids_alt.shape}")

    # Ready for model input
    # model_output = model(input_ids, attention_mask=attention_mask)

except ImportError:
    print("PyTorch not installed, skipping PyTorch examples")

# =============================================================================
# NumPy integration
# =============================================================================

try:
    import numpy as np

    # Zero-copy via DLPack (NumPy 1.23+)
    input_ids_np = np.from_dlpack(batch["input_ids"])
    mask_np = np.from_dlpack(batch["attention_mask"])

    print("\nNumPy arrays (via DLPack):")
    print(f"  input_ids shape: {input_ids_np.shape}")
    print(f"  attention_mask shape: {mask_np.shape}")

except ImportError:
    print("NumPy not installed, skipping NumPy examples")

# =============================================================================
# JAX integration
# =============================================================================

try:
    import jax.numpy as jnp
    from jax.dlpack import from_dlpack as jax_from_dlpack

    # Zero-copy via DLPack
    input_ids_jax = jax_from_dlpack(batch.__dlpack__())

    print("\nJAX array:")
    print(f"  input_ids shape: {input_ids_jax.shape}")

except ImportError:
    print("JAX not installed, skipping JAX examples")

# =============================================================================
# Padding options for ML
# =============================================================================

# Left padding (default) - for decoder/generation models
# Padding goes at the start, real tokens at the end
batch_left = tokenizer.encode(texts)
batch_left.padding_side = "left"

# Right padding - for encoder models (BERT, classification)
# Padding goes at the end, real tokens at the start
batch_right = tokenizer.encode(texts)
batch_right.padding_side = "right"

print("\nPadding comparison:")
result_left = batch_left.to_list()
result_right = batch_right.to_list()
print(f"Left padded first row:  {result_left['input_ids'][0][:8]}...")
print(f"Right padded first row: {result_right['input_ids'][0][:8]}...")

# =============================================================================
# Custom padding
# =============================================================================

# Use a specific pad token ID
batch.pad_token_id = 0  # Override pad token

# Pad to specific length (longer than longest sequence)
result = batch.to_list(max_length=50)
print(f"\nPadded to length 50: {len(result['input_ids'])} sequences, {len(result['input_ids'][0])} tokens each")

# =============================================================================
# Production pattern: DataLoader integration
# =============================================================================


def collate_fn(texts: list[str]) -> dict:
    """Collate function for PyTorch DataLoader."""
    batch = tokenizer.encode(texts)

    try:
        import torch

        return {
            "input_ids": torch.from_dlpack(batch["input_ids"]),
            "attention_mask": torch.from_dlpack(batch["attention_mask"]),
        }
    except ImportError:
        # Fallback to Python lists when PyTorch not available
        return batch.to_list()


# Example usage with DataLoader:
# dataset = ["text1", "text2", ...]
# loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
# for batch in loader:
#     output = model(**batch)

print("\nCollate function result:", collate_fn(texts).keys())

"""
Topics covered:
* tokenizer.batch
* tokenizer.encode
"""
