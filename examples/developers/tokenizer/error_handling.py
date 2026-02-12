"""Error Handling - Handle tokenization errors gracefully.

Primary API: talu.Tokenizer
Scope: Single

The tokenizer raises specific exceptions for different error conditions.
Understanding these helps you build robust applications.

Related:
    - examples/basics/03_tokenize_text.py
"""

import talu
from talu.exceptions import ModelError, TokenizerError, ValidationError

# =============================================================================
# Invalid model path
# =============================================================================

print("Error handling examples:\n")

# ModelError: raised when the model path is invalid
try:
    tokenizer = talu.Tokenizer("/nonexistent/path/to/model")
except (ModelError, TokenizerError) as e:
    print(f"1. Invalid model path:")
    print(f"   {type(e).__name__}: {e}\n")

# =============================================================================
# Invalid parameters
# =============================================================================

tokenizer = talu.Tokenizer("Qwen/Qwen3-0.6B")

# ValidationError: raised for invalid parameter values
try:
    tokenizer.padding_side = "center"  # Only "left" or "right" allowed
except ValidationError as e:
    print(f"2. Invalid padding_side:")
    print(f"   {type(e).__name__}: {e}\n")

try:
    tokenizer.truncation_side = "middle"  # Only "left" or "right" allowed
except ValidationError as e:
    print(f"3. Invalid truncation_side:")
    print(f"   {type(e).__name__}: {e}\n")

# Truncation without max_length
try:
    tokenizer("Hello", truncation=True)  # Missing max_length
except ValidationError as e:
    print(f"4. Truncation without max_length:")
    print(f"   {type(e).__name__}: {e}\n")

# =============================================================================
# ValidationError for wrong argument types
# =============================================================================

try:
    tokenizer.encode(12345)  # text must be str or list[str]
except ValidationError as e:
    print(f"5. Wrong type for encode():")
    print(f"   ValidationError: {e}\n")

try:
    tokenizer.encode("Hello", special_tokens="yes")  # must be bool or set
except ValidationError as e:
    print(f"6. Wrong type for special_tokens:")
    print(f"   ValidationError: {e}\n")

# =============================================================================
# Empty batch issues
# =============================================================================

# Empty batch causes issues with DLPack export
batch = tokenizer.encode([])  # Empty batch is allowed
print(f"7. Empty batch:")
print(f"   len(batch) = {len(batch)}")
print(f"   batch.to_list() = {batch.to_list()}")

try:
    import torch

    torch.from_dlpack(batch)  # Cannot export empty batch
except RuntimeError as e:
    print(f"   DLPack export: RuntimeError: {e}\n")
except ImportError:
    print("   (PyTorch not installed)\n")

# =============================================================================
# Different sequence lengths without padding
# =============================================================================

texts = ["Short", "This is a much longer sentence"]
batch = tokenizer.encode(texts)

try:
    # to_list() with padding=False requires equal lengths
    batch.to_list(padding=False)
except ValidationError as e:
    print(f"8. Unequal lengths without padding:")
    print(f"   {type(e).__name__}: {e}\n")

# Solution: use padding=True (default)
result = batch.to_list(padding=True)
print(f"   With padding: lengths are now equal")

# =============================================================================
# Catching talu errors broadly
# =============================================================================


def safe_tokenize(tokenizer, text):
    """Tokenize with comprehensive error handling."""
    try:
        return tokenizer.encode(text)

    except ValidationError as e:
        # Invalid parameter values
        print(f"Validation error: {e}")
        return None

    except TokenizerError as e:
        # Tokenization failed (rare, usually internal errors)
        print(f"Tokenizer error: {e}")
        return None

    except ModelError as e:
        # Model loading issues
        print(f"Model error: {e}")
        return None

    except talu.TaluError as e:
        # Catch-all for any talu error
        print(f"Talu error ({type(e).__name__}): {e}")
        return None


# Usage
tokens = safe_tokenize(tokenizer, "Hello world")
print(f"9. Safe tokenization result: {tokens.tolist() if tokens else 'None'}")

# =============================================================================
# ValidationError inherits from ValueError
# =============================================================================

# ValidationError inherits from both TaluError and ValueError
# This means you can catch it with either:

try:
    tokenizer.padding_side = "invalid"
except ValueError as e:  # Pythonic way
    print(f"\n10. Caught as ValueError: {e}")

try:
    tokenizer.padding_side = "invalid"
except talu.TaluError as e:  # Talu-specific way
    print(f"    Caught as TaluError: {e}")

# Both work! Use ValueError for Pythonic code, TaluError for talu-specific handling

# =============================================================================
# Error codes
# =============================================================================

# All talu exceptions have an error code attribute
try:
    tokenizer.padding_side = "invalid"
except ValidationError as e:
    print(f"\n11. Error code:")
    print(f"    Exception: {e}")
    print(f"    Code: {e.code}")  # 901 for ValidationError

"""
Topics covered:
* tokenizer.encode
* tokenizer.decode
"""
