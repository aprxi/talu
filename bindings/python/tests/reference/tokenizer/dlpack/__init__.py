"""
DLPack validation tests for tokenizer components.

Tests zero-copy tensor interchange with PyTorch/JAX:
- TokenArray: true zero-copy via refcounted SharedBuffer
- BatchEncoding: allocated padded tensors (not zero-copy)

Maps to: talu/tokenizer/
"""
