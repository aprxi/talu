"""
MFP4Linear - Linear layer using talu.ops.

Uses SIMD-optimized talu.ops.linear (faster than torch.nn.functional.linear).
Supports both numpy and torch inputs/outputs via DLPack.
"""

import numpy as np
from talu import ops as tk_ops
from .module import Module, Parameter


def _is_torch_tensor(x):
    """Check if x is a torch tensor without importing torch."""
    return hasattr(x, 'is_cuda')


class MFP4Linear(Module):
    """Linear layer using talu.ops.linear (SIMD-optimized, zero-copy via DLPack)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter((out_features, in_features))
        if bias:
            self.bias = Parameter((out_features,))
        else:
            self.bias = None

    def forward(self, x):
        """Forward pass. Returns same type as input (torch.Tensor or np.ndarray)."""
        weight = self.weight.get()
        bias = self.bias.get() if self.bias is not None else None

        # For torch bf16 weights, use torch.nn.functional.linear directly
        if _is_torch_tensor(weight):
            import torch
            import torch.nn.functional as F
            return F.linear(x, weight, bias)

        # For numpy weights, use talu ops
        result = tk_ops.linear(x, weight, bias)

        # Return same type as input
        if _is_torch_tensor(x):
            import torch
            return torch.from_dlpack(result)
        else:
            return np.from_dlpack(result)
