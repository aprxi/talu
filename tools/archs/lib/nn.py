"""
Fusable decorator and functional API for talu.ops.

The @fusable decorator marks nn.Module classes for Zig kernel dispatch.
The functional API provides DLPack-compatible tensor operations.
"""

import os
from functools import wraps

# Import talu.ops for native operations
try:
    from talu import ops as _ops
except ImportError:
    _ops = None


# =============================================================================
# Fusable Decorator
# =============================================================================

def _use_talu_backend():
    """Check if Talu backend is enabled."""
    val = os.environ.get("TALU_BACKEND", "").lower()
    return val in ("1", "true", "yes")


def fusable(kernel: str, config: list = None):
    """
    Mark a module for Zig kernel dispatch.

    Args:
        kernel: Kernel type ("attention", "mlp", "norm", "block")
        config: Feature flags (e.g., ["qk_norm", "weight_offset"])

    When TALU_BACKEND=1:
        forward() is replaced with a call to the Zig dispatcher.

    When TALU_BACKEND is unset:
        forward() runs as normal PyTorch (no-op decorator).
    """
    config = config or []

    def decorator(cls):
        original_init = cls.__init__
        original_forward = cls.forward

        # Store metadata on class for tracing
        cls._fusable_kernel = kernel
        cls._fusable_config = config

        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._fusable_kernel = kernel
            self._fusable_config = config

        @wraps(original_forward)
        def new_forward(self, *args, **kwargs):
            if _use_talu_backend():
                raise NotImplementedError(
                    f"Zig kernel dispatch for '{kernel}' is not yet implemented.\n"
                    "Unset TALU_BACKEND to use the PyTorch implementation."
                )
            return original_forward(self, *args, **kwargs)

        cls.__init__ = new_init
        cls.forward = new_forward
        return cls

    return decorator


# =============================================================================
# Functional API (using talu.ops)
# =============================================================================
#
# These functions return DLPack-compatible tensors (OpsTensor).
# Callers can convert to torch via: torch.from_dlpack(result)
#


def rms_norm(x, weight, eps: float = 1e-6):
    """RMS Normalization. Returns DLPack tensor."""
    return _ops.rms_norm(x, weight, eps)


def silu(x):
    """SiLU/Swish activation. Returns DLPack tensor."""
    return _ops.silu(x)


def gelu(x):
    """GELU activation. Returns DLPack tensor."""
    return _ops.gelu(x)


def relu(x):
    """ReLU activation. Returns DLPack tensor."""
    return _ops.relu(x)


def tanh(x):
    """Tanh activation. Returns DLPack tensor."""
    return _ops.tanh(x)


def sigmoid(x):
    """Sigmoid activation. Returns DLPack tensor."""
    return _ops.sigmoid(x)


def softmax(x, dim: int = -1):
    """Softmax activation. Returns DLPack tensor."""
    ndim = x.ndim if hasattr(x, 'ndim') else len(x.shape)
    if dim != -1 and dim != ndim - 1:
        raise NotImplementedError(f"softmax only supports dim=-1, got dim={dim}")
    return _ops.softmax(x)


def matmul(a, b):
    """Matrix multiplication. Returns DLPack tensor."""
    return _ops.matmul(a, b)


def split(x, split_sizes, dim: int = -1):
    """Split tensor into chunks. Returns list of DLPack tensors."""
    return _ops.split(x, split_sizes, dim=dim)


def triu(x, diagonal: int = 0):
    """Upper triangular part of tensor. Returns DLPack tensor."""
    return _ops.triu(x, diagonal=diagonal)


def rsqrt(x):
    """Reciprocal square root. Returns DLPack tensor."""
    return _ops.rsqrt(x)


def zeros_like(x):
    """Zeros with same shape. Returns DLPack tensor."""
    return _ops.zeros_like(x)


def topk(x, k: int):
    """Top-k values and indices. Returns (values, indices) DLPack tensors."""
    return _ops.topk(x, k)


def rope_freqs(seq_len: int, head_dim: int, theta: float = 10000.0):
    """Compute RoPE frequencies. Returns (cos, sin) DLPack tensors."""
    return _ops.rope_freqs(seq_len, head_dim, theta=theta)


def apply_rope(q, k, cos, sin):
    """Apply RoPE in-place."""
    _ops.apply_rope(q, k, cos, sin)


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p: float = 0.0, is_causal: bool = False, scale: float = None):
    """Scaled dot-product attention. Returns DLPack tensor."""
    if attn_mask is not None:
        raise NotImplementedError("attn_mask not supported")
    if dropout_p != 0.0:
        raise NotImplementedError("dropout not supported")
    return _ops.scaled_dot_product_attention(query, key, value, is_causal=is_causal, scale=scale)
