"""
KV Cache utilities using talu.ops.

Pure talu.ops implementation - no torch dependency.
Callers that need torch tensors should use torch.from_dlpack() on results.
"""

from typing import Optional, Tuple
from talu import ops as _ops


# Re-export the native KVCache class
KVCache = _ops.KVCache


def get_past_length(past_key_values: Optional[list]) -> int:
    """Get the sequence length of cached key-value pairs."""
    if past_key_values is None or len(past_key_values) == 0:
        return 0
    # If using native KVCache, check its length
    if hasattr(past_key_values, 'length'):
        return past_key_values.length
    # Legacy format: list of (k, v) tuples
    if past_key_values[0] is None:
        return 0
    return past_key_values[0][0].shape[2]


def update_kv_cache(
    past_key_value: Optional[Tuple],
    key_states,
    value_states,
    sliding_window: Optional[int] = None,
) -> Tuple:
    """
    Update KV cache by concatenating new key/value states.

    Returns OpsTensor objects. Use torch.from_dlpack() if you need torch tensors.
    """
    if past_key_value is None:
        return key_states, value_states

    past_k, past_v = past_key_value

    k = _ops.cat([past_k, key_states], dim=2)
    v = _ops.cat([past_v, value_states], dim=2)

    if sliding_window:
        seq_len = k.shape[2]
        start = max(0, seq_len - sliding_window)
        k = _ops.slice_tensor(k, [slice(None), slice(None), slice(start, seq_len), slice(None)])
        v = _ops.slice_tensor(v, [slice(None), slice(None), slice(start, seq_len), slice(None)])

    return k, v


def repeat_kv(hidden_states, n_rep: int):
    """
    Repeat KV heads for grouped query attention.

    Expands [batch, num_kv_heads, seq, head_dim] to [batch, num_heads, seq, head_dim]
    Returns OpsTensor. Use torch.from_dlpack() if you need torch tensor.

    Note: This creates a copy. For zero-copy, use repeat_kv_view() with 5D attention.
    """
    if n_rep == 1:
        return hidden_states
    return _ops.repeat_interleave(hidden_states, n_rep, dim=1)


def repeat_kv_view(hidden_states, n_rep: int):
    """
    Zero-copy view for GQA - returns unsqueezed tensor for broadcasting.

    Input:  [batch, num_kv_heads, seq, head_dim]
    Output: [batch, num_kv_heads, 1, seq, head_dim]  (5D for broadcast with Q)

    Usage pattern for zero-copy attention:
        # Reshape Q to 5D for broadcast
        Q_5d = Q.reshape(B, H, R, S, D)           # view, no copy

        # Get K/V views (zero-copy)
        K_5d = repeat_kv_view(K, n_rep)           # [B, H, 1, S, D]
        V_5d = repeat_kv_view(V, n_rep)           # [B, H, 1, S, D]

        # Attention with broadcast (K/V broadcast from 1 to R)
        attn = Q_5d @ K_5d.transpose(-2, -1)      # [B, H, R, S, S]
        out_5d = softmax(attn) @ V_5d             # [B, H, R, S, D]

        # Reshape back to 4D (view if contiguous)
        out = out_5d.reshape(B, num_heads, S, D)
    """
    if n_rep == 1:
        return hidden_states
    # Unsqueeze at dim 2: [B, H, S, D] -> [B, H, 1, S, D]
    return _ops.unsqueeze(hidden_states, 2)


def build_causal_mask(seq_length: int):
    """
    Build a causal attention mask [1, 1, seq, seq].

    Returns OpsTensor with -inf above diagonal, 0 on and below.
    Use torch.from_dlpack() if you need torch tensor.
    """
    mask = _ops.causal_mask(seq_length, _ops.DType.FLOAT32)
    return _ops.unsqueeze(_ops.unsqueeze(mask, 0), 0)


def build_sliding_window_mask(seq_length: int, window: int):
    """
    Build a sliding window attention mask [1, 1, seq, seq].

    Returns OpsTensor. Use torch.from_dlpack() if you need torch tensor.
    Note: Currently returns causal mask (sliding window TODO in Zig).
    """
    # TODO: Implement proper sliding window in Zig
    mask = _ops.causal_mask(seq_length, _ops.DType.FLOAT32)
    return _ops.unsqueeze(_ops.unsqueeze(mask, 0), 0)


# RoPE functions - delegate to native ops
def rope_freqs(seq_len: int, head_dim: int, theta: float = 10000.0):
    """Compute RoPE frequencies."""
    return _ops.rope_freqs(seq_len, head_dim, theta=theta)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply rotary position embeddings to query and key tensors.

    Modifies q and k in-place via DLPack.
    """
    _ops.apply_rope(q, k, cos, sin)
    return q, k


# Attention with KV cache
def attention_with_kv_cache(q, k, v, cache: KVCache, layer_idx: int, scale: float = None):
    """Perform attention with KV cache."""
    return _ops.attention_with_kv_cache(q, k, v, cache, layer_idx, scale=scale)


def attention_with_sinks(q, k, v, cache: KVCache, layer_idx: int, sinks=None, sliding_window: int = 0, scale: float = None):
    """Perform attention with KV cache and optional sinks."""
    return _ops.attention_with_sinks(q, k, v, cache, layer_idx, sinks=sinks, sliding_window=sliding_window, scale=scale)
