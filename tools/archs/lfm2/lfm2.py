"""
LFM2 - Pure PyTorch implementation.

Supports LiquidAI LFM2 (Hybrid Conv + Attention).

Usage:
    from lfm2.lfm2 import LFM2

    model, tokenizer = LFM2.from_pretrained("LiquidAI/LFM2-350M")

    ids = torch.tensor([tokenizer.encode("Hello").ids])
    logits = model(ids)

Key features:
- Hybrid architecture with gated short convolutions and grouped query attention
- Double-gated LIV (Linear, Input-dependent, Variational) convolution blocks
- QK LayerNorm on attention layers
- RoPE positional embeddings for attention layers only

Note: This pure PyTorch implementation is for verification/tracing.
Production inference should use optimized kernels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.nn import fusable
from lib.utils import from_pretrained


@fusable(kernel="norm")
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=1000000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, q, k):
        seq_len = q.shape[2]
        if self.cos_cached is None or seq_len > self.cos_cached.shape[0]:
            t = torch.arange(seq_len, device=q.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.cos_cached = emb.cos().to(q.dtype)
            self.sin_cached = emb.sin().to(q.dtype)

        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        def rotate(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat([-x2, x1], dim=-1)

        return (q * cos) + (rotate(q) * sin), (k * cos) + (rotate(k) * sin)


@fusable(kernel="shortconv")
class ShortConv(nn.Module):
    """
    LFM2 Short Convolution block.

    Implements gated short-range convolution:
        output = out_proj(C * conv(B * x))

    where B, C are input-dependent gates and conv is a causal 1D convolution.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.conv_dim = config.get("conv_dim", self.hidden_size)
        self.conv_dim_out = config.get("conv_dim_out", self.hidden_size)
        self.L_cache = config.get("conv_L_cache", 3)

        # Input projection: hidden_size -> 3 * conv_dim (B, C, x)
        self.in_proj = nn.Linear(
            self.hidden_size, 3 * self.conv_dim, bias=config.get("conv_bias", False)
        )

        # Causal 1D convolution (depthwise)
        self.conv = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.L_cache,
            groups=self.conv_dim,
            padding=self.L_cache - 1,  # Causal padding
            bias=config.get("conv_bias", False),
        )

        # Output projection
        self.out_proj = nn.Linear(
            self.conv_dim_out, self.hidden_size, bias=config.get("conv_bias", False)
        )

    def forward(self, x):
        batch, seqlen, _ = x.shape

        # HF: BCx = self.in_proj(x).transpose(-1, -2)
        # Then B, C, x = BCx.chunk(3, dim=-2)
        BCx = self.in_proj(x).transpose(-1, -2)  # (batch, 3*dim, seq)
        B_gate, C_gate, x_proj = BCx.chunk(3, dim=-2)  # Each is (batch, dim, seq)

        # Apply B gate: B * x
        Bx = B_gate * x_proj  # (batch, dim, seq)

        # Causal convolution (already in correct format for Conv1d)
        conv_out = self.conv(Bx)[..., :seqlen]  # (batch, dim, seq)

        # Apply C gate: C * conv(B * x)
        y = C_gate * conv_out  # (batch, dim, seq)

        # Transpose back and project
        y = y.transpose(-1, -2).contiguous()  # (batch, seq, dim)
        return self.out_proj(y)


@fusable(kernel="attention", config=["qk_norm"])
class Attention(nn.Module):
    """
    LFM2 Attention with QK LayerNorm.

    Standard grouped-query attention with:
    - RMSNorm applied to Q and K after projection
    - RoPE positional embeddings
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        # HF uses out_proj, not o_proj
        self.out_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # QK LayerNorm - HF uses q_layernorm/k_layernorm
        self.q_layernorm = RMSNorm(self.head_dim, config.get("norm_eps", 1e-5))
        self.k_layernorm = RMSNorm(self.head_dim, config.get("norm_eps", 1e-5))

        # RoPE
        self.rotary = RotaryEmbedding(self.head_dim, config.get("rope_theta", 1000000.0))

    def forward(self, x):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        # QK LayerNorm
        q = self.q_layernorm(q)
        k = self.k_layernorm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        q, k = self.rotary(q, k)

        # GQA: expand k, v
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return self.out_proj(out.transpose(1, 2).reshape(B, T, -1))


@fusable(kernel="mlp")
class MLP(nn.Module):
    """
    LFM2 MLP with SwiGLU activation.

    HF uses w1/w2/w3 naming:
    - w1: gate projection (with SiLU)
    - w2: down projection
    - w3: up projection
    """

    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]

        # LFM2 uses block_ff_dim for intermediate size
        intermediate_size = config.get("block_ff_dim", config.get("intermediate_size"))

        if intermediate_size is None:
            # Fallback: compute from multiplier
            multiplier = config.get("block_ffn_dim_multiplier", 1.0)
            intermediate_size = int(hidden_size * 4 * multiplier)

        # With SwiGLU, apply 2/3 factor (3 projections instead of 2)
        if config.get("block_use_swiglu", True):
            intermediate_size = int(intermediate_size * 2 / 3)

        # Align to block_multiple_of
        multiple_of = config.get("block_multiple_of", 256)
        intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)

        # HF naming: w1=gate, w2=down, w3=up
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class HybridBlock(nn.Module):
    """
    LFM2 decoder block - either ShortConv or Attention based on layer index.

    Layer type is determined by full_attn_idxs config:
    - If layer_idx in full_attn_idxs: use Attention
    - Otherwise: use ShortConv

    HF naming:
    - Attention layers use "self_attn"
    - Conv layers use "conv"
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config["hidden_size"]

        # Determine layer type
        full_attn_idxs = config.get("full_attn_idxs", [])
        self.is_attention = layer_idx in full_attn_idxs
        self.layer_type = "attention" if self.is_attention else "conv"

        # Pre-norm
        self.operator_norm = RMSNorm(self.hidden_size, config.get("norm_eps", 1e-5))

        # Mixer: either attention (self_attn) or short conv (conv)
        if self.is_attention:
            self.self_attn = Attention(config)
        else:
            self.conv = ShortConv(config)

        # FFN
        self.ffn_norm = RMSNorm(self.hidden_size, config.get("norm_eps", 1e-5))
        self.feed_forward = MLP(config)

    def forward(self, x):
        # Pre-norm architecture
        if self.is_attention:
            x = x + self.self_attn(self.operator_norm(x))
        else:
            x = x + self.conv(self.operator_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class LFM2(nn.Module):
    """
    LFM2 (LiquidAI Foundation Model 2).

    Hybrid architecture with:
    - 10 double-gated short-range LIV convolution blocks
    - 6 grouped query attention (GQA) blocks

    Layer types determined by full_attn_idxs config parameter.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        num_layers = config["num_hidden_layers"]

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        # Heterogeneous layers
        self.layers = nn.ModuleList(
            [HybridBlock(config, i) for i in range(num_layers)]
        )

        # LFM2 applies embedding_norm AFTER all layers (acts as final norm)
        self.embedding_norm = RMSNorm(self.hidden_size, config.get("norm_eps", 1e-5))

        # lm_head is tied to embed_tokens
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x)

        # embedding_norm is applied AFTER all layers in LFM2
        x = self.embedding_norm(x)
        logits = self.lm_head(x)

        return logits

    @staticmethod
    def from_pretrained(model_id):
        return from_pretrained(LFM2, model_id)
