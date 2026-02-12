"""
Phi4 - Pure PyTorch implementation.

Usage:
    from models.phi.phi4 import Phi4

    model, tokenizer = Phi4.from_pretrained("microsoft/Phi-4-mini-instruct")

    ids = torch.tensor([tokenizer.encode("Hello").ids])
    logits = model(ids)

Backend switching:
    # PyTorch (default)
    python -m talu.tests.models.test_phi4 "Hello"

    # Zig (fast)
    TALU_BACKEND=1 python -m talu.tests.models.test_phi4 "Hello"

Key differences from Qwen3:
    - Fused QKV projection (qkv_proj instead of separate q/k/v)
    - Fused gate_up projection
    - Partial rotary (50% of head_dim)
    - No QK normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.nn import fusable
from lib.utils import from_pretrained


@fusable(kernel="norm")
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, rotary_dim, base=10000.0):
        super().__init__()
        self.dim = dim
        self.rotary_dim = rotary_dim  # Partial rotary: only rotate this many dims
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
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
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat([-x2, x1], dim=-1)

        # Partial rotary: only rotate first rotary_dim dimensions
        q_rot = q[..., :self.rotary_dim]
        q_pass = q[..., self.rotary_dim:]
        k_rot = k[..., :self.rotary_dim]
        k_pass = k[..., self.rotary_dim:]

        q_rot = (q_rot * cos) + (rotate(q_rot) * sin)
        k_rot = (k_rot * cos) + (rotate(k_rot) * sin)

        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


@fusable(kernel="attention", config=["fused_qkv"])
class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, rope_theta, partial_rotary_factor):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        # Fused QKV projection
        self.qkv_proj = nn.Linear(hidden_size, q_dim + 2 * kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, hidden_size, bias=False)

        # Partial rotary embedding
        rotary_dim = int(head_dim * partial_rotary_factor)
        self.rotary = RotaryEmbedding(head_dim, rotary_dim, rope_theta)

    def forward(self, x):
        B, T, _ = x.shape
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        # Fused QKV projection + split
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, [q_dim, kv_dim, kv_dim], dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary(q, k)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


@fusable(kernel="mlp", config=["fused_gate_up"])
class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.intermediate_size = intermediate_size
        # Fused gate_up projection
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        # Use static split size for tracing
        gate, up = torch.split(gate_up, [self.intermediate_size, self.intermediate_size], dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, rope_theta, partial_rotary_factor, eps):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.self_attn = Attention(hidden_size, num_heads, num_kv_heads, head_dim, rope_theta, partial_rotary_factor)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.mlp = MLP(hidden_size, intermediate_size)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Phi4(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        num_kv_heads = config.get("num_key_value_heads", num_heads)
        head_dim = config.get("head_dim", hidden_size // num_heads)
        intermediate_size = config["intermediate_size"]
        rope_theta = config.get("rope_theta", 10000.0)
        partial_rotary_factor = config.get("partial_rotary_factor", 0.5)
        eps = config.get("rms_norm_eps", 1e-5)
        num_layers = config["num_hidden_layers"]
        vocab_size = config["vocab_size"]

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            Block(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, rope_theta, partial_rotary_factor, eps)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if config.get("tie_word_embeddings", False):
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))

    @staticmethod
    def from_pretrained(model_id):
        return from_pretrained(Phi4, model_id)
