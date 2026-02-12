"""
Granite3 - Pure PyTorch implementation.

Usage:
    from models.granite.granite3 import Granite3

    model, tokenizer = Granite3.from_pretrained("ibm-granite/granite-3.3-2b-instruct")

    ids = torch.tensor([tokenizer.encode("Hello").ids])
    logits = model(ids)

Supports: IBM Granite models.

Key features:
- LLaMA-style architecture with RMSNorm, RoPE, GQA, SwiGLU
- Residual scaling multipliers (read from config, applied during inference)
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
    def __init__(self, dim, base=10000.0):
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
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat([-x2, x1], dim=-1)

        return (q * cos) + (rotate(q) * sin), (k * cos) + (rotate(k) * sin)


@fusable(kernel="attention")
class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, rope_theta):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.rotary = RotaryEmbedding(head_dim, rope_theta)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = self.rotary(q, k)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


@fusable(kernel="mlp")
class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size,
                 rope_theta, eps, residual_multiplier=1.0):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.self_attn = Attention(hidden_size, num_heads, num_kv_heads, head_dim, rope_theta)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.mlp = MLP(hidden_size, intermediate_size)
        self.residual_multiplier = residual_multiplier

    def forward(self, x):
        # Residual with optional multiplier (Granite-specific)
        h = self.self_attn(self.input_layernorm(x))
        if self.residual_multiplier != 1.0:
            h = h * self.residual_multiplier
        x = x + h

        h = self.mlp(self.post_attention_layernorm(x))
        if self.residual_multiplier != 1.0:
            h = h * self.residual_multiplier
        x = x + h
        return x


class Granite3(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        num_kv_heads = config.get("num_key_value_heads", num_heads)
        head_dim = config.get("head_dim", hidden_size // num_heads)
        intermediate_size = config["intermediate_size"]
        rope_theta = config.get("rope_theta", 10000.0)
        eps = config.get("rms_norm_eps", 1e-6)
        num_layers = config["num_hidden_layers"]
        vocab_size = config["vocab_size"]

        # Granite-specific: residual multiplier
        residual_multiplier = config.get("residual_multiplier", 1.0)
        embedding_multiplier = config.get("embedding_multiplier", 1.0)

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embedding_multiplier = embedding_multiplier
        self.layers = nn.ModuleList([
            Block(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size,
                  rope_theta, eps, residual_multiplier)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if config.get("tie_word_embeddings", True):
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        if self.embedding_multiplier != 1.0:
            x = x * self.embedding_multiplier
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))

    @staticmethod
    def from_pretrained(model_id):
        return from_pretrained(Granite3, model_id)
