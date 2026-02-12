"""
Gemma3 - Pure PyTorch implementation.

Usage:
    from models.gemma.gemma3 import Gemma3

    model, tokenizer = Gemma3.from_pretrained("google/gemma-3-2b")

    ids = torch.tensor([tokenizer.encode("Hello").ids])
    logits = model(ids)

Key differences from LLaMA:
- Weight offset RMSNorm: output * (1.0 + weight) instead of output * weight
- 4 norms per block (pre/post attention AND pre/post feedforward)
- GELU activation instead of SiLU
- QK normalization (like Qwen)
- Embedding scaled by sqrt(hidden_size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.nn import fusable
from lib.utils import from_pretrained


@fusable(kernel="norm", config=["weight_offset"])
class RMSNorm(nn.Module):
    """
    RMSNorm with weight offset: output * (1 + weight).

    Gemma models initialize weights to zeros and add 1.
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight_offset = 1.0
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * (1.0 + self.weight)


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


@fusable(kernel="attention", config=["qk_norm", "weight_offset_norm"])
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

        # QK normalization with weight offset
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

        self.rotary = RotaryEmbedding(head_dim, rope_theta)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = self.rotary(q, k)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


@fusable(kernel="mlp", config=["gelu"])
class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        # Gemma uses GELU instead of SiLU
        return self.down_proj(F.gelu(self.gate_proj(x), approximate='tanh') * self.up_proj(x))


class Block(nn.Module):
    """Gemma block with 4 norms (pre/post attention and pre/post MLP)."""
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, rope_theta, eps):
        super().__init__()
        # 4 norms instead of 2
        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size, eps)
        self.post_feedforward_layernorm = RMSNorm(hidden_size, eps)

        self.self_attn = Attention(hidden_size, num_heads, num_kv_heads, head_dim, rope_theta)
        self.mlp = MLP(hidden_size, intermediate_size)

    def forward(self, x):
        # Attention with pre and post norm
        h = self.input_layernorm(x)
        h = self.self_attn(h)
        h = self.post_attention_layernorm(h)
        x = x + h

        # MLP with pre and post norm
        h = self.pre_feedforward_layernorm(x)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        x = x + h

        return x


class Gemma3(nn.Module):
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

        self.hidden_size = hidden_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            Block(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, rope_theta, eps)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if config.get("tie_word_embeddings", True):
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        # Gemma scales embeddings by sqrt(hidden_size)
        x = x * (self.hidden_size ** 0.5)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))

    @staticmethod
    def from_pretrained(model_id):
        return from_pretrained(Gemma3, model_id)
