"""
Qwen3 MoE - Pure PyTorch implementation.

Supports Qwen3-Coder MoE models (e.g., Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8).

Usage:
    from qwen.qwen3_moe import Qwen3Moe

    model, tokenizer = Qwen3Moe.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")

    ids = torch.tensor([tokenizer.encode("Hello").ids])
    logits = model(ids)

Key features:
- Mixture of Experts with 128 experts, 8 active per token
- QK normalization in attention
- Standard SiLU MLP for each expert (gate_proj * silu(up_proj), no bias)
- Optional top-k probability normalization (norm_topk_prob)
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
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat([-x2, x1], dim=-1)

        return (q * cos) + (rotate(q) * sin), (k * cos) + (rotate(k) * sin)


@fusable(kernel="attention", config=["qk_norm"])
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


class MoEExpertMLP(nn.Module):
    """Single expert MLP with SwiGLU activation."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEExperts(nn.ModuleList):
    """Container for all experts in a MoE layer (indexed directly)."""

    def __init__(self, num_experts, hidden_size, intermediate_size):
        experts = [MoEExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts)]
        super().__init__(experts)
        self.num_experts = num_experts

    def forward(self, hidden_states, selected_experts, routing_weights):
        """
        Args:
            hidden_states: [num_tokens, hidden_size]
            selected_experts: [num_tokens, top_k] - indices of selected experts
            routing_weights: [num_tokens, top_k] - softmax'd routing weights
        """
        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]

        # Accumulate outputs
        final_output = torch.zeros(num_tokens, hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (selected_experts == expert_idx)  # [num_tokens, top_k]
            if not expert_mask.any():
                continue

            # Get token indices and routing weights for this expert
            token_indices, k_indices = expert_mask.nonzero(as_tuple=True)

            # Get the hidden states for these tokens
            expert_input = hidden_states[token_indices]

            # Forward through expert (self[idx] since we inherit from ModuleList)
            expert_output = self[expert_idx](expert_input)

            # Weight by routing weights and accumulate
            weights = routing_weights[token_indices, k_indices].unsqueeze(-1)
            final_output.index_add_(0, token_indices, weights * expert_output)

        return final_output


@fusable(kernel="moe")
class SparseMoeBlock(nn.Module):
    """Sparse Mixture of Experts block for Qwen3 MoE.

    Routes each token to top-k experts based on learned routing weights.
    """

    def __init__(self, hidden_size, moe_intermediate_size, num_experts, num_experts_per_tok, norm_topk_prob=True):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob

        # Router (gate) - no bias, following HF naming
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = MoEExperts(num_experts, hidden_size, moe_intermediate_size)

    def forward(self, hidden_states):
        B, T, D = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, D)

        # Router forward
        router_logits = self.gate(hidden_states_flat)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)

        # Top-k selection
        routing_weights, selected_experts = torch.topk(router_probs, self.num_experts_per_tok, dim=-1)

        # Optional normalization
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        routing_weights = routing_weights.to(hidden_states.dtype)

        # Expert forward
        output = self.experts(hidden_states_flat, selected_experts, routing_weights)

        return output.view(B, T, D)


class Block(nn.Module):
    """Single Qwen3 MoE transformer block."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, moe_intermediate_size,
                 num_experts, num_experts_per_tok, rope_theta, eps, norm_topk_prob=True):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.self_attn = Attention(hidden_size, num_heads, num_kv_heads, head_dim, rope_theta)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.mlp = SparseMoeBlock(hidden_size, moe_intermediate_size, num_experts, num_experts_per_tok, norm_topk_prob)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3Moe(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        num_kv_heads = config["num_key_value_heads"]
        head_dim = config.get("head_dim", hidden_size // num_heads)
        moe_intermediate_size = config["moe_intermediate_size"]
        num_experts = config["num_experts"]
        num_experts_per_tok = config["num_experts_per_tok"]
        rope_theta = config.get("rope_theta", 10000000.0)
        eps = config.get("rms_norm_eps", 1e-6)
        num_layers = config["num_hidden_layers"]
        vocab_size = config["vocab_size"]
        norm_topk_prob = config.get("norm_topk_prob", True)

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            Block(hidden_size, num_heads, num_kv_heads, head_dim, moe_intermediate_size,
                  num_experts, num_experts_per_tok, rope_theta, eps, norm_topk_prob)
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
        return from_pretrained(Qwen3Moe, model_id)
