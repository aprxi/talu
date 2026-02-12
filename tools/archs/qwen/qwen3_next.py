"""Qwen3-Next hybrid architecture (full attention + linear attention)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.nn import fusable
from lib.utils import from_pretrained


@fusable(kernel="norm", config=["weight_offset"])
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight_offset = 1.0
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * (self.weight + 1.0)


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


@fusable(kernel="attention", config=["qk_norm"])
class FullAttention(nn.Module):
    """Qwen3-Next full-attention block with gated q projection output."""

    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = config.get("head_dim", hidden_size // self.num_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        query_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        # Qwen3-Next full-attn uses a gated q projection: [q | gate].
        self.q_proj = nn.Linear(hidden_size, 2 * query_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(query_dim, hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, config.get("rms_norm_eps", 1e-6))
        self.k_norm = RMSNorm(self.head_dim, config.get("rms_norm_eps", 1e-6))
        self.rotary = RotaryEmbedding(self.head_dim, config.get("rope_theta", 1000000.0))

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        query_dim = self.num_heads * self.head_dim

        q_raw = self.q_proj(x).view(bsz, seq_len, self.num_heads, 2 * self.head_dim)
        q, gate = q_raw.split(self.head_dim, dim=-1)

        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        gate = gate.transpose(1, 2)

        q, k = self.rotary(q, k)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out * torch.sigmoid(gate)
        out = out.transpose(1, 2).reshape(bsz, seq_len, query_dim)
        return self.o_proj(out)


@fusable(kernel="mamba")
class LinearAttention(nn.Module):
    """Qwen3-Next linear attention (gated-delta net) traced as mamba_mixer."""

    def __init__(self, config):
        super().__init__()
        self.d_model = config["hidden_size"]
        self.d_state = config.get("linear_key_head_dim", 128)
        self.d_conv = config.get("linear_conv_kernel_dim", 4)
        self.n_heads = config.get("linear_num_value_heads", config["num_attention_heads"])
        self.d_head = config.get("linear_value_head_dim", self.d_model // self.n_heads)
        self.n_groups = config.get("linear_num_key_heads", self.n_heads)
        self.expand = (self.n_heads * self.d_head) // self.d_model

        self.d_inner = self.n_heads * self.d_head
        self.key_dim = self.n_groups * self.d_state
        self.xbc_len = self.d_inner + 2 * self.key_dim

        # Expected checkpoint names:
        # - in_proj_qkvz.weight
        # - in_proj_ba.weight
        self.in_proj_qkvz = nn.Linear(self.d_model, self.xbc_len + self.d_inner, bias=False)
        self.in_proj_ba = nn.Linear(self.d_model, 2 * self.n_heads, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.xbc_len,
            out_channels=self.xbc_len,
            kernel_size=self.d_conv,
            groups=self.xbc_len,
            padding=self.d_conv - 1,
            bias=False,
        )

        self.A_log = nn.Parameter(torch.zeros(self.n_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.n_heads))
        self.norm = RMSNorm(self.d_head, config.get("rms_norm_eps", 1e-6))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        qkvz = self.in_proj_qkvz(x)
        ba = self.in_proj_ba(x)

        xbc = qkvz[..., : self.xbc_len].transpose(1, 2)
        xbc = self.conv1d(xbc)[..., :seq_len].transpose(1, 2)

        values = xbc[..., 2 * self.key_dim : 2 * self.key_dim + self.d_inner]
        gate = qkvz[..., self.xbc_len :]
        gate = gate + ba[..., :1].repeat_interleave(self.d_inner, dim=-1)

        values = values.view(bsz, seq_len, self.n_heads, self.d_head)
        gate = gate.view(bsz, seq_len, self.n_heads, self.d_head)
        values = self.norm(values) * torch.sigmoid(gate)

        return self.out_proj(values.reshape(bsz, seq_len, self.d_inner))


class MoEExpertMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


@fusable(kernel="moe")
class SparseMoeBlock(nn.Module):
    """Qwen3-Next sparse MoE with a gated shared expert branch."""

    def __init__(self, hidden_size, moe_intermediate_size, num_experts, num_experts_per_tok, norm_topk_prob=True):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [MoEExpertMLP(hidden_size, moe_intermediate_size) for _ in range(num_experts)]
        )

        self.shared_expert = MoEExpertMLP(hidden_size, moe_intermediate_size)
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        bsz, seq_len, dim = hidden_states.shape
        flat = hidden_states.view(-1, dim)

        router_logits = self.gate(flat)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(router_probs, self.num_experts_per_tok, dim=-1)

        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        mixed = torch.zeros_like(flat)
        for expert_idx in range(self.num_experts):
            mask = selected_experts == expert_idx
            if not mask.any():
                continue
            token_idx, k_idx = mask.nonzero(as_tuple=True)
            expert_out = self.experts[expert_idx](flat[token_idx])
            weights = routing_weights[token_idx, k_idx].unsqueeze(-1)
            mixed.index_add_(0, token_idx, weights * expert_out)

        shared = self.shared_expert(flat)
        shared_gate = torch.sigmoid(self.shared_expert_gate(flat))
        return (mixed + shared * shared_gate).view(bsz, seq_len, dim)


def layer_types_from_config(config):
    layer_types = config.get("layer_types")
    if layer_types:
        return layer_types

    interval = config.get("full_attention_interval")
    n_layers = config.get("num_hidden_layers", config.get("n_layers"))
    if not interval or not n_layers:
        raise ValueError("Qwen3-Next requires layer_types or full_attention_interval + num_hidden_layers")

    return [
        "full_attention" if (idx + 1) % interval == 0 else "linear_attention"
        for idx in range(n_layers)
    ]


class HybridBlock(nn.Module):
    """Qwen3-Next hybrid block with heterogeneous layer types."""

    def __init__(self, config, layer_idx):
        super().__init__()
        hidden_size = config["hidden_size"]
        eps = config.get("rms_norm_eps", 1e-6)

        layer_types = layer_types_from_config(config)
        self.layer_type = layer_types[layer_idx]

        self.input_layernorm = RMSNorm(hidden_size, eps)
        if self.layer_type == "full_attention":
            self.self_attn = FullAttention(config)
        elif self.layer_type == "linear_attention":
            self.linear_attn = LinearAttention(config)
        else:
            raise ValueError(f"Unsupported Qwen3-Next layer type: {self.layer_type}")

        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.mlp = SparseMoeBlock(
            hidden_size=hidden_size,
            moe_intermediate_size=config["moe_intermediate_size"],
            num_experts=config["num_experts"],
            num_experts_per_tok=config["num_experts_per_tok"],
            norm_topk_prob=config.get("norm_topk_prob", True),
        )

    def forward(self, x):
        if self.layer_type == "full_attention":
            x = x + self.self_attn(self.input_layernorm(x))
        else:
            x = x + self.linear_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3Next(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"]
        n_layers = config["num_hidden_layers"]

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([HybridBlock(config, i) for i in range(n_layers)])
        self.norm = RMSNorm(hidden_size, config.get("rms_norm_eps", 1e-6))
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
        # Qwen3-Next checkpoints are very large; keep reference loading in bf16
        # to avoid OOM during architecture bring-up and parity debugging.
        return from_pretrained(Qwen3Next, model_id, model_dtype=torch.bfloat16)
