"""
GPT-OSS - Pure PyTorch implementation.

Supports OpenAI gpt-oss-20b (MoE with MXFP4 quantized experts).

Usage:
    from gpt_oss.gpt_oss import GptOss

    model, tokenizer = GptOss.from_pretrained("openai/gpt-oss-20b")

    ids = torch.tensor([tokenizer.encode("Hello", add_special_tokens=False)])
    logits = model(ids)

Key features:
- Mixture of Experts with 32 experts, 4 active per token
- MXFP4-quantized expert weights (dequantized to f32 for reference)
- Heterogeneous layers: sliding_attention and full_attention
- RoPE with YaRN scaling
- Attention sinks (per-head learnable logit bias)
- SwiGLU variant: gate*sigmoid(gate*1.702) * (up+1), with clamping

Note: This pure PyTorch implementation dequantizes MXFP4 weights on the fly.
Production inference should use optimized MXFP4 kernels.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.nn import fusable
from gpt_oss.mxfp4 import decode_mxfp4_scales, dequantize_mxfp4_block, mxfp4_linear

# Try to load the optimized C MXFP4 kernel (AVX2, fused MoE).
# Falls back to pure PyTorch dequant if not available.
try:
    import ctypes
    import os as _os
    _KERNEL_PATH = _os.environ.get(
        "MXFP4_KERNEL_PATH",
        str(Path(__file__).resolve().parent / "mxfp4_kernel.so"),
    )
    _MXFP4_C = ctypes.CDLL(_KERNEL_PATH)
    _MXFP4_C.mxfp4_moe_fused.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    _MXFP4_C.mxfp4_moe_fused.restype = None
except (OSError, AttributeError):
    _MXFP4_C = None


# =============================================================================
# Weight loading
# =============================================================================

def _from_pretrained(model_class, model_id):
    """
    Load GPT-OSS from HuggingFace with MXFP4 expert weights.

    Needs custom loading because:
    1. Expert weights are MXFP4 buffers (not nn.Parameter)
    2. Scale tensors need pow(2, x - 127) conversion
    3. Standard from_pretrained doesn't handle register_buffer names
    """
    import json
    from pathlib import Path
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    if Path(model_id).is_dir():
        path = Path(model_id)
    else:
        path = Path(snapshot_download(model_id))

    with open(path / "config.json") as f:
        config = json.load(f)

    # Remove quantization_config — we handle it ourselves
    config.pop("quantization_config", None)

    model = model_class(config)

    # Load all safetensor shards
    index_file = path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        unique_files = sorted(set(index["weight_map"].values()))
    else:
        unique_files = sorted(p.name for p in path.glob("*.safetensors"))

    weights = {}
    for filename in unique_files:
        weights.update(load_file(path / filename))

    # Map HF names to our model names
    # HF: model.layers.0.self_attn.q_proj.weight -> layers.0.self_attn.q_proj.weight
    # state_dict() includes both parameters and buffers
    # Weights are loaded as-is (bf16 stays bf16). Model runs in bf16 for attention,
    # f32 for RMSNorm internals and MXFP4 dequant (matching opencode reference).
    state = model.state_dict()
    mapped = {}
    for our_name in state:
        hf_name = "model." + our_name if not our_name.startswith("lm_head") else our_name
        if hf_name in weights:
            mapped[our_name] = weights[hf_name]

    model.load_state_dict(mapped, strict=False, assign=True)

    # Post-load: convert uint8 scales to f32 powers-of-2, activate MXFP4 path
    for module in model.modules():
        hook = getattr(module, "post_load_hook", None)
        if callable(hook):
            hook()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


# =============================================================================
# YaRN RoPE
# =============================================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
        self.attention_scaling = 1.0
        inv_freq = self._build_inv_freq(config, head_dim)
        self.register_buffer("inv_freq", inv_freq)

    def _build_inv_freq(self, config, head_dim):
        rope_scaling = config.get("rope_scaling")
        rope_theta = config.get("rope_theta", 10000.0)

        if rope_scaling and rope_scaling.get("rope_type") == "yarn":
            inv_freq, attn_factor = _compute_yarn_inv_freq(config, head_dim, rope_theta, rope_scaling)
            self.attention_scaling = attn_factor
            return inv_freq

        return 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

    def forward(self, q, k):
        seq_len = q.shape[2]
        t = torch.arange(seq_len, device=q.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = (emb.cos() * self.attention_scaling).to(q.dtype)
        sin = (emb.sin() * self.attention_scaling).to(q.dtype)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        def rotate(x):
            x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
            return torch.cat([-x2, x1], dim=-1)

        return (q * cos) + (rotate(q) * sin), (k * cos) + (rotate(k) * sin)


def _compute_yarn_inv_freq(config, head_dim, base, rope_scaling):
    """Compute YaRN-scaled inverse frequencies and attention factor."""
    dim = int(head_dim * config.get("partial_rotary_factor", 1.0))
    factor = rope_scaling["factor"]
    attention_factor = rope_scaling.get("attention_factor")
    mscale = rope_scaling.get("mscale")
    mscale_all_dim = rope_scaling.get("mscale_all_dim")
    original_max_pos = rope_scaling.get("original_max_position_embeddings") or config["max_position_embeddings"]

    def get_mscale(scale, mscale_value=1.0):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale_value * math.log(scale) + 1.0

    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
        else:
            attention_factor = get_mscale(factor)

    beta_fast = rope_scaling.get("beta_fast", 32)
    beta_slow = rope_scaling.get("beta_slow", 1)

    def find_correction_dim(num_rotations, dim_value, base_value, max_pos):
        return (dim_value * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base_value))

    def find_correction_range(low_rot, high_rot, dim_value, base_value, max_pos, truncate):
        low = find_correction_dim(low_rot, dim_value, base_value, max_pos)
        high = find_correction_dim(high_rot, dim_value, base_value, max_pos)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim_value - 1)

    def linear_ramp_factor(min_val, max_val, dim_value):
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim_value, dtype=torch.float32) - min_val) / (max_val - min_val)
        return torch.clamp(linear_func, 0, 1)

    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    truncate = rope_scaling.get("truncate", True)
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_pos, truncate)
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    return inv_freq, attention_factor


# =============================================================================
# Building blocks
# =============================================================================

@fusable(kernel="norm")
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


@fusable(kernel="attention")
class Attention(nn.Module):
    """GQA with attention sinks and bias on QKV+O projections."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, rope_theta, attention_bias, config, layer_idx=0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        # Sliding window for sliding_attention layers
        layer_types = config.get("layer_types", [])
        if layer_idx < len(layer_types) and layer_types[layer_idx] == "sliding_attention":
            self.sliding_window = config.get("sliding_window", 0) or 0
        else:
            self.sliding_window = 0

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=attention_bias)

        # Attention sinks: per-head learnable logit added as extra "token"
        self.sinks = nn.Parameter(torch.zeros(num_heads))

        self.rotary = RotaryEmbedding(config)

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

        # Attention with sinks
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask (with optional sliding window)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        if self.sliding_window > 0:
            # Also mask positions outside the sliding window
            positions = torch.arange(T, device=x.device)
            sliding_mask = (positions.unsqueeze(1) - positions.unsqueeze(0)) >= self.sliding_window
            causal_mask = causal_mask | sliding_mask
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Append sink logits as extra column
        sink_logits = self.sinks.view(1, self.num_heads, 1, 1).expand(B, -1, T, 1)
        combined = torch.cat([attn_weights, sink_logits], dim=-1)

        # Numerical stabilization (critical for bf16)
        combined = combined - combined.max(dim=-1, keepdim=True).values

        # Softmax over keys + sink
        probs = F.softmax(combined, dim=-1, dtype=combined.dtype)

        # Drop sink column, attend only to real values
        attn_probs = probs[..., :-1]
        out = torch.matmul(attn_probs, v)

        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


@fusable(kernel="moe")
class MoEMLP(nn.Module):
    """
    Mixture of Experts MLP with MXFP4 quantized expert weights.

    Architecture:
    1. Router: linear(hidden_size -> num_experts) with bias, top-k selection
    2. Per-expert: gate_up = MXFP4_matmul(x, gate_up_blocks) + bias
    3. SwiGLU variant: gate*sigmoid(gate*alpha) * (up+1), with clamping
    4. Down projection: MXFP4_matmul(activated, down_blocks) + bias
    5. Weighted sum by routing scores

    HF weight naming: mlp.router.{weight,bias}, mlp.experts.{gate_up_proj_*,down_proj_*}
    """

    def __init__(self, hidden_size, intermediate_size, num_experts, num_experts_per_tok):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.router = _TopKRouter(hidden_size, num_experts, num_experts_per_tok)
        self.experts = _Experts(hidden_size, intermediate_size, num_experts)

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)

        routing_weights, expert_indices = self.router(x_flat)
        output = self.experts(x_flat, routing_weights, expert_indices, self.num_experts_per_tok)
        return output.reshape(B, T, D)


class _Experts(nn.Module):
    """MXFP4 expert weights container. Matches HF naming: experts.gate_up_proj_*"""

    ALPHA = 1.702
    LIMIT = 7.0

    def __init__(self, hidden_size, intermediate_size, num_experts):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts

        blocks_per_feature_in = hidden_size // 32
        blocks_per_feature_ff = intermediate_size // 32

        self.register_buffer(
            "gate_up_proj_blocks",
            torch.zeros(num_experts, 2 * intermediate_size, blocks_per_feature_in, 16, dtype=torch.uint8),
        )
        self.register_buffer(
            "gate_up_proj_scales",
            torch.zeros(num_experts, 2 * intermediate_size, blocks_per_feature_in, dtype=torch.uint8),
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.zeros(num_experts, 2 * intermediate_size)
        )
        self.register_buffer(
            "down_proj_blocks",
            torch.zeros(num_experts, hidden_size, blocks_per_feature_ff, 16, dtype=torch.uint8),
        )
        self.register_buffer(
            "down_proj_scales",
            torch.zeros(num_experts, hidden_size, blocks_per_feature_ff, dtype=torch.uint8),
        )
        self.down_proj_bias = nn.Parameter(
            torch.zeros(num_experts, hidden_size)
        )

        self._gate_up_scales_f32 = None
        self._down_scales_f32 = None
        self._gate_up_bias_f32 = None
        self._down_bias_f32 = None

    def post_load_hook(self):
        """Convert uint8 scales to f32 power-of-2 values and cache biases as f32."""
        if self.gate_up_proj_scales.numel() > 0:
            self._gate_up_scales_f32 = decode_mxfp4_scales(self.gate_up_proj_scales).contiguous()
        if self.down_proj_scales.numel() > 0:
            self._down_scales_f32 = decode_mxfp4_scales(self.down_proj_scales).contiguous()
        self._gate_up_bias_f32 = self.gate_up_proj_bias.float().contiguous()
        self._down_bias_f32 = self.down_proj_bias.float().contiguous()

    def forward(self, x_flat, routing_weights, expert_indices, num_experts_per_tok):
        input_dtype = x_flat.dtype
        if _MXFP4_C is not None:
            return self._forward_fused(x_flat, routing_weights, expert_indices, num_experts_per_tok).to(input_dtype)
        return self._forward_pytorch(x_flat, routing_weights, expert_indices, num_experts_per_tok).to(input_dtype)

    def _forward_fused(self, x_flat, routing_weights, expert_indices, num_experts_per_tok):
        """Fast path: fused C kernel does gate_up + SwiGLU + down + routing in one call."""
        x_f32 = x_flat.float().contiguous()
        expert_indices_i32 = expert_indices.to(torch.int32).contiguous()
        routing_f32 = routing_weights.float().contiguous()

        batch = x_f32.shape[0]
        in_features = x_f32.shape[-1]
        expert_dim = self.intermediate_size
        blocks_per_feature_gate = in_features // 32
        blocks_per_feature_down = expert_dim // 32

        gu_b = self.gate_up_proj_blocks.contiguous()
        dn_b = self.down_proj_blocks.contiguous()

        expert_stride_gate_blocks = gu_b.shape[1] * gu_b.shape[2] * gu_b.shape[3]
        expert_stride_gate_scales = self._gate_up_scales_f32.shape[1] * self._gate_up_scales_f32.shape[2]
        expert_stride_down_blocks = dn_b.shape[1] * dn_b.shape[2] * dn_b.shape[3]
        expert_stride_down_scales = self._down_scales_f32.shape[1] * self._down_scales_f32.shape[2]

        output = torch.empty((batch, self.hidden_size), device=x_flat.device, dtype=torch.float32)
        _MXFP4_C.mxfp4_moe_fused(
            x_f32.data_ptr(),
            gu_b.data_ptr(), self._gate_up_scales_f32.data_ptr(), self._gate_up_bias_f32.data_ptr(),
            dn_b.data_ptr(), self._down_scales_f32.data_ptr(), self._down_bias_f32.data_ptr(),
            expert_indices_i32.data_ptr(), routing_f32.data_ptr(), output.data_ptr(),
            batch, in_features, expert_dim, self.hidden_size,
            blocks_per_feature_gate, blocks_per_feature_down, num_experts_per_tok,
            expert_stride_gate_blocks, expert_stride_gate_scales,
            expert_stride_down_blocks, expert_stride_down_scales,
        )
        return output

    def _forward_pytorch(self, x_flat, routing_weights, expert_indices, num_experts_per_tok):
        """Fallback: pure PyTorch with on-the-fly MXFP4 dequant."""
        output = torch.zeros(x_flat.shape[0], self.hidden_size, device=x_flat.device, dtype=torch.float32)

        for k in range(num_experts_per_tok):
            indices = expert_indices[:, k]
            weights_k = routing_weights[:, k]

            for expert_idx in range(self.num_experts):
                mask = indices == expert_idx
                if not mask.any():
                    continue

                tokens = x_flat[mask]

                gate_up = mxfp4_linear(
                    tokens,
                    self.gate_up_proj_blocks[expert_idx],
                    self._gate_up_scales_f32[expert_idx],
                    self._gate_up_bias_f32[expert_idx],
                )

                gate = gate_up[..., ::2]
                up = gate_up[..., 1::2]

                gate = gate.clamp(max=self.LIMIT)
                up = up.clamp(min=-self.LIMIT, max=self.LIMIT)
                glu = gate * torch.sigmoid(gate * self.ALPHA)
                activated = (up + 1.0) * glu

                down = mxfp4_linear(
                    activated,
                    self.down_proj_blocks[expert_idx],
                    self._down_scales_f32[expert_idx],
                    self._down_bias_f32[expert_idx],
                )

                output[mask] += weights_k[mask].unsqueeze(-1).float() * down

        return output


class _TopKRouter(nn.Module):
    """Top-k expert router with bias."""

    def __init__(self, hidden_size, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.bias = nn.Parameter(torch.empty(num_experts))

    def forward(self, x):
        logits = F.linear(x, self.weight, self.bias)
        top_values, top_indices = torch.topk(logits, self.top_k, dim=-1)
        top_weights = F.softmax(top_values, dim=-1, dtype=top_values.dtype)
        return top_weights, top_indices


# =============================================================================
# Block + Model
# =============================================================================

class HybridBlock(nn.Module):
    """Single GPT-OSS decoder block (attention + MoE MLP with residuals).

    Heterogeneous: layer type (sliding_attention vs full_attention) is
    determined by config["layer_types"][layer_idx].
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        num_kv_heads = config.get("num_key_value_heads", num_heads)
        head_dim = config.get("head_dim", hidden_size // num_heads)
        intermediate_size = config["intermediate_size"]
        num_experts = config.get("num_local_experts", 32)
        num_experts_per_tok = config.get("num_experts_per_tok", 4)
        eps = config.get("rms_norm_eps", 1e-5)
        attention_bias = config.get("attention_bias", True)

        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.self_attn = Attention(
            hidden_size, num_heads, num_kv_heads, head_dim,
            config.get("rope_theta", 150000.0), attention_bias, config, layer_idx,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.mlp = MoEMLP(hidden_size, intermediate_size, num_experts, num_experts_per_tok)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class GptOss(nn.Module):
    """
    GPT-OSS (OpenAI gpt-oss-20b).

    MoE architecture with:
    - 24 layers (12 sliding_attention + 12 full_attention, alternating)
    - 32 experts per layer, 4 active per token
    - MXFP4-quantized expert weights
    - Attention sinks
    - YaRN RoPE scaling
    """

    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        num_layers = config["num_hidden_layers"]
        vocab_size = config["vocab_size"]
        eps = config.get("rms_norm_eps", 1e-5)

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([HybridBlock(config, i) for i in range(num_layers)])
        self.norm = RMSNorm(hidden_size, eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))

    @staticmethod
    def from_pretrained(model_id):
        return _from_pretrained(GptOss, model_id)
