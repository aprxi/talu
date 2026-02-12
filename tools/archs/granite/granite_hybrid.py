"""
Granite Hybrid - Pure PyTorch implementation.

Supports Granite 4.0 Hybrid (Mamba2 + Attention).

Usage:
    from granite.granite_hybrid import GraniteHybrid

    model, tokenizer = GraniteHybrid.from_pretrained("ibm-granite/granite-4.0-h-350m")

    ids = torch.tensor([tokenizer.encode("Hello").ids])
    logits = model(ids)

Key features:
- Hybrid Mamba2-Attention architecture with heterogeneous layers
- Mamba2 (SSD) state-space layers for efficient sequence modeling
- Standard attention layers interspersed per config.layer_types
- muP (Maximal Update Parameterization) scaling factors
- No positional embeddings for Mamba layers (position_embedding_type: "nope")

Note: This pure PyTorch Mamba2 is sequential (slow) for verification/tracing.
Production inference requires optimized CUDA kernels.
"""

import math
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
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat([-x2, x1], dim=-1)

        return (q * cos) + (rotate(q) * sin), (k * cos) + (rotate(k) * sin)


@fusable(kernel="mamba")
class Mamba2(nn.Module):
    """
    Pure PyTorch implementation of Mamba2 (SSD - State Space Duality).

    This is a sequential implementation for verification/tracing.
    Not optimized for production - use CUDA kernels for inference.

    The Mamba2 layer uses:
    - Input projection to (x, z, B, C, dt)
    - 1D depthwise convolution on x
    - Discretized SSM recurrence
    - Gated output with z
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config["hidden_size"]
        self.d_state = config["mamba_d_state"]  # 128
        self.d_head = config["mamba_d_head"]  # 32
        self.n_heads = config["mamba_n_heads"]  # 48
        self.n_groups = config.get("mamba_n_groups", 1)  # 1
        self.d_conv = config["mamba_d_conv"]  # 4
        self.expand = config.get("mamba_expand", 2)

        # d_inner = expand * d_model (for Granite: 2 * 768 = 1536)
        # But also: d_inner = n_heads * d_head = 48 * 32 = 1536
        self.d_inner = self.n_heads * self.d_head

        # Input projection dimensions:
        # x: d_inner, z: d_inner, B: n_groups * d_state, C: n_groups * d_state, dt: n_heads
        proj_size = (
            self.d_inner  # x
            + self.d_inner  # z
            + self.n_groups * self.d_state  # B
            + self.n_groups * self.d_state  # C
            + self.n_heads  # dt
        )
        self.in_proj = nn.Linear(
            self.d_model, proj_size, bias=config.get("mamba_proj_bias", False)
        )

        # Depthwise conv1d - applied to xBC (x, B, C concatenated)
        # xBC_len = d_inner + 2 * n_groups * d_state
        self.xBC_len = self.d_inner + 2 * self.n_groups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=self.xBC_len,
            out_channels=self.xBC_len,
            kernel_size=self.d_conv,
            groups=self.xBC_len,
            padding=self.d_conv - 1,
            bias=config.get("mamba_conv_bias", True),
        )

        # SSM parameters
        # In Mamba2, A is scalar per head (not a matrix)
        self.A_log = nn.Parameter(torch.zeros(self.n_heads))
        self.D = nn.Parameter(torch.ones(self.n_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.n_heads))

        # Output normalization and projection
        self.norm = RMSNorm(self.d_inner, config.get("rms_norm_eps", 1e-5))
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=config.get("mamba_proj_bias", False)
        )

    def forward(self, x):
        B, L, _ = x.shape

        # 1. Input projection
        xz_bc_dt = self.in_proj(x)

        # Split projections - Mamba2 layout is: z, xBC, dt
        # where xBC = x, B, C concatenated
        d_inner = self.d_inner
        d_bc = self.n_groups * self.d_state

        # z comes first (gating)
        z_proj = xz_bc_dt[..., :d_inner]
        # xBC comes next (to be convolved together)
        xBC = xz_bc_dt[..., d_inner : d_inner + self.xBC_len]
        # dt comes last
        dt_proj = xz_bc_dt[..., d_inner + self.xBC_len :]

        # 2. Conv1D on xBC (causal) - x, B, C are convolved together
        xBC_conv = xBC.transpose(1, 2)  # (B, xBC_len, L)
        xBC_conv = self.conv1d(xBC_conv)[..., :L]  # Causal: trim to original length
        xBC_conv = F.silu(xBC_conv)  # SiLU applied to entire xBC output
        xBC_conv = xBC_conv.transpose(1, 2)  # (B, L, xBC_len)

        # Split xBC after conv+silu into x, B, C
        x_conv = xBC_conv[..., :d_inner]  # (B, L, d_inner)
        B_proj = xBC_conv[..., d_inner : d_inner + d_bc]
        C_proj = xBC_conv[..., d_inner + d_bc :]

        # 3. Discretization
        dt = F.softplus(dt_proj + self.dt_bias)  # (B, L, n_heads)
        A = -torch.exp(self.A_log)  # (n_heads,) - negative for stability

        # 4. SSM scan (sequential - slow but correct)
        # Reshape for heads: (B, L, n_heads, d_head)
        x_heads = x_conv.view(B, L, self.n_heads, self.d_head)

        # B and C are (B, L, d_state) when n_groups=1
        # For Mamba2 with n_groups=1, B/C are shared across heads
        B_ssm = B_proj.view(B, L, self.n_groups, self.d_state)
        C_ssm = C_proj.view(B, L, self.n_groups, self.d_state)

        # Initialize hidden state: (B, n_heads, d_head, d_state)
        h = torch.zeros(
            B, self.n_heads, self.d_head, self.d_state, device=x.device, dtype=x.dtype
        )

        y_list = []
        for t in range(L):
            # Get current timestep values
            x_t = x_heads[:, t]  # (B, n_heads, d_head)
            dt_t = dt[:, t]  # (B, n_heads)
            B_t = B_ssm[:, t, 0]  # (B, d_state) - group 0
            C_t = C_ssm[:, t, 0]  # (B, d_state) - group 0

            # Discretized A: exp(A * dt)
            # A is (n_heads,), dt_t is (B, n_heads)
            dA = torch.exp(A.unsqueeze(0) * dt_t)  # (B, n_heads)

            # Discretized B: dt * B (simplified for diagonal A)
            # x_t: (B, n_heads, d_head), B_t: (B, d_state), dt_t: (B, n_heads)
            # dB * x contribution: outer product x_t with B_t, scaled by dt
            dB_x = (
                dt_t.unsqueeze(-1).unsqueeze(-1)
                * x_t.unsqueeze(-1)
                * B_t.unsqueeze(1).unsqueeze(1)
            )
            # dB_x: (B, n_heads, d_head, d_state)

            # State update: h = dA * h + dB * x
            h = dA.unsqueeze(-1).unsqueeze(-1) * h + dB_x

            # Output: y = C @ h + D * x
            # h: (B, n_heads, d_head, d_state), C_t: (B, d_state)
            y_t = torch.einsum("bhds,bs->bhd", h, C_t)  # (B, n_heads, d_head)

            # Add skip connection with D
            y_t = y_t + self.D.unsqueeze(0).unsqueeze(-1) * x_t

            y_list.append(y_t)

        # Stack outputs: (B, L, n_heads, d_head)
        y = torch.stack(y_list, dim=1)
        y = y.view(B, L, self.d_inner)

        # 5. Gated RMS Norm: norm(y * silu(gate))
        y = y * F.silu(z_proj)
        y = self.norm(y)

        # 6. Output projection
        return self.out_proj(y)


@fusable(kernel="attention")
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.get("attention_bias", False),
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.get("attention_bias", False),
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.get("attention_bias", False),
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.get("attention_bias", False),
        )

        # RoPE (only if not "nope")
        pos_type = config.get("position_embedding_type", "rope")
        if pos_type != "nope":
            self.rotary = RotaryEmbedding(
                self.head_dim, config.get("rope_theta", 10000.0)
            )
        else:
            self.rotary = None

        # muP attention scaling
        self.attention_multiplier = config.get("attention_multiplier", 1.0)

    def forward(self, x):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE if enabled
        if self.rotary is not None:
            q, k = self.rotary(q, k)

        # GQA: expand k, v
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # muP: scale query instead of using 1/sqrt(d) in attention
        # attention_multiplier replaces the standard scaling
        scale = self.attention_multiplier if self.attention_multiplier != 1.0 else None
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)

        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


@fusable(kernel="mlp")
class MLP(nn.Module):
    """
    Gated Linear Unit MLP with combined input projection.

    HF Granite Hybrid uses input_linear (2*intermediate, hidden) -> chunk -> silu(gate) * up.
    """

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        # Combined gate + up projection (HF style)
        self.input_linear = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.output_linear = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        hidden = self.input_linear(x)
        gate, up = hidden.chunk(2, dim=-1)
        return self.output_linear(F.silu(gate) * up)


class HybridBlock(nn.Module):
    """
    A transformer block that can be either Mamba2 or Attention.

    The layer type is determined by config["layer_types"][layer_idx].
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_type = config["layer_types"][layer_idx]
        self.hidden_size = config["hidden_size"]

        self.input_layernorm = RMSNorm(self.hidden_size, config.get("rms_norm_eps", 1e-5))

        if self.layer_type == "mamba":
            self.mixer = Mamba2(config)
        else:
            self.mixer = Attention(config)

        self.post_attention_layernorm = RMSNorm(
            self.hidden_size, config.get("rms_norm_eps", 1e-5)
        )

        # MLP (dense for this model - num_local_experts=0)
        num_experts = config.get("num_local_experts", 0)
        if num_experts > 0:
            raise NotImplementedError(
                "MoE not implemented for hybrid. Use dense (num_local_experts=0)."
            )
        self.mlp = MLP(self.hidden_size, config["intermediate_size"])

        # muP residual scaling
        self.residual_multiplier = config.get("residual_multiplier", 1.0)

        # Align weight names with HF checkpoints (mamba/self_attn, shared_mlp).
        alias = "mamba" if self.layer_type == "mamba" else "self_attn"
        overrides = {}
        for name in self.state_dict().keys():
            if name.startswith("mixer."):
                mapped = name.replace("mixer.", f"{alias}.", 1)
                overrides[name] = [f"model.layers.{{d}}.{mapped}"]
            if name.startswith("mlp."):
                mapped = name.replace("mlp.", "shared_mlp.", 1)
                overrides.setdefault(name, []).append(f"model.layers.{{d}}.{mapped}")
        if overrides:
            self.weight_map_overrides = overrides

    def forward(self, x):
        # Pre-norm architecture with residual scaling
        h = self.mixer(self.input_layernorm(x))
        if self.residual_multiplier != 1.0:
            h = h * self.residual_multiplier
        x = x + h

        h = self.mlp(self.post_attention_layernorm(x))
        if self.residual_multiplier != 1.0:
            h = h * self.residual_multiplier
        x = x + h

        return x


class GraniteHybrid(nn.Module):
    """
    Granite Hybrid model with heterogeneous Mamba2/Attention layers.

    Architecture features:
    - Layer types determined by config.layer_types array
    - muP scaling (embedding_multiplier, logits_scaling, residual_multiplier)
    - No positional embeddings for Mamba layers (position_embedding_type: "nope")
    - GQA for attention layers
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        num_layers = config["num_hidden_layers"]

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        # Heterogeneous layers based on config.layer_types
        self.layers = nn.ModuleList(
            [HybridBlock(config, i) for i in range(num_layers)]
        )

        self.norm = RMSNorm(self.hidden_size, config.get("rms_norm_eps", 1e-5))
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        if config.get("tie_word_embeddings", True):
            self.lm_head.weight = self.embed_tokens.weight

        # muP scaling factors
        self.embedding_multiplier = config.get("embedding_multiplier", 1.0)
        self.logits_scaling = config.get("logits_scaling", 1.0)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)

        # muP embedding scaling
        if self.embedding_multiplier != 1.0:
            x = x * self.embedding_multiplier

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        # muP logits scaling
        if self.logits_scaling != 1.0:
            logits = logits / self.logits_scaling

        return logits

    @staticmethod
    def from_pretrained(model_id):
        from pathlib import Path
        from safetensors.torch import load_file
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        import json

        path = Path(snapshot_download(model_id))

        with open(path / "config.json") as f:
            config = json.load(f)

        model = GraniteHybrid(config)

        weights = {}
        for fp in path.glob("*.safetensors"):
            weights.update(load_file(fp))

        state = model.state_dict()
        mapped = {}
        for name in state:
            hf_name = "model." + name if not name.startswith("lm_head") else name
            # Try name mappings in order until we find a match
            if hf_name not in weights:
                # Granite Hybrid uses "mamba" in safetensors but we use "mixer" for Mamba2 layers
                if ".mixer." in name:
                    mamba_name = hf_name.replace(".mixer.", ".mamba.")
                    if mamba_name in weights:
                        hf_name = mamba_name
                    else:
                        # For attention layers, HF uses "self_attn" but we use "mixer"
                        attn_name = hf_name.replace(".mixer.", ".self_attn.")
                        if attn_name in weights:
                            hf_name = attn_name
                # HF uses "shared_mlp" but we use "mlp"
                if ".mlp." in name and hf_name not in weights:
                    mlp_name = hf_name.replace(".mlp.", ".shared_mlp.")
                    if mlp_name in weights:
                        hf_name = mlp_name
            if hf_name in weights:
                mapped[name] = weights[hf_name]

        model.load_state_dict(mapped, strict=False)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return model, tokenizer
