"""
Youtu-VL - Pure PyTorch implementation.

This is a Vision-Language Model with:
- Siglip2 vision encoder
- MLA (Multi-Latent Attention) language decoder (from DeepSeek-V2)

The language model uses Multi-Latent Attention which compresses Q/KV via
low-rank projections for efficient KV caching.

Usage:
    from youtu_vl.youtu_vl import YoutuVL

    model, tokenizer, processor = YoutuVL.from_pretrained("tencent/Youtu-VL-4B-Instruct")

    # Text-only
    ids = torch.tensor([tokenizer.encode("Hello")])
    logits = model(ids)

    # With image (requires processor)
    inputs = processor(images=image, text="Describe this image", return_tensors="pt")
    logits = model(**inputs)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.nn import fusable


# =============================================================================
# RMSNorm (shared by both language and vision models)
# =============================================================================

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


class LayerNorm(nn.Module):
    """Standard LayerNorm with bias (used in vision encoder)."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


# =============================================================================
# Rotary Position Embedding
# =============================================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=500000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, seq_len, device, dtype):
        if self.cos_cached is None or seq_len > self.cos_cached.shape[0]:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat([freqs, freqs], dim=-1)
            self.cos_cached = emb.cos().to(dtype)
            self.sin_cached = emb.sin().to(dtype)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding (standard, non-interleaved)."""
    # cos, sin: [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_interleave(q, k, cos, sin):
    """Apply rotary position embedding (interleaved layout)."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Interleaved: reshape to (b, h, s, d//2, 2) then transpose to get pairs
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# MLA (Multi-Latent Attention) - from DeepSeek-V2
# =============================================================================

@fusable(kernel="attention", config=["mla"])
class MLAttention(nn.Module):
    """
    Multi-Latent Attention from DeepSeek-V2.

    Key innovation: compress Q and KV via low-rank projections.
    - Q: hidden -> q_lora_rank -> num_heads * qk_head_dim
    - KV: hidden -> kv_lora_rank -> num_heads * (qk_nope_head_dim + v_head_dim)
    - RoPE applied only to qk_rope_head_dim portion
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_head_dim,
        qk_rope_head_dim,
        qk_nope_head_dim,
        v_head_dim,
        rope_theta,
        rope_interleave=True,
        eps=1e-6,
    ):
        super().__init__()
        # Store MLA config as attributes for tracer to extract
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_head_dim = qk_head_dim  # = qk_rope_head_dim + qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_interleave = rope_interleave

        # Q projection: two-stage low-rank
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps)
        self.q_b_proj = nn.Linear(q_lora_rank, num_heads * qk_head_dim, bias=False)

        # KV projection: compressed KV with shared RoPE key
        # Output: kv_lora_rank (for nope part) + qk_rope_head_dim (shared rope key)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps)
        # Expand to per-head nope_k and v
        self.kv_b_proj = nn.Linear(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False
        )

        # Output projection
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)

        # RoPE for the rope portion only
        self.rotary = RotaryEmbedding(qk_rope_head_dim, rope_theta)

        # Scaling factor
        self.scaling = qk_head_dim ** (-0.5)

    def forward(self, x):
        B, T, _ = x.shape

        # === Q projection ===
        # hidden -> q_lora_rank -> num_heads * qk_head_dim
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(B, T, self.num_heads, self.qk_head_dim).transpose(1, 2)
        # Split into nope and rope parts
        q_nope, q_rope = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # === KV projection ===
        # hidden -> (kv_lora_rank + qk_rope_head_dim)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        kv_nope_compressed, k_rope = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # Expand kv_nope_compressed -> per-head (nope_k, v)
        kv_expanded = self.kv_b_proj(self.kv_a_layernorm(kv_nope_compressed))
        kv_expanded = kv_expanded.view(
            B, T, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 2)
        k_nope, v = torch.split(
            kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # k_rope is shared across heads: [B, T, qk_rope_head_dim] -> [B, 1, T, qk_rope_head_dim]
        k_rope = k_rope.view(B, 1, T, self.qk_rope_head_dim)

        # === Apply RoPE to rope portions ===
        cos, sin = self.rotary(T, x.device, x.dtype)
        if self.rope_interleave:
            q_rope, k_rope = apply_rotary_pos_emb_interleave(q_rope, k_rope, cos, sin)
        else:
            q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)

        # Expand k_rope to all heads
        k_rope = k_rope.expand(-1, self.num_heads, -1, -1)

        # === Concatenate nope and rope parts ===
        q = torch.cat([q_nope, q_rope], dim=-1)  # [B, num_heads, T, qk_head_dim]
        k = torch.cat([k_nope, k_rope], dim=-1)  # [B, num_heads, T, qk_head_dim]

        # === Attention ===
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scaling)

        # === Output projection ===
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)


# =============================================================================
# MLP (SwiGLU)
# =============================================================================

@fusable(kernel="mlp")
class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# =============================================================================
# Decoder Block (Language Model)
# =============================================================================

class Block(nn.Module):
    """Single transformer block with MLA attention."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_head_dim,
        qk_rope_head_dim,
        qk_nope_head_dim,
        v_head_dim,
        intermediate_size,
        rope_theta,
        rope_interleave=True,
        eps=1e-6,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.self_attn = MLAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_head_dim=qk_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            v_head_dim=v_head_dim,
            rope_theta=rope_theta,
            rope_interleave=rope_interleave,
            eps=eps,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.mlp = MLP(hidden_size, intermediate_size)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# =============================================================================
# Vision Encoder (Siglip2)
# =============================================================================

class Siglip2Attention(nn.Module):
    """Standard multi-head attention for vision encoder."""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, attention_mask=None):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Vision uses bidirectional attention (not causal)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(out)


class Siglip2MLP(nn.Module):
    """GELU MLP for vision encoder."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Siglip2EncoderLayer(nn.Module):
    """Single vision encoder layer."""

    def __init__(self, hidden_size, num_heads, intermediate_size, eps=1e-6):
        super().__init__()
        self.layer_norm1 = LayerNorm(hidden_size, eps)
        self.self_attn = Siglip2Attention(hidden_size, num_heads)
        self.layer_norm2 = LayerNorm(hidden_size, eps)
        self.mlp = Siglip2MLP(hidden_size, intermediate_size)

    def forward(self, x, attention_mask=None):
        x = x + self.self_attn(self.layer_norm1(x), attention_mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class Siglip2VisionEncoder(nn.Module):
    """Siglip2 Vision Transformer encoder."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.intermediate_size = config["intermediate_size"]
        self.num_layers = config["num_hidden_layers"]
        self.patch_size = config["patch_size"]
        self.eps = config.get("layer_norm_eps", 1e-6)

        # Patch embedding (Conv2d)
        self.patch_embedding = nn.Conv2d(
            in_channels=config.get("num_channels", 3),
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        # Encoder layers
        self.layers = nn.ModuleList([
            Siglip2EncoderLayer(
                self.hidden_size, self.num_heads, self.intermediate_size, self.eps
            )
            for _ in range(self.num_layers)
        ])

        self.post_layernorm = LayerNorm(self.hidden_size, self.eps)

    def forward(self, pixel_values, attention_mask=None):
        # pixel_values: [B, C, H, W]
        # Patch embed: [B, hidden_size, H//patch, W//patch]
        x = self.patch_embedding(pixel_values)
        # Flatten patches: [B, hidden_size, num_patches] -> [B, num_patches, hidden_size]
        x = x.flatten(2).transpose(1, 2)

        for layer in self.layers:
            x = layer(x, attention_mask)

        return self.post_layernorm(x)


# =============================================================================
# Patch Merger (Vision -> Language projection)
# =============================================================================

class VLPatchMerger(nn.Module):
    """Merge vision patches and project to language model dimension."""

    def __init__(self, text_hidden_size, vision_hidden_size, spatial_merge_size=2):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        merged_dim = vision_hidden_size * (spatial_merge_size ** 2)

        self.ln_q = RMSNorm(vision_hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(merged_dim, merged_dim),
            nn.GELU(),
            nn.Linear(merged_dim, text_hidden_size),
        )

    def forward(self, x):
        # x: [B, num_patches, vision_hidden_size]
        # Apply norm then reshape for spatial merging
        x = self.ln_q(x)
        # Merge spatial_merge_size x spatial_merge_size patches
        # This reduces num_patches by factor of spatial_merge_size^2
        x = x.view(-1, self.mlp[0].in_features)
        return self.mlp(x)


# =============================================================================
# Full Model
# =============================================================================

class YoutuVLLanguageModel(nn.Module):
    """Language model decoder with MLA attention (text-only inference)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        intermediate_size = config["intermediate_size"]
        num_layers = config["num_hidden_layers"]
        vocab_size = config["vocab_size"]
        eps = config.get("rms_norm_eps", 1e-6)
        rope_theta = config.get("rope_theta", 500000.0)
        rope_interleave = config.get("rope_interleave", True)

        # MLA-specific config
        q_lora_rank = config["q_lora_rank"]
        kv_lora_rank = config["kv_lora_rank"]
        qk_head_dim = config["qk_head_dim"]
        qk_rope_head_dim = config["qk_rope_head_dim"]
        qk_nope_head_dim = config["qk_nope_head_dim"]
        v_head_dim = config["v_head_dim"]

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            Block(
                hidden_size=hidden_size,
                num_heads=num_heads,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                qk_head_dim=qk_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                qk_nope_head_dim=qk_nope_head_dim,
                v_head_dim=v_head_dim,
                intermediate_size=intermediate_size,
                rope_theta=rope_theta,
                rope_interleave=rope_interleave,
                eps=eps,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Tie embeddings if specified
        if config.get("tie_word_embeddings", True):
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids, inputs_embeds=None):
        if inputs_embeds is None:
            x = self.embed_tokens(input_ids)
        else:
            x = inputs_embeds

        for layer in self.layers:
            x = layer(x)

        return self.lm_head(self.norm(x))


class YoutuVL(nn.Module):
    """Full Youtu-VL model with vision encoder and language decoder."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_token_id = config.get("image_token_id", 128264)

        # Vision encoder
        vision_config = config.get("vision_config", {})
        self.siglip2 = Siglip2VisionEncoder(vision_config)

        # Patch merger
        self.merger = VLPatchMerger(
            text_hidden_size=config["hidden_size"],
            vision_hidden_size=vision_config["hidden_size"],
            spatial_merge_size=2,
        )

        # Language model
        self.model = YoutuVLLanguageModel(config)

        # Share vocab size for lm_head
        self.vocab_size = config["vocab_size"]
        self.lm_head = self.model.lm_head

    def forward(self, input_ids, pixel_values=None, **kwargs):
        """
        Forward pass.

        Args:
            input_ids: [B, T] token IDs (may contain image_token_id placeholders)
            pixel_values: [B, C, H, W] image tensor (optional)
        """
        # Get text embeddings
        inputs_embeds = self.model.embed_tokens(input_ids)

        # If we have images, process and insert them
        if pixel_values is not None:
            # Encode images
            image_embeds = self.siglip2(pixel_values)
            # Project to text dimension
            image_embeds = self.merger(image_embeds)

            # Replace image token positions with image embeddings
            # For simplicity, this assumes batch size 1 and contiguous image tokens
            mask = input_ids == self.image_token_id
            if mask.any():
                # Insert image embeddings at image token positions
                inputs_embeds = inputs_embeds.masked_scatter(
                    mask.unsqueeze(-1).expand_as(inputs_embeds),
                    image_embeds.unsqueeze(0),
                )

        # Forward through language model
        x = inputs_embeds
        for layer in self.model.layers:
            x = layer(x)

        return self.lm_head(self.model.norm(x))

    @staticmethod
    def from_pretrained(model_id):
        """Load pretrained model from HuggingFace."""
        import json
        from pathlib import Path
        from safetensors.torch import load_file
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer, AutoProcessor

        path = Path(snapshot_download(model_id))

        with open(path / "config.json") as f:
            config = json.load(f)

        model = YoutuVL(config)

        # Load weights
        weights = {}
        for fp in path.glob("*.safetensors"):
            weights.update(load_file(fp))

        # Map weights to model
        state = model.state_dict()
        mapped = {}

        for name in state:
            # Try direct mapping first
            if name in weights:
                mapped[name] = weights[name]
            # Try with model. prefix
            elif "model." + name in weights:
                mapped[name] = weights["model." + name]
            # Handle siglip2 prefix
            elif name.startswith("siglip2."):
                # siglip2.vision_model.X -> siglip2.vision_model.X
                hf_name = name.replace("siglip2.", "siglip2.vision_model.")
                if hf_name in weights:
                    mapped[name] = weights[hf_name]

        model.load_state_dict(mapped, strict=False)

        # Load tokenizer and processor
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        try:
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        except Exception:
            processor = None

        return model, tokenizer, processor
