"""
MiniLM (BERT) - Pure PyTorch implementation.

Usage:
    from bert.minilm import MiniLM

    model, tokenizer = MiniLM.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    ids = torch.tensor([tokenizer.encode("Hello world").ids])
    hidden_states = model(ids)          # [batch, seq_len, d_model]
    embedding = model.embed(ids)        # [batch, d_model], L2-normalized

Key differences from LLaMA:
- LayerNorm with bias (not RMSNorm)
- Learned absolute position embeddings (not RoPE)
- Bidirectional attention (is_causal=False)
- All linear layers have bias
- GELU activation (not SiLU gated)
- No lm_head (embedding model)

Supports: BERT, MiniLM, sentence-transformers BERT variants
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.nn import fusable


@fusable(kernel="norm")
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


@fusable(kernel="attention")
class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = self.dense(out.transpose(1, 2).reshape(B, T, -1))
        return self.LayerNorm(out + x)


@fusable(kernel="mlp", config=["gelu"])
class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense_in = nn.Linear(hidden_size, intermediate_size)
        self.dense_out = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)

    def forward(self, x):
        h = self.dense_out(F.gelu(self.dense_in(x)))
        return self.LayerNorm(h + x)


class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, eps=1e-12):
        super().__init__()
        self.attention = Attention(hidden_size, num_heads)
        self.mlp = MLP(hidden_size, intermediate_size)

        # Map our module names to HuggingFace BERT weight paths.
        # The runtime joins weight_prefix + candidate to find weights in checkpoints.
        # HF BERT nests Q/K/V under attention.self.* and output proj under
        # attention.output.*, while MLP weights use intermediate.*/output.*.
        self.weight_map_overrides = {}
        for suffix in ("weight", "bias"):
            for proj in ("query", "key", "value"):
                self.weight_map_overrides[f"attention.{proj}.{suffix}"] = [
                    f"attention.self.{proj}.{suffix}",
                ]
            self.weight_map_overrides[f"attention.dense.{suffix}"] = [
                f"attention.output.dense.{suffix}",
            ]
            self.weight_map_overrides[f"attention.LayerNorm.{suffix}"] = [
                f"attention.output.LayerNorm.{suffix}",
            ]
            self.weight_map_overrides[f"mlp.dense_in.{suffix}"] = [
                f"intermediate.dense.{suffix}",
            ]
            self.weight_map_overrides[f"mlp.dense_out.{suffix}"] = [
                f"output.dense.{suffix}",
            ]
            self.weight_map_overrides[f"mlp.LayerNorm.{suffix}"] = [
                f"output.LayerNorm.{suffix}",
            ]

    def forward(self, x):
        x = self.attention(x)
        x = self.mlp(x)
        return x


class MiniLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        intermediate_size = config["intermediate_size"]
        num_layers = config["num_hidden_layers"]
        vocab_size = config["vocab_size"]
        max_position = config.get("max_position_embeddings", 512)
        type_vocab_size = config.get("type_vocab_size", 2)
        eps = config.get("layer_norm_eps", 1e-12)

        # Embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps)

        # Encoder layers
        self.layers = nn.ModuleList([
            Block(hidden_size, num_heads, intermediate_size, eps)
            for _ in range(num_layers)
        ])

    def forward(self, input_ids):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        token_types = torch.zeros_like(input_ids)

        x = self.word_embeddings(input_ids) + \
            self.position_embeddings(positions) + \
            self.token_type_embeddings(token_types)
        x = self.LayerNorm(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def embed(self, input_ids):
        """Mean-pool hidden states and L2-normalize for sentence embeddings."""
        hidden = self.forward(input_ids)
        pooled = hidden.mean(dim=1)
        return F.normalize(pooled, p=2, dim=-1)

    @staticmethod
    def from_pretrained(model_id):
        """Load from HuggingFace with BERT-style weight name mapping."""
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        path = Path(snapshot_download(model_id))

        with open(path / "config.json") as f:
            config = json.load(f)

        model = MiniLM(config)

        weights = {}
        for fp in path.glob("*.safetensors"):
            weights.update(load_file(fp))

        # Map HuggingFace BERT weight names to our module names.
        # HF: bert.embeddings.word_embeddings.weight -> word_embeddings.weight
        # HF: bert.encoder.layer.{i}.attention.self.query.weight -> layers.{i}.attention.query.weight
        # HF: bert.encoder.layer.{i}.attention.output.dense.weight -> layers.{i}.attention.dense.weight
        # HF: bert.encoder.layer.{i}.attention.output.LayerNorm.weight -> layers.{i}.attention.LayerNorm.weight
        # HF: bert.encoder.layer.{i}.intermediate.dense.weight -> layers.{i}.mlp.dense_in.weight
        # HF: bert.encoder.layer.{i}.output.dense.weight -> layers.{i}.mlp.dense_out.weight
        # HF: bert.encoder.layer.{i}.output.LayerNorm.weight -> layers.{i}.mlp.LayerNorm.weight
        WEIGHT_MAP = {
            # Embeddings
            "bert.embeddings.word_embeddings.weight": "word_embeddings.weight",
            "bert.embeddings.position_embeddings.weight": "position_embeddings.weight",
            "bert.embeddings.token_type_embeddings.weight": "token_type_embeddings.weight",
            "bert.embeddings.LayerNorm.weight": "LayerNorm.weight",
            "bert.embeddings.LayerNorm.bias": "LayerNorm.bias",
        }

        num_layers = config["num_hidden_layers"]
        for i in range(num_layers):
            hf = f"bert.encoder.layer.{i}"
            ours = f"layers.{i}"
            layer_map = {
                f"{hf}.attention.self.query.weight": f"{ours}.attention.query.weight",
                f"{hf}.attention.self.query.bias": f"{ours}.attention.query.bias",
                f"{hf}.attention.self.key.weight": f"{ours}.attention.key.weight",
                f"{hf}.attention.self.key.bias": f"{ours}.attention.key.bias",
                f"{hf}.attention.self.value.weight": f"{ours}.attention.value.weight",
                f"{hf}.attention.self.value.bias": f"{ours}.attention.value.bias",
                f"{hf}.attention.output.dense.weight": f"{ours}.attention.dense.weight",
                f"{hf}.attention.output.dense.bias": f"{ours}.attention.dense.bias",
                f"{hf}.attention.output.LayerNorm.weight": f"{ours}.attention.LayerNorm.weight",
                f"{hf}.attention.output.LayerNorm.bias": f"{ours}.attention.LayerNorm.bias",
                f"{hf}.intermediate.dense.weight": f"{ours}.mlp.dense_in.weight",
                f"{hf}.intermediate.dense.bias": f"{ours}.mlp.dense_in.bias",
                f"{hf}.output.dense.weight": f"{ours}.mlp.dense_out.weight",
                f"{hf}.output.dense.bias": f"{ours}.mlp.dense_out.bias",
                f"{hf}.output.LayerNorm.weight": f"{ours}.mlp.LayerNorm.weight",
                f"{hf}.output.LayerNorm.bias": f"{ours}.mlp.LayerNorm.bias",
            }
            WEIGHT_MAP.update(layer_map)

        state = model.state_dict()
        mapped = {}
        for hf_name, our_name in WEIGHT_MAP.items():
            if hf_name in weights and our_name in state:
                mapped[our_name] = weights[hf_name]

        model.load_state_dict(mapped, strict=False)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return model, tokenizer
