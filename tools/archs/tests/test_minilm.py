#!/usr/bin/env python3
"""
Test MiniLM (BERT) pure PyTorch implementation.

Run from tools/archs/:
    uv run python tests/test_minilm.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from bert.minilm import MiniLM

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def test_forward():
    print(f"Loading {MODEL_ID}...")
    model, tokenizer = MiniLM.from_pretrained(MODEL_ID)
    model.eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Test forward pass
    ids = torch.tensor([tokenizer.encode("Hello world", add_special_tokens=True)])
    with torch.inference_mode():
        hidden_states = model(ids)
    print(f"Hidden states shape: {hidden_states.shape}")
    assert hidden_states.ndim == 3, f"Expected 3D tensor, got {hidden_states.ndim}D"
    assert hidden_states.shape[0] == 1, f"Expected batch=1, got {hidden_states.shape[0]}"
    assert hidden_states.shape[2] == 384, f"Expected d_model=384, got {hidden_states.shape[2]}"

    # Test embedding
    with torch.inference_mode():
        embedding = model.embed(ids)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {embedding.norm().item():.4f}")
    assert embedding.shape == (1, 384), f"Expected (1, 384), got {embedding.shape}"
    assert abs(embedding.norm().item() - 1.0) < 1e-4, "Embedding should be L2-normalized"

    print("\nAll checks passed.")


if __name__ == "__main__":
    test_forward()
