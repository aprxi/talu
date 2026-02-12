#!/usr/bin/env python3
"""
Test Granite Hybrid pure PyTorch implementation.

Run from models/:
    uv run python tests/test_granite_hybrid.py
    uv run python tests/test_granite_hybrid.py "Your custom prompt"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from granite.granite_hybrid import GraniteHybrid

MODEL_ID = "ibm-granite/granite-4.0-h-350m"


def generate(model, tokenizer, prompt, max_tokens=10):
    model.eval()

    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids = torch.tensor([tokenizer.encode(text)])

    with torch.inference_mode():
        for _ in range(max_tokens):
            logits = model(ids)
            next_id = logits[0, -1].argmax().item()
            if next_id == tokenizer.eos_token_id:
                break
            print(tokenizer.decode([next_id]), end="", flush=True)
            ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)
    print()


def main():
    prompt = (
        " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the capital of France?"
    )

    print(f"Loading {MODEL_ID}...")
    model, tokenizer = GraniteHybrid.from_pretrained(MODEL_ID)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Print layer types
    layer_types = [layer.layer_type for layer in model.layers]
    mamba_count = layer_types.count("mamba")
    attn_count = layer_types.count("attention")
    print(f"Layers: {len(layer_types)} total ({mamba_count} Mamba, {attn_count} Attention)\n")

    generate(model, tokenizer, prompt)


if __name__ == "__main__":
    main()
