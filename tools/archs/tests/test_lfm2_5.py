#!/usr/bin/env python3
"""
Test LFM2.5 pure PyTorch implementation.

Run from tools/archs/:
    uv run python tests/test_lfm2_5.py
    uv run python tests/test_lfm2_5.py "Your custom prompt"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lfm2.lfm2_5 import LFM2_5

MODEL_ID = "LiquidAI/LFM2.5-1.2B-Thinking"


def generate(model, tokenizer, prompt, max_tokens=30):
    model.eval()

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


def verify_vs_hf(model, tokenizer):
    """Compare logits against HuggingFace reference."""
    print("\n--- Logit comparison vs HuggingFace ---")
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    hf_model.eval()

    prompt = "The capital of France is"
    ids = torch.tensor([tokenizer.encode(prompt)])

    with torch.inference_mode():
        ours = model(ids)
        theirs = hf_model(ids).logits

    diff = (ours - theirs).abs().max().item()
    print(f"Max logit diff: {diff:.6e}")
    assert diff < 1e-3, f"Logit difference too large: {diff}"
    print("PASS")


def main():
    prompt = (
        " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the capital of France?"
    )

    print(f"Loading {MODEL_ID}...")
    model, tokenizer = LFM2_5.from_pretrained(MODEL_ID)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Print layer types
    layer_types = [layer.layer_type for layer in model.layers]
    conv_count = layer_types.count("conv")
    attn_count = layer_types.count("attention")
    print(f"Layers: {len(layer_types)} total ({conv_count} Conv, {attn_count} Attention)\n")

    generate(model, tokenizer, prompt)
    verify_vs_hf(model, tokenizer)


if __name__ == "__main__":
    main()
