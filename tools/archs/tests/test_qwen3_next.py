#!/usr/bin/env python3
"""Test Qwen3-Next pure PyTorch implementation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from qwen.qwen3_next import Qwen3Next

MODEL_ID = "Qwen/Qwen3-Coder-Next-FP8"


def generate(model, tokenizer, prompt, max_tokens=10):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the capital of France?"

    print(f"Loading {MODEL_ID}...")
    model, tokenizer = Qwen3Next.from_pretrained(MODEL_ID)
    model = model.eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    layer_types = [layer.layer_type for layer in model.layers]
    lin_count = layer_types.count("linear_attention")
    full_count = layer_types.count("full_attention")
    print(f"Layers: {len(layer_types)} total ({lin_count} linear, {full_count} full attention)\n")

    generate(model, tokenizer, prompt)


if __name__ == "__main__":
    main()
