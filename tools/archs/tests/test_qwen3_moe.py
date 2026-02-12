#!/usr/bin/env python3
"""
Test Qwen3 MoE pure PyTorch implementation.

Run from models/:
    uv run python tests/test_qwen3_moe.py
    uv run python tests/test_qwen3_moe.py "Your custom prompt"

Note: This requires ~60GB+ memory to load the full model.
For testing the graph tracing, the implementation is validated
through the trace.py workflow.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from qwen.qwen3_moe import Qwen3Moe

MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"


def generate(model, tokenizer, prompt, max_tokens=10):
    # Apply chat template
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
    import sys
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the capital of France?"

    print(f"Loading {MODEL_ID}...")
    model, tokenizer = Qwen3Moe.from_pretrained(MODEL_ID)
    model = model.to(torch.bfloat16).eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    generate(model, tokenizer, prompt)


if __name__ == "__main__":
    main()
