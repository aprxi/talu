#!/usr/bin/env python3
"""
Test Phi4 pure PyTorch implementation.

Run from models/:
    uv run python tests/test_phi4.py
    uv run python tests/test_phi4.py "Your custom prompt"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from phi.phi4 import Phi4

MODEL_ID = "microsoft/Phi-4-mini-instruct"


def generate(model, tokenizer, prompt, max_tokens=10):
    model.eval()

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
    model, tokenizer = Phi4.from_pretrained(MODEL_ID)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    generate(model, tokenizer, prompt)


if __name__ == "__main__":
    main()
