#!/usr/bin/env python3
"""
Test GPT-OSS pure PyTorch implementation.

This model is ~20B parameters with MXFP4 quantized experts.
Loading requires ~12GB memory and takes several minutes.

Run from tools/archs/:
    uv run python tests/test_gpt_oss.py
    uv run python tests/test_gpt_oss.py "Your custom prompt"
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from gpt_oss.gpt_oss import GptOss

MODEL_ID = "openai/gpt-oss-20b"

# Use HF_HOME-relative path if model is already downloaded, else download
_HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
_LOCAL_PATH = _HF_CACHE / "hub" / "models--openai--gpt-oss-20b" / "snapshots" / "main"

# EOS token IDs from generation_config.json: <|return|>, <|endoftext|>, <|call|>
_EOS_IDS = {200002, 199999, 200012}


def generate(model, tokenizer, prompt, max_tokens=200):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = torch.tensor([tokenizer.encode(text)])

    with torch.inference_mode():
        for _ in range(max_tokens):
            logits = model(ids)
            next_id = logits[0, -1].argmax().item()
            if next_id in _EOS_IDS:
                break
            token = tokenizer.decode([next_id], skip_special_tokens=True)
            print(token, end="", flush=True)
            ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)
    print()


def main():
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the capital of France?"
    model_id = str(_LOCAL_PATH) if _LOCAL_PATH.is_dir() else MODEL_ID

    print(f"Loading {model_id}...")
    model, tokenizer = GptOss.from_pretrained(model_id)
    model = model.eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    generate(model, tokenizer, prompt)


if __name__ == "__main__":
    main()
