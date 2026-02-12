#!/usr/bin/env python3
"""Test Youtu-VL pure PyTorch implementation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from youtu_vl.youtu_vl import YoutuVL, YoutuVLLanguageModel

MODEL_ID = "tencent/Youtu-VL-4B-Instruct"


def load_chat_template(model_id):
    """Load chat template from separate chat_template.json file."""
    import json
    from huggingface_hub import hf_hub_download

    try:
        template_path = hf_hub_download(model_id, "chat_template.json")
        with open(template_path) as f:
            return json.load(f)["chat_template"]
    except Exception:
        return None


def generate(model, tokenizer, prompt, max_tokens=20, chat_template=None):
    """Generate text from the language model."""
    model.eval()

    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    if chat_template:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=chat_template,
        )
    elif tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt

    ids = torch.tensor([tokenizer.encode(text)])

    print(f"Input tokens: {ids.shape[1]}")
    print(f"Generating up to {max_tokens} tokens...")

    with torch.inference_mode():
        for i in range(max_tokens):
            # Forward pass (text-only, no images)
            if hasattr(model, "model"):
                # Full YoutuVL model
                logits = model(ids)
            else:
                # Language model only
                logits = model(ids)

            next_id = logits[0, -1].argmax().item()

            # Check for EOS
            if next_id == tokenizer.eos_token_id:
                print(" [EOS]")
                break

            # Decode and print token
            token_str = tokenizer.decode([next_id])
            print(token_str, end="", flush=True)

            # Append token
            ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)

    print()
    return ids


def main():
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is 2+2?"

    print(f"Loading {MODEL_ID}...")
    print("(This may take a few minutes to download ~10GB of weights)")

    model, tokenizer, processor = YoutuVL.from_pretrained(MODEL_ID)

    # Load chat template from separate file
    chat_template = load_chat_template(MODEL_ID)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print()

    print(f"Prompt: {prompt}")
    print("-" * 40)
    print("Response: ", end="")

    generate(model, tokenizer, prompt, chat_template=chat_template)


if __name__ == "__main__":
    main()
