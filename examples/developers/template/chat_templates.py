"""
Chat Templates - Load and render chat templates from model configs.

Primary API: talu.PromptTemplate, talu.template.PromptTemplate
Scope: Single

Each model has its own chat template in tokenizer_config.json.
Load and inspect these templates to understand how models expect input.

Related:
- examples/developers/template/chat_formats.py
"""

import json
import tempfile
from pathlib import Path

import talu

# Create a mock model directory (in production, use real model paths)
with tempfile.TemporaryDirectory() as model_dir:
    # Simulate a Qwen-style chat template
    config = {
        "chat_template": (
            "{%- for message in messages %}"
            "<|im_start|>{{ message.role }}\n"
            "{{ message.content }}<|im_end|>\n"
            "{%- endfor %}"
            "{%- if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        ),
    }
    Path(model_dir, "tokenizer_config.json").write_text(json.dumps(config))

    # Load template from model
    template = talu.PromptTemplate.from_chat_template(model_dir)

    print("=== Inspect Model Template ===")
    print(f"Source:\n{template.source}\n")
    print(f"Variables: {template.input_variables}")

    # Render chat messages - two equivalent ways:
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]

    print("\n=== Render with apply() ===")
    prompt = template.apply(messages)
    print(prompt)

    print("\n=== Equivalent: direct call ===")
    # apply() is just convenience for this:
    prompt = template(messages=messages, add_generation_prompt=True)
    print(prompt)

    # Customize a model's template
    print("\n=== Customize Template ===")
    custom_source = template.source.replace("assistant", "AI")
    custom = talu.PromptTemplate(custom_source)
    print(f"Modified source:\n{custom.source}")

# With a real model (requires downloaded model):
# template = talu.PromptTemplate.from_chat_template("Qwen/Qwen3-0.6B")
# print(template.source)

"""
Topics covered:
* chat.templates
* template.render
"""
