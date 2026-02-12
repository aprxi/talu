"""
Chat Formats - Use from_preset() for different chat template formats.

Primary API: talu.PromptTemplate, talu.template.PromptTemplate
Scope: Single

Different LLMs expect different chat formats. PromptTemplate.from_preset()
provides built-in templates for common formats, making it easy to switch
between models.

Related:
- examples/developers/template/chat_templates.py
"""

import talu

# ChatML format (Qwen, OpenChat, Hermes, etc.)
chatml = talu.PromptTemplate.from_preset("chatml")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
    {"role": "user", "content": "Give me an example."},
]

print("=== ChatML Format ===")
print(chatml.apply(messages))

# Llama 2 format
llama2 = talu.PromptTemplate.from_preset("llama2")
print("=== Llama 2 Format ===")
print(llama2.apply(messages, bos_token="<s>", eos_token="</s>"))

# Alpaca instruction format
alpaca = talu.PromptTemplate.from_preset("alpaca")
print("=== Alpaca Format ===")
print(alpaca.apply([
    {"role": "user", "content": "Explain machine learning in one sentence."},
]))

# Vicuna format
vicuna = talu.PromptTemplate.from_preset("vicuna")
print("=== Vicuna Format ===")
print(vicuna.apply(messages))

# Training data (no generation prompt at end)
print("=== Training Data (no generation prompt) ===")
training_data = chatml.apply(
    [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ],
    add_generation_prompt=False,
)
print(training_data)

# Inference (add generation prompt)
print("=== Inference (with generation prompt) ===")
inference_prompt = chatml.apply(
    [{"role": "user", "content": "Hi"}],
    add_generation_prompt=True,
)
print(inference_prompt)

# PromptTemplate has all template features
print(f"\n=== Template Features ===")
print(f"Input variables: {chatml.input_variables}")
print(f"Source preview: {chatml.source[:50]}...")

# Check template capabilities
# Useful for Chat implementations to decide how to handle
# system messages and tool definitions
print("\n=== Template Capabilities ===")
for name in ["chatml", "llama2", "alpaca", "vicuna", "zephyr"]:
    t = talu.PromptTemplate.from_preset(name)
    print(f"{name}: system={t.supports_system_role}, tools={t.supports_tools}")

# Note: apply() is convenience syntax for __call__()
# These two are equivalent:
print("\n=== apply() vs __call__() ===")
prompt1 = chatml.apply(messages)
prompt2 = chatml(messages=messages, add_generation_prompt=True)
print(f"Results equal: {prompt1 == prompt2}")

"""
Topics covered:
* chat.templates
* messages.format
"""
