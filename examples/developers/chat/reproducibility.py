"""Reproducibility - Get deterministic outputs with seed.

Primary API: talu.Chat
Scope: Single

When temperature > 0, the model samples randomly from token probabilities.
Setting a seed makes that randomness reproducible.
"""

import talu
from talu.router import GenerationConfig

# Without seed: same prompt gives different outputs
chat = talu.Chat("Qwen/Qwen3-0.6B")
response1 = chat("Write a one-line poem about the sea", temperature=0.8)
response2 = chat("Write a one-line poem about the sea", temperature=0.8)
print(f"Without seed:")
print(f"  Run 1: {response1}")
print(f"  Run 2: {response2}")
print(f"  Same? {str(response1) == str(response2)}")
print()

# With seed: same prompt gives identical outputs
config = GenerationConfig(seed=42, temperature=0.8, max_tokens=50)
chat1 = talu.Chat("Qwen/Qwen3-0.6B", config=config)
chat2 = talu.Chat("Qwen/Qwen3-0.6B", config=config)
response1 = chat1("Write a one-line poem about the sea")
response2 = chat2("Write a one-line poem about the sea")
print(f"With seed=42:")
print(f"  Run 1: {response1}")
print(f"  Run 2: {response2}")
print(f"  Same? {str(response1) == str(response2)}")
print()

# Use case: A/B test prompts fairly
# Same seed ensures differences come from the prompt, not randomness
prompt_a = "Explain quantum computing in one sentence."
prompt_b = "Explain quantum computing simply in one sentence."

for name, prompt in [("A", prompt_a), ("B", prompt_b)]:
    chat = talu.Chat("Qwen/Qwen3-0.6B", config=GenerationConfig(seed=123, temperature=0.7))
    response = chat(prompt)
    print(f"Prompt {name}: {response}")

"""
Topics covered:
* generation.parameters
* config.sampling
"""
