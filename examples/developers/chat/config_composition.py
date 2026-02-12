"""Config Composition - Merge GenerationConfigs with the pipe operator.

Primary API: talu.GenerationConfig
Scope: Single

This example demonstrates how to compose GenerationConfigs using the pipe (|)
operator. This enables clean separation of concerns - define sampling params,
stopping criteria, and formatting separately, then merge them as needed.

Key points:
- Use config1 | config2 to merge (right side wins for conflicts)
- Non-default values override, defaults are preserved
- Original configs are not mutated
- Chain multiple configs: a | b | c

Related:
    - examples/developers/chat/configuration.py (basic config)
    - examples/developers/chat/structured_output.py (response_format)
"""

from talu import Chat
from talu.router import GenerationConfig


# =============================================================================
# Basic Composition
# =============================================================================

print("=== Basic Composition ===")

# Define reusable config "presets"
creative = GenerationConfig(temperature=1.2, top_p=0.95)
precise = GenerationConfig(temperature=0.1, top_k=10)
long_form = GenerationConfig(max_tokens=1000)
short_form = GenerationConfig(max_tokens=50)

# Merge configs with pipe operator
creative_long = creative | long_form
print(f"creative_long: temp={creative_long.temperature}, max_tokens={creative_long.max_tokens}")

precise_short = precise | short_form
print(f"precise_short: temp={precise_short.temperature}, max_tokens={precise_short.max_tokens}")


# =============================================================================
# Right Side Wins for Conflicts
# =============================================================================

print("\n=== Right Side Wins ===")

left = GenerationConfig(temperature=0.5, max_tokens=100)
right = GenerationConfig(temperature=1.0)  # Only temperature set

merged = left | right
print(f"left.temperature = {left.temperature}")
print(f"right.temperature = {right.temperature}")
print(f"merged.temperature = {merged.temperature}")  # 1.0 (right wins)
print(f"merged.max_tokens = {merged.max_tokens}")    # 100 (left preserved)


# =============================================================================
# Defaults Are Preserved
# =============================================================================

print("\n=== Defaults Preserved ===")

custom = GenerationConfig(temperature=0.7, top_k=40)
empty = GenerationConfig()  # All defaults

merged = custom | empty
print(f"custom: temp={custom.temperature}, top_k={custom.top_k}")
print(f"merged: temp={merged.temperature}, top_k={merged.top_k}")
# Custom values preserved because empty has all defaults


# =============================================================================
# Chaining Multiple Configs
# =============================================================================

print("\n=== Chaining ===")

# Define aspect-specific configs
sampling = GenerationConfig(temperature=0.7, top_k=40)
limits = GenerationConfig(max_tokens=500)
stops = GenerationConfig(stop_sequences=["END", "STOP"])

# Merge all three
full_config = sampling | limits | stops

print(f"temperature: {full_config.temperature}")
print(f"top_k: {full_config.top_k}")
print(f"max_tokens: {full_config.max_tokens}")
print(f"stop_sequences: {full_config.stop_sequences}")


# =============================================================================
# Originals Unchanged
# =============================================================================

print("\n=== Immutability ===")

config_a = GenerationConfig(temperature=0.5)
config_b = GenerationConfig(temperature=1.0)

merged = config_a | config_b

print(f"config_a.temperature after merge: {config_a.temperature}")  # Still 0.5
print(f"config_b.temperature after merge: {config_b.temperature}")  # Still 1.0
print(f"merged.temperature: {merged.temperature}")  # 1.0
print(f"merged is config_a: {merged is config_a}")  # False
print(f"merged is config_b: {merged is config_b}")  # False


# =============================================================================
# Practical Use Cases
# =============================================================================

print("\n=== Practical Use Cases ===")

# Use case 1: A/B testing different sampling strategies
sampling_a = GenerationConfig(temperature=0.7)
sampling_b = GenerationConfig(temperature=1.2, top_p=0.9)

base_config = GenerationConfig(max_tokens=100, stop_sequences=["\n\n"])

test_a = base_config | sampling_a
test_b = base_config | sampling_b

print("Test A config:", test_a.temperature, test_a.max_tokens)
print("Test B config:", test_b.temperature, test_b.max_tokens)


# Use case 2: Environment-specific overrides
production = GenerationConfig(temperature=0.7, max_tokens=256)
debug_mode = GenerationConfig(max_tokens=50, seed=42)  # Short, reproducible

# In debug mode, override production settings
config = production | debug_mode if True else production  # Simulating debug flag
print(f"Debug config: max_tokens={config.max_tokens}, seed={config.seed}")


# Use case 3: Per-request customization
session_default = GenerationConfig(temperature=0.7, max_tokens=200)
user_preference = GenerationConfig(temperature=1.0)  # User wants more creative

effective = session_default | user_preference
print(f"Effective config: temp={effective.temperature}, max_tokens={effective.max_tokens}")


# =============================================================================
# With Chat
# =============================================================================

print("\n=== With Chat (conceptual) ===")

# In real usage:
# chat = Chat("Qwen/Qwen3-0.6B", config=session_default)
# response = chat("Write a poem", config=session_default | creative)

# Or use override() for the same effect with kwargs:
# response = chat("Write a poem", temperature=1.2)

creative_thinking = GenerationConfig(
    temperature=1.0,
    allow_thinking=True,
    max_thinking_tokens=256,
)

json_output = GenerationConfig(
    temperature=0.3,
    stop_sequences=["}"],
)

# Combine for creative JSON generation with thinking
creative_json = creative_thinking | json_output
print(f"Creative JSON: temp={creative_json.temperature}, thinking={creative_json.allow_thinking}")


"""
Topics covered:
* config.composition
* config.merge
* config.pipe
* config.presets

Related:
* examples/developers/chat/configuration.py
* examples/basics/03_configuration.py
"""
