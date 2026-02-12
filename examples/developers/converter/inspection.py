"""
Model Inspection - Use describe() to analyze models without loading weights.

Primary API: talu.Converter, talu.converter.describe
Scope: Single

The describe() function reads model configuration without loading weights,
making it fast and memory-efficient for pre-flight checks.

Related:
- examples/basics/15_model_conversion.py
"""

import talu
from talu.converter import describe
from talu.exceptions import ModelError

# =============================================================================
# Basic inspection
# =============================================================================

# Get model info without loading weights
info = describe("Qwen/Qwen3-0.6B")

print(f"Model: {info.model_type}")
print(f"Architecture: {info.architecture}")
print(f"Layers: {info.num_layers}")
print(f"Hidden size: {info.hidden_size}")
print(f"Attention heads: {info.num_heads}")
print(f"KV heads: {info.num_kv_heads}")
print(f"Vocab size: {info.vocab_size}")
print(f"Max sequence length: {info.max_seq_len}")

# =============================================================================
# Check quantization status
# =============================================================================

# Before conversion - check if already quantized
info = describe("Qwen/Qwen3-0.6B")

if info.is_quantized:
    print(f"Already quantized: {info.quant_bits}-bit")
else:
    print("Full precision (FP16) - can be quantized")

# After conversion - verify quantization
path = talu.convert("Qwen/Qwen3-0.6B", scheme="gaf4_64")
info_quantized = describe(path)

print(f"Quantized: {info_quantized.is_quantized}")  # True
print(f"Bits: {info_quantized.quant_bits}")  # 4
print(f"Group size: {info_quantized.quant_group_size}")  # 64

# =============================================================================
# Detect MoE (Mixture of Experts) models
# =============================================================================

# Some models use MoE architecture
info = describe("mistralai/Mixtral-8x7B-v0.1")

if info.is_moe:
    print(f"MoE model with {info.num_experts} experts")
    print(f"Experts per token: {info.experts_per_token}")
    # MoE models benefit from per-tensor quantization overrides
else:
    print("Dense model (not MoE)")

# =============================================================================
# Pre-conversion size estimation
# =============================================================================


def estimate_size_gb(info, bits: int = 4) -> float:
    """Estimate quantized model size based on architecture."""
    # Rough formula: params * bits / 8 / 1e9
    # For transformers: ~12 * hidden^2 * layers params
    params = 12 * info.hidden_size**2 * info.num_layers
    return params * bits / 8 / 1e9


info = describe("Qwen/Qwen3-0.6B")
print(f"Estimated 4-bit size: {estimate_size_gb(info, 4):.1f} GB")
print(f"Estimated 8-bit size: {estimate_size_gb(info, 8):.1f} GB")

# =============================================================================
# Compare models before choosing
# =============================================================================

models = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
]

print("\nModel comparison:")
print(f"{'Model':<25} {'Layers':>8} {'Hidden':>8} {'Heads':>8}")
print("-" * 55)

for model_id in models:
    try:
        info = describe(model_id)
        print(f"{model_id:<25} {info.num_layers:>8} {info.hidden_size:>8} {info.num_heads:>8}")
    except ModelError:
        print(f"{model_id:<25} (not available)")

# =============================================================================
# Using with Converter class
# =============================================================================

from talu.converter import Converter, describe, ModelInfo

# The Converter can use describe internally for validation
converter = Converter()

# Describe a model before conversion
info = describe("Qwen/Qwen3-0.6B")
print(f"\nPre-conversion check:")
print(f"  Type: {info.model_type}")
print(f"  Already quantized: {info.is_quantized}")

# Convert if not already quantized
if not info.is_quantized:
    path = converter("Qwen/Qwen3-0.6B", scheme="gaf4_64")
    print(f"  Converted to: {path}")

"""
Topics covered:
* converter.workflow
* repository.inspect
"""
