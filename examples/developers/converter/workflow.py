"""
Workflow - Convert then chat, use destination/output_dir, and force overwrite.

Primary API: talu.Converter, talu.converter.convert
Scope: Single

Related:
- examples/basics/06_convert_model.py
"""

import talu
from pathlib import Path

# =============================================================================
# Convert and immediately use
# =============================================================================

# Convert with platform-aware defaults (recommended)
path = talu.convert("Qwen/Qwen3-0.6B", platform="cpu")  # -> gaf4_64

# Or with explicit scheme
path = talu.convert("Qwen/Qwen3-0.6B", scheme="gaf4_64")

# Use the converted model directly
chat = talu.Chat(path)
response = chat("What is 2 + 2?")
print(response)

# =============================================================================
# Explicit destination (CI/scripts)
# =============================================================================

# Use `destination` for predictable output paths
# No auto-naming - the model is written exactly where you specify
path = talu.convert(
    "Qwen/Qwen3-0.6B",
    scheme="gaf4_64",
    destination="./models/qwen3-0.6b-q4",
)
assert path.endswith("qwen3-0.6b-q4")  # Predictable!

# Perfect for CI pipelines and deployment scripts
path = talu.convert(
    "Qwen/Qwen3-0.6B",
    scheme="gaf4_64",
    destination="/opt/models/production-model",
    force=True,
)

# =============================================================================
# Auto-named output directory
# =============================================================================

# Without destination, Talu auto-generates a name in output_dir
path = talu.convert(
    "Qwen/Qwen3-0.6B",
    scheme="gaf4_64",
    output_dir="./my-models",  # Parent directory
)
print(f"Auto-named: {path}")  # ./my-models/Qwen/Qwen3-0.6B-GAF4_64

# Using the Converter class for multiple conversions
converter = talu.Converter(output_dir="./quantized")

path1 = converter("Qwen/Qwen3-0.6B", scheme="gaf4_64")
path2 = converter("Qwen/Qwen3-0.6B", scheme="gaf8_64")

print(f"4-bit: {path1}")
print(f"8-bit: {path2}")

# =============================================================================
# Force overwrite existing
# =============================================================================

# By default, convert() errors if output already exists
# Use force=True to overwrite
path = talu.convert("Qwen/Qwen3-0.6B", scheme="gaf4_64", force=True)

# Works with destination too
path = talu.convert(
    "Qwen/Qwen3-0.6B",
    scheme="gaf4_64",
    destination="./models/qwen3",
    force=True,
)

"""
Topics covered:
* converter.workflow
* converter.quantize
"""
