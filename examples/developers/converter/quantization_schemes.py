"""
Quantization - Select schemes with platform/quant or explicit scheme names.

Primary API: talu.convert, talu.list_schemes
Scope: Single

Two ways to specify quantization:

1. **Platform-aware** (recommended): Let Talu choose the optimal format
   - `platform="cpu"` + `quant="4bit"` -> gaf4_64
   - `platform="metal"` + `quant="4bit"` -> gaf4_64

2. **Explicit scheme**: Use scheme aliases or canonical names
   - "4bit" / "q4" / "int4" -> gaf4_64 (balanced 4-bit)
   - "mlx" / "mlx4" / "gaf4" -> gaf4_64 (all platforms)

Related:
- examples/basics/06_convert_model.py
"""

import talu

# =============================================================================
# Platform-Based Conversion (Recommended)
# =============================================================================

# Specify your target platform to get the appropriate format:
# - cpu/cuda -> GAF
# - metal -> GAF
path_cpu = talu.convert("Qwen/Qwen3-0.6B", platform="cpu")  # -> gaf4_64
path_metal = talu.convert("Qwen/Qwen3-0.6B", platform="metal")  # -> gaf4_64

# Specify quantization level with platform
path_cpu_8bit = talu.convert("Qwen/Qwen3-0.6B", platform="cpu", quant="8bit")  # -> gaf8_64
path_metal_8bit = talu.convert("Qwen/Qwen3-0.6B", platform="metal", quant="8bit")  # -> gaf8_64

# Platform aliases
path_mps = talu.convert("Qwen/Qwen3-0.6B", platform="mps")  # same as "metal"
path_apple = talu.convert("Qwen/Qwen3-0.6B", platform="apple")  # same as "metal"
path_cuda = talu.convert("Qwen/Qwen3-0.6B", platform="cuda")  # -> gaf4_64

# =============================================================================
# Explicit Scheme Selection (User-Friendly Aliases)
# =============================================================================

# Use simple aliases for common use cases
path_4bit = talu.convert("Qwen/Qwen3-0.6B", scheme="4bit")  # -> gaf4_64
path_8bit = talu.convert("Qwen/Qwen3-0.6B", scheme="8bit")  # -> gaf8_64
path_mlx = talu.convert("Qwen/Qwen3-0.6B", scheme="mlx")  # -> gaf4_64

# =============================================================================
# Grouped Affine (GAF) - Default format for all platforms
# =============================================================================

# gaf4_64: 4-bit with group_size=64 (default, recommended)
# Alias: "4bit", "q4", "int4", "mlx", "mlx4", "gaf4"
path_gaf4 = talu.convert("Qwen/Qwen3-0.6B", scheme="gaf4_64")

# gaf4_32: 4-bit with group_size=32 (highest quality, larger)
path_gaf4_32 = talu.convert("Qwen/Qwen3-0.6B", scheme="gaf4_32")

# gaf4_128: 4-bit with group_size=128 (smallest, lower quality)
path_gaf4_128 = talu.convert("Qwen/Qwen3-0.6B", scheme="gaf4_128")

# gaf8_64: 8-bit with group_size=64 (near-lossless)
# Alias: "8bit", "mlx8", "gaf8"
path_gaf8 = talu.convert("Qwen/Qwen3-0.6B", scheme="gaf8_64")

# gaf8_32: 8-bit with group_size=32 (highest 8-bit quality)
path_gaf8_32 = talu.convert("Qwen/Qwen3-0.6B", scheme="gaf8_32")

# gaf8_128: 8-bit with group_size=128 (smallest 8-bit)
path_gaf8_128 = talu.convert("Qwen/Qwen3-0.6B", scheme="gaf8_128")

# =============================================================================
# List available schemes and their aliases
# =============================================================================

print("Available quantization schemes:")
for name, info in talu.list_schemes().items():
    aliases = info.get("aliases", [])
    alias_str = f" (aliases: {', '.join(aliases)})" if aliases else ""
    print(f"  {name}: {info['description']}{alias_str}")

# Include unimplemented schemes (for reference)
print("\nAll schemes (including unimplemented):")
for name, info in talu.list_schemes(include_unimplemented=True).items():
    status = info.get("status", "stable")
    aliases = info.get("aliases", [])
    alias_str = f" [{', '.join(aliases)}]" if aliases else ""
    print(f"  {name}: [{status}] {info['description']}{alias_str}")

"""
Topics covered:
* converter.quantize
* converter.schemes
"""
