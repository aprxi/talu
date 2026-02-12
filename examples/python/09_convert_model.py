"""Convert a model for local use.

This example shows:
- Converting models to quantized formats
- Available quantization schemes
- Verification
"""

import os
import sys
from pathlib import Path

import talu
from talu import repository

MODEL_URI = os.environ.get("MODEL_URI", "LiquidAI/LFM2-350M")

if not repository.is_cached(MODEL_URI):
    sys.exit(f"Model '{MODEL_URI}' not found. Run: python examples/python/00_fetch_model.py")


# =============================================================================
# Basic conversion (4-bit quantized, default scheme)
# =============================================================================

# Default scheme is gaf4_64 — good balance of size and quality
# Output lands next to the original: LiquidAI/LFM2-350M -> LiquidAI/LFM2-350M-GAF4
path = talu.convert(MODEL_URI, scheme="gaf4_64")
print(f"Converted model: {path}")

# The converted model is ready to use
chat = talu.Chat(path)
print(chat("What is the capital of France?"))


# =============================================================================
# Available quantization schemes
# =============================================================================

# 4-bit schemes (smaller, slightly less accurate)
#   gaf4_32  — 4-bit, group size 32 (highest accuracy among 4-bit)
#   gaf4_64  — 4-bit, group size 64 (default, balanced)
#   gaf4_128 — 4-bit, group size 128 (smallest)
#
# 8-bit schemes (larger, more accurate)
#   gaf8_32  — 8-bit, group size 32
#   gaf8_64  — 8-bit, group size 64
#   gaf8_128 — 8-bit, group size 128

# Convert with 8-bit for higher accuracy
path_8bit = talu.convert(MODEL_URI, scheme="gaf8_64")
print(f"8-bit model: {path_8bit}")

# Aliases work too: "4bit" -> gaf4_64, "8bit" -> gaf8_64
path_alias = talu.convert(MODEL_URI, scheme="4bit")
print(f"Via alias: {path_alias}")


# =============================================================================
# Verification
# =============================================================================

# Verify after conversion (loads model, generates a few tokens)
verified = talu.convert(MODEL_URI, scheme="gaf4_64", verify=True)
print(f"Verified conversion: {verified}")


# =============================================================================
# Convert from a local path
# =============================================================================

local_path = Path("./my-downloaded-model")
if local_path.exists():
    path_local = talu.convert(str(local_path), scheme="gaf4_64")
    print(f"Local model: {path_local}")
