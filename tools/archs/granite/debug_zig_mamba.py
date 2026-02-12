"""
Debug Zig Mamba implementation by comparing with PyTorch reference.

This script dumps the weights and intermediate values from PyTorch,
which can be compared with Zig's internal values.
"""

import torch
import numpy as np
import sys
sys.path.insert(0, "/home/anthonyp/repositories/talu/tools/archs")

from granite.granite_hybrid import GraniteHybrid, Mamba2

model_id = "ibm-granite/granite-4.0-h-350m"

print("Loading model...")
model, tokenizer = GraniteHybrid.from_pretrained(model_id)
model.eval()

# Get layer 0 Mamba
mamba = model.layers[0].mixer

# Print weight shapes and first few values
print("\n===== Mamba Weight Debug =====")
print(f"in_proj: {mamba.in_proj.weight.shape}")
print(f"  First row: {mamba.in_proj.weight[0, :5].tolist()}")

print(f"\nconv1d: {mamba.conv1d.weight.shape}")
print(f"  Channel 0: {mamba.conv1d.weight[0, 0, :].tolist()}")

print(f"\nA_log: {mamba.A_log.shape}")
print(f"  Values: {mamba.A_log[:5].tolist()}")

print(f"\nD: {mamba.D.shape}")
print(f"  Values: {mamba.D[:5].tolist()}")

print(f"\ndt_bias: {mamba.dt_bias.shape}")
print(f"  Values: {mamba.dt_bias[:5].tolist()}")

print(f"\nnorm: {mamba.norm.weight.shape}")
print(f"  Values: {mamba.norm.weight[:5].tolist()}")

print(f"\nout_proj: {mamba.out_proj.weight.shape}")
print(f"  First row: {mamba.out_proj.weight[0, :5].tolist()}")

# Test with a simple input
print("\n===== Forward Pass Debug =====")
tokens = tokenizer.encode("Hello")
if hasattr(tokens, 'ids'):
    input_ids = torch.tensor([tokens.ids])
else:
    input_ids = torch.tensor([tokens])

print(f"Input tokens: {input_ids.tolist()}")

with torch.no_grad():
    # Get embedding
    embed = model.embed_tokens(input_ids) * model.embedding_multiplier
    print(f"\nEmbedding after multiplier:")
    print(f"  Shape: {embed.shape}")
    print(f"  Mean: {embed.mean().item():.6f}")
    print(f"  First 5 values: {embed[0, 0, :5].tolist()}")

    # Get layer 0 input norm output
    layer0 = model.layers[0]
    normed = layer0.input_layernorm(embed)
    print(f"\nAfter input_layernorm:")
    print(f"  Shape: {normed.shape}")
    print(f"  Mean: {normed.mean().item():.6f}")
    print(f"  First 5 values: {normed[0, 0, :5].tolist()}")

    # Now trace through Mamba step by step
    print("\n===== Mamba Internal Debug =====")
    x = normed
    B, L, _ = x.shape

    # 1. Input projection
    xz_bc_dt = mamba.in_proj(x)
    print(f"\nin_proj output:")
    print(f"  Shape: {xz_bc_dt.shape}")
    print(f"  Mean: {xz_bc_dt.mean().item():.6f}")

    d_inner = mamba.d_inner
    d_bc = mamba.n_groups * mamba.d_state
    xBC_len = mamba.xBC_len

    z_proj = xz_bc_dt[..., :d_inner]
    xBC = xz_bc_dt[..., d_inner:d_inner + xBC_len]
    dt_proj = xz_bc_dt[..., d_inner + xBC_len:]

    print(f"\nSplit outputs:")
    print(f"  z_proj shape: {z_proj.shape}, mean: {z_proj.mean().item():.6f}")
    print(f"  xBC shape: {xBC.shape}, mean: {xBC.mean().item():.6f}")
    print(f"  dt_proj shape: {dt_proj.shape}, mean: {dt_proj.mean().item():.6f}")

    # 2. Conv1D
    xBC_conv = xBC.transpose(1, 2)
    xBC_conv = mamba.conv1d(xBC_conv)[..., :L]
    print(f"\nAfter conv1d (before silu):")
    print(f"  Mean: {xBC_conv.mean().item():.6f}")

    xBC_conv = torch.nn.functional.silu(xBC_conv)
    print(f"After silu:")
    print(f"  Mean: {xBC_conv.mean().item():.6f}")

    xBC_conv = xBC_conv.transpose(1, 2)

    # Split
    x_conv = xBC_conv[..., :d_inner]
    B_proj = xBC_conv[..., d_inner:d_inner + d_bc]
    C_proj = xBC_conv[..., d_inner + d_bc:]

    print(f"\nAfter conv split:")
    print(f"  x_conv mean: {x_conv.mean().item():.6f}")
    print(f"  B_proj mean: {B_proj.mean().item():.6f}")
    print(f"  C_proj mean: {C_proj.mean().item():.6f}")

    # 3. dt discretization
    dt = torch.nn.functional.softplus(dt_proj + mamba.dt_bias)
    print(f"\ndt (after softplus):")
    print(f"  Mean: {dt.mean().item():.6f}")
    print(f"  First 5: {dt[0, 0, :5].tolist()}")

    # Full Mamba forward
    mamba_out = mamba(normed)
    print(f"\nFull Mamba output:")
    print(f"  Shape: {mamba_out.shape}")
    print(f"  Mean: {mamba_out.mean().item():.6f}")
    print(f"  First 5: {mamba_out[0, 0, :5].tolist()}")

    # Full layer 0 output
    layer0_out = layer0(embed)
    print(f"\nFull layer 0 output:")
    print(f"  Shape: {layer0_out.shape}")
    print(f"  Mean: {layer0_out.mean().item():.6f}")

    # Final logits
    logits = model(input_ids)
    print(f"\nFinal logits:")
    print(f"  Shape: {logits.shape}")
    print(f"  Mean: {logits.mean().item():.6f}")
    print(f"  Argmax: {logits[0, -1].argmax().item()}")
