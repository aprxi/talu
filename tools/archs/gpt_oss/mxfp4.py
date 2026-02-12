"""
MXFP4 dequantization utilities (pure PyTorch fallback, no external deps).

Used only when the optimized C kernel (mxfp4_kernel.so) is not available.
"""

import torch


_MXFP4_LUT = torch.tensor(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
    dtype=torch.float32,
)


def decode_mxfp4_scales(scales_uint8: torch.Tensor) -> torch.Tensor:
    """Convert uint8 MXFP4 scale exponents to float32 power-of-2 values."""
    return torch.pow(2.0, scales_uint8.float() - 127.0)


def dequantize_mxfp4_block(
    blocks: torch.Tensor,
    scales_f32: torch.Tensor,
) -> torch.Tensor:
    """
    Dequantize MXFP4 weight blocks to float32.

    Args:
        blocks: [out_features, blocks_per_feature, 16] uint8
        scales_f32: [out_features, blocks_per_feature] float32 (pre-decoded)
    """
    out_features, blocks_per_feature, _ = blocks.shape
    low_nibble = (blocks & 0x0F).long()
    high_nibble = (blocks >> 4).long()
    values = torch.stack([low_nibble, high_nibble], dim=-1)
    values = values.reshape(out_features, blocks_per_feature, 32)
    lut = _MXFP4_LUT.to(blocks.device)
    dequant = lut[values]
    dequant = dequant * (scales_f32.unsqueeze(-1) * 0.5)
    return dequant.reshape(out_features, blocks_per_feature * 32)


def mxfp4_linear(x, blocks, scales_f32, bias=None):
    """Linear layer with MXFP4 weights (dequantize then matmul)."""
    weight = dequantize_mxfp4_block(blocks, scales_f32)
    out = torch.nn.functional.linear(x.float(), weight)
    if bias is not None:
        out = out + bias.float()
    return out
