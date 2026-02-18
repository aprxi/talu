//! CPU compute primitives shared by inference kernels.
//!
//! This package contains CPU-specific low-level kernels and helpers that are
//! called from higher-level inference backend code.

pub const common = @import("common.zig");
pub const tensor_view = @import("tensor_view.zig");
pub const activation_view = @import("activation_view.zig");
pub const norm_primitives = @import("norm_primitives.zig");
pub const transpose = @import("transpose.zig");
pub const attn_primitives = @import("attn_primitives.zig");
pub const matmul_primitives = @import("matmul_primitives.zig");
pub const matmul_prefill = @import("matmul_prefill.zig");
pub const mxfp4 = @import("mxfp4.zig");
pub const math_primitives = @import("math_primitives/root.zig");
pub const simd = @import("simd/root.zig");
pub const matmul = matmul_primitives;
pub const norm = norm_primitives;
pub const attention = attn_primitives;
pub const math = math_primitives;
pub const activation = @import("activation.zig");
pub const elementwise = @import("elementwise.zig");
pub const broadcast = @import("broadcast.zig");
pub const normalization = @import("normalization.zig");
pub const rowwise = @import("rowwise.zig");
pub const layout_transform = @import("layout_transform.zig");
pub const masking = @import("masking.zig");
pub const image_ops = @import("image_ops.zig");
pub const tensor_copy = @import("tensor_copy.zig");
pub const tensor_gather = @import("tensor_gather.zig");
pub const padding = @import("padding.zig");
pub const dot_product = @import("dot_product.zig");
pub const quant_decode = @import("quant_decode.zig");
pub const cache_layout = @import("cache_layout.zig");
pub const cache_store = @import("cache_store.zig");
pub const rotary = @import("rotary.zig");
pub const conv1d_depthwise = @import("conv1d_depthwise.zig");
pub const state_space = @import("state_space.zig");
pub const matvec = @import("matvec.zig");
pub const topk = @import("topk.zig");
pub const reduction = @import("reduction.zig");
pub const softmax = @import("softmax.zig");
pub const sampling_ops = @import("sampling_ops.zig");
pub const sdpa_decode = @import("sdpa_decode.zig");

test {
    _ = common;
    _ = tensor_view;
    _ = activation_view;
    _ = norm_primitives;
    _ = transpose;
    _ = attn_primitives;
    _ = matmul_primitives;
    _ = matmul_prefill;
    _ = mxfp4;
    _ = math_primitives;
    _ = simd;
    _ = matmul;
    _ = norm;
    _ = attention;
    _ = math;
    _ = activation;
    _ = elementwise;
    _ = broadcast;
    _ = normalization;
    _ = rowwise;
    _ = layout_transform;
    _ = masking;
    _ = image_ops;
    _ = tensor_copy;
    _ = tensor_gather;
    _ = padding;
    _ = dot_product;
    _ = quant_decode;
    _ = cache_layout;
    _ = cache_store;
    _ = rotary;
    _ = conv1d_depthwise;
    _ = state_space;
    _ = matvec;
    _ = topk;
    _ = reduction;
    _ = softmax;
    _ = sampling_ops;
    _ = sdpa_decode;
}
