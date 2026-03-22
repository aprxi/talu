//! CPU compute primitives.
//!
//! This package contains CPU-specific low-level kernels and helpers that are
//! called from higher-level runtime orchestration code.

pub const common = @import("common.zig");
pub const tensor_view = @import("tensor_view.zig");
pub const activation_view = @import("activation_view.zig");
pub const mxfp4 = @import("mxfp4.zig");
pub const math = @import("math.zig");
pub const simd = @import("simd/root.zig");
pub const quant = @import("quant/root.zig");
pub const capabilities = @import("capabilities.zig");

// Primitive-first namespaces
pub const linalg = @import("linalg.zig");
pub const layout = @import("layout.zig");
pub const memory = @import("memory.zig");
pub const recurrence = @import("recurrence.zig");
pub const indexing = @import("indexing.zig");

pub const activation = @import("activation.zig");
pub const gated_attention = @import("gated_attention.zig");
pub const gated_delta = @import("gated_delta.zig");
pub const elementwise = @import("elementwise.zig");
pub const normalization = @import("normalization.zig");
pub const rowwise = @import("rowwise.zig");
pub const image_ops = @import("image_ops.zig");
pub const quant_decode = @import("quant_decode.zig");
pub const rotary = @import("rotary.zig");
pub const conv1d_depthwise = @import("conv1d_depthwise.zig");
pub const topk = @import("topk.zig");
pub const reduction = @import("reduction.zig");
pub const softmax = @import("softmax.zig");
pub const sampling_ops = @import("sampling_ops.zig");
pub const sdpa_rowwise = @import("sdpa_rowwise.zig");
pub const linalg_sdpa = @import("linalg_sdpa.zig");
pub const state_space = @import("state_space.zig");
pub const matmul = @import("matmul_primitives.zig");
pub const accelerate = @import("accelerate.zig");

test {
    _ = common;
    _ = tensor_view;
    _ = activation_view;
    _ = mxfp4;
    _ = math;
    _ = simd;
    _ = quant;
    _ = capabilities;
    _ = linalg;
    _ = layout;
    _ = memory;
    _ = recurrence;
    _ = indexing;
    _ = activation;
    _ = gated_attention;
    _ = gated_delta;
    _ = elementwise;
    _ = normalization;
    _ = rowwise;
    _ = image_ops;
    _ = quant_decode;
    _ = rotary;
    _ = conv1d_depthwise;
    _ = topk;
    _ = reduction;
    _ = softmax;
    _ = sampling_ops;
    _ = sdpa_rowwise;
    _ = linalg_sdpa;
    _ = state_space;
    _ = matmul;
    _ = accelerate;
}
