//! CPU compute primitives shared by inference kernels.
//!
//! This package contains CPU-specific low-level kernels and helpers that are
//! called from higher-level inference backend code.

pub const common = @import("common.zig");
pub const tensor_view = @import("tensor_view.zig");
pub const activation_view = @import("activation_view.zig");
pub const attn_primitives = @import("attn_primitives.zig");
pub const mxfp4 = @import("mxfp4.zig");
pub const math_primitives = @import("math_primitives/root.zig");
pub const simd = @import("simd/root.zig");
pub const quant = @import("quant/root.zig");
pub const capabilities = @import("capabilities.zig");

// Primitive-first namespaces
pub const linalg = @import("linalg.zig");
pub const layout = @import("layout.zig");
pub const memory = @import("memory.zig");
pub const cache = @import("cache.zig");
pub const recurrence = @import("recurrence.zig");

pub const activation = @import("activation.zig");
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
pub const sdpa_decode = @import("sdpa_decode.zig");

test {
    _ = common;
    _ = tensor_view;
    _ = activation_view;
    _ = attn_primitives;
    _ = mxfp4;
    _ = math_primitives;
    _ = simd;
    _ = quant;
    _ = capabilities;
    _ = linalg;
    _ = layout;
    _ = memory;
    _ = cache;
    _ = recurrence;
    _ = activation;
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
    _ = sdpa_decode;
}
