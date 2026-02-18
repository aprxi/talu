//! CPU compute primitives shared by inference kernels.
//!
//! This package contains CPU-specific low-level kernels and helpers that are
//! called from higher-level inference backend code.

pub const common = @import("common.zig");
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
