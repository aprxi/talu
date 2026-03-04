//! Backward pass kernels for training.
//!
//! Each sub-module implements the backward pass for a specific operation type.
//! All backward operations work on f32 gradients regardless of forward-pass dtype.
//!
//! Kernel contract: backward functions accumulate into gradient buffers (they do
//! NOT zero the buffer first). Call GradTensor.zero() before each training step.

const std = @import("std");

pub const linear = @import("linear.zig");
pub const cross_entropy = @import("cross_entropy.zig");
pub const embedding = @import("embedding.zig");

pub const rmsnorm = @import("rmsnorm.zig");
pub const activation = @import("activation.zig");
pub const rope = @import("rope.zig");
pub const attention = @import("attention.zig");

pub const moe = @import("moe.zig");
pub const ssm = @import("ssm.zig");
pub const conv1d = @import("conv1d.zig");

test {
    _ = linear;
    _ = cross_entropy;
    _ = embedding;
    _ = rmsnorm;
    _ = activation;
    _ = rope;
    _ = attention;
    _ = moe;
    _ = ssm;
    _ = conv1d;
}
