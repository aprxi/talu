//! Forward pass kernels for training.
//!
//! Each sub-module implements a specific forward operation. All kernels are
//! pub and independently benchmarkable. The pass module orchestrates them
//! into the full forward pass.
//!
//! Kernel contract: forward functions write outputs (they do NOT accumulate).
//! Activations needed by backward are saved into the ActivationCache.

pub const linear = @import("linear.zig");
pub const attention = @import("attention.zig");
pub const norm = @import("norm.zig");
pub const activation = @import("activation.zig");
pub const rope = @import("rope.zig");
pub const embedding = @import("embedding.zig");
pub const loss = @import("loss.zig");
pub const pass = @import("pass.zig");

// Re-export the main forward function for convenience
pub const forward = pass.forward;

test {
    _ = linear;
    _ = attention;
    _ = norm;
    _ = activation;
    _ = rope;
    _ = embedding;
    _ = loss;
    _ = pass;
}
