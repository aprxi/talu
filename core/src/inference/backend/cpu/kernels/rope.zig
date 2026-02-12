//! CPU Rotary Position Embedding (RoPE) Kernel
//!
//! Re-exports RoPE from compute/ops/math_primitives.zig for organization.
//! The actual implementation lives in ops/math_primitives.zig alongside other math operations.

const compute = @import("../../../../compute/root.zig");

// Re-export RoPE from ops/math_primitives.zig
pub const RoPE = compute.ops.math.RoPE;
