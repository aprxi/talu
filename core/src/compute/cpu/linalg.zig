//! Linear algebra primitive namespace for CPU compute.

pub const matmul = @import("matmul_primitives.zig");
pub const prefill = @import("matmul_prefill.zig");
pub const matvec = @import("matvec.zig");
pub const dot = @import("dot_product.zig");

