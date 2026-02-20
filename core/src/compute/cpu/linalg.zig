//! Linear algebra primitive namespace for CPU compute.

pub const matmul = @import("matmul_primitives.zig");
pub const prefill = @import("matmul_prefill.zig");
pub const matvec = @import("matvec.zig");
pub const dot = @import("dot_product.zig");

// Re-export the canonical matmul surface at namespace root so callers can
// depend on `compute.cpu.linalg.*` without importing leaf modules directly.
pub const MatmulScratch = matmul.MatmulScratch;
pub const MatmulFn = matmul.MatmulFn;
pub const DispatchedKernel = matmul.DispatchedKernel;
pub const matmulKernel = matmul.matmulKernel;
pub const matmulAuto = matmul.matmulAuto;
pub const matmulF32 = matmul.matmulF32;
pub const matmulGaffineU4 = matmul.matmulGaffineU4;
