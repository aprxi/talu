//! Kernel interface definitions for compute registry.
//!
//! This defines the kernel IDs, backend selectors, and typed kernel handles
//! used by the registry to provide optimized implementations.

const ops = @import("ops/root.zig");
const dtype_mod = @import("../dtype.zig");

pub const DType = dtype_mod.DType;

/// Backend selector for kernel dispatch.
pub const Backend = enum {
    cpu,
    metal,
};

/// Kernel identifiers for the registry.
pub const KernelId = enum {
    matmul,
    ssm_scan,
    flash_attention,
};

/// Typed kernel function pointers.
pub const MatmulFn = ops.matmul.MatmulFn;
pub const SsmScanFn = ops.simd.ssm_scan.SsmScanFn;
pub const FlashAttentionFn = ops.simd.flash_attention.FlashAttentionFn;

/// Kernel handle returned by registry.
pub const Kernel = union(KernelId) {
    matmul: MatmulFn,
    ssm_scan: SsmScanFn,
    flash_attention: FlashAttentionFn,
};
