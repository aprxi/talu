//! Metal kernel registry for GPU compute.
//!
//! Unlike CPU registry (function pointers), Metal uses lazy graph building.
//! Kernels return ArrayHandle transformations for deferred execution.
//!
//! Note: Current Metal backend uses direct MLX graph API calls. This registry
//! provides a clean interface for future Metal kernel additions.

const graph = @import("graph.zig");
const ArrayHandle = graph.ArrayHandle;

/// Metal kernel identifiers.
pub const MetalKernelId = enum {
    matmul,
    attention,
};

/// Metal kernel function types.
pub const MatmulFn = *const fn (ArrayHandle, ArrayHandle) ArrayHandle;
pub const AttentionFn = *const fn (ArrayHandle, ArrayHandle, ArrayHandle, f32, bool) ArrayHandle;

/// Metal kernel: builds graph node, returns output handle.
pub const MetalKernel = union(MetalKernelId) {
    matmul: MatmulFn,
    attention: AttentionFn,
};

/// Select Metal kernel for given ID.
/// Returns MLX graph-building function for lazy execution.
pub fn selectKernel(id: MetalKernelId) !MetalKernel {
    return switch (id) {
        .matmul => .{ .matmul = graph.mlx_lazy_matmul },
        .attention => .{ .attention = graph.mlx_lazy_attention },
    };
}

// ============================================================================
// Tests
// ============================================================================

test "selectKernel returns matmul" {
    const kernel_val = try selectKernel(.matmul);
    switch (kernel_val) {
        .matmul => {},
        else => return error.TestUnexpectedKernel,
    }
}

test "selectKernel returns attention" {
    const kernel_val = try selectKernel(.attention);
    switch (kernel_val) {
        .attention => {},
        else => return error.TestUnexpectedKernel,
    }
}
