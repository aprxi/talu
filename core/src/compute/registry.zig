//! Kernel registry for compute backends.
//!
//! Provides low-overhead lookup of optimized kernel implementations
//! by ID, backend, and dtype.

const ops = @import("ops/root.zig");
const kernel = @import("kernel.zig");
const dtype_mod = @import("../dtype.zig");

const Kernel = kernel.Kernel;
const KernelId = kernel.KernelId;
const Backend = kernel.Backend;
const DType = dtype_mod.DType;

pub fn selectKernel(id: KernelId, backend: Backend, dtype: DType) !Kernel {
    return switch (id) {
        .matmul => selectMatmul(backend, dtype),
        .ssm_scan => selectSsmScan(backend, dtype),
        .flash_attention => selectFlashAttention(backend, dtype),
    };
}

fn selectMatmul(backend: Backend, dtype: DType) !Kernel {
    if (backend != .cpu) return error.UnsupportedBackend;
    const dk = try ops.matmul.matmulKernel(dtype);
    return .{ .matmul = dk.func };
}

fn selectSsmScan(backend: Backend, dtype: DType) !Kernel {
    if (backend != .cpu) return error.UnsupportedBackend;
    if (dtype != .f32) return error.UnsupportedDType;
    return .{ .ssm_scan = ops.simd.ssm_scan.ssmScanF32 };
}

pub fn selectFlashAttentionForHeadDim(head_dim: usize) !Kernel {
    return switch (head_dim) {
        64 => .{ .flash_attention = ops.simd.flash_attention.flashAttentionF32_64 },
        128 => .{ .flash_attention = ops.simd.flash_attention.flashAttentionF32_128 },
        else => error.UnsupportedHeadDim,
    };
}

fn selectFlashAttention(backend: Backend, dtype: DType) !Kernel {
    if (backend != .cpu) return error.UnsupportedBackend;
    if (dtype != .f32) return error.UnsupportedDType;
    return .{ .flash_attention = ops.simd.flash_attention.flashAttentionF32_128 };
}

test "selectKernel returns matmul kernel for f32" {
    const kernel_val = try selectKernel(.matmul, .cpu, .f32);
    switch (kernel_val) {
        .matmul => {},
        else => return error.TestUnexpectedKernel,
    }
}

test "selectKernel returns ssm_scan kernel for f32" {
    const kernel_val = try selectKernel(.ssm_scan, .cpu, .f32);
    switch (kernel_val) {
        .ssm_scan => {},
        else => return error.TestUnexpectedKernel,
    }
}

test "selectKernel returns flash_attention kernel for f32" {
    const kernel_val = try selectKernel(.flash_attention, .cpu, .f32);
    switch (kernel_val) {
        .flash_attention => {},
        else => return error.TestUnexpectedKernel,
    }
}

test "selectFlashAttentionForHeadDim supports 64 and 128" {
    const k64 = try selectFlashAttentionForHeadDim(64);
    switch (k64) {
        .flash_attention => {},
        else => return error.TestUnexpectedKernel,
    }
    const k128 = try selectFlashAttentionForHeadDim(128);
    switch (k128) {
        .flash_attention => {},
        else => return error.TestUnexpectedKernel,
    }
}
