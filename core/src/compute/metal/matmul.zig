//! Metal GPU matrix multiplication kernels.
//!
//! Provides f32 and quantized (grouped affine U4) matrix multiplication
//! using Metal Performance Shaders on macOS.

const std = @import("std");
const device_mod = @import("device.zig");
const mlx = @import("mlx.zig");
const MetalDevice = device_mod.MetalDevice;

/// C API imports.
extern fn metal_matmul_f32(
    device: *MetalDevice,
    a: [*]const f32,
    m: usize,
    k: usize,
    b: [*]const f32,
    n: usize,
    c: [*]f32,
) bool;

extern fn metal_matmul_f32_transB_scaled(
    device: *MetalDevice,
    a: [*]const f32,
    m: usize,
    k: usize,
    b: [*]const f32,
    n: usize,
    c: [*]f32,
    alpha: f32,
) bool;

extern fn metal_matmul_f32_i8_transB_scaled(
    device: *MetalDevice,
    a: [*]const f32,
    m: usize,
    k: usize,
    b: [*]const i8,
    n: usize,
    b_scales: [*]const f32,
    c: [*]f32,
    alpha: f32,
) bool;

/// F32 matrix multiplication: C = A @ B.
/// A: [m x k], B: [k x n], C: [m x n].
pub fn matmulF32(
    device: *device_mod.Device,
    a: []const f32,
    m: usize,
    k: usize,
    b: []const f32,
    n: usize,
    c: []f32,
) !void {
    std.debug.assert(a.len >= m * k);
    std.debug.assert(b.len >= k * n);
    std.debug.assert(c.len >= m * n);

    const success = metal_matmul_f32(
        device.handle,
        a.ptr,
        m,
        k,
        b.ptr,
        n,
        c.ptr,
    );

    if (!success) return error.MetalMatmulFailed;
}

/// F32 matrix multiplication with transposed B and scaling: C = alpha * A @ B^T.
/// A: [m x k], B: [n x k] (stored row-major, will be transposed), C: [m x n].
/// Optimized for attention: Q @ K^T where Q=[queries, head_dim], K=[history, head_dim].
pub fn matmulF32TransBScaled(
    device: *device_mod.Device,
    a: []const f32,
    m: usize,
    k: usize,
    b: []const f32,
    n: usize,
    c: []f32,
    alpha: f32,
) !void {
    std.debug.assert(a.len >= m * k);
    std.debug.assert(b.len >= n * k);
    std.debug.assert(c.len >= m * n);

    const success = metal_matmul_f32_transB_scaled(
        device.handle,
        a.ptr,
        m,
        k,
        b.ptr,
        n,
        c.ptr,
        alpha,
    );

    if (!success) return error.MetalMatmulFailed;
}

/// F32 Q × I8 K matrix multiplication with dequant: C = alpha * A @ dequant(B)^T.
/// A: [m x k] f32 (queries)
/// B: [n x k] i8 (keys, will be dequantized and transposed)
/// B_scales: [n] f32 (per-row scales for K)
/// C: [m x n] f32 (output scores)
pub fn matmulF32I8TransBScaled(
    device: *device_mod.Device,
    a: []const f32,
    m: usize,
    k: usize,
    b: []const i8,
    n: usize,
    b_scales: []const f32,
    c: []f32,
    alpha: f32,
) !void {
    std.debug.assert(a.len >= m * k);
    std.debug.assert(b.len >= n * k);
    std.debug.assert(b_scales.len >= n);
    std.debug.assert(c.len >= m * n);

    const success = metal_matmul_f32_i8_transB_scaled(
        device.handle,
        a.ptr,
        m,
        k,
        b.ptr,
        n,
        b_scales.ptr,
        c.ptr,
        alpha,
    );

    if (!success) return error.MetalMatmulFailed;
}

/// Grouped-affine u4 quantized matrix multiplication.
pub fn matmulGaffineU4(
    device: *device_mod.Device,
    a: []const f32,
    m: usize,
    k: usize,
    b_data: []const u8,
    b_scales: []const u16,
    b_biases: []const u16,
    n: usize,
    group_size: usize,
    c: []f32,
) !void {
    _ = device;
    std.debug.assert(a.len >= m * k);
    std.debug.assert(c.len >= m * n);
    try mlx.matmulGaffineU4(a, m, k, b_data, b_scales, b_biases, n, group_size, c);
}

// =============================================================================
// Unit Tests - compiled only on macOS where Metal is available
// =============================================================================

const builtin = @import("builtin");

test "matmulF32 computes matrix product" {
    if (comptime builtin.os.tag != .macos) return;

    var device = device_mod.Device.init() catch return;
    defer device.deinit();

    // Test: [2x3] @ [3x2] = [2x2]
    const m: usize = 2;
    const k: usize = 3;
    const n: usize = 2;

    // A = [[1,2,3], [4,5,6]]
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    // B = [[1,2], [3,4], [5,6]]
    const b = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var c: [m * n]f32 = undefined;

    matmulF32(&device, &a, m, k, &b, n, &c) catch |err| {
        try std.testing.expect(err == error.MetalMatmulFailed);
        return;
    };

    // Expected: C[0,0] = 1*1 + 2*3 + 3*5 = 22
    //           C[0,1] = 1*2 + 2*4 + 3*6 = 28
    //           C[1,0] = 4*1 + 5*3 + 6*5 = 49
    //           C[1,1] = 4*2 + 5*4 + 6*6 = 64
    try std.testing.expectApproxEqAbs(@as(f32, 22), c[0], 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 28), c[1], 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 49), c[2], 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 64), c[3], 0.1);
}

test "matmulGaffineU4 computes quantized matrix product" {
    if (comptime builtin.os.tag != .macos) return;

    var device = device_mod.Device.init() catch return;
    defer device.deinit();

    // Test dimensions matching quantization requirements
    const m: usize = 2;
    const k: usize = 8; // Must be multiple of 8 for 4-bit packing
    const n: usize = 4;
    const group_size: usize = 8;

    // A = [[1,1,1,1,1,1,1,1], [2,2,2,2,2,2,2,2]]
    var a: [m * k]f32 = undefined;
    for (0..m) |row| {
        const val: f32 = @floatFromInt(row + 1);
        for (0..k) |col| {
            a[row * k + col] = val;
        }
    }

    // Quantized weights: 0x11 = 1 in both nibbles
    var b_data: [n * k / 2]u8 = undefined;
    for (&b_data) |*v| v.* = 0x11;

    const num_groups = k / group_size;
    var b_scales: [n * num_groups]u16 = undefined;
    var b_biases: [n * num_groups]u16 = undefined;
    for (&b_scales) |*v| v.* = 0x3C00; // 1.0 in fp16
    for (&b_biases) |*v| v.* = 0x0000; // 0.0

    var c: [m * n]f32 = undefined;

    matmulGaffineU4(&device, &a, m, k, &b_data, &b_scales, &b_biases, n, group_size, &c) catch {
        return;
    };

    // Verify output is valid and follows expected pattern
    for (c) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
    // Row 0 outputs should all be equal (same input dotted with same weights)
    try std.testing.expectApproxEqAbs(c[0], c[1], 0.1);
    try std.testing.expectApproxEqAbs(c[0], c[2], 0.1);
    try std.testing.expectApproxEqAbs(c[0], c[3], 0.1);
    // Row 1 should be ~2x row 0 (input is 2x)
    try std.testing.expectApproxEqAbs(c[4], c[0] * 2.0, c[0] * 0.1);
}

test "matmulF32TransBScaled matches CPU reference" {
    if (comptime builtin.os.tag != .macos) return;

    var device = device_mod.Device.init() catch return;
    defer device.deinit();

    // Realistic attention dimensions: Q @ K^T
    // Q: [m, k] = [2, 128] (2 queries, head_dim=128)
    // K: [n, k] = [512, 128] (512 history tokens, head_dim=128)
    // C: [m, n] = [2, 512] (attention scores)
    const m: usize = 2;
    const k: usize = 128;
    const n: usize = 512;
    const alpha: f32 = 1.0 / @sqrt(@as(f32, 128.0)); // typical scale

    // Use page-aligned allocator for zero-copy potential
    const page_alloc = std.heap.page_allocator;

    const a = page_alloc.alloc(f32, m * k) catch return;
    defer page_alloc.free(a);
    const b = page_alloc.alloc(f32, n * k) catch return;
    defer page_alloc.free(b);
    const c_metal = page_alloc.alloc(f32, m * n) catch return;
    defer page_alloc.free(c_metal);
    const c_cpu = page_alloc.alloc(f32, m * n) catch return;
    defer page_alloc.free(c_cpu);

    // Initialize with deterministic values
    var rng = std.Random.DefaultPrng.init(42);
    for (a) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
    for (b) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;

    // Metal computation: C = alpha * A @ B^T
    matmulF32TransBScaled(&device, a, m, k, b, n, c_metal, alpha) catch |err| {
        std.debug.print("Metal matmul failed: {}\n", .{err});
        return;
    };

    // CPU reference: C[i,j] = alpha * sum_l(A[i,l] * B[j,l])
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..k) |l| {
                sum += a[i * k + l] * b[j * k + l];
            }
            c_cpu[i * n + j] = alpha * sum;
        }
    }

    // Compare Metal vs CPU
    var max_diff: f32 = 0.0;
    var max_rel_diff: f32 = 0.0;
    for (0..m * n) |i| {
        const diff = @abs(c_metal[i] - c_cpu[i]);
        const rel = if (@abs(c_cpu[i]) > 1e-6) diff / @abs(c_cpu[i]) else diff;
        max_diff = @max(max_diff, diff);
        max_rel_diff = @max(max_rel_diff, rel);
    }

    std.debug.print("\nMetal vs CPU: max_abs_diff={d:.6}, max_rel_diff={d:.6}\n", .{ max_diff, max_rel_diff });

    // Tolerance: allow small floating point differences
    try std.testing.expect(max_diff < 1e-4);
    try std.testing.expect(max_rel_diff < 1e-4);
}
