//! Metal GPU matrix multiplication kernels.
//!
//! Provides f32 and quantized (grouped affine U4) matrix multiplication
//! using Metal Performance Shaders on macOS.

const std = @import("std");
const device_mod = @import("device.zig");
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

extern fn metal_matmul_mlx4bit(
    device: *MetalDevice,
    a: [*]const f32,
    m: usize,
    k: usize,
    b_data: [*]const u8,
    b_scales: [*]const u16,
    b_biases: [*]const u16,
    n: usize,
    group_size: usize,
    c: [*]f32,
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
    std.debug.assert(a.len >= m * k);
    std.debug.assert(c.len >= m * n);

    const success = metal_matmul_mlx4bit(
        device.handle,
        a.ptr,
        m,
        k,
        b_data.ptr,
        b_scales.ptr,
        b_biases.ptr,
        n,
        group_size,
        c.ptr,
    );

    if (!success) return error.MetalMatmulFailed;
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

    matmulGaffineU4(&device, &a, m, k, &b_data, &b_scales, &b_biases, n, group_size, &c) catch |err| {
        try std.testing.expect(err == error.MetalMatmulFailed);
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
