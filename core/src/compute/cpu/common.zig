//! Shared CPU helpers for inference kernels.

const std = @import("std");
const simd = @import("simd/arch/root.zig");

const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

/// Ensure a reusable f32 buffer has at least `needed` elements.
pub fn ensureF32Slice(allocator: std.mem.Allocator, storage: *[]f32, needed: usize) !void {
    if (storage.*.len >= needed) return;
    if (storage.*.len > 0) allocator.free(storage.*);
    storage.* = try allocator.alloc(f32, needed);
}

/// Ensure a reusable u32 buffer has at least `needed` elements.
pub fn ensureU32Slice(allocator: std.mem.Allocator, storage: *[]u32, needed: usize) !void {
    if (storage.*.len >= needed) return;
    if (storage.*.len > 0) allocator.free(storage.*);
    storage.* = try allocator.alloc(u32, needed);
}

/// Add a 1-D bias vector to each row of a [rows, dim] f32 buffer.
pub fn addBiasRows(data: []f32, bias: []const f32, rows: usize, dim: usize) void {
    std.debug.assert(bias.len == dim);
    std.debug.assert(data.len >= rows * dim);

    for (0..rows) |row_idx| {
        const row = data[row_idx * dim ..][0..dim];
        var vec_idx: usize = 0;
        while (vec_idx + VEC_LEN - 1 < dim) : (vec_idx += VEC_LEN) {
            const row_vec: F32Vec = row[vec_idx..][0..VEC_LEN].*;
            const bias_vec: F32Vec = bias[vec_idx..][0..VEC_LEN].*;
            row[vec_idx..][0..VEC_LEN].* = row_vec + bias_vec;
        }
        while (vec_idx < dim) : (vec_idx += 1) {
            row[vec_idx] += bias[vec_idx];
        }
    }
}

test "ensureF32Slice grows buffer" {
    const allocator = std.testing.allocator;
    var storage: []f32 = &.{};
    defer if (storage.len > 0) allocator.free(storage);

    try ensureF32Slice(allocator, &storage, 16);
    try std.testing.expect(storage.len >= 16);
}

test "addBiasRows adds per-row bias" {
    var data = [_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
    };
    const bias = [_]f32{ 0.5, -1.0, 2.0, 0.0 };

    addBiasRows(&data, &bias, 2, 4);

    try std.testing.expectApproxEqAbs(@as(f32, 1.5), data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), data[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), data[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.5), data[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), data[5], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), data[6], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), data[7], 1e-6);
}

test "ensureU32Slice grows buffer" {
    const allocator = std.testing.allocator;
    var storage: []u32 = &.{};
    defer if (storage.len > 0) allocator.free(storage);

    try ensureU32Slice(allocator, &storage, 8);
    try std.testing.expect(storage.len >= 8);
}
