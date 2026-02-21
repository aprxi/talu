//! Quantized row decode primitives for CPU compute.

const std = @import("std");
const dtype = @import("../../dtype.zig");
const grouped_affine_quant = @import("quant/grouped_affine_quant.zig");

/// Decode one FP16 row to f32.
pub fn decodeF16Row(src: []align(1) const u16, dst: []f32) void {
    std.debug.assert(src.len == dst.len);
    for (src, dst) |v, *d| {
        d.* = dtype.fp16ToF32(v);
    }
}

/// Decode one BF16 row to f32.
pub fn decodeBf16Row(src: []align(1) const u16, dst: []f32) void {
    std.debug.assert(src.len == dst.len);
    for (src, dst) |v, *d| {
        d.* = dtype.bf16ToF32(v);
    }
}

/// Gather and decode FP16 rows by row index.
pub fn gatherDecodeF16Rows(
    src_rows: []align(1) const u16,
    row_count: usize,
    row_width: usize,
    row_indices: []const u32,
    dst_rows: []f32,
) !void {
    if (src_rows.len < row_count * row_width) return error.InvalidShape;
    if (dst_rows.len < row_indices.len * row_width) return error.InvalidShape;

    for (row_indices, 0..) |row_idx_u32, out_row_idx| {
        const row_idx: usize = @intCast(row_idx_u32);
        if (row_idx >= row_count) return error.InvalidTokenId;
        const src_row = src_rows[row_idx * row_width ..][0..row_width];
        const dst_row = dst_rows[out_row_idx * row_width ..][0..row_width];
        decodeF16Row(src_row, dst_row);
    }
}

/// Gather and decode BF16 rows by row index.
pub fn gatherDecodeBf16Rows(
    src_rows: []align(1) const u16,
    row_count: usize,
    row_width: usize,
    row_indices: []const u32,
    dst_rows: []f32,
) !void {
    if (src_rows.len < row_count * row_width) return error.InvalidShape;
    if (dst_rows.len < row_indices.len * row_width) return error.InvalidShape;

    for (row_indices, 0..) |row_idx_u32, out_row_idx| {
        const row_idx: usize = @intCast(row_idx_u32);
        if (row_idx >= row_count) return error.InvalidTokenId;
        const src_row = src_rows[row_idx * row_width ..][0..row_width];
        const dst_row = dst_rows[out_row_idx * row_width ..][0..row_width];
        decodeBf16Row(src_row, dst_row);
    }
}

/// Decode one grouped-affine U4 row to f32.
pub fn decodeGroupedAffineU4Row(
    packed_row: [*]align(1) const u32,
    scale_row: [*]align(1) const u16,
    bias_row: [*]align(1) const u16,
    scales_dtype: dtype.DType,
    group_size: usize,
    group_count: usize,
    out_row: [*]f32,
) void {
    std.debug.assert(group_size % 8 == 0);
    const group_u32_count = group_size / 8;

    var group_idx: usize = 0;
    while (group_idx < group_count) : (group_idx += 1) {
        const scale = grouped_affine_quant.scaleBiasToF32(scales_dtype, scale_row[group_idx]);
        const bias = grouped_affine_quant.scaleBiasToF32(scales_dtype, bias_row[group_idx]);
        const scale_vec: @Vector(8, f32) = @splat(scale);
        const bias_vec: @Vector(8, f32) = @splat(bias);
        const weight_base = packed_row + group_idx * group_u32_count;
        const out_base = out_row + group_idx * group_size;

        var pack_idx: usize = 0;
        while (pack_idx + 3 < group_u32_count) : (pack_idx += 4) {
            const nibs = grouped_affine_quant.extract32NibblesToFloat(weight_base + pack_idx);
            (out_base + pack_idx * 8)[0..8].* = @mulAdd(@Vector(8, f32), nibs.n0, scale_vec, bias_vec);
            (out_base + (pack_idx + 1) * 8)[0..8].* = @mulAdd(@Vector(8, f32), nibs.n1, scale_vec, bias_vec);
            (out_base + (pack_idx + 2) * 8)[0..8].* = @mulAdd(@Vector(8, f32), nibs.n2, scale_vec, bias_vec);
            (out_base + (pack_idx + 3) * 8)[0..8].* = @mulAdd(@Vector(8, f32), nibs.n3, scale_vec, bias_vec);
        }
        while (pack_idx < group_u32_count) : (pack_idx += 1) {
            const word = weight_base[pack_idx];
            const nibble_values = grouped_affine_quant.extractNibbles(word);
            (out_base + pack_idx * 8)[0..8].* = @mulAdd(@Vector(8, f32), nibble_values, scale_vec, bias_vec);
        }
    }
}

/// Decode one grouped-affine U8 row to f32.
pub fn decodeGroupedAffineU8Row(
    packed_row: [*]align(1) const u32,
    scale_row: [*]align(1) const u16,
    bias_row: [*]align(1) const u16,
    scales_dtype: dtype.DType,
    group_size: usize,
    group_count: usize,
    out_row: [*]f32,
) void {
    std.debug.assert(group_size % 4 == 0);
    const group_u32_count = group_size / 4;

    var group_idx: usize = 0;
    while (group_idx < group_count) : (group_idx += 1) {
        const scale = grouped_affine_quant.scaleBiasToF32(scales_dtype, scale_row[group_idx]);
        const bias = grouped_affine_quant.scaleBiasToF32(scales_dtype, bias_row[group_idx]);
        const scale_vec: @Vector(4, f32) = @splat(scale);
        const bias_vec: @Vector(4, f32) = @splat(bias);
        const weight_base = packed_row + group_idx * group_u32_count;
        const out_base = out_row + group_idx * group_size;

        var pack_idx: usize = 0;
        while (pack_idx < group_u32_count) : (pack_idx += 1) {
            const word = weight_base[pack_idx];
            const byte_values = grouped_affine_quant.extractBytes(word);
            (out_base + pack_idx * 4)[0..4].* = @mulAdd(@Vector(4, f32), byte_values, scale_vec, bias_vec);
        }
    }
}

/// Gather and decode grouped-affine U4 rows by row index.
pub fn gatherDecodeGroupedAffineU4Rows(
    packed_values: []align(1) const u32,
    scales: []align(1) const u16,
    biases: []align(1) const u16,
    scales_dtype: dtype.DType,
    group_size: usize,
    row_count: usize,
    row_width: usize,
    row_indices: []const u32,
    dst_rows: []f32,
) !void {
    if (group_size == 0 or (row_width % group_size) != 0 or (row_width % 8) != 0) return error.InvalidShape;
    if (dst_rows.len < row_indices.len * row_width) return error.InvalidShape;

    const packed_row_stride = row_width / 8;
    const group_row_stride = row_width / group_size;
    if (packed_values.len < row_count * packed_row_stride) return error.InvalidShape;
    if (scales.len < row_count * group_row_stride or biases.len < row_count * group_row_stride) return error.InvalidShape;

    for (row_indices, 0..) |row_idx_u32, out_row_idx| {
        const row_idx: usize = @intCast(row_idx_u32);
        if (row_idx >= row_count) return error.InvalidTokenId;

        const packed_row = packed_values.ptr + row_idx * packed_row_stride;
        const scale_row = scales.ptr + row_idx * group_row_stride;
        const bias_row = biases.ptr + row_idx * group_row_stride;
        const out_row = dst_rows.ptr + out_row_idx * row_width;
        decodeGroupedAffineU4Row(
            packed_row,
            scale_row,
            bias_row,
            scales_dtype,
            group_size,
            group_row_stride,
            out_row,
        );
    }
}

/// Gather and decode grouped-affine U8 rows by row index.
pub fn gatherDecodeGroupedAffineU8Rows(
    packed_values: []align(1) const u32,
    scales: []align(1) const u16,
    biases: []align(1) const u16,
    scales_dtype: dtype.DType,
    group_size: usize,
    row_count: usize,
    row_width: usize,
    row_indices: []const u32,
    dst_rows: []f32,
) !void {
    if (group_size == 0 or (row_width % group_size) != 0 or (row_width % 4) != 0) return error.InvalidShape;
    if (dst_rows.len < row_indices.len * row_width) return error.InvalidShape;

    const packed_row_stride = row_width / 4;
    const group_row_stride = row_width / group_size;
    if (packed_values.len < row_count * packed_row_stride) return error.InvalidShape;
    if (scales.len < row_count * group_row_stride or biases.len < row_count * group_row_stride) return error.InvalidShape;

    for (row_indices, 0..) |row_idx_u32, out_row_idx| {
        const row_idx: usize = @intCast(row_idx_u32);
        if (row_idx >= row_count) return error.InvalidTokenId;

        const packed_row = packed_values.ptr + row_idx * packed_row_stride;
        const scale_row = scales.ptr + row_idx * group_row_stride;
        const bias_row = biases.ptr + row_idx * group_row_stride;
        const out_row = dst_rows.ptr + out_row_idx * row_width;
        decodeGroupedAffineU8Row(
            packed_row,
            scale_row,
            bias_row,
            scales_dtype,
            group_size,
            group_row_stride,
            out_row,
        );
    }
}

test "decodeF16Row converts known values" {
    const src = [_]u16{ 0x3C00, 0x4000, 0x3800, 0xBC00 };
    var dst = [_]f32{ 0, 0, 0, 0 };
    decodeF16Row(&src, &dst);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), dst[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), dst[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), dst[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), dst[3], 1e-4);
}

test "decodeBf16Row converts known values" {
    const src = [_]u16{ 0x3F80, 0x4000, 0x3F00, 0xBF80 };
    var dst = [_]f32{ 0, 0, 0, 0 };
    decodeBf16Row(&src, &dst);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), dst[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), dst[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), dst[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), dst[3], 1e-4);
}

test "gatherDecodeF16Rows decodes selected rows" {
    const src = [_]u16{
        0x3C00, 0x4000, // row 0: 1, 2
        0x4200, 0x4400, // row 1: 3, 4
        0x4500, 0x4600, // row 2: 5, 6
    };
    const idx = [_]u32{ 2, 0 };
    var out = [_]f32{0} ** 4;

    try gatherDecodeF16Rows(&src, 3, 2, &idx, &out);

    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), out[1], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[2], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[3], 1e-3);
}

test "gatherDecodeBf16Rows rejects out-of-bounds row index" {
    const src = [_]u16{ 0x3F80, 0x4000, 0x4040, 0x4080 };
    const idx = [_]u32{ 0, 2 };
    var out = [_]f32{0} ** 4;

    try std.testing.expectError(error.InvalidTokenId, gatherDecodeBf16Rows(&src, 2, 2, &idx, &out));
}

test "decodeGroupedAffineU4Row applies scale and bias" {
    const packed_values = [_]u32{0x01234567};
    const scales = [_]u16{0x0000}; // 0.0 f16
    const biases = [_]u16{0x3C00}; // 1.0 f16
    var out = [_]f32{0} ** 8;
    decodeGroupedAffineU4Row(&packed_values, &scales, &biases, .f16, 8, 1, &out);
    for (out) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), v, 1e-6);
    }
}

test "decodeGroupedAffineU8Row applies scale and bias" {
    const packed_values = [_]u32{0x04030201};
    const scales = [_]u16{0x0000}; // 0.0 f16
    const biases = [_]u16{0x4000}; // 2.0 f16
    var out = [_]f32{0} ** 4;
    decodeGroupedAffineU8Row(&packed_values, &scales, &biases, .f16, 4, 1, &out);
    for (out) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 2.0), v, 1e-6);
    }
}

test "gatherDecodeGroupedAffineU4Rows decodes selected row" {
    const packed_values = [_]u32{
        0x00000000, // row0
        0xFFFFFFFF, // row1
    };
    const scales = [_]u16{ 0x0000, 0x0000 }; // both zero scale
    const biases = [_]u16{ 0x3C00, 0x4000 }; // row0=>1 row1=>2
    const idx = [_]u32{1};
    var out = [_]f32{0} ** 8;
    try gatherDecodeGroupedAffineU4Rows(&packed_values, &scales, &biases, .f16, 8, 2, 8, &idx, &out);
    for (out) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 2.0), v, 1e-6);
    }
}

test "gatherDecodeGroupedAffineU8Rows decodes selected row" {
    const packed_values = [_]u32{
        0x00000000, // row0
        0x11111111, // row1
    };
    const scales = [_]u16{ 0x0000, 0x0000 }; // both zero scale
    const biases = [_]u16{ 0x3C00, 0x4200 }; // row0=>1 row1=>3
    const idx = [_]u32{1};
    var out = [_]f32{0} ** 4;
    try gatherDecodeGroupedAffineU8Rows(&packed_values, &scales, &biases, .f16, 4, 2, 4, &idx, &out);
    for (out) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 3.0), v, 1e-6);
    }
}
