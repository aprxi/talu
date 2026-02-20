//! Generic slotted memory copy primitives.

const std = @import("std");
const indexing = @import("indexing.zig");

/// Copy data from `[axis1, axis2, axis3]` into one slot of `[slot, axis1, axis2, axis3]`.
pub fn copy3DToSlotted4D(
    dst_primary: []f32,
    dst_secondary: []f32,
    dst_slot_stride: usize,
    dst_axis1_stride: usize,
    slot_index: usize,
    src_primary: []const f32,
    src_secondary: []const f32,
    src_axis2_capacity: usize,
    axis1_count: usize,
    axis3_width: usize,
    axis2_len: usize,
) void {
    const src_needed = axis1_count * src_axis2_capacity * axis3_width;
    std.debug.assert(src_primary.len >= src_needed);
    std.debug.assert(src_secondary.len >= src_needed);

    const dst_slot_base = slot_index * dst_slot_stride;
    std.debug.assert(dst_primary.len >= dst_slot_base + axis1_count * dst_axis1_stride);
    std.debug.assert(dst_secondary.len >= dst_slot_base + axis1_count * dst_axis1_stride);

    for (0..axis1_count) |axis1| {
        for (0..axis2_len) |axis2| {
            const src_offset = indexing.flatOffset4D(axis1, axis2, 0, src_axis2_capacity * axis3_width, axis3_width, 1);
            const dst_offset = dst_slot_base + indexing.flatOffset4D(axis1, axis2, 0, dst_axis1_stride, axis3_width, 1);
            @memcpy(dst_primary[dst_offset .. dst_offset + axis3_width], src_primary[src_offset .. src_offset + axis3_width]);
            @memcpy(dst_secondary[dst_offset .. dst_offset + axis3_width], src_secondary[src_offset .. src_offset + axis3_width]);
        }
    }
}

/// Append one row `[axis1, axis3]` into slotted `[slot, axis1, axis2, axis3]` at `axis2_position`.
pub fn appendRowToSlotted4D(
    dst_primary: []f32,
    dst_secondary: []f32,
    dst_slot_stride: usize,
    dst_axis1_stride: usize,
    axis3_width: usize,
    axis1_count: usize,
    slot_index: usize,
    axis2_position: usize,
    src_primary: []const f32,
    src_secondary: []const f32,
) void {
    const values_per_row = axis1_count * axis3_width;
    std.debug.assert(src_primary.len == values_per_row);
    std.debug.assert(src_secondary.len == values_per_row);

    const dst_slot_base = slot_index * dst_slot_stride;
    std.debug.assert(dst_primary.len >= dst_slot_base + axis1_count * dst_axis1_stride);
    std.debug.assert(dst_secondary.len >= dst_slot_base + axis1_count * dst_axis1_stride);

    for (0..axis1_count) |axis1| {
        const src_offset = axis1 * axis3_width;
        const dst_offset = dst_slot_base + indexing.flatOffset4D(axis1, axis2_position, 0, dst_axis1_stride, axis3_width, 1);
        @memcpy(dst_primary[dst_offset .. dst_offset + axis3_width], src_primary[src_offset .. src_offset + axis3_width]);
        @memcpy(dst_secondary[dst_offset .. dst_offset + axis3_width], src_secondary[src_offset .. src_offset + axis3_width]);
    }
}

/// Append `axis2_len` rows `[axis2, axis1, axis3]` into slotted `[slot, axis1, axis2, axis3]`.
pub fn appendRowsToSlotted4D(
    dst_primary: []f32,
    dst_secondary: []f32,
    dst_slot_stride: usize,
    dst_axis1_stride: usize,
    axis3_width: usize,
    axis1_count: usize,
    slot_index: usize,
    start_axis2: usize,
    src_primary: []const f32,
    src_secondary: []const f32,
    axis2_len: usize,
) void {
    const values_per_row = axis1_count * axis3_width;
    std.debug.assert(src_primary.len == axis2_len * values_per_row);
    std.debug.assert(src_secondary.len == axis2_len * values_per_row);

    const dst_slot_base = slot_index * dst_slot_stride;
    std.debug.assert(dst_primary.len >= dst_slot_base + axis1_count * dst_axis1_stride);
    std.debug.assert(dst_secondary.len >= dst_slot_base + axis1_count * dst_axis1_stride);

    for (0..axis1_count) |axis1| {
        for (0..axis2_len) |axis2| {
            const src_offset = axis2 * values_per_row + axis1 * axis3_width;
            const dst_offset = dst_slot_base + indexing.flatOffset4D(axis1, start_axis2 + axis2, 0, dst_axis1_stride, axis3_width, 1);
            @memcpy(dst_primary[dst_offset .. dst_offset + axis3_width], src_primary[src_offset .. src_offset + axis3_width]);
            @memcpy(dst_secondary[dst_offset .. dst_offset + axis3_width], src_secondary[src_offset .. src_offset + axis3_width]);
        }
    }
}

test "copy3DToSlotted4D copies rows into slot layout" {
    const axis1_count: usize = 1;
    const axis3_width: usize = 2;
    const axis2_len: usize = 2;
    const src_axis2_capacity: usize = 4;
    const dst_slot_stride: usize = 8;
    const dst_axis1_stride: usize = 4;

    const src_a = [_]f32{ 10, 11, 20, 21, 30, 31, 40, 41 };
    const src_b = [_]f32{ 110, 111, 120, 121, 130, 131, 140, 141 };
    var dst_a = [_]f32{0} ** 16;
    var dst_b = [_]f32{0} ** 16;

    copy3DToSlotted4D(
        &dst_a,
        &dst_b,
        dst_slot_stride,
        dst_axis1_stride,
        1,
        &src_a,
        &src_b,
        src_axis2_capacity,
        axis1_count,
        axis3_width,
        axis2_len,
    );

    try std.testing.expectApproxEqAbs(@as(f32, 10), dst_a[8], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 21), dst_a[11], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 110), dst_b[8], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 121), dst_b[11], 1e-6);
}

test "appendRowToSlotted4D appends one axis2 row for all axis1 entries" {
    const axis1_count: usize = 2;
    const axis3_width: usize = 2;
    const dst_slot_stride: usize = 16;
    const dst_axis1_stride: usize = 8;

    var dst_a = [_]f32{0} ** 32;
    var dst_b = [_]f32{0} ** 32;
    const src_a = [_]f32{ 1, 2, 3, 4 };
    const src_b = [_]f32{ 11, 12, 13, 14 };

    appendRowToSlotted4D(
        &dst_a,
        &dst_b,
        dst_slot_stride,
        dst_axis1_stride,
        axis3_width,
        axis1_count,
        1,
        2,
        &src_a,
        &src_b,
    );

    // slot=1 base=16, axis1=0 row pos=2 => 16 + 0*8 + 2*2 = 20
    try std.testing.expectApproxEqAbs(@as(f32, 1), dst_a[20], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), dst_a[21], 1e-6);
    // axis1=1 row pos=2 => 16 + 1*8 + 2*2 = 28
    try std.testing.expectApproxEqAbs(@as(f32, 3), dst_a[28], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4), dst_a[29], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 11), dst_b[20], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 14), dst_b[29], 1e-6);
}

test "appendRowsToSlotted4D appends multiple axis2 rows in order" {
    const axis1_count: usize = 1;
    const axis3_width: usize = 2;
    const axis2_len: usize = 2;
    const dst_slot_stride: usize = 8;
    const dst_axis1_stride: usize = 8;

    var dst_a = [_]f32{0} ** 16;
    var dst_b = [_]f32{0} ** 16;
    const src_a = [_]f32{ 1, 2, 3, 4 };
    const src_b = [_]f32{ 11, 12, 13, 14 };

    appendRowsToSlotted4D(
        &dst_a,
        &dst_b,
        dst_slot_stride,
        dst_axis1_stride,
        axis3_width,
        axis1_count,
        1,
        1,
        &src_a,
        &src_b,
        axis2_len,
    );

    // row0 -> axis2=1 offset 8 + 2
    try std.testing.expectApproxEqAbs(@as(f32, 1), dst_a[10], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), dst_a[11], 1e-6);
    // row1 -> axis2=2 offset 8 + 4
    try std.testing.expectApproxEqAbs(@as(f32, 3), dst_a[12], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4), dst_a[13], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 11), dst_b[10], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 14), dst_b[13], 1e-6);
}
