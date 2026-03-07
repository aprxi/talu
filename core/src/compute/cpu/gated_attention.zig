//! Gated-attention math primitives.
//!
//! These helpers keep gated-query math in the compute layer
//! rather than inside inference backend orchestration.

const std = @import("std");
const math = @import("math.zig");
const simd = math.simd;

const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

pub fn compactQueryProjection(
    query_projection_values: []const f32,
    compacted_query_values: []f32,
    sequence_len: usize,
    query_dim: usize,
    query_projection_dim: usize,
    head_count: usize,
    head_dim: usize,
) !void {
    if (query_projection_values.len < sequence_len * query_projection_dim) return error.InvalidShape;
    if (compacted_query_values.len < sequence_len * query_dim) return error.InvalidShape;
    if (head_count * head_dim != query_dim) return error.InvalidShape;
    if (query_projection_dim != head_count * head_dim * 2) return error.InvalidShape;

    for (0..sequence_len) |row| {
        const row_proj_offset = row * query_projection_dim;
        const row_compact_offset = row * query_dim;
        const row_proj = query_projection_values[row_proj_offset .. row_proj_offset + query_projection_dim];
        const row_compact = compacted_query_values[row_compact_offset .. row_compact_offset + query_dim];
        for (0..head_count) |head_idx| {
            const src_base = head_idx * head_dim * 2;
            const dst_base = head_idx * head_dim;
            @memcpy(
                row_compact[dst_base .. dst_base + head_dim],
                row_proj[src_base .. src_base + head_dim],
            );
        }
    }
}

pub fn applyOutputGateInPlace(
    context_values: []f32,
    query_projection_values: []const f32,
    sequence_len: usize,
    query_dim: usize,
    query_projection_dim: usize,
    head_count: usize,
    head_dim: usize,
) !void {
    if (context_values.len < sequence_len * query_dim) return error.InvalidShape;
    if (query_projection_values.len < sequence_len * query_projection_dim) return error.InvalidShape;
    if (head_count * head_dim != query_dim) return error.InvalidShape;
    if (query_projection_dim != head_count * head_dim * 2) return error.InvalidShape;

    const one: F32Vec = @splat(1.0);
    for (0..sequence_len) |row| {
        const row_ctx_offset = row * query_dim;
        const row_proj_offset = row * query_projection_dim;
        const context_row = context_values[row_ctx_offset .. row_ctx_offset + query_dim];
        const proj_row = query_projection_values[row_proj_offset .. row_proj_offset + query_projection_dim];
        for (0..head_count) |head_idx| {
            const ctx_base = head_idx * head_dim;
            const gate_base = head_idx * head_dim * 2 + head_dim;
            const context_head = context_row[ctx_base .. ctx_base + head_dim];
            const gate_head = proj_row[gate_base .. gate_base + head_dim];
            var idx: usize = 0;
            while (idx + VEC_LEN - 1 < head_dim) : (idx += VEC_LEN) {
                const gate_vec: F32Vec = gate_head[idx..][0..VEC_LEN].*;
                const sig = one / (one + math.fastExp(-gate_vec));
                const ctx = context_head[idx..][0..VEC_LEN];
                ctx.* = ctx.* * sig;
            }
            while (idx < head_dim) : (idx += 1) {
                context_head[idx] *= 1.0 / (1.0 + math.fastExpScalar(-gate_head[idx]));
            }
        }
    }
}

test "compactQueryProjection extracts per-head q lanes without corrupting gates" {
    const packed_q = [_]f32{
        1, 2, 10, 20, 3, 4, 30, 40,
        5, 6, 50, 60, 7, 8, 70, 80,
    };
    var compacted = [_]f32{0} ** 8;
    try compactQueryProjection(&packed_q, &compacted, 2, 4, 8, 2, 2);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, &compacted);
}

test "applyOutputGateInPlace reads per-head gate lanes" {
    var context = [_]f32{ 2, 4, 6, 8 };
    const packed_q = [_]f32{ 1, 2, 0, 0, 3, 4, 2, -2 };
    try applyOutputGateInPlace(&context, &packed_q, 1, 4, 8, 2, 2);
    try std.testing.expectApproxEqRel(@as(f32, 1.0), context[0], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 2.0), context[1], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 6.0 * (1.0 / (1.0 + @exp(-2.0)))), context[2], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 8.0 * (1.0 / (1.0 + @exp(2.0)))), context[3], 1e-6);
}
