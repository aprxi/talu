//! Tensor layout transform primitives for CPU compute path.

const std = @import("std");
const tensor = @import("../../tensor.zig");
const matmul = @import("matmul_primitives.zig");

const Tensor = tensor.Tensor;
const MatmulFn = matmul.MatmulFn;

/// Fuse gate (`w1`) and up (`w3`) projection tensors into a single tensor.
///
/// Returns `null` for unsupported/quantized layouts.
pub fn fuseTwoProjectionWeights(allocator: std.mem.Allocator, w1: *const Tensor, w3: *const Tensor) !?Tensor {
    if (w1.n_dims != 2 or w3.n_dims != 2) return null;
    if (w1.shape[0] != w3.shape[0] or w1.shape[1] != w3.shape[1]) return null;
    if (w1.dtype != w3.dtype) return null;

    const dtype = w1.dtype;
    const elem_size = dtype.elementSize();
    if (dtype.isQuantized() or elem_size == 0) return null;

    const rows: usize = @intCast(w1.shape[0]);
    const cols: usize = @intCast(w1.shape[1]);

    const fuse_columns = (dtype == .f32);
    if (fuse_columns) {
        const fused_size = rows * 2 * cols * elem_size;
        const fused_data = try allocator.alloc(u8, fused_size);
        errdefer allocator.free(fused_data);

        const w1_bytes = w1.data();
        const w3_bytes = w3.data();
        const row_size = cols * elem_size;
        const fused_row_size = 2 * cols * elem_size;

        for (0..rows) |row| {
            const src_offset = row * row_size;
            const dst_offset = row * fused_row_size;
            @memcpy(fused_data[dst_offset..][0..row_size], w1_bytes[src_offset..][0..row_size]);
            @memcpy(fused_data[dst_offset + row_size ..][0..row_size], w3_bytes[src_offset..][0..row_size]);
        }

        return Tensor{
            .data_ptr = fused_data.ptr,
            .data_size = fused_size,
            .shape = .{ @intCast(rows), @intCast(2 * cols), 0, 0, 0, 0, 0, 0 },
            .n_dims = 2,
            .dtype = dtype,
            .numel = rows * 2 * cols,
        };
    }

    const fused_size = 2 * rows * cols * elem_size;
    const fused_data = try allocator.alloc(u8, fused_size);
    errdefer allocator.free(fused_data);

    const w1_bytes = w1.data();
    const w3_bytes = w3.data();
    const tensor_size = rows * cols * elem_size;
    @memcpy(fused_data[0..tensor_size], w1_bytes[0..tensor_size]);
    @memcpy(fused_data[tensor_size..][0..tensor_size], w3_bytes[0..tensor_size]);

    return Tensor{
        .data_ptr = fused_data.ptr,
        .data_size = fused_size,
        .shape = .{ @intCast(2 * rows), @intCast(cols), 0, 0, 0, 0, 0, 0 },
        .n_dims = 2,
        .dtype = dtype,
        .numel = 2 * rows * cols,
    };
}

/// Resulting views from a fused QKV projection.
pub const QkvViews = struct {
    q: Tensor,
    k: Tensor,
    v: Tensor,
};

/// Run one matmul against concatenated QKV weights and split into Q/K/V views.
/// After matmul, output is [seq, total_dim] where total_dim = q_dim + 2*kv_dim.
/// Each row contains [Q..., K..., V...] which is rearranged into contiguous blocks.
///
/// `qkv_buffer` must be at least 2 * seq * total_dim to allow for rearrangement.
pub inline fn projectQkv(
    input_tensor: *const Tensor,
    fused_weights: *const Tensor,
    qkv_buffer: []f32,
    seq_len: usize,
    query_dim: usize,
    kv_dim: usize,
    matmul_kernel: MatmulFn,
    matmul_scratch: *matmul.MatmulScratch,
) QkvViews {
    const total_dim = query_dim + 2 * kv_dim;
    const total_elems = seq_len * total_dim;
    std.debug.assert(qkv_buffer.len >= 2 * total_elems);

    // Matmul into second half, then rearrange into first half.
    const matmul_output = qkv_buffer[total_elems .. 2 * total_elems];
    var fused_output = Tensor.view2DSlice(matmul_output, seq_len, total_dim);
    matmul_kernel(input_tensor, fused_weights, &fused_output, matmul_scratch);

    const output_buffer = qkv_buffer[0..total_elems];
    for (0..seq_len) |token_idx| {
        const src_base = token_idx * total_dim;

        const q_dst = token_idx * query_dim;
        const q_src = src_base;
        @memcpy(output_buffer[q_dst..][0..query_dim], matmul_output[q_src..][0..query_dim]);

        const k_dst = seq_len * query_dim + token_idx * kv_dim;
        const k_src = src_base + query_dim;
        @memcpy(output_buffer[k_dst..][0..kv_dim], matmul_output[k_src..][0..kv_dim]);

        const v_dst = seq_len * query_dim + seq_len * kv_dim + token_idx * kv_dim;
        const v_src = src_base + query_dim + kv_dim;
        @memcpy(output_buffer[v_dst..][0..kv_dim], matmul_output[v_src..][0..kv_dim]);
    }

    return .{
        .q = Tensor.view2DSlice(output_buffer[0 .. seq_len * query_dim], seq_len, query_dim),
        .k = Tensor.view2DSlice(output_buffer[seq_len * query_dim .. seq_len * query_dim + seq_len * kv_dim], seq_len, kv_dim),
        .v = Tensor.view2DSlice(output_buffer[seq_len * query_dim + seq_len * kv_dim .. seq_len * total_dim], seq_len, kv_dim),
    };
}

/// Extract the first `prefix_len` values from each row in a row-major buffer.
///
/// Source layout is `[row_count, row_stride]`; destination layout is
/// `[row_count, prefix_len]` contiguous.
pub fn extractRowPrefixes(
    src_rows: []const f32,
    row_count: usize,
    row_stride: usize,
    prefix_len: usize,
    dst_rows: []f32,
) !void {
    if (prefix_len > row_stride) return error.InvalidShape;
    if (src_rows.len < row_count * row_stride) return error.InvalidShape;
    if (dst_rows.len < row_count * prefix_len) return error.InvalidShape;

    for (0..row_count) |row_idx| {
        const src_offset = row_idx * row_stride;
        const dst_offset = row_idx * prefix_len;
        @memcpy(
            dst_rows[dst_offset .. dst_offset + prefix_len],
            src_rows[src_offset .. src_offset + prefix_len],
        );
    }
}

/// Split a `[seq_len, total_dim]` contiguous tensor across last dimension.
pub fn splitLastDimContiguous(
    input_data: []const f32,
    seq_len: usize,
    total_dim: usize,
    split_sizes: []const usize,
    out_slices: []const []f32,
) !void {
    if (split_sizes.len != out_slices.len) return error.InvalidShape;
    if (input_data.len < seq_len * total_dim) return error.InvalidShape;

    var sum_sizes: usize = 0;
    for (split_sizes) |s| sum_sizes += s;
    if (sum_sizes != total_dim) return error.InvalidShape;

    for (split_sizes, out_slices) |split_size, out_slice| {
        if (out_slice.len < seq_len * split_size) return error.InvalidShape;
    }

    var dim_offset: usize = 0;
    for (split_sizes, out_slices) |split_size, out_slice| {
        for (0..seq_len) |seq_idx| {
            const src_base = seq_idx * total_dim + dim_offset;
            const dst_base = seq_idx * split_size;
            @memcpy(out_slice[dst_base..][0..split_size], input_data[src_base..][0..split_size]);
        }
        dim_offset += split_size;
    }
}

test "projectQkv rearranges contiguous Q/K/V blocks" {
    const allocator = std.testing.allocator;

    const seq_len = 2;
    const q_dim = 4;
    const kv_dim = 2;
    const total_dim = q_dim + 2 * kv_dim;
    const d_model = 4;

    var input_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{ seq_len, d_model });
    defer input_owned.deinit();
    for (input_owned.asSlice(f32), 0..) |*v, i| v.* = @floatFromInt(i + 1);

    var weight_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{ d_model, total_dim });
    defer weight_owned.deinit();
    const weight_data = weight_owned.asSlice(f32);
    @memset(weight_data, 0.0);
    for (0..d_model) |in_idx| {
        if (in_idx < total_dim) {
            weight_data[in_idx * total_dim + in_idx] = 1.0;
        }
    }

    const qkv_buffer = try allocator.alloc(f32, 2 * seq_len * total_dim);
    defer allocator.free(qkv_buffer);

    var matmul_scratch = try matmul.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();
    const dk = try matmul.matmulKernel(.f32);

    const input_tensor = input_owned.toTensor();
    const weight_tensor = weight_owned.toTensor();
    const views = projectQkv(&input_tensor, &weight_tensor, qkv_buffer, seq_len, q_dim, kv_dim, dk.func, &matmul_scratch);

    const q_data = views.q.asSlice(f32);
    const k_data = views.k.asSlice(f32);
    const v_data = views.v.asSlice(f32);

    try std.testing.expectEqual(@as(usize, seq_len * q_dim), q_data.len);
    try std.testing.expectEqual(@as(usize, seq_len * kv_dim), k_data.len);
    try std.testing.expectEqual(@as(usize, seq_len * kv_dim), v_data.len);

    const q_start = @intFromPtr(q_data.ptr) - @intFromPtr(qkv_buffer.ptr);
    const k_start = @intFromPtr(k_data.ptr) - @intFromPtr(qkv_buffer.ptr);
    const v_start = @intFromPtr(v_data.ptr) - @intFromPtr(qkv_buffer.ptr);

    try std.testing.expectEqual(@as(usize, 0), q_start / @sizeOf(f32));
    try std.testing.expectEqual(@as(usize, seq_len * q_dim), k_start / @sizeOf(f32));
    try std.testing.expectEqual(@as(usize, seq_len * q_dim + seq_len * kv_dim), v_start / @sizeOf(f32));
}

test "projectQkv handles single-token sequence" {
    const allocator = std.testing.allocator;

    const seq_len = 1;
    const q_dim = 8;
    const kv_dim = 4;
    const total_dim = q_dim + 2 * kv_dim;
    const d_model = 8;

    var input_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{ seq_len, d_model });
    defer input_owned.deinit();
    @memset(input_owned.asSlice(f32), 1.0);

    var weight_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{ d_model, total_dim });
    defer weight_owned.deinit();
    @memset(weight_owned.asSlice(f32), 0.1);

    const qkv_buffer = try allocator.alloc(f32, 2 * seq_len * total_dim);
    defer allocator.free(qkv_buffer);

    var matmul_scratch = try matmul.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();
    const dk = try matmul.matmulKernel(.f32);

    const input_tensor = input_owned.toTensor();
    const weight_tensor = weight_owned.toTensor();
    const views = projectQkv(&input_tensor, &weight_tensor, qkv_buffer, seq_len, q_dim, kv_dim, dk.func, &matmul_scratch);

    try std.testing.expectEqual(@as(usize, 1), @as(usize, @intCast(views.q.shape[0])));
    try std.testing.expectEqual(@as(usize, q_dim), @as(usize, @intCast(views.q.shape[1])));
    try std.testing.expectEqual(@as(usize, 1), @as(usize, @intCast(views.k.shape[0])));
    try std.testing.expectEqual(@as(usize, kv_dim), @as(usize, @intCast(views.k.shape[1])));
    try std.testing.expectEqual(@as(usize, 1), @as(usize, @intCast(views.v.shape[0])));
    try std.testing.expectEqual(@as(usize, kv_dim), @as(usize, @intCast(views.v.shape[1])));
}

test "extractRowPrefixes copies contiguous row heads" {
    const src = [_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
    };
    var dst = [_]f32{0} ** 4;
    try extractRowPrefixes(&src, 2, 4, 2, &dst);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 5, 6 }, &dst);
}

test "fuseTwoProjectionWeights concatenates f32 rows by columns" {
    const allocator = std.testing.allocator;
    var w1_data = [_]f32{
        1, 2,
        3, 4,
    };
    var w3_data = [_]f32{
        10, 20,
        30, 40,
    };
    var w1 = Tensor.view2DSlice(&w1_data, 2, 2);
    var w3 = Tensor.view2DSlice(&w3_data, 2, 2);

    const fused_opt = try fuseTwoProjectionWeights(allocator, &w1, &w3);
    try std.testing.expect(fused_opt != null);
    var fused = fused_opt.?;
    defer allocator.free(fused.data());

    try std.testing.expectEqual(@as(i64, 2), fused.shape[0]);
    try std.testing.expectEqual(@as(i64, 4), fused.shape[1]);
    const data = fused.asSlice(f32);
    try std.testing.expectEqualSlices(f32, &[_]f32{
        1, 2, 10, 20,
        3, 4, 30, 40,
    }, data[0..8]);
}

test "splitLastDimContiguous splits per-row contiguous segments" {
    const input = [_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
    };
    var out_a = [_]f32{0} ** 2; // seq_len * 1
    var out_b = [_]f32{0} ** 6; // seq_len * 3
    try splitLastDimContiguous(
        &input,
        2,
        4,
        &.{ 1, 3 },
        &.{ out_a[0..], out_b[0..] },
    );
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 5 }, &out_a);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 2, 3, 4, 6, 7, 8 }, &out_b);
}
