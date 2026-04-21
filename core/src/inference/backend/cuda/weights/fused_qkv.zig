//! Fused QKV weight upload helpers for the CUDA inference backend.

const dense = @import("dense.zig");
const upload_dispatch = @import("upload_dispatch.zig");
const denseMaterializeOutInU16 = dense.materializeDenseOutInU16;
const denseMaterializeOutInF32 = dense.materializeDenseOutInF32;
const uploadLinearWeight = upload_dispatch.uploadLinearWeight;


const std = @import("std");
const compute = @import("compute_pkg");
const tensor = @import("tensor_pkg");
const dtype = @import("dtype_pkg");
const log = @import("log_pkg");
const load_transforms = @import("models_pkg").load.transforms;
const models = @import("models_pkg");
const GateUpLayout = models.runtime_blocks.GateUpLayout;

const LoadedModel = models.LoadedModel;
const Tensor = tensor.Tensor;

/// Convert UE8M0 block scale exponent to f32 scale factor.
inline fn ue8m0ToScale(e8m0: u8) f32 {
    const exp_bits = @as(u32, e8m0) << 23;
    return @bitCast(exp_bits);
}

// --- Shared types from engine_types.zig ---
const engine_types = @import("../runtime/_types_impl.zig");
const KvCacheDtype = engine_types.KvCacheDtype;
const gaffine_scales_dtype_f16 = engine_types.gaffine_scales_dtype_f16;
const gaffine_scales_dtype_bf16 = engine_types.gaffine_scales_dtype_bf16;
const DenseU16Dtype = engine_types.DenseU16Dtype;
const EmbeddingLookupKind = engine_types.EmbeddingLookupKind;
const EmbeddingLookup = engine_types.EmbeddingLookup;
const LinearWeight = engine_types.LinearWeight;
const DeviceTensor = engine_types.DeviceTensor;
const MoEWeightRefs = engine_types.MoEWeightRefs;
const MoEWeights = models.runtime_blocks.MoEWeights;

pub const FusedQkvUpload = struct {
    q: LinearWeight,
    k: LinearWeight,
    v: LinearWeight,
};

fn sliceNvfp4Rows(meta: tensor.Nvfp4Meta, row_start: usize, row_count: usize) !tensor.Nvfp4Meta {
    const total_rows: usize = @intCast(meta.rows);
    const scale_cols: usize = @intCast(meta.scale_cols);
    if (row_count == 0 or scale_cols == 0) return error.InvalidShape;
    if (row_start > total_rows or row_count > total_rows - row_start) return error.InvalidShape;

    const scale_offset = std.math.mul(usize, row_start, scale_cols) catch return error.InvalidShape;
    const scale_len = std.math.mul(usize, row_count, scale_cols) catch return error.InvalidShape;
    if (meta.block_scales_len < scale_offset + scale_len) return error.InvalidShape;
    const base_scales = meta.block_scales_data orelse return error.MissingScales;

    return .{
        .block_scales_data = base_scales + scale_offset,
        .block_scales_len = scale_len,
        .rows = @intCast(row_count),
        .cols = meta.cols,
        .packed_cols = meta.packed_cols,
        .scale_cols = meta.scale_cols,
        .group_size = meta.group_size,
        .weight_global_scale = meta.weight_global_scale,
    };
}

fn sliceFusedNvfp4QkvView(
    fused_qkv: *const Tensor,
    total_out: usize,
    row_start: usize,
    row_count: usize,
    input_dim: usize,
) !Tensor {
    if (fused_qkv.n_dims != 2) return error.UnsupportedModel;
    if (fused_qkv.shape[0] <= 0 or fused_qkv.shape[1] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(fused_qkv.shape[0]);
    const cols: usize = @intCast(fused_qkv.shape[1]);
    if (rows != total_out or cols != input_dim) return error.InvalidArgument;
    if (row_count == 0 or row_start > rows or row_count > rows - row_start) return error.InvalidArgument;

    const bytes_per_row = std.math.divExact(usize, fused_qkv.data_size, rows) catch return error.InvalidArgument;
    var sliced = fused_qkv.*;
    sliced.shape[0] = @intCast(row_count);
    sliced.shape[1] = @intCast(input_dim);
    sliced.numel = row_count * input_dim;
    sliced.data_size = row_count * bytes_per_row;
    if (fused_qkv.data_ptr) |base| sliced.data_ptr = base + row_start * bytes_per_row;
    sliced.nvfp4 = try sliceNvfp4Rows(fused_qkv.nvfp4 orelse return error.MissingScales, row_start, row_count);
    return sliced;
}

pub fn uploadFusedQkvWeights(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    fused_qkv: *const Tensor,
    input_dim: usize,
    q_out: usize,
    kv_out: usize,
) !FusedQkvUpload {
    const total_out = std.math.add(usize, q_out, std.math.mul(usize, kv_out, 2) catch return error.InvalidArgument) catch return error.InvalidArgument;
    if ((fused_qkv.dtype == .u8 or fused_qkv.dtype == .i8) and fused_qkv.nvfp4 != null) {
        var q_tensor = try sliceFusedNvfp4QkvView(fused_qkv, total_out, 0, q_out, input_dim);
        var k_tensor = try sliceFusedNvfp4QkvView(fused_qkv, total_out, q_out, kv_out, input_dim);
        var v_tensor = try sliceFusedNvfp4QkvView(fused_qkv, total_out, q_out + kv_out, kv_out, input_dim);
        const q = try uploadLinearWeight(device, allocator, &q_tensor, input_dim);
        errdefer {
            var q_mut = q;
            q_mut.deinit(device);
        }
        const k = try uploadLinearWeight(device, allocator, &k_tensor, input_dim);
        errdefer {
            var k_mut = k;
            k_mut.deinit(device);
        }
        const v = try uploadLinearWeight(device, allocator, &v_tensor, input_dim);
        return .{ .q = q, .k = k, .v = v };
    }
    if (fused_qkv.dtype == .f16 or fused_qkv.dtype == .bf16) {
        var out_in = try denseMaterializeOutInU16(allocator, fused_qkv, input_dim, total_out);
        defer out_in.deinit(allocator);

        const q_count = std.math.mul(usize, q_out, input_dim) catch return error.InvalidArgument;
        const kv_count = std.math.mul(usize, kv_out, input_dim) catch return error.InvalidArgument;
        const expected = std.math.add(usize, q_count, std.math.mul(usize, kv_count, 2) catch return error.InvalidArgument) catch return error.InvalidArgument;
        if (out_in.values.len != expected) return error.InvalidArgument;

        const q_vals = out_in.values[0..q_count];
        const k_vals = out_in.values[q_count .. q_count + kv_count];
        const v_vals = out_in.values[q_count + kv_count .. q_count + kv_count + kv_count];
        const q_bytes = std.mem.sliceAsBytes(q_vals);
        const k_bytes = std.mem.sliceAsBytes(k_vals);
        const v_bytes = std.mem.sliceAsBytes(v_vals);
        var q_tensor = Tensor.view(@constCast(q_bytes.ptr), &.{ q_out, input_dim }, fused_qkv.dtype, q_bytes.len);
        var k_tensor = Tensor.view(@constCast(k_bytes.ptr), &.{ kv_out, input_dim }, fused_qkv.dtype, k_bytes.len);
        var v_tensor = Tensor.view(@constCast(v_bytes.ptr), &.{ kv_out, input_dim }, fused_qkv.dtype, v_bytes.len);
        const q = try uploadLinearWeight(device, allocator, &q_tensor, input_dim);
        errdefer {
            var q_mut = q;
            q_mut.deinit(device);
        }
        const k = try uploadLinearWeight(device, allocator, &k_tensor, input_dim);
        errdefer {
            var k_mut = k;
            k_mut.deinit(device);
        }
        const v = try uploadLinearWeight(device, allocator, &v_tensor, input_dim);
        return .{ .q = q, .k = k, .v = v };
    }
    if (fused_qkv.dtype == .f32) {
        var out_in = try denseMaterializeOutInF32(allocator, fused_qkv, input_dim, total_out);
        defer out_in.deinit(allocator);

        const q_count = std.math.mul(usize, q_out, input_dim) catch return error.InvalidArgument;
        const kv_count = std.math.mul(usize, kv_out, input_dim) catch return error.InvalidArgument;
        const expected = std.math.add(usize, q_count, std.math.mul(usize, kv_count, 2) catch return error.InvalidArgument) catch return error.InvalidArgument;
        if (out_in.values.len != expected) return error.InvalidArgument;

        const q_vals = out_in.values[0..q_count];
        const k_vals = out_in.values[q_count .. q_count + kv_count];
        const v_vals = out_in.values[q_count + kv_count .. q_count + kv_count + kv_count];
        const q_bytes = std.mem.sliceAsBytes(q_vals);
        const k_bytes = std.mem.sliceAsBytes(k_vals);
        const v_bytes = std.mem.sliceAsBytes(v_vals);
        var q_tensor = Tensor.view(@constCast(q_bytes.ptr), &.{ q_out, input_dim }, .f32, q_bytes.len);
        var k_tensor = Tensor.view(@constCast(k_bytes.ptr), &.{ kv_out, input_dim }, .f32, k_bytes.len);
        var v_tensor = Tensor.view(@constCast(v_bytes.ptr), &.{ kv_out, input_dim }, .f32, v_bytes.len);
        const q = try uploadLinearWeight(device, allocator, &q_tensor, input_dim);
        errdefer {
            var q_mut = q;
            q_mut.deinit(device);
        }
        const k = try uploadLinearWeight(device, allocator, &k_tensor, input_dim);
        errdefer {
            var k_mut = k;
            k_mut.deinit(device);
        }
        const v = try uploadLinearWeight(device, allocator, &v_tensor, input_dim);
        return .{ .q = q, .k = k, .v = v };
    }
    return error.UnsupportedModel;
}

test "sliceFusedNvfp4QkvView preserves packed rows and scales" {
    const packed_bytes = [_]u8{
        0x10, 0x11,
        0x20, 0x21,
        0x30, 0x31,
        0x40, 0x41,
    };
    const scales = [_]u8{
        0xA0,
        0xB0,
        0xC0,
        0xD0,
    };
    var fused = Tensor.view(@constCast(packed_bytes[0..].ptr), &.{ 4, 4 }, .u8, packed_bytes.len);
    fused.nvfp4 = .{
        .block_scales_data = scales[0..].ptr,
        .block_scales_len = scales.len,
        .rows = 4,
        .cols = 4,
        .packed_cols = 2,
        .scale_cols = 1,
        .group_size = 16,
        .weight_global_scale = 1.0,
    };

    const q = try sliceFusedNvfp4QkvView(&fused, 4, 0, 2, 4);
    const k = try sliceFusedNvfp4QkvView(&fused, 4, 2, 1, 4);
    const v = try sliceFusedNvfp4QkvView(&fused, 4, 3, 1, 4);

    try std.testing.expectEqual(@as(i64, 2), q.shape[0]);
    try std.testing.expectEqual(@as(i64, 1), k.shape[0]);
    try std.testing.expectEqual(@as(i64, 1), v.shape[0]);

    try std.testing.expectEqualSlices(u8, packed_bytes[0..4], q.data()[0..q.data_size]);
    try std.testing.expectEqualSlices(u8, packed_bytes[4..6], k.data()[0..k.data_size]);
    try std.testing.expectEqualSlices(u8, packed_bytes[6..8], v.data()[0..v.data_size]);

    const q_meta = q.nvfp4 orelse return error.TestExpectedEqual;
    const k_meta = k.nvfp4 orelse return error.TestExpectedEqual;
    const v_meta = v.nvfp4 orelse return error.TestExpectedEqual;
    try std.testing.expectEqual(@as(u32, 2), q_meta.rows);
    try std.testing.expectEqual(@as(u32, 1), k_meta.rows);
    try std.testing.expectEqual(@as(u32, 1), v_meta.rows);
    try std.testing.expectEqualSlices(u8, scales[0..2], q_meta.block_scales_data.?[0..q_meta.block_scales_len]);
    try std.testing.expectEqualSlices(u8, scales[2..3], k_meta.block_scales_data.?[0..k_meta.block_scales_len]);
    try std.testing.expectEqualSlices(u8, scales[3..4], v_meta.block_scales_data.?[0..v_meta.block_scales_len]);
}
