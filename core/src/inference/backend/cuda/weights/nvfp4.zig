//! NVFP4 weight helpers for the CUDA inference backend.

const std = @import("std");
const compute = @import("compute_pkg");
const tensor = @import("compute_pkg").tensor;
const dtype = @import("compute_pkg").dtype;
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

fn copyNvfp4ScaleRowsCompact(
    dst: []u8,
    src: []const u8,
    row_count: usize,
    source_scale_cols: usize,
    required_scale_cols: usize,
) !void {
    if (required_scale_cols == 0 or source_scale_cols < required_scale_cols) return error.InvalidArgument;
    const required_dst_len = std.math.mul(usize, row_count, required_scale_cols) catch return error.InvalidArgument;
    const required_src_len = std.math.mul(usize, row_count, source_scale_cols) catch return error.InvalidArgument;
    if (dst.len < required_dst_len or src.len < required_src_len) return error.InvalidArgument;

    for (0..row_count) |row| {
        const src_start = row * source_scale_cols;
        const dst_start = row * required_scale_cols;
        @memcpy(dst[dst_start .. dst_start + required_scale_cols], src[src_start .. src_start + required_scale_cols]);
    }
}

// --- Shared types from engine_types.zig ---
const engine_types = @import("../runtime/root.zig");
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

pub fn uploadLinearWeightNvfp4(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    src: *const Tensor,
    input_dim: usize,
) !LinearWeight {
    if (src.n_dims != 2) return error.UnsupportedModel;
    const meta = src.nvfp4 orelse return error.UnsupportedModel;

    const out_dim: usize = @intCast(src.shape[0]);
    const in_dim: usize = @intCast(src.shape[1]);
    if (in_dim != input_dim or out_dim == 0 or in_dim == 0) return error.UnsupportedModel;

    const packed_cols: usize = @intCast(meta.packed_cols);
    if (packed_cols == 0 or packed_cols * 2 != in_dim) return error.InvalidArgument;
    const weight_byte_count = std.math.mul(usize, out_dim, packed_cols) catch return error.InvalidArgument;
    const src_bytes = src.data();
    if (src_bytes.len < weight_byte_count) return error.InvalidArgument;

    const group_size: usize = @intCast(meta.group_size);
    if (group_size == 0) return error.InvalidArgument;
    const required_scale_cols = std.math.divCeil(usize, in_dim, group_size) catch return error.InvalidArgument;
    const scale_cols: usize = @intCast(meta.scale_cols);
    if (scale_cols < required_scale_cols) return error.InvalidArgument;
    const scale_byte_count = std.math.mul(usize, out_dim, scale_cols) catch return error.InvalidArgument;
    const scales_ptr = meta.block_scales_data orelse return error.MissingScales;
    if (meta.block_scales_len < scale_byte_count) return error.InvalidArgument;

    var buffer = try device.allocBuffer(weight_byte_count);
    errdefer buffer.deinit(device);
    try buffer.upload(device, src_bytes[0..weight_byte_count]);

    const compact_scale_byte_count = std.math.mul(usize, out_dim, required_scale_cols) catch return error.InvalidArgument;
    var compact_scales: ?[]u8 = null;
    defer if (compact_scales) |buf| allocator.free(buf);
    const upload_scales = if (scale_cols == required_scale_cols)
        scales_ptr[0..compact_scale_byte_count]
    else blk: {
        const buf = try allocator.alloc(u8, compact_scale_byte_count);
        try copyNvfp4ScaleRowsCompact(buf, scales_ptr[0..scale_byte_count], out_dim, scale_cols, required_scale_cols);
        compact_scales = buf;
        break :blk buf;
    };

    var scales_buffer = try device.allocBuffer(compact_scale_byte_count);
    errdefer scales_buffer.deinit(device);
    try scales_buffer.upload(device, upload_scales);

    var scales_lt_buffer: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 };
    if (meta.group_size == 16) {
        const padded_scale_bytes = engine_types.Nvfp4LinearWeight.cublasLtScaleTensorSize(in_dim, out_dim);
        const padded_sf_k = engine_types.Nvfp4LinearWeight.roundoff(required_scale_cols, 4);
        const n_col_tiles = padded_sf_k / 4;

        const interleaved = try allocator.alloc(u8, padded_scale_bytes);
        defer allocator.free(interleaved);
        @memset(interleaved, 0);

        for (0..out_dim) |m| {
            for (0..required_scale_cols) |k| {
                const src_idx = m * scale_cols + k;
                const dst_idx = (m / 128) * n_col_tiles * 512 +
                    (k / 4) * 512 +
                    (m % 32) * 16 +
                    ((m % 128) / 32) * 4 +
                    (k % 4);
                interleaved[dst_idx] = scales_ptr[src_idx];
            }
        }

        scales_lt_buffer = try device.allocBuffer(padded_scale_bytes);
        errdefer scales_lt_buffer.deinit(device);
        try scales_lt_buffer.upload(device, interleaved);
    }

    return .{ .nvfp4 = .{
        .rows = in_dim,
        .cols = out_dim,
        .buffer = buffer,
        .scales_buffer = scales_buffer,
        .scales_lt_buffer = scales_lt_buffer,
        .packed_cols = meta.packed_cols,
        .scale_cols = @intCast(required_scale_cols),
        .group_size = meta.group_size,
        .weight_global_scale = meta.weight_global_scale,
    } };
}

test "copyNvfp4ScaleRowsCompact removes padded source stride" {
    const src = [_]u8{
        1, 2, 3, 9, 9,
        4, 5, 6, 8, 8,
    };
    var dst = [_]u8{0} ** 6;
    try copyNvfp4ScaleRowsCompact(&dst, &src, 2, 5, 3);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4, 5, 6 }, &dst);
}

test "copyNvfp4ScaleRowsCompact rejects undersized scale rows" {
    const src = [_]u8{ 1, 2, 3, 4 };
    var dst = [_]u8{0} ** 4;
    try std.testing.expectError(error.InvalidArgument, copyNvfp4ScaleRowsCompact(&dst, &src, 2, 1, 2));
}
