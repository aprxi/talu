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

    const scale_cols: usize = @intCast(meta.scale_cols);
    if (scale_cols == 0) return error.InvalidArgument;
    const scale_byte_count = std.math.mul(usize, out_dim, scale_cols) catch return error.InvalidArgument;
    const scales_ptr = meta.block_scales_data orelse return error.MissingScales;
    if (meta.block_scales_len < scale_byte_count) return error.InvalidArgument;

    var buffer = try device.allocBuffer(weight_byte_count);
    errdefer buffer.deinit(device);
    try buffer.upload(device, src_bytes[0..weight_byte_count]);

    var scales_buffer = try device.allocBuffer(scale_byte_count);
    errdefer scales_buffer.deinit(device);
    try scales_buffer.upload(device, scales_ptr[0..scale_byte_count]);

    var scales_lt_buffer: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 };
    if (meta.group_size == 16) {
        const required_scale_cols = (in_dim + 15) / 16;
        if (scale_cols < required_scale_cols) return error.InvalidArgument;
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
        .scale_cols = meta.scale_cols,
        .group_size = meta.group_size,
        .weight_global_scale = meta.weight_global_scale,
    } };
}
