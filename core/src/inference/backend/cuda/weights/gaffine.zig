//! Grouped-affine weight helpers for the CUDA inference backend.

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

pub fn uploadLinearWeightGroupedAffineU4(
    device: *compute.cuda.Device,
    src: *const Tensor,
    input_dim: usize,
) !LinearWeight {
    if (src.n_dims != 2) return error.UnsupportedModel;
    if (src.shape[0] <= 0 or src.shape[1] <= 0) return error.InvalidArgument;
    if (src.data_ptr == null) return error.InvalidArgument;
    const gaffine = src.gaffine orelse return error.UnsupportedModel;
    const out_dim: usize = @intCast(src.shape[0]);
    const in_dim: usize = @intCast(src.shape[1]);

    if (in_dim != input_dim) {
        log.warn("inference", "CUDA grouped-affine U4 orientation unsupported", .{
            .rows = out_dim,
            .cols = in_dim,
            .input_dim = input_dim,
        });
        return error.UnsupportedModel;
    }
    if (in_dim == 0 or out_dim == 0) return error.InvalidArgument;
    if ((in_dim % 8) != 0) return error.UnsupportedModel;
    if (gaffine.group_size == 0 or (in_dim % gaffine.group_size) != 0 or (gaffine.group_size % 8) != 0) {
        return error.UnsupportedModel;
    }

    const packed_words_per_row = in_dim / 8;
    const groups_per_row = in_dim / gaffine.group_size;
    const packed_words = std.math.mul(usize, out_dim, packed_words_per_row) catch return error.InvalidArgument;
    const packed_bytes = std.math.mul(usize, packed_words, @sizeOf(u32)) catch return error.InvalidArgument;
    const sb_count = std.math.mul(usize, out_dim, groups_per_row) catch return error.InvalidArgument;
    const sb_bytes = std.math.mul(usize, sb_count, @sizeOf(u16)) catch return error.InvalidArgument;
    if (src.data_size < packed_bytes) return error.InvalidArgument;
    if (gaffine.scales.len < sb_bytes or gaffine.biases.len < sb_bytes) return error.InvalidArgument;

    const scales_dtype_tag: u32 = switch (gaffine.scales_dtype) {
        .f16 => gaffine_scales_dtype_f16,
        .bf16 => gaffine_scales_dtype_bf16,
        else => return error.UnsupportedModel,
    };

    var packed_dev = try device.allocBuffer(packed_bytes);
    errdefer packed_dev.deinit(device);
    var scales_dev = try device.allocBuffer(sb_bytes);
    errdefer scales_dev.deinit(device);
    var biases_dev = try device.allocBuffer(sb_bytes);
    errdefer biases_dev.deinit(device);

    const packed_host = src.data()[0..packed_bytes];
    try packed_dev.upload(device, packed_host);
    try scales_dev.upload(device, gaffine.scales[0..sb_bytes]);
    try biases_dev.upload(device, gaffine.biases[0..sb_bytes]);

    return .{
        .gaffine_u4 = .{
            .rows = in_dim,
            .cols = out_dim,
            .packed_data = packed_dev,
            .scales = scales_dev,
            .biases = biases_dev,
            .group_size = @intCast(gaffine.group_size),
            .scales_dtype_tag = scales_dtype_tag,
        },
    };
}

pub fn uploadLinearWeightGroupedAffineU8(
    device: *compute.cuda.Device,
    src: *const Tensor,
    input_dim: usize,
) !LinearWeight {
    if (src.n_dims != 2) return error.UnsupportedModel;
    if (src.shape[0] <= 0 or src.shape[1] <= 0) return error.InvalidArgument;
    if (src.data_ptr == null) return error.InvalidArgument;
    const gaffine = src.gaffine orelse return error.UnsupportedModel;
    const out_dim: usize = @intCast(src.shape[0]);
    const in_dim: usize = @intCast(src.shape[1]);

    if (in_dim != input_dim) {
        log.warn("inference", "CUDA grouped-affine U8 orientation unsupported", .{
            .rows = out_dim,
            .cols = in_dim,
            .input_dim = input_dim,
        });
        return error.UnsupportedModel;
    }
    if (in_dim == 0 or out_dim == 0) return error.InvalidArgument;
    if ((in_dim % 4) != 0) return error.UnsupportedModel;
    if (gaffine.group_size == 0 or (in_dim % gaffine.group_size) != 0 or (gaffine.group_size % 4) != 0) {
        return error.UnsupportedModel;
    }

    const packed_words_per_row = in_dim / 4;
    const groups_per_row = in_dim / gaffine.group_size;
    const packed_words = std.math.mul(usize, out_dim, packed_words_per_row) catch return error.InvalidArgument;
    const packed_bytes = std.math.mul(usize, packed_words, @sizeOf(u32)) catch return error.InvalidArgument;
    const sb_count = std.math.mul(usize, out_dim, groups_per_row) catch return error.InvalidArgument;
    const sb_bytes = std.math.mul(usize, sb_count, @sizeOf(u16)) catch return error.InvalidArgument;
    if (src.data_size < packed_bytes) return error.InvalidArgument;
    if (gaffine.scales.len < sb_bytes or gaffine.biases.len < sb_bytes) return error.InvalidArgument;

    const scales_dtype_tag: u32 = switch (gaffine.scales_dtype) {
        .f16 => gaffine_scales_dtype_f16,
        .bf16 => gaffine_scales_dtype_bf16,
        else => return error.UnsupportedModel,
    };

    var packed_dev = try device.allocBuffer(packed_bytes);
    errdefer packed_dev.deinit(device);
    var scales_dev = try device.allocBuffer(sb_bytes);
    errdefer scales_dev.deinit(device);
    var biases_dev = try device.allocBuffer(sb_bytes);
    errdefer biases_dev.deinit(device);

    const packed_host = src.data()[0..packed_bytes];
    try packed_dev.upload(device, packed_host);
    try scales_dev.upload(device, gaffine.scales[0..sb_bytes]);
    try biases_dev.upload(device, gaffine.biases[0..sb_bytes]);

    return .{
        .gaffine_u8 = .{
            .rows = in_dim,
            .cols = out_dim,
            .packed_data = packed_dev,
            .scales = scales_dev,
            .biases = biases_dev,
            .group_size = @intCast(gaffine.group_size),
            .scales_dtype_tag = scales_dtype_tag,
        },
    };
}

pub fn decodeGaffineRow(weight: *const Tensor, row: usize, out: []f32) !void {
    if (weight.dtype != .grouped_affine_u4 and weight.dtype != .grouped_affine_u8) return error.InvalidArgument;
    const gaffine = weight.gaffine orelse return error.InvalidArgument;
    if (weight.n_dims != 2) return error.InvalidArgument;
    if (weight.shape[0] <= 0 or weight.shape[1] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(weight.shape[0]);
    const cols: usize = @intCast(weight.shape[1]);
    if (row >= rows) return error.InvalidArgument;
    if (out.len != cols) return error.InvalidArgument;

    const values_per_word: usize = if (weight.dtype == .grouped_affine_u4) 8 else 4;
    const bits: u5 = if (weight.dtype == .grouped_affine_u4) 4 else 8;
    const mask: u32 = if (weight.dtype == .grouped_affine_u4) 0xF else 0xFF;
    if (values_per_word == 0 or cols % values_per_word != 0) return error.InvalidArgument;
    if (gaffine.group_size == 0 or cols % gaffine.group_size != 0) return error.InvalidArgument;

    const packed_stride = cols / values_per_word;
    const group_stride = cols / gaffine.group_size;
    if (group_stride == 0) return error.InvalidArgument;
    const packed_words = weight.asSliceUnaligned(u32);
    const required_words = std.math.mul(usize, rows, packed_stride) catch return error.InvalidArgument;
    if (packed_words.len < required_words) return error.InvalidArgument;

    var current_group_idx: usize = std.math.maxInt(usize);
    var current_scale: f32 = 0.0;
    var current_bias: f32 = 0.0;
    var col: usize = 0;
    while (col < cols) : (col += 1) {
        const group_idx = col / gaffine.group_size;
        if (group_idx != current_group_idx) {
            current_group_idx = group_idx;
            const sb_idx = row * group_stride + group_idx;
            current_scale = try gaffineScaleBiasToF32(gaffine.scales_dtype, gaffine.scales, sb_idx);
            current_bias = try gaffineScaleBiasToF32(gaffine.scales_dtype, gaffine.biases, sb_idx);
        }

        const pack_idx = row * packed_stride + (col / values_per_word);
        const packed_word = packed_words[pack_idx];
        const shift: u5 = @intCast((col % values_per_word) * bits);
        const quant = (packed_word >> shift) & mask;
        out[col] = @as(f32, @floatFromInt(quant)) * current_scale + current_bias;
    }
}

pub fn gaffineScaleBiasToF32(scales_dtype: tensor.DType, bytes: []const u8, idx: usize) !f32 {
    const byte_offset = std.math.mul(usize, idx, @sizeOf(u16)) catch return error.InvalidArgument;
    if (byte_offset + @sizeOf(u16) > bytes.len) return error.InvalidArgument;

    const v = std.mem.readInt(u16, bytes[byte_offset..][0..2], .little);
    return switch (scales_dtype) {
        .f16 => dtype.fp16ToF32(v),
        .bf16 => dtype.bf16ToF32(v),
        else => error.InvalidArgument,
    };
}

pub fn gaffineValueAt(weight: *const Tensor, row: usize, col: usize) !f32 {
    if (weight.dtype != .grouped_affine_u4 and weight.dtype != .grouped_affine_u8) return error.InvalidArgument;
    const gaffine = weight.gaffine orelse return error.InvalidArgument;
    if (weight.n_dims != 2) return error.InvalidArgument;
    if (weight.shape[0] <= 0 or weight.shape[1] <= 0) return error.InvalidArgument;

    const rows: usize = @intCast(weight.shape[0]);
    const cols: usize = @intCast(weight.shape[1]);
    if (row >= rows or col >= cols) return error.InvalidArgument;

    const values_per_word: usize = if (weight.dtype == .grouped_affine_u4) 8 else 4;
    const bits: u5 = if (weight.dtype == .grouped_affine_u4) 4 else 8;
    const mask: u32 = if (weight.dtype == .grouped_affine_u4) 0xF else 0xFF;
    if (values_per_word == 0 or cols % values_per_word != 0) return error.InvalidArgument;
    if (gaffine.group_size == 0 or cols % gaffine.group_size != 0) return error.InvalidArgument;

    const packed_stride = cols / values_per_word;
    const group_stride = cols / gaffine.group_size;
    if (group_stride == 0) return error.InvalidArgument;

    const pack_idx = row * packed_stride + (col / values_per_word);
    const pack_byte_offset = std.math.mul(usize, pack_idx, @sizeOf(u32)) catch return error.InvalidArgument;
    if (pack_byte_offset + @sizeOf(u32) > weight.data().len) return error.InvalidArgument;
    const packed_word = std.mem.readInt(u32, weight.data()[pack_byte_offset..][0..4], .little);
    const shift: u5 = @intCast((col % values_per_word) * bits);
    const quant = (packed_word >> shift) & mask;

    const group_idx = col / gaffine.group_size;
    const sb_idx = row * group_stride + group_idx;
    const scale = try gaffineScaleBiasToF32(gaffine.scales_dtype, gaffine.scales, sb_idx);
    const bias = try gaffineScaleBiasToF32(gaffine.scales_dtype, gaffine.biases, sb_idx);

    return @as(f32, @floatFromInt(quant)) * scale + bias;
}
