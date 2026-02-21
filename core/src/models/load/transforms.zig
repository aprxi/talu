//! Weight transform helpers for the model loader.
//!
//! Shared routines for orienting weights, dequantizing, and fusing projections.

const std = @import("std");
const tensor = @import("../../tensor.zig");
const dtype = @import("../../dtype.zig");
const st_loader = @import("../../io/safetensors/root.zig");
const log = @import("../../log.zig");
const op_types = @import("../op_types.zig");

const Tensor = tensor.Tensor;
const ModelConfig = tensor.ModelConfig;
const DType = dtype.DType;

/// Loader-side safety bound for grouped-affine metadata.
/// Shared contract constant also asserted by CPU inference backend.
pub const MAX_SUPPORTED_GAFFINE_GROUPS: usize = op_types.MAX_SUPPORTED_GAFFINE_GROUPS;

pub fn maybeConcatQkvWeights(allocator: std.mem.Allocator, q: Tensor, k: Tensor, v: Tensor) ?Tensor {
    // Only support F32 fusion - BF16 weights have different layout ([out, in] vs [in, out])
    if (q.dtype != .f32 or k.dtype != .f32 or v.dtype != .f32) return null;
    if (q.n_dims != 2 or k.n_dims != 2 or v.n_dims != 2) return null;
    if (q.shape[0] == 0) return null;
    const rows = q.shape[0];
    if (k.shape[0] != rows or v.shape[0] != rows) return null;

    const q_cols = q.shape[1];
    const k_cols = k.shape[1];
    const v_cols = v.shape[1];
    const total_cols = q_cols + k_cols + v_cols;

    const concat_buf = allocator.alloc(f32, @intCast(rows * total_cols)) catch return null;
    errdefer allocator.free(concat_buf);

    const q_data = q.asSlice(f32);
    const k_data = k.asSlice(f32);
    const v_data = v.asSlice(f32);

    const rows_usize: usize = @intCast(rows);
    const total_cols_usize: usize = @intCast(total_cols);
    const q_cols_usize: usize = @intCast(q_cols);
    const k_cols_usize: usize = @intCast(k_cols);
    const v_cols_usize: usize = @intCast(v_cols);
    for (0..rows_usize) |r| {
        const dst_row = concat_buf[r * total_cols_usize ..][0..total_cols_usize];
        const q_row = q_data[r * q_cols_usize ..][0..q_cols_usize];
        const k_row = k_data[r * k_cols_usize ..][0..k_cols_usize];
        const v_row = v_data[r * v_cols_usize ..][0..v_cols_usize];
        std.mem.copyForwards(f32, dst_row[0..q_cols_usize], q_row);
        std.mem.copyForwards(f32, dst_row[q_cols_usize .. q_cols_usize + k_cols_usize], k_row);
        std.mem.copyForwards(f32, dst_row[q_cols_usize + k_cols_usize ..], v_row);
    }

    return tensor.Tensor.view2DSlice(concat_buf, @intCast(rows), @intCast(total_cols));
}

// =============================================================================
// Grouped-Affine Inference (shared by orientWeight and orientEmbedding)
// =============================================================================

/// Result of inferring grouped-affine quantization parameters from scales/biases shapes.
const GaffineInferResult = struct {
    dtype: DType,
    values_per_word: usize,
    group_size: usize,
    scales_bytes: []const u8,
    biases_bytes: []const u8,
    scales_dtype: DType,
    shape_override: [4]usize,
};

/// Infer grouped-affine quantization parameters from a tensor and its scales/biases.
/// Returns null if inference fails or the tensor is not grouped-affine.
fn inferGaffineParams(
    st: *st_loader.UnifiedSafeTensors,
    name: []const u8,
    t: *Tensor,
    expected_dim: usize,
) ?GaffineInferResult {
    if (t.dtype != .grouped_affine_u4 and t.dtype != .grouped_affine_u8) return null;

    const base = if (std.mem.endsWith(u8, name, ".weight"))
        name[0 .. name.len - ".weight".len]
    else
        name;
    const packed_shape = t.shape[0..@intCast(t.n_dims)];
    if (packed_shape.len != 2) return null;
    const out_features: usize = @intCast(packed_shape[0]);
    const in_packed: usize = @intCast(packed_shape[1]);

    // Get scales and biases bytes
    const scales_bytes = st.tryGetBytes(base, ".scales") orelse return null;
    const biases_bytes = st.tryGetBytes(base, ".biases") orelse return null;
    if (out_features == 0) return null;
    if (scales_bytes.len % (out_features * 2) != 0) return null;
    if (biases_bytes.len != scales_bytes.len) return null;

    // Get scales dtype (F16 or BF16)
    var scales_name_buf: [256]u8 = undefined;
    const scales_name = std.fmt.bufPrint(&scales_name_buf, "{s}.scales", .{base}) catch return null;
    const scales_tensor = st.getTensor(scales_name, null) catch return null;
    const scales_dtype = scales_tensor.dtype;
    if (scales_dtype != .f16 and scales_dtype != .bf16) return null;

    // scales are f16/bf16 (2 bytes each), shape is [out_features, n_groups]
    const n_groups = scales_bytes.len / (out_features * 2);
    if (n_groups == 0) return null;

    // Auto-detect bits from relationship:
    // unpacked_dim = packed_dim * values_per_word
    // n_groups = unpacked_dim / group_size
    // For 4-bit: values_per_word=8, for 8-bit: values_per_word=4
    const group_size_if_4bit = if (n_groups > 0) (in_packed * 8) / n_groups else 0;
    const group_size_if_8bit = if (n_groups > 0) (in_packed * 4) / n_groups else 0;

    // Typical group sizes are 32, 64, or 128
    const valid_4bit = (group_size_if_4bit == 32 or group_size_if_4bit == 64 or group_size_if_4bit == 128);
    const valid_8bit = (group_size_if_8bit == 32 or group_size_if_8bit == 64 or group_size_if_8bit == 128);

    // Calculate unpacked dimensions for both interpretations
    const unpacked_4bit = in_packed * 8;
    const unpacked_8bit = in_packed * 4;

    // Use expected_dim to disambiguate when both group sizes are valid
    // This handles mixed quantization models (e.g. 8-bit attn/embed, 4-bit MoE)
    const is_4bit = if (valid_4bit and valid_8bit)
        (unpacked_4bit == expected_dim) // Match expected dimension
    else
        valid_4bit; // Only one is valid, use that

    const actual_dtype: DType = if (is_4bit) .grouped_affine_u4 else .grouped_affine_u8;
    const values_per_word: usize = if (is_4bit) 8 else 4;
    const group_size: usize = if (is_4bit) group_size_if_4bit else group_size_if_8bit;

    log.trace("load", "Inferred gaffine params", .{
        .name = name,
        .n_groups = n_groups,
        .expected_dim = expected_dim,
        .unpacked_4bit = unpacked_4bit,
        .unpacked_8bit = unpacked_8bit,
        .bits = if (is_4bit) @as(u8, 4) else 8,
        .group_size = group_size,
    }, @src());

    return .{
        .dtype = actual_dtype,
        .values_per_word = values_per_word,
        .group_size = group_size,
        .scales_bytes = scales_bytes,
        .biases_bytes = biases_bytes,
        .scales_dtype = scales_dtype,
        .shape_override = .{ out_features, in_packed * values_per_word, 0, 0 },
    };
}

/// Apply inferred gaffine params to a tensor, with validation.
fn applyGaffineParams(t: *Tensor, params: GaffineInferResult, name: []const u8) !void {
    // Validate number of groups is within kernel limits
    const k_unpacked = params.shape_override[1];
    const n_groups_actual = k_unpacked / params.group_size;
    if (n_groups_actual > MAX_SUPPORTED_GAFFINE_GROUPS) {
        log.err("load", "Too many groups in tensor", .{
            .name = name,
            .n_groups = n_groups_actual,
            .k = k_unpacked,
            .group_size = params.group_size,
            .max_groups = MAX_SUPPORTED_GAFFINE_GROUPS,
        }, @src());
        return error.TooManyGroups;
    }

    t.dtype = params.dtype;
    for (params.shape_override, 0..) |dim_size, dim_idx| {
        t.shape[dim_idx] = @intCast(dim_size);
    }
    // Note: scales/biases are const slices from safetensors mmap, but GroupedAffineMeta
    // uses []u8 for historical reasons. This is safe since we never write to these.
    t.gaffine = .{
        .scales = @constCast(params.scales_bytes),
        .biases = @constCast(params.biases_bytes),
        .group_size = params.group_size,
        .scales_dtype = params.scales_dtype,
    };
}

pub fn bytesToU16Slice(allocator: std.mem.Allocator, bytes: []const u8, owned: *?[]u16) ![]const u16 {
    if (bytes.len % @sizeOf(u16) != 0) return error.InvalidShape;
    if (@intFromPtr(bytes.ptr) % @alignOf(u16) == 0) {
        const slice = std.mem.bytesAsSlice(u16, bytes);
        return @alignCast(slice);
    }
    const len = bytes.len / @sizeOf(u16);
    const aligned_u16 = try allocator.alloc(u16, len);
    @memcpy(std.mem.sliceAsBytes(aligned_u16), bytes);
    owned.* = aligned_u16;
    return aligned_u16;
}

/// Convert raw bytes to u32 slice, handling potentially unaligned data.
/// SafeTensors files may have tensor data at arbitrary offsets determined
/// by JSON header length, so data may not be aligned to 4 bytes.
fn bytesToU32Slice(allocator: std.mem.Allocator, bytes: []const u8, owned: *?[]u32) ![]const u32 {
    if (bytes.len % @sizeOf(u32) != 0) return error.InvalidShape;
    if (@intFromPtr(bytes.ptr) % @alignOf(u32) == 0) {
        const slice = std.mem.bytesAsSlice(u32, bytes);
        return @alignCast(slice);
    }
    // Data is unaligned - copy to aligned buffer
    const len = bytes.len / @sizeOf(u32);
    const aligned_u32 = try allocator.alloc(u32, len);
    @memcpy(std.mem.sliceAsBytes(aligned_u32), bytes);
    owned.* = aligned_u32;
    return aligned_u32;
}

pub fn maybeConcatGateUpWeights(allocator: std.mem.Allocator, gate: Tensor, up: Tensor) ?Tensor {
    // Only support F32 fusion - BF16 weights have different layout ([out, in] vs [in, out])
    if (gate.dtype != .f32 or up.dtype != .f32) return null;
    if (gate.n_dims != 2 or up.n_dims != 2) return null;
    if (gate.shape[0] == 0) return null;
    const rows: usize = @intCast(gate.shape[0]);
    if (up.shape[0] != gate.shape[0] or gate.shape[1] != up.shape[1]) return null;

    const cols: usize = @intCast(gate.shape[1]);
    const total_cols = cols * 2;

    const concat_buf = allocator.alloc(f32, rows * total_cols) catch return null;
    errdefer allocator.free(concat_buf);

    const gate_data = gate.asSlice(f32);
    const up_data = up.asSlice(f32);

    for (0..rows) |r| {
        const dst_row = concat_buf[r * total_cols ..][0..total_cols];
        const gate_row = gate_data[r * cols ..][0..cols];
        const up_row = up_data[r * cols ..][0..cols];
        std.mem.copyForwards(f32, dst_row[0..cols], gate_row);
        std.mem.copyForwards(f32, dst_row[cols..], up_row);
    }

    return tensor.Tensor.view2DSlice(concat_buf, rows, total_cols);
}

pub fn orientWeight(allocator: std.mem.Allocator, st: *st_loader.UnifiedSafeTensors, name: []const u8, expected_in: usize, config: ModelConfig) !Tensor {
    _ = config; // Not used currently - we use expected_in for disambiguation
    var weight_tensor = try st.getTensor(name, null);
    log.debug("load", "Orient weight", .{
        .name = name,
        .dtype = @tagName(weight_tensor.dtype),
    }, @src());
    log.trace("load", "Orient weight details", .{
        .name = name,
        .expected_in = expected_in,
        .shape_0 = weight_tensor.shape[0],
        .shape_1 = weight_tensor.shape[1],
    }, @src());

    // U32 from safetensors maps to grouped_affine_u4 by default
    // For models with mixed quantization, auto-detect bits from scales shape
    if (inferGaffineParams(st, name, &weight_tensor, expected_in)) |params| {
        try applyGaffineParams(&weight_tensor, params, name);
        return weight_tensor;
    } else if (weight_tensor.dtype == .grouped_affine_u4 or weight_tensor.dtype == .grouped_affine_u8) {
        // Gaffine tensor but inference failed - missing scales/biases
        return error.MissingScales;
    }
    // Handle FP8 E4M3 weights - dequantize to BF16
    if (weight_tensor.dtype == .f8_e4m3) {
        const base = if (std.mem.endsWith(u8, name, ".weight"))
            name[0 .. name.len - ".weight".len]
        else
            name;

        // Try to get scale tensor (need shape info for per-block detection)
        var scale_name_buf: [256]u8 = undefined;
        const scale_name = std.fmt.bufPrint(&scale_name_buf, "{s}.weight_scale_inv", .{base}) catch {
            return dequantizeFp8Weight(allocator, weight_tensor, 1.0, expected_in);
        };

        const scale_tensor = st.getTensor(scale_name, null) catch {
            log.trace("load", "FP8: no weight_scale_inv found, using 1.0", .{}, @src());
            return dequantizeFp8Weight(allocator, weight_tensor, 1.0, expected_in);
        };

        // Check if scale is 2D (per-block) or scalar/1D
        if (scale_tensor.n_dims == 2) {
            // Per-block FP8 quantization (e.g., Qwen3 MoE FP8 models)
            const dequantized = try dequantizeFp8WeightPerBlock(allocator, weight_tensor, scale_tensor, expected_in);
            // Apply same orientation as regular BF16 weights
            return orientWeightTyped(allocator, dequantized, expected_in);
        }

        // Scalar scale (1D or 0D)
        const scale_inv_bytes = scale_tensor.data();
        if (scale_inv_bytes.len >= 2) {
            const scale_inv_bf16 = std.mem.bytesAsValue(u16, scale_inv_bytes[0..2]).*;
            const scale_inv = dtype.bf16ToF32(scale_inv_bf16);
            log.trace("load", "FP8 scale", .{ .scale_inv = scale_inv }, @src());
            const dequantized = try dequantizeFp8Weight(allocator, weight_tensor, scale_inv, expected_in);
            return orientWeightTyped(allocator, dequantized, expected_in);
        }
        const dequantized = try dequantizeFp8Weight(allocator, weight_tensor, 1.0, expected_in);
        return orientWeightTyped(allocator, dequantized, expected_in);
    }

    return switch (weight_tensor.dtype) {
        .f32 => orientWeightF32(allocator, weight_tensor, expected_in),
        .f16, .bf16 => orientWeightTyped(allocator, weight_tensor, expected_in),
        else => weight_tensor,
    };
}

pub fn orientWeightF32(allocator: std.mem.Allocator, t: Tensor, expected_in: usize) !Tensor {
    if (t.n_dims == 1) return t;
    if (t.n_dims != 2) return error.InvalidShape;
    const rows = t.shape[0];
    const cols = t.shape[1];
    if (cols != expected_in and rows != expected_in) return error.InvalidShape;

    // SafeTensors from PyTorch store Linear weights as [out, in]; our matmul expects [in, out].
    if (cols == expected_in) {
        const transposed = try tensor.OwnedTensor.init(allocator, .f32, &.{ @intCast(cols), @intCast(rows) });
        const src_f32 = t.asSlice(f32);
        const dst_f32 = transposed.asSlice(f32);
        const rows_usize: usize = @intCast(rows);
        const cols_usize: usize = @intCast(cols);
        for (0..rows_usize) |r| {
            for (0..cols_usize) |c| {
                dst_f32[c * rows_usize + r] = src_f32[r * cols_usize + c];
            }
        }
        // Return view - arena owns memory
        return transposed.view();
    }

    return t;
}

pub fn orientWeightTyped(allocator: std.mem.Allocator, t: Tensor, expected_in: usize) !Tensor {
    if (t.n_dims == 1) return t;
    if (t.n_dims != 2) return error.InvalidShape;
    const rows: usize = @intCast(t.shape[0]);
    const cols: usize = @intCast(t.shape[1]);
    if (cols != expected_in and rows != expected_in) return error.InvalidShape;

    // BF16/F16 matmul expects weights in [out, in] format.
    // SafeTensors from PyTorch typically store Linear weights as [out, in].
    // If cols == expected_in, the tensor is already in the correct [out, in] format.
    // If rows == expected_in, the tensor is transposed [in, out] and needs fixing.
    if (rows == expected_in and cols != expected_in) {
        // Tensor is in [in, out] format - transpose to [out, in] for cache efficiency.
        // This is critical for performance: strided access patterns destroy cache utilization.
        return transposeToOwned(allocator, t, t.dtype);
    }

    // Already in [out, in] format or ambiguous (square matrix)
    return t;
}

/// Dequantize FP8 E4M3 weight tensor to BF16
fn dequantizeFp8Weight(allocator: std.mem.Allocator, t: Tensor, scale_inv: f32, expected_in: usize) !Tensor {
    if (t.n_dims != 2) {
        log.trace("load", "FP8: expected 2D tensor", .{ .n_dims = t.n_dims }, @src());
        return error.InvalidShape;
    }

    const rows = t.shape[0];
    const cols = t.shape[1];

    // Validate shape (either [out, in] or [in, out])
    if (cols != expected_in and rows != expected_in) {
        log.trace("load", "FP8: shape mismatch", .{ .rows = rows, .cols = cols, .expected_in = expected_in }, @src());
        return error.InvalidShape;
    }

    // Allocate BF16 output tensor (same shape as input)
    const owned = try tensor.OwnedTensor.init(allocator, .bf16, &.{ @intCast(rows), @intCast(cols) });
    const src_bytes = t.data()[0..@as(usize, @intCast(rows * cols))];
    const dst_u16 = owned.asSlice(u16);

    // Dequantize FP8 to BF16
    dtype.dequantizeFp8E4M3ToBf16(src_bytes, scale_inv, dst_u16);

    log.trace("load", "FP8 dequantized to BF16", .{ .rows = rows, .cols = cols }, @src());

    return owned.view();
}

/// Dequantize FP8 E4M3 weight tensor to BF16 with per-block scales.
/// Each block in the weight matrix has its own scale value from a 2D scale tensor.
/// Used by Qwen3 MoE FP8 models where scales have shape [rows/block_size, cols/block_size].
fn dequantizeFp8WeightPerBlock(
    allocator: std.mem.Allocator,
    t: Tensor,
    scale_tensor: Tensor,
    expected_in: usize,
) !Tensor {
    if (t.n_dims != 2) {
        log.trace("load", "FP8 per-block: expected 2D weight tensor", .{ .n_dims = t.n_dims }, @src());
        return error.InvalidShape;
    }
    if (scale_tensor.n_dims != 2) {
        log.trace("load", "FP8 per-block: expected 2D scale tensor", .{ .n_dims = scale_tensor.n_dims }, @src());
        return error.InvalidShape;
    }

    const rows: usize = @intCast(t.shape[0]);
    const cols: usize = @intCast(t.shape[1]);
    const scale_rows: usize = @intCast(scale_tensor.shape[0]);
    const scale_cols: usize = @intCast(scale_tensor.shape[1]);

    // Validate shape
    if (cols != expected_in and rows != expected_in) {
        log.trace("load", "FP8 per-block: shape mismatch", .{ .rows = rows, .cols = cols, .expected_in = expected_in }, @src());
        return error.InvalidShape;
    }

    // Calculate block size (each scale value covers a block)
    if (rows % scale_rows != 0 or cols % scale_cols != 0) {
        log.trace("load", "FP8 per-block: scale shape doesn't divide weight shape", .{
            .weight_rows = rows,
            .weight_cols = cols,
            .scale_rows = scale_rows,
            .scale_cols = scale_cols,
        }, @src());
        return error.InvalidShape;
    }

    const block_row_size = rows / scale_rows;
    const block_col_size = cols / scale_cols;

    log.trace("load", "FP8 per-block dequantizing", .{
        .weight_rows = rows,
        .weight_cols = cols,
        .scale_rows = scale_rows,
        .scale_cols = scale_cols,
        .block_row_size = block_row_size,
        .block_col_size = block_col_size,
    }, @src());

    // Allocate BF16 output tensor
    const owned = try tensor.OwnedTensor.init(allocator, .bf16, &.{ rows, cols });
    const src_bytes = t.data()[0 .. rows * cols];
    const dst_u16 = owned.asSlice(u16);

    // Get scale values (BF16 format)
    const scale_data = scale_tensor.asSliceUnaligned(u16);

    // Dequantize each element with its block's scale
    for (0..rows) |r| {
        const scale_row = r / block_row_size;
        for (0..cols) |c| {
            const scale_col = c / block_col_size;
            const scale_idx = scale_row * scale_cols + scale_col;
            const scale_inv = dtype.bf16ToF32(scale_data[scale_idx]);

            const src_idx = r * cols + c;
            const fp8_value = src_bytes[src_idx];
            const scaled_value = dtype.fp8e4m3ToF32(fp8_value) * scale_inv;
            dst_u16[src_idx] = dtype.f32ToBf16(scaled_value);
        }
    }

    // Debug: print first few dequantized values
    log.debug("load", "FP8 per-block dequant sample", .{
        .first_scale = dtype.bf16ToF32(scale_data[0]),
        .first_src = src_bytes[0],
        .first_fp8_f32 = dtype.fp8e4m3ToF32(src_bytes[0]),
        .first_dequant_bf16 = dtype.bf16ToF32(dst_u16[0]),
    }, @src());

    log.trace("load", "FP8 per-block dequantized to BF16", .{ .rows = rows, .cols = cols }, @src());
    return owned.view();
}

fn transposeToOwned(allocator: std.mem.Allocator, t: Tensor, data_type: DType) !Tensor {
    const rows: usize = @intCast(t.shape[0]);
    const cols: usize = @intCast(t.shape[1]);

    const owned = try tensor.OwnedTensor.init(allocator, data_type, &.{ cols, rows });
    switch (data_type) {
        .f32 => {
            const src_f32 = t.asSlice(f32);
            const dst_f32 = owned.asSlice(f32);
            for (0..rows) |r| {
                for (0..cols) |c| {
                    dst_f32[c * rows + r] = src_f32[r * cols + c];
                }
            }
        },
        .bf16, .f16 => {
            const src_u16 = t.asSliceUnaligned(u16);
            const dst_u16 = owned.asSlice(u16);
            for (0..rows) |r| {
                for (0..cols) |c| {
                    dst_u16[c * rows + r] = src_u16[r * cols + c];
                }
            }
        },
        else => return error.UnsupportedDType,
    }

    return owned.view();
}

pub fn ensureF32(allocator: std.mem.Allocator, t: Tensor) !Tensor {
    return switch (t.dtype) {
        .f32 => t,
        .f16, .bf16, .grouped_affine_u4, .grouped_affine_u8 => convertToF32(allocator, t),
        // MXFP4 is a special quantization format used for MoE expert weights.
        // It should be kept as-is (not converted to F32) - the MoE kernel handles it directly.
        // Note: SafeTensors maps U8 to .i8 (signed), so we need to handle both.
        .mxfp4, .u8, .i8 => t,
        else => error.UnexpectedDType,
    };
}

pub fn orientEmbedding(allocator: std.mem.Allocator, st: *st_loader.UnifiedSafeTensors, name: []const u8, config: ModelConfig) !Tensor {
    var embed_tensor = try st.getTensor(name, null);

    // U32 from safetensors maps to grouped_affine_u4 by default
    // For models with mixed quantization, auto-detect bits from scales shape
    const expected_dim: usize = @intCast(config.d_model);
    if (inferGaffineParams(st, name, &embed_tensor, expected_dim)) |params| {
        try applyGaffineParams(&embed_tensor, params, name);
        return embed_tensor;
    } else if (embed_tensor.dtype == .grouped_affine_u4 or embed_tensor.dtype == .grouped_affine_u8) {
        // Gaffine tensor but inference failed - missing scales/biases
        return error.MissingScales;
    }

    return try ensureF32(allocator, embed_tensor);
}

pub fn convertToF32(allocator: std.mem.Allocator, t: Tensor) !Tensor {
    if (t.n_dims == 0) return error.InvalidShape;
    if (t.dtype == .grouped_affine_u4 or t.dtype == .grouped_affine_u8) {
        const gaffine = t.gaffine orelse return error.InvalidShape;
        const group = gaffine.group_size;
        var scales_owned: ?[]u16 = null;
        defer if (scales_owned) |owned_scales| allocator.free(owned_scales);
        var biases_owned: ?[]u16 = null;
        defer if (biases_owned) |owned_biases| allocator.free(owned_biases);
        var packed_vals_owned: ?[]u32 = null;
        defer if (packed_vals_owned) |owned_packed| allocator.free(owned_packed);
        const scales = try bytesToU16Slice(allocator, gaffine.scales, &scales_owned);
        const biases = try bytesToU16Slice(allocator, gaffine.biases, &biases_owned);
        const packed_vals = try bytesToU32Slice(allocator, t.data(), &packed_vals_owned);
        const rows: usize = @intCast(t.shape[0]);
        const cols: usize = @intCast(t.shape[1]);
        // 4-bit: 8 values per u32, 8-bit: 4 values per u32
        const values_per_word: usize = if (t.dtype == .grouped_affine_u4) 8 else 4;
        const bits: u5 = if (t.dtype == .grouped_affine_u4) 4 else 8;
        const mask: u32 = if (t.dtype == .grouped_affine_u4) 0xF else 0xFF;
        const packed_stride = cols / values_per_word;
        const group_stride = cols / group;
        const shape_usize = t.shapeAsUsize();
        const owned = try tensor.OwnedTensor.init(allocator, .f32, shape_usize[0..@intCast(t.n_dims)]);
        // No errdefer needed: no fallible operations follow before return
        const dst = owned.asSlice(f32);
        for (0..rows) |r| {
            const pack_row = packed_vals[r * packed_stride .. (r + 1) * packed_stride];
            const scale_row = scales[r * group_stride .. (r + 1) * group_stride];
            const bias_row = biases[r * group_stride .. (r + 1) * group_stride];
            var col_offset: usize = 0;
            while (col_offset < cols) : (col_offset += values_per_word) {
                const word = pack_row[col_offset / values_per_word];
                for (0..values_per_word) |val_idx| {
                    const col = col_offset + val_idx;
                    if (col >= cols) break;
                    const shift: u5 = @intCast(val_idx * bits);
                    const group_idx = col / group;
                    const scale = switch (gaffine.scales_dtype) {
                        .f16 => dtype.fp16ToF32(scale_row[group_idx]),
                        .bf16 => dtype.bf16ToF32(scale_row[group_idx]),
                        else => unreachable, // scales_dtype validated at load time
                    };
                    const bias = switch (gaffine.scales_dtype) {
                        .f16 => dtype.fp16ToF32(bias_row[group_idx]),
                        .bf16 => dtype.bf16ToF32(bias_row[group_idx]),
                        else => unreachable, // scales_dtype validated at load time
                    };

                    if (t.dtype == .grouped_affine_u8) {
                        // Grouped-affine u8 uses unsigned u8 values packed into u32.
                        // Zero-point is carried by the per-group bias term.
                        const quant_u8: u8 = @truncate((word >> shift) & mask);
                        dst[r * cols + col] = @as(f32, @floatFromInt(quant_u8)) * scale + bias;
                    } else {
                        // Grouped-affine u4 uses unsigned values 0..15 packed into u32.
                        const q4: u4 = @truncate((word >> shift) & mask);
                        dst[r * cols + col] = @as(f32, @floatFromInt(q4)) * scale + bias;
                    }
                }
            }
        }
        // Return view - arena owns memory
        return owned.view();
    }

    // Convert i64 shape to usize shape for OwnedTensor.init
    var shape_usize: [8]usize = undefined;
    for (0..@intCast(t.n_dims)) |dim_idx| {
        shape_usize[dim_idx] = @intCast(t.shape[dim_idx]);
    }
    var owned = try tensor.OwnedTensor.init(allocator, .f32, shape_usize[0..@intCast(t.n_dims)]);
    errdefer owned.deinit();
    // Use unaligned slice since mmap'd data may not be aligned
    const src_u16 = t.asSliceUnaligned(u16);
    const dst_f32 = owned.asSlice(f32);
    if (dst_f32.len != src_u16.len) return error.InvalidShape;
    if (t.dtype == .f16) {
        for (src_u16, dst_f32) |s, *d| d.* = dtype.fp16ToF32(s);
    } else {
        for (src_u16, dst_f32) |s, *d| d.* = @bitCast(@as(u32, s) << 16);
    }
    // Return view - arena owns memory
    return owned.view();
}

// =============================================================================
// Unit Tests
// =============================================================================

test "bytesToU16Slice aligned data returns view" {
    const allocator = std.testing.allocator;
    var aligned_data: [8]u8 align(@alignOf(u16)) = .{ 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08 };
    var owned: ?[]u16 = null;
    defer if (owned) |o| allocator.free(o);

    const result = try bytesToU16Slice(allocator, &aligned_data, &owned);

    try std.testing.expectEqual(@as(usize, 4), result.len);
    try std.testing.expect(owned == null); // Should not allocate for aligned data
}

test "bytesToU16Slice invalid length returns error" {
    const allocator = std.testing.allocator;
    var data: [3]u8 = .{ 0x01, 0x02, 0x03 }; // Odd length
    var owned: ?[]u16 = null;

    const result = bytesToU16Slice(allocator, &data, &owned);
    try std.testing.expectError(error.InvalidShape, result);
}

test "maybeConcatQkvWeights concatenates f32 tensors" {
    const allocator = std.testing.allocator;

    // Create three 2x2 F32 tensors
    const q_data = try allocator.alloc(f32, 4);
    defer allocator.free(q_data);
    @memcpy(q_data, &[_]f32{ 1, 2, 3, 4 });
    const q = Tensor.view2DSlice(q_data, 2, 2);

    const k_data = try allocator.alloc(f32, 4);
    defer allocator.free(k_data);
    @memcpy(k_data, &[_]f32{ 5, 6, 7, 8 });
    const k = Tensor.view2DSlice(k_data, 2, 2);

    const v_data = try allocator.alloc(f32, 4);
    defer allocator.free(v_data);
    @memcpy(v_data, &[_]f32{ 9, 10, 11, 12 });
    const v = Tensor.view2DSlice(v_data, 2, 2);

    const result = maybeConcatQkvWeights(allocator, q, k, v);
    defer if (result) |r| allocator.free(r.asSlice(f32));

    try std.testing.expect(result != null);
    const concat = result.?;
    try std.testing.expectEqual(@as(i64, 2), concat.shape[0]); // rows
    try std.testing.expectEqual(@as(i64, 6), concat.shape[1]); // cols = 2+2+2

    const data = concat.asSlice(f32);
    // Row 0: q[0,1], k[0,1], v[0,1] = 1,2,5,6,9,10
    try std.testing.expectEqual(@as(f32, 1), data[0]);
    try std.testing.expectEqual(@as(f32, 2), data[1]);
    try std.testing.expectEqual(@as(f32, 5), data[2]);
    try std.testing.expectEqual(@as(f32, 6), data[3]);
    try std.testing.expectEqual(@as(f32, 9), data[4]);
    try std.testing.expectEqual(@as(f32, 10), data[5]);
}

test "maybeConcatQkvWeights returns null for non-f32" {
    const allocator = std.testing.allocator;

    // Create BF16 tensors - should return null
    const q_data = try allocator.alloc(u16, 4);
    defer allocator.free(q_data);
    const q = Tensor.view(@ptrCast(q_data.ptr), &.{ 2, 2 }, .bf16, null);

    const k_data = try allocator.alloc(u16, 4);
    defer allocator.free(k_data);
    const k = Tensor.view(@ptrCast(k_data.ptr), &.{ 2, 2 }, .bf16, null);

    const v_data = try allocator.alloc(u16, 4);
    defer allocator.free(v_data);
    const v = Tensor.view(@ptrCast(v_data.ptr), &.{ 2, 2 }, .bf16, null);

    const result = maybeConcatQkvWeights(allocator, q, k, v);
    try std.testing.expect(result == null);
}

test "maybeConcatQkvWeights returns null for shape mismatch" {
    const allocator = std.testing.allocator;

    const q_data = try allocator.alloc(f32, 4);
    defer allocator.free(q_data);
    const q = Tensor.view2DSlice(q_data, 2, 2);

    const k_data = try allocator.alloc(f32, 6);
    defer allocator.free(k_data);
    const k = Tensor.view2DSlice(k_data, 3, 2); // Different rows

    const v_data = try allocator.alloc(f32, 4);
    defer allocator.free(v_data);
    const v = Tensor.view2DSlice(v_data, 2, 2);

    const result = maybeConcatQkvWeights(allocator, q, k, v);
    try std.testing.expect(result == null);
}

test "maybeConcatGateUpWeights concatenates f32 tensors" {
    const allocator = std.testing.allocator;

    const gate_data = try allocator.alloc(f32, 4);
    defer allocator.free(gate_data);
    @memcpy(gate_data, &[_]f32{ 1, 2, 3, 4 });
    const gate = Tensor.view2DSlice(gate_data, 2, 2);

    const up_data = try allocator.alloc(f32, 4);
    defer allocator.free(up_data);
    @memcpy(up_data, &[_]f32{ 5, 6, 7, 8 });
    const up = Tensor.view2DSlice(up_data, 2, 2);

    const result = maybeConcatGateUpWeights(allocator, gate, up);
    defer if (result) |r| allocator.free(r.asSlice(f32));

    try std.testing.expect(result != null);
    const concat = result.?;
    try std.testing.expectEqual(@as(i64, 2), concat.shape[0]); // rows
    try std.testing.expectEqual(@as(i64, 4), concat.shape[1]); // cols = 2+2

    const data = concat.asSlice(f32);
    // Row 0: gate[0,1], up[0,1] = 1,2,5,6
    try std.testing.expectEqual(@as(f32, 1), data[0]);
    try std.testing.expectEqual(@as(f32, 2), data[1]);
    try std.testing.expectEqual(@as(f32, 5), data[2]);
    try std.testing.expectEqual(@as(f32, 6), data[3]);
}

test "maybeConcatGateUpWeights returns null for non-f32" {
    const allocator = std.testing.allocator;

    const gate_data = try allocator.alloc(u16, 4);
    defer allocator.free(gate_data);
    const gate = Tensor.view(@ptrCast(gate_data.ptr), &.{ 2, 2 }, .bf16, null);

    const up_data = try allocator.alloc(u16, 4);
    defer allocator.free(up_data);
    const up = Tensor.view(@ptrCast(up_data.ptr), &.{ 2, 2 }, .bf16, null);

    const result = maybeConcatGateUpWeights(allocator, gate, up);
    try std.testing.expect(result == null);
}

test "orientWeightF32 returns 1D tensor unchanged" {
    const allocator = std.testing.allocator;

    const data = try allocator.alloc(f32, 4);
    defer allocator.free(data);
    @memcpy(data, &[_]f32{ 1, 2, 3, 4 });
    const t = Tensor.view(@ptrCast(data.ptr), &.{4}, .f32, null);

    const result = try orientWeightF32(allocator, t, 4);
    try std.testing.expectEqual(@as(u8, 1), result.n_dims);
    try std.testing.expectEqual(@as(i64, 4), result.shape[0]);
}

test "orientWeightF32 transposes when cols equals expected_in" {
    const allocator = std.testing.allocator;

    // 2x3 tensor where cols (3) = expected_in
    const data = try allocator.alloc(f32, 6);
    defer allocator.free(data);
    @memcpy(data, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    const t = Tensor.view2DSlice(data, 2, 3);

    const result = try orientWeightF32(allocator, t, 3);
    defer allocator.free(result.asSlice(f32));

    // Should be transposed to 3x2
    try std.testing.expectEqual(@as(i64, 3), result.shape[0]);
    try std.testing.expectEqual(@as(i64, 2), result.shape[1]);

    const out = result.asSlice(f32);
    // Original: [[1,2,3], [4,5,6]]
    // Transposed: [[1,4], [2,5], [3,6]]
    try std.testing.expectEqual(@as(f32, 1), out[0]);
    try std.testing.expectEqual(@as(f32, 4), out[1]);
    try std.testing.expectEqual(@as(f32, 2), out[2]);
    try std.testing.expectEqual(@as(f32, 5), out[3]);
}

test "orientWeightF32 returns unchanged when rows equals expected_in" {
    const allocator = std.testing.allocator;

    // 3x2 tensor where rows (3) = expected_in
    const data = try allocator.alloc(f32, 6);
    defer allocator.free(data);
    @memcpy(data, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    const t = Tensor.view2DSlice(data, 3, 2);

    const result = try orientWeightF32(allocator, t, 3);
    // Should be unchanged
    try std.testing.expectEqual(@as(i64, 3), result.shape[0]);
    try std.testing.expectEqual(@as(i64, 2), result.shape[1]);
}

test "orientWeightF32 returns error for invalid shape" {
    const allocator = std.testing.allocator;

    const data = try allocator.alloc(f32, 6);
    defer allocator.free(data);
    const t = Tensor.view2DSlice(data, 2, 3);

    // Neither 2 nor 3 equals expected_in=5
    const result = orientWeightF32(allocator, t, 5);
    try std.testing.expectError(error.InvalidShape, result);
}

test "orientWeightTyped returns 1D tensor unchanged" {
    const allocator = std.testing.allocator;

    const data = try allocator.alloc(u16, 4);
    defer allocator.free(data);
    const t = Tensor.view(@ptrCast(data.ptr), &.{4}, .bf16, null);

    const result = try orientWeightTyped(allocator, t, 4);
    try std.testing.expectEqual(@as(u8, 1), result.n_dims);
}

test "orientWeightTyped returns 2D tensor unchanged when cols equals expected_in" {
    const allocator = std.testing.allocator;

    const data = try allocator.alloc(u16, 6);
    defer allocator.free(data);
    const t = Tensor.view(@ptrCast(data.ptr), &.{ 2, 3 }, .bf16, null);

    // cols (3) == expected_in, so tensor is already in [out, in] format - no transpose
    const result = try orientWeightTyped(allocator, t, 3);
    try std.testing.expectEqual(@as(i64, 2), result.shape[0]);
    try std.testing.expectEqual(@as(i64, 3), result.shape[1]);
}

test "orientWeightTyped transposes when rows equals expected_in" {
    const allocator = std.testing.allocator;

    // Use arena for orientWeightTyped (follows same pattern as orientWeight tests)
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    // Create a [3, 2] bf16 tensor with known values
    const data = try allocator.alloc(u16, 6);
    defer allocator.free(data);
    // BF16 values: 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
    data[0] = 0x3F80; // 1.0 in bf16
    data[1] = 0x4000; // 2.0 in bf16
    data[2] = 0x4040; // 3.0 in bf16
    data[3] = 0x4080; // 4.0 in bf16
    data[4] = 0x40A0; // 5.0 in bf16
    data[5] = 0x40C0; // 6.0 in bf16
    const t = Tensor.view(@ptrCast(data.ptr), &.{ 3, 2 }, .bf16, null);

    // rows (3) == expected_in, so tensor is in [in, out] format - needs transpose
    const result = try orientWeightTyped(arena_alloc, t, 3);

    // Should be transposed to [2, 3]
    try std.testing.expectEqual(@as(i64, 2), result.shape[0]);
    try std.testing.expectEqual(@as(i64, 3), result.shape[1]);

    // Verify transposition: original [[1,2], [3,4], [5,6]] -> [[1,3,5], [2,4,6]]
    const out = result.asSlice(u16);
    try std.testing.expectEqual(@as(u16, 0x3F80), out[0]); // 1.0
    try std.testing.expectEqual(@as(u16, 0x4040), out[1]); // 3.0
    try std.testing.expectEqual(@as(u16, 0x40A0), out[2]); // 5.0
    try std.testing.expectEqual(@as(u16, 0x4000), out[3]); // 2.0
    try std.testing.expectEqual(@as(u16, 0x4080), out[4]); // 4.0
    try std.testing.expectEqual(@as(u16, 0x40C0), out[5]); // 6.0
}

test "orientWeightTyped returns error for invalid shape" {
    const allocator = std.testing.allocator;

    const data = try allocator.alloc(u16, 6);
    defer allocator.free(data);
    const t = Tensor.view(@ptrCast(data.ptr), &.{ 2, 3 }, .bf16, null);

    const result = orientWeightTyped(allocator, t, 5);
    try std.testing.expectError(error.InvalidShape, result);
}

test "ensureF32 passes through f32 tensor" {
    const allocator = std.testing.allocator;

    const data = try allocator.alloc(f32, 4);
    defer allocator.free(data);
    @memcpy(data, &[_]f32{ 1, 2, 3, 4 });
    const t = Tensor.view(@ptrCast(data.ptr), &.{4}, .f32, null);

    const result = try ensureF32(allocator, t);
    try std.testing.expectEqual(DType.f32, result.dtype);
    // Should be the same tensor (no conversion)
    try std.testing.expectEqual(t.data_ptr, result.data_ptr);
}

test "convertToF32 converts f16 to f32" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    // Create F16 tensor with known values
    const data = try arena_alloc.alloc(u16, 4);
    // F16 representation of 1.0, 2.0, 3.0, 4.0
    data[0] = 0x3C00; // 1.0 in f16
    data[1] = 0x4000; // 2.0 in f16
    data[2] = 0x4200; // 3.0 in f16
    data[3] = 0x4400; // 4.0 in f16
    const t = Tensor.view(@ptrCast(data.ptr), &.{4}, .f16, null);

    const result = try convertToF32(arena_alloc, t);

    try std.testing.expectEqual(DType.f32, result.dtype);
    const out = result.asSlice(f32);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[1], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out[2], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out[3], 1e-3);
}

test "convertToF32 converts bf16 to f32" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    const data = try arena_alloc.alloc(u16, 4);
    // BF16 representation: upper 16 bits of f32
    data[0] = 0x3F80; // 1.0 in bf16
    data[1] = 0x4000; // 2.0 in bf16
    data[2] = 0x4040; // 3.0 in bf16
    data[3] = 0x4080; // 4.0 in bf16
    const t = Tensor.view(@ptrCast(data.ptr), &.{4}, .bf16, null);

    const result = try convertToF32(arena_alloc, t);

    try std.testing.expectEqual(DType.f32, result.dtype);
    const out = result.asSlice(f32);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[1], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out[2], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out[3], 1e-3);
}

test "convertToF32 returns error for empty tensor" {
    const allocator = std.testing.allocator;

    var t = std.mem.zeroes(Tensor);
    t.dtype = .f16;
    t.n_dims = 0;

    const result = convertToF32(allocator, t);
    try std.testing.expectError(error.InvalidShape, result);
}

test "orientWeight transposes f32 weight when needed" {
    const allocator = std.testing.allocator;
    const writer = @import("../../io/safetensors/writer.zig");

    const tmp_dir_path = "/tmp/test_orient_weight";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    // Create a [out=3, in=2] weight matrix in safetensors format
    // Data: [[1,2], [3,4], [5,6]] stored row-major
    const weight_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const entries = [_]writer.TensorEntry{
        .{ .name = "weight", .dtype = .f32, .shape = &[_]usize{ 3, 2 }, .data = std.mem.sliceAsBytes(&weight_data) },
    };
    const model_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, model_path, &entries);

    var st = try st_loader.UnifiedSafeTensors.load(allocator, model_path);
    defer st.deinit();

    var arena_alloc = std.heap.ArenaAllocator.init(allocator);
    defer arena_alloc.deinit();

    // expected_in=2 matches cols, so it should transpose to [2, 3]
    const config = std.mem.zeroes(ModelConfig);
    const result = try orientWeight(arena_alloc.allocator(), &st, "weight", 2, config);

    try std.testing.expectEqual(@as(u8, 2), result.n_dims);
    try std.testing.expectEqual(@as(i64, 2), result.shape[0]); // in dimension first
    try std.testing.expectEqual(@as(i64, 3), result.shape[1]); // out dimension second

    // Verify transposition: original [3,2] -> transposed [2,3]
    // Original row 0: [1,2], row 1: [3,4], row 2: [5,6]
    // Transposed col 0: [1,3,5], col 1: [2,4,6]
    const out = result.asSlice(f32);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), out[5], 1e-6);
}

test "orientWeight returns untransposed when rows equals expected_in" {
    const allocator = std.testing.allocator;
    const writer = @import("../../io/safetensors/writer.zig");

    const tmp_dir_path = "/tmp/test_orient_weight_notr";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    // Create a [in=2, out=3] weight matrix (already in expected format)
    const weight_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const entries = [_]writer.TensorEntry{
        .{ .name = "weight", .dtype = .f32, .shape = &[_]usize{ 2, 3 }, .data = std.mem.sliceAsBytes(&weight_data) },
    };
    const model_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, model_path, &entries);

    var st = try st_loader.UnifiedSafeTensors.load(allocator, model_path);
    defer st.deinit();

    var arena_alloc = std.heap.ArenaAllocator.init(allocator);
    defer arena_alloc.deinit();

    // expected_in=2 matches rows, so no transpose needed
    const config = std.mem.zeroes(ModelConfig);
    const result = try orientWeight(arena_alloc.allocator(), &st, "weight", 2, config);

    try std.testing.expectEqual(@as(u8, 2), result.n_dims);
    try std.testing.expectEqual(@as(i64, 2), result.shape[0]);
    try std.testing.expectEqual(@as(i64, 3), result.shape[1]);

    // Data should be unchanged
    const out = result.asSlice(f32);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out[2], 1e-6);
}

test "orientEmbedding converts f16 embedding to f32" {
    const allocator = std.testing.allocator;
    const writer = @import("../../io/safetensors/writer.zig");

    const tmp_dir_path = "/tmp/test_orient_embed";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    // Create a small f16 embedding table [vocab=4, dim=2]
    // Use simple values that are exact in f16
    const embed_data = [_]u16{
        0x3C00, 0x4000, // 1.0, 2.0
        0x4200, 0x4400, // 3.0, 4.0
        0x4500, 0x4600, // 5.0, 6.0
        0x4700, 0x4800, // 7.0, 8.0
    };
    const entries = [_]writer.TensorEntry{
        .{ .name = "embed", .dtype = .f16, .shape = &[_]usize{ 4, 2 }, .data = std.mem.sliceAsBytes(&embed_data) },
    };
    const model_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, model_path, &entries);

    var st = try st_loader.UnifiedSafeTensors.load(allocator, model_path);
    defer st.deinit();

    var arena_alloc = std.heap.ArenaAllocator.init(allocator);
    defer arena_alloc.deinit();

    var config = std.mem.zeroes(ModelConfig);
    config.d_model = 2; // embedding dimension
    const result = try orientEmbedding(arena_alloc.allocator(), &st, "embed", config);

    try std.testing.expectEqual(DType.f32, result.dtype);
    try std.testing.expectEqual(@as(u8, 2), result.n_dims);
    try std.testing.expectEqual(@as(i64, 4), result.shape[0]);
    try std.testing.expectEqual(@as(i64, 2), result.shape[1]);

    const out = result.asSlice(f32);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[1], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out[2], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out[3], 1e-3);
}

test "orientEmbedding passes through f32 embedding unchanged" {
    const allocator = std.testing.allocator;
    const writer = @import("../../io/safetensors/writer.zig");

    const tmp_dir_path = "/tmp/test_orient_embed_f32";
    std.fs.cwd().makeDir(tmp_dir_path) catch {};
    defer std.fs.cwd().deleteTree(tmp_dir_path) catch {};

    // Create f32 embedding table
    const embed_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const entries = [_]writer.TensorEntry{
        .{ .name = "embed", .dtype = .f32, .shape = &[_]usize{ 2, 2 }, .data = std.mem.sliceAsBytes(&embed_data) },
    };
    const model_path = tmp_dir_path ++ "/model.safetensors";
    try writer.write(allocator, model_path, &entries);

    var st = try st_loader.UnifiedSafeTensors.load(allocator, model_path);
    defer st.deinit();

    var arena_alloc = std.heap.ArenaAllocator.init(allocator);
    defer arena_alloc.deinit();

    var config = std.mem.zeroes(ModelConfig);
    config.d_model = 2;
    const result = try orientEmbedding(arena_alloc.allocator(), &st, "embed", config);

    try std.testing.expectEqual(DType.f32, result.dtype);
    const out = result.asSlice(f32);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out[3], 1e-6);
}
