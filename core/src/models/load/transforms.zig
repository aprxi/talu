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
const GroupedAffineMeta = dtype.GroupedAffineMeta;

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
    config_gaffine_bits: i32,
) ?GaffineInferResult {
    if (t.dtype != .grouped_affine_u4 and t.dtype != .grouped_affine_u8) return null;

    const base = if (std.mem.endsWith(u8, name, ".weight"))
        name[0 .. name.len - ".weight".len]
    else if (std.mem.endsWith(u8, name, ".qweight"))
        name[0 .. name.len - ".qweight".len]
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

    // Disambiguate when both group sizes are valid:
    // 1. Config's gaffine_bits is authoritative when available
    // 2. Fall back to expected_dim matching for mixed quantization models
    const is_4bit = if (valid_4bit and valid_8bit)
        if (config_gaffine_bits == 4) true else if (config_gaffine_bits == 8) false else (unpacked_4bit == expected_dim)
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

const MatrixShape = struct { rows: usize, cols: usize };

fn requireMatrixShape(weight: *const Tensor) !MatrixShape {
    if (weight.n_dims != 2) return error.InvalidShape;
    const rows: usize = @intCast(weight.shape[0]);
    const cols: usize = @intCast(weight.shape[1]);
    if (rows == 0 or cols == 0) return error.InvalidShape;
    return .{ .rows = rows, .cols = cols };
}

pub fn buildGatedDeltaSplitInProj(
    allocator: std.mem.Allocator,
    in_proj_qkv: *const Tensor,
    in_proj_z: *const Tensor,
    in_proj_b: *const Tensor,
    in_proj_a: ?*const Tensor,
) !*const Tensor {
    const qkv_shape = try requireMatrixShape(in_proj_qkv);
    const z_shape = try requireMatrixShape(in_proj_z);
    const b_shape = try requireMatrixShape(in_proj_b);
    var a_shape: ?MatrixShape = null;
    if (in_proj_a) |a| {
        const s = try requireMatrixShape(a);
        a_shape = s;
        if (s.cols != z_shape.cols) return error.InvalidShape;
        if (s.rows != b_shape.rows) return error.InvalidShape;
    }
    if (z_shape.cols != qkv_shape.cols or z_shape.cols != b_shape.cols) return error.InvalidShape;
    // Mixed FP8 + BF16: dequant FP8 tensors to BF16, then fuse as dense.
    // Common in FP8 models where small projections (A, B) stay BF16.
    const has_fp8 = in_proj_qkv.dtype == .f8_e4m3 or in_proj_z.dtype == .f8_e4m3 or
        in_proj_b.dtype == .f8_e4m3 or (if (in_proj_a) |a| a.dtype == .f8_e4m3 else false);
    const all_fp8 = in_proj_qkv.dtype == .f8_e4m3 and in_proj_z.dtype == .f8_e4m3 and
        in_proj_b.dtype == .f8_e4m3 and
        (if (in_proj_a) |a| a.dtype == .f8_e4m3 else true);

    if (has_fp8 and all_fp8) {
        return buildGatedDeltaSplitInProjFp8(allocator, in_proj_qkv, in_proj_z, in_proj_b, in_proj_a);
    }

    if (has_fp8 and !all_fp8) {
        // Dequant FP8 tensors to BF16, keep BF16 as-is, then fuse as BF16.
        const qkv_bf16 = if (in_proj_qkv.dtype == .f8_e4m3)
            try dequantFp8ToBf16ForFusion(allocator, in_proj_qkv)
        else
            in_proj_qkv;
        const z_bf16 = if (in_proj_z.dtype == .f8_e4m3)
            try dequantFp8ToBf16ForFusion(allocator, in_proj_z)
        else
            in_proj_z;
        const b_bf16 = if (in_proj_b.dtype == .f8_e4m3)
            try dequantFp8ToBf16ForFusion(allocator, in_proj_b)
        else
            in_proj_b;
        var a_bf16: ?*const Tensor = null;
        if (in_proj_a) |a| {
            a_bf16 = if (a.dtype == .f8_e4m3)
                try dequantFp8ToBf16ForFusion(allocator, a)
            else
                a;
        }
        // Recurse with all-BF16 tensors
        return buildGatedDeltaSplitInProj(allocator, qkv_bf16, z_bf16, b_bf16, a_bf16);
    }

    if (in_proj_z.dtype != in_proj_qkv.dtype or in_proj_z.dtype != in_proj_b.dtype) return error.InvalidDType;
    if (in_proj_a) |a| {
        if (a.dtype != in_proj_z.dtype) return error.InvalidDType;
    }

    // Quantized tensors (GAF): elementSize() is 0, use actual data lengths.
    const is_quantized = in_proj_z.dtype == .grouped_affine_u4 or in_proj_z.dtype == .grouped_affine_u8;
    if (is_quantized) {
        return buildGatedDeltaSplitInProjQuantized(allocator, in_proj_qkv, in_proj_z, in_proj_b, in_proj_a);
    }

    const element_bytes = in_proj_z.dtype.elementSize();
    const qkv_bytes = in_proj_qkv.data();
    const z_bytes = in_proj_z.data();
    const b_bytes = in_proj_b.data();
    const a_bytes = if (in_proj_a) |a| a.data() else &.{};

    const same_rows = qkv_shape.rows == z_shape.rows and qkv_shape.rows == b_shape.rows and
        (a_shape == null or qkv_shape.rows == a_shape.?.rows);
    const same_cols = qkv_shape.cols == z_shape.cols and qkv_shape.cols == b_shape.cols and
        (a_shape == null or qkv_shape.cols == a_shape.?.cols);

    if (!same_rows and !same_cols) return error.InvalidShape;

    if (same_rows) {
        const rows = qkv_shape.rows;
        const cols = qkv_shape.cols + z_shape.cols + b_shape.cols + if (a_shape != null) a_shape.?.cols else 0;
        const qkv_row_bytes = qkv_shape.cols * element_bytes;
        const z_row_bytes = z_shape.cols * element_bytes;
        const b_row_bytes = b_shape.cols * element_bytes;
        const a_row_bytes: usize = if (a_shape != null) a_shape.?.cols * element_bytes else 0;
        const out_row_bytes = cols * element_bytes;

        if (qkv_bytes.len != rows * qkv_row_bytes or z_bytes.len != rows * z_row_bytes or b_bytes.len != rows * b_row_bytes) {
            return error.InvalidShape;
        }
        if (a_shape != null and a_bytes.len != rows * a_row_bytes) return error.InvalidShape;

        const fused = try Tensor.init(allocator, &.{ @intCast(rows), @intCast(cols) }, in_proj_z.dtype, tensor.Device.cpu());
        const dst = fused.data();
        if (dst.len != rows * out_row_bytes) return error.InvalidShape;

        for (0..rows) |row_idx| {
            const dst_row = dst[row_idx * out_row_bytes ..][0..out_row_bytes];
            const qkv_row = qkv_bytes[row_idx * qkv_row_bytes ..][0..qkv_row_bytes];
            const z_row = z_bytes[row_idx * z_row_bytes ..][0..z_row_bytes];
            const b_row = b_bytes[row_idx * b_row_bytes ..][0..b_row_bytes];
            const a_row = if (a_row_bytes != 0)
                a_bytes[row_idx * a_row_bytes ..][0..a_row_bytes]
            else
                &.{};

            var offset: usize = 0;
            @memcpy(dst_row[offset .. offset + qkv_row_bytes], qkv_row);
            offset += qkv_row_bytes;
            @memcpy(dst_row[offset .. offset + z_row_bytes], z_row);
            offset += z_row_bytes;
            @memcpy(dst_row[offset .. offset + b_row_bytes], b_row);
            offset += b_row_bytes;
            if (a_row_bytes != 0) @memcpy(dst_row[offset .. offset + a_row_bytes], a_row);
        }
        return fused;
    }

    const cols = qkv_shape.cols;
    const rows = qkv_shape.rows + z_shape.rows + b_shape.rows + if (a_shape != null) a_shape.?.rows else 0;
    const row_bytes = cols * element_bytes;

    if (qkv_bytes.len != qkv_shape.rows * row_bytes or z_bytes.len != z_shape.rows * row_bytes or b_bytes.len != b_shape.rows * row_bytes) {
        return error.InvalidShape;
    }
    if (a_shape != null and a_bytes.len != a_shape.?.rows * row_bytes) return error.InvalidShape;

    const fused = try Tensor.init(allocator, &.{ @intCast(rows), @intCast(cols) }, in_proj_z.dtype, tensor.Device.cpu());
    const dst = fused.data();
    if (dst.len != rows * row_bytes) return error.InvalidShape;

    var dst_offset: usize = 0;
    @memcpy(dst[dst_offset .. dst_offset + qkv_bytes.len], qkv_bytes);
    dst_offset += qkv_bytes.len;
    @memcpy(dst[dst_offset .. dst_offset + z_bytes.len], z_bytes);
    dst_offset += z_bytes.len;
    @memcpy(dst[dst_offset .. dst_offset + b_bytes.len], b_bytes);
    dst_offset += b_bytes.len;
    if (a_bytes.len != 0) @memcpy(dst[dst_offset .. dst_offset + a_bytes.len], a_bytes);
    return fused;
}

/// Fuse split gated-delta projections for quantized (GAF) tensors.
///
/// Concatenates along the output dimension (rows): each sub-tensor contributes its
/// packed weight data, scales, and biases. All sub-tensors must share the same input
/// dimension (cols), group_size, scales_dtype, and quantization dtype.
fn buildGatedDeltaSplitInProjQuantized(
    allocator: std.mem.Allocator,
    in_proj_qkv: *const Tensor,
    in_proj_z: *const Tensor,
    in_proj_b: *const Tensor,
    in_proj_a: ?*const Tensor,
) !*const Tensor {
    // All tensors have shape [out_features, unpacked_in] but data is packed.
    // Verify gaffine metadata exists and group params match.
    const qkv_gaf = in_proj_qkv.gaffine orelse return error.MissingScales;
    const z_gaf = in_proj_z.gaffine orelse return error.MissingScales;
    const b_gaf = in_proj_b.gaffine orelse return error.MissingScales;
    if (z_gaf.group_size != qkv_gaf.group_size or b_gaf.group_size != qkv_gaf.group_size) return error.InvalidShape;
    if (z_gaf.scales_dtype != qkv_gaf.scales_dtype or b_gaf.scales_dtype != qkv_gaf.scales_dtype) return error.InvalidDType;
    if (in_proj_a) |a| {
        const a_gaf = a.gaffine orelse return error.MissingScales;
        if (a_gaf.group_size != qkv_gaf.group_size) return error.InvalidShape;
        if (a_gaf.scales_dtype != qkv_gaf.scales_dtype) return error.InvalidDType;
    }

    // Concatenate packed data (row-major along output dim)
    const total_data_len = in_proj_qkv.data_size + in_proj_z.data_size + in_proj_b.data_size +
        (if (in_proj_a) |a| a.data_size else 0);
    const fused_data = try allocator.alloc(u8, total_data_len); // lint:ignore errdefer-alloc - arena freed atomically
    {
        var off: usize = 0;
        @memcpy(fused_data[off .. off + in_proj_qkv.data_size], in_proj_qkv.data());
        off += in_proj_qkv.data_size;
        @memcpy(fused_data[off .. off + in_proj_z.data_size], in_proj_z.data());
        off += in_proj_z.data_size;
        @memcpy(fused_data[off .. off + in_proj_b.data_size], in_proj_b.data());
        off += in_proj_b.data_size;
        if (in_proj_a) |a| @memcpy(fused_data[off .. off + a.data_size], a.data());
    }

    // Concatenate scales and biases
    const total_scales_len = qkv_gaf.scales.len + z_gaf.scales.len + b_gaf.scales.len +
        (if (in_proj_a) |a| (a.gaffine orelse return error.MissingScales).scales.len else 0);
    const total_biases_len = qkv_gaf.biases.len + z_gaf.biases.len + b_gaf.biases.len +
        (if (in_proj_a) |a| (a.gaffine orelse return error.MissingScales).biases.len else 0);

    const fused_scales = try allocator.alloc(u8, total_scales_len); // lint:ignore errdefer-alloc - arena freed atomically
    const fused_biases = try allocator.alloc(u8, total_biases_len); // lint:ignore errdefer-alloc - arena freed atomically
    {
        var s_off: usize = 0;
        var b_off: usize = 0;
        for ([_]GroupedAffineMeta{ qkv_gaf, z_gaf, b_gaf }) |gaf| {
            @memcpy(fused_scales[s_off .. s_off + gaf.scales.len], gaf.scales);
            s_off += gaf.scales.len;
            @memcpy(fused_biases[b_off .. b_off + gaf.biases.len], gaf.biases);
            b_off += gaf.biases.len;
        }
        if (in_proj_a) |a| {
            const a_gaf = a.gaffine orelse return error.MissingScales;
            @memcpy(fused_scales[s_off .. s_off + a_gaf.scales.len], a_gaf.scales);
            @memcpy(fused_biases[b_off .. b_off + a_gaf.biases.len], a_gaf.biases);
        }
    }

    // Build fused tensor
    const qkv_rows: usize = @intCast(in_proj_qkv.shape[0]);
    const z_rows: usize = @intCast(in_proj_z.shape[0]);
    const b_rows: usize = @intCast(in_proj_b.shape[0]);
    const a_rows: usize = if (in_proj_a) |a| @intCast(a.shape[0]) else 0;
    const total_rows = qkv_rows + z_rows + b_rows + a_rows;
    const cols: usize = @intCast(in_proj_qkv.shape[1]);

    const result = try allocator.create(Tensor); // lint:ignore errdefer-alloc - arena freed atomically
    result.* = Tensor.view(fused_data.ptr, &.{ total_rows, cols }, in_proj_qkv.dtype, fused_data.len);
    result.gaffine = .{
        .scales = fused_scales,
        .biases = fused_biases,
        .group_size = qkv_gaf.group_size,
        .scales_dtype = qkv_gaf.scales_dtype,
    };
    return result;
}

/// Dequantize an FP8 tensor with per-block scales to BF16 for fusion with BF16 tensors.
/// Returns a pointer to a heap-allocated BF16 tensor (arena-managed).
fn dequantFp8ToBf16ForFusion(allocator: std.mem.Allocator, t: *const Tensor) !*const Tensor {
    if (t.n_dims != 2) return error.InvalidShape;
    const fp8_meta = t.fp8 orelse return error.MissingScales;
    const rows: usize = @intCast(t.shape[0]);
    const cols: usize = @intCast(t.shape[1]);
    const owned = try tensor.OwnedTensor.init(allocator, .bf16, &.{ rows, cols });
    const src_bytes = t.data()[0 .. rows * cols];
    const dst_u16 = owned.asSlice(u16);

    if (fp8_meta.block_scales_data) |scales_ptr| {
        const scale_data = @as([*]const u16, @ptrCast(@alignCast(scales_ptr)))[0 .. @as(usize, fp8_meta.scale_rows) * @as(usize, fp8_meta.scale_cols)];
        const block_size: usize = fp8_meta.block_size;
        const s_cols: usize = fp8_meta.scale_cols;
        for (0..rows) |r| {
            const sr = r / block_size;
            for (0..cols) |c| {
                const sc = c / block_size;
                const scale_inv = dtype.bf16ToF32(scale_data[sr * s_cols + sc]);
                const fp8_value = src_bytes[r * cols + c];
                dst_u16[r * cols + c] = dtype.f32ToBf16(dtype.fp8e4m3ToF32(fp8_value) * scale_inv);
            }
        }
    } else {
        // Per-tensor scale
        const scale_inv = fp8_meta.scale_inv;
        for (0..rows * cols) |i| {
            dst_u16[i] = dtype.f32ToBf16(dtype.fp8e4m3ToF32(src_bytes[i]) * scale_inv);
        }
    }

    const result = try allocator.create(Tensor);
    result.* = owned.view();
    return result;
}

/// Fuse split gated-delta projections for FP8 E4M3 tensors with per-block scales.
///
/// Concatenates along the output dimension (rows). Each sub-tensor contributes
/// its FP8 weight bytes and BF16 per-block scales. All must share the same
/// input dimension, block_size, and scale_cols.
fn buildGatedDeltaSplitInProjFp8(
    allocator: std.mem.Allocator,
    in_proj_qkv: *const Tensor,
    in_proj_z: *const Tensor,
    in_proj_b: *const Tensor,
    in_proj_a: ?*const Tensor,
) !*const Tensor {
    const qkv_fp8 = in_proj_qkv.fp8 orelse return error.MissingScales;
    const z_fp8 = in_proj_z.fp8 orelse return error.MissingScales;
    const b_fp8 = in_proj_b.fp8 orelse return error.MissingScales;

    // Validate compatible per-block metadata
    if (qkv_fp8.block_size != z_fp8.block_size or qkv_fp8.block_size != b_fp8.block_size)
        return error.InvalidShape;
    if (qkv_fp8.scale_cols != z_fp8.scale_cols or qkv_fp8.scale_cols != b_fp8.scale_cols)
        return error.InvalidShape;

    var a_fp8: ?dtype.Fp8Meta = null;
    if (in_proj_a) |a| {
        a_fp8 = a.fp8 orelse return error.MissingScales;
        if (a_fp8.?.block_size != qkv_fp8.block_size or a_fp8.?.scale_cols != qkv_fp8.scale_cols)
            return error.InvalidShape;
    }

    const qkv_rows: usize = @intCast(in_proj_qkv.shape[0]);
    const z_rows: usize = @intCast(in_proj_z.shape[0]);
    const b_rows: usize = @intCast(in_proj_b.shape[0]);
    const a_rows: usize = if (in_proj_a) |a| @intCast(a.shape[0]) else 0;
    const total_rows = qkv_rows + z_rows + b_rows + a_rows;
    const cols: usize = @intCast(in_proj_z.shape[1]);

    // Fuse FP8 weight bytes (1 byte per element, concatenate rows)
    const total_bytes = total_rows * cols;
    const fused_data = try allocator.alloc(u8, total_bytes);
    var dst_off: usize = 0;
    const qkv_bytes = in_proj_qkv.data()[0 .. qkv_rows * cols];
    @memcpy(fused_data[dst_off .. dst_off + qkv_bytes.len], qkv_bytes);
    dst_off += qkv_bytes.len;
    const z_bytes = in_proj_z.data()[0 .. z_rows * cols];
    @memcpy(fused_data[dst_off .. dst_off + z_bytes.len], z_bytes);
    dst_off += z_bytes.len;
    const b_bytes_data = in_proj_b.data()[0 .. b_rows * cols];
    @memcpy(fused_data[dst_off .. dst_off + b_bytes_data.len], b_bytes_data);
    dst_off += b_bytes_data.len;
    if (in_proj_a) |a| {
        const a_bytes = a.data()[0 .. a_rows * cols];
        @memcpy(fused_data[dst_off .. dst_off + a_bytes.len], a_bytes);
    }

    // Fuse per-block BF16 scales (concatenate scale rows)
    const total_scale_rows = qkv_fp8.scale_rows + z_fp8.scale_rows + b_fp8.scale_rows +
        (if (a_fp8) |a| a.scale_rows else 0);
    const scale_cols = qkv_fp8.scale_cols;
    const scale_entry_bytes = @sizeOf(u16);
    const total_scale_bytes = @as(usize, total_scale_rows) * @as(usize, scale_cols) * scale_entry_bytes;
    const fused_scales = try allocator.alloc(u8, total_scale_bytes);

    var scale_off: usize = 0;
    inline for (.{ qkv_fp8, z_fp8, b_fp8 }) |fp8| {
        if (fp8.block_scales_data) |sdata| {
            const slen = @as(usize, fp8.scale_rows) * @as(usize, fp8.scale_cols) * scale_entry_bytes;
            @memcpy(fused_scales[scale_off .. scale_off + slen], sdata[0..slen]);
            scale_off += slen;
        }
    }
    if (a_fp8) |fp8| {
        if (fp8.block_scales_data) |sdata| {
            const slen = @as(usize, fp8.scale_rows) * @as(usize, fp8.scale_cols) * scale_entry_bytes;
            @memcpy(fused_scales[scale_off .. scale_off + slen], sdata[0..slen]);
        }
    }

    const result = try allocator.create(Tensor);
    result.* = Tensor.view(fused_data.ptr, &.{ total_rows, cols }, .f8_e4m3, total_bytes);
    result.fp8 = .{
        .block_scales_data = fused_scales.ptr,
        .block_scales_len = total_scale_bytes,
        .scale_rows = total_scale_rows,
        .scale_cols = scale_cols,
        .block_size = qkv_fp8.block_size,
    };
    return result;
}

inline fn ue8m0ToScale(e8m0: u8) f32 {
    const exp_bits = @as(u32, e8m0) << 23;
    return @bitCast(exp_bits);
}

fn dequantizeMxfp8WeightToBf16(
    allocator: std.mem.Allocator,
    t: Tensor,
    expected_in: usize,
) !Tensor {
    if (t.n_dims != 2) return error.InvalidShape;
    const meta = t.mxfp8 orelse return error.MissingScales;

    const rows: usize = @intCast(t.shape[0]);
    const cols: usize = @intCast(t.shape[1]);
    if (rows == 0 or cols == 0) return error.InvalidShape;
    if (cols != expected_in and rows != expected_in) return error.InvalidShape;

    const scale_ptr = meta.block_scales_data orelse return error.MissingScales;
    const scale_cols: usize = @intCast(meta.scale_cols);
    if (scale_cols == 0) return error.InvalidShape;

    const required_scale_cols = (cols + 31) / 32;
    if (scale_cols < required_scale_cols) return error.InvalidShape;
    const required_scale_len = rows * scale_cols;
    if (meta.block_scales_len < required_scale_len) return error.InvalidShape;
    const scales = scale_ptr[0..required_scale_len];

    const src = t.data();
    const required_src_len = rows * cols;
    if (src.len < required_src_len) return error.InvalidShape;
    const src_bytes = src[0..required_src_len];

    const owned = try tensor.OwnedTensor.init(allocator, .bf16, &.{ rows, cols });
    const dst_u16 = owned.asSlice(u16);

    for (0..rows) |r| {
        const scale_row = scales[r * scale_cols ..][0..scale_cols];
        for (0..cols) |c| {
            const idx = r * cols + c;
            const scale = ue8m0ToScale(scale_row[c / 32]);
            dst_u16[idx] = dtype.f32ToBf16(dtype.fp8e4m3ToF32(src_bytes[idx]) * scale);
        }
    }

    return owned.view();
}

pub fn orientWeight(
    allocator: std.mem.Allocator,
    st: *st_loader.UnifiedSafeTensors,
    name: []const u8,
    expected_in: usize,
    config: ModelConfig,
    dequantize_mxfp8_to_bf16: bool,
) !Tensor {
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

    // U32/I32 from safetensors maps to grouped_affine_u4 by default.
    // For models with mixed quantization, auto-detect bits from scales shape.
    // GAF format (MLX): .weight + .scales + .biases
    // GPTQ format (AutoRound/AutoGPTQ): .qweight + .scales + .qzeros (transposed layout)
    if (inferGaffineParams(st, name, &weight_tensor, expected_in, config.gaffine_bits)) |params| {
        try applyGaffineParams(&weight_tensor, params, name);
        return weight_tensor;
    } else if (weight_tensor.dtype == .grouped_affine_u4 or weight_tensor.dtype == .grouped_affine_u8) {
        // GAF inference failed (no .biases) - try GPTQ conversion
        return convertGptqToGaffine(allocator, st, name, weight_tensor, expected_in, config.gaffine_bits) catch {
            return error.MissingScales;
        };
    }
    // Handle MXFP8 E4M3 weights — E4M3 data + UE8M0 block-32 scales for cuBLASLt tensor core GEMM
    if (weight_tensor.dtype == .f8_e4m3) mxfp8_check: {
        const base = if (std.mem.endsWith(u8, name, ".weight"))
            name[0 .. name.len - ".weight".len]
        else
            name;

        var mxfp8_scale_name_buf: [256]u8 = undefined;
        const mxfp8_scale_name = std.fmt.bufPrint(&mxfp8_scale_name_buf, "{s}.weight_block_scale", .{base}) catch break :mxfp8_check;
        const mxfp8_scale_tensor = st.getTensor(mxfp8_scale_name, null) catch break :mxfp8_check;

        // MXFP8 scales are 2D: [rows, ceil(cols/32)] with dtype u8/i8 (E8M0)
        // Note: safetensors reader maps U8 → .i8 ("unsigned treated as signed")
        if (mxfp8_scale_tensor.n_dims != 2) break :mxfp8_check;
        if (mxfp8_scale_tensor.dtype != .u8 and mxfp8_scale_tensor.dtype != .i8) break :mxfp8_check;

        const s_rows: usize = @intCast(mxfp8_scale_tensor.shape[0]);
        const s_cols: usize = @intCast(mxfp8_scale_tensor.shape[1]);
        const w_rows: usize = @intCast(weight_tensor.shape[0]);
        if (s_rows != w_rows) break :mxfp8_check;

        const scale_data = mxfp8_scale_tensor.data();
        const scale_byte_len = s_rows * s_cols;
        if (scale_data.len < scale_byte_len) break :mxfp8_check;

        weight_tensor.mxfp8 = .{
            .block_scales_data = scale_data.ptr,
            .block_scales_len = scale_byte_len,
            .rows = @intCast(w_rows),
            .cols = @intCast(weight_tensor.shape[1]),
            .scale_cols = @intCast(s_cols),
        };
        if (dequantize_mxfp8_to_bf16) {
            const dequantized = try dequantizeMxfp8WeightToBf16(allocator, weight_tensor, expected_in);
            return orientWeightTyped(allocator, dequantized, expected_in);
        }
        return weight_tensor;
    }

    // Handle FP8 E4M3 weights — keep native when possible for tensor-core GEMM
    if (weight_tensor.dtype == .f8_e4m3) {
        const base = if (std.mem.endsWith(u8, name, ".weight"))
            name[0 .. name.len - ".weight".len]
        else
            name;

        // Try to get scale tensor (need shape info for per-block detection)
        var scale_name_buf: [256]u8 = undefined;
        const scale_name = std.fmt.bufPrint(&scale_name_buf, "{s}.weight_scale_inv", .{base}) catch {
            weight_tensor.fp8 = .{ .scale_inv = 1.0 };
            return weight_tensor;
        };

        const scale_tensor = st.getTensor(scale_name, null) catch {
            log.trace("load", "FP8: no weight_scale_inv found, using 1.0", .{}, @src());
            weight_tensor.fp8 = .{ .scale_inv = 1.0 };
            return weight_tensor;
        };

        // Per-block (2D) scales: keep FP8 native with per-block scale metadata
        if (scale_tensor.n_dims == 2) {
            const s_rows: usize = @intCast(scale_tensor.shape[0]);
            const s_cols: usize = @intCast(scale_tensor.shape[1]);
            const w_rows: usize = @intCast(weight_tensor.shape[0]);
            const w_cols: usize = @intCast(weight_tensor.shape[1]);
            if (s_rows == 0 or s_cols == 0) {
                const dequantized = try dequantizeFp8WeightPerBlock(allocator, weight_tensor, scale_tensor, expected_in);
                return orientWeightTyped(allocator, dequantized, expected_in);
            }
            // Derive block size via ceil division (handles non-aligned last blocks)
            const block_row_size = (w_rows + s_rows - 1) / s_rows;
            const block_col_size = (w_cols + s_cols - 1) / s_cols;
            if (block_row_size != block_col_size) {
                // Non-square blocks — fall back to BF16 dequant
                const dequantized = try dequantizeFp8WeightPerBlock(allocator, weight_tensor, scale_tensor, expected_in);
                return orientWeightTyped(allocator, dequantized, expected_in);
            }
            const block_size = block_row_size;
            const scale_data = scale_tensor.data();
            const scale_byte_len = s_rows * s_cols * @sizeOf(u16);
            if (scale_data.len < scale_byte_len) {
                const dequantized = try dequantizeFp8WeightPerBlock(allocator, weight_tensor, scale_tensor, expected_in);
                return orientWeightTyped(allocator, dequantized, expected_in);
            }
            weight_tensor.fp8 = .{
                .block_scales_data = scale_data.ptr,
                .block_scales_len = scale_byte_len,
                .scale_rows = @intCast(s_rows),
                .scale_cols = @intCast(s_cols),
                .block_size = @intCast(block_size),
            };
            return weight_tensor;
        }

        // Scalar scale (1D or 0D): keep FP8 native with per-tensor scale metadata
        var scale_inv: f32 = 1.0;
        const scale_inv_bytes = scale_tensor.data();
        if (scale_inv_bytes.len >= 2) {
            const scale_inv_bf16 = std.mem.bytesAsValue(u16, scale_inv_bytes[0..2]).*;
            scale_inv = dtype.bf16ToF32(scale_inv_bf16);
        }
        log.trace("load", "FP8 native", .{ .scale_inv = scale_inv }, @src());
        weight_tensor.fp8 = .{ .scale_inv = scale_inv };
        return weight_tensor;
    }

    return switch (weight_tensor.dtype) {
        .f32 => orientWeightF32(allocator, weight_tensor, expected_in),
        .f16, .bf16 => orientWeightTyped(allocator, weight_tensor, expected_in),
        else => weight_tensor,
    };
}

/// Convert GPTQ/AutoRound packed weights to GAF format at load time.
///
/// GPTQ stores quantized weights as three tensors per projection:
///   .qweight  I32  [packed_in, out]      (packed 4-bit values, transposed vs GAF)
///   .scales   F16  [n_groups, out]        (per-group scales, transposed vs GAF)
///   .qzeros   I32  [n_groups, out/pack]   (packed zero points)
///
/// GAF expects:
///   .weight   U32  [out, packed_in]
///   .scales   F16  [out, n_groups]
///   .biases   F16  [out, n_groups]  where bias = -scale * (stored_zp + 1)
///
/// The +1 offset on qzeros matches the AutoGPTQ packing convention.
fn convertGptqToGaffine(
    allocator: std.mem.Allocator,
    st: *st_loader.UnifiedSafeTensors,
    name: []const u8,
    weight_tensor: Tensor,
    expected_in: usize,
    config_gaffine_bits: i32,
) !Tensor {
    const base = if (std.mem.endsWith(u8, name, ".weight"))
        name[0 .. name.len - ".weight".len]
    else if (std.mem.endsWith(u8, name, ".qweight"))
        name[0 .. name.len - ".qweight".len]
    else
        name;

    // GPTQ requires .qzeros (distinguishes from GAF which uses .biases)
    const qzeros_bytes = st.tryGetBytes(base, ".qzeros") orelse return error.MissingScales;
    const scales_bytes = st.tryGetBytes(base, ".scales") orelse return error.MissingScales;

    const packed_shape = weight_tensor.shape[0..@intCast(weight_tensor.n_dims)];
    if (packed_shape.len != 2) return error.InvalidShape;

    const dim0: usize = @intCast(packed_shape[0]);
    const dim1: usize = @intCast(packed_shape[1]);

    // Get scales tensor to determine dtype (F16 or BF16) and shape.
    // GPTQ scales shape: [n_groups, out_features] — use scales dim1 to identify out_features
    // and resolve which weight axis is packed_in vs out.
    var scales_name_buf: [256]u8 = undefined;
    const scales_name = std.fmt.bufPrint(&scales_name_buf, "{s}.scales", .{base}) catch return error.InvalidShape;
    const scales_tensor = st.getTensor(scales_name, null) catch return error.MissingScales;
    const scales_dtype = scales_tensor.dtype;
    if (scales_dtype != .f16 and scales_dtype != .bf16) return error.InvalidShape;

    const scales_shape = scales_tensor.shape[0..@intCast(scales_tensor.n_dims)];
    if (scales_shape.len != 2) return error.InvalidShape;
    const n_groups: usize = @intCast(scales_shape[0]);
    const scales_out: usize = @intCast(scales_shape[1]);

    // Detect orientation from scales: GPTQ qweight[packed_in, out], scales[n_groups, out].
    // If weight dim1 matches scales dim1 → GPTQ orientation [packed_in, out].
    // If weight dim0 matches scales dim1 → already GAF orientation [out, packed_in].
    var out_features: usize = undefined;
    var in_packed: usize = undefined;
    var needs_transpose: bool = undefined;
    if (dim1 == scales_out and dim0 != scales_out) {
        // GPTQ: [packed_in, out]
        in_packed = dim0;
        out_features = dim1;
        needs_transpose = true;
    } else if (dim0 == scales_out and dim1 != scales_out) {
        // Already GAF: [out, packed_in]
        out_features = dim0;
        in_packed = dim1;
        needs_transpose = false;
    } else if (dim0 == scales_out and dim1 == scales_out) {
        // Square — use expected_in to disambiguate
        if (dim0 * 8 == expected_in or dim0 * 4 == expected_in) {
            in_packed = dim0;
            out_features = dim1;
            needs_transpose = true;
        } else {
            out_features = dim0;
            in_packed = dim1;
            needs_transpose = false;
        }
    } else {
        return error.InvalidShape;
    }

    // Auto-detect bit width from group size validity
    const unpacked_4bit = in_packed * 8;
    const unpacked_8bit = in_packed * 4;
    const group_size_4bit = if (n_groups > 0) unpacked_4bit / n_groups else 0;
    const group_size_8bit = if (n_groups > 0) unpacked_8bit / n_groups else 0;
    const valid_4bit = (group_size_4bit == 32 or group_size_4bit == 64 or group_size_4bit == 128);
    const valid_8bit = (group_size_8bit == 32 or group_size_8bit == 64 or group_size_8bit == 128);

    const is_4bit = if (valid_4bit and valid_8bit)
        (if (config_gaffine_bits == 8) false else true)
    else
        valid_4bit;

    if (!is_4bit and !valid_8bit) return error.InvalidShape;

    const values_per_word: usize = if (is_4bit) 8 else 4;
    const group_size: usize = if (is_4bit) group_size_4bit else group_size_8bit;
    const actual_dtype: DType = if (is_4bit) .grouped_affine_u4 else .grouped_affine_u8;
    const bits_per_val: u5 = if (is_4bit) 4 else 8;

    // --- 1. Transpose weight data: [in_packed, out] → [out, in_packed] ---
    const weight_data = weight_tensor.data();
    const weight_u32_count = weight_data.len / 4;
    if (weight_u32_count != in_packed * out_features) return error.InvalidShape;

    const transposed_weight = try allocator.alloc(u8, weight_data.len); // lint:ignore errdefer-alloc - arena freed atomically
    {
        const src: [*]align(1) const u32 = @ptrCast(weight_data.ptr);
        const dst: [*]align(1) u32 = @ptrCast(transposed_weight.ptr);
        if (needs_transpose) {
            for (0..in_packed) |r| {
                for (0..out_features) |c| {
                    dst[c * in_packed + r] = src[r * out_features + c];
                }
            }
        } else {
            @memcpy(transposed_weight, weight_data);
        }
    }

    // --- 2. Transpose scales: [n_groups, out] → [out, n_groups] ---
    const scales_f16_count = scales_bytes.len / 2;
    if (scales_f16_count != n_groups * out_features) return error.InvalidShape;

    const transposed_scales = try allocator.alloc(u8, scales_bytes.len); // lint:ignore errdefer-alloc - arena freed atomically
    {
        const src: [*]align(1) const u16 = @ptrCast(scales_bytes.ptr);
        const dst: [*]align(1) u16 = @ptrCast(transposed_scales.ptr);
        if (needs_transpose) {
            for (0..n_groups) |g| {
                for (0..out_features) |o| {
                    dst[o * n_groups + g] = src[g * out_features + o];
                }
            }
        } else {
            @memcpy(transposed_scales, scales_bytes);
        }
    }

    // --- 3. Convert qzeros → biases ---
    // qzeros: I32 [n_groups, out_features / vals_per_word], packed zero points.
    // AutoGPTQ convention: actual_zp = stored_zp + 1.
    // GAF bias = -scale * actual_zp, stored as F16/BF16 [out, n_groups].
    const zp_per_word: usize = @as(usize, 32) / bits_per_val;
    const zp_packed_cols = (out_features + zp_per_word - 1) / zp_per_word;
    const qzeros_u32_count = qzeros_bytes.len / 4;
    if (qzeros_u32_count != n_groups * zp_packed_cols) return error.InvalidShape;

    const biases_count = n_groups * out_features;
    const transposed_biases = try allocator.alloc(u8, biases_count * 2); // lint:ignore errdefer-alloc - arena freed atomically
    {
        const qz: [*]align(1) const u32 = @ptrCast(qzeros_bytes.ptr);
        const scales_src: [*]align(1) const u16 = @ptrCast(scales_bytes.ptr);
        const dst: [*]align(1) u16 = @ptrCast(transposed_biases.ptr);
        const mask: u32 = (@as(u32, 1) << bits_per_val) - 1;

        for (0..n_groups) |g| {
            for (0..zp_packed_cols) |j| {
                const packed_zp = qz[g * zp_packed_cols + j];
                for (0..zp_per_word) |k| {
                    const o = j * zp_per_word + k;
                    if (o >= out_features) break;
                    const stored_zp: u32 = (packed_zp >> @as(u5, @intCast(k * bits_per_val))) & mask;
                    const actual_zp_f32: f32 = @floatFromInt(stored_zp + 1);
                    // scale is in GPTQ layout [g][o]
                    const scale_u16 = scales_src[g * out_features + o];
                    const scale_f32 = if (scales_dtype == .f16)
                        dtype.fp16ToF32(scale_u16)
                    else
                        dtype.bf16ToF32(scale_u16);
                    const bias_f32 = -scale_f32 * actual_zp_f32;
                    const bias_u16 = if (scales_dtype == .f16)
                        dtype.f32ToFp16(bias_f32)
                    else
                        @as(u16, @truncate(@as(u32, @bitCast(bias_f32)) >> 16));
                    // Store in GAF layout [o][g]
                    dst[o * n_groups + g] = bias_u16;
                }
            }
        }
    }

    // Build result tensor with GAF metadata
    const k_unpacked = in_packed * values_per_word;
    const n_groups_actual = k_unpacked / group_size;
    if (n_groups_actual > MAX_SUPPORTED_GAFFINE_GROUPS) return error.TooManyGroups;

    var result = weight_tensor;
    result.dtype = actual_dtype;
    result.shape[0] = @intCast(out_features);
    result.shape[1] = @intCast(k_unpacked);
    result.data_ptr = transposed_weight.ptr;
    result.data_size = transposed_weight.len;
    result.gaffine = .{
        .scales = transposed_scales,
        .biases = transposed_biases,
        .group_size = group_size,
        .scales_dtype = scales_dtype,
    };

    log.info("load", "Converted GPTQ weight to GAF format", .{
        .name = name,
        .out_features = out_features,
        .in_features = k_unpacked,
        .n_groups = n_groups,
        .group_size = group_size,
        .bits = @as(u8, if (is_4bit) 4 else 8),
    });

    return result;
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

    // Derive block size from scale grid (handles non-aligned dimensions via ceil division).
    // scale_rows = ceil(rows / block_size), so block_size = ceil(rows / scale_rows).
    if (scale_rows == 0 or scale_cols == 0) return error.InvalidShape;
    const block_row_size = (rows + scale_rows - 1) / scale_rows;
    const block_col_size = (cols + scale_cols - 1) / scale_cols;

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
    if (inferGaffineParams(st, name, &embed_tensor, expected_dim, config.gaffine_bits)) |params| {
        try applyGaffineParams(&embed_tensor, params, name);
        return embed_tensor;
    } else if (embed_tensor.dtype == .grouped_affine_u4 or embed_tensor.dtype == .grouped_affine_u8) {
        // Gaffine tensor but inference failed - missing scales/biases
        return error.MissingScales;
    }

    // Keep dense embedding tables in their native dtype so tied-lm-head models
    // can execute logits projection on native precision paths.
    _ = allocator;
    return switch (embed_tensor.dtype) {
        .f32, .f16, .bf16 => embed_tensor,
        else => error.UnexpectedDType,
    };
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

test "buildGatedDeltaSplitInProj concatenates z qkv dt rows" {
    const allocator = std.testing.allocator;

    var z_data = [_]f32{ 1, 2, 3, 4 };
    var qkv_data = [_]f32{ 5, 6, 7, 8, 9, 10 };
    var dt_data = [_]f32{ 11, 12 };

    var z = Tensor.view2DSlice(z_data[0..], 2, 2);
    var qkv = Tensor.view2DSlice(qkv_data[0..], 3, 2);
    var dt = Tensor.view2DSlice(dt_data[0..], 1, 2);

    const fused = try buildGatedDeltaSplitInProj(allocator, &z, &qkv, &dt, null);
    defer @constCast(fused).deinit(allocator);

    try std.testing.expectEqual(@as(i32, 2), fused.n_dims);
    try std.testing.expectEqual(@as(i64, 6), fused.shape[0]);
    try std.testing.expectEqual(@as(i64, 2), fused.shape[1]);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, fused.asSlice(f32));
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
    const result = try orientWeight(arena_alloc.allocator(), &st, "weight", 2, config, false);

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
    const result = try orientWeight(arena_alloc.allocator(), &st, "weight", 2, config, false);

    try std.testing.expectEqual(@as(u8, 2), result.n_dims);
    try std.testing.expectEqual(@as(i64, 2), result.shape[0]);
    try std.testing.expectEqual(@as(i64, 3), result.shape[1]);

    // Data should be unchanged
    const out = result.asSlice(f32);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out[2], 1e-6);
}

test "orientEmbedding keeps f16 embedding dtype" {
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

    try std.testing.expectEqual(DType.f16, result.dtype);
    try std.testing.expectEqual(@as(u8, 2), result.n_dims);
    try std.testing.expectEqual(@as(i64, 4), result.shape[0]);
    try std.testing.expectEqual(@as(i64, 2), result.shape[1]);

    const out = result.asSliceUnaligned(u16);
    try std.testing.expectEqual(@as(u16, 0x3C00), out[0]);
    try std.testing.expectEqual(@as(u16, 0x4000), out[1]);
    try std.testing.expectEqual(@as(u16, 0x4200), out[2]);
    try std.testing.expectEqual(@as(u16, 0x4400), out[3]);
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
