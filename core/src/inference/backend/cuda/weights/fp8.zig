//! FP8 weight helpers for the CUDA inference backend.

const dense = @import("dense.zig");
const resolveDenseOutInLayout = dense.resolveDenseOutInLayout;

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

pub fn uploadLinearWeightFp8(
    device: *compute.cuda.Device,
    src: *const Tensor,
    input_dim: usize,
) !LinearWeight {
    if (src.n_dims != 2) return error.UnsupportedModel;
    const fp8_meta = src.fp8 orelse return error.UnsupportedModel;

    const dim0: usize = @intCast(src.shape[0]);
    const dim1: usize = @intCast(src.shape[1]);

    // Expect [out_dim, in_dim] layout (standard for HF FP8 models)
    var out_dim: usize = undefined;
    var in_dim: usize = undefined;
    if (dim1 == input_dim) {
        out_dim = dim0;
        in_dim = dim1;
    } else if (dim0 == input_dim) {
        // Transposed layout not supported for raw FP8 bytes
        return error.UnsupportedModel;
    } else {
        return error.UnsupportedModel;
    }

    const byte_count = std.math.mul(usize, out_dim, in_dim) catch return error.InvalidArgument;
    const src_bytes = src.data();
    if (src_bytes.len < byte_count) return error.InvalidArgument;

    var buffer = try device.allocBuffer(byte_count);
    errdefer buffer.deinit(device);
    try buffer.upload(device, src_bytes[0..byte_count]);

    // Upload per-block BF16 scales if present
    var scales_buffer: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 };
    if (fp8_meta.block_scales_data) |scales_ptr| {
        if (fp8_meta.block_scales_len > 0) {
            scales_buffer = try device.allocBuffer(fp8_meta.block_scales_len);
            errdefer scales_buffer.deinit(device);
            try scales_buffer.upload(device, scales_ptr[0..fp8_meta.block_scales_len]);
        }
    }

    return .{ .fp8 = .{
        .rows = in_dim,
        .cols = out_dim,
        .buffer = buffer,
        .scales_buffer = scales_buffer,
        .scale_rows = fp8_meta.scale_rows,
        .scale_cols = fp8_meta.scale_cols,
        .block_size = fp8_meta.block_size,
        .weight_scale_inv = fp8_meta.scale_inv,
    } };
}

pub fn uploadLinearWeightMxfp8(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    src: *const Tensor,
    input_dim: usize,
) !LinearWeight {
    if (src.n_dims != 2) return error.UnsupportedModel;
    const mxfp8_meta = src.mxfp8 orelse return error.UnsupportedModel;

    const dim0: usize = @intCast(src.shape[0]);
    const dim1: usize = @intCast(src.shape[1]);

    // Expect [out_dim, in_dim] layout (standard for HF models)
    if (dim1 != input_dim) return error.UnsupportedModel;
    const out_dim = dim0;
    const in_dim = dim1;

    const byte_count = std.math.mul(usize, out_dim, in_dim) catch return error.InvalidArgument;
    const src_bytes = src.data();
    if (src_bytes.len < byte_count) return error.InvalidArgument;

    var buffer = try device.allocBuffer(byte_count);
    errdefer buffer.deinit(device);
    try buffer.upload(device, src_bytes[0..byte_count]);

    // Upload UE8M0 scales in two layouts:
    // 1. Interleaved layout for cuBLASLt tensor core GEMM (n>8)
    // 2. Simple row-major layout for GEMV kernel (n≤8)
    var scales_buffer: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 };
    var scales_raw_buffer: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 };
    if (mxfp8_meta.block_scales_data) |scales_ptr| {
        if (mxfp8_meta.block_scales_len > 0) {
            const sf_k: usize = mxfp8_meta.scale_cols; // = in_dim / 32
            const raw = scales_ptr[0..mxfp8_meta.block_scales_len];

            // Raw row-major scales for GEMV kernel [out_dim × sf_k]
            scales_raw_buffer = try device.allocBuffer(raw.len);
            errdefer scales_raw_buffer.deinit(device);
            try scales_raw_buffer.upload(device, raw);

            // Interleaved scales for cuBLASLt [padded, tiled layout]
            const padded_scale_bytes = engine_types.Mxfp8LinearWeight.cublasLtScaleTensorSize(in_dim, out_dim);
            const padded_sf_k = engine_types.Mxfp8LinearWeight.roundoff(sf_k, 4);

            const interleaved = try allocator.alloc(u8, padded_scale_bytes);
            defer allocator.free(interleaved);
            @memset(interleaved, 0);

            // Interleave: permute [out_dim × sf_k] row-major into cuBLASLt tile layout.
            const n_col_tiles = padded_sf_k / 4;
            for (0..out_dim) |m| {
                for (0..sf_k) |k| {
                    const src_idx = m * sf_k + k;
                    const dst_idx = (m / 128) * n_col_tiles * 512 +
                        (k / 4) * 512 +
                        (m % 32) * 16 +
                        ((m % 128) / 32) * 4 +
                        (k % 4);
                    interleaved[dst_idx] = raw[src_idx];
                }
            }

            scales_buffer = try device.allocBuffer(padded_scale_bytes);
            errdefer scales_buffer.deinit(device);
            try scales_buffer.upload(device, interleaved);
        }
    }

    return .{ .mxfp8 = .{
        .rows = in_dim,
        .cols = out_dim,
        .buffer = buffer,
        .scales_buffer = scales_buffer,
        .scales_raw_buffer = scales_raw_buffer,
        .scale_cols = mxfp8_meta.scale_cols,
    } };
}

/// Quantize a BF16 weight tensor to FP8 E4M3 with per-block scales on CPU,
/// then upload FP8 bytes + BF16 scales to GPU. Halves DRAM bandwidth per token
/// at the cost of minor quantization error.
pub fn uploadLinearWeightBf16AsFp8(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    src: *const Tensor,
    input_dim: usize,
) !LinearWeight {
    if (src.n_dims != 2) return error.UnsupportedModel;
    if (src.dtype != .bf16) return error.UnsupportedModel;
    if (src.shape[0] <= 0 or src.shape[1] <= 0) return error.InvalidArgument;

    const rows: usize = @intCast(src.shape[0]);
    const cols: usize = @intCast(src.shape[1]);
    const elem_count = std.math.mul(usize, rows, cols) catch return error.InvalidArgument;

    const host_u16 = src.asSliceUnaligned(u16);
    if (host_u16.len < elem_count) return error.InvalidArgument;
    const view = host_u16[0..elem_count];

    // Resolve layout: BF16 weights come in [out_dim, in_dim]
    const layout = resolveDenseOutInLayout(rows, cols, input_dim) catch return error.UnsupportedModel;
    if (layout.needs_transpose) {
        // FP8 upload expects [out_dim, in_dim] row-major; transpose not yet supported
        return error.UnsupportedModel;
    }
    const out_dim = layout.out_dim;
    const in_dim = layout.in_dim;

    // Per-block quantization with block_size=128
    const block_size: u32 = 128;
    const bs: usize = @intCast(block_size);
    const scale_rows: u32 = @intCast((out_dim + bs - 1) / bs);
    const scale_cols: u32 = @intCast((in_dim + bs - 1) / bs);
    const n_blocks = @as(usize, scale_rows) * @as(usize, scale_cols);

    // Allocate output buffers
    const fp8_bytes = try allocator.alloc(u8, elem_count);
    defer allocator.free(fp8_bytes);
    const scale_u16s = try allocator.alloc(u16, n_blocks);
    defer allocator.free(scale_u16s);

    // Quantize each block
    for (0..scale_rows) |br| {
        const row_start = br * bs;
        const row_end = @min(row_start + bs, out_dim);
        for (0..scale_cols) |bc| {
            const col_start = bc * bs;
            const col_end = @min(col_start + bs, in_dim);

            // Find absmax in this block
            var block_max: f32 = 0;
            for (row_start..row_end) |r| {
                const row_offset = r * in_dim;
                for (col_start..col_end) |c| {
                    const f = dtype.bf16ToF32(view[row_offset + c]);
                    const a = @abs(f);
                    if (a > block_max) block_max = a;
                }
            }

            // scale_inv = max / 448, stored as BF16
            const scale_inv: f32 = if (block_max > 0) block_max / 448.0 else 1.0;
            const inv_scale: f32 = 1.0 / scale_inv;
            scale_u16s[br * @as(usize, scale_cols) + bc] = dtype.f32ToBf16(scale_inv);

            // Quantize block values
            for (row_start..row_end) |r| {
                const row_offset = r * in_dim;
                for (col_start..col_end) |c| {
                    const f = dtype.bf16ToF32(view[row_offset + c]);
                    fp8_bytes[row_offset + c] = dtype.f32ToFp8E4M3(f * inv_scale);
                }
            }
        }
    }

    // Upload FP8 weight bytes
    var buffer = try device.allocBuffer(elem_count);
    errdefer buffer.deinit(device);
    try buffer.upload(device, fp8_bytes);

    // Upload BF16 per-block scales
    const scales_byte_len = n_blocks * @sizeOf(u16);
    var scales_buffer = try device.allocBuffer(scales_byte_len);
    errdefer scales_buffer.deinit(device);
    try scales_buffer.upload(device, std.mem.sliceAsBytes(scale_u16s));

    return .{ .fp8 = .{
        .rows = in_dim,
        .cols = out_dim,
        .buffer = buffer,
        .scales_buffer = scales_buffer,
        .scale_rows = scale_rows,
        .scale_cols = scale_cols,
        .block_size = block_size,
        .weight_scale_inv = 1.0,
    } };
}
