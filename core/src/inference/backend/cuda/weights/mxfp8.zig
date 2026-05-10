//! MXFP8 weight helpers for the CUDA inference backend.

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
