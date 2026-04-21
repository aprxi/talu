//! Weight upload dispatch helpers for the CUDA inference backend.

const host = @import("host.zig");
const dense = @import("dense.zig");
const gaffine = @import("gaffine.zig");
const fp8 = @import("fp8.zig");
const mxfp8 = @import("mxfp8.zig");
const nvfp4 = @import("nvfp4.zig");
const materializeTensorF32 = host.materializeTensorF32;
const uploadLinearWeightDense = dense.uploadLinearWeightDense;
const uploadLinearWeightGroupedAffineU4 = gaffine.uploadLinearWeightGroupedAffineU4;
const uploadLinearWeightGroupedAffineU8 = gaffine.uploadLinearWeightGroupedAffineU8;
const uploadLinearWeightMxfp8 = mxfp8.uploadLinearWeightMxfp8;
const uploadLinearWeightNvfp4 = nvfp4.uploadLinearWeightNvfp4;
const uploadLinearWeightFp8 = fp8.uploadLinearWeightFp8;
const uploadLinearWeightBf16AsFp8 = fp8.uploadLinearWeightBf16AsFp8;


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

pub fn uploadTensor(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    src: *const Tensor,
) !DeviceTensor {
    if (src.n_dims < 1 or src.n_dims > 2) return error.UnsupportedModel;
    if (src.shape[0] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(src.shape[0]);
    const cols: usize = if (src.n_dims == 1) 1 else blk: {
        if (src.shape[1] <= 0) return error.InvalidArgument;
        break :blk @intCast(src.shape[1]);
    };

    const host_f32 = try materializeTensorF32(allocator, src);
    defer allocator.free(host_f32);
    var buffer = try device.allocBuffer(host_f32.len * @sizeOf(f32));
    errdefer buffer.deinit(device);
    try buffer.upload(device, std.mem.sliceAsBytes(host_f32));

    return .{
        .rows = rows,
        .cols = cols,
        .buffer = buffer,
    };
}

pub fn uploadLinearWeight(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    src: *const Tensor,
    input_dim: usize,
) !LinearWeight {
    if (src.dtype == .grouped_affine_u4) {
        return uploadLinearWeightGroupedAffineU4(device, src, input_dim);
    }
    if (src.dtype == .grouped_affine_u8) {
        return uploadLinearWeightGroupedAffineU8(device, src, input_dim);
    }
    if (src.dtype == .f8_e4m3 and src.mxfp8 != null) {
        return uploadLinearWeightMxfp8(device, allocator, src, input_dim);
    }
    if ((src.dtype == .i8 or src.dtype == .u8) and src.nvfp4 != null) {
        return uploadLinearWeightNvfp4(device, allocator, src, input_dim);
    }
    if (src.dtype == .f8_e4m3) {
        return uploadLinearWeightFp8(device, src, input_dim);
    }
    // TALU_CUDA_QUANTIZE_FP8=1: runtime-quantize BF16 weights to FP8 E4M3
    // with per-block [128,128] scales. Halves DRAM bandwidth for decode.
    if (src.dtype == .bf16 and engine_types.resolveCudaQuantizeFp8()) {
        return uploadLinearWeightBf16AsFp8(device, allocator, src, input_dim) catch
            uploadLinearWeightDense(device, allocator, src, input_dim);
    }
    return uploadLinearWeightDense(device, allocator, src, input_dim);
}


pub fn uploadLinearWeightWithContext(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    src: *const Tensor,
    input_dim: usize,
    layer_idx: usize,
    weight_name: []const u8,
) !LinearWeight {
    return uploadLinearWeight(device, allocator, src, input_dim) catch |err| {
        if (src.n_dims == 2) {
            log.warn("inference", "CUDA linear weight upload failed", .{
                .layer = layer_idx,
                .weight = weight_name,
                .rows = src.shape[0],
                .cols = src.shape[1],
                .input_dim = input_dim,
                .dtype = @tagName(src.dtype),
                .reason = @errorName(err),
            });
        } else {
            log.warn("inference", "CUDA linear weight upload failed", .{
                .layer = layer_idx,
                .weight = weight_name,
                .n_dims = src.n_dims,
                .input_dim = input_dim,
                .dtype = @tagName(src.dtype),
                .reason = @errorName(err),
            });
        }
        return err;
    };
}


pub fn uploadVectorTensor(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    src: *const Tensor,
    expected_len: usize,
) !DeviceTensor {
    if (expected_len == 0) return error.InvalidArgument;
    const values = try materializeTensorF32(allocator, src);
    defer allocator.free(values);
    if (values.len != expected_len) return error.UnsupportedModel;

    var buffer = try device.allocBuffer(values.len * @sizeOf(f32));
    errdefer buffer.deinit(device);
    try buffer.upload(device, std.mem.sliceAsBytes(values));
    return .{
        .rows = expected_len,
        .cols = 1,
        .buffer = buffer,
    };
}

pub fn allocZeroedF32Buffer(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    count: usize,
) !compute.cuda.Buffer {
    if (count == 0) return error.InvalidArgument;
    const zeros = try allocator.alloc(f32, count);
    defer allocator.free(zeros);
    @memset(zeros, 0.0);
    var buffer = try device.allocBuffer(count * @sizeOf(f32));
    errdefer buffer.deinit(device);
    try buffer.upload(device, std.mem.sliceAsBytes(zeros));
    return buffer;
}

