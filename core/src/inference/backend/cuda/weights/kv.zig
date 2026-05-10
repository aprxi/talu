//! KV and device-buffer helpers for the CUDA inference backend.

const dense = @import("dense.zig");
const host = @import("host.zig");
const transposeRowMajor = dense.transposeRowMajor;
const materializeTensorF32 = host.materializeTensorF32;

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

pub fn bufferSlice(buffer: *const compute.cuda.Buffer, byte_offset: usize, byte_len: usize) !compute.cuda.Buffer {
    if (byte_offset > buffer.size) return error.InvalidArgument;
    const end = std.math.add(usize, byte_offset, byte_len) catch return error.InvalidArgument;
    if (end > buffer.size) return error.InvalidArgument;
    const ptr = std.math.add(u64, buffer.pointer, @intCast(byte_offset)) catch return error.InvalidArgument;
    return .{
        .pointer = ptr,
        .size = byte_len,
    };
}

/// Device KV pair allocation seam.
///
/// Keep all CUDA KV size/allocation logic centralized here so future KV
/// backends (host/offloaded/paged) can replace this path without broad
/// call-site churn.
pub const DeviceKvPair = struct {
    k: compute.cuda.Buffer,
    v: compute.cuda.Buffer,
    k_scale: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 },
    v_scale: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 },
};

/// Element size for f16 KV cache. Use kvCacheElementBytesForDtype for dtype-aware paths.
pub fn kvCacheElementBytes() usize {
    return @sizeOf(u16);
}

pub fn kvCacheElementBytesForDtype(kv_dtype: KvCacheDtype) usize {
    return kv_dtype.elementBytes();
}

pub fn kvCacheBytesForCapacity(capacity: usize, kv_dim: usize) !usize {
    const elems = std.math.mul(usize, capacity, kv_dim) catch return error.InvalidArgument;
    return std.math.mul(usize, elems, kvCacheElementBytes()) catch return error.InvalidArgument;
}

pub fn kvCacheBytesForCapacityDtype(capacity: usize, kv_dim: usize, kv_dtype: KvCacheDtype) !usize {
    const elems = std.math.mul(usize, capacity, kv_dim) catch return error.InvalidArgument;
    return std.math.mul(usize, elems, kvCacheElementBytesForDtype(kv_dtype)) catch return error.InvalidArgument;
}

/// Scale buffer size = capacity × n_kv_heads × sizeof(f32).
pub fn kvScaleBytesForCapacity(capacity: usize, n_kv_heads: usize) !usize {
    const elems = std.math.mul(usize, capacity, n_kv_heads) catch return error.InvalidArgument;
    return std.math.mul(usize, elems, @sizeOf(f32)) catch return error.InvalidArgument;
}

pub fn allocDeviceKvPair(
    device: *compute.cuda.Device,
    capacity: usize,
    kv_dim: usize,
) !DeviceKvPair {
    const bytes = try kvCacheBytesForCapacity(capacity, kv_dim);
    var k = try device.allocBuffer(bytes);
    errdefer k.deinit(device);
    var v = try device.allocBuffer(bytes);
    errdefer v.deinit(device);
    return .{ .k = k, .v = v };
}

pub fn allocDeviceKvPairWithScales(
    device: *compute.cuda.Device,
    capacity: usize,
    kv_dim: usize,
    n_kv_heads: usize,
    kv_dtype: KvCacheDtype,
) !DeviceKvPair {
    const cache_bytes = try kvCacheBytesForCapacityDtype(capacity, kv_dim, kv_dtype);
    var k = try device.allocBuffer(cache_bytes);
    errdefer k.deinit(device);
    var v = try device.allocBuffer(cache_bytes);
    errdefer v.deinit(device);
    if (kv_dtype.hasPerHeadScales()) {
        const scale_bytes = try kvScaleBytesForCapacity(capacity, n_kv_heads);
        var k_scale = try device.allocBuffer(scale_bytes);
        errdefer k_scale.deinit(device);
        var v_scale = try device.allocBuffer(scale_bytes);
        errdefer v_scale.deinit(device);
        return .{ .k = k, .v = v, .k_scale = k_scale, .v_scale = v_scale };
    }
    return .{ .k = k, .v = v };
}

pub fn uploadShortConvWeightTimeMajor(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    src: *const Tensor,
    conv_dim: usize,
    d_conv: usize,
) !DeviceTensor {
    if (src.n_dims < 2 or src.n_dims > 3) return error.UnsupportedModel;
    if (src.shape[0] <= 0 or src.shape[1] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(src.shape[0]);
    const cols: usize = blk: {
        if (src.n_dims == 2) break :blk @intCast(src.shape[1]);
        if (src.shape[2] <= 0) return error.InvalidArgument;
        const dim1: usize = @intCast(src.shape[1]);
        const dim2: usize = @intCast(src.shape[2]);
        if (dim1 == 1) break :blk dim2;
        if (dim2 == 1) break :blk dim1;
        log.warn("inference", "CUDA shortconv conv1d 3D layout unsupported", .{
            .shape0 = src.shape[0],
            .shape1 = src.shape[1],
            .shape2 = src.shape[2],
        });
        return error.UnsupportedModel;
    };
    const expected = std.math.mul(usize, conv_dim, d_conv) catch return error.InvalidArgument;
    const host_f32 = try materializeTensorF32(allocator, src);
    defer allocator.free(host_f32);
    if (host_f32.len != expected) return error.InvalidArgument;

    var oriented: []f32 = host_f32;
    var transposed: ?[]f32 = null;
    defer if (transposed) |t| allocator.free(t);

    if (rows == conv_dim and cols == d_conv) {
        // Convert channel-major [conv_dim, d_conv] -> time-major [d_conv, conv_dim].
        transposed = try transposeRowMajor(f32, allocator, host_f32, rows, cols);
        oriented = transposed.?;
    } else if (!(rows == d_conv and cols == conv_dim)) {
        log.warn("inference", "CUDA shortconv conv1d weight shape unsupported", .{
            .rows = rows,
            .cols = cols,
            .conv_dim = conv_dim,
            .d_conv = d_conv,
        });
        return error.UnsupportedModel;
    }

    var buffer = try device.allocBuffer(oriented.len * @sizeOf(f32));
    errdefer buffer.deinit(device);
    try buffer.upload(device, std.mem.sliceAsBytes(oriented));
    return .{
        .rows = d_conv,
        .cols = conv_dim,
        .buffer = buffer,
    };
}
