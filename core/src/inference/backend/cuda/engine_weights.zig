//! Weight upload, materialization, and host-side utility functions.
//!
//! Free functions extracted from engine.zig to break the circular dependency
//! between engine_types.zig (which contains RuntimeBuffers/BlockRuntime) and
//! the weight upload functions they call during init.
//!
//! Functions that originally took `*CudaBackend` or `*const CudaBackend` use
//! `self: anytype` to avoid importing engine.zig (same pattern as decode.zig).

const std = @import("std");
const compute = @import("../../../compute/root.zig");
const tensor = @import("../../../tensor.zig");
const dtype = @import("../../../dtype.zig");
const log = @import("../../../log.zig");
const load_transforms = @import("../../../models/load/transforms.zig");
const models = @import("../../../models/root.zig");
const GateUpLayout = models.runtime_blocks.GateUpLayout;

const LoadedModel = models.LoadedModel;
const Tensor = tensor.Tensor;

// --- Shared types from engine_types.zig ---
const engine_types = @import("engine_types.zig");
const KvCacheDtype = engine_types.KvCacheDtype;
const gaffine_scales_dtype_f16 = engine_types.gaffine_scales_dtype_f16;
const gaffine_scales_dtype_bf16 = engine_types.gaffine_scales_dtype_bf16;
const DenseU16Dtype = engine_types.DenseU16Dtype;
const EmbeddingLookupKind = engine_types.EmbeddingLookupKind;
const EmbeddingLookup = engine_types.EmbeddingLookup;
const LinearWeight = engine_types.LinearWeight;
const DeviceTensor = engine_types.DeviceTensor;
const missing_device_tensor = engine_types.missing_device_tensor;
const missing_host_tensor = engine_types.missing_host_tensor;
const enable_device_embedding_lookup = engine_types.enable_device_embedding_lookup;

pub fn argmaxHost(values: []const f32) u32 {
    var best_idx: usize = 0;
    var best_val: f32 = -std.math.inf(f32);
    for (values, 0..) |v, idx| {
        if (v > best_val) {
            best_val = v;
            best_idx = idx;
        }
    }
    return @intCast(best_idx);
}

pub fn argminHost(values: []const f32) u32 {
    var best_idx: usize = 0;
    var best_val: f32 = std.math.inf(f32);
    for (values, 0..) |v, idx| {
        if (v < best_val) {
            best_val = v;
            best_idx = idx;
        }
    }
    return @intCast(best_idx);
}

pub fn bytesToMiB(bytes: usize) f32 {
    return @as(f32, @floatFromInt(bytes)) / (1024.0 * 1024.0);
}

pub fn populatePrefillHiddenFromTokens(
    loaded: *const LoadedModel,
    tokens: []const u32,
    d_model: usize,
    out: []f32,
    skip_token_id: ?u32,
) !void {
    if (d_model == 0) return error.InvalidArgument;
    const expected = std.math.mul(usize, tokens.len, d_model) catch return error.InvalidArgument;
    if (out.len != expected) return error.InvalidArgument;

    var idx: usize = 0;
    while (idx < tokens.len) : (idx += 1) {
        const row_start = std.math.mul(usize, idx, d_model) catch return error.InvalidArgument;
        const row = out[row_start .. row_start + d_model];
        if (skip_token_id) |skip_id| {
            if (tokens[idx] == skip_id) {
                @memset(row, 0.0);
                continue;
            }
        }
        const used_model_embeddings = tryPopulateHiddenFromToken(loaded, tokens[idx], row) catch |err| switch (err) {
            error.InvalidArgument => return error.InvalidArgument,
            else => return err,
        };
        if (!used_model_embeddings) return error.UnsupportedModel;
        if (loaded.config.embedding_multiplier != 1.0) {
            for (row) |*value| value.* *= loaded.config.embedding_multiplier;
        }
    }
}

pub fn selectNextTokenFromLogits(self: anytype, logits: []const f32) !u32 {
    if (logits.len != self.vocab_size) return error.InvalidArgument;
    return if (self.loaded.config.logits_scaling < 0.0) argminHost(logits) else argmaxHost(logits);
}

pub fn selectNextTokenFromDeviceLogits(self: anytype) !u32 {
    if (self.runtime_buffers.projected_vocab == 0) return error.InvalidArgument;
    if (self.runtime_buffers.projected_vocab > std.math.maxInt(u32)) return error.InvalidArgument;
    if (self.loaded.config.logits_scaling < 0.0) {
        try self.runtime_buffers.logits_dev.download(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.projected_logits_host));
        return argminHost(self.runtime_buffers.projected_logits_host);
    }

    const argmax_function = self.argmax_function orelse return error.CudaKernelUnavailable;
    const count_u32: u32 = @intCast(self.runtime_buffers.projected_vocab);
    try compute.cuda.argmax.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        argmax_function,
        &self.runtime_buffers.logits_dev,
        &self.argmax_index_dev,
        count_u32,
    );
    var token: u32 = 0;
    try self.argmax_index_dev.download(&self.device, std.mem.asBytes(&token));
    return token;
}

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
    if (kv_dtype == .i8) {
        const scale_bytes = try kvScaleBytesForCapacity(capacity, n_kv_heads);
        var k_scale = try device.allocBuffer(scale_bytes);
        errdefer k_scale.deinit(device);
        var v_scale = try device.allocBuffer(scale_bytes);
        errdefer v_scale.deinit(device);
        return .{ .k = k, .v = v, .k_scale = k_scale, .v_scale = v_scale };
    }
    return .{ .k = k, .v = v };
}

pub fn resizeScratchBuffer(device: *compute.cuda.Device, buffer: *compute.cuda.Buffer, new_size: usize) !void {
    if (new_size == 0) return error.InvalidArgument;
    if (buffer.size == new_size) return;
    var next = try device.allocBuffer(new_size);
    errdefer next.deinit(device);
    buffer.deinit(device);
    buffer.* = next;
}

pub fn freeOwnedTensorView(allocator: std.mem.Allocator, t: Tensor) void {
    if (t.data_ptr) |ptr| {
        const aligned_ptr: [*]align(32) u8 = @alignCast(ptr);
        allocator.free(aligned_ptr[0..t.data_size]);
    }
}

pub fn shouldDownloadPrefillLogits(token_index: usize, token_count: usize) bool {
    std.debug.assert(token_count > 0);
    return token_index + 1 == token_count;
}

pub fn logPrefillTiming(self: anytype, mode: []const u8, token_count: usize, elapsed_ns: u64) void {
    const elapsed_ms: f64 = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const tok_per_s: f64 = if (elapsed_ns == 0)
        0.0
    else
        (@as(f64, @floatFromInt(token_count)) * 1_000_000_000.0) / @as(f64, @floatFromInt(elapsed_ns));
    const dispatches = self.prefillDispatchTotal();
    const dispatches_per_token: f64 = if (token_count == 0)
        0.0
    else
        @as(f64, @floatFromInt(dispatches)) / @as(f64, @floatFromInt(token_count));
    log.info("inference", "CUDA prefill timing", .{
        .mode = mode,
        .tokens = token_count,
        .elapsed_ms = elapsed_ms,
        .tok_per_s = tok_per_s,
        .layer_program_dispatches = dispatches,
        .layer_program_dispatches_per_token = dispatches_per_token,
        .layer_program_rmsnorm = self.prefillDispatchDelta(.rmsnorm),
        .layer_program_attention = self.prefillDispatchDelta(.multihead_attention),
        .layer_program_shortconv = self.prefillDispatchDelta(.shortconv),
        .layer_program_gated_delta = self.prefillDispatchDelta(.gated_delta_net),
        .layer_program_ffn = self.prefillDispatchDelta(.swiglu) + self.prefillDispatchDelta(.moe),
        .layer_program_mamba = self.prefillDispatchDelta(.mamba_mixer),
        .layer_program_residual_add = self.prefillDispatchDelta(.residual_add),
        .layers = self.block_runtime.blocks.len,
        .attention_blocks = self.block_runtime.attention_block_count,
        .shortconv_blocks = self.block_runtime.shortconv_block_count,
        .gated_delta_blocks = self.block_runtime.gated_delta_block_count,
    });
}

pub fn deepstackLayersCompatibleWithPrompt(
    layers: []const []const f32,
    image_positions: usize,
    d_model: usize,
) bool {
    if (d_model == 0) return false;
    for (layers) |layer_features| {
        if (layer_features.len == 0) return false;
        if (layer_features.len % d_model != 0) return false;
        const feature_rows = layer_features.len / d_model;
        if (feature_rows < image_positions) return false;
    }
    return true;
}

pub fn collectTokenPositions(
    allocator: std.mem.Allocator,
    token_ids: []const u32,
    needle: u32,
) ![]usize {
    var count: usize = 0;
    for (token_ids) |token| {
        if (token == needle) count += 1;
    }
    if (count == 0) return &.{};

    const positions = try allocator.alloc(usize, count);
    errdefer allocator.free(positions);

    var write_idx: usize = 0;
    for (token_ids, 0..) |token, idx| {
        if (token != needle) continue;
        positions[write_idx] = idx;
        write_idx += 1;
    }
    std.debug.assert(write_idx == count);
    return positions;
}

pub fn findPositionIndex(positions: []const usize, position: usize) ?usize {
    for (positions, 0..) |value, idx| {
        if (value == position) return idx;
    }
    return null;
}

pub const DenseLinearLayout = struct {
    in_dim: usize,
    out_dim: usize,
    needs_transpose: bool,
};

pub fn resolveDenseInOutLayout(rows: usize, cols: usize, input_dim: usize) !DenseLinearLayout {
    if (input_dim == 0 or rows == 0 or cols == 0) return error.InvalidArgument;
    // Canonical checkpoint layout is [out_dim, in_dim].
    // Prefer this branch first so square matrices (rows == cols == input_dim)
    // are treated as [out,in] and transposed to the kernel layout [in,out].
    if (cols == input_dim) {
        return .{
            .in_dim = cols,
            .out_dim = rows,
            .needs_transpose = true,
        };
    }
    if (rows == input_dim) {
        return .{
            .in_dim = rows,
            .out_dim = cols,
            .needs_transpose = false,
        };
    }
    return error.UnsupportedModel;
}

pub fn resolveDenseOutInLayout(rows: usize, cols: usize, input_dim: usize) !DenseLinearLayout {
    if (input_dim == 0 or rows == 0 or cols == 0) return error.InvalidArgument;
    // Typed (f16/bf16) path follows loader policy: canonical [out_dim, in_dim].
    if (cols == input_dim) {
        return .{
            .in_dim = cols,
            .out_dim = rows,
            .needs_transpose = false,
        };
    }
    if (rows == input_dim) {
        return .{
            .in_dim = rows,
            .out_dim = cols,
            .needs_transpose = true,
        };
    }
    return error.UnsupportedModel;
}

pub fn transposeRowMajor(
    comptime T: type,
    allocator: std.mem.Allocator,
    src: []align(1) const T,
    rows: usize,
    cols: usize,
) ![]T {
    const logical_count = std.math.mul(usize, rows, cols) catch return error.InvalidArgument;
    if (src.len < logical_count) return error.InvalidArgument;
    const out = try allocator.alloc(T, logical_count);
    errdefer allocator.free(out);

    var r: usize = 0;
    while (r < rows) : (r += 1) {
        var c: usize = 0;
        while (c < cols) : (c += 1) {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    return out;
}

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

fn uploadLinearWeightFp8(
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

/// Quantize a BF16 weight tensor to FP8 E4M3 with per-block scales on CPU,
/// then upload FP8 bytes + BF16 scales to GPU. Halves DRAM bandwidth per token
/// at the cost of minor quantization error.
fn uploadLinearWeightBf16AsFp8(
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

pub const DenseOutInU16 = struct {
    values: []align(1) const u16,
    owned: ?[]u16 = null,

    pub fn deinit(self: *DenseOutInU16, allocator: std.mem.Allocator) void {
        if (self.owned) |buf| allocator.free(buf);
        self.* = .{ .values = &.{}, .owned = null };
    }
};

pub const DenseOutInF32 = struct {
    values: []const f32,
    owned: ?[]f32 = null,

    pub fn deinit(self: *DenseOutInF32, allocator: std.mem.Allocator) void {
        if (self.owned) |buf| allocator.free(buf);
        self.* = .{ .values = &.{}, .owned = null };
    }
};

pub const FusedQkvUpload = struct {
    q: LinearWeight,
    k: LinearWeight,
    v: LinearWeight,
};

pub const FusedGateUpUpload = struct {
    gate: LinearWeight,
    up: LinearWeight,
};

pub fn materializeDenseOutInU16(
    allocator: std.mem.Allocator,
    src: *const Tensor,
    input_dim: usize,
    out_dim: usize,
) !DenseOutInU16 {
    if (src.n_dims != 2) return error.UnsupportedModel;
    if (src.shape[0] <= 0 or src.shape[1] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(src.shape[0]);
    const cols: usize = @intCast(src.shape[1]);
    const logical_count = std.math.mul(usize, rows, cols) catch return error.InvalidArgument;
    const view = src.asSliceUnaligned(u16);
    if (view.len < logical_count) return error.InvalidArgument;
    const values = view[0..logical_count];

    if (rows == out_dim and cols == input_dim) {
        return .{ .values = values };
    }
    if (rows == input_dim and cols == out_dim) {
        const transposed = try transposeRowMajor(u16, allocator, values, rows, cols);
        return .{ .values = transposed, .owned = transposed };
    }
    return error.UnsupportedModel;
}

pub fn materializeDenseOutInF32(
    allocator: std.mem.Allocator,
    src: *const Tensor,
    input_dim: usize,
    out_dim: usize,
) !DenseOutInF32 {
    if (src.n_dims != 2) return error.UnsupportedModel;
    if (src.shape[0] <= 0 or src.shape[1] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(src.shape[0]);
    const cols: usize = @intCast(src.shape[1]);
    const logical_count = std.math.mul(usize, rows, cols) catch return error.InvalidArgument;
    const view = src.asSlice(f32);
    if (view.len < logical_count) return error.InvalidArgument;
    const values = view[0..logical_count];

    if (rows == out_dim and cols == input_dim) {
        return .{ .values = values };
    }
    if (rows == input_dim and cols == out_dim) {
        const transposed = try transposeRowMajor(f32, allocator, values, rows, cols);
        return .{ .values = transposed, .owned = transposed };
    }
    return error.UnsupportedModel;
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
    if (fused_qkv.dtype == .f16 or fused_qkv.dtype == .bf16) {
        var out_in = try materializeDenseOutInU16(allocator, fused_qkv, input_dim, total_out);
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
        var out_in = try materializeDenseOutInF32(allocator, fused_qkv, input_dim, total_out);
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

pub fn uploadFusedGateUpWeights(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    fused_gate_up: *const Tensor,
    input_dim: usize,
    layout: GateUpLayout,
) !FusedGateUpUpload {
    if (fused_gate_up.n_dims != 2) return error.UnsupportedModel;
    if (fused_gate_up.shape[0] <= 0 or fused_gate_up.shape[1] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(fused_gate_up.shape[0]);
    const cols: usize = @intCast(fused_gate_up.shape[1]);
    const out_dim = if (rows == input_dim) cols else if (cols == input_dim) rows else return error.UnsupportedModel;
    if ((out_dim % 2) != 0) return error.InvalidArgument;
    const d_ff = out_dim / 2;

    if (fused_gate_up.dtype == .f16 or fused_gate_up.dtype == .bf16) {
        var out_in = try materializeDenseOutInU16(allocator, fused_gate_up, input_dim, out_dim);
        defer out_in.deinit(allocator);

        const part_count = std.math.mul(usize, d_ff, input_dim) catch return error.InvalidArgument;
        var gate_vals: []align(1) const u16 = undefined;
        var up_vals: []align(1) const u16 = undefined;
        var gate_owned: ?[]u16 = null;
        var up_owned: ?[]u16 = null;
        defer if (gate_owned) |buf| allocator.free(buf);
        defer if (up_owned) |buf| allocator.free(buf);
        switch (layout) {
            .concat => {
                gate_vals = out_in.values[0..part_count];
                up_vals = out_in.values[part_count .. part_count * 2];
            },
            .interleaved => {
                const gate_tmp = try allocator.alloc(u16, part_count);
                errdefer allocator.free(gate_tmp);
                const up_tmp = try allocator.alloc(u16, part_count);
                errdefer allocator.free(up_tmp);
                var row: usize = 0;
                while (row < d_ff) : (row += 1) {
                    const gate_src_row = (2 * row) * input_dim;
                    const up_src_row = (2 * row + 1) * input_dim;
                    const dst_row = row * input_dim;
                    @memcpy(gate_tmp[dst_row .. dst_row + input_dim], out_in.values[gate_src_row .. gate_src_row + input_dim]);
                    @memcpy(up_tmp[dst_row .. dst_row + input_dim], out_in.values[up_src_row .. up_src_row + input_dim]);
                }
                gate_vals = gate_tmp;
                up_vals = up_tmp;
                gate_owned = gate_tmp;
                up_owned = up_tmp;
            },
        }

        const gate_bytes = std.mem.sliceAsBytes(gate_vals);
        const up_bytes = std.mem.sliceAsBytes(up_vals);
        var gate_tensor = Tensor.view(@constCast(gate_bytes.ptr), &.{ d_ff, input_dim }, fused_gate_up.dtype, gate_bytes.len);
        var up_tensor = Tensor.view(@constCast(up_bytes.ptr), &.{ d_ff, input_dim }, fused_gate_up.dtype, up_bytes.len);
        const gate = try uploadLinearWeight(device, allocator, &gate_tensor, input_dim);
        errdefer {
            var gate_mut = gate;
            gate_mut.deinit(device);
        }
        const up = try uploadLinearWeight(device, allocator, &up_tensor, input_dim);
        return .{ .gate = gate, .up = up };
    }

    if (fused_gate_up.dtype == .f32) {
        var out_in = try materializeDenseOutInF32(allocator, fused_gate_up, input_dim, out_dim);
        defer out_in.deinit(allocator);

        const part_count = std.math.mul(usize, d_ff, input_dim) catch return error.InvalidArgument;
        var gate_vals: []const f32 = undefined;
        var up_vals: []const f32 = undefined;
        var gate_owned: ?[]f32 = null;
        var up_owned: ?[]f32 = null;
        defer if (gate_owned) |buf| allocator.free(buf);
        defer if (up_owned) |buf| allocator.free(buf);
        switch (layout) {
            .concat => {
                gate_vals = out_in.values[0..part_count];
                up_vals = out_in.values[part_count .. part_count * 2];
            },
            .interleaved => {
                const gate_tmp = try allocator.alloc(f32, part_count);
                errdefer allocator.free(gate_tmp);
                const up_tmp = try allocator.alloc(f32, part_count);
                errdefer allocator.free(up_tmp);
                var row: usize = 0;
                while (row < d_ff) : (row += 1) {
                    const gate_src_row = (2 * row) * input_dim;
                    const up_src_row = (2 * row + 1) * input_dim;
                    const dst_row = row * input_dim;
                    @memcpy(gate_tmp[dst_row .. dst_row + input_dim], out_in.values[gate_src_row .. gate_src_row + input_dim]);
                    @memcpy(up_tmp[dst_row .. dst_row + input_dim], out_in.values[up_src_row .. up_src_row + input_dim]);
                }
                gate_vals = gate_tmp;
                up_vals = up_tmp;
                gate_owned = gate_tmp;
                up_owned = up_tmp;
            },
        }

        const gate_bytes = std.mem.sliceAsBytes(gate_vals);
        const up_bytes = std.mem.sliceAsBytes(up_vals);
        var gate_tensor = Tensor.view(@constCast(gate_bytes.ptr), &.{ d_ff, input_dim }, .f32, gate_bytes.len);
        var up_tensor = Tensor.view(@constCast(up_bytes.ptr), &.{ d_ff, input_dim }, .f32, up_bytes.len);
        const gate = try uploadLinearWeight(device, allocator, &gate_tensor, input_dim);
        errdefer {
            var gate_mut = gate;
            gate_mut.deinit(device);
        }
        const up = try uploadLinearWeight(device, allocator, &up_tensor, input_dim);
        return .{ .gate = gate, .up = up };
    }

    return error.UnsupportedModel;
}

pub fn uploadLinearWeightDense(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    src: *const Tensor,
    input_dim: usize,
) !LinearWeight {
    if (src.dtype == .f16 or src.dtype == .bf16) {
        return uploadLinearWeightDenseU16(device, allocator, src, input_dim);
    }
    if (src.n_dims != 2) return error.UnsupportedModel;
    if (src.shape[0] <= 0 or src.shape[1] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(src.shape[0]);
    const cols: usize = @intCast(src.shape[1]);

    const host_f32 = try materializeTensorF32(allocator, src);
    defer allocator.free(host_f32);

    const layout = resolveDenseInOutLayout(rows, cols, input_dim) catch |err| {
        log.warn("inference", "CUDA dense linear weight orientation unsupported", .{
            .rows = rows,
            .cols = cols,
            .input_dim = input_dim,
            .dtype = @tagName(src.dtype),
        });
        return err;
    };

    var oriented: []f32 = host_f32;
    var transposed: ?[]f32 = null;
    defer if (transposed) |t| allocator.free(t);
    if (layout.needs_transpose) {
        transposed = try transposeRowMajor(f32, allocator, host_f32, rows, cols);
        oriented = transposed.?;
    }

    var buffer = try device.allocBuffer(oriented.len * @sizeOf(f32));
    errdefer buffer.deinit(device);
    try buffer.upload(device, std.mem.sliceAsBytes(oriented));

    return .{
        .dense_f32 = .{
            .rows = layout.in_dim,
            .cols = layout.out_dim,
            .buffer = buffer,
        },
    };
}

pub fn uploadLinearWeightDenseU16(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    src: *const Tensor,
    input_dim: usize,
) !LinearWeight {
    if (src.n_dims != 2) return error.UnsupportedModel;
    if (src.shape[0] <= 0 or src.shape[1] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(src.shape[0]);
    const cols: usize = @intCast(src.shape[1]);
    const logical_count = std.math.mul(usize, rows, cols) catch return error.InvalidArgument;

    const host_u16 = src.asSliceUnaligned(u16);
    if (host_u16.len < logical_count) return error.InvalidArgument;
    const view = host_u16[0..logical_count];

    const layout = resolveDenseOutInLayout(rows, cols, input_dim) catch |err| {
        log.warn("inference", "CUDA u16 linear weight orientation unsupported", .{
            .rows = rows,
            .cols = cols,
            .input_dim = input_dim,
            .dtype = @tagName(src.dtype),
        });
        return err;
    };

    var oriented: []align(1) const u16 = view;
    var transposed: ?[]u16 = null;
    defer if (transposed) |t| allocator.free(t);
    if (layout.needs_transpose) {
        transposed = try transposeRowMajor(u16, allocator, view, rows, cols);
        oriented = transposed.?;
    }

    const bytes = std.math.mul(usize, oriented.len, @sizeOf(u16)) catch return error.InvalidArgument;
    var buffer = try device.allocBuffer(bytes);
    errdefer buffer.deinit(device);
    try buffer.upload(device, std.mem.sliceAsBytes(oriented));

    const dense_dtype: DenseU16Dtype = switch (src.dtype) {
        .f16 => .f16,
        .bf16 => .bf16,
        else => return error.UnsupportedModel,
    };

    return .{
        .dense_u16 = .{
            .rows = layout.in_dim,
            .cols = layout.out_dim,
            .buffer = buffer,
            .dtype = dense_dtype,
        },
    };
}

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

pub fn tryUploadEmbeddingLookup(
    device: *compute.cuda.Device,
    loaded: *const LoadedModel,
    d_model: usize,
) !?EmbeddingLookup {
    const embeddings = &loaded.token_embeddings;
    if (embeddings.data_ptr == null) return null;
    if (embeddings.n_dims != 2) return null;
    if (embeddings.shape[0] <= 0 or embeddings.shape[1] <= 0) return null;
    const kind: EmbeddingLookupKind = switch (embeddings.dtype) {
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
        .grouped_affine_u4 => .gaffine_u4,
        else => return null,
    };

    const dim0: usize = @intCast(embeddings.shape[0]);
    const dim1: usize = @intCast(embeddings.shape[1]);
    var layout_tag: u32 = undefined;
    var hidden_dim: usize = undefined;
    if (dim1 == d_model) {
        layout_tag = compute.cuda.embedding_lookup_f32.layout_vocab_hidden;
        hidden_dim = dim1;
    } else if (dim0 == d_model) {
        if (kind == .gaffine_u4) return null;
        layout_tag = compute.cuda.embedding_lookup_f32.layout_hidden_vocab;
        hidden_dim = dim0;
    } else {
        return null;
    }
    const dim0_u32 = std.math.cast(u32, dim0) orelse return error.InvalidArgument;
    const dim1_u32 = std.math.cast(u32, dim1) orelse return error.InvalidArgument;
    const hidden_dim_u32 = std.math.cast(u32, hidden_dim) orelse return error.InvalidArgument;

    if (kind == .gaffine_u4) {
        const gaffine = embeddings.gaffine orelse return null;
        if (gaffine.group_size == 0) return null;
        const group_size: usize = gaffine.group_size;
        if ((hidden_dim % group_size) != 0 or (group_size % 8) != 0) return null;
        const groups_per_row = hidden_dim / group_size;
        const packed_words_per_row = hidden_dim / 8;
        const packed_words_total = std.math.mul(usize, dim0, packed_words_per_row) catch return error.InvalidArgument;
        const sb_count = std.math.mul(usize, dim0, groups_per_row) catch return error.InvalidArgument;
        const packed_bytes = std.math.mul(usize, packed_words_total, @sizeOf(u32)) catch return error.InvalidArgument;
        const sb_bytes = std.math.mul(usize, sb_count, @sizeOf(u16)) catch return error.InvalidArgument;
        const packed_vals = embeddings.asSliceUnaligned(u32);
        if (packed_vals.len < packed_words_total) return error.InvalidArgument;
        if (gaffine.scales.len < sb_bytes or gaffine.biases.len < sb_bytes) return error.InvalidArgument;
        const scales_dtype_tag = switch (gaffine.scales_dtype) {
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
        try packed_dev.upload(device, std.mem.sliceAsBytes(packed_vals[0..packed_words_total]));
        try scales_dev.upload(device, gaffine.scales[0..sb_bytes]);
        try biases_dev.upload(device, gaffine.biases[0..sb_bytes]);

        return .{
            .kind = .gaffine_u4,
            .dim0 = dim0_u32,
            .dim1 = dim1_u32,
            .hidden_dim = hidden_dim_u32,
            .layout_tag = layout_tag,
            .group_size = std.math.cast(u32, group_size) orelse return error.InvalidArgument,
            .scales_dtype_tag = scales_dtype_tag,
            .scales = scales_dev,
            .biases = biases_dev,
            .multiplier = loaded.config.embedding_multiplier,
            .buffer = packed_dev,
        };
    }

    const elem_count = std.math.mul(usize, dim0, dim1) catch return error.InvalidArgument;
    const elem_bytes: usize = switch (kind) {
        .f32 => @sizeOf(f32),
        .f16, .bf16 => @sizeOf(u16),
        .gaffine_u4 => unreachable,
    };
    const bytes = std.math.mul(usize, elem_count, elem_bytes) catch return error.InvalidArgument;
    var buffer = try device.allocBuffer(bytes);
    errdefer buffer.deinit(device);
    switch (kind) {
        .f32 => {
            const src = embeddings.asSlice(f32);
            if (src.len < elem_count) return error.InvalidArgument;
            try buffer.upload(device, std.mem.sliceAsBytes(src[0..elem_count]));
        },
        .f16, .bf16 => {
            const src_u16 = embeddings.asSliceUnaligned(u16);
            if (src_u16.len < elem_count) return error.InvalidArgument;
            try buffer.upload(device, std.mem.sliceAsBytes(src_u16[0..elem_count]));
        },
        .gaffine_u4 => unreachable,
    }

    return .{
        .kind = kind,
        .dim0 = dim0_u32,
        .dim1 = dim1_u32,
        .hidden_dim = hidden_dim_u32,
        .layout_tag = layout_tag,
        .multiplier = loaded.config.embedding_multiplier,
        .buffer = buffer,
    };
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

pub fn materializeTensorF32(allocator: std.mem.Allocator, src: *const Tensor) ![]f32 {
    if (src.data_ptr == null) return error.InvalidArgument;
    if (src.n_dims < 1 or src.n_dims > 3) return error.UnsupportedModel;
    if (src.shape[0] <= 0) return error.InvalidArgument;
    var logical_count: usize = @intCast(src.shape[0]);
    if (src.n_dims >= 2) {
        if (src.shape[1] <= 0) return error.InvalidArgument;
        logical_count = std.math.mul(usize, logical_count, @as(usize, @intCast(src.shape[1]))) catch return error.InvalidArgument;
    }
    if (src.n_dims >= 3) {
        if (src.shape[2] <= 0) return error.InvalidArgument;
        logical_count = std.math.mul(usize, logical_count, @as(usize, @intCast(src.shape[2]))) catch return error.InvalidArgument;
    }

    const out = try allocator.alloc(f32, logical_count);
    errdefer allocator.free(out);

    switch (src.dtype) {
        .f32 => {
            const values = src.asSlice(f32);
            if (values.len < logical_count) return error.InvalidArgument;
            @memcpy(out, values[0..logical_count]);
        },
        .f16, .bf16 => {
            const values = src.asSliceUnaligned(u16);
            if (values.len < logical_count) return error.InvalidArgument;
            for (out, 0..) |*dst, i| {
                dst.* = if (src.dtype == .f16) dtype.fp16ToF32(values[i]) else dtype.bf16ToF32(values[i]);
            }
        },
        .grouped_affine_u4, .grouped_affine_u8 => {
            const dequantized = try load_transforms.convertToF32(allocator, src.*);
            defer freeOwnedTensorView(allocator, dequantized);
            const src_f32 = dequantized.asSlice(f32);
            if (src_f32.len < logical_count) return error.InvalidArgument;
            @memcpy(out, src_f32[0..logical_count]);
        },
        else => return error.UnsupportedModel,
    }

    return out;
}

pub fn canUseModelEmbeddings(loaded: *const LoadedModel, d_model: usize) bool {
    if (d_model == 0) return false;
    const embeddings = loaded.token_embeddings;
    if (embeddings.data_ptr == null) return false;
    if (embeddings.n_dims != 2) return false;
    if (embeddings.shape[0] <= 0 or embeddings.shape[1] <= 0) return false;
    const dim0: usize = @intCast(embeddings.shape[0]);
    const dim1: usize = @intCast(embeddings.shape[1]);
    if (dim0 != d_model and dim1 != d_model) return false;
    return switch (embeddings.dtype) {
        .f32, .f16, .bf16, .grouped_affine_u4, .grouped_affine_u8 => true,
        else => false,
    };
}

pub fn tryPopulateHiddenFromToken(
    loaded: *const LoadedModel,
    token: u32,
    out: []f32,
) !bool {
    const embeddings = &loaded.token_embeddings;
    if (embeddings.data_ptr == null) return false;
    if (embeddings.n_dims != 2) return false;
    if (embeddings.shape[0] <= 0 or embeddings.shape[1] <= 0) return false;

    const dim0: usize = @intCast(embeddings.shape[0]);
    const dim1: usize = @intCast(embeddings.shape[1]);
    const token_idx: usize = @intCast(token);
    const hidden_dim = out.len;

    switch (embeddings.dtype) {
        .f32 => {
            const src = embeddings.asSlice(f32);
            // Common layout: [vocab, d_model].
            if (dim1 == hidden_dim) {
                if (token_idx >= dim0) return error.InvalidArgument;
                const row_start = token_idx * dim1;
                @memcpy(out, src[row_start .. row_start + hidden_dim]);
                return true;
            }

            // Transposed layout: [d_model, vocab].
            if (dim0 == hidden_dim) {
                if (token_idx >= dim1) return error.InvalidArgument;
                var i: usize = 0;
                while (i < hidden_dim) : (i += 1) {
                    out[i] = src[i * dim1 + token_idx];
                }
                return true;
            }

            return false;
        },
        .f16, .bf16 => {
            const src_u16 = embeddings.asSliceUnaligned(u16);
            // Common layout: [vocab, d_model].
            if (dim1 == hidden_dim) {
                if (token_idx >= dim0) return error.InvalidArgument;
                var i: usize = 0;
                while (i < hidden_dim) : (i += 1) {
                    const raw = src_u16[token_idx * dim1 + i];
                    out[i] = if (embeddings.dtype == .f16) dtype.fp16ToF32(raw) else dtype.bf16ToF32(raw);
                }
                return true;
            }

            // Transposed layout: [d_model, vocab].
            if (dim0 == hidden_dim) {
                if (token_idx >= dim1) return error.InvalidArgument;
                var i: usize = 0;
                while (i < hidden_dim) : (i += 1) {
                    const raw = src_u16[i * dim1 + token_idx];
                    out[i] = if (embeddings.dtype == .f16) dtype.fp16ToF32(raw) else dtype.bf16ToF32(raw);
                }
                return true;
            }

            return false;
        },
        .grouped_affine_u4, .grouped_affine_u8 => {
            // Common layout: [vocab, d_model].
            if (dim1 == hidden_dim) {
                if (token_idx >= dim0) return error.InvalidArgument;
                try decodeGaffineRow(embeddings, token_idx, out);
                return true;
            }

            // Transposed layout: [d_model, vocab].
            if (dim0 == hidden_dim) {
                if (token_idx >= dim1) return error.InvalidArgument;
                var i: usize = 0;
                while (i < hidden_dim) : (i += 1) {
                    out[i] = try gaffineValueAt(embeddings, i, token_idx);
                }
                return true;
            }

            return false;
        },
        else => return false,
    }
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

pub fn tryPopulateFinalNormWeight(loaded: *const LoadedModel, out: []f32) bool {
    if (loaded.ln_final) |ln_final| {
        if (ln_final.data_ptr == null or ln_final.numel < out.len) return false;
        switch (ln_final.dtype) {
            .f32 => {
                const src = ln_final.asSlice(f32);
                if (src.len < out.len) return false;
                @memcpy(out, src[0..out.len]);
                return true;
            },
            .f16, .bf16 => {
                const src = ln_final.asSliceUnaligned(u16);
                if (src.len < out.len) return false;
                for (out, 0..) |*v, i| {
                    const raw = src[i];
                    v.* = if (ln_final.dtype == .f16) dtype.fp16ToF32(raw) else dtype.bf16ToF32(raw);
                }
                return true;
            },
            else => return false,
        }
    }
    return false;
}

pub fn tryPopulateProjectionFromWeight(
    allocator: std.mem.Allocator,
    weight: *const Tensor,
    d_model: usize,
    projected_vocab: usize,
    out: []f32,
) bool {
    if (weight.data_ptr == null) return false;
    if (weight.n_dims != 2) return false;

    if (weight.shape[0] <= 0 or weight.shape[1] <= 0) return false;
    const dim0: usize = @intCast(weight.shape[0]);
    const dim1: usize = @intCast(weight.shape[1]);
    switch (weight.dtype) {
        .f32 => {
            const expected_len = std.math.mul(usize, dim0, dim1) catch return false;
            const src = weight.asSlice(f32);
            if (src.len < expected_len) return false;

            // Direct layout: [d_model, vocab] so each row can be copied contiguously.
            if (dim0 == d_model and dim1 >= projected_vocab) {
                var row: usize = 0;
                while (row < d_model) : (row += 1) {
                    const src_start = row * dim1;
                    const dst_start = row * projected_vocab;
                    @memcpy(out[dst_start .. dst_start + projected_vocab], src[src_start .. src_start + projected_vocab]);
                }
                return true;
            }

            // Transposed layout: [vocab, d_model], so gather one column per token row.
            if (dim1 == d_model and dim0 >= projected_vocab) {
                var row: usize = 0;
                while (row < d_model) : (row += 1) {
                    var col: usize = 0;
                    while (col < projected_vocab) : (col += 1) {
                        out[row * projected_vocab + col] = src[col * dim1 + row];
                    }
                }
                return true;
            }

            return false;
        },
        .f16, .bf16 => {
            const expected_len = std.math.mul(usize, dim0, dim1) catch return false;
            const src_u16 = weight.asSliceUnaligned(u16);
            if (src_u16.len < expected_len) return false;

            // Direct layout: [d_model, vocab]
            if (dim0 == d_model and dim1 >= projected_vocab) {
                var row: usize = 0;
                while (row < d_model) : (row += 1) {
                    var col: usize = 0;
                    while (col < projected_vocab) : (col += 1) {
                        const raw = src_u16[row * dim1 + col];
                        out[row * projected_vocab + col] = if (weight.dtype == .f16) dtype.fp16ToF32(raw) else dtype.bf16ToF32(raw);
                    }
                }
                return true;
            }

            // Transposed layout: [vocab, d_model]
            if (dim1 == d_model and dim0 >= projected_vocab) {
                var row: usize = 0;
                while (row < d_model) : (row += 1) {
                    var col: usize = 0;
                    while (col < projected_vocab) : (col += 1) {
                        const raw = src_u16[col * dim1 + row];
                        out[row * projected_vocab + col] = if (weight.dtype == .f16) dtype.fp16ToF32(raw) else dtype.bf16ToF32(raw);
                    }
                }
                return true;
            }

            return false;
        },
        .grouped_affine_u4, .grouped_affine_u8 => {
            const dequantized = load_transforms.convertToF32(allocator, weight.*) catch return false;
            defer freeOwnedTensorView(allocator, dequantized);
            const src = dequantized.asSlice(f32);

            if (dim0 == d_model and dim1 >= projected_vocab) {
                var row: usize = 0;
                while (row < d_model) : (row += 1) {
                    const src_start = row * dim1;
                    const dst_start = row * projected_vocab;
                    @memcpy(out[dst_start .. dst_start + projected_vocab], src[src_start .. src_start + projected_vocab]);
                }
                return true;
            }
            if (dim1 == d_model and dim0 >= projected_vocab) {
                var row: usize = 0;
                while (row < d_model) : (row += 1) {
                    var col: usize = 0;
                    while (col < projected_vocab) : (col += 1) {
                        out[row * projected_vocab + col] = src[col * dim1 + row];
                    }
                }
                return true;
            }
            return false;
        },
        else => return false,
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
