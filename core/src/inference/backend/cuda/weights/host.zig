//! Host-side utility helpers for the CUDA inference backend.

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
