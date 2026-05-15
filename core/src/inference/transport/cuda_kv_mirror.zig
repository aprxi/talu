//! CPU KV-cache to CUDA mirror transport helpers.
//!
//! This module owns the host staging, f16 conversion, row transposition, and
//! CUDA buffer upload mechanics for process-local KV mirror handoff.

const std = @import("std");

const cuda_activation = @import("cuda_activation.zig");

pub fn uploadCpuKvToCudaMirrors(
    gpu_backend: anytype,
    cpu_backend: anytype,
    slot_index: usize,
    position_start: usize,
    position_count: usize,
) !void {
    const GpuType = @TypeOf(gpu_backend.*);
    if (comptime !@hasField(GpuType, "block_runtime")) return;
    const BlockRuntimeType = @TypeOf(gpu_backend.block_runtime);
    if (comptime !@hasField(BlockRuntimeType, "replicated_kv_sources")) return;
    const replicated_sources = gpu_backend.block_runtime.replicated_kv_sources;
    if (replicated_sources.len == 0) return;

    const mirrors = gpu_backend.block_runtime.mirror_kv;
    const head_dim = gpu_backend.head_dim;
    const allocator = gpu_backend.allocator;

    if (gpu_backend.kv_cache_dtype != .f16) return error.UnsupportedModel;

    for (replicated_sources, 0..) |source, mirror_index| {
        const mirror = &mirrors[mirror_index];
        const cpu_kv = cpu_backend.kv_cache.getLayer(source.global_layer_idx);
        const kv_dim = source.kv_dim;
        if (head_dim == 0 or kv_dim == 0 or kv_dim % head_dim != 0) return error.InvalidArgument;
        const kv_heads = kv_dim / head_dim;
        const total_elems = std.math.mul(usize, position_count, kv_dim) catch return error.InvalidArgument;
        const total_bytes = std.math.mul(usize, total_elems, @sizeOf(f16)) catch return error.InvalidArgument;
        const row_bytes = std.math.mul(usize, kv_dim, @sizeOf(f16)) catch return error.InvalidArgument;

        const staging_f16 = try allocator.alloc(f16, total_elems);
        defer allocator.free(staging_f16);

        try stageCpuKvRowsToF16(cpu_kv, slot_index, position_start, position_count, kv_heads, head_dim, .key, staging_f16);
        const gpu_k_offset = std.math.mul(usize, position_start, row_bytes) catch return error.InvalidArgument;
        const k_slice = try cuda_activation.cudaBufferSlice(&mirror.k, gpu_k_offset, total_bytes);
        try k_slice.upload(&gpu_backend.device, std.mem.sliceAsBytes(staging_f16));

        try stageCpuKvRowsToF16(cpu_kv, slot_index, position_start, position_count, kv_heads, head_dim, .value, staging_f16);
        const gpu_v_offset = std.math.mul(usize, position_start, row_bytes) catch return error.InvalidArgument;
        const v_slice = try cuda_activation.cudaBufferSlice(&mirror.v, gpu_v_offset, total_bytes);
        try v_slice.upload(&gpu_backend.device, std.mem.sliceAsBytes(staging_f16));
    }
}

const CpuKvPlane = enum {
    key,
    value,
};

fn stageCpuKvRowsToF16(
    cpu_kv: anytype,
    slot_index: usize,
    position_start: usize,
    position_count: usize,
    kv_heads: usize,
    head_dim: usize,
    comptime plane: CpuKvPlane,
    staging_f16: []f16,
) !void {
    const kv_dim = std.math.mul(usize, kv_heads, head_dim) catch return error.InvalidArgument;
    const expected_elems = std.math.mul(usize, position_count, kv_dim) catch return error.InvalidArgument;
    if (staging_f16.len != expected_elems) return error.InvalidArgument;
    for (0..position_count) |position_offset| {
        const position = std.math.add(usize, position_start, position_offset) catch return error.InvalidArgument;
        const row_offset = std.math.mul(usize, position_offset, kv_dim) catch return error.InvalidArgument;
        for (0..kv_heads) |kv_head| {
            const source = switch (plane) {
                .key => cpu_kv.getK(slot_index, kv_head, position),
                .value => cpu_kv.getV(slot_index, kv_head, position),
            };
            if (source.len < head_dim) return error.InvalidArgument;
            const head_base = std.math.mul(usize, kv_head, head_dim) catch return error.InvalidArgument;
            const head_offset = std.math.add(usize, row_offset, head_base) catch return error.InvalidArgument;
            for (0..head_dim) |dim| {
                staging_f16[head_offset + dim] = @floatCast(source[dim]);
            }
        }
    }
}
