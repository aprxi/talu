//! Forward pass computation functions.
//!
//! Contains the main forward-pass entry points (single-token decode, batched
//! decode, prefill), KV capacity management, and recurrent state resets.
//! Functions use `self: anytype` to avoid circular imports with engine.zig.

const std = @import("std");
const compute = @import("compute_pkg");
const tensor = @import("compute_pkg").tensor;
const log = @import("log_pkg");
const trace = @import("xray_pkg").trace;
const orchestrator = @import("../../../bridge/orchestrator.zig");
const per_layer_branch_feature = @import("../per_layer_branch.zig");

// --- Shared types from engine_types.zig ---
const engine_types = @import("../runtime/root.zig");
const BatchDecodeInfo = engine_types.BatchDecodeInfo;
const KvCacheDtype = engine_types.KvCacheDtype;
const enable_dispatch_observability = engine_types.enable_dispatch_observability;
const enable_device_embedding_lookup = engine_types.enable_device_embedding_lookup;
const AttentionKernelSet = engine_types.AttentionKernelSet;

// --- Compute ops from engine_ops.zig ---
const engine_ops = @import("../operators/root.zig");

// --- Utilities from engine_weights.zig ---
const engine_weights = @import("../weights/root.zig");
const bufferSlice = engine_weights.bufferSlice;
const populatePrefillHiddenFromTokens = engine_weights.populatePrefillHiddenFromTokens;
const tryPopulateHiddenFromToken = engine_weights.tryPopulateHiddenFromToken;

const saturatingU64FromU128 = engine_types.saturatingU64FromU128;
const logicalF32RowSlice = engine_types.logicalF32RowSlice;

fn topologyModeTag(self: anytype) ?[]const u8 {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasField(SelfType, "topology_mode")) return null;
    return @tagName(self.topology_mode);
}

fn topologyModeIs(self: anytype, comptime expected: []const u8) bool {
    const tag = topologyModeTag(self) orelse return false;
    return std.mem.eql(u8, tag, expected);
}

/// Resolve staged prefill chunk rows for a specific request length.
/// Keeps explicit env override behavior unchanged.
pub fn uploadCpuKvToMirrors(
    gpu_backend: anytype,
    cpu_backend: anytype,
    slot_index: usize,
    pos_start: usize,
    n_positions: usize,
) !void {
    const GpuType = @TypeOf(gpu_backend.*);
    if (comptime !@hasField(GpuType, "block_runtime")) return;
    const BrtType = @TypeOf(gpu_backend.block_runtime);
    if (comptime !@hasField(BrtType, "replicated_kv_sources")) return;
    const replicated = gpu_backend.block_runtime.replicated_kv_sources;
    if (replicated.len == 0) return;

    const mirrors = gpu_backend.block_runtime.mirror_kv;
    const head_dim = gpu_backend.head_dim;
    const allocator = gpu_backend.allocator;

    // Only f16 KV cache is supported for cross-device mirrors.
    if (gpu_backend.kv_cache_dtype != .f16) return error.UnsupportedModel;

    for (replicated, 0..) |src, mi| {
        const mk = &mirrors[mi];
        const cpu_kv = cpu_backend.kv_cache.getLayer(src.global_layer_idx);
        const n_kv_heads = src.kv_dim / head_dim;
        const kv_dim = src.kv_dim;
        const total_elems = n_positions * kv_dim;
        const total_bytes = total_elems * @sizeOf(f16);

        const staging_f16 = try allocator.alloc(f16, total_elems);
        defer allocator.free(staging_f16);

        // Convert + transpose K: CPU [slot][head][pos][dim] → GPU [pos][head*dim]
        for (0..n_positions) |pi| {
            const pos = pos_start + pi;
            const row_off = pi * kv_dim;
            for (0..n_kv_heads) |kh| {
                const cpu_k = cpu_kv.getK(slot_index, kh, pos);
                const dst_off = row_off + kh * head_dim;
                for (0..head_dim) |di| {
                    staging_f16[dst_off + di] = @floatCast(cpu_k[di]);
                }
            }
        }
        const gpu_k_offset = pos_start * kv_dim * @sizeOf(f16);
        const k_slice = try bufferSlice(&mk.k, gpu_k_offset, total_bytes);
        try k_slice.upload(&gpu_backend.device, std.mem.sliceAsBytes(staging_f16));

        // Convert + transpose V.
        for (0..n_positions) |pi| {
            const pos = pos_start + pi;
            const row_off = pi * kv_dim;
            for (0..n_kv_heads) |kh| {
                const cpu_v = cpu_kv.getV(slot_index, kh, pos);
                const dst_off = row_off + kh * head_dim;
                for (0..head_dim) |di| {
                    staging_f16[dst_off + di] = @floatCast(cpu_v[di]);
                }
            }
        }
        const gpu_v_offset = pos_start * kv_dim * @sizeOf(f16);
        const v_slice = try bufferSlice(&mk.v, gpu_v_offset, total_bytes);
        try v_slice.upload(&gpu_backend.device, std.mem.sliceAsBytes(staging_f16));
    }
}

pub fn transferPipelineActivationMultiRow(self: anytype, dst: anytype, total_bytes: usize) !void {
    if (total_bytes == 0) return;
    switch (self.pipeline_transfer_mode) {
        .peer_to_peer => {
            // Prefer event-based non-blocking transfer to avoid host stalls.
            if (self.pipeline_stage0_event) |event| {
                try self.device.recordEvent(event, self.compute_stream);
                try dst.device.streamWaitEvent(dst.compute_stream, event);
                try dst.device.makeCurrent();
                try self.device.memcpyPeerAsync(
                    &dst.device,
                    &dst.runtime_buffers.input_dev,
                    &self.runtime_buffers.input_dev,
                    total_bytes,
                    dst.compute_stream,
                );
            } else {
                try self.device.memcpyPeerAsync(
                    &dst.device,
                    &dst.runtime_buffers.input_dev,
                    &self.runtime_buffers.input_dev,
                    total_bytes,
                    self.compute_stream,
                );
                if (self.compute_stream) |stream| {
                    try self.device.synchronizeStream(stream);
                } else {
                    try self.device.synchronize();
                }
            }
        },
        .host_staged => {
            const staging = self.pipeline_host_staging orelse return error.PipelineTransferNotInitialized;
            if (staging.len == 0) return error.PipelineTransferBufferTooSmall;
            var offset: usize = 0;
            while (offset < total_bytes) {
                const chunk = @min(staging.len, total_bytes - offset);
                var src_slice = try bufferSlice(&self.runtime_buffers.input_dev, offset, chunk);
                try src_slice.download(&self.device, staging[0..chunk]);
                var dst_slice = try bufferSlice(&dst.runtime_buffers.input_dev, offset, chunk);
                try dst_slice.upload(&dst.device, staging[0..chunk]);
                offset += chunk;
            }
        },
        .none => return error.InvalidTopologyConfig,
    }
}

pub fn transferPipelineActivationStage12MultiRow(self: anytype, src: anytype, total_bytes: usize) !void {
    if (total_bytes == 0) return;

    if (self.device.canAccessPeer(&src.device)) {
        // Peer access may already be enabled; the peer copy below is authoritative.
        self.device.enablePeerAccess(&src.device) catch {};
        src.device.enablePeerAccess(&self.device) catch {};
        const peer_copy_started = if (src.device.memcpyPeerAsync(
            &self.device,
            &self.runtime_buffers.input_dev,
            &src.runtime_buffers.input_dev,
            total_bytes,
            src.compute_stream,
        )) true else |_| false;
        if (peer_copy_started) {
            if (src.compute_stream) |stream| {
                try src.device.synchronizeStream(stream);
            } else {
                try src.device.synchronize();
            }
            return;
        }
    }

    const staging = self.pipeline_host_staging_stage12 orelse return error.PipelineTransferNotInitialized;
    if (staging.len == 0) return error.PipelineTransferBufferTooSmall;

    var offset: usize = 0;
    while (offset < total_bytes) {
        const chunk = @min(staging.len, total_bytes - offset);
        var src_slice = try bufferSlice(&src.runtime_buffers.input_dev, offset, chunk);
        try src_slice.download(&src.device, staging[0..chunk]);
        var dst_slice = try bufferSlice(&self.runtime_buffers.input_dev, offset, chunk);
        try dst_slice.upload(&self.device, staging[0..chunk]);
        offset += chunk;
    }
}
