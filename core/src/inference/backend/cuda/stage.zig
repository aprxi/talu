//! CUDA pipeline stage wrapper implementing the backend-agnostic stage contract.
//!
//! CudaStage wraps a CudaBackend pointer and provides the executeLayers /
//! downloadActivation / uploadActivation / synchronize interface required
//! by PipelineRuntime. The backend itself owns all device-bound state;
//! the stage provides the contract surface for pipeline orchestration.
//!
//! CudaP2PTransfer provides an optimized transfer path using CUDA peer-to-peer
//! copies when hardware supports it, falling back to host-staged transfer otherwise.

const std = @import("std");
const compute = @import("../../../compute/root.zig");
const pipeline = @import("../pipeline.zig");

/// Wrapper around a CudaBackend pointer that satisfies the pipeline stage contract.
/// Each stage corresponds to one CudaBackend instance operating on a disjoint
/// range of decoder layers on a specific CUDA device.
pub const CudaStage = struct {
    /// The underlying backend instance for this stage.
    /// Each stage owns its own backend with its own device and layer range.
    backend: *@import("root.zig").BackendType,

    pub const layer_execution_input_magic: u32 = 0x32475550; // "PUG2" little-endian marker

    /// Internal stage execution payload.
    /// This is process-local and build-local only; it is not a stable wire ABI.
    pub const LayerExecutionInput = extern struct {
        abi_magic: u32 = layer_execution_input_magic,
        abi_size: u32 = @sizeOf(LayerExecutionInput),
        token: u32,
        position: usize,
        slot_index: usize,
        trace_seq_len_u32: u32,
        trace_pos_offset: usize,
        logits_out_ptr: ?[*]f32 = null,
        logits_out_len: usize = 0,
        compute_logits: bool = false,
        download_logits: bool = false,
        ensure_kv_capacity: bool = true,
        use_preloaded_input: bool = false,
    };

    /// Execute decoder layers [layer_start, layer_end) through this stage's backend.
    /// The input pointer carries a LayerExecutionInput payload for stage execution.
    pub fn executeLayers(self: *CudaStage, input: []const u8, layer_start: usize, layer_end: usize) !void {
        if (input.len != @sizeOf(LayerExecutionInput)) return error.InvalidArgument;
        var exec_input: LayerExecutionInput = undefined;
        @memcpy(std.mem.asBytes(&exec_input), input);
        if (exec_input.abi_magic != layer_execution_input_magic or exec_input.abi_size != @sizeOf(LayerExecutionInput)) {
            return error.InvalidArgument;
        }
        if ((exec_input.logits_out_ptr == null) != (exec_input.logits_out_len == 0)) return error.InvalidArgument;
        if (layer_end < layer_start) return error.InvalidArgument;
        const local_layer_limit = layer_end - layer_start;
        if (local_layer_limit > self.backend.block_runtime.blocks.len) return error.InvalidArgument;
        const logits_out_opt: ?[]f32 = if (exec_input.logits_out_ptr) |ptr|
            ptr[0..exec_input.logits_out_len]
        else
            null;
        try self.backend.computeGpuPrototypeLogitsWithLayerLimit(
            exec_input.token,
            exec_input.position,
            exec_input.slot_index,
            logits_out_opt,
            local_layer_limit,
            exec_input.compute_logits,
            exec_input.download_logits,
            exec_input.ensure_kv_capacity,
            exec_input.trace_seq_len_u32,
            exec_input.trace_pos_offset,
            null,
            null,
            null,
            exec_input.use_preloaded_input,
        );
    }

    /// Download this stage's output activation from device to host.
    pub fn downloadActivation(self: *CudaStage, host_buf: []u8, byte_count: usize) !void {
        _ = byte_count;
        const backend = self.backend;
        try backend.runtime_buffers.input_dev.download(&backend.device, host_buf);
    }

    /// Upload activation from host to this stage's input buffer on device.
    pub fn uploadActivation(self: *CudaStage, host_buf: []const u8, byte_count: usize) !void {
        _ = byte_count;
        const backend = self.backend;
        try backend.runtime_buffers.input_dev.upload(&backend.device, host_buf);
    }

    /// Synchronize this stage's compute stream.
    pub fn synchronize(self: *CudaStage) !void {
        const backend = self.backend;
        if (backend.compute_stream) |stream| {
            try backend.device.synchronizeStream(stream);
        }
    }

    pub fn deinit(_: *CudaStage, _: std.mem.Allocator) void {
        // The underlying backend is owned by the caller; we don't free it here.
    }
};

/// CUDA-specific activation transfer using P2P copies when available,
/// falling back to host-staged copies otherwise.
pub const CudaP2PTransfer = struct {
    mode: TransferMode,
    /// Pinned host staging buffer (allocated only for host_staged mode).
    host_staging: ?[]align(4096) u8,

    pub const TransferMode = enum {
        peer_to_peer,
        host_staged,
    };

    /// Probe P2P capability and initialize the transfer mechanism.
    pub fn init(
        allocator: std.mem.Allocator,
        dev0: *compute.cuda.Device,
        dev1: *compute.cuda.Device,
        max_transfer_bytes: usize,
    ) !CudaP2PTransfer {
        if (dev0.canAccessPeer(dev1)) {
            try dev0.enablePeerAccess(dev1);
            try dev1.enablePeerAccess(dev0);
            return .{ .mode = .peer_to_peer, .host_staging = null };
        }
        // Host-staged fallback: allocate a pinned buffer for activation transfer.
        const staging = try allocator.alignedAlloc(u8, .fromByteUnits(4096), max_transfer_bytes);
        return .{ .mode = .host_staged, .host_staging = staging };
    }

    /// Transfer activation from stage0's output buffer to stage1's input buffer.
    pub fn transfer(self: *CudaP2PTransfer, src: *CudaStage, dst: *CudaStage, byte_count: usize) !void {
        const src_backend = src.backend;
        const dst_backend = dst.backend;
        switch (self.mode) {
            .peer_to_peer => {
                try src_backend.device.memcpyPeerAsync(
                    dst_backend.runtime_buffers.input_dev.pointer,
                    dst_backend.device.context,
                    src_backend.runtime_buffers.input_dev.pointer,
                    src_backend.device.context,
                    byte_count,
                    src_backend.compute_stream,
                );
                // Synchronize source stream so destination can safely read.
                if (src_backend.compute_stream) |stream| {
                    try src_backend.device.synchronizeStream(stream);
                }
            },
            .host_staged => {
                const staging = self.host_staging orelse return error.PipelineTransferNotInitialized;
                if (byte_count > staging.len) return error.PipelineTransferBufferTooSmall;
                try src.downloadActivation(staging[0..byte_count], byte_count);
                try dst.uploadActivation(staging[0..byte_count], byte_count);
            },
        }
    }

    pub fn deinit(self: *CudaP2PTransfer, allocator: std.mem.Allocator) void {
        if (self.host_staging) |buf| {
            allocator.free(buf);
            self.host_staging = null;
        }
    }
};

/// CUDA pipeline type: PipelineRuntime specialized for CudaStage + CudaP2PTransfer.
pub const CudaPipeline = pipeline.PipelineRuntime(CudaStage, CudaP2PTransfer);

test "CudaStage.executeLayers rejects payload with incorrect byte length" {
    var stage = CudaStage{
        .backend = @ptrFromInt(64),
    };
    const too_small = [_]u8{0} ** (@sizeOf(CudaStage.LayerExecutionInput) - 1);
    try std.testing.expectError(error.InvalidArgument, stage.executeLayers(too_small[0..], 0, 0));
}

test "CudaStage.executeLayers rejects payload with invalid ABI marker" {
    var stage = CudaStage{
        .backend = @ptrFromInt(64),
    };
    var payload = CudaStage.LayerExecutionInput{};
    payload.abi_magic = 0;
    const bytes = std.mem.asBytes(&payload);
    try std.testing.expectError(error.InvalidArgument, stage.executeLayers(bytes, 0, 0));
}

test "CudaStage.executeLayers rejects payload with logits pointer length mismatch" {
    var stage = CudaStage{
        .backend = @ptrFromInt(64),
    };
    var payload = CudaStage.LayerExecutionInput{};
    payload.logits_out_len = 4;
    const bytes = std.mem.asBytes(&payload);
    try std.testing.expectError(error.InvalidArgument, stage.executeLayers(bytes, 0, 0));
}
