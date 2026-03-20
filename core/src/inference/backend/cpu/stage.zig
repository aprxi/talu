//! CPU pipeline stage wrapper implementing the backend-agnostic stage contract.
//!
//! CpuStage wraps a FusedCpuBackend pointer and provides the executeLayers /
//! downloadActivation / uploadActivation / synchronize surface required by
//! PipelineRuntime. The stage stores the active slot index from executeLayers
//! so transfer methods can address per-slot activations deterministically.

const std = @import("std");
const pipeline = @import("../pipeline.zig");
const FusedCpuBackend = @import("engine.zig").FusedCpuBackend;

pub const CpuStage = struct {
    backend: *FusedCpuBackend,
    active_slot_index: usize = 0,

    pub const layer_execution_input_magic: u32 = 0x32505543; // "CPU2" little-endian marker
    pub const supported_boundary_dtypes = [_]pipeline.BoundaryDType{
        .f32,
    };

    /// Internal stage execution payload.
    /// This is process-local and build-local only; it is not a stable wire ABI.
    pub const LayerExecutionInput = extern struct {
        abi_magic: u32 = layer_execution_input_magic,
        abi_size: u32 = @sizeOf(LayerExecutionInput),
        token: u32,
        position: usize,
        slot_index: usize,
        logits_out_ptr: ?[*]f32 = null,
        logits_out_len: usize = 0,
        compute_logits: bool = false,
        download_logits: bool = false,
        ensure_kv_capacity: bool = true,
        use_preloaded_input: bool = false,
    };

    pub fn executeLayers(self: *CpuStage, input: []const u8, layer_start: usize, layer_end: usize) !void {
        if (input.len != @sizeOf(LayerExecutionInput)) return error.InvalidArgument;
        var exec_input: LayerExecutionInput = undefined;
        @memcpy(std.mem.asBytes(&exec_input), input);
        if (exec_input.abi_magic != layer_execution_input_magic or exec_input.abi_size != @sizeOf(LayerExecutionInput)) {
            return error.InvalidArgument;
        }
        if ((exec_input.logits_out_ptr == null) != (exec_input.logits_out_len == 0)) return error.InvalidArgument;
        if (layer_end < layer_start) return error.InvalidArgument;

        self.active_slot_index = exec_input.slot_index;
        const logits_out_opt: ?[]f32 = if (exec_input.logits_out_ptr) |ptr|
            ptr[0..exec_input.logits_out_len]
        else
            null;
        try self.backend.computePrototypeLogitsWithLayerRange(
            exec_input.token,
            exec_input.position,
            exec_input.slot_index,
            logits_out_opt,
            layer_start,
            layer_end,
            exec_input.compute_logits,
            exec_input.download_logits,
            exec_input.ensure_kv_capacity,
            exec_input.use_preloaded_input,
        );
    }

    pub fn downloadActivation(self: *CpuStage, host_buf: []u8, byte_count: usize) !void {
        const source = self.backend.slotActivationBytes(self.active_slot_index);
        if (byte_count > source.len or byte_count > host_buf.len) return error.InvalidArgument;
        @memcpy(host_buf[0..byte_count], source[0..byte_count]);
    }

    pub fn uploadActivation(self: *CpuStage, host_buf: []const u8, byte_count: usize) !void {
        const dst = self.backend.slotActivationBytesMut(self.active_slot_index);
        if (byte_count > dst.len or byte_count > host_buf.len) return error.InvalidArgument;
        @memcpy(dst[0..byte_count], host_buf[0..byte_count]);
    }

    pub fn synchronize(_: *CpuStage) !void {}

    pub fn deinit(_: *CpuStage, _: std.mem.Allocator) void {}
};

/// Host-staged CPU pipeline alias for symmetric CPU stage testing.
pub const CpuPipeline = pipeline.PipelineRuntime(CpuStage, CpuStage, null);

test "CpuStage.executeLayers rejects payload with incorrect byte length" {
    var stage = CpuStage{
        .backend = @ptrFromInt(64),
    };
    const too_small = [_]u8{0} ** (@sizeOf(CpuStage.LayerExecutionInput) - 1);
    try std.testing.expectError(error.InvalidArgument, stage.executeLayers(too_small[0..], 0, 0));
}

test "CpuStage.executeLayers rejects payload with invalid ABI marker" {
    var stage = CpuStage{
        .backend = @ptrFromInt(64),
    };
    var payload = CpuStage.LayerExecutionInput{
        .token = 0,
        .position = 0,
        .slot_index = 0,
    };
    payload.abi_magic = 0;
    try std.testing.expectError(error.InvalidArgument, stage.executeLayers(std.mem.asBytes(&payload), 0, 0));
}

test "CpuStage.executeLayers rejects payload with logits pointer length mismatch" {
    var stage = CpuStage{
        .backend = @ptrFromInt(64),
    };
    var payload = CpuStage.LayerExecutionInput{
        .token = 0,
        .position = 0,
        .slot_index = 0,
    };
    payload.logits_out_len = 3;
    try std.testing.expectError(error.InvalidArgument, stage.executeLayers(std.mem.asBytes(&payload), 0, 0));
}
