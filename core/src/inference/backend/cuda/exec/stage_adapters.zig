//! Shared adapter helpers for CUDA staged decode routes.
//!
//! The route files keep topology-specific ordering. This module owns common
//! stage-method validation and activation movement used by PipelineRuntime
//! adapters.

const std = @import("std");

pub const DecodeContext = struct {
    token: u32,
    position: usize,
    slot_index: usize,
    logits_out_opt: ?[]f32,
    compute_logits: bool,
    download_logits: bool,
    ensure_kv_capacity: bool,
    trace_seq_len_u32: u32,
    trace_pos_offset: usize,
};

pub fn validateEmptyInput(input: []const u8) !void {
    if (input.len != 0) return error.InvalidArgument;
}

pub fn decodeLayerLimit(layer_start: usize, layer_end: usize) !usize {
    if (layer_end < layer_start) return error.InvalidArgument;
    return layer_end - layer_start;
}

pub fn executeCpuDecodeLayerRange(
    backend: anytype,
    ctx: *const DecodeContext,
    layer_start: usize,
    layer_end: usize,
    use_preloaded_input: bool,
) !void {
    if (comptime !hasDecl(@TypeOf(backend.*), "executeDecodeLayerRange")) {
        return error.InvalidTopologyConfig;
    }
    try backend.executeDecodeLayerRange(
        ctx.token,
        ctx.position,
        ctx.slot_index,
        null,
        layer_start,
        layer_end,
        false,
        false,
        ctx.ensure_kv_capacity,
        use_preloaded_input,
    );
}

pub fn executeCudaDecodeLayerRange(
    comptime execute_decode_with_layer_limit: anytype,
    backend: anytype,
    ctx: *const DecodeContext,
    layer_start: usize,
    layer_end: usize,
    logits_out_opt: ?[]f32,
    compute_logits: bool,
    download_logits: bool,
    use_preloaded_input: bool,
) !void {
    const local_layer_limit = try decodeLayerLimit(layer_start, layer_end);
    try execute_decode_with_layer_limit(
        backend,
        ctx.token,
        ctx.position,
        ctx.slot_index,
        logits_out_opt,
        local_layer_limit,
        compute_logits,
        download_logits,
        ctx.ensure_kv_capacity,
        ctx.trace_seq_len_u32,
        ctx.trace_pos_offset,
        null,
        null,
        null,
        use_preloaded_input,
    );
}

pub fn downloadCpuActivation(backend: anytype, slot_index: usize, host_buf: []u8, byte_count: usize) !void {
    if (byte_count > host_buf.len) return error.InvalidArgument;
    const source = backend.slotActivationBytes(slot_index);
    if (byte_count > source.len) return error.InvalidArgument;
    @memcpy(host_buf[0..byte_count], source[0..byte_count]);
}

pub fn uploadCpuActivation(backend: anytype, slot_index: usize, host_buf: []const u8, byte_count: usize) !void {
    if (byte_count > host_buf.len) return error.InvalidArgument;
    const dst = backend.slotActivationBytesMut(slot_index);
    if (byte_count > dst.len) return error.InvalidArgument;
    @memcpy(dst[0..byte_count], host_buf[0..byte_count]);
}

pub fn downloadCudaActivation(backend: anytype, host_buf: []u8, byte_count: usize) !void {
    if (byte_count > host_buf.len) return error.InvalidArgument;
    const BackendType = @TypeOf(backend.*);
    if (comptime @hasField(BackendType, "runtime_buffers") and @hasField(BackendType, "device")) {
        return backend.runtime_buffers.input_dev.download(&backend.device, host_buf[0..byte_count]);
    }
    return error.InvalidTopologyConfig;
}

pub fn uploadCudaActivation(backend: anytype, slot_index: usize, host_buf: []const u8, byte_count: usize) !void {
    if (byte_count > host_buf.len) return error.InvalidArgument;
    const BackendType = @TypeOf(backend.*);
    if (comptime @hasDecl(BackendType, "uploadPipelineActivationFromHost")) {
        return backend.uploadPipelineActivationFromHost(slot_index, host_buf[0..byte_count], byte_count);
    }
    if (comptime @hasField(BackendType, "runtime_buffers") and @hasField(BackendType, "device")) {
        return backend.runtime_buffers.input_dev.upload(&backend.device, host_buf[0..byte_count]);
    }
    return error.InvalidTopologyConfig;
}

pub fn synchronizeCudaBackend(backend: anytype) !void {
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasField(BackendType, "device")) return error.InvalidTopologyConfig;
    if (comptime @hasField(BackendType, "compute_stream")) {
        if (backend.compute_stream) |stream| {
            try backend.device.synchronizeStream(stream);
            return;
        }
    }
    try backend.device.synchronize();
}

fn hasDecl(comptime T: type, comptime name: []const u8) bool {
    return switch (@typeInfo(T)) {
        .@"struct", .@"enum", .@"union", .@"opaque" => @hasDecl(T, name),
        else => false,
    };
}

test "decodeLayerLimit rejects inverted ranges" {
    try std.testing.expectError(error.InvalidArgument, decodeLayerLimit(4, 3));
}

test "validateEmptyInput rejects route payloads" {
    try std.testing.expectError(error.InvalidArgument, validateEmptyInput(&.{1}));
}
