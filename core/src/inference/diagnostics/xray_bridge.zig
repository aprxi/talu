//! Bridge-metadata adapters for xray diagnostics.
//!
//! This module keeps staged-boundary xray event construction out of backend
//! private state. Callers pass validated bridge metadata plus a host-readable
//! payload pointer.

const std = @import("std");
const tensor_frame = @import("../bridge/tensor_frame.zig");
const trace = @import("xray_pkg").trace;

const test_activation_batch = [_]tensor_frame.TensorFrameBatchEntry{.{
    .batch_index = 0,
    .request_id = 3,
    .slot_id = 7,
    .sequence_start = 9,
    .token_count = 1,
}};

pub const BoundaryTraceError = tensor_frame.TensorFrameValidationError || error{
    UnsupportedTraceBackend,
};

pub const BoundaryActivationTrace = struct {
    point: trace.TracePoint,
    layer: u16,
    token: u32,
    position: u32,
    backend: trace.Backend,
    dtype: trace.DType,
    shape: [4]u32,
    ndim: u8,
    kernel_name: []const u8,
    work: trace.Work,
};

pub fn activationHandoffLayerInputTrace(
    metadata: *const tensor_frame.TensorFrameMetadata,
) BoundaryTraceError!BoundaryActivationTrace {
    try metadata.validate();
    if (metadata.role != .activation) return error.InvalidActivationRole;
    if (metadata.payload.location_hint == null) return error.UnsupportedTraceBackend;
    if (metadata.batch.entries.len != 1) return error.InvalidBatch;
    const entry = metadata.batch.entries[0];

    return .{
        .point = .layer_input,
        .layer = std.math.cast(u16, metadata.boundary.consumer_layer_start) orelse return error.InvalidConsumerLayerRange,
        .token = 0,
        .position = std.math.cast(u32, entry.sequence_start) orelse return error.InvalidSequenceRange,
        .backend = try traceBackend(metadata.payload.location_hint.?),
        .dtype = traceDType(metadata.tensor.dtype),
        .shape = try traceShape(metadata.tensor),
        .ndim = metadata.tensor.rank,
        .kernel_name = "bridge_activation_handoff",
        .work = .{ .bytes = metadata.payload.byte_count },
    };
}

pub fn emitActivationHandoffLayerInput(
    metadata: *const tensor_frame.TensorFrameMetadata,
    ptr: [*]const u8,
) void {
    if (!trace.shouldEmit(.layer_input)) return;
    const emission = activationHandoffLayerInputTrace(metadata) catch return;
    const previous_backend = trace.setBackendContext(emission.backend);
    defer _ = trace.setBackendContext(previous_backend);
    trace.emitWithWork(
        emission.point,
        emission.layer,
        emission.token,
        emission.position,
        ptr,
        emission.dtype,
        emission.shape,
        emission.ndim,
        emission.kernel_name,
        emission.work,
    );
}

fn traceBackend(location_hint: tensor_frame.TensorFramePayloadLocationHint) BoundaryTraceError!trace.Backend {
    return switch (location_hint) {
        .cpu => .cpu,
        .cuda => .cuda,
        .metal => .metal,
        .opaque_local => error.UnsupportedTraceBackend,
    };
}

fn traceDType(dtype: tensor_frame.TensorFrameDType) trace.DType {
    return switch (dtype) {
        .bf16 => .bf16,
        .f16 => .f16,
        .f32 => .f32,
    };
}

fn traceShape(tensor: tensor_frame.TensorFrameTensorDesc) tensor_frame.TensorFrameValidationError![4]u32 {
    var result = [_]u32{0} ** 4;
    for (0..tensor.rank) |idx| {
        result[idx] = std.math.cast(u32, tensor.shape[idx]) orelse return error.InvalidTensorShape;
    }
    return result;
}

fn testActivationMetadata(location_hint: ?tensor_frame.TensorFramePayloadLocationHint) !tensor_frame.TensorFrameMetadata {
    const boundary = tensor_frame.TensorFrameBoundaryRef{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .producer_layer_start = 0,
        .producer_layer_end = 4,
        .consumer_layer_start = 4,
        .consumer_layer_end = 12,
    };
    const tensor = try tensor_frame.TensorFrameTensorDesc.contiguousActivation(.f32, .{ 1, 1, 16, 0 });
    return .{
        .frame_id = try tensor_frame.TensorFrameInstanceId.init(1),
        .plan = .{
            .graph_digest = [_]u8{1} ** 32,
            .graph_contract_version = 1,
            .stage_plan_contract_version = 1,
            .stage_plan_id = .{ .digest = [_]u8{2} ** 32 },
        },
        .boundary = boundary,
        .selected_contract = .{
            .boundary = boundary,
            .dtype = .f32,
            .layout = .row_major,
            .source = .explicit,
        },
        .role = .activation,
        .step_kind = .decode,
        .shape_context = .{ .expected_hidden_size = 16 },
        .tensor = tensor,
        .batch = .{ .entries = &test_activation_batch },
        .payload = .{
            .byte_count = tensor.payload_byte_count,
            .location_hint = location_hint,
            .ownership = .borrowed_until_next_stage_call,
            .lifetime = .step_scoped,
        },
    };
}

test "inference diagnostics activationHandoffLayerInputTrace maps bridge metadata to xray layer input" {
    const metadata = try testActivationMetadata(.{ .cuda = 0 });
    const emission = try activationHandoffLayerInputTrace(&metadata);

    try std.testing.expectEqual(trace.TracePoint.layer_input, emission.point);
    try std.testing.expectEqual(@as(u16, 4), emission.layer);
    try std.testing.expectEqual(@as(u32, 0), emission.token);
    try std.testing.expectEqual(@as(u32, 9), emission.position);
    try std.testing.expectEqual(trace.Backend.cuda, emission.backend);
    try std.testing.expectEqual(trace.DType.f32, emission.dtype);
    try std.testing.expectEqual(@as(u8, 3), emission.ndim);
    try std.testing.expectEqual([4]u32{ 1, 1, 16, 0 }, emission.shape);
    try std.testing.expectEqual(@as(u64, 64), emission.work.bytes);
    try std.testing.expectEqualStrings("bridge_activation_handoff", emission.kernel_name);
}

test "inference diagnostics activationHandoffLayerInputTrace rejects multi-entry frame" {
    const multi_batch = [_]tensor_frame.TensorFrameBatchEntry{
        .{
            .batch_index = 0,
            .request_id = 3,
            .slot_id = 7,
            .sequence_start = 9,
            .token_count = 1,
        },
        .{
            .batch_index = 1,
            .request_id = 4,
            .slot_id = 8,
            .sequence_start = 12,
            .token_count = 1,
        },
    };
    var metadata = try testActivationMetadata(.{ .cuda = 0 });
    metadata.tensor = try tensor_frame.TensorFrameTensorDesc.contiguousActivation(.f32, .{ 2, 1, 16, 0 });
    metadata.batch = .{ .entries = &multi_batch };
    metadata.payload.byte_count = metadata.tensor.payload_byte_count;

    try std.testing.expectError(error.InvalidBatch, activationHandoffLayerInputTrace(&metadata));
}

test "inference diagnostics activationHandoffLayerInputTrace rejects unsupported xray backend" {
    const metadata = try testActivationMetadata(.{ .opaque_local = 42 });
    try std.testing.expectError(error.UnsupportedTraceBackend, activationHandoffLayerInputTrace(&metadata));
}

test "inference diagnostics emitActivationHandoffLayerInput no-ops when layer input trace point is inactive" {
    const Capture = struct {
        var count: usize = 0;

        fn handler(_: trace.TraceEmission) void {
            count += 1;
        }

        fn reset() void {
            count = 0;
        }
    };

    Capture.reset();
    trace.setHandler(&Capture.handler);
    trace.setActiveBuiltInPointMask(0);
    trace.setActiveExactEmissionFilter(null);
    defer {
        trace.setActiveBuiltInPointMask(0);
        trace.setActiveExactEmissionFilter(null);
        trace.setHandler(null);
    }

    const metadata = try testActivationMetadata(.{ .cuda = 0 });
    var payload = [_]u8{0} ** 64;
    emitActivationHandoffLayerInput(&metadata, payload[0..].ptr);
    try std.testing.expectEqual(@as(usize, 0), Capture.count);
}
