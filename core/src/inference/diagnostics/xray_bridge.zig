//! Bridge-metadata adapters for xray diagnostics.
//!
//! This module keeps staged-boundary xray event construction out of backend
//! private state. Callers pass validated bridge metadata plus a host-readable
//! payload pointer.

const std = @import("std");
const tensor_frame = @import("../bridge/tensor_frame.zig");
const trace = @import("xray_pkg").trace;

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
    if (metadata.role != .activation) return error.InvalidStageBoundary;

    return .{
        .point = .layer_input,
        .layer = std.math.cast(u16, metadata.boundary.consumer_layer_start) orelse return error.InvalidLayerRange,
        .token = 0,
        .position = metadata.sequence_len,
        .backend = try traceBackend(metadata.boundary.target.backend),
        .dtype = traceDType(metadata.boundary.dtype),
        .shape = try traceShape(metadata.shape),
        .ndim = metadata.shape.rank,
        .kernel_name = "bridge_activation_handoff",
        .work = .{ .bytes = metadata.byte_count },
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

fn traceBackend(backend: tensor_frame.StageBackend) BoundaryTraceError!trace.Backend {
    return switch (backend) {
        .cpu => .cpu,
        .cuda => .cuda,
        .metal => .metal,
        .remote => error.UnsupportedTraceBackend,
    };
}

fn traceDType(dtype: tensor_frame.TensorFrameDType) trace.DType {
    return switch (dtype) {
        .bf16 => .bf16,
        .f16 => .f16,
        .f32 => .f32,
    };
}

fn traceShape(shape: tensor_frame.TensorFrameShape) tensor_frame.TensorFrameValidationError![4]u32 {
    try shape.validate();
    var result = [_]u32{0} ** 4;
    for (0..shape.rank) |idx| {
        result[idx] = std.math.cast(u32, shape.dims[idx]) orelse return error.InvalidTensorShape;
    }
    return result;
}

fn testActivationMetadata(target_backend: tensor_frame.StageBackend) !tensor_frame.TensorFrameMetadata {
    return tensor_frame.activationHandoffFrame(.{
        .frame_id = 1,
        .graph_id = 2,
        .request_id = 3,
        .source = .{ .stage_id = 0, .backend = .cpu },
        .target = .{ .stage_id = 1, .backend = target_backend },
        .producer_layer_start = 0,
        .producer_layer_end = 4,
        .consumer_layer_start = 4,
        .consumer_layer_end = 12,
        .dtype = .f32,
        .shape = try tensor_frame.TensorFrameShape.contiguous(3, .{ 1, 1, 16, 0 }),
        .device = .{ .cuda = 0 },
        .sequence_start = 9,
        .sequence_len = 1,
        .batch_size = 1,
        .slot_index = 7,
    });
}

test "activationHandoffLayerInputTrace maps bridge metadata to xray layer input" {
    const metadata = try testActivationMetadata(.cuda);
    const emission = try activationHandoffLayerInputTrace(&metadata);

    try std.testing.expectEqual(trace.TracePoint.layer_input, emission.point);
    try std.testing.expectEqual(@as(u16, 4), emission.layer);
    try std.testing.expectEqual(@as(u32, 0), emission.token);
    try std.testing.expectEqual(@as(u32, 1), emission.position);
    try std.testing.expectEqual(trace.Backend.cuda, emission.backend);
    try std.testing.expectEqual(trace.DType.f32, emission.dtype);
    try std.testing.expectEqual(@as(u8, 3), emission.ndim);
    try std.testing.expectEqual([4]u32{ 1, 1, 16, 0 }, emission.shape);
    try std.testing.expectEqual(@as(u64, 64), emission.work.bytes);
    try std.testing.expectEqualStrings("bridge_activation_handoff", emission.kernel_name);
}

test "activationHandoffLayerInputTrace rejects remote xray backend" {
    const metadata = try testActivationMetadata(.remote);
    try std.testing.expectError(error.UnsupportedTraceBackend, activationHandoffLayerInputTrace(&metadata));
}

test "emitActivationHandoffLayerInput no-ops when layer input trace point is inactive" {
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

    const metadata = try testActivationMetadata(.cuda);
    var payload = [_]u8{0} ** 64;
    emitActivationHandoffLayerInput(&metadata, payload[0..].ptr);
    try std.testing.expectEqual(@as(usize, 0), Capture.count);
}
