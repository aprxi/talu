//! Tests for CUDA transport endpoint descriptors.

const std = @import("std");
const main = @import("main");

const endpoint = main.inference.backend.cuda.interface.transport_endpoint;

test "CUDA transport endpoint exposes descriptors without performing copies" {
    const MockDevice = struct {
        pub fn ordinal(_: *@This()) usize {
            return 7;
        }
    };
    const MockBackend = struct {
        const RuntimeBuffers = struct {
            input_dev: u8 = 0,
        };

        device: MockDevice = .{},
        runtime_buffers: RuntimeBuffers = .{},
        compute_stream: ?*anyopaque = @ptrFromInt(0x1000),
    };

    var backend = MockBackend{};
    const hint = try endpoint.deviceLocationHint(&backend);
    try std.testing.expectEqual(main.inference.bridge.TensorFramePayloadLocationHint{ .cuda = 7 }, hint);

    const decode_target = try endpoint.decodeInputBuffer(&backend, 3, 64);
    try std.testing.expectEqual(@as(*anyopaque, @ptrCast(&backend.runtime_buffers.input_dev)), decode_target.buffer);
    try std.testing.expectEqual(@as(*anyopaque, @ptrCast(&backend.device)), decode_target.device);
    try std.testing.expectEqual(@as(u16, 7), decode_target.device_ordinal);
    try std.testing.expectEqual(@as(?*anyopaque, @ptrFromInt(0x1000)), decode_target.stream);
    try std.testing.expectEqual(@as(usize, 64), decode_target.byte_count);

    const prefill_target = try endpoint.prefillInputBuffer(&backend, 128);
    try std.testing.expectEqual(@as(*anyopaque, @ptrCast(&backend.runtime_buffers.input_dev)), prefill_target.buffer);
    try std.testing.expectEqual(@as(usize, 128), prefill_target.byte_count);

    try std.testing.expectError(error.InvalidTopologyConfig, endpoint.hostDecodeActivationSlice(&backend, 0, 4));
    try std.testing.expectError(error.InvalidTopologyConfig, endpoint.hostPrefillActivationSlice(&backend, 4));
    try std.testing.expectError(error.UnsupportedContentType, endpoint.sideInputBuffer(&backend, 4));
}
