//! Tests for CUDA transport endpoint descriptors.

const std = @import("std");
const main = @import("main");

const endpoint = main.inference.backend.cuda.interface.transport_endpoint;

test "CUDA transport endpoint exposes only CUDA-owned external activation surfaces" {
    const MockDevice = struct {
        pub fn ordinal(_: *const @This()) usize {
            return 7;
        }

        pub fn synchronize(_: *@This()) !void {}

        pub fn synchronizeStream(_: *@This(), _: *anyopaque) !void {}

        pub fn canAccessPeer(_: *@This(), _: *@This()) bool {
            return true;
        }

        pub fn enablePeerAccess(_: *@This(), _: *@This()) !void {}

        pub fn memcpyPeerAsync(_: *@This(), _: *@This(), _: anytype, _: anytype, _: usize, _: ?*anyopaque) !void {}
    };
    const MockBuffer = struct {
        size: usize = 256,
        pointer: u64 = 0,

        pub fn download(_: *@This(), _: anytype, _: []u8) !void {}

        pub fn upload(_: *@This(), _: anytype, _: []const u8) !void {}
    };
    const MockBackend = struct {
        const RuntimeBuffers = struct {
            input_dev: MockBuffer = .{},
        };

        device: MockDevice = .{},
        runtime_buffers: RuntimeBuffers = .{},
        compute_stream: ?*anyopaque = @ptrFromInt(0x1000),
    };

    var backend = MockBackend{};
    const hint = try endpoint.deviceLocationHint(&backend);
    try std.testing.expectEqual(main.inference.pipeline.TensorFramePayloadLocationHint{ .cuda = 7 }, hint);

    const decode_output = try endpoint.decodeExternalOutput(&backend, 3, 64);
    try std.testing.expectEqual(@as(*anyopaque, @ptrCast(&backend.runtime_buffers.input_dev)), decode_output.buffer);
    try std.testing.expectEqual(@as(*anyopaque, @ptrCast(&backend.device)), decode_output.device);
    try std.testing.expectEqual(@as(u16, 7), decode_output.device_ordinal);
    try std.testing.expectEqual(@as(?*anyopaque, @ptrFromInt(0x1000)), decode_output.stream);
    try std.testing.expectEqual(@as(usize, 3), decode_output.slot_index);
    try std.testing.expectEqual(@as(usize, 64), decode_output.byte_count);

    const decode_input = try endpoint.decodeExternalInput(&backend, 4, 96);
    try std.testing.expectEqual(@as(*anyopaque, @ptrCast(&backend.runtime_buffers.input_dev)), decode_input.buffer);
    try std.testing.expectEqual(@as(usize, 4), decode_input.slot_index);
    try std.testing.expectEqual(@as(usize, 96), decode_input.byte_count);

    const prefill_output = try endpoint.prefillExternalOutput(&backend, 128);
    try std.testing.expectEqual(@as(*anyopaque, @ptrCast(&backend.runtime_buffers.input_dev)), prefill_output.buffer);
    try std.testing.expectEqual(@as(usize, 128), prefill_output.byte_count);

    const prefill_input = try endpoint.prefillExternalInput(&backend, 160);
    try std.testing.expectEqual(@as(*anyopaque, @ptrCast(&backend.runtime_buffers.input_dev)), prefill_input.buffer);
    try std.testing.expectEqual(@as(usize, 160), prefill_input.byte_count);

    try std.testing.expectError(error.UnsupportedContentType, endpoint.sideExternalInput(&backend, 4));
}
