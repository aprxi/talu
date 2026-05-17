//! Tests for CPU transport endpoint descriptors.

const std = @import("std");
const main = @import("main");

const endpoint = main.inference.backend.cpu.interface.transport_endpoint;

test "CPU transport endpoint exposes only CPU-owned external activation surfaces" {
    const MockCpuBackend = struct {
        slot_bytes: [8]u8 = .{ 1, 2, 3, 4, 5, 6, 7, 8 },
        prefill_bytes: [6]u8 = .{ 9, 10, 11, 12, 13, 14 },

        pub fn slotActivationBytes(self: *@This(), slot_index: usize) []const u8 {
            _ = slot_index;
            return self.slot_bytes[0..];
        }

        pub fn slotActivationBytesMut(self: *@This(), slot_index: usize) []u8 {
            _ = slot_index;
            return self.slot_bytes[0..];
        }

        pub fn localPrefillActivationBytes(self: *@This(), byte_count: usize) []const u8 {
            _ = byte_count;
            return self.prefill_bytes[0..];
        }

        pub fn localPrefillActivationBytesMut(self: *@This(), byte_count: usize) []u8 {
            _ = byte_count;
            return self.prefill_bytes[0..];
        }
    };

    var backend = MockCpuBackend{};
    const hint = try endpoint.deviceLocationHint(&backend);
    try std.testing.expectEqual(main.inference.pipeline.TensorFramePayloadLocationHint{ .cpu = {} }, hint);

    const decode_source = try endpoint.decodeExternalOutput(&backend, 0, 4);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, decode_source.bytes);
    const prefill_source = try endpoint.prefillExternalOutput(&backend, 3);
    try std.testing.expectEqualSlices(u8, &.{ 9, 10, 11 }, prefill_source.bytes);

    const decode_target = try endpoint.decodeExternalInput(&backend, 0, 2);
    decode_target.bytes[0] = 42;
    try std.testing.expectEqual(@as(u8, 42), backend.slot_bytes[0]);

    const prefill_target = try endpoint.prefillExternalInput(&backend, 2);
    prefill_target.bytes[1] = 77;
    try std.testing.expectEqual(@as(u8, 77), backend.prefill_bytes[1]);
}
