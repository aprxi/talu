//! Metal backend integration tests.
//!
//! Tests Metal GPU availability, device creation, buffer operations,
//! and matrix multiplication. Skipped on non-macOS platforms.

const std = @import("std");
const metal = @import("root.zig");

test "metal availability" {
    if (!metal.isAvailable()) {
        std.debug.print("Metal is not available on this system\n", .{});
        return error.SkipZigTest;
    }

}

test "metal device creation" {
    if (!metal.isAvailable()) {
        return error.SkipZigTest;
    }

    var metal_device = try metal.Device.init();
    defer metal_device.deinit();

    const device_name = metal_device.name();
    try std.testing.expect(device_name.len > 0);
}

test "metal buffer allocation" {
    if (!metal.isAvailable()) {
        return error.SkipZigTest;
    }

    var metal_device = try metal.Device.init();
    defer metal_device.deinit();

    // Allocate a 1KB buffer
    var metal_buffer = try metal_device.allocBuffer(1024);
    defer metal_buffer.deinit();

    // Test upload/download
    const test_data = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    metal_buffer.upload(&test_data);

    var result = [_]u8{0} ** 8;
    metal_buffer.download(&result);

    try std.testing.expectEqualSlices(u8, &test_data, &result);
}

test "metal f32 matmul" {
    if (!metal.isAvailable()) {
        return error.SkipZigTest;
    }

    var metal_device = try metal.Device.init();
    defer metal_device.deinit();

    // Test small matmul: [2x3] @ [3x2] = [2x2]
    const left_matrix = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    const right_matrix = [_]f32{
        1, 0,
        0, 1,
        1, 1,
    };
    var output_matrix = [_]f32{0} ** 4;

    try metal.matmul.matmulF32(
        &metal_device,
        &left_matrix,
        2, // m
        3, // k
        &right_matrix,
        2, // n
        &output_matrix,
    );

    // Expected result:
    // [1*1 + 2*0 + 3*1, 1*0 + 2*1 + 3*1] = [4, 5]
    // [4*1 + 5*0 + 6*1, 4*0 + 5*1 + 6*1] = [10, 11]
    const expected = [_]f32{ 4, 5, 10, 11 };

    for (output_matrix, expected) |actual, expected_value| {
        try std.testing.expectApproxEqAbs(expected_value, actual, 0.001);
    }
}
