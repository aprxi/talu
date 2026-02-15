const std = @import("std");
const main = @import("main");
const image = main.core.image;

const png_red = @embedFile("corpus/1x1_red.png");

test "image.toModelInput packs f32 NHWC zero_to_one" {
    var out = try image.toModelInput(std.testing.allocator, png_red, .{
        .width = 1,
        .height = 1,
        .dtype = .f32,
        .layout = .nhwc,
        .normalize = .zero_to_one,
    });
    defer out.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 3 * @sizeOf(f32)), out.data.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), readF32LE(out.data, 0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), readF32LE(out.data, 1), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), readF32LE(out.data, 2), 1e-6);
}

test "image.toModelInput packs u8 NCHW without normalization" {
    var out = try image.toModelInput(std.testing.allocator, png_red, .{
        .width = 1,
        .height = 1,
        .dtype = .u8,
        .layout = .nchw,
        .normalize = .none,
    });
    defer out.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 3), out.data.len);
    try std.testing.expectEqual(@as(u8, 255), out.data[0]);
    try std.testing.expectEqual(@as(u8, 0), out.data[1]);
    try std.testing.expectEqual(@as(u8, 0), out.data[2]);
}

test "image.toModelInput rejects normalized u8 output" {
    try std.testing.expectError(
        error.InvalidArgument,
        image.toModelInput(std.testing.allocator, png_red, .{
            .width = 1,
            .height = 1,
            .dtype = .u8,
            .layout = .nhwc,
            .normalize = .zero_to_one,
        }),
    );
}

fn readF32LE(bytes: []const u8, elem_idx: usize) f32 {
    const off = elem_idx * @sizeOf(f32);
    const ptr: *const [4]u8 = @ptrCast(bytes[off .. off + 4].ptr);
    const bits = std.mem.readInt(u32, ptr, .little);
    return @bitCast(bits);
}
