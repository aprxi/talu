const std = @import("std");
const pixel = @import("pixel.zig");
const codecs = @import("codecs/root.zig");

pub const EncodeFormat = enum(u8) {
    jpeg,
    png,
};

pub const EncodeOptions = struct {
    format: EncodeFormat,
    jpeg_quality: u8 = 85,
};

pub fn encode(
    allocator: std.mem.Allocator,
    img: pixel.Image,
    opts: EncodeOptions,
) ![]u8 {
    return switch (opts.format) {
        .jpeg => codecs.jpeg.encode(allocator, img, opts.jpeg_quality),
        .png => codecs.png.encode(allocator, img),
    };
}

test "encode produces JPEG with correct magic bytes" {
    const data = [_]u8{ 255, 0, 0 };
    const img: pixel.Image = .{ .width = 1, .height = 1, .stride = 3, .format = .rgb8, .data = @constCast(&data) };
    const out = try encode(std.testing.allocator, img, .{ .format = .jpeg });
    defer std.testing.allocator.free(out);
    try std.testing.expect(out.len >= 3);
    try std.testing.expectEqual(@as(u8, 0xFF), out[0]);
    try std.testing.expectEqual(@as(u8, 0xD8), out[1]);
    try std.testing.expectEqual(@as(u8, 0xFF), out[2]);
}

test "encode produces PNG with correct magic bytes" {
    const data = [_]u8{ 255, 0, 0 };
    const img: pixel.Image = .{ .width = 1, .height = 1, .stride = 3, .format = .rgb8, .data = @constCast(&data) };
    const out = try encode(std.testing.allocator, img, .{ .format = .png });
    defer std.testing.allocator.free(out);
    try std.testing.expect(out.len >= 8);
    try std.testing.expect(std.mem.eql(u8, out[0..8], &.{ 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A }));
}
