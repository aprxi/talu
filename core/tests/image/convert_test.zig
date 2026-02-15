const std = @import("std");
const main = @import("main");
const image = main.core.image;

test "image.convert applies alpha composite and format conversion" {
    var rgba = try std.testing.allocator.alloc(u8, 2 * 4);
    defer std.testing.allocator.free(rgba);

    // pixel0: fully opaque red, pixel1: 50% green
    rgba[0] = 255;
    rgba[1] = 0;
    rgba[2] = 0;
    rgba[3] = 255;
    rgba[4] = 0;
    rgba[5] = 255;
    rgba[6] = 0;
    rgba[7] = 128;

    const src: image.Image = .{
        .width = 2,
        .height = 1,
        .stride = 8,
        .format = .rgba8,
        .data = rgba,
    };

    var out = try image.convert(std.testing.allocator, src, .{
        .format = .rgb8,
        .alpha = .composite,
        .alpha_background = .{ .r = 0, .g = 0, .b = 0 },
    });
    defer out.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 2), out.width);
    try std.testing.expectEqual(@as(u32, 1), out.height);
    try std.testing.expectEqual(@as(u8, 255), out.data[0]);
    try std.testing.expect(out.data[4] >= 126 and out.data[4] <= 129);
}

test "image.convert resizes with contain framing" {
    const rgb = try std.testing.allocator.alloc(u8, 3 * 2 * 2);
    defer std.testing.allocator.free(rgb);
    @memset(rgb, 255);

    const src: image.Image = .{
        .width = 2,
        .height = 2,
        .stride = 6,
        .format = .rgb8,
        .data = rgb,
    };

    var out = try image.convert(std.testing.allocator, src, .{
        .format = .rgb8,
        .resize = .{
            .out_w = 4,
            .out_h = 2,
            .fit = .contain,
            .filter = .nearest,
            .pad_color = .{ .r = 10, .g = 20, .b = 30 },
        },
    });
    defer out.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 4), out.width);
    try std.testing.expectEqual(@as(u32, 2), out.height);
    // Contain with 2x2 -> 4x2 introduces horizontal padding on both sides.
    try std.testing.expectEqual(@as(u8, 10), out.data[0]);
    try std.testing.expectEqual(@as(u8, 255), out.data[3]);
}
