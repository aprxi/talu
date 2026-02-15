//! Alpha channel handling: strip, composite over background, or keep.

const std = @import("std");
const pixel = @import("pixel.zig");

pub const AlphaMode = enum(u8) {
    keep,
    discard,
    composite,
};

pub fn rgbaToRgb(allocator: std.mem.Allocator, src: pixel.Image, mode: AlphaMode, bg: pixel.Rgb8) !pixel.Image {
    if (src.format != .rgba8) return error.InvalidPixelFormat;
    return switch (mode) {
        .keep => cloneImage(allocator, src),
        .discard => stripAlpha(allocator, src),
        .composite => compositeRgbaToRgb(allocator, src, bg),
    };
}

pub fn stripAlpha(allocator: std.mem.Allocator, src: pixel.Image) !pixel.Image {
    if (src.format != .rgba8) return error.InvalidPixelFormat;

    const out_stride: u32 = src.width * 3;
    const out_len_u64 = @as(u64, out_stride) * @as(u64, src.height);
    const out_len = std.math.cast(usize, out_len_u64) orelse return error.ImageOutputTooLarge;
    const out = try allocator.alloc(u8, out_len);
    errdefer allocator.free(out);

    var y: u32 = 0;
    while (y < src.height) : (y += 1) {
        var x: u32 = 0;
        while (x < src.width) : (x += 1) {
            const s = @as(usize, y) * @as(usize, src.stride) + @as(usize, x) * 4;
            const d = @as(usize, y) * @as(usize, out_stride) + @as(usize, x) * 3;
            out[d + 0] = src.data[s + 0];
            out[d + 1] = src.data[s + 1];
            out[d + 2] = src.data[s + 2];
        }
    }

    return .{
        .width = src.width,
        .height = src.height,
        .stride = out_stride,
        .format = .rgb8,
        .data = out,
    };
}

pub fn compositeRgbaToRgb(allocator: std.mem.Allocator, src: pixel.Image, bg: pixel.Rgb8) !pixel.Image {
    if (src.format != .rgba8) return error.InvalidPixelFormat;

    const out_stride: u32 = src.width * 3;
    const out_len_u64 = @as(u64, out_stride) * @as(u64, src.height);
    const out_len = std.math.cast(usize, out_len_u64) orelse return error.ImageOutputTooLarge;
    const out = try allocator.alloc(u8, out_len);
    errdefer allocator.free(out);

    var y: u32 = 0;
    while (y < src.height) : (y += 1) {
        var x: u32 = 0;
        while (x < src.width) : (x += 1) {
            const s = @as(usize, y) * @as(usize, src.stride) + @as(usize, x) * 4;
            const d = @as(usize, y) * @as(usize, out_stride) + @as(usize, x) * 3;

            const a = @as(u32, src.data[s + 3]);
            const inv_a = 255 - a;

            out[d + 0] = @intCast((@as(u32, src.data[s + 0]) * a + @as(u32, bg.r) * inv_a + 127) / 255);
            out[d + 1] = @intCast((@as(u32, src.data[s + 1]) * a + @as(u32, bg.g) * inv_a + 127) / 255);
            out[d + 2] = @intCast((@as(u32, src.data[s + 2]) * a + @as(u32, bg.b) * inv_a + 127) / 255);
        }
    }

    return .{
        .width = src.width,
        .height = src.height,
        .stride = out_stride,
        .format = .rgb8,
        .data = out,
    };
}

fn makeRgba(comptime n: u32, pixels: [n][4]u8) [n * 4]u8 {
    var buf: [n * 4]u8 = undefined;
    for (0..n) |i| {
        buf[i * 4 + 0] = pixels[i][0];
        buf[i * 4 + 1] = pixels[i][1];
        buf[i * 4 + 2] = pixels[i][2];
        buf[i * 4 + 3] = pixels[i][3];
    }
    return buf;
}

fn rgbaImg(comptime n: u32, data: *const [n * 4]u8) pixel.Image {
    return .{ .width = n, .height = 1, .stride = n * 4, .format = .rgba8, .data = @constCast(@as([]const u8, data)) };
}

test "stripAlpha drops alpha channel preserving RGB" {
    const buf = makeRgba(2, .{ .{ 255, 0, 0, 255 }, .{ 0, 0, 255, 0 } });
    var out = try stripAlpha(std.testing.allocator, rgbaImg(2, &buf));
    defer out.deinit(std.testing.allocator);
    try std.testing.expectEqual(pixel.PixelFormat.rgb8, out.format);
    try std.testing.expectEqual(@as(u8, 255), out.data[0]);
    try std.testing.expectEqual(@as(u8, 0), out.data[3]);
    try std.testing.expectEqual(@as(u8, 0), out.data[4]);
    try std.testing.expectEqual(@as(u8, 255), out.data[5]);
}

test "stripAlpha rejects non-RGBA input" {
    var d = [_]u8{ 0, 0, 0 };
    const rgb: pixel.Image = .{ .width = 1, .height = 1, .stride = 3, .format = .rgb8, .data = &d };
    try std.testing.expectError(error.InvalidPixelFormat, stripAlpha(std.testing.allocator, rgb));
}

test "compositeRgbaToRgb opaque pixel unchanged" {
    const buf = makeRgba(1, .{.{ 200, 100, 50, 255 }});
    const bg: pixel.Rgb8 = .{ .r = 0, .g = 0, .b = 0 };
    var out = try compositeRgbaToRgb(std.testing.allocator, rgbaImg(1, &buf), bg);
    defer out.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u8, 200), out.data[0]);
    try std.testing.expectEqual(@as(u8, 100), out.data[1]);
    try std.testing.expectEqual(@as(u8, 50), out.data[2]);
}

test "compositeRgbaToRgb transparent pixel shows background" {
    const buf = makeRgba(1, .{.{ 200, 100, 50, 0 }});
    const bg: pixel.Rgb8 = .{ .r = 10, .g = 20, .b = 30 };
    var out = try compositeRgbaToRgb(std.testing.allocator, rgbaImg(1, &buf), bg);
    defer out.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u8, 10), out.data[0]);
    try std.testing.expectEqual(@as(u8, 20), out.data[1]);
    try std.testing.expectEqual(@as(u8, 30), out.data[2]);
}

test "compositeRgbaToRgb semi-transparent blend" {
    const buf = makeRgba(1, .{.{ 255, 0, 0, 128 }});
    const bg: pixel.Rgb8 = .{ .r = 0, .g = 0, .b = 255 };
    var out = try compositeRgbaToRgb(std.testing.allocator, rgbaImg(1, &buf), bg);
    defer out.deinit(std.testing.allocator);
    // R: (255*128 + 0*127 + 127) / 255 = 128
    try std.testing.expectEqual(@as(u8, 128), out.data[0]);
    // B: (0*128 + 255*127 + 127) / 255 = 127
    try std.testing.expectEqual(@as(u8, 127), out.data[2]);
}

test "compositeRgbaToRgb rejects non-RGBA" {
    var d = [_]u8{ 0, 0, 0 };
    const rgb: pixel.Image = .{ .width = 1, .height = 1, .stride = 3, .format = .rgb8, .data = &d };
    try std.testing.expectError(error.InvalidPixelFormat, compositeRgbaToRgb(std.testing.allocator, rgb, .{ .r = 0, .g = 0, .b = 0 }));
}

test "rgbaToRgb keep mode clones RGBA" {
    const buf = makeRgba(1, .{.{ 100, 150, 200, 128 }});
    var out = try rgbaToRgb(std.testing.allocator, rgbaImg(1, &buf), .keep, .{ .r = 0, .g = 0, .b = 0 });
    defer out.deinit(std.testing.allocator);
    try std.testing.expectEqual(pixel.PixelFormat.rgba8, out.format);
    try std.testing.expectEqual(@as(u8, 128), out.data[3]);
}

test "rgbaToRgb discard mode strips alpha" {
    const buf = makeRgba(1, .{.{ 100, 150, 200, 128 }});
    var out = try rgbaToRgb(std.testing.allocator, rgbaImg(1, &buf), .discard, .{ .r = 0, .g = 0, .b = 0 });
    defer out.deinit(std.testing.allocator);
    try std.testing.expectEqual(pixel.PixelFormat.rgb8, out.format);
    try std.testing.expectEqual(@as(u8, 100), out.data[0]);
}

test "rgbaToRgb composite mode blends" {
    const buf = makeRgba(1, .{.{ 255, 0, 0, 255 }});
    var out = try rgbaToRgb(std.testing.allocator, rgbaImg(1, &buf), .composite, .{ .r = 0, .g = 255, .b = 0 });
    defer out.deinit(std.testing.allocator);
    try std.testing.expectEqual(pixel.PixelFormat.rgb8, out.format);
    try std.testing.expectEqual(@as(u8, 255), out.data[0]);
    try std.testing.expectEqual(@as(u8, 0), out.data[1]);
}

test "rgbaToRgb rejects non-RGBA" {
    var d = [_]u8{128};
    const gray: pixel.Image = .{ .width = 1, .height = 1, .stride = 1, .format = .gray8, .data = &d };
    try std.testing.expectError(error.InvalidPixelFormat, rgbaToRgb(std.testing.allocator, gray, .discard, .{ .r = 0, .g = 0, .b = 0 }));
}

fn cloneImage(allocator: std.mem.Allocator, src: pixel.Image) !pixel.Image {
    const out = try allocator.alloc(u8, src.data.len);
    @memcpy(out, src.data);
    return .{
        .width = src.width,
        .height = src.height,
        .stride = src.stride,
        .format = src.format,
        .data = out,
    };
}
