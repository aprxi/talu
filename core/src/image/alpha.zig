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
