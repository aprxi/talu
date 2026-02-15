const std = @import("std");
const pixel = @import("pixel.zig");
const limits_mod = @import("limits.zig");
const alpha_mod = @import("alpha.zig");
const resize_mod = @import("resize.zig");

pub const ConvertSpec = struct {
    format: pixel.PixelFormat = .rgb8,
    resize: ?resize_mod.ResizeOptions = null,
    alpha: alpha_mod.AlphaMode = .composite,
    alpha_background: pixel.Rgb8 = .{ .r = 0, .g = 0, .b = 0 },
    orientation: enum { preserve, normalize_to_normal } = .preserve,
    limits: limits_mod.Limits = .{},
};

pub fn convert(
    allocator: std.mem.Allocator,
    src: pixel.Image,
    spec: ConvertSpec,
) !pixel.Image {
    _ = spec.orientation;

    var current = try cloneImage(allocator, src);
    errdefer current.deinit(allocator);

    if (current.format == .rgba8 and spec.format != .rgba8) {
        const no_alpha = switch (spec.alpha) {
            .keep => return error.InvalidArgument,
            .discard => try alpha_mod.stripAlpha(allocator, current),
            .composite => try alpha_mod.compositeRgbaToRgb(allocator, current, spec.alpha_background),
        };
        current.deinit(allocator);
        current = no_alpha;
    }

    if (current.format != spec.format) {
        const converted = try convertFormat(allocator, current, spec.format, spec.alpha_background);
        current.deinit(allocator);
        current = converted;
    }

    if (spec.resize) |resize_opts| {
        const resized = try resize_mod.resize(allocator, current, resize_opts, spec.limits);
        current.deinit(allocator);
        current = resized;
    }

    const required = try current.requiredLen();
    try spec.limits.validateOutputBytes(required);

    return current;
}

fn convertFormat(
    allocator: std.mem.Allocator,
    src: pixel.Image,
    dst_format: pixel.PixelFormat,
    alpha_background: pixel.Rgb8,
) !pixel.Image {
    if (src.format == dst_format) return cloneImage(allocator, src);

    return switch (dst_format) {
        .gray8 => toGray8(allocator, src, alpha_background),
        .rgb8 => toRgb8(allocator, src, alpha_background),
        .rgba8 => toRgba8(allocator, src),
    };
}

fn toGray8(allocator: std.mem.Allocator, src: pixel.Image, alpha_background: pixel.Rgb8) !pixel.Image {
    const out_stride: u32 = src.width;
    const out_len_u64 = @as(u64, out_stride) * @as(u64, src.height);
    const out_len = std.math.cast(usize, out_len_u64) orelse return error.ImageOutputTooLarge;
    const out = try allocator.alloc(u8, out_len);
    errdefer allocator.free(out);

    var y: u32 = 0;
    while (y < src.height) : (y += 1) {
        var x: u32 = 0;
        while (x < src.width) : (x += 1) {
            const d = @as(usize, y) * @as(usize, out_stride) + @as(usize, x);
            const rgb = readPixelAsRgb(src, x, y, alpha_background);
            out[d] = luminance(rgb.r, rgb.g, rgb.b);
        }
    }

    return .{
        .width = src.width,
        .height = src.height,
        .stride = out_stride,
        .format = .gray8,
        .data = out,
    };
}

fn toRgb8(allocator: std.mem.Allocator, src: pixel.Image, alpha_background: pixel.Rgb8) !pixel.Image {
    const out_stride: u32 = src.width * 3;
    const out_len_u64 = @as(u64, out_stride) * @as(u64, src.height);
    const out_len = std.math.cast(usize, out_len_u64) orelse return error.ImageOutputTooLarge;
    const out = try allocator.alloc(u8, out_len);
    errdefer allocator.free(out);

    var y: u32 = 0;
    while (y < src.height) : (y += 1) {
        var x: u32 = 0;
        while (x < src.width) : (x += 1) {
            const d = @as(usize, y) * @as(usize, out_stride) + @as(usize, x) * 3;
            const rgb = readPixelAsRgb(src, x, y, alpha_background);
            out[d + 0] = rgb.r;
            out[d + 1] = rgb.g;
            out[d + 2] = rgb.b;
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

fn toRgba8(allocator: std.mem.Allocator, src: pixel.Image) !pixel.Image {
    const out_stride: u32 = src.width * 4;
    const out_len_u64 = @as(u64, out_stride) * @as(u64, src.height);
    const out_len = std.math.cast(usize, out_len_u64) orelse return error.ImageOutputTooLarge;
    const out = try allocator.alloc(u8, out_len);
    errdefer allocator.free(out);

    var y: u32 = 0;
    while (y < src.height) : (y += 1) {
        var x: u32 = 0;
        while (x < src.width) : (x += 1) {
            const d = @as(usize, y) * @as(usize, out_stride) + @as(usize, x) * 4;
            switch (src.format) {
                .gray8 => {
                    const off = @as(usize, y) * @as(usize, src.stride) + @as(usize, x);
                    const v = src.data[off];
                    out[d + 0] = v;
                    out[d + 1] = v;
                    out[d + 2] = v;
                    out[d + 3] = 255;
                },
                .rgb8 => {
                    const off = @as(usize, y) * @as(usize, src.stride) + @as(usize, x) * 3;
                    out[d + 0] = src.data[off + 0];
                    out[d + 1] = src.data[off + 1];
                    out[d + 2] = src.data[off + 2];
                    out[d + 3] = 255;
                },
                .rgba8 => {
                    const off = @as(usize, y) * @as(usize, src.stride) + @as(usize, x) * 4;
                    @memcpy(out[d .. d + 4], src.data[off .. off + 4]);
                },
            }
        }
    }

    return .{
        .width = src.width,
        .height = src.height,
        .stride = out_stride,
        .format = .rgba8,
        .data = out,
    };
}

fn readPixelAsRgb(src: pixel.Image, x: u32, y: u32, bg: pixel.Rgb8) pixel.Rgb8 {
    return switch (src.format) {
        .gray8 => blk: {
            const off = @as(usize, y) * @as(usize, src.stride) + @as(usize, x);
            const v = src.data[off];
            break :blk .{ .r = v, .g = v, .b = v };
        },
        .rgb8 => blk: {
            const off = @as(usize, y) * @as(usize, src.stride) + @as(usize, x) * 3;
            break :blk .{ .r = src.data[off + 0], .g = src.data[off + 1], .b = src.data[off + 2] };
        },
        .rgba8 => blk: {
            const off = @as(usize, y) * @as(usize, src.stride) + @as(usize, x) * 4;
            const a = @as(u32, src.data[off + 3]);
            const inv = 255 - a;
            break :blk .{
                .r = @intCast((@as(u32, src.data[off + 0]) * a + @as(u32, bg.r) * inv + 127) / 255),
                .g = @intCast((@as(u32, src.data[off + 1]) * a + @as(u32, bg.g) * inv + 127) / 255),
                .b = @intCast((@as(u32, src.data[off + 2]) * a + @as(u32, bg.b) * inv + 127) / 255),
            };
        },
    };
}

fn luminance(r: u8, g: u8, b: u8) u8 {
    const y = @as(u16, r) * 30 + @as(u16, g) * 59 + @as(u16, b) * 11;
    return @intCast(y / 100);
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
