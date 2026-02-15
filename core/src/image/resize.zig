const std = @import("std");
const pixel = @import("pixel.zig");
const limits_mod = @import("limits.zig");

pub const ResizeFilter = enum(u8) {
    nearest,
    bilinear,
    bicubic,
};

pub const FitMode = enum(u8) {
    stretch,
    contain,
    cover,
};

pub const ResizeOptions = struct {
    out_w: u32,
    out_h: u32,
    fit: FitMode = .contain,
    filter: ResizeFilter = .bicubic,
    pad_color: pixel.Rgb8 = .{ .r = 0, .g = 0, .b = 0 },
};

pub fn resize(
    allocator: std.mem.Allocator,
    src: pixel.Image,
    opts: ResizeOptions,
    lim: limits_mod.Limits,
) !pixel.Image {
    if (opts.out_w == 0 or opts.out_h == 0) return error.InvalidImageDimensions;
    const bpp = pixel.bytesPerPixel(src.format);
    _ = try lim.checkedOutputSize(opts.out_w, opts.out_h, bpp);

    return switch (opts.fit) {
        .stretch => resampleToDims(allocator, src, opts.out_w, opts.out_h, opts.filter),
        .contain => resizeContain(allocator, src, opts, lim),
        .cover => resizeCover(allocator, src, opts, lim),
    };
}

fn resizeContain(
    allocator: std.mem.Allocator,
    src: pixel.Image,
    opts: ResizeOptions,
    lim: limits_mod.Limits,
) !pixel.Image {
    const scale_x = @as(f64, @floatFromInt(opts.out_w)) / @as(f64, @floatFromInt(src.width));
    const scale_y = @as(f64, @floatFromInt(opts.out_h)) / @as(f64, @floatFromInt(src.height));
    const scale = @min(scale_x, scale_y);

    const scaled_w = @max(@as(u32, 1), @as(u32, @intFromFloat(@round(@as(f64, @floatFromInt(src.width)) * scale))));
    const scaled_h = @max(@as(u32, 1), @as(u32, @intFromFloat(@round(@as(f64, @floatFromInt(src.height)) * scale))));

    var scaled = try resampleToDims(allocator, src, scaled_w, scaled_h, opts.filter);
    defer scaled.deinit(allocator);

    const canvas = try allocCanvas(allocator, opts.out_w, opts.out_h, src.format, opts.pad_color, lim);
    var out = canvas;

    const offset_x = (opts.out_w - scaled_w) / 2;
    const offset_y = (opts.out_h - scaled_h) / 2;
    blitCentered(scaled, &out, offset_x, offset_y);

    return out;
}

fn resizeCover(
    allocator: std.mem.Allocator,
    src: pixel.Image,
    opts: ResizeOptions,
    lim: limits_mod.Limits,
) !pixel.Image {
    const scale_x = @as(f64, @floatFromInt(opts.out_w)) / @as(f64, @floatFromInt(src.width));
    const scale_y = @as(f64, @floatFromInt(opts.out_h)) / @as(f64, @floatFromInt(src.height));
    const scale = @max(scale_x, scale_y);

    const scaled_w = @max(@as(u32, 1), @as(u32, @intFromFloat(@round(@as(f64, @floatFromInt(src.width)) * scale))));
    const scaled_h = @max(@as(u32, 1), @as(u32, @intFromFloat(@round(@as(f64, @floatFromInt(src.height)) * scale))));

    var scaled = try resampleToDims(allocator, src, scaled_w, scaled_h, opts.filter);
    defer scaled.deinit(allocator);

    const out = try allocCanvas(allocator, opts.out_w, opts.out_h, src.format, opts.pad_color, lim);
    var result = out;

    const crop_x = (scaled_w - opts.out_w) / 2;
    const crop_y = (scaled_h - opts.out_h) / 2;
    cropInto(scaled, &result, crop_x, crop_y);

    return result;
}

fn allocCanvas(
    allocator: std.mem.Allocator,
    w: u32,
    h: u32,
    format: pixel.PixelFormat,
    pad_color: pixel.Rgb8,
    lim: limits_mod.Limits,
) !pixel.Image {
    const bpp = pixel.bytesPerPixel(format);
    const len = try lim.checkedOutputSize(w, h, bpp);
    const stride: u32 = w * bpp;
    const data = try allocator.alloc(u8, len);

    fillCanvas(data, stride, w, h, format, pad_color);

    return .{
        .width = w,
        .height = h,
        .stride = stride,
        .format = format,
        .data = data,
    };
}

fn fillCanvas(
    data: []u8,
    stride: u32,
    w: u32,
    h: u32,
    format: pixel.PixelFormat,
    pad_color: pixel.Rgb8,
) void {
    switch (format) {
        .gray8 => {
            const y_val: u8 = @intCast((@as(u16, pad_color.r) * 30 + @as(u16, pad_color.g) * 59 + @as(u16, pad_color.b) * 11) / 100);
            @memset(data, y_val);
        },
        .rgb8 => {
            var row: u32 = 0;
            while (row < h) : (row += 1) {
                var x: u32 = 0;
                while (x < w) : (x += 1) {
                    const off = @as(usize, row) * @as(usize, stride) + @as(usize, x) * 3;
                    data[off + 0] = pad_color.r;
                    data[off + 1] = pad_color.g;
                    data[off + 2] = pad_color.b;
                }
            }
        },
        .rgba8 => {
            var row: u32 = 0;
            while (row < h) : (row += 1) {
                var x: u32 = 0;
                while (x < w) : (x += 1) {
                    const off = @as(usize, row) * @as(usize, stride) + @as(usize, x) * 4;
                    data[off + 0] = pad_color.r;
                    data[off + 1] = pad_color.g;
                    data[off + 2] = pad_color.b;
                    data[off + 3] = 255;
                }
            }
        },
    }
}

fn blitCentered(src: pixel.Image, dst: *pixel.Image, offset_x: u32, offset_y: u32) void {
    const bpp = @as(usize, pixel.bytesPerPixel(src.format));
    var y: u32 = 0;
    while (y < src.height) : (y += 1) {
        const src_row = @as(usize, y) * @as(usize, src.stride);
        const dst_row = @as(usize, y + offset_y) * @as(usize, dst.stride) + @as(usize, offset_x) * bpp;
        const row_len = @as(usize, src.width) * bpp;
        @memcpy(dst.data[dst_row .. dst_row + row_len], src.data[src_row .. src_row + row_len]);
    }
}

fn cropInto(src: pixel.Image, dst: *pixel.Image, crop_x: u32, crop_y: u32) void {
    const bpp = @as(usize, pixel.bytesPerPixel(src.format));
    var y: u32 = 0;
    while (y < dst.height) : (y += 1) {
        const src_row = @as(usize, y + crop_y) * @as(usize, src.stride) + @as(usize, crop_x) * bpp;
        const dst_row = @as(usize, y) * @as(usize, dst.stride);
        const row_len = @as(usize, dst.width) * bpp;
        @memcpy(dst.data[dst_row .. dst_row + row_len], src.data[src_row .. src_row + row_len]);
    }
}

fn resampleToDims(
    allocator: std.mem.Allocator,
    src: pixel.Image,
    out_w: u32,
    out_h: u32,
    filter: ResizeFilter,
) !pixel.Image {
    if (out_w == src.width and out_h == src.height) {
        return cloneImage(allocator, src);
    }

    const bpp = @as(usize, pixel.bytesPerPixel(src.format));
    const out_stride = out_w * @as(u32, @intCast(bpp));
    const out_len_u64 = @as(u64, out_stride) * @as(u64, out_h);
    const out_len = std.math.cast(usize, out_len_u64) orelse return error.ImageOutputTooLarge;
    const out = try allocator.alloc(u8, out_len);
    errdefer allocator.free(out);

    var y: u32 = 0;
    while (y < out_h) : (y += 1) {
        const src_y = mapCoord(y, out_h, src.height);
        var x: u32 = 0;
        while (x < out_w) : (x += 1) {
            const src_x = mapCoord(x, out_w, src.width);
            const dst_off = @as(usize, y) * @as(usize, out_stride) + @as(usize, x) * bpp;

            var c_idx: usize = 0;
            while (c_idx < bpp) : (c_idx += 1) {
                const value = sampleChannel(src, src_x, src_y, c_idx, filter);
                out[dst_off + c_idx] = @intFromFloat(@min(@max(value, 0.0), 255.0));
            }
        }
    }

    return .{
        .width = out_w,
        .height = out_h,
        .stride = out_stride,
        .format = src.format,
        .data = out,
    };
}

fn mapCoord(dst: u32, dst_size: u32, src_size: u32) f32 {
    const d = @as(f32, @floatFromInt(dst));
    const ds = @as(f32, @floatFromInt(dst_size));
    const ss = @as(f32, @floatFromInt(src_size));
    return ((d + 0.5) * ss / ds) - 0.5;
}

fn sampleChannel(src: pixel.Image, x: f32, y: f32, c_idx: usize, filter: ResizeFilter) f32 {
    return switch (filter) {
        .nearest => sampleNearest(src, x, y, c_idx),
        .bilinear => sampleBilinear(src, x, y, c_idx),
        .bicubic => sampleBicubic(src, x, y, c_idx),
    };
}

fn sampleNearest(src: pixel.Image, x: f32, y: f32, c_idx: usize) f32 {
    const xi = clampInt(@intFromFloat(@round(x)), src.width);
    const yi = clampInt(@intFromFloat(@round(y)), src.height);
    return @floatFromInt(readChannel(src, xi, yi, c_idx));
}

fn sampleBilinear(src: pixel.Image, x: f32, y: f32, c_idx: usize) f32 {
    const x0f = @floor(x);
    const y0f = @floor(y);
    const tx = x - x0f;
    const ty = y - y0f;

    const x0 = clampInt(@intFromFloat(x0f), src.width);
    const y0 = clampInt(@intFromFloat(y0f), src.height);
    const x1 = clampInt(@as(i32, @intCast(x0)) + 1, src.width);
    const y1 = clampInt(@as(i32, @intCast(y0)) + 1, src.height);

    const c00 = @as(f32, @floatFromInt(readChannel(src, x0, y0, c_idx)));
    const c10 = @as(f32, @floatFromInt(readChannel(src, x1, y0, c_idx)));
    const c01 = @as(f32, @floatFromInt(readChannel(src, x0, y1, c_idx)));
    const c11 = @as(f32, @floatFromInt(readChannel(src, x1, y1, c_idx)));

    const a = c00 + (c10 - c00) * tx;
    const b = c01 + (c11 - c01) * tx;
    return a + (b - a) * ty;
}

fn sampleBicubic(src: pixel.Image, x: f32, y: f32, c_idx: usize) f32 {
    const x_base_f = @floor(x);
    const y_base_f = @floor(y);
    const tx = x - x_base_f;
    const ty = y - y_base_f;

    const x_base = @as(i32, @intFromFloat(x_base_f));
    const y_base = @as(i32, @intFromFloat(y_base_f));

    var accum: f32 = 0;
    var m: i32 = -1;
    while (m <= 2) : (m += 1) {
        const wy = cubicKernel(@as(f32, @floatFromInt(m)) - ty);
        var n: i32 = -1;
        while (n <= 2) : (n += 1) {
            const wx = cubicKernel(@as(f32, @floatFromInt(n)) - tx);
            const sx = clampInt(x_base + n, src.width);
            const sy = clampInt(y_base + m, src.height);
            const sample = @as(f32, @floatFromInt(readChannel(src, sx, sy, c_idx)));
            accum += sample * wx * wy;
        }
    }

    return accum;
}

fn cubicKernel(t: f32) f32 {
    const a: f32 = -0.5;
    const x = @abs(t);
    if (x < 1.0) {
        return (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0;
    }
    if (x < 2.0) {
        return a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a;
    }
    return 0.0;
}

fn clampInt(v: i32, bound: u32) u32 {
    if (v < 0) return 0;
    const vu = @as(u32, @intCast(v));
    return if (vu >= bound) bound - 1 else vu;
}

fn readChannel(src: pixel.Image, x: u32, y: u32, c_idx: usize) u8 {
    const bpp = @as(usize, pixel.bytesPerPixel(src.format));
    const off = @as(usize, y) * @as(usize, src.stride) + @as(usize, x) * bpp + c_idx;
    return src.data[off];
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
