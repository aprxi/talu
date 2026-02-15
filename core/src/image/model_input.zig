const std = @import("std");
const pixel = @import("pixel.zig");
const limits_mod = @import("limits.zig");

pub const TensorLayout = enum(u8) {
    nhwc,
    nchw,
};

pub const DType = enum(u8) {
    u8,
    f32,
};

pub const Normalize = enum(u8) {
    none,
    zero_to_one,
    imagenet,
};

pub const ModelInputSpec = struct {
    width: u32,
    height: u32,
    dtype: DType = .u8,
    layout: TensorLayout = .nhwc,
    normalize: Normalize = .zero_to_one,
    fit: @import("resize.zig").FitMode = .contain,
    filter: @import("resize.zig").ResizeFilter = .bicubic,
    pad_color: pixel.Rgb8 = .{ .r = 0, .g = 0, .b = 0 },
    limits: limits_mod.Limits = .{},
};

pub const ModelBuffer = struct {
    data: []u8,
    width: u32,
    height: u32,
    channels: u8,
    layout: TensorLayout,
    dtype: DType,

    pub fn deinit(self: *ModelBuffer, allocator: std.mem.Allocator) void {
        if (self.data.len != 0) allocator.free(self.data);
        self.* = .{
            .data = &.{},
            .width = 0,
            .height = 0,
            .channels = 0,
            .layout = .nhwc,
            .dtype = .u8,
        };
    }
};

pub fn packModelInput(
    allocator: std.mem.Allocator,
    img: pixel.Image,
    spec: ModelInputSpec,
) !ModelBuffer {
    if (img.format != .rgb8) return error.InvalidPixelFormat;
    if (img.width != spec.width or img.height != spec.height) return error.InvalidImageDimensions;

    const channels: u8 = 3;
    const elem_count_u64 = @as(u64, spec.width) * @as(u64, spec.height) * channels;
    const elem_count = std.math.cast(usize, elem_count_u64) orelse return error.ImageOutputTooLarge;

    return switch (spec.dtype) {
        .u8 => packU8(allocator, img, spec, elem_count, channels),
        .f32 => packF32(allocator, img, spec, elem_count, channels),
    };
}

fn packU8(
    allocator: std.mem.Allocator,
    img: pixel.Image,
    spec: ModelInputSpec,
    elem_count: usize,
    channels: u8,
) !ModelBuffer {
    if (spec.normalize != .none) return error.InvalidArgument;

    const out = try allocator.alloc(u8, elem_count);
    errdefer allocator.free(out);

    writeU8(img, spec.layout, out);

    return .{
        .data = out,
        .width = spec.width,
        .height = spec.height,
        .channels = channels,
        .layout = spec.layout,
        .dtype = .u8,
    };
}

fn packF32(
    allocator: std.mem.Allocator,
    img: pixel.Image,
    spec: ModelInputSpec,
    elem_count: usize,
    channels: u8,
) !ModelBuffer {
    const bytes_len_u64 = @as(u64, elem_count) * @sizeOf(f32);
    const bytes_len = std.math.cast(usize, bytes_len_u64) orelse return error.ImageOutputTooLarge;
    try spec.limits.validateOutputBytes(bytes_len);

    const out = try allocator.alloc(u8, bytes_len);
    errdefer allocator.free(out);

    writeF32(img, spec.layout, spec.normalize, out);

    return .{
        .data = out,
        .width = spec.width,
        .height = spec.height,
        .channels = channels,
        .layout = spec.layout,
        .dtype = .f32,
    };
}

fn writeU8(img: pixel.Image, layout: TensorLayout, out: []u8) void {
    const w = img.width;
    const h = img.height;

    switch (layout) {
        .nhwc => {
            var y: u32 = 0;
            while (y < h) : (y += 1) {
                var x: u32 = 0;
                while (x < w) : (x += 1) {
                    const src_off = @as(usize, y) * @as(usize, img.stride) + @as(usize, x) * 3;
                    const dst_off = (@as(usize, y) * @as(usize, w) + @as(usize, x)) * 3;
                    out[dst_off + 0] = img.data[src_off + 0];
                    out[dst_off + 1] = img.data[src_off + 1];
                    out[dst_off + 2] = img.data[src_off + 2];
                }
            }
        },
        .nchw => {
            const plane = @as(usize, w) * @as(usize, h);
            var y: u32 = 0;
            while (y < h) : (y += 1) {
                var x: u32 = 0;
                while (x < w) : (x += 1) {
                    const src_off = @as(usize, y) * @as(usize, img.stride) + @as(usize, x) * 3;
                    const idx = @as(usize, y) * @as(usize, w) + @as(usize, x);
                    out[idx] = img.data[src_off + 0];
                    out[plane + idx] = img.data[src_off + 1];
                    out[2 * plane + idx] = img.data[src_off + 2];
                }
            }
        },
    }
}

fn writeF32(img: pixel.Image, layout: TensorLayout, normalize: Normalize, out: []u8) void {
    const w = img.width;
    const h = img.height;

    switch (layout) {
        .nhwc => {
            var y: u32 = 0;
            while (y < h) : (y += 1) {
                var x: u32 = 0;
                while (x < w) : (x += 1) {
                    const src_off = @as(usize, y) * @as(usize, img.stride) + @as(usize, x) * 3;
                    const dst_elem = (@as(usize, y) * @as(usize, w) + @as(usize, x)) * 3;
                    writeF32At(out, dst_elem + 0, normalizeChannel(img.data[src_off + 0], 0, normalize));
                    writeF32At(out, dst_elem + 1, normalizeChannel(img.data[src_off + 1], 1, normalize));
                    writeF32At(out, dst_elem + 2, normalizeChannel(img.data[src_off + 2], 2, normalize));
                }
            }
        },
        .nchw => {
            const plane = @as(usize, w) * @as(usize, h);
            var y: u32 = 0;
            while (y < h) : (y += 1) {
                var x: u32 = 0;
                while (x < w) : (x += 1) {
                    const src_off = @as(usize, y) * @as(usize, img.stride) + @as(usize, x) * 3;
                    const idx = @as(usize, y) * @as(usize, w) + @as(usize, x);
                    writeF32At(out, idx, normalizeChannel(img.data[src_off + 0], 0, normalize));
                    writeF32At(out, plane + idx, normalizeChannel(img.data[src_off + 1], 1, normalize));
                    writeF32At(out, 2 * plane + idx, normalizeChannel(img.data[src_off + 2], 2, normalize));
                }
            }
        },
    }
}

fn writeF32At(out: []u8, elem_idx: usize, value: f32) void {
    const off = elem_idx * @sizeOf(f32);
    const bits: u32 = @bitCast(value);
    const dst: *[4]u8 = @ptrCast(out[off .. off + 4].ptr);
    std.mem.writeInt(u32, dst, bits, .little);
}

fn normalizeChannel(v: u8, channel: u8, mode: Normalize) f32 {
    const f = @as(f32, @floatFromInt(v));
    return switch (mode) {
        .none => f,
        .zero_to_one => f / 255.0,
        .imagenet => blk: {
            const n = f / 255.0;
            const mean: f32 = switch (channel) {
                0 => @as(f32, 0.485),
                1 => @as(f32, 0.456),
                else => @as(f32, 0.406),
            };
            const stddev: f32 = switch (channel) {
                0 => @as(f32, 0.229),
                1 => @as(f32, 0.224),
                else => @as(f32, 0.225),
            };
            break :blk (n - mean) / stddev;
        },
    };
}
