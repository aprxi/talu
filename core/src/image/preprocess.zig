//! Vision preprocessing: smart resize, planar F32 conversion, and grid tiling.

const std = @import("std");
const pixel = @import("pixel.zig");
const limits_mod = @import("limits.zig");
const resize_mod = @import("resize.zig");
const convert_mod = @import("convert.zig");
const alpha_mod = @import("alpha.zig");

pub const VisionNormalize = enum(u8) {
    none,
    zero_to_one,
    minus_one_to_one,
    imagenet,
};

pub const PlanarF32Spec = struct {
    temporal_frames: u32 = 1,
    normalize: VisionNormalize = .zero_to_one,
};

pub const SmartResizeOptions = struct {
    factor: u32 = 1,
    min_pixels: u64 = 0,
    max_pixels: u64 = 0,
};

pub const SmartResizeResult = struct {
    width: u32,
    height: u32,
};

pub const VisionGrid = struct {
    temporal: u32,
    height: u32,
    width: u32,
};

pub const TokenCountOptions = struct {
    patch_size: u32,
    spatial_merge_size: u32,
    temporal_frames: u32 = 1,
    temporal_patch_size: u32 = 1,
    smart_resize: ?SmartResizeOptions = null,
};

pub const ExplicitResize = struct {
    width: u32,
    height: u32,
    fit: resize_mod.FitMode = .stretch,
    filter: resize_mod.ResizeFilter = .bicubic,
    pad_color: pixel.Rgb8 = .{ .r = 0, .g = 0, .b = 0 },
};

pub const VisionPreprocessOptions = struct {
    normalize: VisionNormalize = .minus_one_to_one,
    temporal_frames: u32 = 1,
    patch_size: u32 = 1,
    temporal_patch_size: u32 = 1,
    spatial_merge_size: u32 = 1,
    smart_resize: ?SmartResizeOptions = null,
    explicit_resize: ?ExplicitResize = null,
    alpha: alpha_mod.AlphaMode = .composite,
    alpha_background: pixel.Rgb8 = .{ .r = 0, .g = 0, .b = 0 },
    limits: limits_mod.Limits = .{},
};

pub const VisionPreprocessResult = struct {
    pixels: []f32,
    width: u32,
    height: u32,
    grid: VisionGrid,
    token_count: u32,

    pub fn deinit(self: *VisionPreprocessResult, allocator: std.mem.Allocator) void {
        if (self.pixels.len != 0) allocator.free(self.pixels);
        self.* = .{
            .pixels = &.{},
            .width = 0,
            .height = 0,
            .grid = .{ .temporal = 0, .height = 0, .width = 0 },
            .token_count = 0,
        };
    }
};

pub fn smartResize(width: u32, height: u32, opts: SmartResizeOptions) !SmartResizeResult {
    if (width == 0 or height == 0) return error.InvalidImageDimensions;
    if (opts.factor == 0) return error.InvalidArgument;
    if (opts.min_pixels > 0 and opts.max_pixels > 0 and opts.min_pixels > opts.max_pixels) {
        return error.InvalidArgument;
    }

    const factor = opts.factor;
    if (opts.max_pixels > 0) {
        const factor_pixels = try checkedPixels(factor, factor);
        if (opts.max_pixels < factor_pixels) return error.InvalidArgument;
    }

    var out_w = alignNearest(width, factor);
    var out_h = alignNearest(height, factor);

    const current_pixels = try checkedPixels(out_w, out_h);
    if (opts.max_pixels > 0 and current_pixels > opts.max_pixels) {
        const scaled = try scaleAndAlign(out_w, out_h, opts.max_pixels, factor, .down);
        out_w = scaled.width;
        out_h = scaled.height;
    } else if (opts.min_pixels > 0 and current_pixels < opts.min_pixels) {
        const scaled = try scaleAndAlign(out_w, out_h, opts.min_pixels, factor, .up);
        out_w = scaled.width;
        out_h = scaled.height;
    }

    const final_pixels = try checkedPixels(out_w, out_h);
    if (opts.max_pixels > 0 and final_pixels > opts.max_pixels) {
        return error.InvalidArgument;
    }
    if (opts.min_pixels > 0 and final_pixels < opts.min_pixels) {
        return error.InvalidArgument;
    }

    return .{ .width = out_w, .height = out_h };
}

pub fn toPlanarF32(
    allocator: std.mem.Allocator,
    img: pixel.Image,
    spec: PlanarF32Spec,
) ![]f32 {
    if (img.format != .rgb8) return error.InvalidPixelFormat;
    if (spec.temporal_frames == 0) return error.InvalidArgument;

    const h = @as(usize, img.height);
    const w = @as(usize, img.width);
    const t = @as(usize, spec.temporal_frames);
    const channels: usize = 3;
    const plane = try std.math.mul(usize, h, w);
    const per_channel = try std.math.mul(usize, t, plane);
    const total = try std.math.mul(usize, channels, per_channel);

    const out = try allocator.alloc(f32, total);
    errdefer allocator.free(out);

    var row: usize = 0;
    while (row < h) : (row += 1) {
        var col: usize = 0;
        while (col < w) : (col += 1) {
            const src_idx = row * @as(usize, img.stride) + col * 3;
            const r = normalizeChannel(img.data[src_idx + 0], 0, spec.normalize);
            const g = normalizeChannel(img.data[src_idx + 1], 1, spec.normalize);
            const b = normalizeChannel(img.data[src_idx + 2], 2, spec.normalize);
            const pixel_idx = row * w + col;

            var frame: usize = 0;
            while (frame < t) : (frame += 1) {
                const frame_base = frame * plane + pixel_idx;
                out[0 * per_channel + frame_base] = r;
                out[1 * per_channel + frame_base] = g;
                out[2 * per_channel + frame_base] = b;
            }
        }
    }

    return out;
}

pub fn calculateVisionGrid(
    width: u32,
    height: u32,
    patch_size: u32,
    temporal_frames: u32,
    temporal_patch_size: u32,
) !VisionGrid {
    if (width == 0 or height == 0) return error.InvalidImageDimensions;
    if (patch_size == 0 or temporal_patch_size == 0) return error.InvalidArgument;
    if (width % patch_size != 0 or height % patch_size != 0) return error.InvalidImageDimensions;
    if (temporal_frames == 0 or temporal_frames % temporal_patch_size != 0) return error.InvalidImageDimensions;

    return .{
        .temporal = temporal_frames / temporal_patch_size,
        .height = height / patch_size,
        .width = width / patch_size,
    };
}

pub fn calculateMergedTokenCount(grid: VisionGrid, spatial_merge_size: u32) !u32 {
    if (spatial_merge_size == 0) return error.InvalidArgument;
    if (grid.height == 0 or grid.width == 0 or grid.temporal == 0) return error.InvalidImageDimensions;
    if (grid.height % spatial_merge_size != 0 or grid.width % spatial_merge_size != 0) {
        return error.InvalidImageDimensions;
    }

    const merged_h = grid.height / spatial_merge_size;
    const merged_w = grid.width / spatial_merge_size;
    const tmp = try std.math.mul(u64, @as(u64, grid.temporal), @as(u64, merged_h));
    const tokens = try std.math.mul(u64, tmp, @as(u64, merged_w));
    return std.math.cast(u32, tokens) orelse error.ImageOutputTooLarge;
}

pub fn calculateTokenCountForImage(width: u32, height: u32, opts: TokenCountOptions) !u32 {
    const dims = if (opts.smart_resize) |sr|
        try smartResize(width, height, sr)
    else
        SmartResizeResult{ .width = width, .height = height };

    const grid = try calculateVisionGrid(
        dims.width,
        dims.height,
        opts.patch_size,
        opts.temporal_frames,
        opts.temporal_patch_size,
    );
    return calculateMergedTokenCount(grid, opts.spatial_merge_size);
}

pub fn preprocessImage(
    allocator: std.mem.Allocator,
    src: pixel.Image,
    opts: VisionPreprocessOptions,
) !VisionPreprocessResult {
    if (opts.smart_resize != null and opts.explicit_resize != null) return error.InvalidArgument;

    var working = try convert_mod.convert(allocator, src, .{
        .format = .rgb8,
        .alpha = opts.alpha,
        .alpha_background = opts.alpha_background,
        .limits = opts.limits,
    });
    defer working.deinit(allocator);

    const final_dims = if (opts.smart_resize) |sr|
        try smartResize(working.width, working.height, sr)
    else if (opts.explicit_resize) |er|
        SmartResizeResult{ .width = er.width, .height = er.height }
    else
        SmartResizeResult{ .width = working.width, .height = working.height };

    var resized = working;
    var resized_owned = false;
    if (final_dims.width != working.width or final_dims.height != working.height) {
        const resize_opts = if (opts.explicit_resize) |er| resize_mod.ResizeOptions{
            .out_w = final_dims.width,
            .out_h = final_dims.height,
            .fit = er.fit,
            .filter = er.filter,
            .pad_color = er.pad_color,
        } else resize_mod.ResizeOptions{
            .out_w = final_dims.width,
            .out_h = final_dims.height,
            .fit = .stretch,
            .filter = .bicubic,
            .pad_color = opts.alpha_background,
        };

        resized = try convert_mod.convert(allocator, working, .{
            .format = .rgb8,
            .resize = resize_opts,
            .alpha = .composite,
            .alpha_background = opts.alpha_background,
            .limits = opts.limits,
        });
        resized_owned = true;
    }
    defer if (resized_owned) resized.deinit(allocator);

    const grid = try calculateVisionGrid(
        resized.width,
        resized.height,
        opts.patch_size,
        opts.temporal_frames,
        opts.temporal_patch_size,
    );
    const token_count = try calculateMergedTokenCount(grid, opts.spatial_merge_size);

    const pixel_count = try std.math.mul(u64, @as(u64, resized.width), @as(u64, resized.height));
    const total_values = try std.math.mul(u64, pixel_count, @as(u64, 3 * opts.temporal_frames));
    const out_bytes_u64 = try std.math.mul(u64, total_values, @sizeOf(f32));
    const out_bytes = std.math.cast(usize, out_bytes_u64) orelse return error.ImageOutputTooLarge;
    try opts.limits.validateOutputBytes(out_bytes);

    const pixels = try toPlanarF32(allocator, resized, .{
        .temporal_frames = opts.temporal_frames,
        .normalize = opts.normalize,
    });

    return .{
        .pixels = pixels,
        .width = resized.width,
        .height = resized.height,
        .grid = grid,
        .token_count = token_count,
    };
}

const AlignMode = enum {
    down,
    up,
};

fn scaleAndAlign(
    width: u32,
    height: u32,
    target_pixels: u64,
    factor: u32,
    mode: AlignMode,
) !SmartResizeResult {
    const current_pixels = try checkedPixels(width, height);
    if (current_pixels == 0 or target_pixels == 0) return error.InvalidArgument;

    const scale = std.math.sqrt(@as(f64, @floatFromInt(target_pixels)) / @as(f64, @floatFromInt(current_pixels)));
    const raw_w = @max(@as(u32, 1), @as(u32, @intFromFloat(@round(@as(f64, @floatFromInt(width)) * scale))));
    const raw_h = @max(@as(u32, 1), @as(u32, @intFromFloat(@round(@as(f64, @floatFromInt(height)) * scale))));

    var out_w = switch (mode) {
        .down => alignDown(raw_w, factor),
        .up => alignUp(raw_w, factor),
    };
    var out_h = switch (mode) {
        .down => alignDown(raw_h, factor),
        .up => alignUp(raw_h, factor),
    };

    if (mode == .down) {
        while (try checkedPixels(out_w, out_h) > target_pixels) {
            if (out_w >= out_h and out_w > factor) {
                out_w -= factor;
            } else if (out_h > factor) {
                out_h -= factor;
            } else {
                break;
            }
        }
    } else {
        while (try checkedPixels(out_w, out_h) < target_pixels) {
            if (out_w <= out_h) {
                out_w += factor;
            } else {
                out_h += factor;
            }
        }
    }

    return .{ .width = out_w, .height = out_h };
}

fn alignNearest(value: u32, factor: u32) u32 {
    if (factor <= 1) return value;
    const down = alignDown(value, factor);
    const up = alignUp(value, factor);
    const d_down = if (down > value) down - value else value - down;
    const d_up = if (up > value) up - value else value - up;
    return if (d_down < d_up) down else up;
}

fn alignDown(value: u32, factor: u32) u32 {
    if (factor <= 1) return value;
    const out = (value / factor) * factor;
    return if (out == 0) factor else out;
}

fn alignUp(value: u32, factor: u32) u32 {
    if (factor <= 1) return value;
    const q = (value + factor - 1) / factor;
    return q * factor;
}

fn checkedPixels(width: u32, height: u32) !u64 {
    return std.math.mul(u64, @as(u64, width), @as(u64, height));
}

fn normalizeChannel(value: u8, channel: u8, mode: VisionNormalize) f32 {
    const v = @as(f32, @floatFromInt(value));
    return switch (mode) {
        .none => v,
        .zero_to_one => v / 255.0,
        .minus_one_to_one => v / 127.5 - 1.0,
        .imagenet => blk: {
            const n = v / 255.0;
            const mean: f32 = switch (channel) {
                0 => 0.485,
                1 => 0.456,
                else => 0.406,
            };
            const stddev: f32 = switch (channel) {
                0 => 0.229,
                1 => 0.224,
                else => 0.225,
            };
            break :blk (n - mean) / stddev;
        },
    };
}
