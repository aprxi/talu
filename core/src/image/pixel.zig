const std = @import("std");

pub const PixelFormat = enum(u8) {
    gray8,
    rgb8,
    rgba8,
};

pub const ColorPolicy = enum(u8) {
    assume_srgb,
};

pub const Rgb8 = struct {
    r: u8,
    g: u8,
    b: u8,
};

pub fn bytesPerPixel(format: PixelFormat) u32 {
    return switch (format) {
        .gray8 => 1,
        .rgb8 => 3,
        .rgba8 => 4,
    };
}

pub const Image = struct {
    width: u32,
    height: u32,
    stride: u32,
    format: PixelFormat,
    data: []u8,

    pub fn bytesPerPixel(self: Image) u32 {
        return switch (self.format) {
            .gray8 => 1,
            .rgb8 => 3,
            .rgba8 => 4,
        };
    }

    pub fn requiredLen(self: Image) !usize {
        const rows = @as(u64, self.height);
        const row_stride = @as(u64, self.stride);
        const total = try std.math.mul(u64, rows, row_stride);
        return std.math.cast(usize, total) orelse error.ImageOutputTooLarge;
    }

    pub fn deinit(self: *Image, allocator: std.mem.Allocator) void {
        if (self.data.len != 0) allocator.free(self.data);
        self.* = .{
            .width = 0,
            .height = 0,
            .stride = 0,
            .format = .rgb8,
            .data = &.{},
        };
    }
};
