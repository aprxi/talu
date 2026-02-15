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
