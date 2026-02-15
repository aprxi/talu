const std = @import("std");

pub const Limits = struct {
    max_input_bytes: usize = 256 * 1024 * 1024,
    max_dimension: u32 = 32768,
    max_pixels: u64 = 64 * 1024 * 1024,
    max_output_bytes: usize = 512 * 1024 * 1024,

    pub fn validateInput(self: Limits, n: usize) !void {
        if (n > self.max_input_bytes) return error.ImageInputTooLarge;
    }

    pub fn validateDims(self: Limits, w: u32, h: u32) !void {
        if (w == 0 or h == 0) return error.InvalidImageDimensions;
        if (w > self.max_dimension or h > self.max_dimension) {
            return error.ImageDimensionExceeded;
        }
        const pixels = @as(u64, w) * @as(u64, h);
        if (pixels > self.max_pixels) return error.ImagePixelCountExceeded;
    }

    pub fn validateOutputBytes(self: Limits, n: usize) !void {
        if (n > self.max_output_bytes) return error.ImageOutputTooLarge;
    }

    pub fn checkedOutputSize(self: Limits, w: u32, h: u32, bpp: u32) !usize {
        try self.validateDims(w, h);
        const total_u64 = try std.math.mul(u64, @as(u64, w), @as(u64, h));
        const bytes_u64 = try std.math.mul(u64, total_u64, @as(u64, bpp));
        const bytes = std.math.cast(usize, bytes_u64) orelse return error.ImageOutputTooLarge;
        try self.validateOutputBytes(bytes);
        return bytes;
    }
};

pub const default_limits = Limits{};
