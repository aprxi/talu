//! WebP image decoder using libwebp.

const std = @import("std");
const pixel = @import("../pixel.zig");
const limits_mod = @import("../limits.zig");

const c = @cImport({
    @cInclude("webp/decode.h");
});

pub fn decode(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    lim: limits_mod.Limits,
) !pixel.Image {
    // Validate RIFF/WEBP magic before calling C library — WebPGetInfo is not
    // safe with arbitrary input and can read out of bounds on non-WebP data.
    if (!isValidWebpContainer(bytes)) return error.WebpHeaderFailed;

    var w: c_int = 0;
    var h: c_int = 0;
    if (c.WebPGetInfo(bytes.ptr, bytes.len, &w, &h) == 0) return error.WebpHeaderFailed;
    if (w <= 0 or h <= 0) return error.InvalidImageDimensions;

    const width: u32 = @intCast(w);
    const height: u32 = @intCast(h);
    const out_len = try lim.checkedOutputSize(width, height, 4);

    const out = try allocator.alloc(u8, out_len);
    errdefer allocator.free(out);

    if (c.WebPDecodeRGBAInto(bytes.ptr, bytes.len, out.ptr, out_len, @intCast(width * 4)) == null) {
        return error.WebpDecodeFailed;
    }

    return .{
        .width = width,
        .height = height,
        .stride = width * 4,
        .format = .rgba8,
        .data = out,
    };
}

/// RIFF container: 4 "RIFF" + 4 file-size + 4 "WEBP" + at least 8 for a VP8 chunk header.
const MIN_WEBP_SIZE = 20;

fn isValidWebpContainer(bytes: []const u8) bool {
    if (bytes.len < MIN_WEBP_SIZE) return false;
    if (!std.mem.eql(u8, bytes[0..4], "RIFF")) return false;
    if (!std.mem.eql(u8, bytes[8..12], "WEBP")) return false;

    // RIFF file-size field (little-endian u32) must not exceed actual data.
    const riff_size = std.mem.readInt(u32, bytes[4..8], .little);
    if (@as(u64, riff_size) + 8 > bytes.len) return false;

    return true;
}

test "decode rejects non-WebP data" {
    var garbage = [_]u8{ 0x00, 0x01, 0x02, 0x03 };
    try std.testing.expectError(error.WebpHeaderFailed, decode(std.testing.allocator, &garbage, .{}));
}

test "decode rejects truncated WebP container" {
    // Valid RIFF+WEBP magic but riff_size claims more data than available.
    var buf = [_]u8{0} ** 20;
    @memcpy(buf[0..4], "RIFF");
    std.mem.writeInt(u32, buf[4..8], 100, .little); // claims 108 bytes total
    @memcpy(buf[8..12], "WEBP");
    try std.testing.expectError(error.WebpHeaderFailed, decode(std.testing.allocator, &buf, .{}));
}

test "decode rejects data too short for WebP" {
    var short = [_]u8{ 'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'E', 'B', 'P' };
    try std.testing.expectError(error.WebpHeaderFailed, decode(std.testing.allocator, &short, .{}));
}
