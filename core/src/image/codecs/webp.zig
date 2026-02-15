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
