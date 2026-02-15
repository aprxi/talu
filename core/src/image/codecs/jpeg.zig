//! JPEG image decoder and encoder using libturbojpeg.

const std = @import("std");
const pixel = @import("../pixel.zig");
const limits_mod = @import("../limits.zig");
const exif = @import("../exif.zig");

const tjhandle = ?*anyopaque;

extern fn tjInitDecompress() tjhandle;
extern fn tjInitCompress() tjhandle;
extern fn tjDestroy(handle: tjhandle) c_int;
extern fn tjDecompressHeader3(
    handle: tjhandle,
    jpeg_buf: [*]const u8,
    jpeg_size: c_ulong,
    width: *c_int,
    height: *c_int,
    jpeg_subsamp: *c_int,
    jpeg_colorspace: *c_int,
) c_int;
extern fn tjDecompress2(
    handle: tjhandle,
    jpeg_buf: [*]const u8,
    jpeg_size: c_ulong,
    dst_buf: [*]u8,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
    flags: c_int,
) c_int;
extern fn tjCompress2(
    handle: tjhandle,
    src_buf: [*]const u8,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
    jpeg_buf: *[*c]u8,
    jpeg_size: *c_ulong,
    jpeg_subsamp: c_int,
    jpeg_qual: c_int,
    flags: c_int,
) c_int;
extern fn tjFree(buffer: [*c]u8) void;

const TJPF_RGB: c_int = 0;
const TJPF_GRAY: c_int = 6;
const TJPF_RGBA: c_int = 7;
const TJSAMP_420: c_int = 2;

pub fn decode(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    lim: limits_mod.Limits,
    apply_orientation: bool,
) !pixel.Image {
    const handle = tjInitDecompress() orelse return error.JpegInitFailed;
    defer _ = tjDestroy(handle);

    var w: c_int = 0;
    var h: c_int = 0;
    var subsamp: c_int = 0;
    var colorspace: c_int = 0;

    if (tjDecompressHeader3(
        handle,
        bytes.ptr,
        @intCast(bytes.len),
        &w,
        &h,
        &subsamp,
        &colorspace,
    ) != 0) {
        return error.JpegHeaderFailed;
    }

    if (w <= 0 or h <= 0) return error.InvalidImageDimensions;

    const width: u32 = @intCast(w);
    const height: u32 = @intCast(h);

    const out_len = try lim.checkedOutputSize(width, height, 3);
    const out = try allocator.alloc(u8, out_len);
    errdefer allocator.free(out);

    if (tjDecompress2(
        handle,
        bytes.ptr,
        @intCast(bytes.len),
        out.ptr,
        @intCast(width),
        0,
        @intCast(height),
        TJPF_RGB,
        0,
    ) != 0) {
        return error.JpegDecodeFailed;
    }

    var img: pixel.Image = .{
        .width = width,
        .height = height,
        .stride = width * 3,
        .format = .rgb8,
        .data = out,
    };

    if (apply_orientation) {
        const orientation = exif.parseJpegOrientation(bytes);
        img = try exif.applyOrientation(allocator, img, orientation);
    }

    return img;
}

pub fn encode(
    allocator: std.mem.Allocator,
    img: pixel.Image,
    quality: u8,
) ![]u8 {
    const handle = tjInitCompress() orelse return error.JpegInitFailed;
    defer _ = tjDestroy(handle);

    const pixel_format: c_int = switch (img.format) {
        .gray8 => TJPF_GRAY,
        .rgb8 => TJPF_RGB,
        .rgba8 => TJPF_RGBA,
    };

    var jpeg_buf: [*c]u8 = null;
    var jpeg_size: c_ulong = 0;

    if (tjCompress2(
        handle,
        img.data.ptr,
        @intCast(img.width),
        @intCast(img.stride),
        @intCast(img.height),
        pixel_format,
        &jpeg_buf,
        &jpeg_size,
        TJSAMP_420,
        @intCast(quality),
        0,
    ) != 0) {
        return error.JpegEncodeFailed;
    }
    defer tjFree(jpeg_buf);

    const out_len = std.math.cast(usize, jpeg_size) orelse return error.ImageOutputTooLarge;
    const out = try allocator.alloc(u8, out_len);
    @memcpy(out, jpeg_buf[0..out_len]);
    return out;
}

test "decode roundtrips an encoded JPEG" {
    // Encode a synthetic 1x1 red pixel, then decode it back.
    var data = [_]u8{ 200, 100, 50 };
    const src: pixel.Image = .{ .width = 1, .height = 1, .stride = 3, .format = .rgb8, .data = &data };
    const jpeg_bytes = try encode(std.testing.allocator, src, 95);
    defer std.testing.allocator.free(jpeg_bytes);

    var img = try decode(std.testing.allocator, jpeg_bytes, .{}, false);
    defer img.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 1), img.width);
    try std.testing.expectEqual(@as(u32, 1), img.height);
    try std.testing.expectEqual(pixel.PixelFormat.rgb8, img.format);
}

test "decode rejects invalid JPEG data" {
    var garbage = [_]u8{ 0x00, 0x01, 0x02, 0x03 };
    try std.testing.expectError(error.JpegHeaderFailed, decode(std.testing.allocator, &garbage, .{}, false));
}

test "encode produces JPEG with SOI marker" {
    var data = [_]u8{ 200, 100, 50 };
    const img: pixel.Image = .{ .width = 1, .height = 1, .stride = 3, .format = .rgb8, .data = &data };
    const jpeg_bytes = try encode(std.testing.allocator, img, 90);
    defer std.testing.allocator.free(jpeg_bytes);
    try std.testing.expect(jpeg_bytes.len >= 3);
    try std.testing.expectEqual(@as(u8, 0xFF), jpeg_bytes[0]);
    try std.testing.expectEqual(@as(u8, 0xD8), jpeg_bytes[1]);
}
