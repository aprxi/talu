const std = @import("std");
const pixel = @import("../pixel.zig");
const limits_mod = @import("../limits.zig");

const c = @cImport({
    @cInclude("spng.h");
});

pub fn decode(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    lim: limits_mod.Limits,
) !pixel.Image {
    const ctx = c.spng_ctx_new(0) orelse return error.PngInitFailed;
    defer c.spng_ctx_free(ctx);

    if (c.spng_set_png_buffer(ctx, bytes.ptr, bytes.len) != 0) return error.PngSetBufferFailed;

    var ihdr: c.struct_spng_ihdr = std.mem.zeroes(c.struct_spng_ihdr);
    if (c.spng_get_ihdr(ctx, &ihdr) != 0) return error.PngHeaderFailed;

    try lim.validateDims(ihdr.width, ihdr.height);

    var out_size: usize = 0;
    if (c.spng_decoded_image_size(ctx, c.SPNG_FMT_RGBA8, &out_size) != 0) return error.PngSizeFailed;
    try lim.validateOutputBytes(out_size);

    const out = try allocator.alloc(u8, out_size);
    errdefer allocator.free(out);

    if (c.spng_decode_image(ctx, out.ptr, out_size, c.SPNG_FMT_RGBA8, 0) != 0) return error.PngDecodeFailed;

    return .{
        .width = ihdr.width,
        .height = ihdr.height,
        .stride = ihdr.width * 4,
        .format = .rgba8,
        .data = out,
    };
}

pub fn encode(allocator: std.mem.Allocator, img: pixel.Image) ![]u8 {
    const color_type: u8 = switch (img.format) {
        .gray8 => c.SPNG_COLOR_TYPE_GRAYSCALE,
        .rgb8 => c.SPNG_COLOR_TYPE_TRUECOLOR,
        .rgba8 => c.SPNG_COLOR_TYPE_TRUECOLOR_ALPHA,
    };

    const expected_row = img.width * pixel.bytesPerPixel(img.format);
    if (img.stride != expected_row) return error.UnsupportedStride;

    const ctx = c.spng_ctx_new(c.SPNG_CTX_ENCODER) orelse return error.PngInitFailed;
    defer c.spng_ctx_free(ctx);

    if (c.spng_set_option(ctx, c.SPNG_ENCODE_TO_BUFFER, 1) != 0) return error.PngEncodeFailed;

    var ihdr: c.struct_spng_ihdr = std.mem.zeroes(c.struct_spng_ihdr);
    ihdr.width = img.width;
    ihdr.height = img.height;
    ihdr.bit_depth = 8;
    ihdr.color_type = color_type;
    ihdr.compression_method = 0;
    ihdr.filter_method = 0;
    ihdr.interlace_method = c.SPNG_INTERLACE_NONE;

    if (c.spng_set_ihdr(ctx, &ihdr) != 0) return error.PngEncodeFailed;

    if (c.spng_encode_image(ctx, img.data.ptr, img.data.len, c.SPNG_FMT_RAW, c.SPNG_ENCODE_FINALIZE) != 0) {
        return error.PngEncodeFailed;
    }

    var png_size: usize = 0;
    var enc_err: c_int = 0;
    const png_ptr = c.spng_get_png_buffer(ctx, &png_size, &enc_err);
    if (png_ptr == null or enc_err != 0) return error.PngEncodeFailed;
    defer std.c.free(png_ptr);

    const out = try allocator.alloc(u8, png_size);
    @memcpy(out, @as([*]u8, @ptrCast(png_ptr.?))[0..png_size]);
    return out;
}
