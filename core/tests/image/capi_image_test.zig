const std = @import("std");
const main = @import("main");
const capi = main.capi;

const png_red = @embedFile("corpus/1x1_red.png");

test "capi image decode and free" {
    var out = std.mem.zeroes(capi.TaluImage);
    const rc = capi.talu_image_decode(png_red.ptr, png_red.len, null, &out);
    try std.testing.expectEqual(@as(i32, 0), rc);
    defer capi.talu_image_free(&out);

    try std.testing.expect(out.data != null);
    try std.testing.expectEqual(@as(u32, 1), out.width);
    try std.testing.expectEqual(@as(u32, 1), out.height);
    try std.testing.expectEqual(@as(c_int, 1), out.format); // rgb8
}

test "capi image encode PNG and free" {
    var decoded = std.mem.zeroes(capi.TaluImage);
    const decode_rc = capi.talu_image_decode(png_red.ptr, png_red.len, null, &decoded);
    try std.testing.expectEqual(@as(i32, 0), decode_rc);
    defer capi.talu_image_free(&decoded);

    var opts = std.mem.zeroes(capi.TaluImageEncodeOptions);
    opts.format = 1; // png
    opts.jpeg_quality = 85;

    var out_bytes: ?[*]u8 = null;
    var out_len: usize = 0;
    const encode_rc = capi.talu_image_encode(&decoded, &opts, &out_bytes, &out_len);
    try std.testing.expectEqual(@as(i32, 0), encode_rc);
    defer capi.talu_image_encode_free(out_bytes, out_len);

    const ptr = out_bytes orelse return error.TestUnexpectedResult;
    try std.testing.expect(out_len > 8);
    try std.testing.expect(ptr[0] == 0x89 and ptr[1] == 0x50 and ptr[2] == 0x4E and ptr[3] == 0x47);
}

test "capi image to_model_input produces f32 buffer" {
    var spec = std.mem.zeroes(capi.TaluModelInputSpec);
    spec.width = 1;
    spec.height = 1;
    spec.dtype = 1; // f32
    spec.layout = 0; // nhwc
    spec.normalize = 1; // zero_to_one
    spec.fit_mode = 1; // contain
    spec.filter = 2; // bicubic

    var out = std.mem.zeroes(capi.TaluModelBuffer);
    const rc = capi.talu_image_to_model_input(png_red.ptr, png_red.len, &spec, &out);
    try std.testing.expectEqual(@as(i32, 0), rc);
    defer capi.talu_model_buffer_free(&out);

    try std.testing.expect(out.data != null);
    try std.testing.expectEqual(@as(usize, 3 * @sizeOf(f32)), out.len);
    try std.testing.expectEqual(@as(u32, 1), out.width);
    try std.testing.expectEqual(@as(u32, 1), out.height);
}

test "capi image decode rejects null input" {
    var out = std.mem.zeroes(capi.TaluImage);
    const rc = capi.talu_image_decode(null, 0, null, &out);
    try std.testing.expect(rc != 0);
}
