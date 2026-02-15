const std = @import("std");
const main = @import("main");
const capi = main.capi;

const png_red = @embedFile("corpus/1x1_red.png");
const webp_red = @embedFile("corpus/1x1_red.webp");

test "capi file inspect image returns metadata" {
    var info = std.mem.zeroes(capi.TaluFileInfo);
    const rc = capi.talu_file_inspect(png_red.ptr, png_red.len, &info);
    try std.testing.expectEqual(@as(i32, 0), rc);
    defer capi.talu_file_info_free(&info);

    try std.testing.expectEqual(@as(c_int, 1), info.kind);
    try std.testing.expectEqual(@as(c_int, 2), info.image_format); // png
    try std.testing.expectEqual(@as(u32, 1), info.width);
    try std.testing.expectEqual(@as(u32, 1), info.height);
    try std.testing.expect(info.mime_len > 0);
    try std.testing.expect(info.description_len > 0);
}

test "capi file inspect unknown bytes still reports mime/description" {
    const random = "not an image";

    var info = std.mem.zeroes(capi.TaluFileInfo);
    const rc = capi.talu_file_inspect(random.ptr, random.len, &info);
    try std.testing.expectEqual(@as(i32, 0), rc);
    defer capi.talu_file_info_free(&info);

    try std.testing.expectEqual(@as(c_int, 0), info.kind);
    try std.testing.expectEqual(@as(c_int, 0), info.image_format);
    try std.testing.expect(info.mime_len > 0 or info.description_len > 0);
}

test "capi file transform resize webp to png" {
    var opts = std.mem.zeroes(capi.TaluFileTransformOptions);
    opts.resize_enabled = 1;
    opts.out_w = 2;
    opts.out_h = 2;
    opts.fit_mode = 1; // contain
    opts.filter = 2; // bicubic
    opts.output_format = 2; // png
    opts.jpeg_quality = 90;

    var out_bytes: ?[*]u8 = null;
    var out_len: usize = 0;
    var out_info = std.mem.zeroes(capi.TaluFileInfo);

    const rc = capi.talu_file_transform(
        webp_red.ptr,
        webp_red.len,
        &opts,
        &out_bytes,
        &out_len,
        &out_info,
    );
    try std.testing.expectEqual(@as(i32, 0), rc);
    defer capi.talu_file_bytes_free(out_bytes, out_len);
    defer capi.talu_file_info_free(&out_info);

    const ptr = out_bytes orelse return error.TestUnexpectedResult;
    try std.testing.expect(out_len > 8);
    try std.testing.expect(ptr[0] == 0x89 and ptr[1] == 0x50 and ptr[2] == 0x4E and ptr[3] == 0x47);
    try std.testing.expectEqual(@as(c_int, 1), out_info.kind);
    try std.testing.expectEqual(@as(c_int, 2), out_info.image_format); // png
    try std.testing.expectEqual(@as(u32, 2), out_info.width);
    try std.testing.expectEqual(@as(u32, 2), out_info.height);
}
