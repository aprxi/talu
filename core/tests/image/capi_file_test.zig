const std = @import("std");
const main = @import("main");
const capi = main.capi;

const png_red = @embedFile("corpus/1x1_red.png");
const webp_red = @embedFile("corpus/1x1_red.webp");
const pdf_1page = @embedFile("corpus/1x1_page.pdf");

test "capi file inspect image returns metadata" {
    var info = std.mem.zeroes(capi.TaluFileInfo);
    var img = std.mem.zeroes(capi.TaluImageInfo);
    const rc = capi.talu_file_inspect(png_red.ptr, png_red.len, &info, &img);
    try std.testing.expectEqual(@as(i32, 0), rc);
    defer capi.talu_file_info_free(&info);

    try std.testing.expectEqual(@as(c_int, 1), info.kind);
    try std.testing.expectEqual(@as(c_int, 2), img.format); // png
    try std.testing.expectEqual(@as(u32, 1), img.width);
    try std.testing.expectEqual(@as(u32, 1), img.height);
    try std.testing.expect(info.mime_len > 0);
    try std.testing.expect(info.description_len > 0);
}

test "capi file inspect unknown bytes still reports mime/description" {
    const random = "not an image";

    var info = std.mem.zeroes(capi.TaluFileInfo);
    var img = std.mem.zeroes(capi.TaluImageInfo);
    const rc = capi.talu_file_inspect(random.ptr, random.len, &info, &img);
    try std.testing.expectEqual(@as(i32, 0), rc);
    defer capi.talu_file_info_free(&info);

    // Plain ASCII bytes are classified as text (kind=5).
    try std.testing.expectEqual(@as(c_int, 5), info.kind);
    try std.testing.expectEqual(@as(c_int, 0), img.format);
    try std.testing.expect(info.mime_len > 0 or info.description_len > 0);
}

test "capi file inspect with null out_image succeeds" {
    var info = std.mem.zeroes(capi.TaluFileInfo);
    const rc = capi.talu_file_inspect(png_red.ptr, png_red.len, &info, null);
    try std.testing.expectEqual(@as(i32, 0), rc);
    defer capi.talu_file_info_free(&info);

    try std.testing.expectEqual(@as(c_int, 1), info.kind);
    try std.testing.expect(info.mime_len > 0);
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
    var out_image = std.mem.zeroes(capi.TaluImageInfo);

    const rc = capi.talu_file_transform(
        webp_red.ptr,
        webp_red.len,
        &opts,
        &out_bytes,
        &out_len,
        &out_image,
    );
    try std.testing.expectEqual(@as(i32, 0), rc);
    defer capi.talu_file_bytes_free(out_bytes, out_len);

    const ptr = out_bytes orelse return error.TestUnexpectedResult;
    try std.testing.expect(out_len > 8);
    try std.testing.expect(ptr[0] == 0x89 and ptr[1] == 0x50 and ptr[2] == 0x4E and ptr[3] == 0x47);
    try std.testing.expectEqual(@as(c_int, 2), out_image.format); // png
    try std.testing.expectEqual(@as(u32, 2), out_image.width);
    try std.testing.expectEqual(@as(u32, 2), out_image.height);
}

test "talu_file_inspect recognizes pdf as document" {
    var info = std.mem.zeroes(capi.TaluFileInfo);
    var img = std.mem.zeroes(capi.TaluImageInfo);
    const rc = capi.talu_file_inspect(pdf_1page.ptr, pdf_1page.len, &info, &img);
    try std.testing.expectEqual(@as(i32, 0), rc);
    defer capi.talu_file_info_free(&info);

    // PDF is kind=2 (document) — a rendered format with no intrinsic pixel
    // dimensions. TaluImageInfo stays zeroed.
    try std.testing.expectEqual(@as(c_int, 2), info.kind);
    try std.testing.expectEqual(@as(c_int, 0), img.format);
    try std.testing.expectEqual(@as(u32, 0), img.width);
    try std.testing.expectEqual(@as(u32, 0), img.height);
}

test "talu_pdf_page_count returns page count" {
    var count: u32 = 0;
    const rc = capi.talu_pdf_page_count(pdf_1page.ptr, pdf_1page.len, &count);
    try std.testing.expectEqual(@as(i32, 0), rc);
    try std.testing.expectEqual(@as(u32, 1), count);
}

test "talu_pdf_page_count rejects null bytes" {
    var count: u32 = 0;
    const rc = capi.talu_pdf_page_count(null, 0, &count);
    try std.testing.expect(rc != 0);
}

test "talu_pdf_render_page renders first page" {
    var img = std.mem.zeroes(capi.TaluImage);
    const rc = capi.talu_pdf_render_page(pdf_1page.ptr, pdf_1page.len, 0, 72, &img);
    try std.testing.expectEqual(@as(i32, 0), rc);
    defer capi.talu_image_free(&img);

    // 72pt page at 72 DPI → 72px
    try std.testing.expectEqual(@as(u32, 72), img.width);
    try std.testing.expectEqual(@as(u32, 72), img.height);
    try std.testing.expectEqual(@as(c_int, 2), img.format); // rgba8 = 2
    try std.testing.expect(img.data != null);
    try std.testing.expect(img.len > 0);
}

test "talu_pdf_render_page rejects null bytes" {
    var img = std.mem.zeroes(capi.TaluImage);
    const rc = capi.talu_pdf_render_page(null, 0, 0, 72, &img);
    try std.testing.expect(rc != 0);
}

test "talu_pdf_render_page rejects invalid page index" {
    var img = std.mem.zeroes(capi.TaluImage);
    const rc = capi.talu_pdf_render_page(pdf_1page.ptr, pdf_1page.len, 99, 72, &img);
    try std.testing.expect(rc != 0);
}
