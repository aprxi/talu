const std = @import("std");
const main = @import("main");
const image = main.core.image;

const png_red = @embedFile("corpus/1x1_red.png");
const jpg_red = @embedFile("corpus/1x1_red.jpg");
const webp_red = @embedFile("corpus/1x1_red.webp");
const jpg_2x3 = @embedFile("corpus/2x3_blue.jpg");
const pdf_1page = @embedFile("corpus/1x1_page.pdf");

test "image.detectFormat recognizes jpeg/png/webp/pdf" {
    try std.testing.expectEqual(image.Format.png, image.detectFormat(png_red).?);
    try std.testing.expectEqual(image.Format.jpeg, image.detectFormat(jpg_red).?);
    try std.testing.expectEqual(image.Format.webp, image.detectFormat(webp_red).?);
    try std.testing.expectEqual(image.Format.pdf, image.detectFormat(pdf_1page).?);
}

test "image.decode decodes minimal jpeg/png/webp" {
    var png_img = try image.decode(std.testing.allocator, png_red, .{});
    defer png_img.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 1), png_img.width);
    try std.testing.expectEqual(@as(u32, 1), png_img.height);

    var jpg_img = try image.decode(std.testing.allocator, jpg_red, .{});
    defer jpg_img.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 1), jpg_img.width);
    try std.testing.expectEqual(@as(u32, 1), jpg_img.height);

    var webp_img = try image.decode(std.testing.allocator, webp_red, .{});
    defer webp_img.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 1), webp_img.width);
    try std.testing.expectEqual(@as(u32, 1), webp_img.height);
}

test "image.decode decodes minimal pdf" {
    var pdf_img = try image.decode(std.testing.allocator, pdf_1page, .{});
    defer pdf_img.deinit(std.testing.allocator);

    // 72pt page at default 150 DPI → ceil(72 * 150/72) = 150px
    try std.testing.expectEqual(@as(u32, 150), pdf_img.width);
    try std.testing.expectEqual(@as(u32, 150), pdf_img.height);
    try std.testing.expectEqual(image.PixelFormat.rgb8, pdf_img.format);
}

test "image.decode rejects unsupported format" {
    try std.testing.expectError(
        error.UnsupportedImageFormat,
        image.decode(std.testing.allocator, "not an image", .{}),
    );
}

test "image.decode applies JPEG EXIF orientation" {
    const oriented = try injectExifOrientation(std.testing.allocator, jpg_2x3, 6);
    defer std.testing.allocator.free(oriented);

    var out = try image.decode(std.testing.allocator, oriented, .{ .apply_orientation = true });
    defer out.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 3), out.width);
    try std.testing.expectEqual(@as(u32, 2), out.height);
}

test "image.decode enforces dimensions before allocation" {
    const huge = try patchJpegDimensions(std.testing.allocator, jpg_2x3, 50_000, 50_000);
    defer std.testing.allocator.free(huge);

    try std.testing.expectError(
        error.ImageDimensionExceeded,
        image.decode(std.testing.allocator, huge, .{
            .limits = .{
                .max_dimension = 4096,
                .max_pixels = 16 * 1024 * 1024,
                .max_input_bytes = 256 * 1024 * 1024,
                .max_output_bytes = 512 * 1024 * 1024,
            },
        }),
    );
}

fn injectExifOrientation(allocator: std.mem.Allocator, jpeg: []const u8, orientation: u16) ![]u8 {
    if (jpeg.len < 2 or !(jpeg[0] == 0xFF and jpeg[1] == 0xD8)) return error.InvalidArgument;

    var payload = [_]u8{
        // Exif header
        'E',  'x',  'i',  'f',  0x00, 0x00,
        // TIFF header (MM endian)
        'M',  'M',  0x00, 0x2A, 0x00, 0x00,
        0x00, 0x08,
        // IFD0 entry count: 1
        0x00, 0x01,
        // Tag 0x0112, type SHORT (3), count 1
        0x01, 0x12,
        0x00, 0x03, 0x00, 0x00, 0x00, 0x01,
        // Value (2 bytes) + padding
        0x00, 0x00, 0x00, 0x00,
        // Next IFD offset
        0x00, 0x00,
        0x00, 0x00,
    };
    payload[24] = @intCast((orientation >> 8) & 0xFF);
    payload[25] = @intCast(orientation & 0xFF);

    const app1_len: u16 = @intCast(payload.len + 2);
    const seg = [_]u8{
        0xFF,                             0xE1,
        @intCast((app1_len >> 8) & 0xFF), @intCast(app1_len & 0xFF),
    };

    const out = try allocator.alloc(u8, jpeg.len + seg.len + payload.len);
    @memcpy(out[0..2], jpeg[0..2]);
    @memcpy(out[2 .. 2 + seg.len], &seg);
    @memcpy(out[2 + seg.len .. 2 + seg.len + payload.len], &payload);
    @memcpy(out[2 + seg.len + payload.len ..], jpeg[2..]);
    return out;
}

fn patchJpegDimensions(allocator: std.mem.Allocator, jpeg: []const u8, w: u16, h: u16) ![]u8 {
    var out = try allocator.dupe(u8, jpeg);

    var i: usize = 2;
    while (i + 8 < out.len) {
        if (out[i] != 0xFF) {
            i += 1;
            continue;
        }
        const marker = out[i + 1];
        if (marker == 0xD9 or marker == 0xDA) break;
        if (i + 4 > out.len) break;

        const seg_len = (@as(u16, out[i + 2]) << 8) | out[i + 3];
        if (seg_len < 2) break;

        if (marker >= 0xC0 and marker <= 0xCF and marker != 0xC4 and marker != 0xC8 and marker != 0xCC) {
            out[i + 5] = @intCast((h >> 8) & 0xFF);
            out[i + 6] = @intCast(h & 0xFF);
            out[i + 7] = @intCast((w >> 8) & 0xFF);
            out[i + 8] = @intCast(w & 0xFF);
            return out;
        }

        i += 2 + seg_len;
    }

    allocator.free(out);
    return error.TestUnexpectedResult;
}
