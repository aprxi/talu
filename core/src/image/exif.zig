const std = @import("std");
const pixel = @import("pixel.zig");

pub const Orientation = enum(u8) {
    normal = 1,
    mirror_h = 2,
    rotate_180 = 3,
    mirror_v = 4,
    transpose = 5,
    rotate_90_cw = 6,
    transverse = 7,
    rotate_270_cw = 8,
};

pub fn parseJpegOrientation(bytes: []const u8) Orientation {
    if (bytes.len < 4) return .normal;
    if (!(bytes[0] == 0xFF and bytes[1] == 0xD8)) return .normal;

    var i: usize = 2;
    while (i + 3 < bytes.len) {
        if (bytes[i] != 0xFF) {
            i += 1;
            continue;
        }

        var marker_idx = i + 1;
        while (marker_idx < bytes.len and bytes[marker_idx] == 0xFF) {
            marker_idx += 1;
        }
        if (marker_idx >= bytes.len) break;

        const marker = bytes[marker_idx];
        if (marker == 0xD9 or marker == 0xDA) break;
        if (marker_idx + 2 >= bytes.len) break;

        const seg_len = readBeU16(bytes, marker_idx + 1) orelse break;
        if (seg_len < 2) break;
        const seg_data_start = marker_idx + 3;
        const seg_data_len = seg_len - 2;
        if (seg_data_start + seg_data_len > bytes.len) break;

        if (marker == 0xE1) {
            const seg = bytes[seg_data_start .. seg_data_start + seg_data_len];
            if (seg.len >= 6 and std.mem.eql(u8, seg[0..6], "Exif\x00\x00")) {
                return parseExifBlock(seg[6..]);
            }
        }

        i = seg_data_start + seg_data_len;
    }

    return .normal;
}

pub fn applyOrientation(allocator: std.mem.Allocator, img: pixel.Image, orientation: Orientation) !pixel.Image {
    if (orientation == .normal) return img;

    const bpp_u32 = pixel.bytesPerPixel(img.format);
    const bpp = @as(usize, bpp_u32);

    const dims = orientedDims(img.width, img.height, orientation);
    const out_w = dims.w;
    const out_h = dims.h;
    const out_stride = out_w * bpp_u32;
    const out_len_u64 = @as(u64, out_stride) * @as(u64, out_h);
    const out_len = std.math.cast(usize, out_len_u64) orelse return error.ImageOutputTooLarge;

    const out = try allocator.alloc(u8, out_len);
    errdefer allocator.free(out);

    var y: u32 = 0;
    while (y < out_h) : (y += 1) {
        var x: u32 = 0;
        while (x < out_w) : (x += 1) {
            const src = mapDstToSrc(orientation, x, y, img.width, img.height);
            const src_off = @as(usize, src.y) * @as(usize, img.stride) + @as(usize, src.x) * bpp;
            const dst_off = @as(usize, y) * @as(usize, out_stride) + @as(usize, x) * bpp;
            @memcpy(out[dst_off .. dst_off + bpp], img.data[src_off .. src_off + bpp]);
        }
    }

    allocator.free(img.data);
    return .{
        .width = out_w,
        .height = out_h,
        .stride = out_stride,
        .format = img.format,
        .data = out,
    };
}

fn parseExifBlock(tiff: []const u8) Orientation {
    if (tiff.len < 8) return .normal;

    const endian = parseEndian(tiff[0..2]) orelse return .normal;
    if (readU16(tiff, 2, endian) != 42) return .normal;

    const ifd0_off = readU32(tiff, 4, endian);
    if (ifd0_off + 2 > tiff.len) return .normal;

    const entry_count = readU16(tiff, ifd0_off, endian);
    var entry_off: usize = ifd0_off + 2;

    var idx: u16 = 0;
    while (idx < entry_count and entry_off + 12 <= tiff.len) : ({
        idx += 1;
        entry_off += 12;
    }) {
        const tag = readU16(tiff, entry_off, endian);
        if (tag != 0x0112) continue;

        const typ = readU16(tiff, entry_off + 2, endian);
        const count = readU32(tiff, entry_off + 4, endian);
        if (typ != 3 or count < 1) return .normal;

        const raw_value = readU16(tiff, entry_off + 8, endian);
        return orientationFromValue(raw_value);
    }

    return .normal;
}

fn orientationFromValue(v: u16) Orientation {
    return switch (v) {
        1 => .normal,
        2 => .mirror_h,
        3 => .rotate_180,
        4 => .mirror_v,
        5 => .transpose,
        6 => .rotate_90_cw,
        7 => .transverse,
        8 => .rotate_270_cw,
        else => .normal,
    };
}

const Endian = enum {
    little,
    big,
};

fn parseEndian(two: []const u8) ?Endian {
    if (two.len != 2) return null;
    if (two[0] == 'I' and two[1] == 'I') return .little;
    if (two[0] == 'M' and two[1] == 'M') return .big;
    return null;
}

fn readBeU16(bytes: []const u8, off: usize) ?u16 {
    if (off + 2 > bytes.len) return null;
    return (@as(u16, bytes[off]) << 8) | @as(u16, bytes[off + 1]);
}

fn readU16(bytes: []const u8, off: usize, endian: Endian) u16 {
    if (off + 2 > bytes.len) return 0;
    const buf: *const [2]u8 = @ptrCast(bytes[off .. off + 2].ptr);
    return switch (endian) {
        .little => std.mem.readInt(u16, buf, .little),
        .big => std.mem.readInt(u16, buf, .big),
    };
}

fn readU32(bytes: []const u8, off: usize, endian: Endian) u32 {
    if (off + 4 > bytes.len) return 0;
    const buf: *const [4]u8 = @ptrCast(bytes[off .. off + 4].ptr);
    return switch (endian) {
        .little => std.mem.readInt(u32, buf, .little),
        .big => std.mem.readInt(u32, buf, .big),
    };
}

fn orientedDims(w: u32, h: u32, orientation: Orientation) struct { w: u32, h: u32 } {
    return switch (orientation) {
        .normal, .mirror_h, .rotate_180, .mirror_v => .{ .w = w, .h = h },
        .transpose, .rotate_90_cw, .transverse, .rotate_270_cw => .{ .w = h, .h = w },
    };
}

fn mapDstToSrc(orientation: Orientation, x: u32, y: u32, src_w: u32, src_h: u32) struct { x: u32, y: u32 } {
    return switch (orientation) {
        .normal => .{ .x = x, .y = y },
        .mirror_h => .{ .x = src_w - 1 - x, .y = y },
        .rotate_180 => .{ .x = src_w - 1 - x, .y = src_h - 1 - y },
        .mirror_v => .{ .x = x, .y = src_h - 1 - y },
        .transpose => .{ .x = y, .y = x },
        .rotate_90_cw => .{ .x = y, .y = src_h - 1 - x },
        .transverse => .{ .x = src_w - 1 - y, .y = src_h - 1 - x },
        .rotate_270_cw => .{ .x = src_w - 1 - y, .y = x },
    };
}

// --- Test helpers ---

/// Build a minimal JPEG byte stream with an EXIF APP1 segment containing the
/// given orientation tag.  `endian` selects II (little) or MM (big).
fn buildExifJpeg(endian: Endian, orient_val: u8) [64]u8 {
    var buf = [_]u8{0} ** 64;
    // SOI
    buf[0] = 0xFF;
    buf[1] = 0xD8;
    // APP1 marker
    buf[2] = 0xFF;
    buf[3] = 0xE1;
    // Segment length (big-endian): covers rest of APP1 (58 bytes)
    buf[4] = 0x00;
    buf[5] = 58;
    // "Exif\0\0"
    @memcpy(buf[6..12], "Exif\x00\x00");
    // TIFF header starts at offset 12
    const tiff_base: usize = 12;
    switch (endian) {
        .little => {
            buf[tiff_base] = 'I';
            buf[tiff_base + 1] = 'I';
        },
        .big => {
            buf[tiff_base] = 'M';
            buf[tiff_base + 1] = 'M';
        },
    }
    // Magic 42
    writeU16(&buf, tiff_base + 2, 42, endian);
    // IFD0 offset (8 = right after TIFF header)
    writeU32(&buf, tiff_base + 4, 8, endian);
    // IFD0 at tiff_base+8: 1 entry
    const ifd: usize = tiff_base + 8;
    writeU16(&buf, ifd, 1, endian);
    // Entry: tag=0x0112, type=3(SHORT), count=1, value=orient_val
    writeU16(&buf, ifd + 2, 0x0112, endian);
    writeU16(&buf, ifd + 4, 3, endian);
    writeU32(&buf, ifd + 6, 1, endian);
    writeU16(&buf, ifd + 10, orient_val, endian);
    return buf;
}

fn writeU16(buf: []u8, off: usize, val: u16, endian: Endian) void {
    const b: *[2]u8 = @ptrCast(buf[off .. off + 2].ptr);
    switch (endian) {
        .little => std.mem.writeInt(u16, b, val, .little),
        .big => std.mem.writeInt(u16, b, val, .big),
    }
}

fn writeU32(buf: []u8, off: usize, val: u32, endian: Endian) void {
    const b: *[4]u8 = @ptrCast(buf[off .. off + 4].ptr);
    switch (endian) {
        .little => std.mem.writeInt(u32, b, val, .little),
        .big => std.mem.writeInt(u32, b, val, .big),
    }
}

test "parseJpegOrientation reads orientation 6 little-endian" {
    const jpeg = buildExifJpeg(.little, 6);
    try std.testing.expectEqual(Orientation.rotate_90_cw, parseJpegOrientation(&jpeg));
}

test "parseJpegOrientation reads orientation 3 big-endian" {
    const jpeg = buildExifJpeg(.big, 3);
    try std.testing.expectEqual(Orientation.rotate_180, parseJpegOrientation(&jpeg));
}

test "parseJpegOrientation returns normal for all 8 values" {
    // Values 1..8 should each map to the corresponding enum variant.
    inline for (.{ 1, 2, 3, 4, 5, 6, 7, 8 }) |v| {
        const jpeg = buildExifJpeg(.little, v);
        const orient = parseJpegOrientation(&jpeg);
        try std.testing.expectEqual(@as(u8, v), @intFromEnum(orient));
    }
}

test "parseJpegOrientation returns normal for non-JPEG input" {
    try std.testing.expectEqual(Orientation.normal, parseJpegOrientation("not a jpeg"));
}

test "parseJpegOrientation returns normal for empty input" {
    try std.testing.expectEqual(Orientation.normal, parseJpegOrientation(""));
}

test "parseJpegOrientation returns normal for JPEG without EXIF" {
    // SOI + SOS (no APP1 segment)
    const jpeg = [_]u8{ 0xFF, 0xD8, 0xFF, 0xDA, 0x00, 0x02 };
    try std.testing.expectEqual(Orientation.normal, parseJpegOrientation(&jpeg));
}

test "applyOrientation normal returns input unchanged" {
    // 2x1 grayscale: [10, 20]
    var data = [_]u8{ 10, 20 };
    const img: pixel.Image = .{ .width = 2, .height = 1, .stride = 2, .format = .gray8, .data = &data };
    // .normal should return the original image without allocating.
    const result = try applyOrientation(std.testing.allocator, img, .normal);
    try std.testing.expectEqual(@as(u32, 2), result.width);
    try std.testing.expectEqual(@as(u32, 1), result.height);
    try std.testing.expectEqual(@as(u8, 10), result.data[0]);
}

test "applyOrientation rotate_180 reverses pixels" {
    // 2x1 grayscale: [10, 20] → [20, 10]
    const alloc = std.testing.allocator;
    const src = try alloc.alloc(u8, 2);
    src[0] = 10;
    src[1] = 20;
    var img: pixel.Image = .{ .width = 2, .height = 1, .stride = 2, .format = .gray8, .data = src };
    img = try applyOrientation(alloc, img, .rotate_180);
    defer alloc.free(img.data);
    try std.testing.expectEqual(@as(u8, 20), img.data[0]);
    try std.testing.expectEqual(@as(u8, 10), img.data[1]);
}

test "applyOrientation rotate_90_cw swaps dimensions" {
    // 2x3 grid → should become 3x2
    const alloc = std.testing.allocator;
    const src = try alloc.alloc(u8, 6);
    // Row 0: 1 2, Row 1: 3 4, Row 2: 5 6
    @memcpy(src, &[_]u8{ 1, 2, 3, 4, 5, 6 });
    var img: pixel.Image = .{ .width = 2, .height = 3, .stride = 2, .format = .gray8, .data = src };
    img = try applyOrientation(alloc, img, .rotate_90_cw);
    defer alloc.free(img.data);
    try std.testing.expectEqual(@as(u32, 3), img.width);
    try std.testing.expectEqual(@as(u32, 2), img.height);
    // Top-left of rotated image should be bottom-left of original (5)
    try std.testing.expectEqual(@as(u8, 5), img.data[0]);
}

test "applyOrientation mirror_h flips horizontally" {
    const alloc = std.testing.allocator;
    const src = try alloc.alloc(u8, 4);
    // 2x2: [1 2 / 3 4] → [2 1 / 4 3]
    @memcpy(src, &[_]u8{ 1, 2, 3, 4 });
    var img: pixel.Image = .{ .width = 2, .height = 2, .stride = 2, .format = .gray8, .data = src };
    img = try applyOrientation(alloc, img, .mirror_h);
    defer alloc.free(img.data);
    try std.testing.expectEqual(@as(u8, 2), img.data[0]);
    try std.testing.expectEqual(@as(u8, 1), img.data[1]);
    try std.testing.expectEqual(@as(u8, 4), img.data[2]);
    try std.testing.expectEqual(@as(u8, 3), img.data[3]);
}
