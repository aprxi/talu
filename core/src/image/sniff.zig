const std = @import("std");

pub const Format = enum {
    jpeg,
    png,
    webp,
};

const MAGIC_MIME_TYPE: c_int = 0x0000010;

extern fn magic_open(flags: c_int) ?*anyopaque;
extern fn magic_close(cookie: ?*anyopaque) void;
extern fn magic_load_buffers(cookie: ?*anyopaque, buffers: [*]?*anyopaque, sizes: [*]usize, n: usize) c_int;
extern fn magic_buffer(cookie: ?*anyopaque, buffer: ?*const anyopaque, length: usize) ?[*:0]const u8;

pub fn detect(bytes: []const u8) ?Format {
    if (isJpeg(bytes)) return .jpeg;
    if (isPng(bytes)) return .png;
    if (isWebp(bytes)) return .webp;
    return null;
}

pub fn detectWithFallback(bytes: []const u8) ?Format {
    if (detect(bytes)) |fmt| return fmt;
    return detectWithLibmagic(bytes);
}

fn isJpeg(bytes: []const u8) bool {
    return bytes.len >= 3 and bytes[0] == 0xFF and bytes[1] == 0xD8 and bytes[2] == 0xFF;
}

fn isPng(bytes: []const u8) bool {
    return bytes.len >= 8 and std.mem.eql(u8, bytes[0..8], &.{ 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A });
}

fn isWebp(bytes: []const u8) bool {
    return bytes.len >= 12 and std.mem.eql(u8, bytes[0..4], "RIFF") and std.mem.eql(u8, bytes[8..12], "WEBP");
}

test "detect recognizes JPEG magic bytes" {
    const jpeg = [_]u8{ 0xFF, 0xD8, 0xFF, 0xE0 };
    try std.testing.expectEqual(Format.jpeg, detect(&jpeg).?);
}

test "detect recognizes PNG magic bytes" {
    const png = [_]u8{ 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };
    try std.testing.expectEqual(Format.png, detect(&png).?);
}

test "detect recognizes WebP magic bytes" {
    const webp = [_]u8{ 'R', 'I', 'F', 'F', 0x00, 0x00, 0x00, 0x00, 'W', 'E', 'B', 'P' };
    try std.testing.expectEqual(Format.webp, detect(&webp).?);
}

test "detect returns null for unknown bytes" {
    try std.testing.expectEqual(@as(?Format, null), detect("not an image"));
}

test "detect returns null for empty input" {
    try std.testing.expectEqual(@as(?Format, null), detect(""));
}

test "detect returns null for truncated JPEG" {
    try std.testing.expectEqual(@as(?Format, null), detect(&[_]u8{ 0xFF, 0xD8 }));
}

test "detectWithFallback recognizes JPEG via magic bytes" {
    const jpeg = [_]u8{ 0xFF, 0xD8, 0xFF, 0xE0 };
    try std.testing.expectEqual(Format.jpeg, detectWithFallback(&jpeg).?);
}

test "detectWithFallback returns null for non-image data" {
    try std.testing.expectEqual(@as(?Format, null), detectWithFallback("Hello, world!"));
}

test "detectWithFallback returns null for empty input" {
    try std.testing.expectEqual(@as(?Format, null), detectWithFallback(""));
}

fn detectWithLibmagic(bytes: []const u8) ?Format {
    if (bytes.len == 0) return null;

    const cookie = magic_open(MAGIC_MIME_TYPE) orelse return null;
    defer magic_close(cookie);

    const magic_db = @import("magic_db").data;
    if (magic_db.len == 0) return null;

    const db_ptr: ?*anyopaque = @ptrCast(@constCast(magic_db.ptr));
    var db_ptrs = [_]?*anyopaque{db_ptr};
    var db_sizes = [_]usize{magic_db.len};

    if (magic_load_buffers(cookie, &db_ptrs, &db_sizes, 1) != 0) return null;

    const mime_cstr = magic_buffer(cookie, bytes.ptr, bytes.len) orelse return null;
    const mime = std.mem.sliceTo(mime_cstr, 0);

    if (std.mem.startsWith(u8, mime, "image/jpeg")) return .jpeg;
    if (std.mem.startsWith(u8, mime, "image/png")) return .png;
    if (std.mem.startsWith(u8, mime, "image/webp")) return .webp;
    return null;
}
