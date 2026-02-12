//! Session ID generation utilities.

const std = @import("std");

/// Generate a UUIDv4 session identifier.
/// Caller owns returned memory.
pub fn generateSessionId(allocator: std.mem.Allocator) ![]const u8 {
    var uuid_bytes: [16]u8 = undefined;
    std.crypto.random.bytes(&uuid_bytes);

    // Set version (4) and variant (RFC 4122)
    uuid_bytes[6] = (uuid_bytes[6] & 0x0f) | 0x40;
    uuid_bytes[8] = (uuid_bytes[8] & 0x3f) | 0x80;

    var out = try allocator.alloc(u8, 36);
    const hex = "0123456789abcdef";

    var idx: usize = 0;
    for (uuid_bytes, 0..) |byte, i| {
        if (i == 4 or i == 6 or i == 8 or i == 10) {
            out[idx] = '-';
            idx += 1;
        }
        out[idx] = hex[(byte >> 4) & 0x0f];
        out[idx + 1] = hex[byte & 0x0f];
        idx += 2;
    }

    return out;
}

test "generateSessionId returns UUIDv4 format" {
    const allocator = std.testing.allocator;
    const id = try generateSessionId(allocator);
    defer allocator.free(id);

    try std.testing.expectEqual(@as(usize, 36), id.len);
    try std.testing.expectEqual(@as(u8, '-'), id[8]);
    try std.testing.expectEqual(@as(u8, '-'), id[13]);
    try std.testing.expectEqual(@as(u8, '-'), id[18]);
    try std.testing.expectEqual(@as(u8, '-'), id[23]);
}
