//! KvBuf Writer - Serializes fields into a directory-last binary format.
//!
//! KvBuf (Key-Value Buffer) separates field values from structural metadata,
//! enabling zero-copy field access during reads.
//!
//! Binary layout:
//!   [ Values Area ]  [ Directory ]  [ Footer ]
//!
//! - Values Area: Raw concatenated field bytes (no length prefixes or type markers).
//! - Directory: Array of 10-byte entries (field_id:u16, offset:u32, length:u32).
//! - Footer: directory_offset:u32 + magic:u8 (5 bytes).
//!
//! Thread safety: NOT thread-safe (single-writer use).

const std = @import("std");

/// Magic byte identifying KvBuf format version 1.
pub const KVBUF_MAGIC: u8 = 0x01;

/// Size of each directory entry in bytes.
pub const ENTRY_SIZE: usize = 10;

/// Size of the footer in bytes (4-byte directory_offset + 1-byte magic).
pub const FOOTER_SIZE: usize = 5;

/// Pending field entry, accumulated during writing.
const PendingEntry = struct {
    field_id: u16,
    offset: u32,
    length: u32,
};

/// Builds a KvBuf blob from a sequence of field additions.
///
/// Usage:
///   var w = KvBufWriter.init();
///   try w.addString(1, "hello");
///   try w.addU64(2, 42);
///   const blob = try w.finish(allocator);
///   defer allocator.free(blob);
///
/// Caller owns the returned blob from finish().
pub const KvBufWriter = struct {
    values: std.ArrayListUnmanaged(u8),
    entries: std.ArrayListUnmanaged(PendingEntry),

    pub fn init() KvBufWriter {
        return .{
            .values = .{},
            .entries = .{},
        };
    }

    /// Release internal buffers without producing output.
    pub fn deinit(self: *KvBufWriter, allocator: std.mem.Allocator) void {
        self.values.deinit(allocator);
        self.entries.deinit(allocator);
    }

    /// Append a raw byte slice as a field value.
    pub fn addBytes(self: *KvBufWriter, allocator: std.mem.Allocator, field_id: u16, data: []const u8) !void {
        if (data.len > std.math.maxInt(u32)) return error.ValueTooLarge;
        const offset: u32 = @intCast(self.values.items.len);
        try self.values.appendSlice(allocator, data);
        try self.entries.append(allocator, .{
            .field_id = field_id,
            .offset = offset,
            .length = @intCast(data.len),
        });
    }

    /// Append a string field.
    pub fn addString(self: *KvBufWriter, allocator: std.mem.Allocator, field_id: u16, value: []const u8) !void {
        return self.addBytes(allocator, field_id, value);
    }

    /// Append an optional string field. Skips if null.
    pub fn addOptionalString(self: *KvBufWriter, allocator: std.mem.Allocator, field_id: u16, value: ?[]const u8) !void {
        if (value) |v| {
            return self.addString(allocator, field_id, v);
        }
    }

    /// Append a u64 field as 8 little-endian bytes.
    pub fn addU64(self: *KvBufWriter, allocator: std.mem.Allocator, field_id: u16, value: u64) !void {
        var buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &buf, value, .little);
        return self.addBytes(allocator, field_id, &buf);
    }

    /// Append an i64 field as 8 little-endian bytes.
    pub fn addI64(self: *KvBufWriter, allocator: std.mem.Allocator, field_id: u16, value: i64) !void {
        var buf: [8]u8 = undefined;
        std.mem.writeInt(i64, &buf, value, .little);
        return self.addBytes(allocator, field_id, &buf);
    }

    /// Append a u32 field as 4 little-endian bytes.
    pub fn addU32(self: *KvBufWriter, allocator: std.mem.Allocator, field_id: u16, value: u32) !void {
        var buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &buf, value, .little);
        return self.addBytes(allocator, field_id, &buf);
    }

    /// Append a u8 field as a single byte.
    pub fn addU8(self: *KvBufWriter, allocator: std.mem.Allocator, field_id: u16, value: u8) !void {
        const buf = [_]u8{value};
        return self.addBytes(allocator, field_id, &buf);
    }

    /// Finalize and return the complete KvBuf blob.
    /// Caller owns the returned memory.
    pub fn finish(self: *KvBufWriter, allocator: std.mem.Allocator) ![]u8 {
        const values_len = self.values.items.len;
        const dir_len = self.entries.items.len * ENTRY_SIZE;
        const total = values_len + dir_len + FOOTER_SIZE;

        const blob = try allocator.alloc(u8, total);
        errdefer allocator.free(blob);

        // 1. Copy values area
        @memcpy(blob[0..values_len], self.values.items);

        // 2. Write directory entries (10 bytes each: u16 field_id + u32 offset + u32 length)
        var pos = values_len;
        for (self.entries.items) |entry| {
            std.mem.writeInt(u16, blob[pos..][0..2], entry.field_id, .little);
            std.mem.writeInt(u32, blob[pos + 2 ..][0..4], entry.offset, .little);
            std.mem.writeInt(u32, blob[pos + 6 ..][0..4], entry.length, .little);
            pos += ENTRY_SIZE;
        }

        // 3. Write footer
        std.mem.writeInt(u32, blob[pos..][0..4], @intCast(values_len), .little);
        blob[pos + 4] = KVBUF_MAGIC;

        return blob;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "KvBufWriter init and finish empty" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    try std.testing.expectEqual(@as(usize, FOOTER_SIZE), blob.len);
    try std.testing.expectEqual(KVBUF_MAGIC, blob[blob.len - 1]);
    try std.testing.expectEqual(@as(u32, 0), std.mem.readInt(u32, blob[0..4], .little));
}

test "KvBufWriter addString single field" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addString(allocator, 1, "hello");
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    // Values: "hello" (5 bytes) + 1 directory entry (10 bytes) + footer (5 bytes) = 20
    try std.testing.expectEqual(@as(usize, 20), blob.len);
    try std.testing.expectEqualStrings("hello", blob[0..5]);
    try std.testing.expectEqual(KVBUF_MAGIC, blob[blob.len - 1]);
}

test "KvBufWriter addU64 stores little-endian" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addU64(allocator, 10, 0x0102030405060708);
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    try std.testing.expectEqual(@as(usize, 23), blob.len);
    const value = std.mem.readInt(u64, blob[0..8], .little);
    try std.testing.expectEqual(@as(u64, 0x0102030405060708), value);
}

test "KvBufWriter addOptionalString skips null" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addOptionalString(allocator, 1, null);
    try w.addOptionalString(allocator, 2, "present");
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    try std.testing.expectEqual(@as(usize, 22), blob.len);
}

test "KvBufWriter multiple fields" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addString(allocator, 1, "abc");
    try w.addString(allocator, 2, "defgh");
    try w.addU64(allocator, 3, 42);
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    try std.testing.expectEqual(@as(usize, 51), blob.len);
    try std.testing.expectEqualStrings("abc", blob[0..3]);
    try std.testing.expectEqualStrings("defgh", blob[3..8]);
}

test "KvBufWriter large value over 64KB" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    const large = try allocator.alloc(u8, 100_000);
    defer allocator.free(large);
    @memset(large, 'x');

    try w.addBytes(allocator, 1, large);
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    try std.testing.expectEqual(@as(usize, 100_000 + ENTRY_SIZE + FOOTER_SIZE), blob.len);
    try std.testing.expectEqualStrings(large, blob[0..100_000]);
}
