//! KvBuf Reader - Zero-copy field access from KvBuf binary blobs.
//!
//! Wraps a `[]const u8` buffer and provides O(n) field lookup by scanning
//! the directory. The directory is typically small (< 20 entries), so linear
//! scan is faster than hashing for this use case.
//!
//! All returned slices borrow from the original buffer — no allocations.
//!
//! Thread safety: Immutable after initialization; safe for concurrent reads.

const std = @import("std");
const writer_mod = @import("writer.zig");

const KVBUF_MAGIC = writer_mod.KVBUF_MAGIC;
const FOOTER_SIZE = writer_mod.FOOTER_SIZE;
const ENTRY_SIZE = writer_mod.ENTRY_SIZE; // 10 bytes: u16 field_id + u32 offset + u32 length

/// Zero-copy reader over a KvBuf-encoded blob.
///
/// Usage:
///   const reader = KvBufReader.init(blob) catch return error.InvalidPayload;
///   const text = reader.get(FieldIds.content_text) orelse return null;
///   // text is a slice into blob — no allocation.
///
/// Returned slices borrow from the input blob. The blob must outlive all slices.
pub const KvBufReader = struct {
    data: []const u8,
    dir_offset: u32,
    entry_count: usize,

    /// Parse footer and validate magic. Returns error if blob is malformed.
    pub fn init(blob: []const u8) !KvBufReader {
        if (blob.len < FOOTER_SIZE) return error.InvalidKvBuf;

        const magic = blob[blob.len - 1];
        if (magic != KVBUF_MAGIC) return error.InvalidKvBuf;

        const dir_offset = std.mem.readInt(u32, blob[blob.len - FOOTER_SIZE ..][0..4], .little);
        const dir_area_len = blob.len - FOOTER_SIZE - @as(usize, dir_offset);

        if (dir_area_len % ENTRY_SIZE != 0) return error.InvalidKvBuf;

        return .{
            .data = blob,
            .dir_offset = dir_offset,
            .entry_count = dir_area_len / ENTRY_SIZE,
        };
    }

    /// Look up a field by ID. Returns a zero-copy slice into the blob,
    /// or null if the field is not present.
    pub fn get(self: KvBufReader, field_id: u16) ?[]const u8 {
        var i: usize = 0;
        while (i < self.entry_count) : (i += 1) {
            const base = @as(usize, self.dir_offset) + i * ENTRY_SIZE;
            const entry_fid = std.mem.readInt(u16, self.data[base..][0..2], .little);
            if (entry_fid == field_id) {
                const offset = std.mem.readInt(u32, self.data[base + 2 ..][0..4], .little);
                const length = std.mem.readInt(u32, self.data[base + 6 ..][0..4], .little);
                const start = @as(usize, offset);
                const end = start + @as(usize, length);
                if (end > @as(usize, self.dir_offset)) return null; // corrupt: overlaps directory
                return self.data[start..end];
            }
        }
        return null;
    }

    /// Look up a field and interpret as little-endian u64.
    /// Returns null if field not found or length is not 8.
    pub fn getU64(self: KvBufReader, field_id: u16) ?u64 {
        const slice = self.get(field_id) orelse return null;
        if (slice.len != 8) return null;
        return std.mem.readInt(u64, slice[0..8], .little);
    }

    /// Look up a field and interpret as little-endian i64.
    /// Returns null if field not found or length is not 8.
    pub fn getI64(self: KvBufReader, field_id: u16) ?i64 {
        const slice = self.get(field_id) orelse return null;
        if (slice.len != 8) return null;
        return std.mem.readInt(i64, slice[0..8], .little);
    }

    /// Look up a field and interpret as little-endian u32.
    /// Returns null if field not found or length is not 4.
    pub fn getU32(self: KvBufReader, field_id: u16) ?u32 {
        const slice = self.get(field_id) orelse return null;
        if (slice.len != 4) return null;
        return std.mem.readInt(u32, slice[0..4], .little);
    }

    /// Look up a field and interpret as a single u8.
    /// Returns null if field not found or length is not 1.
    pub fn getU8(self: KvBufReader, field_id: u16) ?u8 {
        const slice = self.get(field_id) orelse return null;
        if (slice.len != 1) return null;
        return slice[0];
    }

    /// Return the number of directory entries.
    pub fn fieldCount(self: KvBufReader) usize {
        return self.entry_count;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "KvBufReader init rejects short blob" {
    const result = KvBufReader.init("");
    try std.testing.expectError(error.InvalidKvBuf, result);
}

test "KvBufReader init rejects bad magic" {
    const blob = [_]u8{ 0, 0, 0, 0, 0xFF };
    try std.testing.expectError(error.InvalidKvBuf, KvBufReader.init(&blob));
}

test "KvBufReader roundtrip with KvBufWriter" {
    const allocator = std.testing.allocator;
    var w = writer_mod.KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addString(allocator, 1, "hello");
    try w.addString(allocator, 2, "world");
    try w.addU64(allocator, 3, 999);

    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);

    const f1 = reader.get(1) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("hello", f1);

    const f2 = reader.get(2) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("world", f2);

    const f3 = reader.getU64(3) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u64, 999), f3);

    try std.testing.expect(reader.get(99) == null);
}

test "KvBufReader fieldCount" {
    const allocator = std.testing.allocator;
    var w = writer_mod.KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addString(allocator, 1, "a");
    try w.addString(allocator, 2, "b");
    try w.addString(allocator, 3, "c");

    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    try std.testing.expectEqual(@as(usize, 3), reader.fieldCount());
}

test "KvBufReader getU8" {
    const allocator = std.testing.allocator;
    var w = writer_mod.KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addU8(allocator, 5, 42);

    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    const val = reader.getU8(5) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u8, 42), val);
}

test "KvBufReader getI64" {
    const allocator = std.testing.allocator;
    var w = writer_mod.KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addI64(allocator, 7, -12345);

    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    const val = reader.getI64(7) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(i64, -12345), val);
}

test "KvBufReader empty blob (no fields)" {
    const allocator = std.testing.allocator;
    var w = writer_mod.KvBufWriter.init();
    defer w.deinit(allocator);

    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    try std.testing.expectEqual(@as(usize, 0), reader.fieldCount());
    try std.testing.expect(reader.get(1) == null);
}

test "KvBufReader getU32" {
    const allocator = std.testing.allocator;
    var w = writer_mod.KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addU32(allocator, 8, 0xDEADBEEF);

    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    const val = reader.getU32(8) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u32, 0xDEADBEEF), val);
}

test "KvBufReader large value roundtrip" {
    const allocator = std.testing.allocator;
    var w = writer_mod.KvBufWriter.init();
    defer w.deinit(allocator);

    const large = try allocator.alloc(u8, 200_000);
    defer allocator.free(large);
    @memset(large, 'A');

    try w.addBytes(allocator, 1, large);
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    const val = reader.get(1) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 200_000), val.len);
    try std.testing.expectEqualStrings(large, val);
}
