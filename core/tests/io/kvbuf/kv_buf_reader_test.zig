//! Integration tests for io.kvbuf.KvBufReader
//!
//! KvBufReader provides zero-copy field access from KvBuf binary blobs.
//! All returned slices borrow from the original buffer — no allocations.

const std = @import("std");
const main = @import("main");

const KvBufReader = main.io.kvbuf.KvBufReader;
const KvBufWriter = main.io.kvbuf.KvBufWriter;
const KVBUF_MAGIC = main.io.kvbuf.KVBUF_MAGIC;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "KvBufReader type is accessible" {
    const T = KvBufReader;
    _ = T;
}

test "KvBufReader is a struct" {
    const info = @typeInfo(KvBufReader);
    try std.testing.expect(info == .@"struct");
}

test "KvBufReader has expected fields" {
    const info = @typeInfo(KvBufReader);
    const fields = info.@"struct".fields;

    var has_data = false;
    var has_dir_offset = false;
    var has_entry_count = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "data")) has_data = true;
        if (comptime std.mem.eql(u8, field.name, "dir_offset")) has_dir_offset = true;
        if (comptime std.mem.eql(u8, field.name, "entry_count")) has_entry_count = true;
    }

    try std.testing.expect(has_data);
    try std.testing.expect(has_dir_offset);
    try std.testing.expect(has_entry_count);
}

// =============================================================================
// Method Tests
// =============================================================================

test "KvBufReader has init method" {
    try std.testing.expect(@hasDecl(KvBufReader, "init"));
}

test "KvBufReader has get method" {
    try std.testing.expect(@hasDecl(KvBufReader, "get"));
}

test "KvBufReader has getU64 method" {
    try std.testing.expect(@hasDecl(KvBufReader, "getU64"));
}

test "KvBufReader has getI64 method" {
    try std.testing.expect(@hasDecl(KvBufReader, "getI64"));
}

test "KvBufReader has getU32 method" {
    try std.testing.expect(@hasDecl(KvBufReader, "getU32"));
}

test "KvBufReader has getU8 method" {
    try std.testing.expect(@hasDecl(KvBufReader, "getU8"));
}

test "KvBufReader has fieldCount method" {
    try std.testing.expect(@hasDecl(KvBufReader, "fieldCount"));
}

// =============================================================================
// init Tests
// =============================================================================

test "KvBufReader.init rejects empty blob" {
    try std.testing.expectError(error.InvalidKvBuf, KvBufReader.init(""));
}

test "KvBufReader.init rejects short blob" {
    try std.testing.expectError(error.InvalidKvBuf, KvBufReader.init("abc"));
}

test "KvBufReader.init rejects bad magic" {
    const blob = [_]u8{ 0, 0, 0, 0, 0xFF };
    try std.testing.expectError(error.InvalidKvBuf, KvBufReader.init(&blob));
}

test "KvBufReader.init accepts valid empty blob" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    try std.testing.expectEqual(@as(usize, 0), reader.fieldCount());
}

// =============================================================================
// get Tests
// =============================================================================

test "KvBufReader.get returns field value" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addString(allocator, 1, "hello");
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    const val = reader.get(1) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("hello", val);
}

test "KvBufReader.get returns null for missing field" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addString(allocator, 1, "hello");
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    try std.testing.expect(reader.get(99) == null);
}

test "KvBufReader.get finds correct field among multiple" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addString(allocator, 1, "first");
    try w.addString(allocator, 2, "second");
    try w.addString(allocator, 3, "third");

    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);

    const f1 = reader.get(1) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("first", f1);

    const f2 = reader.get(2) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("second", f2);

    const f3 = reader.get(3) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("third", f3);
}

// =============================================================================
// getU64 Tests
// =============================================================================

test "KvBufReader.getU64 returns value" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addU64(allocator, 1, 0x0102030405060708);
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    const val = reader.getU64(1) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u64, 0x0102030405060708), val);
}

test "KvBufReader.getU64 returns null for missing field" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    try std.testing.expect(reader.getU64(1) == null);
}

test "KvBufReader.getU64 returns null for wrong size field" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addString(allocator, 1, "short"); // 5 bytes, not 8
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    try std.testing.expect(reader.getU64(1) == null);
}

// =============================================================================
// getI64 Tests
// =============================================================================

test "KvBufReader.getI64 returns negative value" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addI64(allocator, 1, -12345);
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    const val = reader.getI64(1) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(i64, -12345), val);
}

test "KvBufReader.getI64 returns null for wrong size field" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addU32(allocator, 1, 42); // 4 bytes, not 8
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    try std.testing.expect(reader.getI64(1) == null);
}

// =============================================================================
// getU32 Tests
// =============================================================================

test "KvBufReader.getU32 returns value" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addU32(allocator, 1, 0xDEADBEEF);
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    const val = reader.getU32(1) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u32, 0xDEADBEEF), val);
}

test "KvBufReader.getU32 returns null for wrong size field" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addU64(allocator, 1, 42); // 8 bytes, not 4
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    try std.testing.expect(reader.getU32(1) == null);
}

// =============================================================================
// getU8 Tests
// =============================================================================

test "KvBufReader.getU8 returns single byte" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addU8(allocator, 1, 42);
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    const val = reader.getU8(1) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u8, 42), val);
}

test "KvBufReader.getU8 returns null for wrong size field" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addString(allocator, 1, "ab"); // 2 bytes, not 1
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    try std.testing.expect(reader.getU8(1) == null);
}

// =============================================================================
// fieldCount Tests
// =============================================================================

test "KvBufReader.fieldCount returns zero for empty blob" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    try std.testing.expectEqual(@as(usize, 0), reader.fieldCount());
}

test "KvBufReader.fieldCount returns correct count" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addString(allocator, 1, "a");
    try w.addString(allocator, 2, "b");
    try w.addString(allocator, 3, "c");

    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    try std.testing.expectEqual(@as(usize, 3), reader.fieldCount());
}

// =============================================================================
// Zero-Copy Verification Tests
// =============================================================================

test "KvBufReader.get returns slice into original blob" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addString(allocator, 1, "hello");
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    const val = reader.get(1) orelse return error.TestUnexpectedResult;

    // Verify slice points into blob memory (zero-copy)
    const blob_start = @intFromPtr(blob.ptr);
    const blob_end = blob_start + blob.len;
    const val_start = @intFromPtr(val.ptr);
    const val_end = val_start + val.len;

    try std.testing.expect(val_start >= blob_start);
    try std.testing.expect(val_end <= blob_end);
}

// =============================================================================
// Large Value Tests
// =============================================================================

test "KvBufReader handles large values" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
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
