//! Integration tests for io.kvbuf.KvBufWriter
//!
//! KvBufWriter serializes fields into a directory-last binary format,
//! separating field values from structural metadata for zero-copy access.

const std = @import("std");
const main = @import("main");

const KvBufWriter = main.io.kvbuf.KvBufWriter;
const KvBufReader = main.io.kvbuf.KvBufReader;
const KVBUF_MAGIC = main.io.kvbuf.KVBUF_MAGIC;
const FOOTER_SIZE = main.io.kvbuf.FOOTER_SIZE;
const ENTRY_SIZE = main.io.kvbuf.ENTRY_SIZE;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "KvBufWriter type is accessible" {
    const T = KvBufWriter;
    _ = T;
}

test "KvBufWriter is a struct" {
    const info = @typeInfo(KvBufWriter);
    try std.testing.expect(info == .@"struct");
}

// =============================================================================
// Method Tests
// =============================================================================

test "KvBufWriter has init method" {
    try std.testing.expect(@hasDecl(KvBufWriter, "init"));
}

test "KvBufWriter has deinit method" {
    try std.testing.expect(@hasDecl(KvBufWriter, "deinit"));
}

test "KvBufWriter has addBytes method" {
    try std.testing.expect(@hasDecl(KvBufWriter, "addBytes"));
}

test "KvBufWriter has addString method" {
    try std.testing.expect(@hasDecl(KvBufWriter, "addString"));
}

test "KvBufWriter has addOptionalString method" {
    try std.testing.expect(@hasDecl(KvBufWriter, "addOptionalString"));
}

test "KvBufWriter has addU64 method" {
    try std.testing.expect(@hasDecl(KvBufWriter, "addU64"));
}

test "KvBufWriter has addI64 method" {
    try std.testing.expect(@hasDecl(KvBufWriter, "addI64"));
}

test "KvBufWriter has addU32 method" {
    try std.testing.expect(@hasDecl(KvBufWriter, "addU32"));
}

test "KvBufWriter has addU8 method" {
    try std.testing.expect(@hasDecl(KvBufWriter, "addU8"));
}

test "KvBufWriter has finish method" {
    try std.testing.expect(@hasDecl(KvBufWriter, "finish"));
}

// =============================================================================
// init/deinit Tests
// =============================================================================

test "KvBufWriter.init creates empty writer" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    // Should be able to finish immediately with just footer
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    try std.testing.expectEqual(@as(usize, FOOTER_SIZE), blob.len);
}

// =============================================================================
// addString Tests
// =============================================================================

test "KvBufWriter.addString stores string data" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addString(allocator, 1, "hello");
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    // Verify via reader
    const reader = try KvBufReader.init(blob);
    const val = reader.get(1) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("hello", val);
}

test "KvBufWriter.addString with empty string" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addString(allocator, 1, "");
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    const val = reader.get(1) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 0), val.len);
}

// =============================================================================
// addOptionalString Tests
// =============================================================================

test "KvBufWriter.addOptionalString with value" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addOptionalString(allocator, 1, "present");
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    const val = reader.get(1) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("present", val);
}

test "KvBufWriter.addOptionalString with null skips field" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addOptionalString(allocator, 1, null);
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    try std.testing.expect(reader.get(1) == null);
    try std.testing.expectEqual(@as(usize, 0), reader.fieldCount());
}

// =============================================================================
// addU64 Tests
// =============================================================================

test "KvBufWriter.addU64 stores little-endian value" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addU64(allocator, 10, 0x0102030405060708);
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    const val = reader.getU64(10) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u64, 0x0102030405060708), val);
}

test "KvBufWriter.addU64 with zero" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addU64(allocator, 1, 0);
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    const val = reader.getU64(1) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u64, 0), val);
}

test "KvBufWriter.addU64 with max value" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addU64(allocator, 1, std.math.maxInt(u64));
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    const val = reader.getU64(1) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(std.math.maxInt(u64), val);
}

// =============================================================================
// addI64 Tests
// =============================================================================

test "KvBufWriter.addI64 stores negative value" {
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

// =============================================================================
// addU32 Tests
// =============================================================================

test "KvBufWriter.addU32 stores value" {
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

// =============================================================================
// addU8 Tests
// =============================================================================

test "KvBufWriter.addU8 stores single byte" {
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

// =============================================================================
// Multiple Fields Tests
// =============================================================================

test "KvBufWriter multiple fields preserve order and values" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addString(allocator, 1, "first");
    try w.addString(allocator, 2, "second");
    try w.addU64(allocator, 3, 999);
    try w.addU8(allocator, 4, 0xFF);

    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    const reader = try KvBufReader.init(blob);
    try std.testing.expectEqual(@as(usize, 4), reader.fieldCount());

    const f1 = reader.get(1) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("first", f1);

    const f2 = reader.get(2) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("second", f2);

    const f3 = reader.getU64(3) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u64, 999), f3);

    const f4 = reader.getU8(4) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u8, 0xFF), f4);
}

// =============================================================================
// finish Tests
// =============================================================================

test "KvBufWriter.finish produces valid magic byte" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addString(allocator, 1, "test");
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    try std.testing.expectEqual(KVBUF_MAGIC, blob[blob.len - 1]);
}

test "KvBufWriter.finish blob size calculation" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    // 5 bytes value + 1 entry (10 bytes) + footer (5 bytes) = 20 bytes
    try w.addString(allocator, 1, "hello");
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    try std.testing.expectEqual(@as(usize, 20), blob.len);
}

// =============================================================================
// Large Value Tests
// =============================================================================

test "KvBufWriter handles large values" {
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

    const reader = try KvBufReader.init(blob);
    const val = reader.get(1) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 100_000), val.len);
}
