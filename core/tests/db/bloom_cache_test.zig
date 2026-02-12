//! Integration tests for BloomCache.
//!
//! Tests the BloomCache type exported from core/src/db/root.zig.

const std = @import("std");
const db = @import("db");
const BloomCache = db.BloomCache;
const BloomFilter = db.BloomFilter;
const hashString = db.bloom.hashString;

// ============================================================================
// Lifecycle Tests
// ============================================================================

test "BloomCache init and deinit" {
    var cache = BloomCache.init(std.testing.allocator);
    defer cache.deinit();

    // Empty cache should not find any blocks
    const result = cache.mayContainInBlock("/any/path", 0x1000, hashString("doc-123"));
    try std.testing.expect(result == null);
}

// ============================================================================
// getOrCreate Tests
// ============================================================================

test "BloomCache getOrCreate returns filter for new block" {
    var cache = BloomCache.init(std.testing.allocator);
    defer cache.deinit();

    const filter = try cache.getOrCreate("/path/to/file.talu", 0x1000, 100);

    // Should return a valid filter
    try std.testing.expect(filter.bits.len > 0);
    try std.testing.expectEqual(@as(usize, 0), filter.popCount());
}

test "BloomCache getOrCreate returns same filter for same block" {
    var cache = BloomCache.init(std.testing.allocator);
    defer cache.deinit();

    const filter1 = try cache.getOrCreate("/path/to/file.talu", 0x1000, 100);
    filter1.add(hashString("doc-123"));

    const filter2 = try cache.getOrCreate("/path/to/file.talu", 0x1000, 100);

    // Should be the same filter (pointer comparison)
    try std.testing.expectEqual(filter1, filter2);
    // Should still have the item we added
    try std.testing.expect(filter2.mayContain(hashString("doc-123")));
}

test "BloomCache getOrCreate returns different filter for different offset" {
    var cache = BloomCache.init(std.testing.allocator);
    defer cache.deinit();

    const filter1 = try cache.getOrCreate("/path/to/file.talu", 0x1000, 100);
    filter1.add(hashString("doc-123"));

    const filter2 = try cache.getOrCreate("/path/to/file.talu", 0x2000, 100);

    // Different offset = different filter
    try std.testing.expect(filter1 != filter2);
    // New filter should not contain the item
    try std.testing.expect(!filter2.mayContain(hashString("doc-123")));
}

test "BloomCache getOrCreate returns different filter for different path" {
    var cache = BloomCache.init(std.testing.allocator);
    defer cache.deinit();

    const filter1 = try cache.getOrCreate("/path/to/file1.talu", 0x1000, 100);
    filter1.add(hashString("doc-123"));

    const filter2 = try cache.getOrCreate("/path/to/file2.talu", 0x1000, 100);

    // Different path = different filter
    try std.testing.expect(filter1 != filter2);
}

// ============================================================================
// mayContainInBlock Tests
// ============================================================================

test "BloomCache mayContainInBlock finds added item" {
    var cache = BloomCache.init(std.testing.allocator);
    defer cache.deinit();

    const filter = try cache.getOrCreate("/path/to/file.talu", 0x1000, 100);
    filter.add(hashString("doc-123"));
    filter.add(hashString("doc-456"));

    const result1 = cache.mayContainInBlock("/path/to/file.talu", 0x1000, hashString("doc-123"));
    try std.testing.expect(result1 != null);
    try std.testing.expect(result1.?);

    const result2 = cache.mayContainInBlock("/path/to/file.talu", 0x1000, hashString("doc-456"));
    try std.testing.expect(result2 != null);
    try std.testing.expect(result2.?);
}

test "BloomCache mayContainInBlock returns null for uncached block" {
    var cache = BloomCache.init(std.testing.allocator);
    defer cache.deinit();

    // Create a filter for one block
    _ = try cache.getOrCreate("/path/to/file.talu", 0x1000, 100);

    // Query a different block that has no filter
    const result = cache.mayContainInBlock("/other/path.talu", 0x2000, hashString("doc-123"));
    try std.testing.expect(result == null);
}

test "BloomCache mayContainInBlock returns false for missing item" {
    var cache = BloomCache.init(std.testing.allocator);
    defer cache.deinit();

    const filter = try cache.getOrCreate("/path/to/file.talu", 0x1000, 100);
    filter.add(hashString("doc-123"));

    // Query for an item that was never added
    const result = cache.mayContainInBlock("/path/to/file.talu", 0x1000, hashString("doc-999"));

    // Should return definite false (not present) or false positive
    // Due to bloom filter semantics, we can only test that it returns a value
    try std.testing.expect(result != null);
}

// ============================================================================
// hashString Tests
// ============================================================================

test "hashString is deterministic" {
    const h1 = hashString("test-document");
    const h2 = hashString("test-document");
    const h3 = hashString("different-document");

    try std.testing.expectEqual(h1, h2);
    try std.testing.expect(h1 != h3);
}

test "hashString produces different hashes for different inputs" {
    const h1 = hashString("doc-001");
    const h2 = hashString("doc-002");
    const h3 = hashString("doc-003");

    try std.testing.expect(h1 != h2);
    try std.testing.expect(h2 != h3);
    try std.testing.expect(h1 != h3);
}
