//! Integration tests for BloomFilter.
//!
//! Tests the BloomFilter type exported from core/src/db/root.zig.

const std = @import("std");
const db = @import("db");
const BloomFilter = db.BloomFilter;

// ============================================================================
// Lifecycle Tests
// ============================================================================

test "BloomFilter init and deinit" {
    var bloom = try BloomFilter.init(std.testing.allocator, 100);
    defer bloom.deinit();

    // Should initialize with all bits clear
    try std.testing.expectEqual(@as(usize, 0), bloom.popCount());
}

test "BloomFilter initFromBytes round-trip" {
    var bloom1 = try BloomFilter.init(std.testing.allocator, 100);
    defer bloom1.deinit();

    bloom1.add(12345);
    bloom1.add(67890);

    const bytes = bloom1.toBytes();
    var bloom2 = try BloomFilter.initFromBytes(std.testing.allocator, bytes);
    defer bloom2.deinit();

    try std.testing.expect(bloom2.mayContain(12345));
    try std.testing.expect(bloom2.mayContain(67890));
}

// ============================================================================
// Add and mayContain Tests
// ============================================================================

test "BloomFilter add and mayContain basic" {
    var bloom = try BloomFilter.init(std.testing.allocator, 100);
    defer bloom.deinit();

    // Add some items
    bloom.add(12345);
    bloom.add(67890);
    bloom.add(11111);

    // Items added should be found (no false negatives)
    try std.testing.expect(bloom.mayContain(12345));
    try std.testing.expect(bloom.mayContain(67890));
    try std.testing.expect(bloom.mayContain(11111));
}

test "BloomFilter add increments popCount" {
    var bloom = try BloomFilter.init(std.testing.allocator, 100);
    defer bloom.deinit();

    const initial_count = bloom.popCount();
    bloom.add(12345);
    const after_add = bloom.popCount();

    // Adding an item should set some bits
    try std.testing.expect(after_add > initial_count);
}

test "BloomFilter mayContain returns false for missing items" {
    var bloom = try BloomFilter.init(std.testing.allocator, 100);
    defer bloom.deinit();

    // Only add one item
    bloom.add(12345);

    // Check many items that weren't added
    var false_positives: usize = 0;
    for (0..1000) |i| {
        const test_val = @as(u64, i) + 100000;
        if (bloom.mayContain(test_val)) {
            false_positives += 1;
        }
    }

    // Expect low false positive rate (generous margin for small filter)
    try std.testing.expect(false_positives < 50);
}

// ============================================================================
// Sizing Tests
// ============================================================================

test "BloomFilter has minimum size" {
    var bloom = try BloomFilter.init(std.testing.allocator, 10);
    defer bloom.deinit();

    // Minimum size is 64 bytes
    try std.testing.expect(bloom.bits.len >= 64);
}

test "BloomFilter scales with expected items" {
    var small = try BloomFilter.init(std.testing.allocator, 100);
    defer small.deinit();

    var large = try BloomFilter.init(std.testing.allocator, 10000);
    defer large.deinit();

    // Larger expected count should result in larger filter
    try std.testing.expect(large.bits.len > small.bits.len);
    // 10000 items * 10 bits / 8 = 12500 bytes minimum
    try std.testing.expect(large.bits.len >= 12500);
}

// ============================================================================
// toBytes Tests
// ============================================================================

test "BloomFilter toBytes returns correct length" {
    var bloom = try BloomFilter.init(std.testing.allocator, 100);
    defer bloom.deinit();

    const bytes = bloom.toBytes();
    try std.testing.expectEqual(bloom.bits.len, bytes.len);
}

// ============================================================================
// popCount Tests
// ============================================================================

test "BloomFilter popCount reflects additions" {
    var bloom = try BloomFilter.init(std.testing.allocator, 100);
    defer bloom.deinit();

    const count0 = bloom.popCount();
    bloom.add(12345);
    const count1 = bloom.popCount();
    bloom.add(67890);
    const count2 = bloom.popCount();

    try std.testing.expectEqual(@as(usize, 0), count0);
    try std.testing.expect(count1 > 0);
    try std.testing.expect(count2 >= count1);
}

// ============================================================================
// estimateFalsePositiveRate Tests
// ============================================================================

test "BloomFilter estimateFalsePositiveRate increases with fills" {
    var bloom = try BloomFilter.init(std.testing.allocator, 100);
    defer bloom.deinit();

    const rate0 = bloom.estimateFalsePositiveRate();

    // Add many items to increase fill rate
    for (0..50) |i| {
        bloom.add(@as(u64, i) * 12345);
    }

    const rate1 = bloom.estimateFalsePositiveRate();

    // Empty filter should have 0 FPR
    try std.testing.expectEqual(@as(f64, 0), rate0);
    // After adding items, FPR should be positive
    try std.testing.expect(rate1 > 0);
}
