//! Integration tests for TokenMask
//!
//! TokenMask is a bit vector for tracking valid tokens during constrained sampling.
//! It supports efficient set/check operations and bulk masking of logits.

const std = @import("std");
const main = @import("main");
const TokenMask = main.validate.TokenMask;

// =============================================================================
// Lifecycle Tests
// =============================================================================

test "TokenMask: init allocates and zeros bits" {
    const allocator = std.testing.allocator;

    var mask = try TokenMask.init(allocator, 100);
    defer mask.deinit();

    // All bits should be clear after init
    for (0..100) |i| {
        try std.testing.expect(!mask.isSet(i));
    }
}

test "TokenMask: init handles zero length" {
    const allocator = std.testing.allocator;

    var mask = try TokenMask.init(allocator, 0);
    defer mask.deinit();

    try std.testing.expectEqual(@as(usize, 0), mask.len);
}

test "TokenMask: init handles non-64-aligned length" {
    const allocator = std.testing.allocator;

    // 70 is not divisible by 64
    var mask = try TokenMask.init(allocator, 70);
    defer mask.deinit();

    try std.testing.expectEqual(@as(usize, 70), mask.len);
}

test "TokenMask: deinit frees memory" {
    const allocator = std.testing.allocator;

    var mask = try TokenMask.init(allocator, 128);
    mask.deinit();
    // No leak = test passes (std.testing.allocator checks)
}

// =============================================================================
// allValid Tests
// =============================================================================

test "TokenMask: allValid sets all bits" {
    const allocator = std.testing.allocator;

    var mask = try TokenMask.allValid(allocator, 100);
    defer mask.deinit();

    for (0..100) |i| {
        try std.testing.expect(mask.isSet(i));
    }
}

test "TokenMask: allValid clears padding bits" {
    const allocator = std.testing.allocator;

    // 70 bits means padding in the last word
    var mask = try TokenMask.allValid(allocator, 70);
    defer mask.deinit();

    // Bits 0-69 should be set
    for (0..70) |i| {
        try std.testing.expect(mask.isSet(i));
    }

    // Bits beyond len should return false (not in valid range)
    try std.testing.expect(!mask.isSet(70));
    try std.testing.expect(!mask.isSet(100));
}

// =============================================================================
// set Tests
// =============================================================================

test "TokenMask: set marks bit as valid" {
    const allocator = std.testing.allocator;

    var mask = try TokenMask.init(allocator, 100);
    defer mask.deinit();

    mask.set(42);
    try std.testing.expect(mask.isSet(42));
}

test "TokenMask: set multiple bits" {
    const allocator = std.testing.allocator;

    var mask = try TokenMask.init(allocator, 100);
    defer mask.deinit();

    mask.set(0);
    mask.set(63); // Last bit of first word
    mask.set(64); // First bit of second word
    mask.set(99); // Near end

    try std.testing.expect(mask.isSet(0));
    try std.testing.expect(mask.isSet(63));
    try std.testing.expect(mask.isSet(64));
    try std.testing.expect(mask.isSet(99));
}

test "TokenMask: set out of range is no-op" {
    const allocator = std.testing.allocator;

    var mask = try TokenMask.init(allocator, 100);
    defer mask.deinit();

    // Setting beyond len should be silently ignored
    mask.set(100);
    mask.set(200);
    // Should not crash, and isSet should return false
    try std.testing.expect(!mask.isSet(100));
}

test "TokenMask: setValid is alias for set" {
    const allocator = std.testing.allocator;

    var mask = try TokenMask.init(allocator, 100);
    defer mask.deinit();

    mask.setValid(25);
    try std.testing.expect(mask.isSet(25));
}

// =============================================================================
// isSet Tests
// =============================================================================

test "TokenMask: isSet returns false for unset bit" {
    const allocator = std.testing.allocator;

    var mask = try TokenMask.init(allocator, 100);
    defer mask.deinit();

    try std.testing.expect(!mask.isSet(50));
}

test "TokenMask: isSet returns true for set bit" {
    const allocator = std.testing.allocator;

    var mask = try TokenMask.init(allocator, 100);
    defer mask.deinit();

    mask.set(50);
    try std.testing.expect(mask.isSet(50));
}

test "TokenMask: isSet returns false for out of range index" {
    const allocator = std.testing.allocator;

    var mask = try TokenMask.init(allocator, 100);
    defer mask.deinit();

    try std.testing.expect(!mask.isSet(100));
    try std.testing.expect(!mask.isSet(1000));
}

// =============================================================================
// setAll Tests
// =============================================================================

test "TokenMask: setAll sets all valid bits" {
    const allocator = std.testing.allocator;

    var mask = try TokenMask.init(allocator, 100);
    defer mask.deinit();

    mask.setAll();

    for (0..100) |i| {
        try std.testing.expect(mask.isSet(i));
    }
}

test "TokenMask: setAll clears padding" {
    const allocator = std.testing.allocator;

    // 70 bits - padding in last word
    var mask = try TokenMask.init(allocator, 70);
    defer mask.deinit();

    mask.setAll();

    // Valid range should be set
    for (0..70) |i| {
        try std.testing.expect(mask.isSet(i));
    }

    // Beyond len should return false
    try std.testing.expect(!mask.isSet(70));
}

// =============================================================================
// clearAll Tests
// =============================================================================

test "TokenMask: clearAll clears all bits" {
    const allocator = std.testing.allocator;

    var mask = try TokenMask.allValid(allocator, 100);
    defer mask.deinit();

    mask.clearAll();

    for (0..100) |i| {
        try std.testing.expect(!mask.isSet(i));
    }
}

// =============================================================================
// Word Boundary Tests
// =============================================================================

test "TokenMask: handles exact 64-bit boundary" {
    const allocator = std.testing.allocator;

    var mask = try TokenMask.init(allocator, 64);
    defer mask.deinit();

    mask.set(0);
    mask.set(63);

    try std.testing.expect(mask.isSet(0));
    try std.testing.expect(mask.isSet(63));
    try std.testing.expect(!mask.isSet(64)); // Out of range
}

test "TokenMask: handles 128 bits (two words)" {
    const allocator = std.testing.allocator;

    var mask = try TokenMask.init(allocator, 128);
    defer mask.deinit();

    mask.set(0);
    mask.set(63);
    mask.set(64);
    mask.set(127);

    try std.testing.expect(mask.isSet(0));
    try std.testing.expect(mask.isSet(63));
    try std.testing.expect(mask.isSet(64));
    try std.testing.expect(mask.isSet(127));
}

// =============================================================================
// Large Mask Tests
// =============================================================================

test "TokenMask: handles large vocabulary size" {
    const allocator = std.testing.allocator;

    // Typical vocab size
    var mask = try TokenMask.init(allocator, 32000);
    defer mask.deinit();

    mask.set(0);
    mask.set(15999);
    mask.set(31999);

    try std.testing.expect(mask.isSet(0));
    try std.testing.expect(mask.isSet(15999));
    try std.testing.expect(mask.isSet(31999));
    try std.testing.expect(!mask.isSet(1));
}
