//! FFI Conversion Utilities
//!
//! Core logic for converting between Zig types and C-compatible types.
//! The capi/ layer should delegate to these functions rather than implementing
//! conversion logic inline.
//!
//! ## Design Principle
//!
//! The capi/ functions are thin wrappers - they only do argument validation,
//! conversion, and error mapping. Any logic involving loops, complex allocations,
//! or multi-step conversions belongs here in helpers/ffi.zig instead.
//!
//! ## Usage
//!
//! ```zig
//! const ffi = @import("helpers/ffi.zig");
//!
//! // Convert Zig string slice to C string list
//! const list = try ffi.StringList.fromSlices(allocator, strings);
//! defer list.deinit(allocator);
//! ```

const std = @import("std");

/// A list of null-terminated strings for C interop.
/// This is the core type - capi modules can re-export or wrap as needed.
pub const StringList = struct {
    items: [][:0]const u8,

    const Self = @This();

    /// Convert a slice of Zig strings to a C-compatible StringList.
    /// All strings are copied with null terminators.
    /// Caller owns the result and must call deinit().
    pub fn fromSlices(allocator: std.mem.Allocator, strings: []const []const u8) !Self {
        var items = std.ArrayListUnmanaged([:0]const u8){};
        errdefer {
            for (items.items) |item| allocator.free(item);
            items.deinit(allocator);
        }

        for (strings) |s| {
            const cstr = try allocator.allocSentinel(u8, s.len, 0);
            errdefer allocator.free(cstr);
            @memcpy(cstr, s);
            try items.append(allocator, cstr);
        }

        return Self{
            .items = try items.toOwnedSlice(allocator),
        };
    }

    /// Free all strings and the items array.
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        for (self.items) |item| {
            allocator.free(item);
        }
        allocator.free(self.items);
        self.items = &.{};
    }

    /// Get the number of strings.
    pub fn count(self: *const Self) usize {
        return self.items.len;
    }

    /// Get string at index, or null if out of bounds.
    pub fn get(self: *const Self, idx: usize) ?[:0]const u8 {
        if (idx >= self.items.len) return null;
        return self.items[idx];
    }
};

// =============================================================================
// C Callback Adapters
// =============================================================================

/// C callback type for getting token info by ID.
/// Returns pointer to token bytes and sets out_len to the length.
/// Returns null if token ID is invalid.
pub const TokenInfoCallback = *const fn (
    token_id: u32,
    out_len: *usize,
    ctx: ?*anyopaque,
) callconv(.c) ?[*]const u8;

// =============================================================================
// JSON Utilities
// =============================================================================

/// Builds a JSON array of strings from multiple slices.
/// Returns a null-terminated string like `["a","b","c"]`.
/// Caller owns the result.
pub fn buildJsonStringArray(alloc: std.mem.Allocator, slices: []const []const []const u8) ![:0]u8 {
    var buffer = std.ArrayListUnmanaged(u8){};
    errdefer buffer.deinit(alloc);

    try buffer.append(alloc, '[');

    var is_first = true;
    for (slices) |slice| {
        for (slice) |s| {
            if (!is_first) try buffer.append(alloc, ',');
            try buffer.append(alloc, '"');
            try buffer.appendSlice(alloc, s);
            try buffer.append(alloc, '"');
            is_first = false;
        }
    }

    try buffer.append(alloc, ']');

    // Allocate with null terminator
    const result = try alloc.allocSentinel(u8, buffer.items.len, 0);
    @memcpy(result, buffer.items);
    buffer.deinit(alloc);
    return result;
}

// =============================================================================
// Tests
// =============================================================================

test "buildJsonStringArray creates valid JSON" {
    const alloc = std.testing.allocator;
    const slice1 = &[_][]const u8{ "a", "b" };
    const slice2 = &[_][]const u8{"c"};

    const result = try buildJsonStringArray(alloc, &.{ slice1, slice2 });
    defer alloc.free(result);

    try std.testing.expectEqualStrings("[\"a\",\"b\",\"c\"]", result);
}

test "buildJsonStringArray handles empty input" {
    const alloc = std.testing.allocator;
    const empty: []const []const u8 = &.{};

    const result = try buildJsonStringArray(alloc, &.{empty});
    defer alloc.free(result);

    try std.testing.expectEqualStrings("[]", result);
}

test "StringList.fromSlices creates valid list" {
    const allocator = std.testing.allocator;
    const strings = &[_][]const u8{ "hello", "world", "test" };

    var list = try StringList.fromSlices(allocator, strings);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 3), list.count());
    try std.testing.expectEqualStrings("hello", list.get(0).?);
    try std.testing.expectEqualStrings("world", list.get(1).?);
    try std.testing.expectEqualStrings("test", list.get(2).?);
    try std.testing.expect(list.get(3) == null);
}

test "StringList.fromSlices handles empty input" {
    const allocator = std.testing.allocator;
    const strings = &[_][]const u8{};

    var list = try StringList.fromSlices(allocator, strings);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), list.count());
    try std.testing.expect(list.get(0) == null);
}

test "StringList strings are null-terminated" {
    const allocator = std.testing.allocator;
    const strings = &[_][]const u8{"abc"};

    var list = try StringList.fromSlices(allocator, strings);
    defer list.deinit(allocator);

    const str = list.get(0).?;
    // Sentinel-terminated slice should have null at the end
    try std.testing.expectEqual(@as(u8, 0), str[str.len]);
}
