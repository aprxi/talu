//! C API FFI conversion utilities.
//!
//! Shared ownership/lifetime helpers for C-facing modules. Domain modules should
//! not import this file.

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
// Tests
// =============================================================================

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
