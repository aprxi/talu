//! Tokenizer Internal Types
//!
//! Common type definitions used across tokenizer modules.
//! Includes allocator, token, normalized text, and range types.

const std = @import("std");

pub const Allocator = std.heap.c_allocator;

pub const Range = struct {
    start: usize,
    end: usize,
};

pub const Normalized = struct {
    text: []u8,
    map: []i32, // normalized byte index -> original byte index

    pub fn deinit(self: *Normalized) void {
        Allocator.free(self.text);
        Allocator.free(self.map);
    }
};

pub const Token = struct {
    ptr: [*]u8,
    len: usize,

    pub fn slice(self: Token) []u8 {
        return self.ptr[0..self.len];
    }

    pub fn sliceConst(self: Token) []const u8 {
        return self.ptr[0..self.len];
    }
};

pub const PretokenizeResult = struct {
    tokens: std.ArrayListUnmanaged(Token),
    ranges: std.ArrayListUnmanaged(Range),

    pub fn deinit(self: *PretokenizeResult) void {
        for (self.tokens.items) |token| {
            Allocator.free(token.ptr[0 .. token.len + 1]); // +1 for null terminator we still add
        }
        self.tokens.deinit(Allocator);
        self.ranges.deinit(Allocator);
    }
};

// =============================================================================
// Tests
// =============================================================================

test "Token slice conversions" {
    const data = try Allocator.alloc(u8, 5);
    defer Allocator.free(data);
    @memcpy(data, "hello");

    const token = Token{ .ptr = data.ptr, .len = 5 };
    const slice = token.slice();
    try std.testing.expectEqual(@as(usize, 5), slice.len);
    try std.testing.expectEqualStrings("hello", slice);

    const const_slice = token.sliceConst();
    try std.testing.expectEqualStrings("hello", const_slice);
}

test "Normalized deinit frees resources" {
    var normalized = Normalized{
        .text = try Allocator.alloc(u8, 10),
        .map = try Allocator.alloc(i32, 10),
    };
    // Just ensure deinit doesn't crash - memory leak detection will catch issues
    normalized.deinit();
}

test "PretokenizeResult deinit frees token memory" {
    var result = PretokenizeResult{
        .tokens = .{},
        .ranges = .{},
    };

    // Add a token with null terminator
    const token_data = try Allocator.alloc(u8, 6);
    @memcpy(token_data, "hello\x00");
    try result.tokens.append(Allocator, .{ .ptr = token_data.ptr, .len = 5 });
    try result.ranges.append(Allocator, .{ .start = 0, .end = 5 });

    // Deinit should free the token data
    result.deinit();
}

test "sliceConst returns const slice from Token" {
    const data = try Allocator.alloc(u8, 5);
    defer Allocator.free(data);
    @memcpy(data, "hello");

    const token = Token{ .ptr = data.ptr, .len = 5 };
    const slice = token.sliceConst();

    try std.testing.expectEqual(@as(usize, 5), slice.len);
    try std.testing.expectEqualStrings("hello", slice);
}
