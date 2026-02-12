//! String Utilities
//!
//! String duplication and manipulation helpers for tokenizer internals.
//! Handles C-string interop with null-termination.

const std = @import("std");
const types = @import("types.zig");

const Allocator = types.Allocator;

pub fn strdup_range(start_ptr: [*]const u8, len: usize) ?[*:0]u8 {
    const copy_buf = Allocator.alloc(u8, len + 1) catch return null;
    @memcpy(copy_buf[0..len], start_ptr[0..len]);
    copy_buf[len] = 0;
    return @ptrCast(copy_buf.ptr);
}

pub fn tokenizer_strdup(input: ?[*:0]const u8) ?[*:0]u8 {
    if (input == null) return null;
    const input_slice = std.mem.sliceTo(input.?, 0);
    return strdup_range(input.?, input_slice.len);
}

pub fn dupTokenString(token_bytes: []const u8) ?[*:0]u8 {
    const token_copy = Allocator.allocSentinel(u8, token_bytes.len, 0) catch return null;
    if (token_bytes.len > 0) @memcpy(token_copy, token_bytes);
    return token_copy;
}

// =============================================================================
// Tests
// =============================================================================

test "strdup_range copies correctly" {
    const input = "hello";
    const result = strdup_range(input.ptr, input.len);
    try std.testing.expect(result != null);
    defer Allocator.free(std.mem.sliceTo(result.?, 0));

    const slice = std.mem.sliceTo(result.?, 0);
    try std.testing.expectEqualStrings(input, slice);
}

test "tokenizer_strdup copies null-terminated string" {
    const input: [*:0]const u8 = "world";
    const result = tokenizer_strdup(input);
    try std.testing.expect(result != null);
    defer Allocator.free(std.mem.sliceTo(result.?, 0));

    const slice = std.mem.sliceTo(result.?, 0);
    try std.testing.expectEqualStrings("world", slice);
}

test "dupTokenString copies slice correctly" {
    const input = "test123";
    const result = dupTokenString(input);
    try std.testing.expect(result != null);
    defer Allocator.free(std.mem.sliceTo(result.?, 0));

    const slice = std.mem.sliceTo(result.?, 0);
    try std.testing.expectEqualStrings(input, slice);
}
