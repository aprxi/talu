//! Memory allocation utilities for C API.
//!
//! Provides functions for allocating and freeing memory using c_allocator,
//! enabling FFI callers to allocate buffers that Zig code can later free.

const std = @import("std");

/// Allocate a byte buffer using c_allocator.
///
/// The caller must free the returned pointer using talu_free_string()
/// with the same length. Returns null on allocation failure.
///
/// This function enables FFI callers (e.g., Python ctypes) to allocate
/// memory that Zig code can safely free with c_allocator.
pub export fn talu_alloc_string(len: usize) callconv(.c) ?[*]u8 {
    if (len == 0) return null;
    const slice = std.heap.c_allocator.alloc(u8, len) catch return null;
    return slice.ptr;
}

/// Free a buffer allocated by talu_alloc_string().
///
/// The ptr and len must match a previous talu_alloc_string() call.
/// Passing null is safe (no-op). Passing mismatched len is undefined behavior.
pub export fn talu_free_string(ptr: ?[*]u8, len: usize) callconv(.c) void {
    if (ptr) |p| {
        if (len == 0) return;
        std.heap.c_allocator.free(p[0..len]);
    }
}

// =============================================================================
// Tests
// =============================================================================

test "talu_alloc_string: basic allocation and free" {
    const len: usize = 64;
    const ptr = talu_alloc_string(len);
    try std.testing.expect(ptr != null);

    // Write to the memory to verify it's usable
    @memset(ptr.?[0..len], 'x');
    try std.testing.expectEqual(@as(u8, 'x'), ptr.?[0]);
    try std.testing.expectEqual(@as(u8, 'x'), ptr.?[len - 1]);

    // Free should not crash
    talu_free_string(ptr, len);
}

test "talu_alloc_string: zero length returns null" {
    const ptr = talu_alloc_string(0);
    try std.testing.expect(ptr == null);
}

test "talu_free_string: null pointer is no-op" {
    // Should not crash
    talu_free_string(null, 100);
    talu_free_string(null, 0);
}

test "talu_free_string: zero length with non-null is no-op" {
    // Allocate something valid first
    const ptr = talu_alloc_string(10);
    try std.testing.expect(ptr != null);

    // Free with zero length should not free (but also not crash)
    // Note: This is a guard against misuse - don't rely on this behavior
    talu_free_string(ptr, 0);

    // Now actually free it
    talu_free_string(ptr, 10);
}

test "talu_alloc_string: can store and retrieve string data" {
    const test_str = "Hello, World!";
    const len = test_str.len + 1; // Include null terminator

    const ptr = talu_alloc_string(len);
    try std.testing.expect(ptr != null);

    // Copy string including null terminator
    @memcpy(ptr.?[0..test_str.len], test_str);
    ptr.?[test_str.len] = 0;

    // Verify contents
    try std.testing.expectEqualStrings(test_str, ptr.?[0..test_str.len]);

    talu_free_string(ptr, len);
}
