//! Core error context storage.
//!
//! This thread-local context is owned by core and can be set by internal modules
//! before they return an error. Boundary layers (like C API) can consume this
//! context to enrich reported error messages.

const std = @import("std");

const CONTEXT_BUF_SIZE: usize = 256;

threadlocal var context_buffer: [CONTEXT_BUF_SIZE]u8 = undefined;
threadlocal var context_length: usize = 0;

/// Set internal error context. Call before returning an error.
pub fn setContext(comptime fmt: []const u8, args: anytype) void {
    var stream = std.io.fixedBufferStream(&context_buffer);
    stream.writer().print(fmt, args) catch {};
    context_length = stream.pos;
}

/// Clear internal error context.
pub fn clearContext() void {
    context_length = 0;
}

/// Consume and clear the current context payload.
pub fn consumeContext() ?[]const u8 {
    if (context_length == 0) return null;
    defer context_length = 0;
    return context_buffer[0..context_length];
}

test "consumeContext clears stored value" {
    clearContext();
    setContext("value={d}", .{@as(u32, 7)});
    const first = consumeContext() orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("value=7", first);
    try std.testing.expect(consumeContext() == null);
}
