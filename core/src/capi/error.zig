//! Thread-local error context for C API.
//!
//! Provides stable error codes and human-readable messages for FFI callers.

const std = @import("std");
const error_codes = @import("error_codes.zig");
const ErrorCode = error_codes.ErrorCode;
const errorToCode = error_codes.errorToCode;

const ERROR_BUF_SIZE: usize = 512;
const TRUNCATED_SUFFIX = "... (truncated)";
const MIN_ERROR_CONTENT: usize = 64;

comptime {
    std.debug.assert(ERROR_BUF_SIZE > TRUNCATED_SUFFIX.len + 1 + MIN_ERROR_CONTENT);
}

threadlocal var error_buffer: [ERROR_BUF_SIZE]u8 = undefined; // Safe: setError writes before any read
threadlocal var error_length: usize = 0;
threadlocal var last_error_code: ErrorCode = .ok;
threadlocal var is_truncated: bool = false;

// Internal error context - set by internal code before returning errors
const CONTEXT_BUF_SIZE: usize = 256;
threadlocal var context_buffer: [CONTEXT_BUF_SIZE]u8 = undefined; // Safe: setContext writes before any read
threadlocal var context_length: usize = 0;

/// Set internal error context. Call this before returning an error to provide
/// additional diagnostic information. The context is accumulated silently (not
/// logged) and only surfaces when setError() is called at the C API boundary.
///
/// Example:
/// ```zig
/// if (k == 0) {
///     setContext("MoE k=0, must be >= 1");
///     return error.InvalidMoEConfig;
/// }
/// ```
pub fn setContext(comptime fmt: []const u8, args: anytype) void {
    var stream = std.io.fixedBufferStream(&context_buffer);
    stream.writer().print(fmt, args) catch {};
    context_length = stream.pos;
}

/// Clear internal error context.
pub fn clearContext() void {
    context_length = 0;
}

/// Get current context (for use in setError).
fn getContext() ?[]const u8 {
    if (context_length == 0) return null;
    return context_buffer[0..context_length];
}

/// Store error for retrieval by talu_take_last_error().
pub fn setError(err: anyerror, comptime fmt: []const u8, args: anytype) void {
    setErrorWithCode(errorToCode(err), fmt, args);
}

/// Store error with explicit error code.
/// Use this when you need to specify a specific error code rather than
/// having it derived from the Zig error type.
/// If internal context was set via setContext(), it will be appended.
pub fn setErrorWithCode(code: ErrorCode, comptime fmt: []const u8, args: anytype) void {
    last_error_code = code;
    is_truncated = false;

    const max_message_len = ERROR_BUF_SIZE - 1 - TRUNCATED_SUFFIX.len;
    var buffer_stream = std.io.fixedBufferStream(error_buffer[0..max_message_len]);
    const writer = buffer_stream.writer();

    // Write the main message
    writer.print(fmt, args) catch {
        is_truncated = true;
    };

    // Append internal context if set
    if (!is_truncated) {
        if (getContext()) |ctx| {
            writer.print(" ({s})", .{ctx}) catch {
                is_truncated = true;
            };
        }
    }

    // Clear context after use
    clearContext();

    if (is_truncated) {
        @memcpy(error_buffer[buffer_stream.pos..][0..TRUNCATED_SUFFIX.len], TRUNCATED_SUFFIX);
        error_buffer[buffer_stream.pos + TRUNCATED_SUFFIX.len] = 0;
        error_length = buffer_stream.pos + TRUNCATED_SUFFIX.len;
    } else {
        error_buffer[buffer_stream.pos] = 0;
        error_length = buffer_stream.pos;
    }
}

/// Set error with cause chain (flattened into message).
/// IMPORTANT: cause_msg must NOT alias the TLS error buffer.
/// NOTE: Currently unused, but kept for future error chaining support.
fn setErrorWithCause(
    err: anyerror,
    cause_msg: ?[]const u8,
    comptime fmt: []const u8,
    args: anytype,
) void {
    last_error_code = errorToCode(err);
    is_truncated = false;

    const max_message_len = ERROR_BUF_SIZE - 1 - TRUNCATED_SUFFIX.len;
    var buffer_stream = std.io.fixedBufferStream(error_buffer[0..max_message_len]);
    const writer = buffer_stream.writer();

    writer.print(fmt, args) catch {
        is_truncated = true;
    };

    if (!is_truncated) {
        if (cause_msg) |cause_message| {
            writer.print("\n  caused by: {s}", .{cause_message}) catch {
                is_truncated = true;
            };
        }
    }

    if (is_truncated) {
        @memcpy(error_buffer[buffer_stream.pos..][0..TRUNCATED_SUFFIX.len], TRUNCATED_SUFFIX);
        error_buffer[buffer_stream.pos + TRUNCATED_SUFFIX.len] = 0;
        error_length = buffer_stream.pos + TRUNCATED_SUFFIX.len;
    } else {
        error_buffer[buffer_stream.pos] = 0;
        error_length = buffer_stream.pos;
    }
}

/// Clear error state.
pub fn clearError() void {
    last_error_code = .ok;
    error_length = 0;
    is_truncated = false;
}

/// Set error from a simple string message.
pub fn set_last_error(msg: []const u8) void {
    last_error_code = .internal_error;
    is_truncated = false;

    const copy_len = @min(msg.len, ERROR_BUF_SIZE - 1);
    @memcpy(error_buffer[0..copy_len], msg[0..copy_len]);
    error_buffer[copy_len] = 0;
    error_length = copy_len;
}

/// Set error from an error value.
pub fn set_last_error_from_err(err: anyerror) void {
    setError(err, "{s}", .{@errorName(err)});
}

/// Get the last error message (for internal use).
pub fn get_last_error() ?[*:0]const u8 {
    return talu_last_error();
}

/// Retrieve last error message.
/// Returns NUL-terminated string valid until next talu_* call on same thread.
pub export fn talu_last_error() callconv(.c) ?[*:0]const u8 {
    if (last_error_code == .ok) return null;
    return @ptrCast(&error_buffer[0]);
}

/// Retrieve last error code.
pub export fn talu_last_error_code() callconv(.c) i32 {
    return @intFromEnum(last_error_code);
}

/// Clear error state.
pub export fn talu_clear_error() callconv(.c) void {
    clearError();
}

/// Get recommended buffer size for error messages.
pub export fn talu_error_buf_size() callconv(.c) usize {
    return ERROR_BUF_SIZE;
}

/// Atomically retrieve error code and message, then clear.
pub export fn talu_take_last_error(
    out_buf: ?[*]u8,
    out_buf_size: usize,
    out_code: *i32,
) callconv(.c) usize {
    out_code.* = @intFromEnum(last_error_code);

    if (last_error_code == .ok) return 0;

    const message_len = error_length;
    if (out_buf == null) return message_len + 1;

    const copy_len = @min(message_len, out_buf_size -| 1);
    if (copy_len > 0) {
        @memcpy(out_buf.?[0..copy_len], error_buffer[0..copy_len]);
    }
    if (out_buf_size > 0) {
        out_buf.?[copy_len] = 0;
    }

    clearError();
    return copy_len;
}

test "talu_take_last_error copies message and code" {
    setError(error.FileNotFound, "Test error: {s}", .{"details"});

    var scratch_buf: [256]u8 = undefined;
    var code: i32 = 0;
    const len = talu_take_last_error(&scratch_buf, scratch_buf.len, &code);

    try std.testing.expectEqual(@as(i32, 500), code);
    try std.testing.expect(len > 0);
    try std.testing.expectEqualStrings("Test error: details", scratch_buf[0..len]);
    try std.testing.expect(talu_last_error() == null);
}

test "talu_take_last_error: query mode does not clear" {
    setError(error.FileNotFound, "Hello", .{});

    var code: i32 = 0;
    const required = talu_take_last_error(null, 0, &code);

    try std.testing.expectEqual(@as(usize, 6), required);
    try std.testing.expect(talu_last_error() != null);
}

test "talu_last_error: truncation appends indicator" {
    var long_path: [1000]u8 = undefined;
    @memset(&long_path, 'a');

    setError(error.FileNotFound, "Path: {s}", .{&long_path});

    const msg = talu_last_error();
    try std.testing.expect(msg != null);
    try std.testing.expect(std.mem.endsWith(u8, std.mem.span(msg.?), "... (truncated)"));
}

test "setContext adds internal context to error message" {
    // Internal code sets context before returning error
    setContext("k={d}, must be >= 1", .{@as(u32, 0)});

    // C API boundary catches error and calls setError
    setError(error.InvalidArgument, "MoE configuration invalid", .{});

    const msg = talu_last_error();
    try std.testing.expect(msg != null);
    try std.testing.expectEqualStrings("MoE configuration invalid (k=0, must be >= 1)", std.mem.span(msg.?));

    // Context is cleared after use
    clearError();
    setError(error.InvalidArgument, "Another error", .{});
    const msg2 = talu_last_error();
    try std.testing.expectEqualStrings("Another error", std.mem.span(msg2.?));
}

test "setContext without setError does nothing" {
    clearError();
    setContext("some context", .{});

    // If setError is never called, context is just sitting there
    try std.testing.expect(talu_last_error() == null);
    try std.testing.expectEqual(@as(i32, 0), talu_last_error_code());
}
