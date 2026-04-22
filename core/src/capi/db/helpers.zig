//! Shared DB C-API helpers.
//!
//! Small validation and conversion utilities used across DB submodules
//! (table, vector, ops). Kept here to avoid duplication.

const std = @import("std");
const capi_error = @import("../error.zig");
const error_codes = @import("../error_codes.zig");

/// Convert an optional null-terminated C string to an optional slice.
pub fn optSlice(s: ?[*:0]const u8) ?[]const u8 {
    return if (s) |p| std.mem.span(p) else null;
}

/// Validate and convert a C db_path to a slice. Sets error on failure.
pub fn validateDbPath(db_path: ?[*:0]const u8) ?[]const u8 {
    const slice = std.mem.sliceTo(db_path orelse {
        capi_error.setErrorWithCode(.invalid_argument, "db_path is null", .{});
        return null;
    }, 0);
    if (slice.len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, "db_path is empty", .{});
        return null;
    }
    return slice;
}

/// Validate a required C string parameter. Sets error on null or empty.
pub fn validateRequiredArg(s: ?[*:0]const u8, comptime arg_name: []const u8) ?[]const u8 {
    const slice = std.mem.sliceTo(s orelse {
        capi_error.setErrorWithCode(.invalid_argument, arg_name ++ " is null", .{});
        return null;
    }, 0);
    if (slice.len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, arg_name ++ " is empty", .{});
        return null;
    }
    return slice;
}

/// Set error and return invalid_argument code (for compact null validation).
pub fn setArgError(comptime msg: []const u8) i32 {
    capi_error.setErrorWithCode(.invalid_argument, msg, .{});
    return @intFromEnum(error_codes.ErrorCode.invalid_argument);
}
