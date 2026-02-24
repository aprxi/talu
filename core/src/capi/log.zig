//! C API for log configuration
//!
//! Allows bindings to control log level and format programmatically.
//! Used by CLI for -v/-vv/-vvv flags and --log-format option.

const log = @import("../log.zig");

/// Log levels (matches log.Level enum values)
pub const LogLevel = enum(c_int) {
    trace = 1,
    debug = 5,
    info = 9,
    warn = 13,
    err = 17,
    fatal = 21,
    off = 255,
};

/// Log formats (matches log.Format enum values)
pub const LogFormat = enum(c_int) {
    json = 0,
    human = 1,
};

/// Callback receiving core log records.
pub const CLogCallback = ?*const fn (
    level: c_int,
    scope_ptr: [*]const u8,
    scope_len: usize,
    body_ptr: [*]const u8,
    body_len: usize,
    attrs_json_ptr: [*]const u8,
    attrs_json_len: usize,
    file_ptr: [*]const u8,
    file_len: usize,
    line: u32,
    user_data: ?*anyopaque,
) callconv(.c) void;

/// Set the log level.
///
/// Levels:
///   1 = TRACE (most verbose, includes code locations)
///   5 = DEBUG (includes code locations)
///   9 = INFO (progress, status)
///  13 = WARN (default - silent for normal operation)
///  17 = ERROR (failures only)
///  21 = FATAL (unrecoverable errors)
/// 255 = OFF (completely silent)
///
/// Typical CLI mapping:
///   (none) -> 13 (WARN)
///   -v     ->  9 (INFO)
///   -vv    ->  5 (DEBUG)
///   -vvv   ->  1 (TRACE)
pub export fn talu_set_log_level(level: c_int) callconv(.c) void {
    const log_level: log.Level = @enumFromInt(@as(u8, @intCast(@min(level, 255))));
    log.setLogLevel(log_level);
}

/// Set the log format.
///
/// Formats:
///   0 = JSON (OpenTelemetry compliant, machine-readable)
///   1 = HUMAN (colored, human-readable)
///
/// By default, format is auto-detected:
///   - HUMAN when stderr is a TTY
///   - JSON when stderr is piped/redirected
pub export fn talu_set_log_format(format: c_int) callconv(.c) void {
    const log_format: log.Format = if (format == 0) .json else .human;
    log.setLogFormat(log_format);
}

/// Get the current log level.
pub export fn talu_get_log_level() callconv(.c) c_int {
    return @intFromEnum(log.getLogLevel());
}

/// Get the current log format.
pub export fn talu_get_log_format() callconv(.c) c_int {
    return @intFromEnum(log.getLogFormat());
}

/// Set a scope filter pattern (glob-style).
///
/// The filter is matched against the fully-qualified scope (e.g. "core::inference").
/// Supports trailing `*` for prefix match: "core::*" matches all core scopes.
/// Empty string or "*" disables filtering (shows all).
///
/// Note: this only filters Zig core logs. Rust server logs have their own filter.
pub export fn talu_set_log_filter(ptr: [*]const u8, len: usize) callconv(.c) void {
    log.setLogFilter(ptr[0..len]);
}

/// Set a callback for core log records.
///
/// Passing `null` disables callback forwarding.
pub export fn talu_set_log_callback(callback: CLogCallback, user_data: ?*anyopaque) callconv(.c) void {
    log.setLogCallback(callback, user_data);
}
