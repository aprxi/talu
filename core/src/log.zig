//! Structured Logging (OpenTelemetry-compliant)
//!
//! Produces structured log output following the OpenTelemetry Logging Data Model.
//!
//! Usage:
//!     const log = @import("log.zig");
//!
//!     // Info message (no code location)
//!     log.info("fetch", "Fetching file list", .{ .model_id = model_id });
//!
//!     // Debug message (includes code location)
//!     log.debug("fetch", "Cache lookup", .{}, @src());
//!
//!     // Error message (includes code location)
//!     log.err("fetch", "Model not found", .{ .model_id = model_id }, @src());
//!
//! Configuration:
//!     - Default level: WARN (silent for normal operation)
//!     - Use setLogLevel() to change programmatically (CLI uses this for -v flags)
//!     - Environment variables override: TALU_LOG_LEVEL, TALU_LOG_FORMAT
//!
//! Performance:
//!     - Level check happens FIRST via inline functions (~1-2 cycles when filtered)
//!     - Level/format are cached at first use (no syscalls per log call)
//!     - Logging is prohibited in hot paths (enforced by lint rule)

const std = @import("std");
const build_options = @import("build_options");

// =============================================================================
// Types
// =============================================================================

/// Severity levels (OpenTelemetry numeric ranges)
pub const Level = enum(u8) {
    trace = 1,
    debug = 5,
    info = 9,
    warn = 13,
    err = 17,
    fatal = 21,
    off = 255,

    pub fn toString(self: Level) []const u8 {
        return switch (self) {
            .trace => "TRACE",
            .debug => "DEBUG",
            .info => "INFO",
            .warn => "WARN",
            .err => "ERROR",
            .fatal => "FATAL",
            .off => "OFF",
        };
    }

    pub fn fromString(s: []const u8) Level {
        if (std.ascii.eqlIgnoreCase(s, "trace")) return .trace;
        if (std.ascii.eqlIgnoreCase(s, "debug")) return .debug;
        if (std.ascii.eqlIgnoreCase(s, "info")) return .info;
        if (std.ascii.eqlIgnoreCase(s, "warn") or std.ascii.eqlIgnoreCase(s, "warning")) return .warn;
        if (std.ascii.eqlIgnoreCase(s, "error") or std.ascii.eqlIgnoreCase(s, "err")) return .err;
        if (std.ascii.eqlIgnoreCase(s, "fatal")) return .fatal;
        if (std.ascii.eqlIgnoreCase(s, "off") or std.ascii.eqlIgnoreCase(s, "none")) return .off;
        return .warn; // Default for unknown strings
    }

    /// Returns true if code location should be included for this level
    pub fn includesCodeLocation(self: Level) bool {
        return switch (self) {
            .trace, .debug, .err, .fatal => true,
            .info, .warn, .off => false,
        };
    }
};

pub const Format = enum(u8) {
    json = 0,
    human = 1,

    pub fn fromString(s: []const u8) Format {
        if (std.ascii.eqlIgnoreCase(s, "json")) return .json;
        return .human;
    }
};

/// Source location info
pub const SourceLocation = struct {
    file: []const u8,
    line: u32,
};

// =============================================================================
// Cached Configuration
// =============================================================================

/// Cached log level - initialized on first use or via setLogLevel()
/// Default: .warn (silent for normal operation)
var cached_level: Level = .warn; // Single-threaded: set once at startup via getLogLevel()
var level_initialized: bool = false; // Single-threaded: set once at startup

/// Cached log format - initialized on first use or via setLogFormat()
var cached_format: Format = .human; // Single-threaded: set once at startup via getLogFormat()
var format_initialized: bool = false; // Single-threaded: set once at startup

/// Get the current log level (cached after first call)
pub fn getLogLevel() Level {
    if (level_initialized) return cached_level;

    // Check environment variables (one-time cost)
    if (std.posix.getenv("TALU_LOG_LEVEL")) |s| {
        cached_level = Level.fromString(s);
    } else if (std.posix.getenv("TALU_LOG")) |s| {
        cached_level = Level.fromString(s);
    }
    // else keep default (.warn)

    level_initialized = true;
    return cached_level;
}

/// Get the current log format (cached after first call)
pub fn getLogFormat() Format {
    if (format_initialized) return cached_format;

    // Check environment variable (one-time cost)
    if (std.posix.getenv("TALU_LOG_FORMAT")) |s| {
        cached_format = Format.fromString(s);
    } else {
        // Auto-detect: human for TTY, json for pipe
        cached_format = if (std.fs.File.stderr().isTty()) .human else .json;
    }

    format_initialized = true;
    return cached_format;
}

/// Set log level programmatically (used by CLI for -v/-vv/-vvv flags)
pub fn setLogLevel(level: Level) void {
    cached_level = level;
    level_initialized = true;
}

/// Set log format programmatically (used by CLI for --log-format flag)
pub fn setLogFormat(format: Format) void {
    cached_format = format;
    format_initialized = true;
}

fn getVersion() []const u8 {
    return build_options.version;
}

// =============================================================================
// Timestamp
// =============================================================================

fn writeTimestamp(writer: anytype) !void {
    // Get current time as nanoseconds since epoch
    const nanos = std.time.nanoTimestamp();
    const secs = @divFloor(nanos, std.time.ns_per_s);
    const nsec_part: u64 = @intCast(@mod(nanos, std.time.ns_per_s));

    // Convert to datetime
    const epoch_secs: u64 = @intCast(secs);
    const epoch = std.time.epoch.EpochSeconds{ .secs = epoch_secs };
    const day = epoch.getEpochDay();
    const year_day = day.calculateYearDay();
    const month_day = year_day.calculateMonthDay();
    const day_secs = epoch.getDaySeconds();

    const year = year_day.year;
    const month = month_day.month.numeric();
    const day_of_month = month_day.day_index + 1;
    const hour = day_secs.getHoursIntoDay();
    const minute = day_secs.getMinutesIntoHour();
    const second = day_secs.getSecondsIntoMinute();

    // RFC3339 with nanoseconds: 2024-01-09T14:32:15.123456789Z
    try writer.print("{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}.{d:0>9}Z", .{
        year,
        month,
        day_of_month,
        hour,
        minute,
        second,
        nsec_part,
    });
}

// =============================================================================
// JSON Output
// =============================================================================

/// Check if a type is string-like (can be coerced to []const u8)
fn isStringLike(comptime T: type) bool {
    const type_info = @typeInfo(T);
    return switch (type_info) {
        .pointer => |ptr| blk: {
            // Handle slices: []const u8
            if (ptr.size == .slice and ptr.child == u8) break :blk true;
            // Handle sentinel-terminated arrays: *const [N:0]u8 (string literals)
            if (ptr.size == .one) {
                const child_info = @typeInfo(ptr.child);
                if (child_info == .array) {
                    const arr = child_info.array;
                    if (arr.child == u8 and arr.sentinel_ptr != null) break :blk true;
                }
            }
            break :blk false;
        },
        else => false,
    };
}

fn writeJsonString(writer: anytype, s: []const u8) !void {
    try writer.writeByte('"');
    for (s) |c| {
        switch (c) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => {
                if (c < 0x20) {
                    try writer.print("\\u{x:0>4}", .{c});
                } else {
                    try writer.writeByte(c);
                }
            },
        }
    }
    try writer.writeByte('"');
}

fn writeJsonRecord(
    writer: anytype,
    level: Level,
    scope: []const u8,
    body: []const u8,
    src: ?SourceLocation,
    attrs: anytype,
) !void {
    try writer.writeAll("{\"timestamp\":\"");
    try writeTimestamp(writer);
    try writer.writeAll("\",\"severityText\":\"");
    try writer.writeAll(level.toString());
    try writer.writeAll("\",\"body\":");
    try writeJsonString(writer, body);

    // Attributes
    try writer.writeAll(",\"attributes\":{\"scope\":");
    try writeJsonString(writer, scope);

    // Custom attributes from struct
    const AttrType = @TypeOf(attrs);
    if (@typeInfo(AttrType) == .@"struct") {
        inline for (std.meta.fields(AttrType)) |field| {
            try writer.writeAll(",");
            try writeJsonString(writer, field.name);
            try writer.writeAll(":");
            const value = @field(attrs, field.name);
            const FieldType = @TypeOf(value);
            if (FieldType == []const u8) {
                try writeJsonString(writer, value);
            } else if (comptime isStringLike(FieldType)) {
                // Handle [:0]const u8 and other string-like types
                try writeJsonString(writer, value);
            } else if (@typeInfo(FieldType) == .int) {
                try writer.print("{d}", .{value});
            } else if (@typeInfo(FieldType) == .float) {
                try writer.print("{d}", .{value});
            } else {
                try writeJsonString(writer, @typeName(FieldType));
            }
        }
    }

    // Code location for debug/trace/error/fatal
    if (src) |s| {
        // Strip "core/src/" prefix
        const filepath = if (std.mem.startsWith(u8, s.file, "core/src/"))
            s.file["core/src/".len..]
        else
            s.file;
        try writer.writeAll(",\"code.filepath\":");
        try writeJsonString(writer, filepath);
        try writer.print(",\"code.lineno\":{d}", .{s.line});
    }

    // Resource
    try writer.writeAll("},\"resource\":{\"service.name\":\"talu\",\"service.version\":");
    try writeJsonString(writer, getVersion());
    try writer.writeAll("}}\n");
}

// =============================================================================
// Human Output
// =============================================================================

const Color = struct {
    const reset = "\x1b[0m";
    const dim = "\x1b[2m";
    const red = "\x1b[31m";
    const yellow = "\x1b[33m";
    const cyan = "\x1b[36m";
};

fn writeHumanRecord(
    writer: anytype,
    level: Level,
    scope: []const u8,
    body: []const u8,
    src: ?SourceLocation,
    attrs: anytype,
    use_colors: bool,
) !void {
    // Timestamp (time only for human)
    const nanos = std.time.nanoTimestamp();
    const secs = @divFloor(nanos, std.time.ns_per_s);
    const epoch_secs: u64 = @intCast(secs);
    const epoch = std.time.epoch.EpochSeconds{ .secs = epoch_secs };
    const day_secs = epoch.getDaySeconds();

    try writer.print("{d:0>2}:{d:0>2}:{d:0>2} ", .{
        day_secs.getHoursIntoDay(),
        day_secs.getMinutesIntoHour(),
        day_secs.getSecondsIntoMinute(),
    });

    // Level with color and padding
    const level_str = level.toString();
    if (use_colors) {
        const color = switch (level) {
            .trace, .debug => Color.dim,
            .info => "",
            .warn => Color.yellow,
            .err, .fatal => Color.red,
            .off => "",
        };
        if (color.len > 0) try writer.writeAll(color);
        try writer.print("{s: <5} ", .{level_str});
        if (color.len > 0) try writer.writeAll(Color.reset);
    } else {
        try writer.print("{s: <5} ", .{level_str});
    }

    // Scope
    if (use_colors) try writer.writeAll(Color.cyan);
    try writer.print("[{s}] ", .{scope});
    if (use_colors) try writer.writeAll(Color.reset);

    // Body
    try writer.writeAll(body);

    // Inline attributes for human readability
    // Keep it concise: <80 chars total, show only most useful info
    // Special case: progress/total pattern shows as "134/311"
    const AttrType = @TypeOf(attrs);
    if (@typeInfo(AttrType) == .@"struct") {
        const fields = std.meta.fields(AttrType);
        if (fields.len > 0) {
            // Check for progress/total pattern (common for loops)
            comptime var has_progress = false;
            comptime var has_total = false;
            inline for (fields) |field| {
                if (comptime std.mem.eql(u8, field.name, "progress")) has_progress = true;
                if (comptime std.mem.eql(u8, field.name, "total")) has_total = true;
            }

            try writer.writeAll(" (");

            if (comptime has_progress and has_total) {
                // Loop progress: "134/311"
                inline for (fields) |field| {
                    if (comptime std.mem.eql(u8, field.name, "progress")) {
                        try writer.print("{d}", .{@field(attrs, field.name)});
                    }
                }
                try writer.writeByte('/');
                inline for (fields) |field| {
                    if (comptime std.mem.eql(u8, field.name, "total")) {
                        try writer.print("{d}", .{@field(attrs, field.name)});
                    }
                }
            } else {
                // General case: show string attributes only (most useful)
                var first = true;
                inline for (fields) |field| {
                    const value = @field(attrs, field.name);
                    const FieldType = @TypeOf(value);

                    if (comptime isStringType(FieldType)) {
                        if (!first) try writer.writeAll(", ");
                        first = false;
                        const str = asString(value);
                        // Truncate long strings unless trace level (-vvv)
                        if (str.len > 40 and getLogLevel() != .trace) {
                            try writer.print("{s}...", .{str[0..37]});
                        } else {
                            try writer.print("{s}", .{str});
                        }
                    }
                }
                // If no strings, fall back to first numeric
                if (first) {
                    inline for (fields) |field| {
                        const value = @field(attrs, field.name);
                        const FieldType = @TypeOf(value);
                        if (comptime isIntType(FieldType)) {
                            if (!first) break;
                            try writer.print("{s}={d}", .{ field.name, value });
                            first = false;
                        }
                    }
                }
            }

            try writer.writeByte(')');
        }
    }

    // Code location is intentionally omitted from human format.
    // File/line info is only useful for machine processing (JSON format).
    // Human readers (developers or users) don't need it cluttering their terminal.
    _ = src;

    try writer.writeByte('\n');
}

// Type detection helpers for comptime
fn isStringType(comptime T: type) bool {
    return T == []const u8 or T == []u8 or
        (@typeInfo(T) == .pointer and @typeInfo(T).pointer.size == .slice and
        (@typeInfo(T).pointer.child == u8));
}

fn isIntType(comptime T: type) bool {
    return @typeInfo(T) == .int or @typeInfo(T) == .comptime_int;
}

fn asString(value: anytype) []const u8 {
    const T = @TypeOf(value);
    if (T == []const u8) return value;
    if (T == []u8) return value;
    if (@typeInfo(T) == .pointer and @typeInfo(T).pointer.size == .slice) {
        return value;
    }
    return "";
}

// =============================================================================
// Core Write Function (not inlined - called only when level check passes)
// =============================================================================

fn writeLogImpl(
    level: Level,
    scope: []const u8,
    body: []const u8,
    src: ?SourceLocation,
    attrs: anytype,
) void {
    const stderr = std.fs.File.stderr();
    const format = getLogFormat();

    var buf: [8192]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    const writer = fbs.writer();

    switch (format) {
        .json => writeJsonRecord(writer, level, scope, body, src, attrs) catch return,
        .human => writeHumanRecord(writer, level, scope, body, src, attrs, stderr.isTty()) catch return,
    }

    stderr.writeAll(fbs.getWritten()) catch {};
}

// =============================================================================
// Public API (inline functions - level check happens FIRST at call site)
// =============================================================================

/// Log at TRACE level (includes code location)
/// Cost when filtered: ~1-2 cycles (inline level check only)
pub inline fn trace(
    scope: []const u8,
    body: []const u8,
    attrs: anytype,
    comptime src: std.builtin.SourceLocation,
) void {
    if (@intFromEnum(Level.trace) < @intFromEnum(getLogLevel())) return;
    writeLogImpl(.trace, scope, body, .{ .file = src.file, .line = src.line }, attrs);
}

/// Log at DEBUG level (includes code location)
/// Cost when filtered: ~1-2 cycles (inline level check only)
pub inline fn debug(
    scope: []const u8,
    body: []const u8,
    attrs: anytype,
    comptime src: std.builtin.SourceLocation,
) void {
    if (@intFromEnum(Level.debug) < @intFromEnum(getLogLevel())) return;
    writeLogImpl(.debug, scope, body, .{ .file = src.file, .line = src.line }, attrs);
}

/// Log at INFO level (no code location)
/// Cost when filtered: ~1-2 cycles (inline level check only)
pub inline fn info(scope: []const u8, body: []const u8, attrs: anytype) void {
    if (@intFromEnum(Level.info) < @intFromEnum(getLogLevel())) return;
    writeLogImpl(.info, scope, body, null, attrs);
}

/// Log at WARN level (no code location)
/// Cost when filtered: ~1-2 cycles (inline level check only)
pub inline fn warn(scope: []const u8, body: []const u8, attrs: anytype) void {
    if (@intFromEnum(Level.warn) < @intFromEnum(getLogLevel())) return;
    writeLogImpl(.warn, scope, body, null, attrs);
}

/// Log at ERROR level (includes code location)
/// Cost when filtered: ~1-2 cycles (inline level check only)
pub inline fn err(
    scope: []const u8,
    body: []const u8,
    attrs: anytype,
    comptime src: std.builtin.SourceLocation,
) void {
    if (@intFromEnum(Level.err) < @intFromEnum(getLogLevel())) return;
    writeLogImpl(.err, scope, body, .{ .file = src.file, .line = src.line }, attrs);
}

/// Log at FATAL level (includes code location)
/// Cost when filtered: ~1-2 cycles (inline level check only)
pub inline fn fatal(
    scope: []const u8,
    body: []const u8,
    attrs: anytype,
    comptime src: std.builtin.SourceLocation,
) void {
    if (@intFromEnum(Level.fatal) < @intFromEnum(getLogLevel())) return;
    writeLogImpl(.fatal, scope, body, .{ .file = src.file, .line = src.line }, attrs);
}

// =============================================================================
// Tests
// =============================================================================

test "Level.fromString" {
    try std.testing.expectEqual(Level.trace, Level.fromString("trace"));
    try std.testing.expectEqual(Level.debug, Level.fromString("DEBUG"));
    try std.testing.expectEqual(Level.info, Level.fromString("info"));
    try std.testing.expectEqual(Level.warn, Level.fromString("warn"));
    try std.testing.expectEqual(Level.warn, Level.fromString("warning"));
    try std.testing.expectEqual(Level.err, Level.fromString("error"));
    try std.testing.expectEqual(Level.err, Level.fromString("err"));
    try std.testing.expectEqual(Level.fatal, Level.fromString("fatal"));
    try std.testing.expectEqual(Level.off, Level.fromString("off"));
    try std.testing.expectEqual(Level.warn, Level.fromString("unknown")); // Default is warn now
}

test "Level.includesCodeLocation" {
    try std.testing.expect(Level.trace.includesCodeLocation());
    try std.testing.expect(Level.debug.includesCodeLocation());
    try std.testing.expect(!Level.info.includesCodeLocation());
    try std.testing.expect(!Level.warn.includesCodeLocation());
    try std.testing.expect(Level.err.includesCodeLocation());
    try std.testing.expect(Level.fatal.includesCodeLocation());
}

test "setLogLevel changes cached level" {
    const original = getLogLevel();
    defer setLogLevel(original);

    setLogLevel(.trace);
    try std.testing.expectEqual(Level.trace, getLogLevel());

    setLogLevel(.err);
    try std.testing.expectEqual(Level.err, getLogLevel());
}

test "setLogFormat changes cached format" {
    const original = getLogFormat();
    defer setLogFormat(original);

    setLogFormat(.json);
    try std.testing.expectEqual(Format.json, getLogFormat());

    setLogFormat(.human);
    try std.testing.expectEqual(Format.human, getLogFormat());
}

test "getVersion returns build version" {
    const version = getVersion();
    try std.testing.expect(version.len > 0);
}
