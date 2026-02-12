//! JSON parsing helpers with size limits and consistent errors.

const std = @import("std");
const log = @import("../../log.zig");

var parse_id_counter: std.atomic.Value(u64) = std.atomic.Value(u64).init(0);

const default_max_string_bytes: usize = 256 * 1024;
const default_max_value_bytes: usize = 1 * 1024 * 1024;

pub const ParseOptions = struct {
    /// Maximum input size in bytes. Parsing is rejected before it starts
    /// if input exceeds this. Caller must set this consciously.
    max_size_bytes: usize,

    /// Forward-compat: ignore fields not in the target struct.
    /// Use true for external formats (HF configs, policy files).
    /// Use false for internal formats where unknown fields signal a bug.
    ignore_unknown_fields: bool = true,

    /// Maximum size for any single JSON value token (string/number/etc).
    /// Defaults to min(default_max_value_bytes, max_size_bytes) when null.
    /// Set to max_size_bytes to allow a value as large as the full payload.
    max_value_bytes: ?usize = null,

    /// Maximum nesting depth of arrays/objects.
    max_depth: usize = 256,

    /// Maximum length of any JSON string (measured in source bytes).
    /// Defaults to min(default_max_string_bytes, max_size_bytes) when null.
    /// Set to max_size_bytes to allow a string as large as the full payload.
    max_string_bytes: ?usize = null,
};

pub const ParseError = error{
    InputTooLarge,
    InputTooDeep,
    StringTooLong,
    InvalidJson,
    OutOfMemory,
};

fn logRejected(reason: []const u8, parse_id: u64, input_bytes: usize, max_bytes: usize) void {
    log.info("json", "Rejected JSON", .{
        .reason = reason,
        .parse_id = parse_id,
        .input_bytes = input_bytes,
        .max_bytes = max_bytes,
    });
}

fn logAccepted(parse_id: u64, input_bytes: usize, max_bytes: usize, target: []const u8) void {
    log.debug("json", "Parsed JSON", .{
        .target = target,
        .parse_id = parse_id,
        .input_bytes = input_bytes,
        .max_bytes = max_bytes,
    }, @src());
}

fn enforceLimits(json: []const u8, options: ParseOptions, parse_id: u64) ParseError!void {
    const max_string = @min(options.max_string_bytes orelse default_max_string_bytes, options.max_size_bytes);
    var depth: usize = 0;
    var in_string = false;
    var escape = false;
    var current_string_len: usize = 0;

    for (json) |byte| {
        if (in_string) {
            if (escape) {
                escape = false;
                current_string_len += 1;
                if (current_string_len > max_string) {
                    logRejected("string_too_long", parse_id, json.len, max_string);
                    return ParseError.StringTooLong;
                }
                continue;
            }
            switch (byte) {
                '\\' => {
                    escape = true;
                },
                '"' => {
                    in_string = false;
                },
                else => {
                    current_string_len += 1;
                    if (current_string_len > max_string) {
                        logRejected("string_too_long", parse_id, json.len, max_string);
                        return ParseError.StringTooLong;
                    }
                },
            }
            continue;
        }

        switch (byte) {
            '"' => {
                in_string = true;
                current_string_len = 0;
            },
            '{', '[' => {
                depth += 1;
                if (depth > options.max_depth) {
                    logRejected("input_too_deep", parse_id, json.len, options.max_depth);
                    return ParseError.InputTooDeep;
                }
            },
            '}', ']' => {
                if (depth == 0) {
                    logRejected("invalid_json", parse_id, json.len, options.max_size_bytes);
                    return ParseError.InvalidJson;
                }
                depth -= 1;
            },
            else => {},
        }
    }

    if (in_string or escape or depth != 0) {
        logRejected("invalid_json", parse_id, json.len, options.max_size_bytes);
        return ParseError.InvalidJson;
    }
}

fn mapParseError(err: anytype) ParseError {
    return switch (err) {
        error.OutOfMemory => ParseError.OutOfMemory,
        else => ParseError.InvalidJson,
    };
}

/// Parse JSON bytes into std.json.Value.
/// Enforces size limit, logs rejections.
pub fn parseValue(
    allocator: std.mem.Allocator,
    json: []const u8,
    options: ParseOptions,
) ParseError!std.json.Parsed(std.json.Value) {
    const parse_id = parse_id_counter.fetchAdd(1, .monotonic);
    if (json.len > options.max_size_bytes) {
        logRejected("input_too_large", parse_id, json.len, options.max_size_bytes);
        return ParseError.InputTooLarge;
    }
    try enforceLimits(json, options, parse_id);
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, json, .{
        .max_value_len = @min(options.max_value_bytes orelse default_max_value_bytes, options.max_size_bytes),
    }) catch |err| {
        if (err == error.OutOfMemory) {
            logRejected("out_of_memory", parse_id, json.len, options.max_size_bytes);
            return ParseError.OutOfMemory;
        }
        if (err == error.ValueTooLong) {
            logRejected("value_too_long", parse_id, json.len, options.max_size_bytes);
            return ParseError.InvalidJson;
        }
        logRejected("invalid_json", parse_id, json.len, options.max_size_bytes);
        return mapParseError(err);
    };
    logAccepted(parse_id, json.len, options.max_size_bytes, "Value");
    return parsed;
}

/// Parse JSON bytes into a typed struct.
/// Enforces size limit, maps ignore_unknown_fields, logs rejections.
pub fn parseStruct(
    allocator: std.mem.Allocator,
    comptime T: type,
    json: []const u8,
    options: ParseOptions,
) ParseError!std.json.Parsed(T) {
    const parse_id = parse_id_counter.fetchAdd(1, .monotonic);
    if (json.len > options.max_size_bytes) {
        logRejected("input_too_large", parse_id, json.len, options.max_size_bytes);
        return ParseError.InputTooLarge;
    }
    try enforceLimits(json, options, parse_id);
    const parsed = std.json.parseFromSlice(T, allocator, json, .{
        .ignore_unknown_fields = options.ignore_unknown_fields,
        .allocate = .alloc_always,
        .max_value_len = @min(options.max_value_bytes orelse default_max_value_bytes, options.max_size_bytes),
    }) catch |err| {
        if (err == error.OutOfMemory) {
            logRejected("out_of_memory", parse_id, json.len, options.max_size_bytes);
            return ParseError.OutOfMemory;
        }
        if (err == error.ValueTooLong) {
            logRejected("value_too_long", parse_id, json.len, options.max_size_bytes);
            return ParseError.InvalidJson;
        }
        logRejected("invalid_json", parse_id, json.len, options.max_size_bytes);
        return mapParseError(err);
    };
    logAccepted(parse_id, json.len, options.max_size_bytes, @typeName(T));
    return parsed;
}

/// Parse a JSON value into a typed struct.
/// Enforces size limit using the provided input_size_bytes for logging and checks.
pub fn parseStructFromValue(
    allocator: std.mem.Allocator,
    comptime T: type,
    value: std.json.Value,
    input_size_bytes: usize,
    options: ParseOptions,
) ParseError!std.json.Parsed(T) {
    const parse_id = parse_id_counter.fetchAdd(1, .monotonic);
    if (input_size_bytes > options.max_size_bytes) {
        logRejected("input_too_large", parse_id, input_size_bytes, options.max_size_bytes);
        return ParseError.InputTooLarge;
    }
    const parsed = std.json.parseFromValue(T, allocator, value, .{
        .ignore_unknown_fields = options.ignore_unknown_fields,
    }) catch |err| {
        if (err == error.OutOfMemory) {
            logRejected("out_of_memory", parse_id, input_size_bytes, options.max_size_bytes);
            return ParseError.OutOfMemory;
        }
        logRejected("invalid_json", parse_id, input_size_bytes, options.max_size_bytes);
        return mapParseError(err);
    };
    logAccepted(parse_id, input_size_bytes, options.max_size_bytes, @typeName(T));
    return parsed;
}

/// Extract a single string field from JSON object. Returns owned memory.
/// Convenience for the common pattern: parse, get field, dupe, free.
/// Returns null if field is missing or not a string.
pub fn extractStringField(
    allocator: std.mem.Allocator,
    json: []const u8,
    field_name: []const u8,
    options: ParseOptions,
) ParseError!?[]u8 {
    const parsed = try parseValue(allocator, json, options);
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const val = parsed.value.object.get(field_name) orelse return null;
    return switch (val) {
        .string => |s| allocator.dupe(u8, s) catch ParseError.OutOfMemory,
        else => null,
    };
}

test "parseValue enforces size limit" {
    const parsed = parseValue(std.testing.allocator, "{\"a\":1}", .{ .max_size_bytes = 4 });
    try std.testing.expectError(ParseError.InputTooLarge, parsed);
}

test "parseValue rejects malformed JSON" {
    const parsed = parseValue(std.testing.allocator, "{\"a\":", .{ .max_size_bytes = 64 });
    try std.testing.expectError(ParseError.InvalidJson, parsed);
}

test "parseValue enforces max depth" {
    const json = "[[[1]]]";
    const parsed = parseValue(std.testing.allocator, json, .{
        .max_size_bytes = 64,
        .max_depth = 2,
    });
    try std.testing.expectError(ParseError.InputTooDeep, parsed);
}

test "parseValue enforces max string length" {
    const parsed = parseValue(std.testing.allocator, "{\"s\":\"abcd\"}", .{
        .max_size_bytes = 64,
        .max_string_bytes = 3,
    });
    try std.testing.expectError(ParseError.StringTooLong, parsed);
}

test "parseValue enforces default max string length" {
    const alloc = std.testing.allocator;
    const string_len = default_max_string_bytes + 1;
    const prefix = "{\"s\":\"";
    const suffix = "\"}";
    const json_len = prefix.len + string_len + suffix.len;

    var json_buf = try alloc.alloc(u8, json_len);
    defer alloc.free(json_buf);

    std.mem.copyForwards(u8, json_buf[0..prefix.len], prefix);
    @memset(json_buf[prefix.len .. prefix.len + string_len], 'a');
    std.mem.copyForwards(u8, json_buf[prefix.len + string_len ..], suffix);

    const parsed = parseValue(alloc, json_buf, .{ .max_size_bytes = json_len });
    try std.testing.expectError(ParseError.StringTooLong, parsed);
}

test "parseValue accepts valid JSON within limit" {
    var parsed = try parseValue(std.testing.allocator, "{\"a\":1}", .{ .max_size_bytes = 64 });
    defer parsed.deinit();
    try std.testing.expect(parsed.value == .object);
}

test "parseStruct maps ignore_unknown_fields" {
    const Example = struct {
        value: u8,
    };
    var parsed = try parseStruct(std.testing.allocator, Example, "{\"value\": 7, \"extra\": 1}", .{
        .max_size_bytes = 64,
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();
    try std.testing.expectEqual(@as(u8, 7), parsed.value.value);
}

test "parseStruct rejects unknown fields when strict" {
    const Example = struct {
        value: u8,
    };
    const parsed = parseStruct(std.testing.allocator, Example, "{\"value\": 7, \"extra\": 1}", .{
        .max_size_bytes = 64,
        .ignore_unknown_fields = false,
    });
    try std.testing.expectError(ParseError.InvalidJson, parsed);
}

test "parseStructFromValue parses from Value" {
    const Example = struct {
        value: u8,
    };
    var parsed_value = try parseValue(std.testing.allocator, "{\"value\": 9}", .{ .max_size_bytes = 64 });
    defer parsed_value.deinit();
    var parsed = try parseStructFromValue(std.testing.allocator, Example, parsed_value.value, 10, .{
        .max_size_bytes = 64,
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();
    try std.testing.expectEqual(@as(u8, 9), parsed.value.value);
}

test "extractStringField returns owned string" {
    const alloc = std.testing.allocator;
    const result = try extractStringField(alloc, "{\"command\":\"ls\"}", "command", .{ .max_size_bytes = 64 });
    defer if (result) |s| alloc.free(s);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("ls", result.?);
}

test "extractStringField returns null on missing field" {
    const result = try extractStringField(std.testing.allocator, "{\"a\":1}", "command", .{ .max_size_bytes = 64 });
    try std.testing.expect(result == null);
}

test "extractStringField returns null on non-string field" {
    const result = try extractStringField(std.testing.allocator, "{\"command\":2}", "command", .{ .max_size_bytes = 64 });
    try std.testing.expect(result == null);
}

test "extractStringField handles unicode escapes correctly" {
    const alloc = std.testing.allocator;
    const result = try extractStringField(alloc, "{\"command\":\"echo \\u0048ello\"}", "command", .{ .max_size_bytes = 128 });
    defer if (result) |s| alloc.free(s);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("echo Hello", result.?);
}

test "extractStringField handles escaped quotes correctly" {
    const alloc = std.testing.allocator;
    const result = try extractStringField(alloc, "{\"command\":\"echo \\\"hello\\\"\"}", "command", .{ .max_size_bytes = 128 });
    defer if (result) |s| alloc.free(s);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("echo \"hello\"", result.?);
}
