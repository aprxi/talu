//! JSON Value Extraction Helpers
//!
//! Utility functions for extracting typed values from std.json.ObjectMap
//! with default values. Reduces boilerplate in config parsing.

const std = @import("std");

/// Extract an integer from JSON value, returning default if missing or wrong type.
pub fn getInt(comptime T: type, object: std.json.ObjectMap, key: []const u8, default_value: T) T {
    const value = object.get(key) orelse return default_value;
    return switch (value) {
        .integer => |int_val| if (int_val >= 0) @intCast(int_val) else default_value,
        else => default_value,
    };
}

/// Extract an optional integer from JSON value.
pub fn getOptionalInt(comptime T: type, object: std.json.ObjectMap, key: []const u8) ?T {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .integer => |int_val| if (int_val >= 0) @intCast(int_val) else null,
        else => null,
    };
}

/// Extract a float from JSON value, returning default if missing or wrong type.
pub fn getFloat(comptime T: type, object: std.json.ObjectMap, key: []const u8, default_value: T) T {
    const value = object.get(key) orelse return default_value;
    return switch (value) {
        .float => |f| @floatCast(f),
        .integer => |int_val| @floatFromInt(int_val),
        else => default_value,
    };
}

/// Extract a boolean from JSON value, returning default if missing or wrong type.
pub fn getBool(object: std.json.ObjectMap, key: []const u8, default_value: bool) bool {
    const value = object.get(key) orelse return default_value;
    return switch (value) {
        .bool => |b| b,
        else => default_value,
    };
}

/// Extract a string from JSON value.
fn getString(object: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .string => |s| s,
        else => null,
    };
}

/// Extract an array of integers from JSON value (handles both single int and array).
/// Caller owns returned slice.
pub fn getIntArray(comptime T: type, allocator: std.mem.Allocator, object: std.json.ObjectMap, key: []const u8) ![]T {
    const value = object.get(key) orelse return &.{};
    return switch (value) {
        .integer => |int_val| blk: {
            if (int_val < 0) break :blk &.{};
            const ids = try allocator.alloc(T, 1);
            ids[0] = @intCast(int_val);
            break :blk ids;
        },
        .array => |arr| blk: {
            var ids = try allocator.alloc(T, arr.items.len);
            var count: usize = 0;
            for (arr.items) |item| {
                if (item == .integer and item.integer >= 0) {
                    ids[count] = @intCast(item.integer);
                    count += 1;
                }
            }
            if (count < ids.len) {
                ids = allocator.realloc(ids, count) catch ids;
            }
            break :blk ids[0..count];
        },
        else => &.{},
    };
}

// =============================================================================
// Tests
// =============================================================================

test "getInt - valid integer" {
    const json_str = "{\"count\": 42}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = getInt(u32, parsed.value.object, "count", 0);
    try std.testing.expectEqual(@as(u32, 42), result);
}

test "getInt - missing key returns default" {
    const json_str = "{}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = getInt(u32, parsed.value.object, "count", 99);
    try std.testing.expectEqual(@as(u32, 99), result);
}

test "getInt - wrong type returns default" {
    const json_str = "{\"count\": \"not a number\"}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = getInt(u32, parsed.value.object, "count", 99);
    try std.testing.expectEqual(@as(u32, 99), result);
}

test "getInt - negative value returns default" {
    const json_str = "{\"count\": -5}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = getInt(u32, parsed.value.object, "count", 99);
    try std.testing.expectEqual(@as(u32, 99), result);
}

test "getInt - zero value" {
    const json_str = "{\"count\": 0}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = getInt(u32, parsed.value.object, "count", 99);
    try std.testing.expectEqual(@as(u32, 0), result);
}

test "getOptionalInt - valid integer" {
    const json_str = "{\"count\": 42}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = getOptionalInt(u32, parsed.value.object, "count");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(u32, 42), result.?);
}

test "getOptionalInt - missing key returns null" {
    const json_str = "{}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = getOptionalInt(u32, parsed.value.object, "count");
    try std.testing.expect(result == null);
}

test "getOptionalInt - wrong type returns null" {
    const json_str = "{\"count\": \"not a number\"}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = getOptionalInt(u32, parsed.value.object, "count");
    try std.testing.expect(result == null);
}

test "getOptionalInt - negative value returns null" {
    const json_str = "{\"count\": -5}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = getOptionalInt(u32, parsed.value.object, "count");
    try std.testing.expect(result == null);
}

test "getFloat - valid float" {
    const json_str = "{\"value\": 3.14}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = getFloat(f32, parsed.value.object, "value", 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 3.14), result, 0.001);
}

test "getFloat - integer converts to float" {
    const json_str = "{\"value\": 42}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = getFloat(f32, parsed.value.object, "value", 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 42.0), result, 0.001);
}

test "getFloat - missing key returns default" {
    const json_str = "{}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = getFloat(f32, parsed.value.object, "value", 99.9);
    try std.testing.expectApproxEqAbs(@as(f32, 99.9), result, 0.001);
}

test "getFloat - wrong type returns default" {
    const json_str = "{\"value\": \"not a number\"}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = getFloat(f32, parsed.value.object, "value", 99.9);
    try std.testing.expectApproxEqAbs(@as(f32, 99.9), result, 0.001);
}

test "getFloat - negative float" {
    const json_str = "{\"value\": -2.5}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = getFloat(f32, parsed.value.object, "value", 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, -2.5), result, 0.001);
}

test "getBool - true value" {
    const json_str = "{\"flag\": true}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = getBool(parsed.value.object, "flag", false);
    try std.testing.expect(result == true);
}

test "getBool - false value" {
    const json_str = "{\"flag\": false}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = getBool(parsed.value.object, "flag", true);
    try std.testing.expect(result == false);
}

test "getBool - missing key returns default" {
    const json_str = "{}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = getBool(parsed.value.object, "flag", true);
    try std.testing.expect(result == true);
}

test "getBool - wrong type returns default" {
    const json_str = "{\"flag\": \"yes\"}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = getBool(parsed.value.object, "flag", true);
    try std.testing.expect(result == true);
}

test "getString - valid string" {
    const json_str = "{\"name\": \"hello\"}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = getString(parsed.value.object, "name");
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("hello", result.?);
}

test "getString - missing key returns null" {
    const json_str = "{}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = getString(parsed.value.object, "name");
    try std.testing.expect(result == null);
}

test "getString - wrong type returns null" {
    const json_str = "{\"name\": 42}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = getString(parsed.value.object, "name");
    try std.testing.expect(result == null);
}

test "getIntArray - single integer" {
    const json_str = "{\"ids\": 42}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = try getIntArray(u32, std.testing.allocator, parsed.value.object, "ids");
    defer std.testing.allocator.free(result);

    try std.testing.expectEqual(@as(usize, 1), result.len);
    try std.testing.expectEqual(@as(u32, 42), result[0]);
}

test "getIntArray - array of integers" {
    const json_str = "{\"ids\": [1, 2, 3, 4, 5]}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = try getIntArray(u32, std.testing.allocator, parsed.value.object, "ids");
    defer std.testing.allocator.free(result);

    try std.testing.expectEqual(@as(usize, 5), result.len);
    try std.testing.expectEqual(@as(u32, 1), result[0]);
    try std.testing.expectEqual(@as(u32, 5), result[4]);
}

test "getIntArray - empty array" {
    const json_str = "{\"ids\": []}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = try getIntArray(u32, std.testing.allocator, parsed.value.object, "ids");
    defer if (result.len > 0) std.testing.allocator.free(result);

    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test "getIntArray - missing key returns empty" {
    const json_str = "{}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = try getIntArray(u32, std.testing.allocator, parsed.value.object, "ids");
    defer if (result.len > 0) std.testing.allocator.free(result);

    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test "getIntArray - negative integer returns empty" {
    const json_str = "{\"ids\": -5}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = try getIntArray(u32, std.testing.allocator, parsed.value.object, "ids");
    defer if (result.len > 0) std.testing.allocator.free(result);

    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test "getIntArray - mixed array filters negatives" {
    const json_str = "{\"ids\": [1, -2, 3, -4, 5]}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = try getIntArray(u32, std.testing.allocator, parsed.value.object, "ids");
    defer std.testing.allocator.free(result);

    try std.testing.expectEqual(@as(usize, 3), result.len);
    try std.testing.expectEqual(@as(u32, 1), result[0]);
    try std.testing.expectEqual(@as(u32, 3), result[1]);
    try std.testing.expectEqual(@as(u32, 5), result[2]);
}

test "getIntArray - array with non-integers" {
    const json_str = "{\"ids\": [1, \"two\", 3, null, 5]}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = try getIntArray(u32, std.testing.allocator, parsed.value.object, "ids");
    defer std.testing.allocator.free(result);

    try std.testing.expectEqual(@as(usize, 3), result.len);
    try std.testing.expectEqual(@as(u32, 1), result[0]);
    try std.testing.expectEqual(@as(u32, 3), result[1]);
    try std.testing.expectEqual(@as(u32, 5), result[2]);
}

test "getIntArray - wrong type returns empty" {
    const json_str = "{\"ids\": \"not an array\"}";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();

    const result = try getIntArray(u32, std.testing.allocator, parsed.value.object, "ids");
    defer if (result.len > 0) std.testing.allocator.free(result);

    try std.testing.expectEqual(@as(usize, 0), result.len);
}
