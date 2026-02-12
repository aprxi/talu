//! Integration tests for io.json.

const std = @import("std");
const main = @import("main");
const io = main.io;

test "parseValue enforces size limit (integration)" {
    const parsed = io.json.parseValue(std.testing.allocator, "{\"a\":1}", .{ .max_size_bytes = 4 });
    try std.testing.expectError(io.json.ParseError.InputTooLarge, parsed);
}

test "parseStruct respects ignore_unknown_fields (integration)" {
    const Example = struct {
        value: u8,
    };
    var parsed = try io.json.parseStruct(std.testing.allocator, Example, "{\"value\": 2, \"extra\": 1}", .{
        .max_size_bytes = 64,
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();
    try std.testing.expectEqual(@as(u8, 2), parsed.value.value);
}

test "parseValue works with json_helpers extraction (integration)" {
    const json_helpers = io.json_helpers;
    var parsed = try io.json.parseValue(std.testing.allocator, "{\"count\": 5}", .{ .max_size_bytes = 64 });
    defer parsed.deinit();
    const value = json_helpers.getInt(u8, parsed.value.object, "count", 0);
    try std.testing.expectEqual(@as(u8, 5), value);
}
