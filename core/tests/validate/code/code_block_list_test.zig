//! Integration tests for CodeBlockList.
//!
//! Tests the CodeBlockList type exported from core/src/validate/code/root.zig.

const std = @import("std");
const main = @import("main");
const code = main.validate.code;
const CodeBlock = code.CodeBlock;
const CodeBlockList = code.CodeBlockList;

// ============================================================================
// Lifecycle Tests
// ============================================================================

test "CodeBlockList init and deinit" {
    var list = CodeBlockList.init(std.testing.allocator);
    defer list.deinit();

    try std.testing.expectEqual(@as(usize, 0), list.count());
}

// ============================================================================
// Append and Count Tests
// ============================================================================

test "CodeBlockList append increments count" {
    var list = CodeBlockList.init(std.testing.allocator);
    defer list.deinit();

    try list.append(.{
        .index = 0,
        .fence_start = 0,
        .fence_end = 10,
        .language_start = 3,
        .language_end = 6,
        .content_start = 7,
        .content_end = 9,
        .complete = true,
    });

    try std.testing.expectEqual(@as(usize, 1), list.count());

    try list.append(.{
        .index = 1,
        .fence_start = 20,
        .fence_end = 40,
        .language_start = 23,
        .language_end = 27,
        .content_start = 28,
        .content_end = 38,
        .complete = true,
    });

    try std.testing.expectEqual(@as(usize, 2), list.count());
}

// ============================================================================
// Get Tests
// ============================================================================

test "CodeBlockList get returns correct block" {
    var list = CodeBlockList.init(std.testing.allocator);
    defer list.deinit();

    try list.append(.{
        .index = 0,
        .fence_start = 0,
        .fence_end = 10,
        .language_start = 3,
        .language_end = 6,
        .content_start = 7,
        .content_end = 9,
        .complete = true,
    });

    try list.append(.{
        .index = 1,
        .fence_start = 20,
        .fence_end = 30,
        .language_start = 23,
        .language_end = 26,
        .content_start = 27,
        .content_end = 29,
        .complete = false,
    });

    const block0 = list.get(0);
    try std.testing.expect(block0 != null);
    try std.testing.expectEqual(@as(u32, 0), block0.?.index);
    try std.testing.expect(block0.?.complete);

    const block1 = list.get(1);
    try std.testing.expect(block1 != null);
    try std.testing.expectEqual(@as(u32, 1), block1.?.index);
    try std.testing.expect(!block1.?.complete);
}

test "CodeBlockList get returns null for out of bounds" {
    var list = CodeBlockList.init(std.testing.allocator);
    defer list.deinit();

    try std.testing.expect(list.get(0) == null);
    try std.testing.expect(list.get(100) == null);
}

// ============================================================================
// toJson Tests
// ============================================================================

test "CodeBlockList toJson serializes empty list" {
    var list = CodeBlockList.init(std.testing.allocator);
    defer list.deinit();

    const json = try list.toJson(std.testing.allocator);
    defer std.testing.allocator.free(json);

    try std.testing.expectEqualStrings("[]", json);
}

test "CodeBlockList toJson serializes single block" {
    var list = CodeBlockList.init(std.testing.allocator);
    defer list.deinit();

    try list.append(.{
        .index = 0,
        .fence_start = 0,
        .fence_end = 25,
        .language_start = 3,
        .language_end = 9,
        .content_start = 10,
        .content_end = 22,
        .complete = true,
    });

    const json = try list.toJson(std.testing.allocator);
    defer std.testing.allocator.free(json);

    // Verify it's valid JSON array with one object
    try std.testing.expect(std.mem.startsWith(u8, json, "[{"));
    try std.testing.expect(std.mem.endsWith(u8, json, "}]"));
    try std.testing.expect(std.mem.indexOf(u8, json, "\"index\":0") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"complete\":true") != null);
}

test "CodeBlockList toJson serializes multiple blocks" {
    var list = CodeBlockList.init(std.testing.allocator);
    defer list.deinit();

    try list.append(.{
        .index = 0,
        .fence_start = 0,
        .fence_end = 10,
        .language_start = 3,
        .language_end = 6,
        .content_start = 7,
        .content_end = 9,
        .complete = true,
    });

    try list.append(.{
        .index = 1,
        .fence_start = 20,
        .fence_end = 30,
        .language_start = 23,
        .language_end = 26,
        .content_start = 27,
        .content_end = 29,
        .complete = false,
    });

    const json = try list.toJson(std.testing.allocator);
    defer std.testing.allocator.free(json);

    // Verify JSON has two objects separated by comma
    try std.testing.expect(std.mem.indexOf(u8, json, "},{") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"index\":0") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"index\":1") != null);
}
