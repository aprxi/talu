//! Integration tests for tokenizer.VocabListC
//!
//! VocabListC is a C-compatible struct that converts VocabEntry slices
//! into parallel arrays suitable for FFI. It owns copied token strings.

const std = @import("std");
const main = @import("main");

const VocabListC = main.tokenizer.VocabListC;
const VocabEntryC = main.tokenizer.VocabEntryC;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "VocabListC type is accessible" {
    const T = VocabListC;
    _ = T;
}

test "VocabListC is a struct" {
    const info = @typeInfo(VocabListC);
    try std.testing.expect(info == .@"struct");
}

test "VocabListC has expected fields" {
    const info = @typeInfo(VocabListC);
    const fields = info.@"struct".fields;

    var has_tokens = false;
    var has_lengths = false;
    var has_ids = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "tokens")) has_tokens = true;
        if (comptime std.mem.eql(u8, field.name, "lengths")) has_lengths = true;
        if (comptime std.mem.eql(u8, field.name, "ids")) has_ids = true;
    }

    try std.testing.expect(has_tokens);
    try std.testing.expect(has_lengths);
    try std.testing.expect(has_ids);
}

// =============================================================================
// Method Tests
// =============================================================================

test "VocabListC has fromEntries method" {
    try std.testing.expect(@hasDecl(VocabListC, "fromEntries"));
}

test "VocabListC has deinit method" {
    try std.testing.expect(@hasDecl(VocabListC, "deinit"));
}

test "VocabListC has count method" {
    try std.testing.expect(@hasDecl(VocabListC, "count"));
}

// =============================================================================
// fromEntries Tests
// =============================================================================

test "VocabListC.fromEntries with empty slice" {
    const allocator = std.testing.allocator;
    const entries: []const VocabEntryC = &.{};

    var list = try VocabListC.fromEntries(allocator, entries);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), list.count());
    try std.testing.expectEqual(@as(usize, 0), list.tokens.len);
    try std.testing.expectEqual(@as(usize, 0), list.lengths.len);
    try std.testing.expectEqual(@as(usize, 0), list.ids.len);
}

test "VocabListC.fromEntries with single entry" {
    const allocator = std.testing.allocator;
    const entries = [_]VocabEntryC{
        .{ .token = "hello", .id = 42 },
    };

    var list = try VocabListC.fromEntries(allocator, &entries);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), list.count());
    try std.testing.expectEqual(@as(u32, 5), list.lengths[0]);
    try std.testing.expectEqual(@as(u32, 42), list.ids[0]);
    try std.testing.expectEqualStrings("hello", std.mem.span(list.tokens[0]));
}

test "VocabListC.fromEntries with multiple entries" {
    const allocator = std.testing.allocator;
    const entries = [_]VocabEntryC{
        .{ .token = "foo", .id = 1 },
        .{ .token = "bar", .id = 2 },
        .{ .token = "baz", .id = 3 },
    };

    var list = try VocabListC.fromEntries(allocator, &entries);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 3), list.count());

    try std.testing.expectEqualStrings("foo", std.mem.span(list.tokens[0]));
    try std.testing.expectEqual(@as(u32, 3), list.lengths[0]);
    try std.testing.expectEqual(@as(u32, 1), list.ids[0]);

    try std.testing.expectEqualStrings("bar", std.mem.span(list.tokens[1]));
    try std.testing.expectEqual(@as(u32, 3), list.lengths[1]);
    try std.testing.expectEqual(@as(u32, 2), list.ids[1]);

    try std.testing.expectEqualStrings("baz", std.mem.span(list.tokens[2]));
    try std.testing.expectEqual(@as(u32, 3), list.lengths[2]);
    try std.testing.expectEqual(@as(u32, 3), list.ids[2]);
}

// =============================================================================
// deinit Tests
// =============================================================================

test "VocabListC.deinit clears arrays" {
    const allocator = std.testing.allocator;
    const entries = [_]VocabEntryC{
        .{ .token = "test", .id = 100 },
    };

    var list = try VocabListC.fromEntries(allocator, &entries);
    list.deinit(allocator);

    // After deinit, arrays should be empty
    try std.testing.expectEqual(@as(usize, 0), list.tokens.len);
    try std.testing.expectEqual(@as(usize, 0), list.lengths.len);
    try std.testing.expectEqual(@as(usize, 0), list.ids.len);
}

// =============================================================================
// count Tests
// =============================================================================

test "VocabListC.count returns correct count" {
    const allocator = std.testing.allocator;
    const entries = [_]VocabEntryC{
        .{ .token = "a", .id = 0 },
        .{ .token = "b", .id = 1 },
    };

    var list = try VocabListC.fromEntries(allocator, &entries);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), list.count());
}
