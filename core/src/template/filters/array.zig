//! Array/List Filters
//!
//! Filters that operate on arrays and sequences: access, transformation, sorting.

const std = @import("std");
const types = @import("types.zig");

const TemplateInput = types.TemplateInput;
const EvalError = types.EvalError;
const Evaluator = types.Evaluator;
const Expr = types.Expr;

// ============================================================================
// Element Access
// ============================================================================

pub fn filterFirst(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    return switch (value) {
        .array => |a| if (a.len > 0) a[0] else .none,
        .string => |s| if (s.len > 0) .{ .string = s[0..1] } else .none,
        else => EvalError.TypeError,
    };
}

pub fn filterLast(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    return switch (value) {
        .array => |a| if (a.len > 0) a[a.len - 1] else .none,
        .string => |s| if (s.len > 0) .{ .string = s[s.len - 1 ..] } else .none,
        else => EvalError.TypeError,
    };
}

pub fn filterRandom(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    switch (value) {
        .array => |arr| {
            if (arr.len == 0) return .none;
            // Use a simple deterministic selection for reproducibility in templates
            // For true randomness, users should handle this in application code
            const idx = @mod(@as(usize, @truncate(@as(u128, @bitCast(std.time.nanoTimestamp())))), arr.len);
            return arr[idx];
        },
        .string => |s| {
            if (s.len == 0) return .{ .string = "" };
            const idx = @mod(@as(usize, @truncate(@as(u128, @bitCast(std.time.nanoTimestamp())))), s.len);
            return .{ .string = s[idx .. idx + 1] };
        },
        else => return EvalError.TypeError,
    }
}

// ============================================================================
// Combining
// ============================================================================

pub fn filterJoin(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    const arr = switch (value) {
        .array => |a| a,
        else => return EvalError.TypeError,
    };

    var sep: []const u8 = "";
    if (args.len > 0) {
        const sep_val = try e.evalExpr(args[0]);
        sep = switch (sep_val) {
            .string => |s| s,
            else => return EvalError.TypeError,
        };
    }

    const arena = e.ctx.arena.allocator();
    var result = std.ArrayListUnmanaged(u8){};
    for (arr, 0..) |item, i| {
        if (i > 0) result.appendSlice(arena, sep) catch return EvalError.OutOfMemory;
        const str = item.asString(arena) catch return EvalError.OutOfMemory;
        result.appendSlice(arena, str) catch return EvalError.OutOfMemory;
    }
    return .{ .string = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

// ============================================================================
// Ordering
// ============================================================================

pub fn filterReverse(e: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    switch (value) {
        .array => |a| {
            var result = e.ctx.arena.allocator().alloc(TemplateInput, a.len) catch return EvalError.OutOfMemory;
            for (a, 0..) |item, i| {
                result[a.len - 1 - i] = item;
            }
            return .{ .array = result };
        },
        .string => |s| {
            // UTF-8 aware string reversal: reverse codepoints, not bytes
            const arena = e.ctx.arena.allocator();

            // First pass: count codepoints and collect their byte slices
            var codepoint_starts = std.ArrayListUnmanaged(usize){};
            var iter = std.unicode.Utf8View.initUnchecked(s).iterator();
            var pos: usize = 0;
            while (iter.nextCodepointSlice()) |cp_slice| {
                codepoint_starts.append(arena, pos) catch return EvalError.OutOfMemory;
                pos += cp_slice.len;
            }

            if (codepoint_starts.items.len == 0) {
                return .{ .string = "" };
            }

            // Build reversed string by iterating codepoints in reverse
            var result = arena.alloc(u8, s.len) catch return EvalError.OutOfMemory;
            var write_pos: usize = 0;
            var i: usize = codepoint_starts.items.len;
            while (i > 0) {
                i -= 1;
                const start = codepoint_starts.items[i];
                const end = if (i + 1 < codepoint_starts.items.len) codepoint_starts.items[i + 1] else s.len;
                const cp_bytes = s[start..end];
                @memcpy(result[write_pos .. write_pos + cp_bytes.len], cp_bytes);
                write_pos += cp_bytes.len;
            }

            return .{ .string = result };
        },
        else => return EvalError.TypeError,
    }
}

pub fn filterSort(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    const arr = switch (value) {
        .array => |a| a,
        else => return EvalError.TypeError,
    };

    // Check for reverse argument (first positional or keyword arg)
    var reverse = false;
    if (args.len > 0) {
        const arg_val = try e.evalExpr(args[0]);
        if (arg_val == .boolean) {
            reverse = arg_val.boolean;
        }
    }

    const arena = e.ctx.arena.allocator();
    const result = arena.alloc(TemplateInput, arr.len) catch return EvalError.OutOfMemory;
    @memcpy(result, arr);

    // Sort by string representation
    std.mem.sort(TemplateInput, result, arena, struct {
        fn lessThan(alloc: std.mem.Allocator, a: TemplateInput, b: TemplateInput) bool {
            const a_str = a.asString(alloc) catch return false;
            const b_str = b.asString(alloc) catch return false;
            return std.mem.order(u8, a_str, b_str) == .lt;
        }
    }.lessThan);

    // Reverse if requested
    if (reverse) {
        std.mem.reverse(TemplateInput, result);
    }

    return .{ .array = result };
}

// ============================================================================
// Uniqueness
// ============================================================================

pub fn filterUnique(e: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    const arr = switch (value) {
        .array => |a| a,
        else => return EvalError.TypeError,
    };

    const arena = e.ctx.arena.allocator();
    var result = std.ArrayListUnmanaged(TemplateInput){};
    var seen = std.StringHashMapUnmanaged(void){};

    for (arr) |item| {
        // Use JSON representation as key for uniqueness
        const key = item.toJson(arena) catch return EvalError.OutOfMemory;
        if (!seen.contains(key)) {
            seen.put(arena, key, {}) catch return EvalError.OutOfMemory;
            result.append(arena, item) catch return EvalError.OutOfMemory;
        }
    }

    return .{ .array = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

// ============================================================================
// Slicing and Batching
// ============================================================================

pub fn filterBatch(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    const arr = switch (value) {
        .array => |a| a,
        else => return EvalError.TypeError,
    };

    if (args.len < 1) return EvalError.TypeError;

    const size_val = try e.evalExpr(args[0]);
    const size: usize = switch (size_val) {
        .integer => |int_val| if (int_val > 0) @intCast(int_val) else return EvalError.TypeError,
        else => return EvalError.TypeError,
    };

    const arena = e.ctx.arena.allocator();
    var result = std.ArrayListUnmanaged(TemplateInput){};
    var batch = std.ArrayListUnmanaged(TemplateInput){};

    for (arr) |item| {
        batch.append(arena, item) catch return EvalError.OutOfMemory;
        if (batch.items.len >= size) {
            result.append(arena, .{ .array = batch.toOwnedSlice(arena) catch return EvalError.OutOfMemory }) catch return EvalError.OutOfMemory;
            batch = .{};
        }
    }

    // Add remaining items
    if (batch.items.len > 0) {
        result.append(arena, .{ .array = batch.toOwnedSlice(arena) catch return EvalError.OutOfMemory }) catch return EvalError.OutOfMemory;
    }

    return .{ .array = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

pub fn filterSlice(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    // slice(n) divides the sequence into n groups
    const arr = switch (value) {
        .array => |a| a,
        else => return EvalError.TypeError,
    };

    if (args.len < 1) return EvalError.TypeError;

    const slices_val = try e.evalExpr(args[0]);
    const num_slices: usize = switch (slices_val) {
        .integer => |int_val| if (int_val > 0) @intCast(int_val) else return EvalError.TypeError,
        else => return EvalError.TypeError,
    };

    const arena = e.ctx.arena.allocator();
    var result = std.ArrayListUnmanaged(TemplateInput){};

    if (arr.len == 0 or num_slices == 0) {
        return .{ .array = &.{} };
    }

    // Calculate items per slice
    const items_per_slice = arr.len / num_slices;
    const extra = arr.len % num_slices;
    var offset: usize = 0;

    for (0..num_slices) |slice_idx| {
        // First 'extra' slices get one more item
        const slice_size = items_per_slice + (if (slice_idx < extra) @as(usize, 1) else @as(usize, 0));
        const end = offset + slice_size;

        const slice_items = arena.alloc(TemplateInput, slice_size) catch return EvalError.OutOfMemory;
        @memcpy(slice_items, arr[offset..end]);

        result.append(arena, .{ .array = slice_items }) catch return EvalError.OutOfMemory;
        offset = end;
    }

    return .{ .array = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

// ============================================================================
// Dict/Map Operations
// ============================================================================

pub fn filterDictsort(e: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    const map_value = switch (value) {
        .map => |map| map,
        else => return EvalError.TypeError,
    };

    const arena = e.ctx.arena.allocator();

    // Collect keys and sort them
    var keys = std.ArrayListUnmanaged([]const u8){};
    var it = map_value.iterator();
    while (it.next()) |entry| {
        keys.append(arena, entry.key_ptr.*) catch return EvalError.OutOfMemory;
    }

    std.mem.sort([]const u8, keys.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lessThan);

    // Build array of [key, value] pairs
    var result = std.ArrayListUnmanaged(TemplateInput){};
    for (keys.items) |key| {
        const entry_value = map_value.get(key) orelse .none;
        const pair = arena.alloc(TemplateInput, 2) catch return EvalError.OutOfMemory;
        pair[0] = .{ .string = key };
        pair[1] = entry_value;
        result.append(arena, .{ .array = pair }) catch return EvalError.OutOfMemory;
    }

    return .{ .array = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

pub fn filterItems(e: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    const map_value = switch (value) {
        .map => |map| map,
        else => return EvalError.TypeError,
    };

    const arena = e.ctx.arena.allocator();
    var result = std.ArrayListUnmanaged(TemplateInput){};

    var it = map_value.iterator();
    while (it.next()) |entry| {
        const pair = arena.alloc(TemplateInput, 2) catch return EvalError.OutOfMemory;
        pair[0] = .{ .string = entry.key_ptr.* };
        pair[1] = entry.value_ptr.*;
        result.append(arena, .{ .array = pair }) catch return EvalError.OutOfMemory;
    }

    return .{ .array = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

// ============================================================================
// Unit Tests
// ============================================================================

const TemplateParser = @import("../eval.zig").TemplateParser;

test "filterFirst - array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .string = "first" },
        .{ .string = "second" },
    };
    const input = TemplateInput{ .array = &items };
    const result = try filterFirst(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("first", result.string);
}

test "filterFirst - string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "hello" };
    const result = try filterFirst(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("h", result.string);
}

test "filterLast - array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .string = "first" },
        .{ .string = "last" },
    };
    const input = TemplateInput{ .array = &items };
    const result = try filterLast(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("last", result.string);
}

test "filterLast - string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "hello" };
    const result = try filterLast(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("o", result.string);
}

test "filterJoin" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .string = "a" },
        .{ .string = "b" },
        .{ .string = "c" },
    };
    const input = TemplateInput{ .array = &items };

    const sep_expr = Expr{ .string = ", " };
    const args = [_]*const Expr{&sep_expr};
    const result = try filterJoin(&eval_ctx,input, &args);
    try std.testing.expectEqualStrings("a, b, c", result.string);
}

test "filterReverse - array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 2 },
        .{ .integer = 3 },
    };
    const input = TemplateInput{ .array = &items };
    const result = try filterReverse(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(i64, 3), result.array[0].integer);
    try std.testing.expectEqual(@as(i64, 2), result.array[1].integer);
    try std.testing.expectEqual(@as(i64, 1), result.array[2].integer);
}

test "filterReverse - string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "hello" };
    const result = try filterReverse(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("olleh", result.string);
}

test "filterReverse - UTF-8 CJK" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    // Each CJK character is 3 bytes - must reverse codepoints, not bytes
    const input = TemplateInput{ .string = "你好" };
    const result = try filterReverse(&eval_ctx, input, &.{});
    try std.testing.expectEqualStrings("好你", result.string);
}

test "filterReverse - UTF-8 emoji" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    // Emoji is 4 bytes - must reverse codepoints, not bytes
    const input = TemplateInput{ .string = "a🌍b" };
    const result = try filterReverse(&eval_ctx, input, &.{});
    try std.testing.expectEqualStrings("b🌍a", result.string);
}

test "filterReverse - UTF-8 mixed" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    // Mixed ASCII and multi-byte characters
    const input = TemplateInput{ .string = "hello世界" };
    const result = try filterReverse(&eval_ctx, input, &.{});
    try std.testing.expectEqualStrings("界世olleh", result.string);
}

test "filterUnique" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 2 },
        .{ .integer = 1 },
        .{ .integer = 3 },
        .{ .integer = 2 },
    };
    const input = TemplateInput{ .array = &items };
    const result = try filterUnique(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(usize, 3), result.array.len);
}

test "filterSort" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 3 },
        .{ .integer = 1 },
        .{ .integer = 2 },
    };
    const input = TemplateInput{ .array = &items };
    const result = try filterSort(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(i64, 1), result.array[0].integer);
    try std.testing.expectEqual(@as(i64, 2), result.array[1].integer);
    try std.testing.expectEqual(@as(i64, 3), result.array[2].integer);
}

test "filterItems from map" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    var map = std.StringHashMapUnmanaged(TemplateInput){};
    try map.put(ctx.arena.allocator(), "a", .{ .integer = 1 });
    const input = TemplateInput{ .map = map };
    const result = try filterItems(&eval_ctx, input, &.{});
    try std.testing.expectEqual(@as(usize, 1), result.array.len);
}

test "filterRandom - from array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .string = "a" },
        .{ .string = "b" },
        .{ .string = "c" },
    };
    const input = TemplateInput{ .array = &items };
    const result = try filterRandom(&eval_ctx, input, &.{});
    // Result should be one of the items
    try std.testing.expect(result == .string);
    try std.testing.expect(result.string.len == 1);
}

test "filterRandom - from string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "abc" };
    const result = try filterRandom(&eval_ctx, input, &.{});
    try std.testing.expect(result == .string);
    try std.testing.expect(result.string.len == 1);
}

test "filterFirst - empty array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{};
    const input = TemplateInput{ .array = &items };
    const result = try filterFirst(&eval_ctx,input, &.{});
    try std.testing.expect(result == .none);
}

test "filterFirst - empty string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "" };
    const result = try filterFirst(&eval_ctx,input, &.{});
    try std.testing.expect(result == .none);
}

test "filterLast - empty array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{};
    const input = TemplateInput{ .array = &items };
    const result = try filterLast(&eval_ctx,input, &.{});
    try std.testing.expect(result == .none);
}

test "filterLast - empty string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "" };
    const result = try filterLast(&eval_ctx,input, &.{});
    try std.testing.expect(result == .none);
}

test "filterJoin - empty array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{};
    const input = TemplateInput{ .array = &items };
    const result = try filterJoin(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("", result.string);
}

test "filterJoin - single element" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{.{ .string = "only" }};
    const input = TemplateInput{ .array = &items };
    const sep_expr = Expr{ .string = ", " };
    const args = [_]*const Expr{&sep_expr};
    const result = try filterJoin(&eval_ctx,input, &args);
    try std.testing.expectEqualStrings("only", result.string);
}

test "filterJoin - no separator" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .string = "a" },
        .{ .string = "b" },
        .{ .string = "c" },
    };
    const input = TemplateInput{ .array = &items };
    const result = try filterJoin(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("abc", result.string);
}

test "filterReverse - single element array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{.{ .integer = 1 }};
    const input = TemplateInput{ .array = &items };
    const result = try filterReverse(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(usize, 1), result.array.len);
    try std.testing.expectEqual(@as(i64, 1), result.array[0].integer);
}

test "filterSort - strings" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .string = "zebra" },
        .{ .string = "apple" },
        .{ .string = "banana" },
    };
    const input = TemplateInput{ .array = &items };
    const result = try filterSort(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(usize, 3), result.array.len);
    try std.testing.expectEqualStrings("apple", result.array[0].string);
    try std.testing.expectEqualStrings("banana", result.array[1].string);
    try std.testing.expectEqualStrings("zebra", result.array[2].string);
}

test "filterUnique - all unique" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 2 },
        .{ .integer = 3 },
    };
    const input = TemplateInput{ .array = &items };
    const result = try filterUnique(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(usize, 3), result.array.len);
}

test "filterSlice - basic division" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 2 },
        .{ .integer = 3 },
        .{ .integer = 4 },
        .{ .integer = 5 },
        .{ .integer = 6 },
    };
    const input = TemplateInput{ .array = &items };
    const n_expr = Expr{ .integer = 3 };
    const args = [_]*const Expr{&n_expr};
    const result = try filterSlice(&eval_ctx,input, &args);

    try std.testing.expectEqual(@as(usize, 3), result.array.len);
    // First slice: [1, 2]
    try std.testing.expectEqual(@as(usize, 2), result.array[0].array.len);
    try std.testing.expectEqual(@as(i64, 1), result.array[0].array[0].integer);
    try std.testing.expectEqual(@as(i64, 2), result.array[0].array[1].integer);
    // Second slice: [3, 4]
    try std.testing.expectEqual(@as(usize, 2), result.array[1].array.len);
    try std.testing.expectEqual(@as(i64, 3), result.array[1].array[0].integer);
    try std.testing.expectEqual(@as(i64, 4), result.array[1].array[1].integer);
    // Third slice: [5, 6]
    try std.testing.expectEqual(@as(usize, 2), result.array[2].array.len);
    try std.testing.expectEqual(@as(i64, 5), result.array[2].array[0].integer);
    try std.testing.expectEqual(@as(i64, 6), result.array[2].array[1].integer);
}

test "filterSlice - uneven division" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 2 },
        .{ .integer = 3 },
        .{ .integer = 4 },
        .{ .integer = 5 },
    };
    const input = TemplateInput{ .array = &items };
    const n_expr = Expr{ .integer = 3 };
    const args = [_]*const Expr{&n_expr};
    const result = try filterSlice(&eval_ctx,input, &args);

    try std.testing.expectEqual(@as(usize, 3), result.array.len);
    // First slice gets extra item: [1, 2]
    try std.testing.expectEqual(@as(usize, 2), result.array[0].array.len);
    // Second slice gets extra item: [3, 4]
    try std.testing.expectEqual(@as(usize, 2), result.array[1].array.len);
    // Third slice: [5]
    try std.testing.expectEqual(@as(usize, 1), result.array[2].array.len);
    try std.testing.expectEqual(@as(i64, 5), result.array[2].array[0].integer);
}

test "filterSlice - empty array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{};
    const input = TemplateInput{ .array = &items };
    const n_expr = Expr{ .integer = 3 };
    const args = [_]*const Expr{&n_expr};
    const result = try filterSlice(&eval_ctx,input, &args);

    try std.testing.expectEqual(@as(usize, 0), result.array.len);
}

test "filterSlice - single slice" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 2 },
        .{ .integer = 3 },
    };
    const input = TemplateInput{ .array = &items };
    const n_expr = Expr{ .integer = 1 };
    const args = [_]*const Expr{&n_expr};
    const result = try filterSlice(&eval_ctx,input, &args);

    try std.testing.expectEqual(@as(usize, 1), result.array.len);
    try std.testing.expectEqual(@as(usize, 3), result.array[0].array.len);
    try std.testing.expectEqual(@as(i64, 1), result.array[0].array[0].integer);
    try std.testing.expectEqual(@as(i64, 2), result.array[0].array[1].integer);
    try std.testing.expectEqual(@as(i64, 3), result.array[0].array[2].integer);
}

test "filterBatch - basic batching" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 2 },
        .{ .integer = 3 },
        .{ .integer = 4 },
        .{ .integer = 5 },
    };
    const input = TemplateInput{ .array = &items };

    const size_expr = Expr{ .integer = 2 };
    const args = [_]*const Expr{&size_expr};
    const result = try filterBatch(&eval_ctx,input, &args);

    try std.testing.expectEqual(@as(usize, 3), result.array.len);
    try std.testing.expectEqual(@as(usize, 2), result.array[0].array.len);
    try std.testing.expectEqual(@as(usize, 2), result.array[1].array.len);
    try std.testing.expectEqual(@as(usize, 1), result.array[2].array.len);
    try std.testing.expectEqual(@as(i64, 1), result.array[0].array[0].integer);
    try std.testing.expectEqual(@as(i64, 2), result.array[0].array[1].integer);
    try std.testing.expectEqual(@as(i64, 5), result.array[2].array[0].integer);
}

test "filterBatch - exact division" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 2 },
        .{ .integer = 3 },
        .{ .integer = 4 },
    };
    const input = TemplateInput{ .array = &items };

    const size_expr = Expr{ .integer = 2 };
    const args = [_]*const Expr{&size_expr};
    const result = try filterBatch(&eval_ctx,input, &args);

    try std.testing.expectEqual(@as(usize, 2), result.array.len);
    try std.testing.expectEqual(@as(usize, 2), result.array[0].array.len);
    try std.testing.expectEqual(@as(usize, 2), result.array[1].array.len);
}

test "filterBatch - single element batches" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 2 },
        .{ .integer = 3 },
    };
    const input = TemplateInput{ .array = &items };

    const size_expr = Expr{ .integer = 1 };
    const args = [_]*const Expr{&size_expr};
    const result = try filterBatch(&eval_ctx,input, &args);

    try std.testing.expectEqual(@as(usize, 3), result.array.len);
    try std.testing.expectEqual(@as(usize, 1), result.array[0].array.len);
    try std.testing.expectEqual(@as(usize, 1), result.array[1].array.len);
    try std.testing.expectEqual(@as(usize, 1), result.array[2].array.len);
}

test "filterBatch - empty array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{};
    const input = TemplateInput{ .array = &items };

    const size_expr = Expr{ .integer = 2 };
    const args = [_]*const Expr{&size_expr};
    const result = try filterBatch(&eval_ctx,input, &args);

    try std.testing.expectEqual(@as(usize, 0), result.array.len);
}

test "filterBatch - large batch size" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 2 },
    };
    const input = TemplateInput{ .array = &items };

    const size_expr = Expr{ .integer = 10 };
    const args = [_]*const Expr{&size_expr};
    const result = try filterBatch(&eval_ctx,input, &args);

    try std.testing.expectEqual(@as(usize, 1), result.array.len);
    try std.testing.expectEqual(@as(usize, 2), result.array[0].array.len);
}

test "filterDictsort - basic sorting" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    var map = std.StringHashMapUnmanaged(TemplateInput){};
    try map.put(ctx.arena.allocator(), "zebra", .{ .integer = 3 });
    try map.put(ctx.arena.allocator(), "apple", .{ .integer = 1 });
    try map.put(ctx.arena.allocator(), "banana", .{ .integer = 2 });
    const input = TemplateInput{ .map = map };

    const result = try filterDictsort(&eval_ctx,input, &.{});

    try std.testing.expectEqual(@as(usize, 3), result.array.len);
    // Should be sorted alphabetically by key
    try std.testing.expectEqualStrings("apple", result.array[0].array[0].string);
    try std.testing.expectEqual(@as(i64, 1), result.array[0].array[1].integer);
    try std.testing.expectEqualStrings("banana", result.array[1].array[0].string);
    try std.testing.expectEqual(@as(i64, 2), result.array[1].array[1].integer);
    try std.testing.expectEqualStrings("zebra", result.array[2].array[0].string);
    try std.testing.expectEqual(@as(i64, 3), result.array[2].array[1].integer);
}

test "filterDictsort - single entry" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    var map = std.StringHashMapUnmanaged(TemplateInput){};
    try map.put(ctx.arena.allocator(), "key", .{ .string = "value" });
    const input = TemplateInput{ .map = map };

    const result = try filterDictsort(&eval_ctx,input, &.{});

    try std.testing.expectEqual(@as(usize, 1), result.array.len);
    try std.testing.expectEqualStrings("key", result.array[0].array[0].string);
    try std.testing.expectEqualStrings("value", result.array[0].array[1].string);
}

test "filterDictsort - empty map" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const map = std.StringHashMapUnmanaged(TemplateInput){};
    const input = TemplateInput{ .map = map };

    const result = try filterDictsort(&eval_ctx,input, &.{});

    try std.testing.expectEqual(@as(usize, 0), result.array.len);
}

test "filterDictsort - numeric string keys" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    var map = std.StringHashMapUnmanaged(TemplateInput){};
    try map.put(ctx.arena.allocator(), "3", .{ .string = "third" });
    try map.put(ctx.arena.allocator(), "1", .{ .string = "first" });
    try map.put(ctx.arena.allocator(), "2", .{ .string = "second" });
    const input = TemplateInput{ .map = map };

    const result = try filterDictsort(&eval_ctx,input, &.{});

    try std.testing.expectEqual(@as(usize, 3), result.array.len);
    // Lexicographic sort, not numeric
    try std.testing.expectEqualStrings("1", result.array[0].array[0].string);
    try std.testing.expectEqualStrings("2", result.array[1].array[0].string);
    try std.testing.expectEqualStrings("3", result.array[2].array[0].string);
}

test "filterDictsort - mixed value types" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    var map = std.StringHashMapUnmanaged(TemplateInput){};
    try map.put(ctx.arena.allocator(), "int", .{ .integer = 42 });
    try map.put(ctx.arena.allocator(), "str", .{ .string = "hello" });
    try map.put(ctx.arena.allocator(), "bool", .{ .boolean = true });
    const input = TemplateInput{ .map = map };

    const result = try filterDictsort(&eval_ctx,input, &.{});

    try std.testing.expectEqual(@as(usize, 3), result.array.len);
    // Sorted by key: bool, int, str
    try std.testing.expectEqualStrings("bool", result.array[0].array[0].string);
    try std.testing.expectEqualStrings("int", result.array[1].array[0].string);
    try std.testing.expectEqualStrings("str", result.array[2].array[0].string);
}
