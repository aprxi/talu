//! Selection Filters
//!
//! Filters for selecting, filtering, and mapping over arrays based on conditions.

const std = @import("std");
const types = @import("types.zig");
const predicates = @import("../predicates.zig");

// Import sibling filter modules for filterMap
const string = @import("string.zig");
const array = @import("array.zig");
const numeric = @import("numeric.zig");
const format = @import("format.zig");
const misc = @import("misc.zig");

const TemplateInput = types.TemplateInput;
const EvalError = types.EvalError;
const Evaluator = types.Evaluator;
const Expr = types.Expr;
const FilterFn = types.FilterFn;
const applyTest = predicates.applyTest;

// ============================================================================
// Item Selection
// ============================================================================

pub fn filterSelect(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    // Select items matching test (default: truthy)
    const arr = switch (value) {
        .array => |a| a,
        else => return EvalError.TypeError,
    };

    // Get optional test name from first argument
    var test_name: ?[]const u8 = null;
    if (args.len > 0) {
        const name_val = try e.evalExpr(args[0]);
        if (name_val == .string) {
            test_name = name_val.string;
        }
    }

    // Test arguments (everything after the test name)
    const test_args = if (args.len > 1) args[1..] else &[_]*const Expr{};

    const arena = e.ctx.arena.allocator();
    var result = std.ArrayListUnmanaged(TemplateInput){};

    for (arr) |item| {
        const matches = if (test_name) |name|
            try applyTest(e, name, item, test_args)
        else
            item.isTruthy();

        if (matches) {
            result.append(arena, item) catch return EvalError.OutOfMemory;
        }
    }

    return .{ .array = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

pub fn filterReject(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    // Reject items matching test (default: truthy)
    const arr = switch (value) {
        .array => |a| a,
        else => return EvalError.TypeError,
    };

    // Get optional test name from first argument
    var test_name: ?[]const u8 = null;
    if (args.len > 0) {
        const name_val = try e.evalExpr(args[0]);
        if (name_val == .string) {
            test_name = name_val.string;
        }
    }

    // Test arguments (everything after the test name)
    const test_args = if (args.len > 1) args[1..] else &[_]*const Expr{};

    const arena = e.ctx.arena.allocator();
    var result = std.ArrayListUnmanaged(TemplateInput){};

    for (arr) |item| {
        const matches = if (test_name) |name|
            try applyTest(e, name, item, test_args)
        else
            item.isTruthy();

        if (!matches) {
            result.append(arena, item) catch return EvalError.OutOfMemory;
        }
    }

    return .{ .array = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

// ============================================================================
// Attribute Selection
// ============================================================================

pub fn filterSelectattr(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    const arr = switch (value) {
        .array => |a| a,
        else => return EvalError.TypeError,
    };

    if (args.len < 1) return EvalError.TypeError;

    const arena = e.ctx.arena.allocator();
    var result = std.ArrayListUnmanaged(TemplateInput){};

    // Get attribute name
    const attr_val = try e.evalExpr(args[0]);
    const attr = switch (attr_val) {
        .string => |s| s,
        else => return EvalError.TypeError,
    };

    for (arr) |item| {
        switch (item) {
            .map => |m| {
                if (m.get(attr)) |val| {
                    // If we have a test (args[1]), apply it
                    if (args.len > 1) {
                        const test_name_val = try e.evalExpr(args[1]);
                        const test_name = switch (test_name_val) {
                            .string => |s| s,
                            else => return EvalError.TypeError,
                        };
                        // Simple tests - just check equality if args[2] provided
                        if (args.len > 2) {
                            const expected = try e.evalExpr(args[2]);
                            if (val.eql(expected)) {
                                result.append(arena, item) catch return EvalError.OutOfMemory;
                            }
                        } else if (std.mem.eql(u8, test_name, "defined")) {
                            result.append(arena, item) catch return EvalError.OutOfMemory;
                        } else if (std.mem.eql(u8, test_name, "true") and val.isTruthy()) {
                            result.append(arena, item) catch return EvalError.OutOfMemory;
                        }
                    } else if (val.isTruthy()) {
                        result.append(arena, item) catch return EvalError.OutOfMemory;
                    }
                }
            },
            else => {},
        }
    }

    return .{ .array = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

pub fn filterRejectattr(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    const arr = switch (value) {
        .array => |a| a,
        else => return EvalError.TypeError,
    };

    if (args.len < 1) return EvalError.TypeError;

    const arena = e.ctx.arena.allocator();
    var result = std.ArrayListUnmanaged(TemplateInput){};

    const attr_val = try e.evalExpr(args[0]);
    const attr = switch (attr_val) {
        .string => |s| s,
        else => return EvalError.TypeError,
    };

    for (arr) |item| {
        switch (item) {
            .map => |m| {
                if (m.get(attr)) |val| {
                    if (args.len > 1) {
                        const test_name_val = try e.evalExpr(args[1]);
                        const test_name = switch (test_name_val) {
                            .string => |s| s,
                            else => return EvalError.TypeError,
                        };
                        if (args.len > 2) {
                            const expected = try e.evalExpr(args[2]);
                            if (!val.eql(expected)) {
                                result.append(arena, item) catch return EvalError.OutOfMemory;
                            }
                        } else if (std.mem.eql(u8, test_name, "defined")) {
                            // defined means it exists, so reject it
                        } else if (!val.isTruthy()) {
                            result.append(arena, item) catch return EvalError.OutOfMemory;
                        }
                    } else if (!val.isTruthy()) {
                        result.append(arena, item) catch return EvalError.OutOfMemory;
                    }
                } else {
                    // Attribute doesn't exist - include it (reject only those that match)
                    result.append(arena, item) catch return EvalError.OutOfMemory;
                }
            },
            else => result.append(arena, item) catch return EvalError.OutOfMemory,
        }
    }

    return .{ .array = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

pub fn filterAttr(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    if (args.len < 1) return EvalError.TypeError;

    const attr_val = try e.evalExpr(args[0]);
    const attr = switch (attr_val) {
        .string => |s| s,
        else => return EvalError.TypeError,
    };

    switch (value) {
        .map => |m| {
            if (m.get(attr)) |v| return v;
            if (e.ctx.strict) return EvalError.KeyError;
            return .none;
        },
        .namespace => |ns| {
            if (ns.get(attr)) |v| return v;
            if (e.ctx.strict) return EvalError.KeyError;
            return .none;
        },
        .none => {
            if (e.ctx.strict) return EvalError.UndefinedVariable;
            return .none;
        },
        else => {
            if (e.ctx.strict) return EvalError.TypeError;
            return .none;
        },
    }
}

// ============================================================================
// Mapping
// ============================================================================

pub fn filterMap(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    const arr = switch (value) {
        .array => |a| a,
        else => return EvalError.TypeError,
    };

    if (args.len < 1) return EvalError.TypeError;

    const arena = e.ctx.arena.allocator();
    var result = std.ArrayListUnmanaged(TemplateInput){};

    // Get first argument as string
    const first_arg = args[0];
    const arg_val = try e.evalExpr(first_arg);
    const arg_str = switch (arg_val) {
        .string => |s| s,
        else => return EvalError.TypeError,
    };

    // Check if arg is a known filter name
    // If so, apply as filter: map('upper'), map('trim'), etc.
    // Otherwise, treat as attribute name: map(attribute='name') or map('name')
    const filter_names = std.StaticStringMap(void).initComptime(.{
        .{ "upper", {} },
        .{ "lower", {} },
        .{ "trim", {} },
        .{ "strip", {} },
        .{ "string", {} },
        .{ "int", {} },
        .{ "float", {} },
        .{ "abs", {} },
        .{ "capitalize", {} },
        .{ "title", {} },
        .{ "length", {} },
        .{ "first", {} },
        .{ "last", {} },
        .{ "reverse", {} },
        .{ "sort", {} },
        .{ "unique", {} },
        .{ "list", {} },
        .{ "tojson", {} },
        .{ "escape", {} },
        .{ "safe", {} },
        .{ "striptags", {} },
        .{ "urlencode", {} },
    });

    if (filter_names.get(arg_str) != null) {
        // Apply as filter to each item
        const filter_args = if (args.len > 1) args[1..] else &[_]*const Expr{};

        // Local filter dispatch for map filter (avoids circular dependency with root.zig)
        const map_filters = std.StaticStringMap(FilterFn).initComptime(.{
            .{ "upper", &string.filterUpper },
            .{ "lower", &string.filterLower },
            .{ "trim", &string.filterTrim },
            .{ "strip", &string.filterTrim },
            .{ "string", &misc.filterString },
            .{ "int", &numeric.filterInt },
            .{ "float", &numeric.filterFloat },
            .{ "abs", &numeric.filterAbs },
            .{ "capitalize", &string.filterCapitalize },
            .{ "title", &string.filterTitle },
            .{ "length", &misc.filterLength },
            .{ "first", &array.filterFirst },
            .{ "last", &array.filterLast },
            .{ "reverse", &array.filterReverse },
            .{ "sort", &array.filterSort },
            .{ "unique", &array.filterUnique },
            .{ "list", &misc.filterList },
            .{ "tojson", &format.filterTojson },
            .{ "escape", &string.filterEscape },
            .{ "safe", &string.filterSafe },
            .{ "striptags", &string.filterStriptags },
            .{ "urlencode", &string.filterUrlencode },
        });

        const filter_fn = map_filters.get(arg_str) orelse return EvalError.UnsupportedFilter;
        for (arr) |item| {
            const filtered = try filter_fn(e, item, filter_args);
            result.append(arena, filtered) catch return EvalError.OutOfMemory;
        }
    } else {
        // Treat as attribute name
        for (arr) |item| {
            switch (item) {
                .map => |map_value| {
                    if (map_value.get(arg_str)) |entry_value| {
                        result.append(arena, entry_value) catch return EvalError.OutOfMemory;
                    } else {
                        if (e.ctx.strict) return EvalError.KeyError;
                        result.append(arena, .none) catch return EvalError.OutOfMemory;
                    }
                },
                else => {
                    if (e.ctx.strict) return EvalError.TypeError;
                    result.append(arena, .none) catch return EvalError.OutOfMemory;
                },
            }
        }
    }

    return .{ .array = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

// ============================================================================
// Grouping
// ============================================================================

pub fn filterGroupby(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    const arr = switch (value) {
        .array => |a| a,
        else => return EvalError.TypeError,
    };

    if (args.len < 1) return EvalError.TypeError;

    const arena = e.ctx.arena.allocator();

    // Get attribute name
    const attr_val = try e.evalExpr(args[0]);
    const attr = switch (attr_val) {
        .string => |s| s,
        else => return EvalError.TypeError,
    };

    // Group items by attribute value
    var groups = std.StringHashMapUnmanaged(std.ArrayListUnmanaged(TemplateInput)){};

    for (arr) |item| {
        const key = switch (item) {
            .map => |m| blk: {
                if (m.get(attr)) |v| {
                    break :blk v.asString(arena) catch continue;
                }
                continue;
            },
            else => continue,
        };

        const gop = groups.getOrPut(arena, key) catch return EvalError.OutOfMemory;
        if (!gop.found_existing) {
            gop.value_ptr.* = std.ArrayListUnmanaged(TemplateInput){};
        }
        gop.value_ptr.append(arena, item) catch return EvalError.OutOfMemory;
    }

    // Convert to array of (key, items) tuples
    var result = std.ArrayListUnmanaged(TemplateInput){};
    var it = groups.iterator();
    while (it.next()) |entry| {
        var tuple = std.ArrayListUnmanaged(TemplateInput){};
        tuple.append(arena, .{ .string = entry.key_ptr.* }) catch return EvalError.OutOfMemory;
        tuple.append(arena, .{ .array = entry.value_ptr.toOwnedSlice(arena) catch return EvalError.OutOfMemory }) catch return EvalError.OutOfMemory;
        result.append(arena, .{ .array = tuple.toOwnedSlice(arena) catch return EvalError.OutOfMemory }) catch return EvalError.OutOfMemory;
    }

    return .{ .array = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

// ============================================================================
// Unit Tests
// ============================================================================

const TemplateParser = @import("../eval.zig").TemplateParser;

test "filterAttr - basic map attribute" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    var map = std.StringHashMapUnmanaged(TemplateInput){};
    try map.put(ctx.arena.allocator(), "name", .{ .string = "Alice" });
    try map.put(ctx.arena.allocator(), "age", .{ .integer = 30 });
    const input = TemplateInput{ .map = map };

    const attr_expr = Expr{ .string = "name" };
    const args = [_]*const Expr{&attr_expr};
    const result = try filterAttr(&eval_ctx,input, &args);
    try std.testing.expectEqualStrings("Alice", result.string);
}

test "filterAttr - integer attribute" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    var map = std.StringHashMapUnmanaged(TemplateInput){};
    try map.put(ctx.arena.allocator(), "age", .{ .integer = 30 });
    const input = TemplateInput{ .map = map };

    const attr_expr = Expr{ .string = "age" };
    const args = [_]*const Expr{&attr_expr};
    const result = try filterAttr(&eval_ctx,input, &args);
    try std.testing.expectEqual(@as(i64, 30), result.integer);
}

test "filterAttr - missing attribute" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    var map = std.StringHashMapUnmanaged(TemplateInput){};
    try map.put(ctx.arena.allocator(), "name", .{ .string = "Alice" });
    const input = TemplateInput{ .map = map };

    const attr_expr = Expr{ .string = "nonexistent" };
    const args = [_]*const Expr{&attr_expr};
    const result = try filterAttr(&eval_ctx,input, &args);
    try std.testing.expect(result == .none);
}

test "filterAttr - namespace attribute" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    var namespace = try ctx.arena.allocator().create(std.StringHashMapUnmanaged(TemplateInput));
    namespace.* = std.StringHashMapUnmanaged(TemplateInput){};
    try namespace.put(ctx.arena.allocator(), "value", .{ .integer = 42 });
    const input = TemplateInput{ .namespace = namespace };

    const attr_expr = Expr{ .string = "value" };
    const args = [_]*const Expr{&attr_expr};
    const result = try filterAttr(&eval_ctx,input, &args);
    try std.testing.expectEqual(@as(i64, 42), result.integer);
}

test "filterAttr - empty map" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const map = std.StringHashMapUnmanaged(TemplateInput){};
    const input = TemplateInput{ .map = map };

    const attr_expr = Expr{ .string = "anything" };
    const args = [_]*const Expr{&attr_expr};
    const result = try filterAttr(&eval_ctx,input, &args);
    try std.testing.expect(result == .none);
}

test "filterGroupby - basic grouping" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    // Create items with a "category" attribute
    var map1 = std.StringHashMapUnmanaged(TemplateInput){};
    try map1.put(ctx.arena.allocator(), "name", .{ .string = "apple" });
    try map1.put(ctx.arena.allocator(), "category", .{ .string = "fruit" });

    var map2 = std.StringHashMapUnmanaged(TemplateInput){};
    try map2.put(ctx.arena.allocator(), "name", .{ .string = "carrot" });
    try map2.put(ctx.arena.allocator(), "category", .{ .string = "vegetable" });

    var map3 = std.StringHashMapUnmanaged(TemplateInput){};
    try map3.put(ctx.arena.allocator(), "name", .{ .string = "banana" });
    try map3.put(ctx.arena.allocator(), "category", .{ .string = "fruit" });

    const items = [_]TemplateInput{
        .{ .map = map1 },
        .{ .map = map2 },
        .{ .map = map3 },
    };
    const input = TemplateInput{ .array = &items };

    const attr_expr = Expr{ .string = "category" };
    const args = [_]*const Expr{&attr_expr};
    const result = try filterGroupby(&eval_ctx,input, &args);

    // Should have 2 groups: fruit and vegetable
    try std.testing.expectEqual(@as(usize, 2), result.array.len);

    // Each group is a tuple [key, items]
    // Find the fruit group
    var found_fruit = false;
    var found_vegetable = false;

for (result.array) |group| {
        const key = group.array[0].string;
        const group_items = group.array[1].array;

if (std.mem.eql(u8, key, "fruit")) {
            found_fruit = true;
            try std.testing.expectEqual(@as(usize, 2), group_items.len);
        } else if (std.mem.eql(u8, key, "vegetable")) {
            found_vegetable = true;
            try std.testing.expectEqual(@as(usize, 1), group_items.len);
        }
    }

    try std.testing.expect(found_fruit);
    try std.testing.expect(found_vegetable);
}

test "filterGroupby - single group" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    var map1 = std.StringHashMapUnmanaged(TemplateInput){};
    try map1.put(ctx.arena.allocator(), "type", .{ .string = "same" });

    var map2 = std.StringHashMapUnmanaged(TemplateInput){};
    try map2.put(ctx.arena.allocator(), "type", .{ .string = "same" });

    const items = [_]TemplateInput{
        .{ .map = map1 },
        .{ .map = map2 },
    };
    const input = TemplateInput{ .array = &items };

    const attr_expr = Expr{ .string = "type" };
    const args = [_]*const Expr{&attr_expr};
    const result = try filterGroupby(&eval_ctx,input, &args);

    try std.testing.expectEqual(@as(usize, 1), result.array.len);
    try std.testing.expectEqualStrings("same", result.array[0].array[0].string);
    try std.testing.expectEqual(@as(usize, 2), result.array[0].array[1].array.len);
}

test "filterGroupby - empty array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{};
    const input = TemplateInput{ .array = &items };

    const attr_expr = Expr{ .string = "attribute" };
    const args = [_]*const Expr{&attr_expr};
    const result = try filterGroupby(&eval_ctx,input, &args);

    try std.testing.expectEqual(@as(usize, 0), result.array.len);
}

test "filterGroupby - missing attribute" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    var map1 = std.StringHashMapUnmanaged(TemplateInput){};
    try map1.put(ctx.arena.allocator(), "name", .{ .string = "apple" });
    // No "category" attribute

    const items = [_]TemplateInput{.{ .map = map1 }};
    const input = TemplateInput{ .array = &items };

    const attr_expr = Expr{ .string = "category" };
    const args = [_]*const Expr{&attr_expr};
    const result = try filterGroupby(&eval_ctx,input, &args);

    // Items without the attribute are skipped
    try std.testing.expectEqual(@as(usize, 0), result.array.len);
}

test "filterGroupby - numeric values" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    var map1 = std.StringHashMapUnmanaged(TemplateInput){};
    try map1.put(ctx.arena.allocator(), "name", .{ .string = "item1" });
    try map1.put(ctx.arena.allocator(), "priority", .{ .integer = 1 });

    var map2 = std.StringHashMapUnmanaged(TemplateInput){};
    try map2.put(ctx.arena.allocator(), "name", .{ .string = "item2" });
    try map2.put(ctx.arena.allocator(), "priority", .{ .integer = 2 });

    var map3 = std.StringHashMapUnmanaged(TemplateInput){};
    try map3.put(ctx.arena.allocator(), "name", .{ .string = "item3" });
    try map3.put(ctx.arena.allocator(), "priority", .{ .integer = 1 });

    const items = [_]TemplateInput{
        .{ .map = map1 },
        .{ .map = map2 },
        .{ .map = map3 },
    };
    const input = TemplateInput{ .array = &items };

    const attr_expr = Expr{ .string = "priority" };
    const args = [_]*const Expr{&attr_expr};
    const result = try filterGroupby(&eval_ctx,input, &args);

    // Should have 2 groups: "1" and "2" (converted to strings)
    try std.testing.expectEqual(@as(usize, 2), result.array.len);
}

test "filterSelect - default truthy" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 0 },
        .{ .boolean = true },
        .{ .boolean = false },
        .{ .string = "hello" },
        .{ .string = "" },
    };
    const input = TemplateInput{ .array = &items };
    const result = try filterSelect(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(usize, 3), result.array.len);
    try std.testing.expectEqual(@as(i64, 1), result.array[0].integer);
    try std.testing.expectEqual(true, result.array[1].boolean);
    try std.testing.expectEqualStrings("hello", result.array[2].string);
}

test "filterSelect - empty array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{};
    const input = TemplateInput{ .array = &items };
    const result = try filterSelect(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(usize, 0), result.array.len);
}

test "filterSelect - no matches" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 0 },
        .{ .boolean = false },
        .{ .string = "" },
        .none,
    };
    const input = TemplateInput{ .array = &items };
    const result = try filterSelect(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(usize, 0), result.array.len);
}

test "filterSelect - named test" {
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
    const test_expr = Expr{ .string = "odd" };
    const args = [_]*const Expr{&test_expr};
    const result = try filterSelect(&eval_ctx,input, &args);
    try std.testing.expectEqual(@as(usize, 2), result.array.len);
    try std.testing.expectEqual(@as(i64, 1), result.array[0].integer);
    try std.testing.expectEqual(@as(i64, 3), result.array[1].integer);
}

test "filterReject - default truthy" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 0 },
        .{ .boolean = true },
        .{ .boolean = false },
        .{ .string = "hello" },
        .{ .string = "" },
    };
    const input = TemplateInput{ .array = &items };
    const result = try filterReject(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(usize, 3), result.array.len);
    try std.testing.expectEqual(@as(i64, 0), result.array[0].integer);
    try std.testing.expectEqual(false, result.array[1].boolean);
    try std.testing.expectEqualStrings("", result.array[2].string);
}

test "filterReject - empty array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{};
    const input = TemplateInput{ .array = &items };
    const result = try filterReject(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(usize, 0), result.array.len);
}

test "filterReject - all match" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .boolean = true },
        .{ .string = "hello" },
    };
    const input = TemplateInput{ .array = &items };
    const result = try filterReject(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(usize, 0), result.array.len);
}

test "filterReject - named test" {
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
    const test_expr = Expr{ .string = "odd" };
    const args = [_]*const Expr{&test_expr};
    const result = try filterReject(&eval_ctx,input, &args);
    try std.testing.expectEqual(@as(usize, 2), result.array.len);
    try std.testing.expectEqual(@as(i64, 2), result.array[0].integer);
    try std.testing.expectEqual(@as(i64, 4), result.array[1].integer);
}

test "filterSelectattr - default truthy" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arena = eval_ctx.ctx.arena.allocator();

    var map1 = std.StringHashMapUnmanaged(TemplateInput){};
    try map1.put(arena, "active", .{ .boolean = true });
    var map2 = std.StringHashMapUnmanaged(TemplateInput){};
    try map2.put(arena, "active", .{ .boolean = false });
    var map3 = std.StringHashMapUnmanaged(TemplateInput){};
    try map3.put(arena, "active", .{ .boolean = true });

    const items = [_]TemplateInput{
        .{ .map = map1 },
        .{ .map = map2 },
        .{ .map = map3 },
    };
    const input = TemplateInput{ .array = &items };
    const attr_expr = Expr{ .string = "active" };
    const args = [_]*const Expr{&attr_expr};
    const result = try filterSelectattr(&eval_ctx,input, &args);
    try std.testing.expectEqual(@as(usize, 2), result.array.len);
}

test "filterSelectattr - defined test" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arena = eval_ctx.ctx.arena.allocator();

    var map1 = std.StringHashMapUnmanaged(TemplateInput){};
    try map1.put(arena, "name", .{ .string = "alice" });
    var map2 = std.StringHashMapUnmanaged(TemplateInput){};
    try map2.put(arena, "age", .{ .integer = 30 });
    var map3 = std.StringHashMapUnmanaged(TemplateInput){};
    try map3.put(arena, "name", .{ .string = "bob" });

    const items = [_]TemplateInput{
        .{ .map = map1 },
        .{ .map = map2 },
        .{ .map = map3 },
    };
    const input = TemplateInput{ .array = &items };
    const attr_expr = Expr{ .string = "name" };
    const test_expr = Expr{ .string = "defined" };
    const args = [_]*const Expr{ &attr_expr, &test_expr };
    const result = try filterSelectattr(&eval_ctx,input, &args);
    try std.testing.expectEqual(@as(usize, 2), result.array.len);
}

test "filterSelectattr - equality test" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arena = eval_ctx.ctx.arena.allocator();

    var map1 = std.StringHashMapUnmanaged(TemplateInput){};
    try map1.put(arena, "role", .{ .string = "admin" });
    var map2 = std.StringHashMapUnmanaged(TemplateInput){};
    try map2.put(arena, "role", .{ .string = "user" });
    var map3 = std.StringHashMapUnmanaged(TemplateInput){};
    try map3.put(arena, "role", .{ .string = "admin" });

    const items = [_]TemplateInput{
        .{ .map = map1 },
        .{ .map = map2 },
        .{ .map = map3 },
    };
    const input = TemplateInput{ .array = &items };
    const attr_expr = Expr{ .string = "role" };
    const test_expr = Expr{ .string = "equalto" };
    const value_expr = Expr{ .string = "admin" };
    const args = [_]*const Expr{ &attr_expr, &test_expr, &value_expr };
    const result = try filterSelectattr(&eval_ctx,input, &args);
    try std.testing.expectEqual(@as(usize, 2), result.array.len);
}

test "filterSelectattr - empty array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{};
    const input = TemplateInput{ .array = &items };
    const attr_expr = Expr{ .string = "name" };
    const args = [_]*const Expr{&attr_expr};
    const result = try filterSelectattr(&eval_ctx,input, &args);
    try std.testing.expectEqual(@as(usize, 0), result.array.len);
}

test "filterSelectattr - no matches" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arena = eval_ctx.ctx.arena.allocator();

    var map1 = std.StringHashMapUnmanaged(TemplateInput){};
    try map1.put(arena, "active", .{ .boolean = false });
    var map2 = std.StringHashMapUnmanaged(TemplateInput){};
    try map2.put(arena, "active", .{ .integer = 0 });

    const items = [_]TemplateInput{
        .{ .map = map1 },
        .{ .map = map2 },
    };
    const input = TemplateInput{ .array = &items };
    const attr_expr = Expr{ .string = "active" };
    const args = [_]*const Expr{&attr_expr};
    const result = try filterSelectattr(&eval_ctx,input, &args);
    try std.testing.expectEqual(@as(usize, 0), result.array.len);
}

test "filterRejectattr - default truthy" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arena = eval_ctx.ctx.arena.allocator();

    var map1 = std.StringHashMapUnmanaged(TemplateInput){};
    try map1.put(arena, "active", .{ .boolean = true });
    var map2 = std.StringHashMapUnmanaged(TemplateInput){};
    try map2.put(arena, "active", .{ .boolean = false });
    var map3 = std.StringHashMapUnmanaged(TemplateInput){};
    try map3.put(arena, "active", .{ .boolean = true });

    const items = [_]TemplateInput{
        .{ .map = map1 },
        .{ .map = map2 },
        .{ .map = map3 },
    };
    const input = TemplateInput{ .array = &items };
    const attr_expr = Expr{ .string = "active" };
    const args = [_]*const Expr{&attr_expr};
    const result = try filterRejectattr(&eval_ctx,input, &args);
    try std.testing.expectEqual(@as(usize, 1), result.array.len);
}

test "filterRejectattr - defined test" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arena = eval_ctx.ctx.arena.allocator();

    var map1 = std.StringHashMapUnmanaged(TemplateInput){};
    try map1.put(arena, "name", .{ .string = "alice" });
    var map2 = std.StringHashMapUnmanaged(TemplateInput){};
    try map2.put(arena, "age", .{ .integer = 30 });
    var map3 = std.StringHashMapUnmanaged(TemplateInput){};
    try map3.put(arena, "name", .{ .string = "bob" });

    const items = [_]TemplateInput{
        .{ .map = map1 },
        .{ .map = map2 },
        .{ .map = map3 },
    };
    const input = TemplateInput{ .array = &items };
    const attr_expr = Expr{ .string = "name" };
    const test_expr = Expr{ .string = "defined" };
    const args = [_]*const Expr{ &attr_expr, &test_expr };
    const result = try filterRejectattr(&eval_ctx,input, &args);
    // Rejects items where "name" is defined, so only map2 should remain
    try std.testing.expectEqual(@as(usize, 1), result.array.len);
}

test "filterRejectattr - equality test" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arena = eval_ctx.ctx.arena.allocator();

    var map1 = std.StringHashMapUnmanaged(TemplateInput){};
    try map1.put(arena, "role", .{ .string = "admin" });
    var map2 = std.StringHashMapUnmanaged(TemplateInput){};
    try map2.put(arena, "role", .{ .string = "user" });
    var map3 = std.StringHashMapUnmanaged(TemplateInput){};
    try map3.put(arena, "role", .{ .string = "admin" });

    const items = [_]TemplateInput{
        .{ .map = map1 },
        .{ .map = map2 },
        .{ .map = map3 },
    };
    const input = TemplateInput{ .array = &items };
    const attr_expr = Expr{ .string = "role" };
    const test_expr = Expr{ .string = "equalto" };
    const value_expr = Expr{ .string = "admin" };
    const args = [_]*const Expr{ &attr_expr, &test_expr, &value_expr };
    const result = try filterRejectattr(&eval_ctx,input, &args);
    // Rejects items where role == "admin", so only map2 should remain
    try std.testing.expectEqual(@as(usize, 1), result.array.len);
}

test "filterRejectattr - empty array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{};
    const input = TemplateInput{ .array = &items };
    const attr_expr = Expr{ .string = "name" };
    const args = [_]*const Expr{&attr_expr};
    const result = try filterRejectattr(&eval_ctx,input, &args);
    try std.testing.expectEqual(@as(usize, 0), result.array.len);
}

test "filterRejectattr - attribute missing" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arena = eval_ctx.ctx.arena.allocator();

    var map1 = std.StringHashMapUnmanaged(TemplateInput){};
    try map1.put(arena, "name", .{ .string = "alice" });
    var map2 = std.StringHashMapUnmanaged(TemplateInput){};
    try map2.put(arena, "age", .{ .integer = 30 });

    const items = [_]TemplateInput{
        .{ .map = map1 },
        .{ .map = map2 },
    };
    const input = TemplateInput{ .array = &items };
    const attr_expr = Expr{ .string = "active" };
    const args = [_]*const Expr{&attr_expr};
    const result = try filterRejectattr(&eval_ctx,input, &args);
    // Both items don't have "active" attribute, so both should be kept
    try std.testing.expectEqual(@as(usize, 2), result.array.len);
}

test "filterRejectattr - non-map items" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .string = "hello" },
        .{ .boolean = true },
    };
    const input = TemplateInput{ .array = &items };
    const attr_expr = Expr{ .string = "name" };
    const args = [_]*const Expr{&attr_expr};
    const result = try filterRejectattr(&eval_ctx,input, &args);
    // Non-map items should be kept since they can't match
    try std.testing.expectEqual(@as(usize, 3), result.array.len);
}

test "filterMap - apply upper filter" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .string = "hello" },
        .{ .string = "world" },
    };
    const input = TemplateInput{ .array = &items };
    const filter_expr = Expr{ .string = "upper" };
    const args = [_]*const Expr{&filter_expr};
    const result = try filterMap(&eval_ctx, input, &args);
    try std.testing.expectEqual(@as(usize, 2), result.array.len);
    try std.testing.expectEqualStrings("HELLO", result.array[0].string);
    try std.testing.expectEqualStrings("WORLD", result.array[1].string);
}
