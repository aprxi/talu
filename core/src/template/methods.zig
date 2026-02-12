//! Jinja2 Built-in Methods
//!
//! Object.method() support for strings, arrays, maps, and loop contexts.

const std = @import("std");
const ast = @import("ast.zig");
const eval = @import("eval.zig");

const TemplateInput = eval.TemplateInput;
const EvalError = eval.EvalError;
const Evaluator = eval.Evaluator;
const CyclerState = eval.CyclerState;
const Expr = ast.Expr;

// ============================================================================
// Methods (object.method())
// ============================================================================

pub fn callMethod(e: *Evaluator, obj: TemplateInput, method: []const u8, args: []const *const Expr) EvalError!TemplateInput {
    switch (obj) {
        .string => |s| return stringMethod(e, s, method, args),
        .array => |a| return arrayMethod(e, a, method, args),
        .map => |m| return mapMethod(e, m, method, args),
        .namespace => |ns| return mapMethod(e, ns.*, method, args),
        .cycler => |c| return cyclerMethod(c, method),
        .loop_ctx => |lp| return loopMethod(e, lp, method, args),
        else => {},
    }
    return EvalError.UnsupportedMethod;
}

const TrimMode = enum { both, left, right };

fn evalStringArg(e: *Evaluator, args: []const *const Expr, index: usize) EvalError![]const u8 {
    if (args.len <= index) return EvalError.TypeError;
    const value = try e.evalExpr(args[index]);
    return switch (value) {
        .string => |s| s,
        else => EvalError.TypeError,
    };
}

fn evalOptionalStringArg(e: *Evaluator, args: []const *const Expr, index: usize, default: []const u8) EvalError![]const u8 {
    if (args.len <= index) return default;
    return evalStringArg(e, args, index);
}

fn trimString(e: *Evaluator, s: []const u8, args: []const *const Expr, mode: TrimMode) EvalError!TemplateInput {
    const chars = try evalOptionalStringArg(e, args, 0, " \t\n\r");
    const trimmed = switch (mode) {
        .both => std.mem.trim(u8, s, chars),
        .left => std.mem.trimLeft(u8, s, chars),
        .right => std.mem.trimRight(u8, s, chars),
    };
    return .{ .string = e.ctx.arena.allocator().dupe(u8, trimmed) catch return EvalError.OutOfMemory };
}

fn loopMethod(e: *Evaluator, lp: *eval.LoopContext, method: []const u8, args: []const *const Expr) EvalError!TemplateInput {
    if (std.mem.eql(u8, method, "cycle")) {
        // loop.cycle('a', 'b', 'c') returns args[index0 % len(args)]
        if (args.len == 0) return .none;
        const idx = lp.index0 % args.len;
        return e.evalExpr(args[idx]);
    }
    return EvalError.UnsupportedMethod;
}

fn cyclerMethod(c: *CyclerState, method: []const u8) EvalError!TemplateInput {
    if (std.mem.eql(u8, method, "next")) {
        if (c.items.len == 0) return .none;
        const item = c.items[c.index];
        c.index = (c.index + 1) % c.items.len;
        return item;
    }
    if (std.mem.eql(u8, method, "current")) {
        if (c.items.len == 0) return .none;
        return c.items[c.index];
    }
    if (std.mem.eql(u8, method, "reset")) {
        c.index = 0;
        return .{ .string = "" };
    }
    return EvalError.UnsupportedMethod;
}

fn stringMethod(e: *Evaluator, s: []const u8, method: []const u8, args: []const *const Expr) EvalError!TemplateInput {
    if (std.mem.eql(u8, method, "strip") or std.mem.eql(u8, method, "trim")) {
        return trimString(e, s, args, .both);
    }

    if (std.mem.eql(u8, method, "lstrip")) {
        return trimString(e, s, args, .left);
    }

    if (std.mem.eql(u8, method, "rstrip")) {
        return trimString(e, s, args, .right);
    }

    if (std.mem.eql(u8, method, "upper")) {
        var result = e.ctx.arena.allocator().alloc(u8, s.len) catch return EvalError.OutOfMemory;
        for (s, 0..) |c, char_idx| result[char_idx] = std.ascii.toUpper(c);
        return .{ .string = result };
    }

    if (std.mem.eql(u8, method, "lower")) {
        var result = e.ctx.arena.allocator().alloc(u8, s.len) catch return EvalError.OutOfMemory;
        for (s, 0..) |c, char_idx| result[char_idx] = std.ascii.toLower(c);
        return .{ .string = result };
    }

    if (std.mem.eql(u8, method, "startswith")) {
        const prefix = try evalStringArg(e, args, 0);
        return .{ .boolean = std.mem.startsWith(u8, s, prefix) };
    }

    if (std.mem.eql(u8, method, "endswith")) {
        const suffix = try evalStringArg(e, args, 0);
        return .{ .boolean = std.mem.endsWith(u8, s, suffix) };
    }

    if (std.mem.eql(u8, method, "split")) {
        const sep = try evalOptionalStringArg(e, args, 0, " ");

        const arena = e.ctx.arena.allocator();
        var result = std.ArrayListUnmanaged(TemplateInput){};
        var it = std.mem.splitSequence(u8, s, sep);
        while (it.next()) |part| {
            const part_copy = arena.dupe(u8, part) catch return EvalError.OutOfMemory;
            result.append(arena, .{ .string = part_copy }) catch return EvalError.OutOfMemory;
        }
        return .{ .array = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
    }

    if (std.mem.eql(u8, method, "replace")) {
        const old = try evalStringArg(e, args, 0);
        const new = try evalStringArg(e, args, 1);
        const result = std.mem.replaceOwned(u8, e.ctx.arena.allocator(), s, old, new) catch return EvalError.OutOfMemory;
        return .{ .string = result };
    }

    if (std.mem.eql(u8, method, "find")) {
        const needle = try evalStringArg(e, args, 0);
        if (std.mem.indexOf(u8, s, needle)) |idx| {
            return .{ .integer = @intCast(idx) };
        }
        return .{ .integer = -1 };
    }

    if (std.mem.eql(u8, method, "count")) {
        const needle = try evalStringArg(e, args, 0);
        var count: i64 = 0;
        var idx: usize = 0;
        while (std.mem.indexOfPos(u8, s, idx, needle)) |pos| {
            count += 1;
            idx = pos + needle.len;
        }
        return .{ .integer = count };
    }

    if (std.mem.eql(u8, method, "title")) {
        if (s.len == 0) return .{ .string = s };
        var result = e.ctx.arena.allocator().alloc(u8, s.len) catch return EvalError.OutOfMemory;
        var capitalize_next = true;
        for (s, 0..) |c, char_idx| {
            if (std.ascii.isAlphabetic(c)) {
                result[char_idx] = if (capitalize_next) std.ascii.toUpper(c) else std.ascii.toLower(c);
                capitalize_next = false;
            } else {
                result[char_idx] = c;
                capitalize_next = (c == ' ' or c == '\t' or c == '\n' or c == '-' or c == '_');
            }
        }
        return .{ .string = result };
    }

    if (std.mem.eql(u8, method, "capitalize")) {
        if (s.len == 0) return .{ .string = s };
        var result = e.ctx.arena.allocator().alloc(u8, s.len) catch return EvalError.OutOfMemory;
        result[0] = std.ascii.toUpper(s[0]);
        for (s[1..], 1..) |c, char_idx| result[char_idx] = std.ascii.toLower(c);
        return .{ .string = result };
    }

    if (std.mem.eql(u8, method, "join")) {
        // "sep".join(items) - join array with separator
        if (args.len < 1) return EvalError.TypeError;
        const items_val = try e.evalExpr(args[0]);
        const items = switch (items_val) {
            .array => |a| a,
            else => return EvalError.TypeError,
        };
        const arena = e.ctx.arena.allocator();
        var result = std.ArrayListUnmanaged(u8){};
        for (items, 0..) |item, item_idx| {
            if (item_idx > 0) result.appendSlice(arena, s) catch return EvalError.OutOfMemory;
            const item_str = item.asString(arena) catch return EvalError.OutOfMemory;
            result.appendSlice(arena, item_str) catch return EvalError.OutOfMemory;
        }
        return .{ .string = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
    }

    return EvalError.UnsupportedMethod;
}

fn arrayMethod(e: *Evaluator, a: []const TemplateInput, method: []const u8, args: []const *const Expr) EvalError!TemplateInput {
    _ = args;

    if (std.mem.eql(u8, method, "reverse")) {
        var result = e.ctx.arena.allocator().alloc(TemplateInput, a.len) catch return EvalError.OutOfMemory;
        for (a, 0..) |item, item_idx| {
            result[a.len - 1 - item_idx] = item;
        }
        return .{ .array = result };
    }

    return EvalError.UnsupportedMethod;
}

fn mapMethod(e: *Evaluator, m: std.StringHashMapUnmanaged(TemplateInput), method: []const u8, args: []const *const Expr) EvalError!TemplateInput {
    const arena = e.ctx.arena.allocator();

    if (std.mem.eql(u8, method, "items")) {
        var result = std.ArrayListUnmanaged(TemplateInput){};
        var it = m.iterator();
        while (it.next()) |entry| {
            const pair = arena.alloc(TemplateInput, 2) catch return EvalError.OutOfMemory;
            pair[0] = .{ .string = entry.key_ptr.* };
            pair[1] = entry.value_ptr.*;
            result.append(arena, .{ .array = pair }) catch return EvalError.OutOfMemory;
        }
        return .{ .array = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
    }

    if (std.mem.eql(u8, method, "keys")) {
        var result = std.ArrayListUnmanaged(TemplateInput){};
        var it = m.iterator();
        while (it.next()) |entry| {
            result.append(arena, .{ .string = entry.key_ptr.* }) catch return EvalError.OutOfMemory;
        }
        return .{ .array = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
    }

    if (std.mem.eql(u8, method, "values")) {
        var result = std.ArrayListUnmanaged(TemplateInput){};
        var it = m.iterator();
        while (it.next()) |entry| {
            result.append(arena, entry.value_ptr.*) catch return EvalError.OutOfMemory;
        }
        return .{ .array = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
    }

    if (std.mem.eql(u8, method, "get")) {
        const key = try evalStringArg(e, args, 0);
        if (m.get(key)) |val| {
            return val;
        }
        // Return default if provided, else none
        if (args.len > 1) {
            return try e.evalExpr(args[1]);
        }
        return .none;
    }

    return EvalError.UnsupportedMethod;
}

// ============================================================================
// Tests
// ============================================================================

const TemplateParser = eval.TemplateParser;

test "callMethod string.upper" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const obj = TemplateInput{ .string = "hello" };
    const result = try callMethod(&eval_ctx, obj, "upper", &.{});
    try std.testing.expectEqualStrings("HELLO", result.string);
}

test "callMethod string.lower" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const obj = TemplateInput{ .string = "HELLO" };
    const result = try callMethod(&eval_ctx, obj, "lower", &.{});
    try std.testing.expectEqualStrings("hello", result.string);
}

test "callMethod string.strip" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const obj = TemplateInput{ .string = "  hello  " };
    const result = try callMethod(&eval_ctx, obj, "strip", &.{});
    try std.testing.expectEqualStrings("hello", result.string);
}

test "callMethod string.trim" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const obj = TemplateInput{ .string = "\t\nhello\n\t" };
    const result = try callMethod(&eval_ctx, obj, "trim", &.{});
    try std.testing.expectEqualStrings("hello", result.string);
}

test "callMethod string.lstrip" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const obj = TemplateInput{ .string = "  hello  " };
    const result = try callMethod(&eval_ctx, obj, "lstrip", &.{});
    try std.testing.expectEqualStrings("hello  ", result.string);
}

test "callMethod string.rstrip" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const obj = TemplateInput{ .string = "  hello  " };
    const result = try callMethod(&eval_ctx, obj, "rstrip", &.{});
    try std.testing.expectEqualStrings("  hello", result.string);
}

test "callMethod string.startswith" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const obj = TemplateInput{ .string = "hello world" };
    const prefix_expr = Expr{ .string = "hello" };
    const args = [_]*const Expr{&prefix_expr};
    const result = try callMethod(&eval_ctx, obj, "startswith", &args);
    try std.testing.expectEqual(true, result.boolean);
}

test "callMethod string.endswith" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const obj = TemplateInput{ .string = "hello world" };
    const suffix_expr = Expr{ .string = "world" };
    const args = [_]*const Expr{&suffix_expr};
    const result = try callMethod(&eval_ctx, obj, "endswith", &args);
    try std.testing.expectEqual(true, result.boolean);
}

test "callMethod string.split" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const obj = TemplateInput{ .string = "a,b,c" };
    const sep_expr = Expr{ .string = "," };
    const args = [_]*const Expr{&sep_expr};
    const result = try callMethod(&eval_ctx, obj, "split", &args);
    try std.testing.expectEqual(@as(usize, 3), result.array.len);
    try std.testing.expectEqualStrings("a", result.array[0].string);
    try std.testing.expectEqualStrings("b", result.array[1].string);
    try std.testing.expectEqualStrings("c", result.array[2].string);
}

test "callMethod string.replace" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const obj = TemplateInput{ .string = "hello world" };
    const old_expr = Expr{ .string = "world" };
    const new_expr = Expr{ .string = "there" };
    const args = [_]*const Expr{ &old_expr, &new_expr };
    const result = try callMethod(&eval_ctx, obj, "replace", &args);
    try std.testing.expectEqualStrings("hello there", result.string);
}

test "callMethod string.find - found" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const obj = TemplateInput{ .string = "hello world" };
    const needle_expr = Expr{ .string = "world" };
    const args = [_]*const Expr{&needle_expr};
    const result = try callMethod(&eval_ctx, obj, "find", &args);
    try std.testing.expectEqual(@as(i64, 6), result.integer);
}

test "callMethod string.find - not found" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const obj = TemplateInput{ .string = "hello world" };
    const needle_expr = Expr{ .string = "xyz" };
    const args = [_]*const Expr{&needle_expr};
    const result = try callMethod(&eval_ctx, obj, "find", &args);
    try std.testing.expectEqual(@as(i64, -1), result.integer);
}

test "callMethod string.count" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const obj = TemplateInput{ .string = "hello hello world" };
    const needle_expr = Expr{ .string = "hello" };
    const args = [_]*const Expr{&needle_expr};
    const result = try callMethod(&eval_ctx, obj, "count", &args);
    try std.testing.expectEqual(@as(i64, 2), result.integer);
}

test "callMethod string.title" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const obj = TemplateInput{ .string = "hello world" };
    const result = try callMethod(&eval_ctx, obj, "title", &.{});
    try std.testing.expectEqualStrings("Hello World", result.string);
}

test "callMethod string.capitalize" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const obj = TemplateInput{ .string = "hello WORLD" };
    const result = try callMethod(&eval_ctx, obj, "capitalize", &.{});
    try std.testing.expectEqualStrings("Hello world", result.string);
}

test "callMethod string.join" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .string = "a" },
        .{ .string = "b" },
        .{ .string = "c" },
    };
    const items_val = TemplateInput{ .array = &items };
    try ctx.set("items", items_val);

    const obj = TemplateInput{ .string = ", " };
    const items_expr = Expr{ .variable = "items" };
    const args = [_]*const Expr{&items_expr};
    const result = try callMethod(&eval_ctx, obj, "join", &args);
    try std.testing.expectEqualStrings("a, b, c", result.string);
}

test "callMethod array.reverse" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 2 },
        .{ .integer = 3 },
    };
    const obj = TemplateInput{ .array = &items };
    const result = try callMethod(&eval_ctx, obj, "reverse", &.{});
    try std.testing.expectEqual(@as(i64, 3), result.array[0].integer);
    try std.testing.expectEqual(@as(i64, 2), result.array[1].integer);
    try std.testing.expectEqual(@as(i64, 1), result.array[2].integer);
}

test "callMethod map.items" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    var map = std.StringHashMapUnmanaged(TemplateInput){};
    try map.put(ctx.arena.allocator(), "key1", .{ .string = "value1" });
    try map.put(ctx.arena.allocator(), "key2", .{ .string = "value2" });
    const obj = TemplateInput{ .map = map };

    const result = try callMethod(&eval_ctx, obj, "items", &.{});
    try std.testing.expectEqual(@as(usize, 2), result.array.len);
    // Each item should be a [key, value] pair
    try std.testing.expectEqual(@as(usize, 2), result.array[0].array.len);
}

test "callMethod map.keys" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    var map = std.StringHashMapUnmanaged(TemplateInput){};
    try map.put(ctx.arena.allocator(), "key1", .{ .string = "value1" });
    try map.put(ctx.arena.allocator(), "key2", .{ .string = "value2" });
    const obj = TemplateInput{ .map = map };

    const result = try callMethod(&eval_ctx, obj, "keys", &.{});
    try std.testing.expectEqual(@as(usize, 2), result.array.len);
}

test "callMethod map.values" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    var map = std.StringHashMapUnmanaged(TemplateInput){};
    try map.put(ctx.arena.allocator(), "key1", .{ .string = "value1" });
    try map.put(ctx.arena.allocator(), "key2", .{ .string = "value2" });
    const obj = TemplateInput{ .map = map };

    const result = try callMethod(&eval_ctx, obj, "values", &.{});
    try std.testing.expectEqual(@as(usize, 2), result.array.len);
}

test "callMethod map.get - with key" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    var map = std.StringHashMapUnmanaged(TemplateInput){};
    try map.put(ctx.arena.allocator(), "key1", .{ .string = "value1" });
    const obj = TemplateInput{ .map = map };

    const key_expr = Expr{ .string = "key1" };
    const args = [_]*const Expr{&key_expr};
    const result = try callMethod(&eval_ctx, obj, "get", &args);
    try std.testing.expectEqualStrings("value1", result.string);
}

test "callMethod map.get - with default" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const map = std.StringHashMapUnmanaged(TemplateInput){};
    const obj = TemplateInput{ .map = map };

    const key_expr = Expr{ .string = "missing" };
    const default_expr = Expr{ .string = "default" };
    const args = [_]*const Expr{ &key_expr, &default_expr };
    const result = try callMethod(&eval_ctx, obj, "get", &args);
    try std.testing.expectEqualStrings("default", result.string);
}

test "callMethod cycler.next" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .string = "a" },
        .{ .string = "b" },
        .{ .string = "c" },
    };
    var cycler_state = CyclerState{ .items = &items, .index = 0 };
    const obj = TemplateInput{ .cycler = &cycler_state };

    const result1 = try callMethod(&eval_ctx, obj, "next", &.{});
    try std.testing.expectEqualStrings("a", result1.string);

    const result2 = try callMethod(&eval_ctx, obj, "next", &.{});
    try std.testing.expectEqualStrings("b", result2.string);

    const result3 = try callMethod(&eval_ctx, obj, "next", &.{});
    try std.testing.expectEqualStrings("c", result3.string);

    // Should wrap around
    const result4 = try callMethod(&eval_ctx, obj, "next", &.{});
    try std.testing.expectEqualStrings("a", result4.string);
}

test "callMethod cycler.current" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .string = "a" },
        .{ .string = "b" },
    };
    var cycler_state = CyclerState{ .items = &items, .index = 1 };
    const obj = TemplateInput{ .cycler = &cycler_state };

    const result = try callMethod(&eval_ctx, obj, "current", &.{});
    try std.testing.expectEqualStrings("b", result.string);
}

test "callMethod cycler.reset" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .string = "a" },
        .{ .string = "b" },
    };
    var cycler_state = CyclerState{ .items = &items, .index = 1 };
    const obj = TemplateInput{ .cycler = &cycler_state };

    _ = try callMethod(&eval_ctx, obj, "reset", &.{});
    try std.testing.expectEqual(@as(usize, 0), cycler_state.index);
}
