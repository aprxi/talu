//! Miscellaneous Filters
//!
//! Utility filters: length, default, type conversion, parsing.

const std = @import("std");
const types = @import("types.zig");
const function_docs = @import("../function_docs.zig");

const TemplateInput = types.TemplateInput;
const EvalError = types.EvalError;
const Evaluator = types.Evaluator;
const Expr = types.Expr;

// ============================================================================
// Length/Count
// ============================================================================

pub fn filterLength(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    return switch (value) {
        .string => |s| .{ .integer = @intCast(s.len) },
        .array => |a| .{ .integer = @intCast(a.len) },
        .map => |m| .{ .integer = @intCast(m.count()) },
        else => EvalError.TypeError,
    };
}

// ============================================================================
// Default Value
// ============================================================================

pub fn filterDefault(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    // Check if second arg (boolean) requests checking for any falsy value
    var check_falsy = false;
    if (args.len > 1) {
        const bool_arg = try e.evalExpr(args[1]);
        check_falsy = bool_arg.isTruthy();
    }

    // Get default value
    const default_val = if (args.len > 0) try e.evalExpr(args[0]) else TemplateInput{ .string = "" };

    // If check_falsy is true, check isTruthy; otherwise just check for none
    if (check_falsy) {
        return if (value.isTruthy()) value else default_val;
    } else {
        return if (value == .none) default_val else value;
    }
}

// ============================================================================
// Type Conversion
// ============================================================================

pub fn filterString(e: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    const str = value.asString(e.ctx.arena.allocator()) catch return EvalError.OutOfMemory;
    return .{ .string = str };
}

pub fn filterList(e: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    const arena = e.ctx.arena.allocator();
    switch (value) {
        .array => return value,
        .string => |input_str| {
            var arr = std.ArrayListUnmanaged(TemplateInput){};
            for (input_str) |c| {
                const char_str = arena.alloc(u8, 1) catch return EvalError.OutOfMemory;
                char_str[0] = c;
                arr.append(arena, .{ .string = char_str }) catch return EvalError.OutOfMemory;
            }
            return .{ .array = arr.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
        },
        else => return EvalError.TypeError,
    }
}

// ============================================================================
// Parsing
// ============================================================================

/// Parse Python source code and extract function documentation.
/// Returns an array of function metadata objects.
pub fn filterParseFunctions(e: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    const source = switch (value) {
        .string => |s| s,
        else => return EvalError.TypeError,
    };

    const arena = e.ctx.arena.allocator();

    // Parse Python source to extract function documentation
    const functions = function_docs.parseFunctions(arena, source) catch return EvalError.OutOfMemory;

    // Convert to TemplateInput
    return function_docs.toTemplateInput(arena, functions) catch return EvalError.OutOfMemory;
}

// ============================================================================
// Unit Tests
// ============================================================================

const TemplateParser = @import("../eval.zig").TemplateParser;

test "filterLength - string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "hello" };
    const result = try filterLength(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(i64, 5), result.integer);
}

test "filterLength - array" {
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
    const result = try filterLength(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(i64, 3), result.integer);
}

test "filterDefault - with none" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .none = {} };
    const default_expr = Expr{ .string = "fallback" };
    const args = [_]*const Expr{&default_expr};
    const result = try filterDefault(&eval_ctx,input, &args);
    try std.testing.expectEqualStrings("fallback", result.string);
}

test "filterDefault - with value" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "value" };
    const default_expr = Expr{ .string = "fallback" };
    const args = [_]*const Expr{&default_expr};
    const result = try filterDefault(&eval_ctx,input, &args);
    try std.testing.expectEqualStrings("value", result.string);
}

test "filterList from array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{ .{ .integer = 1 }, .{ .integer = 2 } };
    const input = TemplateInput{ .array = &items };
    const result = try filterList(&eval_ctx, input, &.{});
    try std.testing.expectEqual(@as(usize, 2), result.array.len);
}

test "filterLength - empty string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "" };
    const result = try filterLength(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(i64, 0), result.integer);
}

test "filterLength - empty array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{};
    const input = TemplateInput{ .array = &items };
    const result = try filterLength(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(i64, 0), result.integer);
}

test "filterLength - map with multiple entries" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    var map = std.StringHashMapUnmanaged(TemplateInput){};
    try map.put(ctx.arena.allocator(), "key1", .{ .integer = 1 });
    try map.put(ctx.arena.allocator(), "key2", .{ .integer = 2 });
    try map.put(ctx.arena.allocator(), "key3", .{ .integer = 3 });
    const input = TemplateInput{ .map = map };
    const result = try filterLength(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(i64, 3), result.integer);
}

test "filterDefault - empty string keeps empty" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "" };
    const default_expr = Expr{ .string = "fallback" };
    const args = [_]*const Expr{&default_expr};
    const result = try filterDefault(&eval_ctx,input, &args);
    // Jinja2 spec: without boolean=True, empty string is defined, not replaced
    try std.testing.expectEqualStrings("", result.string);
}

test "filterDefault - empty string with boolean=True" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "" };
    const default_expr = Expr{ .string = "fallback" };
    const check_falsy_expr = Expr{ .boolean = true };
    const args = [_]*const Expr{ &default_expr, &check_falsy_expr };
    const result = try filterDefault(&eval_ctx,input, &args);
    // With boolean=True, empty string is falsy, so use fallback
    try std.testing.expectEqualStrings("fallback", result.string);
}

test "filterDefault - falsy check with false boolean" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .boolean = false };
    const default_expr = Expr{ .string = "fallback" };
    const check_falsy_expr = Expr{ .boolean = true };
    const args = [_]*const Expr{ &default_expr, &check_falsy_expr };
    const result = try filterDefault(&eval_ctx,input, &args);
    try std.testing.expectEqualStrings("fallback", result.string);
}

test "filterDefault - falsy check with truthy value" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "exists" };
    const default_expr = Expr{ .string = "fallback" };
    const check_falsy_expr = Expr{ .boolean = true };
    const args = [_]*const Expr{ &default_expr, &check_falsy_expr };
    const result = try filterDefault(&eval_ctx,input, &args);
    try std.testing.expectEqualStrings("exists", result.string);
}

test "filterDefault - d alias" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .none = {} };
    const default_expr = Expr{ .string = "fallback" };
    const args = [_]*const Expr{&default_expr};
    const result = try filterDefault(&eval_ctx,input, &args);
    try std.testing.expectEqualStrings("fallback", result.string);
}

test "filterLength - count alias" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "test" };
    const result = try filterLength(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(i64, 4), result.integer);
}

test "filterString - from integer" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .integer = 123 };
    const result = try filterString(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("123", result.string);
}

test "filterString - from boolean" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .boolean = true };
    const result = try filterString(&eval_ctx,input, &.{});
    // Jinja2/Python convention: booleans stringify to "True"/"False"
    try std.testing.expectEqualStrings("True", result.string);
}

test "filterParseFunctions - empty source" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "" };
    const result = try filterParseFunctions(&eval_ctx, input, &.{});
    // Empty source returns empty array
    try std.testing.expectEqual(@as(usize, 0), result.array.len);
}
