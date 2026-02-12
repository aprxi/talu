//! Numeric Filters
//!
//! Filters for numeric operations: conversion, math, aggregation.

const std = @import("std");
const types = @import("types.zig");

const TemplateInput = types.TemplateInput;
const EvalError = types.EvalError;
const Evaluator = types.Evaluator;
const Expr = types.Expr;

// ============================================================================
// Type Conversion
// ============================================================================

pub fn filterInt(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    return switch (value) {
        .integer => value,
        .float => |f| .{ .integer = @intFromFloat(f) },
        .string => |s| blk: {
            const parsed_int = std.fmt.parseInt(i64, s, 10) catch return .{ .integer = 0 };
            break :blk .{ .integer = parsed_int };
        },
        .boolean => |b| .{ .integer = if (b) 1 else 0 },
        else => .{ .integer = 0 },
    };
}

pub fn filterFloat(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    return switch (value) {
        .float => value,
        .integer => |int_val| .{ .float = @floatFromInt(int_val) },
        .string => |input_str| blk: {
            const parsed_float = std.fmt.parseFloat(f64, input_str) catch return .{ .float = 0.0 };
            break :blk .{ .float = parsed_float };
        },
        else => .{ .float = 0.0 },
    };
}

// ============================================================================
// Math Operations
// ============================================================================

pub fn filterAbs(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    return switch (value) {
        .integer => |int_val| .{ .integer = if (int_val < 0) -int_val else int_val },
        .float => |f| .{ .float = @abs(f) },
        else => EvalError.TypeError,
    };
}

pub fn filterRound(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    const f: f64 = switch (value) {
        .integer => |int_val| @floatFromInt(int_val),
        .float => |fl| fl,
        else => return EvalError.TypeError,
    };

    var precision: i64 = 0;
    if (args.len > 0) {
        const prec_val = try e.evalExpr(args[0]);
        precision = switch (prec_val) {
            .integer => |int_val| int_val,
            else => return EvalError.TypeError,
        };
    }

    if (precision == 0) {
        return .{ .float = @round(f) };
    }

    const factor = std.math.pow(f64, 10.0, @floatFromInt(precision));
    return .{ .float = @round(f * factor) / factor };
}

// ============================================================================
// Aggregation
// ============================================================================

pub fn filterSum(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    const arr = switch (value) {
        .array => |a| a,
        else => return EvalError.TypeError,
    };

    // Get optional start value
    var total: f64 = 0;
    if (args.len > 0) {
        const start_val = try e.evalExpr(args[0]);
        total = start_val.asNumber() orelse 0;
    }

    var is_float = false;
    for (arr) |item| {
        switch (item) {
            .integer => |int_val| total += @floatFromInt(int_val),
            .float => |f| {
                total += f;
                is_float = true;
            },
            else => {},
        }
    }

    if (is_float) {
        return .{ .float = total };
    } else {
        return .{ .integer = @intFromFloat(total) };
    }
}

pub fn filterMin(e: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    const arr = switch (value) {
        .array => |a| a,
        else => return EvalError.TypeError,
    };

    if (arr.len == 0) return .none;

    var min_val = arr[0];
    for (arr[1..]) |item| {
        const cmp = try compareValues(e, min_val, item);
        if (cmp > 0) min_val = item;
    }
    return min_val;
}

pub fn filterMax(e: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    const arr = switch (value) {
        .array => |a| a,
        else => return EvalError.TypeError,
    };

    if (arr.len == 0) return .none;

    var max_val = arr[0];
    for (arr[1..]) |item| {
        const cmp = try compareValues(e, max_val, item);
        if (cmp < 0) max_val = item;
    }
    return max_val;
}

// ============================================================================
// Helper Functions
// ============================================================================

pub fn compareValues(_: *Evaluator, a: TemplateInput, b: TemplateInput) EvalError!i32 {
    // Compare two values, return -1, 0, or 1
    switch (a) {
        .integer => |ai| {
            const bi = switch (b) {
                .integer => |int_val| int_val,
                .float => |f| return if (@as(f64, @floatFromInt(ai)) < f) @as(i32, -1) else if (@as(f64, @floatFromInt(ai)) > f) @as(i32, 1) else @as(i32, 0),
                else => return EvalError.TypeError,
            };
            return if (ai < bi) @as(i32, -1) else if (ai > bi) @as(i32, 1) else @as(i32, 0);
        },
        .float => |af| {
            const bf: f64 = switch (b) {
                .integer => |int_val| @floatFromInt(int_val),
                .float => |f| f,
                else => return EvalError.TypeError,
            };
            return if (af < bf) @as(i32, -1) else if (af > bf) @as(i32, 1) else @as(i32, 0);
        },
        .string => |as| {
            const bs = switch (b) {
                .string => |s| s,
                else => return EvalError.TypeError,
            };
            return if (std.mem.lessThan(u8, as, bs)) @as(i32, -1) else if (std.mem.lessThan(u8, bs, as)) @as(i32, 1) else @as(i32, 0);
        },
        else => return EvalError.TypeError,
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

const TemplateParser = @import("../eval.zig").TemplateParser;

test "filterInt - from string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "42" };
    const result = try filterInt(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(i64, 42), result.integer);
}

test "filterInt - from float" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .float = 42.7 };
    const result = try filterInt(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(i64, 42), result.integer);
}

test "filterAbs - negative integer" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .integer = -42 };
    const result = try filterAbs(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(i64, 42), result.integer);
}

test "filterAbs - negative float" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .float = -3.14 };
    const result = try filterAbs(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(f64, 3.14), result.float);
}

test "filterRound - basic" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .float = 3.7 };
    const result = try filterRound(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(f64, 4.0), result.float);
}

test "filterSum" {
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
    const result = try filterSum(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(i64, 6), result.integer);
}

test "filterMin" {
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
    const result = try filterMin(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(i64, 1), result.integer);
}

test "filterMax" {
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
    const result = try filterMax(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(i64, 3), result.integer);
}

test "filterInt from string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "42" };
    const result = try filterInt(&eval_ctx, input, &.{});
    try std.testing.expectEqual(@as(i64, 42), result.integer);
}

test "filterAbs negative integer" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .integer = -42 };
    const result = try filterAbs(&eval_ctx, input, &.{});
    try std.testing.expectEqual(@as(i64, 42), result.integer);
}

test "filterRound float" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .float = 3.7 };
    const result = try filterRound(&eval_ctx, input, &.{});
    // round(3.7) = 4.0, returns float
    try std.testing.expectEqual(@as(f64, 4.0), result.float);
}

test "filterFloat - from string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "3.14" };
    const result = try filterFloat(&eval_ctx, input, &.{});
    try std.testing.expectEqual(@as(f64, 3.14), result.float);
}

test "filterFloat - from integer" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .integer = 42 };
    const result = try filterFloat(&eval_ctx, input, &.{});
    try std.testing.expectEqual(@as(f64, 42.0), result.float);
}

test "filterRound - with precision" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .float = 3.14159 };
    const precision_expr = Expr{ .integer = 2 };
    const args = [_]*const Expr{&precision_expr};
    const result = try filterRound(&eval_ctx,input, &args);
    try std.testing.expectEqual(@as(f64, 3.14), result.float);
}

test "filterAbs - positive integer" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .integer = 42 };
    const result = try filterAbs(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(i64, 42), result.integer);
}

test "filterAbs - zero" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .integer = 0 };
    const result = try filterAbs(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(i64, 0), result.integer);
}

test "compareValues - integers" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const a = TemplateInput{ .integer = 5 };
    const b = TemplateInput{ .integer = 10 };
    const c = TemplateInput{ .integer = 5 };

    try std.testing.expectEqual(@as(i32, -1), try compareValues(&eval_ctx, a, b));
    try std.testing.expectEqual(@as(i32, 1), try compareValues(&eval_ctx, b, a));
    try std.testing.expectEqual(@as(i32, 0), try compareValues(&eval_ctx, a, c));
}
