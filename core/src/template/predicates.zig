//! Jinja2 Built-in Tests
//!
//! Implements the `is test` predicates.

const std = @import("std");
const ast = @import("ast.zig");
const eval = @import("eval.zig");

const TemplateInput = eval.TemplateInput;
const EvalError = eval.EvalError;
const Evaluator = eval.Evaluator;
const Expr = ast.Expr;

// ============================================================================
// Tests (is test)
// ============================================================================

pub fn applyTest(e: *Evaluator, name: []const u8, value: TemplateInput, args: []const *const Expr) EvalError!bool {
    const map = std.StaticStringMap(*const fn (*Evaluator, TemplateInput, []const *const Expr) EvalError!bool).initComptime(.{
        .{ "divisibleby", testDivisibleBy },
        .{ "defined", testDefined },
        .{ "undefined", testUndefined },
        .{ "none", testNone },
        .{ "string", testString },
        .{ "number", testNumber },
        .{ "integer", testInteger },
        .{ "float", testFloat },
        .{ "sequence", testSequence },
        .{ "iterable", testSequence },
        .{ "mapping", testMapping },
        .{ "true", testTrue },
        .{ "false", testFalse },
        .{ "odd", testOdd },
        .{ "even", testEven },
        .{ "equalto", testEqualTo },
        .{ "eq", testEqualTo },
        .{ "sameas", testEqualTo },
        .{ "boolean", testBoolean },
        .{ "callable", testCallable },
    });

    if (map.get(name)) |test_fn| {
        return test_fn(e, value, args);
    }

    return EvalError.UnsupportedTest;
}

fn testDivisibleBy(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!bool {
    if (value != .integer) return false;
    if (args.len == 0) return EvalError.TypeError;
    const arg = try e.evalExpr(args[0]);
    if (arg != .integer or arg.integer == 0) return false;
    return @mod(value.integer, arg.integer) == 0;
}

fn testDefined(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!bool {
    // A variable is "defined" if it's not .none (which is returned for undefined variables)
    return value != .none;
}

fn testUndefined(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!bool {
    return value == .none;
}

fn testNone(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!bool {
    return value == .none;
}

fn testString(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!bool {
    return value == .string;
}

fn testNumber(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!bool {
    return value == .integer or value == .float;
}

fn testInteger(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!bool {
    return value == .integer;
}

fn testFloat(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!bool {
    return value == .float;
}

fn testSequence(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!bool {
    return value == .array or value == .string;
}

fn testMapping(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!bool {
    return value == .map;
}

fn testTrue(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!bool {
    return value == .boolean and value.boolean == true;
}

fn testFalse(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!bool {
    return value == .boolean and value.boolean == false;
}

fn testOdd(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!bool {
    return value == .integer and @mod(value.integer, 2) != 0;
}

fn testEven(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!bool {
    return value == .integer and @mod(value.integer, 2) == 0;
}

fn testEqualTo(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!bool {
    if (args.len == 0) return EvalError.TypeError;
    const other = try e.evalExpr(args[0]);
    return value.eql(other);
}

fn testBoolean(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!bool {
    return value == .boolean;
}

fn testCallable(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!bool {
    return value == .macro or value == .joiner or value == .cycler;
}

// ============================================================================
// Tests
// ============================================================================

const TemplateParser = eval.TemplateParser;

test "applyTest defined - with value" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .string = "test" };
    const result = try applyTest(&eval_ctx, "defined", value, &.{});
    try std.testing.expectEqual(true, result);
}

test "applyTest defined - with none" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .none = {} };
    const result = try applyTest(&eval_ctx, "defined", value, &.{});
    try std.testing.expectEqual(false, result);
}

test "applyTest undefined - with none" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .none = {} };
    const result = try applyTest(&eval_ctx, "undefined", value, &.{});
    try std.testing.expectEqual(true, result);
}

test "applyTest undefined - with value" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .string = "test" };
    const result = try applyTest(&eval_ctx, "undefined", value, &.{});
    try std.testing.expectEqual(false, result);
}

test "applyTest none" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .none = {} };
    const result = try applyTest(&eval_ctx, "none", value, &.{});
    try std.testing.expectEqual(true, result);
}

test "applyTest string - with string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .string = "test" };
    const result = try applyTest(&eval_ctx, "string", value, &.{});
    try std.testing.expectEqual(true, result);
}

test "applyTest string - with number" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .integer = 42 };
    const result = try applyTest(&eval_ctx, "string", value, &.{});
    try std.testing.expectEqual(false, result);
}

test "applyTest number - with integer" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .integer = 42 };
    const result = try applyTest(&eval_ctx, "number", value, &.{});
    try std.testing.expectEqual(true, result);
}

test "applyTest number - with float" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .float = 3.14 };
    const result = try applyTest(&eval_ctx, "number", value, &.{});
    try std.testing.expectEqual(true, result);
}

test "applyTest number - with string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .string = "test" };
    const result = try applyTest(&eval_ctx, "number", value, &.{});
    try std.testing.expectEqual(false, result);
}

test "applyTest integer" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .integer = 42 };
    const result = try applyTest(&eval_ctx, "integer", value, &.{});
    try std.testing.expectEqual(true, result);
}

test "applyTest float" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .float = 3.14 };
    const result = try applyTest(&eval_ctx, "float", value, &.{});
    try std.testing.expectEqual(true, result);
}

test "applyTest sequence - with array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{.{ .integer = 1 }};
    const value = TemplateInput{ .array = &items };
    const result = try applyTest(&eval_ctx, "sequence", value, &.{});
    try std.testing.expectEqual(true, result);
}

test "applyTest sequence - with string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .string = "test" };
    const result = try applyTest(&eval_ctx, "sequence", value, &.{});
    try std.testing.expectEqual(true, result);
}

test "applyTest iterable - alias for sequence" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{.{ .integer = 1 }};
    const value = TemplateInput{ .array = &items };
    const result = try applyTest(&eval_ctx, "iterable", value, &.{});
    try std.testing.expectEqual(true, result);
}

test "applyTest mapping" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const map = std.StringHashMapUnmanaged(TemplateInput){};
    const value = TemplateInput{ .map = map };
    const result = try applyTest(&eval_ctx, "mapping", value, &.{});
    try std.testing.expectEqual(true, result);
}

test "applyTest true - with true" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .boolean = true };
    const result = try applyTest(&eval_ctx, "true", value, &.{});
    try std.testing.expectEqual(true, result);
}

test "applyTest true - with false" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .boolean = false };
    const result = try applyTest(&eval_ctx, "true", value, &.{});
    try std.testing.expectEqual(false, result);
}

test "applyTest false - with false" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .boolean = false };
    const result = try applyTest(&eval_ctx, "false", value, &.{});
    try std.testing.expectEqual(true, result);
}

test "applyTest false - with true" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .boolean = true };
    const result = try applyTest(&eval_ctx, "false", value, &.{});
    try std.testing.expectEqual(false, result);
}

test "applyTest odd" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .integer = 3 };
    const result = try applyTest(&eval_ctx, "odd", value, &.{});
    try std.testing.expectEqual(true, result);
}

test "applyTest even" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .integer = 4 };
    const result = try applyTest(&eval_ctx, "even", value, &.{});
    try std.testing.expectEqual(true, result);
}

test "applyTest divisibleby - true" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .integer = 10 };
    const divisor_expr = Expr{ .integer = 5 };
    const args = [_]*const Expr{&divisor_expr};
    const result = try applyTest(&eval_ctx, "divisibleby", value, &args);
    try std.testing.expectEqual(true, result);
}

test "applyTest divisibleby - false" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .integer = 10 };
    const divisor_expr = Expr{ .integer = 3 };
    const args = [_]*const Expr{&divisor_expr};
    const result = try applyTest(&eval_ctx, "divisibleby", value, &args);
    try std.testing.expectEqual(false, result);
}

test "applyTest equalto - true" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .integer = 42 };
    const other_expr = Expr{ .integer = 42 };
    const args = [_]*const Expr{&other_expr};
    const result = try applyTest(&eval_ctx, "equalto", value, &args);
    try std.testing.expectEqual(true, result);
}

test "applyTest equalto - false" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .integer = 42 };
    const other_expr = Expr{ .integer = 43 };
    const args = [_]*const Expr{&other_expr};
    const result = try applyTest(&eval_ctx, "equalto", value, &args);
    try std.testing.expectEqual(false, result);
}

test "applyTest eq - alias for equalto" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .string = "test" };
    const other_expr = Expr{ .string = "test" };
    const args = [_]*const Expr{&other_expr};
    const result = try applyTest(&eval_ctx, "eq", value, &args);
    try std.testing.expectEqual(true, result);
}

test "applyTest sameas - alias for equalto" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .boolean = true };
    const other_expr = Expr{ .boolean = true };
    const args = [_]*const Expr{&other_expr};
    const result = try applyTest(&eval_ctx, "sameas", value, &args);
    try std.testing.expectEqual(true, result);
}

test "applyTest boolean" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const value = TemplateInput{ .boolean = true };
    const result = try applyTest(&eval_ctx, "boolean", value, &.{});
    try std.testing.expectEqual(true, result);
}

test "applyTest callable - with joiner" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const joiner_state = try ctx.arena.allocator().create(eval.JoinerState);
    joiner_state.* = .{ .separator = ", ", .called = false };
    const value = TemplateInput{ .joiner = joiner_state };
    const result = try applyTest(&eval_ctx, "callable", value, &.{});
    try std.testing.expectEqual(true, result);
}

test "applyTest callable - with cycler" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{.{ .string = "a" }};
    const cycler_state = try ctx.arena.allocator().create(eval.CyclerState);
    cycler_state.* = .{ .items = &items, .index = 0 };
    const value = TemplateInput{ .cycler = cycler_state };
    const result = try applyTest(&eval_ctx, "callable", value, &.{});
    try std.testing.expectEqual(true, result);
}
