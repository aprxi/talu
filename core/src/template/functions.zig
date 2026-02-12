//! Jinja2 Built-in Functions
//!
//! Implements callable built-ins like range(), len(), and joiner().

const std = @import("std");
const ast = @import("ast.zig");
const eval = @import("eval.zig");
const filters = @import("filters/root.zig");

const TemplateInput = eval.TemplateInput;
const EvalError = eval.EvalError;
const Evaluator = eval.Evaluator;
const JoinerState = eval.JoinerState;
const CyclerState = eval.CyclerState;
const Expr = ast.Expr;

pub fn callFunction(e: *Evaluator, name: []const u8, args: []const *const Expr) EvalError!TemplateInput {
    const map = std.StaticStringMap(*const fn (*Evaluator, []const *const Expr) EvalError!TemplateInput).initComptime(.{
        .{ "range", functionRange },
        .{ "len", functionLen },
        .{ "length", functionLen },
        .{ "count", functionLen },
        .{ "str", functionStr },
        .{ "string", functionStr },
        .{ "int", functionInt },
        .{ "float", functionFloat },
        .{ "dict", functionDict },
        .{ "list", functionList },
        .{ "items", functionItems },
        .{ "raise_exception", functionRaiseException },
        .{ "cycler", functionCycler },
        .{ "joiner", functionJoiner },
        .{ "lipsum", functionLipsum },
        .{ "equalto", functionEqualTo },
        .{ "sameas", functionEqualTo },
        .{ "eq", functionEqualTo },
        .{ "defined", functionDefined },
        .{ "abs", functionAbs },
        .{ "round", functionRound },
        .{ "max", functionMax },
        .{ "min", functionMin },
        .{ "sum", functionSum },
        .{ "strftime_now", functionStrftimeNow },
        .{ "caller", functionCaller },
    });

    if (map.get(name)) |fn_ptr| {
        return fn_ptr(e, args);
    }

    return EvalError.UnsupportedMethod;
}

fn evalArg(e: *Evaluator, args: []const *const Expr, index: usize) EvalError!TemplateInput {
    if (args.len <= index) return EvalError.TypeError;
    return e.evalExpr(args[index]);
}

fn evalStringArg(e: *Evaluator, args: []const *const Expr, index: usize) EvalError![]const u8 {
    const value = try evalArg(e, args, index);
    return switch (value) {
        .string => |s| s,
        else => EvalError.TypeError,
    };
}

fn evalIntArg(e: *Evaluator, args: []const *const Expr, index: usize) EvalError!i64 {
    const value = try evalArg(e, args, index);
    return switch (value) {
        .integer => |int_val| int_val,
        else => EvalError.TypeError,
    };
}

fn functionRange(e: *Evaluator, args: []const *const Expr) EvalError!TemplateInput {
    if (args.len < 1) return EvalError.TypeError;

    var start: i64 = 0;
    var stop: i64 = undefined; // Safe: both branches assign before use
    var step: i64 = 1;

    if (args.len == 1) {
        stop = try evalIntArg(e, args, 0);
    } else {
        start = try evalIntArg(e, args, 0);
        stop = try evalIntArg(e, args, 1);
        if (args.len > 2) {
            step = try evalIntArg(e, args, 2);
        }
    }

    if (step == 0) return EvalError.InvalidOperation;

    const arena = e.ctx.arena.allocator();
    var result = std.ArrayListUnmanaged(TemplateInput){};

    if (step > 0) {
        var range_val = start;
        while (range_val < stop) : (range_val += step) {
            result.append(arena, .{ .integer = range_val }) catch return EvalError.OutOfMemory;
        }
    } else {
        var range_val = start;
        while (range_val > stop) : (range_val += step) {
            result.append(arena, .{ .integer = range_val }) catch return EvalError.OutOfMemory;
        }
    }

    return .{ .array = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

fn functionLen(e: *Evaluator, args: []const *const Expr) EvalError!TemplateInput {
    const value = try evalArg(e, args, 0);
    return switch (value) {
        .string => |s| .{ .integer = @intCast(s.len) },
        .array => |a| .{ .integer = @intCast(a.len) },
        .map => |m| .{ .integer = @intCast(m.count()) },
        else => EvalError.TypeError,
    };
}

fn functionStr(e: *Evaluator, args: []const *const Expr) EvalError!TemplateInput {
    const value = try evalArg(e, args, 0);
    const str = value.asString(e.ctx.arena.allocator()) catch return EvalError.OutOfMemory;
    return .{ .string = str };
}

fn functionInt(e: *Evaluator, args: []const *const Expr) EvalError!TemplateInput {
    const value = try evalArg(e, args, 0);
    return filters.filterInt(e, value, &.{});
}

fn functionFloat(e: *Evaluator, args: []const *const Expr) EvalError!TemplateInput {
    const value = try evalArg(e, args, 0);
    return switch (value) {
        .integer => |int_val| .{ .float = @floatFromInt(int_val) },
        .float => value,
        .string => |s| blk: {
            const parsed_float = std.fmt.parseFloat(f64, s) catch return .{ .float = 0.0 };
            break :blk .{ .float = parsed_float };
        },
        else => .{ .float = 0.0 },
    };
}

fn functionDict(_: *Evaluator, _: []const *const Expr) EvalError!TemplateInput {
    // dict() creates an empty dict, dict(key=value, ...) creates with values
    // For simplicity, we return an empty map - keyword args would need parser support
    return .{ .map = .{} };
}

fn functionList(e: *Evaluator, args: []const *const Expr) EvalError!TemplateInput {
    if (args.len < 1) return .{ .array = &.{} };
    const value = try e.evalExpr(args[0]);
    return filters.filterList(e, value, &.{});
}

fn functionItems(e: *Evaluator, args: []const *const Expr) EvalError!TemplateInput {
    const value = try evalArg(e, args, 0);
    return filters.filterItems(e, value, &.{});
}

fn functionRaiseException(e: *Evaluator, args: []const *const Expr) EvalError!TemplateInput {
    // raise_exception(message) - throws an error for template validation
    // Store the message in the context (TemplateParser) so it survives after Evaluator is destroyed
    if (args.len > 0) {
        const msg = try evalArg(e, args, 0);
        if (msg == .string) {
            // Store in arena so it lives until context is freed
            const msg_copy = e.ctx.arena.allocator().dupe(u8, msg.string) catch return EvalError.OutOfMemory;
            e.ctx.raise_exception_message = msg_copy;
        }
    }
    return EvalError.RaiseException;
}

fn functionCycler(e: *Evaluator, args: []const *const Expr) EvalError!TemplateInput {
    if (args.len < 1) return .none;
    const arena = e.ctx.arena.allocator();

    const items = arena.alloc(TemplateInput, args.len) catch return EvalError.OutOfMemory;
    for (args, 0..) |arg, i| {
        items[i] = try e.evalExpr(arg);
    }

    const state = arena.create(CyclerState) catch return EvalError.OutOfMemory;
    state.* = .{ .items = items, .index = 0 };
    return .{ .cycler = state };
}

fn functionJoiner(e: *Evaluator, args: []const *const Expr) EvalError!TemplateInput {
    const arena = e.ctx.arena.allocator();
    var sep: []const u8 = ", ";
    if (args.len > 0) {
        sep = try evalStringArg(e, args, 0);
    }
    const state = arena.create(JoinerState) catch return EvalError.OutOfMemory;
    state.* = .{ .separator = sep, .called = false };
    return .{ .joiner = state };
}

fn functionLipsum(e: *Evaluator, args: []const *const Expr) EvalError!TemplateInput {
    const text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit.";
    var use_html = false;
    if (args.len >= 2) {
        const html_arg = try evalArg(e, args, 1);
        if (html_arg == .boolean and html_arg.boolean) {
            use_html = true;
        }
    }
    if (use_html) {
        return .{ .string = "<p>" ++ text ++ "</p>" };
    }
    return .{ .string = text };
}

fn functionEqualTo(e: *Evaluator, args: []const *const Expr) EvalError!TemplateInput {
    if (args.len >= 2) {
        const left_value = try e.evalExpr(args[0]);
        const right_value = try e.evalExpr(args[1]);
        return .{ .boolean = left_value.eql(right_value) };
    }
    return .none;
}

fn functionDefined(e: *Evaluator, args: []const *const Expr) EvalError!TemplateInput {
    if (args.len < 1) return .{ .boolean = false };
    const value = try e.evalExpr(args[0]);
    return .{ .boolean = value != .none };
}

fn functionAbs(e: *Evaluator, args: []const *const Expr) EvalError!TemplateInput {
    const value = try evalArg(e, args, 0);
    return filters.filterAbs(e, value, &.{});
}

fn functionRound(e: *Evaluator, args: []const *const Expr) EvalError!TemplateInput {
    const value = try evalArg(e, args, 0);
    return filters.filterRound(e, value, args[1..]);
}

fn functionMax(e: *Evaluator, args: []const *const Expr) EvalError!TemplateInput {
    const value = try evalArg(e, args, 0);
    switch (value) {
        .array => |arr| {
            if (arr.len == 0) return .none;
            var max_val = arr[0];
            for (arr[1..]) |item| {
                const a_num = max_val.asNumber() orelse continue;
                const b_num = item.asNumber() orelse continue;
                if (b_num > a_num) max_val = item;
            }
            return max_val;
        },
        else => return value,
    }
}

fn functionMin(e: *Evaluator, args: []const *const Expr) EvalError!TemplateInput {
    const value = try evalArg(e, args, 0);
    switch (value) {
        .array => |arr| {
            if (arr.len == 0) return .none;
            var min_val = arr[0];
            for (arr[1..]) |item| {
                const a_num = min_val.asNumber() orelse continue;
                const b_num = item.asNumber() orelse continue;
                if (b_num < a_num) min_val = item;
            }
            return min_val;
        },
        else => return value,
    }
}

fn functionSum(e: *Evaluator, args: []const *const Expr) EvalError!TemplateInput {
    const value = try evalArg(e, args, 0);
    switch (value) {
        .array => |arr| {
            var total: f64 = 0;
            for (arr) |item| {
                if (item.asNumber()) |n| total += n;
            }
            return .{ .float = total };
        },
        else => return value,
    }
}

fn functionStrftimeNow(e: *Evaluator, args: []const *const Expr) EvalError!TemplateInput {
    var format: []const u8 = "%Y-%m-%d";
    if (args.len > 0) {
        const fmt_val = try e.evalExpr(args[0]);
        format = switch (fmt_val) {
            .string => |s| s,
            else => "%Y-%m-%d",
        };
    }

    const timestamp = std.time.timestamp();
    const epoch_secs: std.time.epoch.EpochSeconds = .{ .secs = @intCast(timestamp) };
    const day_secs = epoch_secs.getDaySeconds();
    const epoch_day = epoch_secs.getEpochDay();
    const year_day = epoch_day.calculateYearDay();
    const month_day = year_day.calculateMonthDay();

    const arena = e.ctx.arena.allocator();
    var result = std.ArrayListUnmanaged(u8){};

    var i: usize = 0;
    while (i < format.len) : (i += 1) {
        if (format[i] == '%' and i + 1 < format.len) {
            const spec = format[i + 1];
            i += 1;
            switch (spec) {
                'Y' => {
                    const year_str = std.fmt.allocPrint(arena, "{d}", .{year_day.year}) catch return EvalError.OutOfMemory;
                    result.appendSlice(arena, year_str) catch return EvalError.OutOfMemory;
                },
                'm' => {
                    const month_str = std.fmt.allocPrint(arena, "{d:0>2}", .{month_day.month.numeric()}) catch return EvalError.OutOfMemory;
                    result.appendSlice(arena, month_str) catch return EvalError.OutOfMemory;
                },
                'd' => {
                    const day_str = std.fmt.allocPrint(arena, "{d:0>2}", .{month_day.day_index + 1}) catch return EvalError.OutOfMemory;
                    result.appendSlice(arena, day_str) catch return EvalError.OutOfMemory;
                },
                'H' => {
                    const hour_str = std.fmt.allocPrint(arena, "{d:0>2}", .{day_secs.getHoursIntoDay()}) catch return EvalError.OutOfMemory;
                    result.appendSlice(arena, hour_str) catch return EvalError.OutOfMemory;
                },
                'M' => {
                    const min_str = std.fmt.allocPrint(arena, "{d:0>2}", .{day_secs.getMinutesIntoHour()}) catch return EvalError.OutOfMemory;
                    result.appendSlice(arena, min_str) catch return EvalError.OutOfMemory;
                },
                'S' => {
                    const sec_str = std.fmt.allocPrint(arena, "{d:0>2}", .{day_secs.getSecondsIntoMinute()}) catch return EvalError.OutOfMemory;
                    result.appendSlice(arena, sec_str) catch return EvalError.OutOfMemory;
                },
                'B' => {
                    const month_names = [_][]const u8{
                        "January", "February", "March",     "April",   "May",      "June",
                        "July",    "August",   "September", "October", "November", "December",
                    };
                    const month_idx = month_day.month.numeric() - 1;
                    if (month_idx < 12) {
                        result.appendSlice(arena, month_names[month_idx]) catch return EvalError.OutOfMemory;
                    }
                },
                'b' => {
                    const month_names = [_][]const u8{
                        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
                    };
                    const month_idx = month_day.month.numeric() - 1;
                    if (month_idx < 12) {
                        result.appendSlice(arena, month_names[month_idx]) catch return EvalError.OutOfMemory;
                    }
                },
                '%' => result.append(arena, '%') catch return EvalError.OutOfMemory,
                else => {
                    result.append(arena, '%') catch return EvalError.OutOfMemory;
                    result.append(arena, spec) catch return EvalError.OutOfMemory;
                },
            }
        } else {
            result.append(arena, format[i]) catch return EvalError.OutOfMemory;
        }
    }

    return .{ .string = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

fn functionCaller(e: *Evaluator, _: []const *const Expr) EvalError!TemplateInput {
    if (e.caller_body) |body| {
        const content = try e.renderNodesToString(body);
        return .{ .string = content };
    }
    return .{ .string = "" };
}

// ============================================================================
// Tests
// ============================================================================

const TemplateParser = eval.TemplateParser;

test "callFunction range - single arg" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arg_expr = Expr{ .integer = 5 };
    const args = [_]*const Expr{&arg_expr};
    const result = try callFunction(&eval_ctx, "range", &args);
    try std.testing.expectEqual(@as(usize, 5), result.array.len);
    try std.testing.expectEqual(@as(i64, 0), result.array[0].integer);
    try std.testing.expectEqual(@as(i64, 4), result.array[4].integer);
}

test "callFunction range - two args" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const start_expr = Expr{ .integer = 2 };
    const stop_expr = Expr{ .integer = 5 };
    const args = [_]*const Expr{ &start_expr, &stop_expr };
    const result = try callFunction(&eval_ctx, "range", &args);
    try std.testing.expectEqual(@as(usize, 3), result.array.len);
    try std.testing.expectEqual(@as(i64, 2), result.array[0].integer);
    try std.testing.expectEqual(@as(i64, 4), result.array[2].integer);
}

test "callFunction range - with step" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const start_expr = Expr{ .integer = 0 };
    const stop_expr = Expr{ .integer = 10 };
    const step_expr = Expr{ .integer = 2 };
    const args = [_]*const Expr{ &start_expr, &stop_expr, &step_expr };
    const result = try callFunction(&eval_ctx, "range", &args);
    try std.testing.expectEqual(@as(usize, 5), result.array.len);
    try std.testing.expectEqual(@as(i64, 0), result.array[0].integer);
    try std.testing.expectEqual(@as(i64, 8), result.array[4].integer);
}

test "callFunction len - string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arg_expr = Expr{ .string = "hello" };
    const args = [_]*const Expr{&arg_expr};
    const result = try callFunction(&eval_ctx, "len", &args);
    try std.testing.expectEqual(@as(i64, 5), result.integer);
}

test "callFunction len - array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .string = "a" },
        .{ .string = "b" },
        .{ .string = "c" },
    };
    const array_val = TemplateInput{ .array = &items };
    try ctx.set("arr", array_val);

    const arg_expr = Expr{ .variable = "arr" };
    const args = [_]*const Expr{&arg_expr};
    const result = try callFunction(&eval_ctx, "len", &args);
    try std.testing.expectEqual(@as(i64, 3), result.integer);
}

test "callFunction str" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arg_expr = Expr{ .integer = 42 };
    const args = [_]*const Expr{&arg_expr};
    const result = try callFunction(&eval_ctx, "str", &args);
    try std.testing.expectEqualStrings("42", result.string);
}

test "callFunction int - from string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arg_expr = Expr{ .string = "123" };
    const args = [_]*const Expr{&arg_expr};
    const result = try callFunction(&eval_ctx, "int", &args);
    try std.testing.expectEqual(@as(i64, 123), result.integer);
}

test "callFunction int - from float" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arg_expr = Expr{ .float = 42.7 };
    const args = [_]*const Expr{&arg_expr};
    const result = try callFunction(&eval_ctx, "int", &args);
    try std.testing.expectEqual(@as(i64, 42), result.integer);
}

test "callFunction float - from string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arg_expr = Expr{ .string = "3.14" };
    const args = [_]*const Expr{&arg_expr};
    const result = try callFunction(&eval_ctx, "float", &args);
    try std.testing.expectEqual(@as(f64, 3.14), result.float);
}

test "callFunction float - from int" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arg_expr = Expr{ .integer = 42 };
    const args = [_]*const Expr{&arg_expr};
    const result = try callFunction(&eval_ctx, "float", &args);
    try std.testing.expectEqual(@as(f64, 42.0), result.float);
}

test "callFunction dict" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const args = [_]*const Expr{};
    const result = try callFunction(&eval_ctx, "dict", &args);
    try std.testing.expect(result == .map);
    try std.testing.expectEqual(@as(usize, 0), result.map.count());
}

test "callFunction list - empty" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const args = [_]*const Expr{};
    const result = try callFunction(&eval_ctx, "list", &args);
    try std.testing.expect(result == .array);
    try std.testing.expectEqual(@as(usize, 0), result.array.len);
}

test "callFunction list - from string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arg_expr = Expr{ .string = "abc" };
    const args = [_]*const Expr{&arg_expr};
    const result = try callFunction(&eval_ctx, "list", &args);
    try std.testing.expectEqual(@as(usize, 3), result.array.len);
    try std.testing.expectEqualStrings("a", result.array[0].string);
    try std.testing.expectEqualStrings("b", result.array[1].string);
    try std.testing.expectEqualStrings("c", result.array[2].string);
}

test "callFunction joiner" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const sep_expr = Expr{ .string = ", " };
    const args = [_]*const Expr{&sep_expr};
    const result = try callFunction(&eval_ctx, "joiner", &args);
    try std.testing.expect(result == .joiner);
    try std.testing.expectEqualStrings(", ", result.joiner.separator);
    try std.testing.expectEqual(false, result.joiner.called);
}

test "callFunction cycler" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arg1 = Expr{ .string = "a" };
    const arg2 = Expr{ .string = "b" };
    const arg3 = Expr{ .string = "c" };
    const args = [_]*const Expr{ &arg1, &arg2, &arg3 };
    const result = try callFunction(&eval_ctx, "cycler", &args);
    try std.testing.expect(result == .cycler);
    try std.testing.expectEqual(@as(usize, 3), result.cycler.items.len);
    try std.testing.expectEqual(@as(usize, 0), result.cycler.index);
}

test "callFunction lipsum - plain" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const args = [_]*const Expr{};
    const result = try callFunction(&eval_ctx, "lipsum", &args);
    try std.testing.expectEqualStrings("Lorem ipsum dolor sit amet, consectetur adipiscing elit.", result.string);
}

test "callFunction lipsum - with html" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arg1 = Expr{ .integer = 1 };
    const arg2 = Expr{ .boolean = true };
    const args = [_]*const Expr{ &arg1, &arg2 };
    const result = try callFunction(&eval_ctx, "lipsum", &args);
    try std.testing.expectEqualStrings("<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>", result.string);
}

test "callFunction equalto - true" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arg1 = Expr{ .integer = 42 };
    const arg2 = Expr{ .integer = 42 };
    const args = [_]*const Expr{ &arg1, &arg2 };
    const result = try callFunction(&eval_ctx, "equalto", &args);
    try std.testing.expectEqual(true, result.boolean);
}

test "callFunction equalto - false" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arg1 = Expr{ .integer = 42 };
    const arg2 = Expr{ .integer = 43 };
    const args = [_]*const Expr{ &arg1, &arg2 };
    const result = try callFunction(&eval_ctx, "equalto", &args);
    try std.testing.expectEqual(false, result.boolean);
}

test "callFunction defined - with value" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arg_expr = Expr{ .string = "value" };
    const args = [_]*const Expr{&arg_expr};
    const result = try callFunction(&eval_ctx, "defined", &args);
    try std.testing.expectEqual(true, result.boolean);
}

test "callFunction defined - with none" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arg_expr: Expr = .none;
    const args = [_]*const Expr{&arg_expr};
    const result = try callFunction(&eval_ctx, "defined", &args);
    try std.testing.expectEqual(false, result.boolean);
}

test "callFunction abs" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const arg_expr = Expr{ .integer = -42 };
    const args = [_]*const Expr{&arg_expr};
    const result = try callFunction(&eval_ctx, "abs", &args);
    try std.testing.expectEqual(@as(i64, 42), result.integer);
}

test "callFunction max - array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 5 },
        .{ .integer = 3 },
    };
    const array_val = TemplateInput{ .array = &items };
    try ctx.set("nums", array_val);

    const arg_expr = Expr{ .variable = "nums" };
    const args = [_]*const Expr{&arg_expr};
    const result = try callFunction(&eval_ctx, "max", &args);
    try std.testing.expectEqual(@as(i64, 5), result.integer);
}

test "callFunction min - array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 5 },
        .{ .integer = 3 },
    };
    const array_val = TemplateInput{ .array = &items };
    try ctx.set("nums", array_val);

    const arg_expr = Expr{ .variable = "nums" };
    const args = [_]*const Expr{&arg_expr};
    const result = try callFunction(&eval_ctx, "min", &args);
    try std.testing.expectEqual(@as(i64, 1), result.integer);
}

test "callFunction sum - array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 2 },
        .{ .integer = 3 },
    };
    const array_val = TemplateInput{ .array = &items };
    try ctx.set("nums", array_val);

    const arg_expr = Expr{ .variable = "nums" };
    const args = [_]*const Expr{&arg_expr};
    const result = try callFunction(&eval_ctx, "sum", &args);
    try std.testing.expectEqual(@as(f64, 6.0), result.float);
}
