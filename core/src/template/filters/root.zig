//! Jinja2 Built-in Filters
//!
//! Main entry point for the filters module. Implements the subset of
//! Jinja2 filters needed for LLM chat templates.

const std = @import("std");
const io = @import("../../io/root.zig");
const types = @import("types.zig");

// Import filter modules
pub const string = @import("string.zig");
pub const array = @import("array.zig");
pub const numeric = @import("numeric.zig");
pub const format = @import("format.zig");
pub const select = @import("select.zig");
pub const misc = @import("misc.zig");

const TemplateInput = types.TemplateInput;
const EvalError = types.EvalError;
const Evaluator = types.Evaluator;
const CustomFilter = types.CustomFilter;
const Expr = types.Expr;

// Re-export commonly used types
pub const FilterFn = types.FilterFn;

// Re-export public filter functions for direct access
pub const filterInt = numeric.filterInt;
pub const filterList = misc.filterList;
pub const filterItems = array.filterItems;
pub const filterAbs = numeric.filterAbs;
pub const filterRound = numeric.filterRound;

// ============================================================================
// Main Filter Dispatcher
// ============================================================================

pub fn applyFilter(e: *Evaluator, name: []const u8, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    // 1. Check custom filters first (allows overriding built-ins)
    if (e.custom_filters) |custom| {
        if (custom.get(name)) |filter| {
            return applyCustomFilter(e, filter, value, args);
        }
    }

    // 2. Check built-in filters
    const map = std.StaticStringMap(FilterFn).initComptime(.{
        // Format filters
        .{ "tojson", &format.filterTojson },
        .{ "pprint", &format.filterTojson }, // alias
        .{ "indent", &format.filterIndent },
        .{ "filesizeformat", &format.filterFilesizeformat },
        .{ "xmlattr", &format.filterXmlattr },
        .{ "format", &format.filterFormat },
        .{ "json_schema", &format.filterJsonSchema },

        // String filters
        .{ "lower", &string.filterLower },
        .{ "upper", &string.filterUpper },
        .{ "capitalize", &string.filterCapitalize },
        .{ "title", &string.filterTitle },
        .{ "trim", &string.filterTrim },
        .{ "strip", &string.filterTrim }, // alias
        .{ "replace", &string.filterReplace },
        .{ "split", &string.filterSplit },
        .{ "escape", &string.filterEscape },
        .{ "e", &string.filterEscape }, // alias
        .{ "forceescape", &string.filterEscape }, // alias
        .{ "safe", &string.filterSafe },
        .{ "striptags", &string.filterStriptags },
        .{ "urlencode", &string.filterUrlencode },
        .{ "urlize", &string.filterUrlize },
        .{ "wordwrap", &string.filterWordwrap },
        .{ "center", &string.filterCenter },
        .{ "truncate", &string.filterTruncate },
        .{ "wordcount", &string.filterWordcount },

        // Array filters
        .{ "first", &array.filterFirst },
        .{ "last", &array.filterLast },
        .{ "join", &array.filterJoin },
        .{ "reverse", &array.filterReverse },
        .{ "sort", &array.filterSort },
        .{ "unique", &array.filterUnique },
        .{ "batch", &array.filterBatch },
        .{ "slice", &array.filterSlice },
        .{ "dictsort", &array.filterDictsort },
        .{ "items", &array.filterItems },
        .{ "random", &array.filterRandom },

        // Numeric filters
        .{ "int", &numeric.filterInt },
        .{ "float", &numeric.filterFloat },
        .{ "abs", &numeric.filterAbs },
        .{ "round", &numeric.filterRound },
        .{ "sum", &numeric.filterSum },
        .{ "min", &numeric.filterMin },
        .{ "max", &numeric.filterMax },

        // Select filters
        .{ "select", &select.filterSelect },
        .{ "reject", &select.filterReject },
        .{ "selectattr", &select.filterSelectattr },
        .{ "rejectattr", &select.filterRejectattr },
        .{ "attr", &select.filterAttr },
        .{ "map", &select.filterMap },
        .{ "groupby", &select.filterGroupby },

        // Misc filters
        .{ "length", &misc.filterLength },
        .{ "count", &misc.filterLength }, // alias
        .{ "default", &misc.filterDefault },
        .{ "d", &misc.filterDefault }, // alias
        .{ "string", &misc.filterString },
        .{ "list", &misc.filterList },
        .{ "parse_functions", &misc.filterParseFunctions },
    });

    if (map.get(name)) |filter_fn| {
        return filter_fn(e, value, args);
    }

    return EvalError.UnsupportedFilter;
}

// ============================================================================
// Custom Filter Support
// ============================================================================

/// Apply a custom filter by calling back to Python/C.
/// Serializes value and args to JSON, calls the callback, deserializes result.
fn applyCustomFilter(
    e: *Evaluator,
    filter: CustomFilter,
    value: TemplateInput,
    args: []const *const Expr,
) EvalError!TemplateInput {
    const arena = e.ctx.arena.allocator();

    // 1. Serialize value to JSON
    var value_buf = std.ArrayListUnmanaged(u8){};
    value.writeJson(value_buf.writer(arena)) catch return EvalError.OutOfMemory;
    const value_json = arena.dupeZ(u8, value_buf.items) catch return EvalError.OutOfMemory;

    // 2. Evaluate and serialize args to JSON array
    var args_buf = std.ArrayListUnmanaged(u8){};
    const writer = args_buf.writer(arena);
    writer.writeByte('[') catch return EvalError.OutOfMemory;
    for (args, 0..) |arg_expr, i| {
        if (i > 0) writer.writeByte(',') catch return EvalError.OutOfMemory;
        const arg_value = try e.evalExpr(arg_expr);
        arg_value.writeJson(writer) catch return EvalError.OutOfMemory;
    }
    writer.writeByte(']') catch return EvalError.OutOfMemory;
    const args_json = arena.dupeZ(u8, args_buf.items) catch return EvalError.OutOfMemory;

    // 3. Call the Python/C callback
    const result_json = filter.callback(value_json.ptr, args_json.ptr, filter.user_data);
    if (result_json == null) {
        return EvalError.CustomFilterError;
    }

    // Result must be freed with c_allocator (allocated by Python)
    const result_slice = std.mem.span(result_json.?);
    defer std.heap.c_allocator.free(result_slice);

    // 4. Parse result JSON back to TemplateInput
    return parseJsonToTemplateInput(arena, result_slice) catch {
        return EvalError.CustomFilterError;
    };
}

/// Parse a JSON string into a TemplateInput value.
fn parseJsonToTemplateInput(allocator: std.mem.Allocator, json_str: []const u8) !TemplateInput {
    const parsed = io.json.parseValue(allocator, json_str, .{
        .max_size_bytes = 10 * 1024 * 1024,
        .max_value_bytes = 10 * 1024 * 1024,
        .max_string_bytes = 1 * 1024 * 1024,
    }) catch |err| {
        return switch (err) {
            error.InputTooLarge => error.InvalidJson,
            error.InputTooDeep => error.InvalidJson,
            error.StringTooLong => error.InvalidJson,
            error.InvalidJson => error.InvalidJson,
            error.OutOfMemory => error.OutOfMemory,
        };
    };
    // Note: parsed.deinit() not called - arena will clean up
    return jsonValueToTemplateInput(allocator, parsed.value);
}

/// Convert std.json.Value to TemplateInput.
fn jsonValueToTemplateInput(allocator: std.mem.Allocator, value: std.json.Value) !TemplateInput {
    return switch (value) {
        .null => .none,
        .bool => |b| .{ .boolean = b },
        .integer => |i| .{ .integer = i },
        .float => |f| .{ .float = f },
        .string => |s| .{ .string = s },
        .array => |arr| {
            var items = try allocator.alloc(TemplateInput, arr.items.len);
            errdefer allocator.free(items);
            for (arr.items, 0..) |item, i| {
                items[i] = try jsonValueToTemplateInput(allocator, item);
            }
            return .{ .array = items };
        },
        .object => |obj| {
            var map_result = std.StringHashMapUnmanaged(TemplateInput){};
            var iter = obj.iterator();
            while (iter.next()) |entry| {
                const val = try jsonValueToTemplateInput(allocator, entry.value_ptr.*);
                try map_result.put(allocator, entry.key_ptr.*, val);
            }
            return .{ .map = map_result };
        },
        .number_string => .none, // Not supported
    };
}

// ============================================================================
// Tests
// ============================================================================

const TemplateParser = @import("../eval.zig").TemplateParser;

test "applyFilter - upper" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "hello" };
    const result = try applyFilter(&eval_ctx, "upper", input, &.{});
    try std.testing.expectEqualStrings("HELLO", result.string);
}

test "applyFilter - lower" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "HELLO" };
    const result = try applyFilter(&eval_ctx, "lower", input, &.{});
    try std.testing.expectEqualStrings("hello", result.string);
}

test "applyFilter - length" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "hello" };
    const result = try applyFilter(&eval_ctx, "length", input, &.{});
    try std.testing.expectEqual(@as(i64, 5), result.integer);
}

test "applyFilter - unsupported filter" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "hello" };
    const result = applyFilter(&eval_ctx, "nonexistent_filter", input, &.{});
    try std.testing.expectError(EvalError.UnsupportedFilter, result);
}
