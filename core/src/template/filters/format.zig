//! Format Filters
//!
//! Filters for output formatting: JSON, indentation, file sizes, XML.

const std = @import("std");
const types = @import("types.zig");

const TemplateInput = types.TemplateInput;
const EvalError = types.EvalError;
const Evaluator = types.Evaluator;
const Expr = types.Expr;

// ============================================================================
// JSON
// ============================================================================

pub fn filterTojson(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    // Check for indent argument
    var indent: ?usize = null;
    if (args.len > 0) {
        const indent_val = try e.evalExpr(args[0]);
        if (indent_val == .integer and indent_val.integer > 0) {
            indent = @intCast(indent_val.integer);
        }
    }

    const arena = e.ctx.arena.allocator();
    var buffer = std.ArrayListUnmanaged(u8){};
    const writer = buffer.writer(arena);

    if (indent) |ind| {
        writeJsonIndented(value, writer, 0, ind) catch return EvalError.OutOfMemory;
    } else {
        value.writeJson(writer) catch return EvalError.OutOfMemory;
    }

    return .{ .string = buffer.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

fn writeJsonIndented(value: TemplateInput, writer: anytype, depth: usize, indent: usize) !void {
    switch (value) {
        .string => |s| {
            try writer.writeByte('"');
            for (s) |c| {
                switch (c) {
                    '"' => try writer.writeAll("\\\""),
                    '\\' => try writer.writeAll("\\\\"),
                    '\n' => try writer.writeAll("\\n"),
                    '\r' => try writer.writeAll("\\r"),
                    '\t' => try writer.writeAll("\\t"),
                    else => try writer.writeByte(c),
                }
            }
            try writer.writeByte('"');
        },
        .integer => |int_val| try writer.print("{d}", .{int_val}),
        .float => |f| try writer.print("{d}", .{f}),
        .boolean => |b| try writer.writeAll(if (b) "true" else "false"),
        .none => try writer.writeAll("null"),
        .array => |arr| {
            try writer.writeByte('[');
            if (arr.len > 0) {
                try writer.writeByte('\n');
                for (arr, 0..) |item, item_idx| {
                    try writer.writeByteNTimes(' ', (depth + 1) * indent);
                    try writeJsonIndented(item, writer, depth + 1, indent);
                    if (item_idx < arr.len - 1) try writer.writeByte(',');
                    try writer.writeByte('\n');
                }
                try writer.writeByteNTimes(' ', depth * indent);
            }
            try writer.writeByte(']');
        },
        .map => |m| {
            try writer.writeByte('{');
            const count = m.count();
            if (count > 0) {
                try writer.writeByte('\n');
                var it = m.iterator();
                var i: usize = 0;
                while (it.next()) |entry| {
                    try writer.writeByteNTimes(' ', (depth + 1) * indent);
                    try writer.writeByte('"');
                    try writer.writeAll(entry.key_ptr.*);
                    try writer.writeAll("\": ");
                    try writeJsonIndented(entry.value_ptr.*, writer, depth + 1, indent);
                    if (i < count - 1) try writer.writeByte(',');
                    try writer.writeByte('\n');
                    i += 1;
                }
                try writer.writeByteNTimes(' ', depth * indent);
            }
            try writer.writeByte('}');
        },
        .namespace => |ns| {
            try writer.writeByte('{');
            const count = ns.count();
            if (count > 0) {
                try writer.writeByte('\n');
                var it = ns.iterator();
                var i: usize = 0;
                while (it.next()) |entry| {
                    try writer.writeByteNTimes(' ', (depth + 1) * indent);
                    try writer.writeByte('"');
                    try writer.writeAll(entry.key_ptr.*);
                    try writer.writeAll("\": ");
                    try writeJsonIndented(entry.value_ptr.*, writer, depth + 1, indent);
                    if (i < count - 1) try writer.writeByte(',');
                    try writer.writeByte('\n');
                    i += 1;
                }
                try writer.writeByteNTimes(' ', depth * indent);
            }
            try writer.writeByte('}');
        },
        .macro => try writer.writeAll("null"),
        .joiner => try writer.writeAll("null"),
        .cycler => try writer.writeAll("null"),
        .loop_ctx => try writer.writeAll("null"),
    }
}

// ============================================================================
// Indentation
// ============================================================================

pub fn filterIndent(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    const input_str = switch (value) {
        .string => |str| str,
        else => return EvalError.TypeError,
    };

    // Default indent is 4 spaces
    var indent_width: i64 = 4;
    var first_line = false;

    if (args.len > 0) {
        const width_val = try e.evalExpr(args[0]);
        indent_width = switch (width_val) {
            .integer => |int_val| int_val,
            else => return EvalError.TypeError,
        };
    }
    if (args.len > 1) {
        const first_val = try e.evalExpr(args[1]);
        first_line = first_val.isTruthy();
    }

    const arena = e.ctx.arena.allocator();
    const indent_str = arena.alloc(u8, @intCast(indent_width)) catch return EvalError.OutOfMemory;
    @memset(indent_str, ' ');

    var result = std.ArrayListUnmanaged(u8){};
    var line_start = true;
    var is_first_line = true;

    for (input_str) |c| {
        if (line_start and c != '\n') {
            if (!is_first_line or first_line) {
                result.appendSlice(arena, indent_str) catch return EvalError.OutOfMemory;
            }
            line_start = false;
        }
        result.append(arena, c) catch return EvalError.OutOfMemory;
        if (c == '\n') {
            line_start = true;
            is_first_line = false;
        }
    }

    return .{ .string = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

// ============================================================================
// File Size
// ============================================================================

pub fn filterFilesizeformat(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    const size: f64 = switch (value) {
        .integer => |int_val| @floatFromInt(int_val),
        .float => |f| f,
        else => return EvalError.TypeError,
    };

    // Check for binary argument (default: false = decimal/SI units)
    var binary = false;
    if (args.len > 0) {
        const bin_val = try e.evalExpr(args[0]);
        binary = bin_val.isTruthy();
    }

    const arena = e.ctx.arena.allocator();
    var buffer = std.ArrayListUnmanaged(u8){};
    const writer = buffer.writer(arena);

    const base: f64 = if (binary) 1024.0 else 1000.0;
    const units_decimal = [_][]const u8{ "Bytes", "kB", "MB", "GB", "TB", "PB" };
    const units_binary = [_][]const u8{ "Bytes", "KiB", "MiB", "GiB", "TiB", "PiB" };
    const units = if (binary) &units_binary else &units_decimal;

    var scaled = size;
    var unit_idx: usize = 0;

    while (scaled >= base and unit_idx < units.len - 1) {
        scaled /= base;
        unit_idx += 1;
    }

    if (unit_idx == 0) {
        // Bytes - show as integer
        writer.print("{d} {s}", .{ @as(i64, @intFromFloat(size)), units[0] }) catch return EvalError.OutOfMemory;
    } else {
        // Show with one decimal place
        writer.print("{d:.1} {s}", .{ scaled, units[unit_idx] }) catch return EvalError.OutOfMemory;
    }

    return .{ .string = buffer.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

// ============================================================================
// XML
// ============================================================================

pub fn filterXmlattr(e: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    const map_value = switch (value) {
        .map => |map| map,
        else => return EvalError.TypeError,
    };

    const arena = e.ctx.arena.allocator();
    var result = std.ArrayListUnmanaged(u8){};

    var it = map_value.iterator();
    var first = true;
    while (it.next()) |entry| {
        if (!first) result.append(arena, ' ') catch return EvalError.OutOfMemory;
        result.appendSlice(arena, entry.key_ptr.*) catch return EvalError.OutOfMemory;
        result.appendSlice(arena, "=\"") catch return EvalError.OutOfMemory;
        const val_str = entry.value_ptr.asString(arena) catch return EvalError.OutOfMemory;
        result.appendSlice(arena, val_str) catch return EvalError.OutOfMemory;
        result.append(arena, '"') catch return EvalError.OutOfMemory;
        first = false;
    }

    return .{ .string = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

// ============================================================================
// JSON Schema Inference
// ============================================================================

/// Infer a JSON Schema from a data value.
///
/// Takes any TemplateInput value and generates a JSON Schema that describes it.
/// Useful for injecting schema definitions into prompts for structured output.
///
/// Example:
///   {{ example_data | json_schema }}
///   -> {"type": "object", "properties": {"name": {"type": "string"}}}
///
///   {{ example_data | json_schema | tojson(2) }}
///   -> Pretty-printed schema
pub fn filterJsonSchema(e: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    const arena = e.ctx.arena.allocator();
    return inferSchema(arena, value) catch return EvalError.OutOfMemory;
}

/// Infer JSON Schema from a TemplateInput value.
fn inferSchema(allocator: std.mem.Allocator, value: TemplateInput) !TemplateInput {
    switch (value) {
        .string => {
            var map = std.StringHashMapUnmanaged(TemplateInput){};
            try map.put(allocator, "type", .{ .string = "string" });
            return .{ .map = map };
        },
        .integer => {
            var map = std.StringHashMapUnmanaged(TemplateInput){};
            try map.put(allocator, "type", .{ .string = "integer" });
            return .{ .map = map };
        },
        .float => {
            var map = std.StringHashMapUnmanaged(TemplateInput){};
            try map.put(allocator, "type", .{ .string = "number" });
            return .{ .map = map };
        },
        .boolean => {
            var map = std.StringHashMapUnmanaged(TemplateInput){};
            try map.put(allocator, "type", .{ .string = "boolean" });
            return .{ .map = map };
        },
        .none => {
            var map = std.StringHashMapUnmanaged(TemplateInput){};
            try map.put(allocator, "type", .{ .string = "null" });
            return .{ .map = map };
        },
        .array => |arr| {
            var map = std.StringHashMapUnmanaged(TemplateInput){};
            try map.put(allocator, "type", .{ .string = "array" });

            // Infer items schema from first element (if any)
            if (arr.len > 0) {
                const items_schema = try inferSchema(allocator, arr[0]);
                try map.put(allocator, "items", items_schema);
            }

            return .{ .map = map };
        },
        .map => |m| {
            var schema_map = std.StringHashMapUnmanaged(TemplateInput){};
            try schema_map.put(allocator, "type", .{ .string = "object" });

            // Build properties schema
            var properties = std.StringHashMapUnmanaged(TemplateInput){};
            var it = m.iterator();
            while (it.next()) |entry| {
                const prop_schema = try inferSchema(allocator, entry.value_ptr.*);
                try properties.put(allocator, entry.key_ptr.*, prop_schema);
            }

            if (properties.count() > 0) {
                try schema_map.put(allocator, "properties", .{ .map = properties });
            }

            return .{ .map = schema_map };
        },
        .namespace => |ns| {
            // Treat namespace same as map
            var schema_map = std.StringHashMapUnmanaged(TemplateInput){};
            try schema_map.put(allocator, "type", .{ .string = "object" });

            var properties = std.StringHashMapUnmanaged(TemplateInput){};
            var it = ns.iterator();
            while (it.next()) |entry| {
                const prop_schema = try inferSchema(allocator, entry.value_ptr.*);
                try properties.put(allocator, entry.key_ptr.*, prop_schema);
            }

            if (properties.count() > 0) {
                try schema_map.put(allocator, "properties", .{ .map = properties });
            }

            return .{ .map = schema_map };
        },
        // For special types, return generic object
        .macro, .joiner, .cycler, .loop_ctx => {
            var map = std.StringHashMapUnmanaged(TemplateInput){};
            try map.put(allocator, "type", .{ .string = "object" });
            return .{ .map = map };
        },
    }
}

// ============================================================================
// Printf-style Formatting
// ============================================================================

pub fn filterFormat(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    const fmt = switch (value) {
        .string => |s| s,
        else => return EvalError.TypeError,
    };

    const arena = e.ctx.arena.allocator();
    var result = std.ArrayListUnmanaged(u8){};

    // Simple %s replacement
    var arg_idx: usize = 0;
    var i: usize = 0;
    while (i < fmt.len) {
        if (fmt[i] == '%' and i + 1 < fmt.len) {
            const spec = fmt[i + 1];
            if (spec == 's' or spec == 'd') {
                // Replace with argument
                if (arg_idx < args.len) {
                    const arg_val = try e.evalExpr(args[arg_idx]);
                    const arg_str = arg_val.asString(arena) catch return EvalError.OutOfMemory;
                    result.appendSlice(arena, arg_str) catch return EvalError.OutOfMemory;
                    arg_idx += 1;
                }
                i += 2;
                continue;
            } else if (spec == '%') {
                result.append(arena, '%') catch return EvalError.OutOfMemory;
                i += 2;
                continue;
            }
        }
        result.append(arena, fmt[i]) catch return EvalError.OutOfMemory;
        i += 1;
    }

    return .{ .string = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

// ============================================================================
// Unit Tests
// ============================================================================

const TemplateParser = @import("../eval.zig").TemplateParser;

test "filterTojson - simple object" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "test" };
    const result = try filterTojson(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("\"test\"", result.string);
}

test "filterFilesizeformat - bytes" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .integer = 512 };
    const result = try filterFilesizeformat(&eval_ctx, input, &.{});
    try std.testing.expectEqualStrings("512 Bytes", result.string);
}

test "filterFilesizeformat - kilobytes" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .integer = 1500 };
    const result = try filterFilesizeformat(&eval_ctx, input, &.{});
    try std.testing.expectEqualStrings("1.5 kB", result.string);
}

test "filterFilesizeformat - megabytes" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .integer = 5000000 };
    const result = try filterFilesizeformat(&eval_ctx, input, &.{});
    try std.testing.expectEqualStrings("5.0 MB", result.string);
}

test "filterTojson - integer" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .integer = 42 };
    const result = try filterTojson(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("42", result.string);
}

test "filterTojson - boolean" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .boolean = false };
    const result = try filterTojson(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("false", result.string);
}

test "filterTojson - none" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .none = {} };
    const result = try filterTojson(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("null", result.string);
}

test "filterIndent - default width" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "line1\nline2\nline3" };
    const result = try filterIndent(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("line1\n    line2\n    line3", result.string);
}

test "filterIndent - custom width" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "line1\nline2" };
    const width_expr = Expr{ .integer = 2 };
    const args = [_]*const Expr{&width_expr};
    const result = try filterIndent(&eval_ctx,input, &args);
    try std.testing.expectEqualStrings("line1\n  line2", result.string);
}

test "filterIndent - first line true" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "line1\nline2" };
    const width_expr = Expr{ .integer = 4 };
    const first_expr = Expr{ .boolean = true };
    const args = [_]*const Expr{ &width_expr, &first_expr };
    const result = try filterIndent(&eval_ctx,input, &args);
    try std.testing.expectEqualStrings("    line1\n    line2", result.string);
}

test "filterIndent - empty string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "" };
    const result = try filterIndent(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("", result.string);
}

test "filterIndent - single line" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "single line" };
    const result = try filterIndent(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("single line", result.string);
}

test "filterIndent - empty lines" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "line1\n\nline3" };
    const result = try filterIndent(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("line1\n\n    line3", result.string);
}

test "filterXmlattr - basic map" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    var map = std.StringHashMapUnmanaged(TemplateInput){};
    try map.put(ctx.arena.allocator(), "class", .{ .string = "main" });
    const input = TemplateInput{ .map = map };
    const result = try filterXmlattr(&eval_ctx, input, &.{});
    try std.testing.expectEqualStrings("class=\"main\"", result.string);
}

test "filterFormat - string substitution" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "Hello %s!" };
    const arg_expr = Expr{ .string = "World" };
    const args = [_]*const Expr{&arg_expr};
    const result = try filterFormat(&eval_ctx, input, &args);
    try std.testing.expectEqualStrings("Hello World!", result.string);
}

// ============================================================================
// JSON Schema Tests
// ============================================================================

test "filterJsonSchema - string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "hello" };
    const result = try filterJsonSchema(&eval_ctx, input, &.{});
    try std.testing.expectEqualStrings("string", result.map.get("type").?.string);
}

test "filterJsonSchema - integer" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .integer = 42 };
    const result = try filterJsonSchema(&eval_ctx, input, &.{});
    try std.testing.expectEqualStrings("integer", result.map.get("type").?.string);
}

test "filterJsonSchema - float" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .float = 3.14 };
    const result = try filterJsonSchema(&eval_ctx, input, &.{});
    try std.testing.expectEqualStrings("number", result.map.get("type").?.string);
}

test "filterJsonSchema - boolean" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .boolean = true };
    const result = try filterJsonSchema(&eval_ctx, input, &.{});
    try std.testing.expectEqualStrings("boolean", result.map.get("type").?.string);
}

test "filterJsonSchema - none/null" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .none = {} };
    const result = try filterJsonSchema(&eval_ctx, input, &.{});
    try std.testing.expectEqualStrings("null", result.map.get("type").?.string);
}

test "filterJsonSchema - array with items" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{
        .{ .string = "a" },
        .{ .string = "b" },
    };
    const input = TemplateInput{ .array = &items };
    const result = try filterJsonSchema(&eval_ctx, input, &.{});

    try std.testing.expectEqualStrings("array", result.map.get("type").?.string);

    // Check items schema
    const items_schema = result.map.get("items").?;
    try std.testing.expectEqualStrings("string", items_schema.map.get("type").?.string);
}

test "filterJsonSchema - empty array" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const items = [_]TemplateInput{};
    const input = TemplateInput{ .array = &items };
    const result = try filterJsonSchema(&eval_ctx, input, &.{});

    try std.testing.expectEqualStrings("array", result.map.get("type").?.string);
    // No items schema for empty array
    try std.testing.expect(result.map.get("items") == null);
}

test "filterJsonSchema - object with properties" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    var map = std.StringHashMapUnmanaged(TemplateInput){};
    try map.put(ctx.arena.allocator(), "name", .{ .string = "Alice" });
    try map.put(ctx.arena.allocator(), "age", .{ .integer = 30 });
    const input = TemplateInput{ .map = map };

    const result = try filterJsonSchema(&eval_ctx, input, &.{});
    try std.testing.expectEqualStrings("object", result.map.get("type").?.string);

    // Check properties
    const properties = result.map.get("properties").?.map;
    const name_schema = properties.get("name").?;
    try std.testing.expectEqualStrings("string", name_schema.map.get("type").?.string);
    const age_schema = properties.get("age").?;
    try std.testing.expectEqualStrings("integer", age_schema.map.get("type").?.string);
}

test "filterJsonSchema - nested object" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    // Create nested structure: {user: {name: "Alice"}}
    var inner_map = std.StringHashMapUnmanaged(TemplateInput){};
    try inner_map.put(ctx.arena.allocator(), "name", .{ .string = "Alice" });

    var outer_map = std.StringHashMapUnmanaged(TemplateInput){};
    try outer_map.put(ctx.arena.allocator(), "user", .{ .map = inner_map });

    const input = TemplateInput{ .map = outer_map };
    const result = try filterJsonSchema(&eval_ctx, input, &.{});

    try std.testing.expectEqualStrings("object", result.map.get("type").?.string);

    // Check nested schema
    const properties = result.map.get("properties").?.map;
    const user_schema = properties.get("user").?;
    try std.testing.expectEqualStrings("object", user_schema.map.get("type").?.string);

    const user_props = user_schema.map.get("properties").?.map;
    const name_schema = user_props.get("name").?;
    try std.testing.expectEqualStrings("string", name_schema.map.get("type").?.string);
}
