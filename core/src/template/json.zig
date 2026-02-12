//! JSON Integration for Template Engine
//!
//! Provides high-level JSON-based rendering API:
//! - Parse JSON variables and render templates in one call
//! - Convert between std.json and TemplateInput values
//! - Serialize template results to JSON
//!
//! This module is the primary interface for C API and Python bindings.

const std = @import("std");
const io = @import("../io/root.zig");
const root = @import("root.zig");
const input_mod = @import("input.zig");
const eval_mod = @import("eval.zig");
const validate_mod = @import("validate.zig");
const capi_error = @import("../capi/error.zig");

pub const TemplateInput = input_mod.TemplateInput;
pub const TemplateParser = eval_mod.TemplateParser;
pub const ValidationResult = validate_mod.ValidationResult;
pub const Error = root.Error;
pub const OutputSpan = root.OutputSpan;
pub const SpanSource = root.SpanSource;

// =============================================================================
// High-Level JSON Rendering API
// =============================================================================

/// Result of a render operation with detailed error information.
pub const RenderResult = struct {
    /// Rendered output (owned, caller must free)
    output: ?[]const u8 = null,
    /// Error code if render failed
    err: ?Error = null,
    /// Human-readable error message
    error_message: ?[]const u8 = null,
    /// Path to undefined variable (if applicable)
    undefined_path: ?[]const u8 = null,
    /// User message from raise_exception() (if applicable)
    raise_message: ?[]const u8 = null,
    /// Context for parse errors (e.g., "if", "for" block name)
    parse_error_context: ?[]const u8 = null,

    pub fn deinit(self: *RenderResult, alloc: std.mem.Allocator) void {
        if (self.output) |o| alloc.free(o);
        if (self.error_message) |m| alloc.free(m);
        if (self.undefined_path) |p| alloc.free(p);
        if (self.raise_message) |m| alloc.free(m);
        // parse_error_context is a static string literal, not owned
    }

    pub fn success(self: *const RenderResult) bool {
        return self.err == null;
    }
};

/// Render a template with JSON variables.
/// This is the primary entry point for external callers (C API, Python).
///
/// Parameters:
/// - allocator: Memory allocator
/// - template_source: Jinja2 template string
/// - json_vars: JSON object string with template variables
/// - strict: If true, undefined variables raise an error
///
/// Returns RenderResult with either output or error details.
pub fn renderFromJson(
    alloc: std.mem.Allocator,
    template_source: []const u8,
    json_vars: []const u8,
    strict: bool,
) RenderResult {
    // Parse JSON
    const parsed_json = io.json.parseValue(alloc, json_vars, .{
        .max_size_bytes = 10 * 1024 * 1024,
        .max_value_bytes = 10 * 1024 * 1024,
        .max_string_bytes = 1 * 1024 * 1024,
    }) catch |err| {
        return switch (err) {
            error.InputTooLarge, error.InputTooDeep, error.StringTooLong, error.InvalidJson => .{
                .err = null,
                .error_message = alloc.dupe(u8, "failed to parse JSON variables") catch null,
            },
            error.OutOfMemory => .{ .err = error.OutOfMemory },
        };
    };
    defer parsed_json.deinit();

    // Build context
    var ctx = TemplateParser.init(alloc);
    ctx.strict = strict;
    defer ctx.deinit();

    if (parsed_json.value == .object) {
        var iter = parsed_json.value.object.iterator();
        while (iter.next()) |entry| {
            const value = jsonToTemplateInput(alloc, entry.value_ptr.*);
            ctx.set(entry.key_ptr.*, value) catch continue;
        }
    }

    // Render
    const output = root.render(alloc, template_source, &ctx) catch |err| {
        // Set context for error message (used by capi)
        if (ctx.undefined_path) |p| {
            capi_error.setContext("'{s}' is undefined", .{p});
        } else if (ctx.raise_exception_message) |m| {
            capi_error.setContext("{s}", .{m});
        } else if (ctx.parse_error_context) |block| {
            capi_error.setContext("unclosed '{s}' block", .{block});
        }

        // Must copy strings before ctx is freed by defer
        return .{
            .err = err,
            .undefined_path = if (ctx.undefined_path) |p| alloc.dupe(u8, p) catch null else null,
            .raise_message = if (ctx.raise_exception_message) |m| alloc.dupe(u8, m) catch null else null,
            .parse_error_context = ctx.parse_error_context,
        };
    };

    return .{ .output = output };
}

/// Render a template with JSON variables and custom filters.
pub fn renderFromJsonWithFilters(
    alloc: std.mem.Allocator,
    template_source: []const u8,
    json_vars: []const u8,
    strict: bool,
    custom_filters: ?*const root.CustomFilterSet,
) RenderResult {
    // Parse JSON
    const parsed_json = io.json.parseValue(alloc, json_vars, .{
        .max_size_bytes = 10 * 1024 * 1024,
        .max_value_bytes = 10 * 1024 * 1024,
        .max_string_bytes = 1 * 1024 * 1024,
    }) catch |err| {
        return switch (err) {
            error.InputTooLarge, error.InputTooDeep, error.StringTooLong, error.InvalidJson => .{
                .err = null,
                .error_message = alloc.dupe(u8, "failed to parse JSON variables") catch null,
            },
            error.OutOfMemory => .{ .err = error.OutOfMemory },
        };
    };
    defer parsed_json.deinit();

    // Build context
    var ctx = TemplateParser.init(alloc);
    ctx.strict = strict;
    defer ctx.deinit();

    if (parsed_json.value == .object) {
        var iter = parsed_json.value.object.iterator();
        while (iter.next()) |entry| {
            const value = jsonToTemplateInput(alloc, entry.value_ptr.*);
            ctx.set(entry.key_ptr.*, value) catch continue;
        }
    }

    // Render with filters
    const output = root.renderWithFilters(alloc, template_source, &ctx, custom_filters) catch |err| {
        // Set context for error message (used by capi)
        if (ctx.undefined_path) |p| {
            capi_error.setContext("'{s}' is undefined", .{p});
        } else if (ctx.raise_exception_message) |m| {
            capi_error.setContext("{s}", .{m});
        } else if (ctx.parse_error_context) |block| {
            capi_error.setContext("unclosed '{s}' block", .{block});
        }

        // Must copy strings before ctx is freed by defer
        return .{
            .err = err,
            .undefined_path = if (ctx.undefined_path) |p| alloc.dupe(u8, p) catch null else null,
            .raise_message = if (ctx.raise_exception_message) |m| alloc.dupe(u8, m) catch null else null,
            .parse_error_context = ctx.parse_error_context,
        };
    };

    return .{ .output = output };
}

/// Result of debug render with span tracking.
pub const RenderDebugResult = struct {
    /// Rendered output (owned)
    output: ?[]const u8 = null,
    /// Output spans showing variable sources
    spans: ?[]const root.OutputSpan = null,
    /// Error if render failed
    err: ?Error = null,
    /// Path to undefined variable
    undefined_path: ?[]const u8 = null,
    /// User message from raise_exception()
    raise_message: ?[]const u8 = null,
    /// Context for parse errors (e.g., "if", "for" block name)
    parse_error_context: ?[]const u8 = null,

    pub fn deinit(self: *RenderDebugResult, alloc: std.mem.Allocator) void {
        if (self.output) |o| alloc.free(o);
        if (self.spans) |s| alloc.free(s);
        if (self.undefined_path) |p| alloc.free(p);
        if (self.raise_message) |m| alloc.free(m);
        // parse_error_context is a static string literal, not owned
    }

    pub fn success(self: *const RenderDebugResult) bool {
        return self.err == null;
    }
};

/// Render with span tracking for debug visualization.
pub fn renderFromJsonDebug(
    alloc: std.mem.Allocator,
    template_source: []const u8,
    json_vars: []const u8,
    strict: bool,
) RenderDebugResult {
    // Parse JSON
    const parsed_json = io.json.parseValue(alloc, json_vars, .{
        .max_size_bytes = 10 * 1024 * 1024,
        .max_value_bytes = 10 * 1024 * 1024,
        .max_string_bytes = 1 * 1024 * 1024,
    }) catch |err| {
        return switch (err) {
            error.InputTooLarge, error.InputTooDeep, error.StringTooLong, error.InvalidJson => .{ .err = null },
            error.OutOfMemory => .{ .err = error.OutOfMemory },
        };
    };
    defer parsed_json.deinit();

    // Build context
    var ctx = TemplateParser.init(alloc);
    ctx.strict = strict;
    defer ctx.deinit();

    if (parsed_json.value == .object) {
        var iter = parsed_json.value.object.iterator();
        while (iter.next()) |entry| {
            const value = jsonToTemplateInput(alloc, entry.value_ptr.*);
            ctx.set(entry.key_ptr.*, value) catch continue;
        }
    }

    // Render with spans
    const result = root.renderWithSpans(alloc, template_source, &ctx) catch |err| {
        // Set context for error message (used by capi)
        if (ctx.undefined_path) |p| {
            capi_error.setContext("'{s}' is undefined", .{p});
        } else if (ctx.raise_exception_message) |m| {
            capi_error.setContext("{s}", .{m});
        } else if (ctx.parse_error_context) |block| {
            capi_error.setContext("unclosed '{s}' block", .{block});
        }

        // Must copy strings before ctx is freed by defer
        return .{
            .err = err,
            .undefined_path = if (ctx.undefined_path) |p| alloc.dupe(u8, p) catch null else null,
            .raise_message = if (ctx.raise_exception_message) |m| alloc.dupe(u8, m) catch null else null,
            .parse_error_context = ctx.parse_error_context,
        };
    };

    // Must copy spans' variable paths before ctx is freed (they're allocated from ctx.arena)
    const copied_spans = copySpans(alloc, result.spans) catch {
        alloc.free(result.output);
        alloc.free(result.spans);
        return .{ .err = error.OutOfMemory };
    };
    alloc.free(result.spans);

    return .{
        .output = result.output,
        .spans = copied_spans,
    };
}

// =============================================================================
// Internal Helpers
// =============================================================================

/// Copy spans with owned variable paths (original paths may be arena-allocated).
fn copySpans(alloc: std.mem.Allocator, spans: []const OutputSpan) ![]const OutputSpan {
    const copied = try alloc.alloc(OutputSpan, spans.len);
    errdefer alloc.free(copied);

    for (spans, 0..) |span, i| {
        copied[i] = .{
            .start = span.start,
            .end = span.end,
            .source = switch (span.source) {
                .static_text => .static_text,
                .expression => .expression,
                .variable => |path| .{ .variable = try alloc.dupe(u8, path) },
            },
        };
    }
    return copied;
}

// =============================================================================
// JSON to TemplateInput Conversion
// =============================================================================

/// Convert std.json.Value to TemplateInput recursively.
/// Caller is responsible for freeing any allocated arrays/maps.
pub fn jsonToTemplateInput(allocator: std.mem.Allocator, json_value: std.json.Value) TemplateInput {
    return switch (json_value) {
        .null => .none,
        .bool => |bool_value| .{ .boolean = bool_value },
        .integer => |int_value| .{ .integer = int_value },
        .float => |float_value| .{ .float = float_value },
        .string => |string_value| .{ .string = string_value },
        .array => |array_value| {
            const array_items = allocator.alloc(TemplateInput, array_value.items.len) catch return .none;
            for (array_value.items, 0..) |item, item_idx| {
                array_items[item_idx] = jsonToTemplateInput(allocator, item);
            }
            return .{ .array = array_items };
        },
        .object => |object_value| {
            var map_values = std.StringHashMapUnmanaged(TemplateInput){};
            var object_iter = object_value.iterator();
            while (object_iter.next()) |entry| {
                map_values.put(allocator, entry.key_ptr.*, jsonToTemplateInput(allocator, entry.value_ptr.*)) catch continue;
            }
            return .{ .map = map_values };
        },
        .number_string => |number_text| {
            // Try to parse as integer first, then float
            if (std.fmt.parseInt(i64, number_text, 10)) |int_value| {
                return .{ .integer = int_value };
            } else |_| {
                if (std.fmt.parseFloat(f64, number_text)) |float_value| {
                    return .{ .float = float_value };
                } else |_| {
                    return .{ .string = number_text };
                }
            }
        },
    };
}

// =============================================================================
// ValidationResult JSON Serialization
// =============================================================================

/// Serialize ValidationResult to JSON string.
/// Caller must free the returned string.
pub fn validationResultToJson(allocator: std.mem.Allocator, result: ValidationResult) ![]u8 {
    var json_buffer = std.ArrayListUnmanaged(u8){};
    errdefer json_buffer.deinit(allocator);
    const writer = json_buffer.writer(allocator);

    try writer.writeAll("{\"valid\":");
    try writer.writeAll(if (result.valid) "true" else "false");

    try writer.writeAll(",\"required\":[");
    for (result.required, 0..) |name, i| {
        if (i > 0) try writer.writeByte(',');
        try writer.writeByte('"');
        try writer.writeAll(name);
        try writer.writeByte('"');
    }
    try writer.writeAll("]");

    try writer.writeAll(",\"optional\":[");
    for (result.optional, 0..) |name, i| {
        if (i > 0) try writer.writeByte(',');
        try writer.writeByte('"');
        try writer.writeAll(name);
        try writer.writeByte('"');
    }
    try writer.writeAll("]");

    try writer.writeAll(",\"extra\":[");
    for (result.extra, 0..) |name, i| {
        if (i > 0) try writer.writeByte(',');
        try writer.writeByte('"');
        try writer.writeAll(name);
        try writer.writeByte('"');
    }
    try writer.writeAll("]}");

    return json_buffer.toOwnedSlice(allocator);
}

// =============================================================================
// Tests
// =============================================================================

test "jsonToTemplateInput: null" {
    const allocator = std.testing.allocator;
    const result = jsonToTemplateInput(allocator, .null);
    try std.testing.expectEqual(TemplateInput.none, result);
}

test "jsonToTemplateInput: boolean" {
    const allocator = std.testing.allocator;

    const true_result = jsonToTemplateInput(allocator, .{ .bool = true });
    try std.testing.expectEqual(true, true_result.boolean);

    const false_result = jsonToTemplateInput(allocator, .{ .bool = false });
    try std.testing.expectEqual(false, false_result.boolean);
}

test "jsonToTemplateInput: integer" {
    const allocator = std.testing.allocator;
    const result = jsonToTemplateInput(allocator, .{ .integer = 42 });
    try std.testing.expectEqual(@as(i64, 42), result.integer);
}

test "jsonToTemplateInput: negative integer" {
    const allocator = std.testing.allocator;
    const result = jsonToTemplateInput(allocator, .{ .integer = -17 });
    try std.testing.expectEqual(@as(i64, -17), result.integer);
}

test "jsonToTemplateInput: float" {
    const allocator = std.testing.allocator;
    const result = jsonToTemplateInput(allocator, .{ .float = 3.14 });
    try std.testing.expectEqual(@as(f64, 3.14), result.float);
}

test "jsonToTemplateInput: string" {
    const allocator = std.testing.allocator;
    const result = jsonToTemplateInput(allocator, .{ .string = "hello" });
    try std.testing.expectEqualStrings("hello", result.string);
}

test "jsonToTemplateInput: array" {
    const allocator = std.testing.allocator;

    var json_array = std.json.Array.init(allocator);
    defer json_array.deinit();
    try json_array.append(.{ .integer = 1 });
    try json_array.append(.{ .integer = 2 });
    try json_array.append(.{ .integer = 3 });

    const result = jsonToTemplateInput(allocator, .{ .array = json_array });
    defer allocator.free(result.array);

    try std.testing.expectEqual(@as(usize, 3), result.array.len);
    try std.testing.expectEqual(@as(i64, 1), result.array[0].integer);
    try std.testing.expectEqual(@as(i64, 2), result.array[1].integer);
    try std.testing.expectEqual(@as(i64, 3), result.array[2].integer);
}

test "jsonToTemplateInput: object" {
    const allocator = std.testing.allocator;

    var json_object = std.json.ObjectMap.init(allocator);
    defer json_object.deinit();
    try json_object.put("name", .{ .string = "Alice" });
    try json_object.put("age", .{ .integer = 30 });

    const result = jsonToTemplateInput(allocator, .{ .object = json_object });
    defer {
        var map = result.map;
        map.deinit(allocator);
    }

    try std.testing.expectEqualStrings("Alice", result.map.get("name").?.string);
    try std.testing.expectEqual(@as(i64, 30), result.map.get("age").?.integer);
}

test "jsonToTemplateInput: number_string as integer" {
    const allocator = std.testing.allocator;
    const result = jsonToTemplateInput(allocator, .{ .number_string = "12345" });
    try std.testing.expectEqual(@as(i64, 12345), result.integer);
}

test "jsonToTemplateInput: number_string as float" {
    const allocator = std.testing.allocator;
    const result = jsonToTemplateInput(allocator, .{ .number_string = "3.14159" });
    try std.testing.expectEqual(@as(f64, 3.14159), result.float);
}

test "jsonToTemplateInput: number_string unparseable" {
    const allocator = std.testing.allocator;
    const result = jsonToTemplateInput(allocator, .{ .number_string = "not-a-number" });
    try std.testing.expectEqualStrings("not-a-number", result.string);
}

test "validationResultToJson: valid result" {
    const allocator = std.testing.allocator;

    const required = [_][]const u8{ "messages", "add_generation_prompt" };
    const optional = [_][]const u8{"tools"};
    const extra = [_][]const u8{};

    const result = ValidationResult{
        .valid = true,
        .required = &required,
        .optional = &optional,
        .extra = &extra,
    };

    const json = try validationResultToJson(allocator, result);
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"valid\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"messages\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"add_generation_prompt\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"tools\"") != null);
}

test "validationResultToJson: invalid result with missing vars" {
    const allocator = std.testing.allocator;

    const required = [_][]const u8{"messages"};
    const optional = [_][]const u8{};
    const extra = [_][]const u8{ "unused1", "unused2" };

    const result = ValidationResult{
        .valid = false,
        .required = &required,
        .optional = &optional,
        .extra = &extra,
    };

    const json = try validationResultToJson(allocator, result);
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"valid\":false") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"unused1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"unused2\"") != null);
}

test "validationResultToJson: empty arrays" {
    const allocator = std.testing.allocator;

    const empty = [_][]const u8{};

    const result = ValidationResult{
        .valid = true,
        .required = &empty,
        .optional = &empty,
        .extra = &empty,
    };

    const json = try validationResultToJson(allocator, result);
    defer allocator.free(json);

    try std.testing.expectEqualStrings("{\"valid\":true,\"required\":[],\"optional\":[],\"extra\":[]}", json);
}
