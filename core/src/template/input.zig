//! Template Input Types
//!
//! Core value types for template evaluation:
//!
//! - `TemplateInput`: Dynamic values (strings, numbers, lists, objects)
//! - `LoopContext`: Loop state during iteration
//! - `MacroDef`: Macro definitions
//! - `JoinerState`: State for joiner() function
//! - `CyclerState`: State for cycler() function

const std = @import("std");
const ast = @import("ast.zig");

/// Macro definition stored in context
pub const MacroDef = struct {
    name: []const u8,
    params: []const ast.Node.MacroParam,
    body: []const *const ast.Node,
};

/// Joiner state - returns empty on first call, separator on subsequent calls
pub const JoinerState = struct {
    separator: []const u8,
    called: bool = false,
};

/// Cycler state - cycles through a list of values
pub const CyclerState = struct {
    items: []const TemplateInput,
    index: usize = 0,
};

/// Loop context available during for loops
pub const LoopContext = struct {
    index0: usize,
    index: usize,
    first: bool,
    last: bool,
    length: usize,
    revindex: usize, // remaining from end, 1-based
    revindex0: usize, // remaining from end, 0-based
    previtem: ?TemplateInput = null,
    nextitem: ?TemplateInput = null,
    depth: usize = 1, // recursion depth (1-based)
    depth0: usize = 0, // recursion depth (0-based)
    // For recursive loops
    recursive_body: ?[]const *const ast.Node = null,
    recursive_target: ?[]const u8 = null,
    recursive_target2: ?[]const u8 = null,
};

/// A dynamic value that can be passed to templates.
///
/// Templates need to work with different data types: strings, numbers, lists,
/// and objects. Since Zig is statically typed, `TemplateInput` provides a
/// tagged union that can hold any of these types.
///
/// ## Common Usage
///
/// ```zig
/// // Simple values
/// parser.set("name", .{ .string = "Alice" });
/// parser.set("count", .{ .integer = 42 });
/// parser.set("enabled", .{ .boolean = true });
///
/// // Lists (for {% for item in items %})
/// const items = [_]TemplateInput{ .{ .string = "a" }, .{ .string = "b" } };
/// parser.set("items", .{ .array = &items });
///
/// // Objects (for {{ message.role }})
/// var msg = std.StringHashMapUnmanaged(TemplateInput){};
/// try msg.put(allocator, "role", .{ .string = "user" });
/// parser.set("message", .{ .map = msg });
/// ```
pub const TemplateInput = union(enum) {
    string: []const u8,
    integer: i64,
    float: f64,
    boolean: bool,
    array: []const TemplateInput,
    map: std.StringHashMapUnmanaged(TemplateInput),
    none,
    /// Namespace object - mutable container for loop variables
    namespace: *std.StringHashMapUnmanaged(TemplateInput),
    /// Macro definition
    macro: MacroDef,
    /// Joiner callable
    joiner: *JoinerState,
    /// Cycler callable
    cycler: *CyclerState,
    /// Loop context (for loop.cycle() support)
    loop_ctx: *LoopContext,

    pub fn isTruthy(self: TemplateInput) bool {
        return switch (self) {
            .string => |s| s.len > 0,
            .integer => |int_val| int_val != 0,
            .float => |f| f != 0.0,
            .boolean => |b| b,
            .array => |a| a.len > 0,
            .map => |m| m.count() > 0,
            .namespace => true,
            .macro => true,
            .joiner => true,
            .cycler => true,
            .loop_ctx => true,
            .none => false,
        };
    }


    pub fn asString(self: TemplateInput, allocator: std.mem.Allocator) ![]const u8 {
        return switch (self) {
            .string => |s| s,
            .integer => |int_val| try std.fmt.allocPrint(allocator, "{d}", .{int_val}),
            .float => |f| blk: {
                // Always show decimal point for floats (Python/Jinja behavior)
                const str = try std.fmt.allocPrint(allocator, "{d}", .{f});
                // Check if it has a decimal point
                for (str) |c| {
                    if (c == '.') break :blk str;
                }
                // No decimal point - append .0
                const with_decimal = try std.fmt.allocPrint(allocator, "{s}.0", .{str});
                allocator.free(str);
                break :blk with_decimal;
            },
            .boolean => |b| if (b) "True" else "False",
            .none => "",
            .array => try self.toJson(allocator),
            .map => try self.toJson(allocator),
            .namespace => "[namespace]",
            .macro => |m| try std.fmt.allocPrint(allocator, "<macro {s}>", .{m.name}),
            .joiner => "[joiner]",
            .cycler => "[cycler]",
            .loop_ctx => "[loop]",
        };
    }

    pub fn toJson(self: TemplateInput, allocator: std.mem.Allocator) ![]const u8 {
        var json_buffer = std.ArrayListUnmanaged(u8){};
        const writer = json_buffer.writer(allocator);
        try self.writeJson(writer);
        return json_buffer.toOwnedSlice(allocator);
    }

    pub fn writeJson(self: TemplateInput, writer: anytype) !void {
        switch (self) {
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
                for (arr, 0..) |item, idx| {
                    if (idx > 0) try writer.writeAll(", ");
                    try item.writeJson(writer);
                }
                try writer.writeByte(']');
            },
            .map => |m| {
                try writer.writeByte('{');
                var first = true;
                var it = m.iterator();
                while (it.next()) |entry| {
                    if (!first) try writer.writeAll(", ");
                    first = false;
                    try writer.print("\"{s}\": ", .{entry.key_ptr.*});
                    try entry.value_ptr.writeJson(writer);
                }
                try writer.writeByte('}');
            },
            .namespace => try writer.writeAll("{}"),
            .macro => try writer.writeAll("null"),
            .joiner => try writer.writeAll("null"),
            .cycler => try writer.writeAll("null"),
            .loop_ctx => try writer.writeAll("null"),
        }
    }

    pub fn eql(self: TemplateInput, other: TemplateInput) bool {
        if (@as(std.meta.Tag(TemplateInput), self) != @as(std.meta.Tag(TemplateInput), other)) {
            // Type coercion for numeric comparison
            const self_num = self.asNumber();
            const other_num = other.asNumber();
            if (self_num != null and other_num != null) {
                return self_num.? == other_num.?;
            }
            return false;
        }

        return switch (self) {
            .string => |s| std.mem.eql(u8, s, other.string),
            .integer => |int_val| int_val == other.integer,
            .float => |f| f == other.float,
            .boolean => |b| b == other.boolean,
            .none => true,
            else => false,
        };
    }

    pub fn asNumber(self: TemplateInput) ?f64 {
        return switch (self) {
            .integer => |int_val| @floatFromInt(int_val),
            .float => |f| f,
            else => null,
        };
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "TemplateInput.isTruthy - string" {
    const empty_str = TemplateInput{ .string = "" };
    try std.testing.expect(!empty_str.isTruthy());

    const non_empty_str = TemplateInput{ .string = "hello" };
    try std.testing.expect(non_empty_str.isTruthy());
}

test "TemplateInput.isTruthy - integer" {
    const zero = TemplateInput{ .integer = 0 };
    try std.testing.expect(!zero.isTruthy());

    const positive = TemplateInput{ .integer = 42 };
    try std.testing.expect(positive.isTruthy());

    const negative = TemplateInput{ .integer = -1 };
    try std.testing.expect(negative.isTruthy());
}

test "TemplateInput.isTruthy - float" {
    const zero = TemplateInput{ .float = 0.0 };
    try std.testing.expect(!zero.isTruthy());

    const positive = TemplateInput{ .float = 3.14 };
    try std.testing.expect(positive.isTruthy());

    const negative = TemplateInput{ .float = -2.5 };
    try std.testing.expect(negative.isTruthy());
}

test "TemplateInput.isTruthy - boolean" {
    const true_val = TemplateInput{ .boolean = true };
    try std.testing.expect(true_val.isTruthy());

    const false_val = TemplateInput{ .boolean = false };
    try std.testing.expect(!false_val.isTruthy());
}

test "TemplateInput.isTruthy - array" {
    const empty_array = TemplateInput{ .array = &.{} };
    try std.testing.expect(!empty_array.isTruthy());

    const items = [_]TemplateInput{.{ .integer = 1 }};
    const non_empty_array = TemplateInput{ .array = &items };
    try std.testing.expect(non_empty_array.isTruthy());
}

test "TemplateInput.isTruthy - map" {
    const empty_map = TemplateInput{ .map = .{} };
    try std.testing.expect(!empty_map.isTruthy());

    var map = std.StringHashMapUnmanaged(TemplateInput){};
    const allocator = std.testing.allocator;
    try map.put(allocator, "key", .{ .string = "value" });
    defer map.deinit(allocator);

    const non_empty_map = TemplateInput{ .map = map };
    try std.testing.expect(non_empty_map.isTruthy());
}

test "TemplateInput.isTruthy - none" {
    const none: TemplateInput = .none;
    try std.testing.expect(!none.isTruthy());
}

test "TemplateInput.isTruthy - namespace" {
    var ns = std.StringHashMapUnmanaged(TemplateInput){};
    const namespace = TemplateInput{ .namespace = &ns };
    try std.testing.expect(namespace.isTruthy());
}

test "TemplateInput.isTruthy - macro" {
    const macro = TemplateInput{ .macro = .{
        .name = "test_macro",
        .params = &.{},
        .body = &.{},
    } };
    try std.testing.expect(macro.isTruthy());
}

test "TemplateInput.isTruthy - joiner" {
    var joiner_state = JoinerState{ .separator = ", " };
    const joiner = TemplateInput{ .joiner = &joiner_state };
    try std.testing.expect(joiner.isTruthy());
}

test "TemplateInput.isTruthy - cycler" {
    var cycler_state = CyclerState{ .items = &.{} };
    const cycler = TemplateInput{ .cycler = &cycler_state };
    try std.testing.expect(cycler.isTruthy());
}

test "TemplateInput.isTruthy - loop_ctx" {
    var loop_ctx = LoopContext{
        .index0 = 0,
        .index = 1,
        .first = true,
        .last = false,
        .length = 5,
        .revindex = 5,
        .revindex0 = 4,
    };
    const loop_input = TemplateInput{ .loop_ctx = &loop_ctx };
    try std.testing.expect(loop_input.isTruthy());
}

test "TemplateInput.asString - string" {
    const allocator = std.testing.allocator;
    const input = TemplateInput{ .string = "hello" };
    const str = try input.asString(allocator);
    try std.testing.expectEqualStrings("hello", str);
}

test "TemplateInput.asString - integer" {
    const allocator = std.testing.allocator;
    const input = TemplateInput{ .integer = 42 };
    const str = try input.asString(allocator);
    defer allocator.free(str);
    try std.testing.expectEqualStrings("42", str);
}

test "TemplateInput.asString - negative integer" {
    const allocator = std.testing.allocator;
    const input = TemplateInput{ .integer = -17 };
    const str = try input.asString(allocator);
    defer allocator.free(str);
    try std.testing.expectEqualStrings("-17", str);
}

test "TemplateInput.asString - float" {
    const allocator = std.testing.allocator;
    const input = TemplateInput{ .float = 3.14 };
    const str = try input.asString(allocator);
    defer allocator.free(str);
    try std.testing.expectEqualStrings("3.14", str);
}

test "TemplateInput.asString - float whole number adds .0" {
    const allocator = std.testing.allocator;
    const input = TemplateInput{ .float = 5.0 };
    const str = try input.asString(allocator);
    defer allocator.free(str);
    // Should have decimal point
    try std.testing.expect(std.mem.indexOf(u8, str, ".") != null);
}

test "TemplateInput.asString - boolean true" {
    const allocator = std.testing.allocator;
    const input = TemplateInput{ .boolean = true };
    const str = try input.asString(allocator);
    try std.testing.expectEqualStrings("True", str);
}

test "TemplateInput.asString - boolean false" {
    const allocator = std.testing.allocator;
    const input = TemplateInput{ .boolean = false };
    const str = try input.asString(allocator);
    try std.testing.expectEqualStrings("False", str);
}

test "TemplateInput.asString - none" {
    const allocator = std.testing.allocator;
    const input: TemplateInput = .none;
    const str = try input.asString(allocator);
    try std.testing.expectEqualStrings("", str);
}

test "TemplateInput.asString - namespace" {
    const allocator = std.testing.allocator;
    var ns = std.StringHashMapUnmanaged(TemplateInput){};
    const input = TemplateInput{ .namespace = &ns };
    const str = try input.asString(allocator);
    try std.testing.expectEqualStrings("[namespace]", str);
}

test "TemplateInput.asString - macro" {
    const allocator = std.testing.allocator;
    const input = TemplateInput{ .macro = .{
        .name = "test_macro",
        .params = &.{},
        .body = &.{},
    } };
    const str = try input.asString(allocator);
    defer allocator.free(str);
    try std.testing.expectEqualStrings("<macro test_macro>", str);
}

test "TemplateInput.asString - joiner" {
    const allocator = std.testing.allocator;
    var joiner_state = JoinerState{ .separator = ", " };
    const input = TemplateInput{ .joiner = &joiner_state };
    const str = try input.asString(allocator);
    try std.testing.expectEqualStrings("[joiner]", str);
}

test "TemplateInput.asString - cycler" {
    const allocator = std.testing.allocator;
    var cycler_state = CyclerState{ .items = &.{} };
    const input = TemplateInput{ .cycler = &cycler_state };
    const str = try input.asString(allocator);
    try std.testing.expectEqualStrings("[cycler]", str);
}

test "TemplateInput.asString - loop_ctx" {
    const allocator = std.testing.allocator;
    var loop_ctx = LoopContext{
        .index0 = 0,
        .index = 1,
        .first = true,
        .last = false,
        .length = 5,
        .revindex = 5,
        .revindex0 = 4,
    };
    const input = TemplateInput{ .loop_ctx = &loop_ctx };
    const str = try input.asString(allocator);
    try std.testing.expectEqualStrings("[loop]", str);
}

test "TemplateInput.toJson - string" {
    const allocator = std.testing.allocator;
    const input = TemplateInput{ .string = "hello" };
    const json = try input.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expectEqualStrings("\"hello\"", json);
}

test "TemplateInput.toJson - string with escapes" {
    const allocator = std.testing.allocator;
    const input = TemplateInput{ .string = "hello\nworld\"test\\" };
    const json = try input.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expectEqualStrings("\"hello\\nworld\\\"test\\\\\"", json);
}

test "TemplateInput.toJson - integer" {
    const allocator = std.testing.allocator;
    const input = TemplateInput{ .integer = 42 };
    const json = try input.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expectEqualStrings("42", json);
}

test "TemplateInput.toJson - float" {
    const allocator = std.testing.allocator;
    const input = TemplateInput{ .float = 3.14 };
    const json = try input.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expectEqualStrings("3.14", json);
}

test "TemplateInput.toJson - boolean" {
    const allocator = std.testing.allocator;
    const true_input = TemplateInput{ .boolean = true };
    const true_json = try true_input.toJson(allocator);
    defer allocator.free(true_json);
    try std.testing.expectEqualStrings("true", true_json);

    const false_input = TemplateInput{ .boolean = false };
    const false_json = try false_input.toJson(allocator);
    defer allocator.free(false_json);
    try std.testing.expectEqualStrings("false", false_json);
}

test "TemplateInput.toJson - none" {
    const allocator = std.testing.allocator;
    const input: TemplateInput = .none;
    const json = try input.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expectEqualStrings("null", json);
}

test "TemplateInput.toJson - array" {
    const allocator = std.testing.allocator;
    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .string = "hello" },
        .{ .boolean = true },
    };
    const input = TemplateInput{ .array = &items };
    const json = try input.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expectEqualStrings("[1, \"hello\", true]", json);
}

test "TemplateInput.toJson - empty array" {
    const allocator = std.testing.allocator;
    const input = TemplateInput{ .array = &.{} };
    const json = try input.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expectEqualStrings("[]", json);
}

test "TemplateInput.toJson - map" {
    const allocator = std.testing.allocator;
    var map = std.StringHashMapUnmanaged(TemplateInput){};
    defer map.deinit(allocator);
    try map.put(allocator, "name", .{ .string = "Alice" });
    try map.put(allocator, "age", .{ .integer = 30 });

    const input = TemplateInput{ .map = map };
    const json = try input.toJson(allocator);
    defer allocator.free(json);
    // Map order is not guaranteed, just check it's valid JSON with braces
    try std.testing.expect(json[0] == '{');
    try std.testing.expect(json[json.len - 1] == '}');
    try std.testing.expect(std.mem.indexOf(u8, json, "\"name\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"age\"") != null);
}

test "TemplateInput.toJson - namespace" {
    const allocator = std.testing.allocator;
    var ns = std.StringHashMapUnmanaged(TemplateInput){};
    const input = TemplateInput{ .namespace = &ns };
    const json = try input.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expectEqualStrings("{}", json);
}

test "TemplateInput.toJson - macro" {
    const allocator = std.testing.allocator;
    const input = TemplateInput{ .macro = .{
        .name = "test",
        .params = &.{},
        .body = &.{},
    } };
    const json = try input.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expectEqualStrings("null", json);
}

test "TemplateInput.toJson - joiner" {
    const allocator = std.testing.allocator;
    var joiner_state = JoinerState{ .separator = ", " };
    const input = TemplateInput{ .joiner = &joiner_state };
    const json = try input.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expectEqualStrings("null", json);
}

test "TemplateInput.toJson - cycler" {
    const allocator = std.testing.allocator;
    var cycler_state = CyclerState{ .items = &.{} };
    const input = TemplateInput{ .cycler = &cycler_state };
    const json = try input.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expectEqualStrings("null", json);
}

test "TemplateInput.toJson - loop_ctx" {
    const allocator = std.testing.allocator;
    var loop_ctx = LoopContext{
        .index0 = 0,
        .index = 1,
        .first = true,
        .last = false,
        .length = 5,
        .revindex = 5,
        .revindex0 = 4,
    };
    const input = TemplateInput{ .loop_ctx = &loop_ctx };
    const json = try input.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expectEqualStrings("null", json);
}

test "TemplateInput.eql - same type comparisons" {
    // String equality
    const str1 = TemplateInput{ .string = "hello" };
    const str2 = TemplateInput{ .string = "hello" };
    const str3 = TemplateInput{ .string = "world" };
    try std.testing.expect(str1.eql(str2));
    try std.testing.expect(!str1.eql(str3));

    // Integer equality
    const int1 = TemplateInput{ .integer = 42 };
    const int2 = TemplateInput{ .integer = 42 };
    const int3 = TemplateInput{ .integer = 17 };
    try std.testing.expect(int1.eql(int2));
    try std.testing.expect(!int1.eql(int3));

    // Float equality
    const float1 = TemplateInput{ .float = 3.14 };
    const float2 = TemplateInput{ .float = 3.14 };
    const float3 = TemplateInput{ .float = 2.71 };
    try std.testing.expect(float1.eql(float2));
    try std.testing.expect(!float1.eql(float3));

    // Boolean equality
    const bool_true1 = TemplateInput{ .boolean = true };
    const bool_true2 = TemplateInput{ .boolean = true };
    const bool_false = TemplateInput{ .boolean = false };
    try std.testing.expect(bool_true1.eql(bool_true2));
    try std.testing.expect(!bool_true1.eql(bool_false));

    // None equality
    const none1: TemplateInput = .none;
    const none2: TemplateInput = .none;
    try std.testing.expect(none1.eql(none2));
}

test "TemplateInput.eql - numeric coercion" {
    // Integer to float coercion
    const int_val = TemplateInput{ .integer = 5 };
    const float_val = TemplateInput{ .float = 5.0 };
    try std.testing.expect(int_val.eql(float_val));
    try std.testing.expect(float_val.eql(int_val));

    // Non-matching numeric values
    const int_val2 = TemplateInput{ .integer = 5 };
    const float_val2 = TemplateInput{ .float = 5.5 };
    try std.testing.expect(!int_val2.eql(float_val2));
}

test "TemplateInput.eql - different non-numeric types" {
    const str_val = TemplateInput{ .string = "42" };
    const int_val = TemplateInput{ .integer = 42 };
    try std.testing.expect(!str_val.eql(int_val));

    const bool_val = TemplateInput{ .boolean = true };
    const int_one = TemplateInput{ .integer = 1 };
    try std.testing.expect(!bool_val.eql(int_one));
}

test "TemplateInput.asNumber - integer" {
    const input = TemplateInput{ .integer = 42 };
    const num = input.asNumber();
    try std.testing.expect(num != null);
    try std.testing.expectEqual(@as(f64, 42.0), num.?);
}

test "TemplateInput.asNumber - negative integer" {
    const input = TemplateInput{ .integer = -17 };
    const num = input.asNumber();
    try std.testing.expect(num != null);
    try std.testing.expectEqual(@as(f64, -17.0), num.?);
}

test "TemplateInput.asNumber - float" {
    const input = TemplateInput{ .float = 3.14 };
    const num = input.asNumber();
    try std.testing.expect(num != null);
    try std.testing.expectEqual(@as(f64, 3.14), num.?);
}

test "TemplateInput.asNumber - non-numeric types" {
    const string_input = TemplateInput{ .string = "42" };
    try std.testing.expect(string_input.asNumber() == null);

    const bool_input = TemplateInput{ .boolean = true };
    try std.testing.expect(bool_input.asNumber() == null);

    const none_input: TemplateInput = .none;
    try std.testing.expect(none_input.asNumber() == null);

    const array_input = TemplateInput{ .array = &.{} };
    try std.testing.expect(array_input.asNumber() == null);
}

// ============================================================================
// Tests for TemplateInput.writeJson
// ============================================================================

test "TemplateInput.writeJson - string" {
    const allocator = std.testing.allocator;
    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(allocator);

    const input = TemplateInput{ .string = "hello" };
    try input.writeJson(buffer.writer(allocator));

    try std.testing.expectEqualStrings("\"hello\"", buffer.items);
}

test "TemplateInput.writeJson - string with special characters" {
    const allocator = std.testing.allocator;
    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(allocator);

    const input = TemplateInput{ .string = "hello\nworld" };
    try input.writeJson(buffer.writer(allocator));

    try std.testing.expectEqualStrings("\"hello\\nworld\"", buffer.items);
}

test "TemplateInput.writeJson - integer" {
    const allocator = std.testing.allocator;
    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(allocator);

    const input = TemplateInput{ .integer = 42 };
    try input.writeJson(buffer.writer(allocator));

    try std.testing.expectEqualStrings("42", buffer.items);
}

test "TemplateInput.writeJson - float" {
    const allocator = std.testing.allocator;
    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(allocator);

    const input = TemplateInput{ .float = 3.14 };
    try input.writeJson(buffer.writer(allocator));

    try std.testing.expectEqualStrings("3.14", buffer.items);
}

test "TemplateInput.writeJson - boolean" {
    const allocator = std.testing.allocator;
    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(allocator);

    const true_input = TemplateInput{ .boolean = true };
    try true_input.writeJson(buffer.writer(allocator));
    try std.testing.expectEqualStrings("true", buffer.items);

    buffer.clearRetainingCapacity();

    const false_input = TemplateInput{ .boolean = false };
    try false_input.writeJson(buffer.writer(allocator));
    try std.testing.expectEqualStrings("false", buffer.items);
}

test "TemplateInput.writeJson - none" {
    const allocator = std.testing.allocator;
    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(allocator);

    const input: TemplateInput = .none;
    try input.writeJson(buffer.writer(allocator));

    try std.testing.expectEqualStrings("null", buffer.items);
}

test "TemplateInput.writeJson - array" {
    const allocator = std.testing.allocator;
    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(allocator);

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .string = "hello" },
    };
    const input = TemplateInput{ .array = &items };
    try input.writeJson(buffer.writer(allocator));

    try std.testing.expectEqualStrings("[1, \"hello\"]", buffer.items);
}

test "TemplateInput.writeJson - nested array" {
    const allocator = std.testing.allocator;
    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(allocator);

    const inner = [_]TemplateInput{.{ .integer = 1 }};
    const outer = [_]TemplateInput{.{ .array = &inner }};
    const input = TemplateInput{ .array = &outer };
    try input.writeJson(buffer.writer(allocator));

    try std.testing.expectEqualStrings("[[1]]", buffer.items);
}

test "TemplateInput.writeJson - map" {
    const allocator = std.testing.allocator;
    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(allocator);

    var map = std.StringHashMapUnmanaged(TemplateInput){};
    defer map.deinit(allocator);
    try map.put(allocator, "key", .{ .integer = 42 });

    const input = TemplateInput{ .map = map };
    try input.writeJson(buffer.writer(allocator));

    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "\"key\": 42") != null);
}

test "TemplateInput.writeJson - namespace" {
    const allocator = std.testing.allocator;
    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(allocator);

    var ns = std.StringHashMapUnmanaged(TemplateInput){};
    const input = TemplateInput{ .namespace = &ns };
    try input.writeJson(buffer.writer(allocator));

    try std.testing.expectEqualStrings("{}", buffer.items);
}

// ============================================================================
// Tests for helper types
// ============================================================================

test "LoopContext initialization with all fields" {
    const loop_ctx = LoopContext{
        .index0 = 2,
        .index = 3,
        .first = false,
        .last = false,
        .length = 10,
        .revindex = 8,
        .revindex0 = 7,
        .previtem = .{ .integer = 1 },
        .nextitem = .{ .integer = 2 },
        .depth = 2,
        .depth0 = 1,
        .recursive_body = null,
        .recursive_target = null,
        .recursive_target2 = null,
    };

    try std.testing.expectEqual(@as(usize, 2), loop_ctx.index0);
    try std.testing.expectEqual(@as(usize, 3), loop_ctx.index);
    try std.testing.expectEqual(false, loop_ctx.first);
    try std.testing.expectEqual(false, loop_ctx.last);
    try std.testing.expectEqual(@as(usize, 10), loop_ctx.length);
    try std.testing.expectEqual(@as(usize, 8), loop_ctx.revindex);
    try std.testing.expectEqual(@as(usize, 7), loop_ctx.revindex0);
    try std.testing.expectEqual(@as(usize, 2), loop_ctx.depth);
    try std.testing.expectEqual(@as(usize, 1), loop_ctx.depth0);
}

test "LoopContext minimal initialization" {
    const loop_ctx = LoopContext{
        .index0 = 0,
        .index = 1,
        .first = true,
        .last = true,
        .length = 1,
        .revindex = 1,
        .revindex0 = 0,
    };

    try std.testing.expectEqual(@as(usize, 0), loop_ctx.index0);
    try std.testing.expectEqual(@as(usize, 1), loop_ctx.index);
    try std.testing.expectEqual(true, loop_ctx.first);
    try std.testing.expectEqual(true, loop_ctx.last);
    try std.testing.expectEqual(@as(usize, 1), loop_ctx.length);
    try std.testing.expect(loop_ctx.previtem == null);
    try std.testing.expect(loop_ctx.nextitem == null);
    try std.testing.expectEqual(@as(usize, 1), loop_ctx.depth);
    try std.testing.expectEqual(@as(usize, 0), loop_ctx.depth0);
}

test "JoinerState initialization" {
    const joiner = JoinerState{ .separator = ", " };
    try std.testing.expectEqualStrings(", ", joiner.separator);
    try std.testing.expectEqual(false, joiner.called);
}

test "JoinerState state tracking" {
    var joiner = JoinerState{ .separator = " | " };
    try std.testing.expectEqual(false, joiner.called);

    joiner.called = true;
    try std.testing.expectEqual(true, joiner.called);
    try std.testing.expectEqualStrings(" | ", joiner.separator);
}

test "CyclerState initialization" {
    const items = [_]TemplateInput{
        .{ .string = "a" },
        .{ .string = "b" },
        .{ .string = "c" },
    };
    const cycler = CyclerState{ .items = &items };
    try std.testing.expectEqual(@as(usize, 3), cycler.items.len);
    try std.testing.expectEqual(@as(usize, 0), cycler.index);
}

test "CyclerState empty items" {
    const cycler = CyclerState{ .items = &.{} };
    try std.testing.expectEqual(@as(usize, 0), cycler.items.len);
    try std.testing.expectEqual(@as(usize, 0), cycler.index);
}

test "MacroDef initialization" {
    const macro = MacroDef{
        .name = "test_macro",
        .params = &.{},
        .body = &.{},
    };
    try std.testing.expectEqualStrings("test_macro", macro.name);
    try std.testing.expectEqual(@as(usize, 0), macro.params.len);
    try std.testing.expectEqual(@as(usize, 0), macro.body.len);
}
