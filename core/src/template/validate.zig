//! Template Validation
//!
//! Extract required variables from template AST and validate inputs.
//! This enables catching errors before expensive LLM API calls.

const std = @import("std");
const io = @import("../io/root.zig");
const ast = @import("ast.zig");
const lexer_mod = @import("lexer.zig");
const parser_mod = @import("parser.zig");

const Expr = ast.Expr;
const Node = ast.Node;
const Lexer = lexer_mod.Lexer;
const Parser = parser_mod.Parser;

/// Result of template validation
pub const ValidationResult = struct {
    /// Variables that are required (used without default() filter) and not provided
    required: []const []const u8,
    /// Variables that are optional (used with default() filter) and not provided
    optional: []const []const u8,
    /// Variables in inputs but not used by template
    extra: []const []const u8,
    /// Whether validation passed (no required variables missing)
    valid: bool,

    pub fn deinit(self: *ValidationResult, allocator: std.mem.Allocator) void {
        allocator.free(self.required);
        allocator.free(self.optional);
        allocator.free(self.extra);
    }
};

/// Built-in names that should not be considered input variables
const builtin_names = std.StaticStringMap(void).initComptime(.{
    .{ "loop", {} },
    .{ "true", {} },
    .{ "false", {} },
    .{ "none", {} },
    .{ "True", {} },
    .{ "False", {} },
    .{ "None", {} },
    .{ "range", {} },
    .{ "dict", {} },
    .{ "namespace", {} },
    .{ "joiner", {} },
    .{ "cycler", {} },
    .{ "lipsum", {} },
});

/// Result of variable extraction from template AST
pub const ExtractedVariables = struct {
    /// Variables used without default() filter - truly required
    required: std.StringHashMapUnmanaged(void),
    /// Variables used with default() filter - optional
    optional: std.StringHashMapUnmanaged(void),

    pub fn deinit(self: *ExtractedVariables, allocator: std.mem.Allocator) void {
        self.required.deinit(allocator);
        self.optional.deinit(allocator);
    }
};

/// Extract all variable names referenced in a template.
/// Returns two sets: required (naked variables) and optional (variables with default() filter).
/// For `user.name`, returns `user` (the root variable).
pub fn extractVariables(allocator: std.mem.Allocator, template: []const u8) !ExtractedVariables {
    // Tokenize
    var lexer_ctx = Lexer.init(allocator, template);
    defer lexer_ctx.deinit();

    const tokens = lexer_ctx.tokenize() catch return error.LexError;

    // Parse
    var parser_ctx = Parser.init(allocator, tokens);
    defer parser_ctx.deinit();

    const nodes = parser_ctx.parse() catch return error.ParseError;
    defer allocator.free(nodes);

    // Extract variables from AST
    var required = std.StringHashMapUnmanaged(void){};
    var optional = std.StringHashMapUnmanaged(void){};
    var defined_vars = std.StringHashMapUnmanaged(void){}; // Loop/set variables
    defer defined_vars.deinit(allocator);

    for (nodes) |node| {
        try extractFromNode(allocator, node, &required, &optional, &defined_vars);
    }

    return .{ .required = required, .optional = optional };
}

fn extractFromNode(
    allocator: std.mem.Allocator,
    node: *const Node,
    required: *std.StringHashMapUnmanaged(void),
    optional: *std.StringHashMapUnmanaged(void),
    defined_vars: *std.StringHashMapUnmanaged(void),
) !void {
    switch (node.*) {
        .text => {},
        .print => |expr| try extractFromExpr(allocator, expr, required, optional, defined_vars),
        .if_stmt => |stmt| {
            for (stmt.branches) |branch| {
                try extractFromExpr(allocator, branch.condition, required, optional, defined_vars);
                for (branch.body) |n| try extractFromNode(allocator, n, required, optional, defined_vars);
            }
            for (stmt.else_body) |n| try extractFromNode(allocator, n, required, optional, defined_vars);
        },
        .for_stmt => |stmt| {
            // The iterator is an input variable
            try extractFromExpr(allocator, stmt.iterable, required, optional, defined_vars);
            // The loop variable(s) are defined, not inputs
            try defined_vars.put(allocator, stmt.target, {});
            if (stmt.target2) |v2| try defined_vars.put(allocator, v2, {});
            // Optional filter condition
            if (stmt.filter) |f| try extractFromExpr(allocator, f, required, optional, defined_vars);
            // Process body
            for (stmt.body) |n| try extractFromNode(allocator, n, required, optional, defined_vars);
            for (stmt.else_body) |n| try extractFromNode(allocator, n, required, optional, defined_vars);
        },
        .set_stmt => |stmt| {
            // Value is extracted first (may reference existing vars)
            try extractFromExpr(allocator, stmt.value, required, optional, defined_vars);
            // Then the variable is defined
            try defined_vars.put(allocator, stmt.target, {});
        },
        .macro_def => |stmt| {
            // Macro name is defined
            try defined_vars.put(allocator, stmt.name, {});
            // Macro params are defined within macro scope
            for (stmt.params) |p| try defined_vars.put(allocator, p.name, {});
            for (stmt.body) |n| try extractFromNode(allocator, n, required, optional, defined_vars);
        },
        .macro_call_stmt => |stmt| {
            for (stmt.args) |arg| try extractFromExpr(allocator, arg, required, optional, defined_vars);
            for (stmt.kwargs) |kw| try extractFromExpr(allocator, kw.value, required, optional, defined_vars);
        },
        .call_block => |stmt| {
            for (stmt.args) |arg| try extractFromExpr(allocator, arg, required, optional, defined_vars);
            for (stmt.body) |n| try extractFromNode(allocator, n, required, optional, defined_vars);
        },
        .filter_block => |stmt| {
            for (stmt.body) |n| try extractFromNode(allocator, n, required, optional, defined_vars);
        },
        .generation_block => |stmt| {
            for (stmt.body) |n| try extractFromNode(allocator, n, required, optional, defined_vars);
        },
        .include => |stmt| {
            // Extract variables from the include expression
            // Note: We can't know what variables the included template needs at static analysis time
            try extractFromExpr(allocator, stmt.template_expr, required, optional, defined_vars);
        },
        .break_stmt, .continue_stmt => {},
    }
}

fn extractFromExpr(
    allocator: std.mem.Allocator,
    expr: *const Expr,
    required: *std.StringHashMapUnmanaged(void),
    optional: *std.StringHashMapUnmanaged(void),
    defined_vars: *std.StringHashMapUnmanaged(void),
) !void {
    switch (expr.*) {
        .string, .integer, .float, .boolean, .none => {},
        .variable => |name| {
            // Only add if not a built-in and not locally defined
            if (!builtin_names.has(name) and !defined_vars.contains(name)) {
                try required.put(allocator, name, {});
            }
        },
        .getattr => |ga| {
            // For user.name, we want to extract "user" (the root)
            try extractRootVariable(allocator, ga.object, required, defined_vars);
        },
        .getitem => |gi| {
            try extractRootVariable(allocator, gi.object, required, defined_vars);
            try extractFromExpr(allocator, gi.key, required, optional, defined_vars);
        },
        .slice => |sl| {
            try extractRootVariable(allocator, sl.object, required, defined_vars);
            if (sl.start) |s| try extractFromExpr(allocator, s, required, optional, defined_vars);
            if (sl.stop) |s| try extractFromExpr(allocator, s, required, optional, defined_vars);
            if (sl.step) |s| try extractFromExpr(allocator, s, required, optional, defined_vars);
        },
        .binop => |bo| {
            try extractFromExpr(allocator, bo.left, required, optional, defined_vars);
            try extractFromExpr(allocator, bo.right, required, optional, defined_vars);
        },
        .unaryop => |uo| {
            try extractFromExpr(allocator, uo.operand, required, optional, defined_vars);
        },
        .call => |c| {
            try extractFromExpr(allocator, c.func, required, optional, defined_vars);
            for (c.args) |arg| try extractFromExpr(allocator, arg, required, optional, defined_vars);
        },
        .filter => |f| {
            // Check if this is a default() or d() filter
            const is_default = std.mem.eql(u8, f.name, "default") or std.mem.eql(u8, f.name, "d");
            if (is_default) {
                // Variables under default() filter go to optional set
                try extractFromExprAsOptional(allocator, f.value, optional, defined_vars);
            } else {
                try extractFromExpr(allocator, f.value, required, optional, defined_vars);
            }
            // Filter arguments are always required
            for (f.args) |arg| try extractFromExpr(allocator, arg, required, optional, defined_vars);
        },
        .test_expr => |t| {
            try extractFromExpr(allocator, t.value, required, optional, defined_vars);
            for (t.args) |arg| try extractFromExpr(allocator, arg, required, optional, defined_vars);
        },
        .conditional => |c| {
            try extractFromExpr(allocator, c.test_val, required, optional, defined_vars);
            try extractFromExpr(allocator, c.true_val, required, optional, defined_vars);
            try extractFromExpr(allocator, c.false_val, required, optional, defined_vars);
        },
        .list => |items| {
            for (items) |item| try extractFromExpr(allocator, item, required, optional, defined_vars);
        },
        .dict => |pairs| {
            for (pairs) |pair| {
                try extractFromExpr(allocator, pair.key, required, optional, defined_vars);
                try extractFromExpr(allocator, pair.value, required, optional, defined_vars);
            }
        },
        .namespace_call => |args| {
            for (args) |arg| try extractFromExpr(allocator, arg.value, required, optional, defined_vars);
        },
        .macro_call => |mc| {
            for (mc.args) |arg| try extractFromExpr(allocator, arg, required, optional, defined_vars);
            for (mc.kwargs) |kw| try extractFromExpr(allocator, kw.value, required, optional, defined_vars);
        },
    }
}

/// Extract variables from expression, marking them as optional (used under default() filter)
fn extractFromExprAsOptional(
    allocator: std.mem.Allocator,
    expr: *const Expr,
    optional: *std.StringHashMapUnmanaged(void),
    defined_vars: *std.StringHashMapUnmanaged(void),
) !void {
    switch (expr.*) {
        .string, .integer, .float, .boolean, .none => {},
        .variable => |name| {
            if (!builtin_names.has(name) and !defined_vars.contains(name)) {
                try optional.put(allocator, name, {});
            }
        },
        .getattr => |ga| {
            try extractRootVariableAsOptional(allocator, ga.object, optional, defined_vars);
        },
        .getitem => |gi| {
            try extractRootVariableAsOptional(allocator, gi.object, optional, defined_vars);
        },
        // For complex expressions under default(), extract all as optional
        .binop => |bo| {
            try extractFromExprAsOptional(allocator, bo.left, optional, defined_vars);
            try extractFromExprAsOptional(allocator, bo.right, optional, defined_vars);
        },
        .unaryop => |uo| {
            try extractFromExprAsOptional(allocator, uo.operand, optional, defined_vars);
        },
        .call => |c| {
            try extractFromExprAsOptional(allocator, c.func, optional, defined_vars);
            for (c.args) |arg| try extractFromExprAsOptional(allocator, arg, optional, defined_vars);
        },
        .filter => |f| {
            try extractFromExprAsOptional(allocator, f.value, optional, defined_vars);
            for (f.args) |arg| try extractFromExprAsOptional(allocator, arg, optional, defined_vars);
        },
        else => {},
    }
}

/// Extract root variable as optional
fn extractRootVariableAsOptional(
    allocator: std.mem.Allocator,
    expr: *const Expr,
    optional: *std.StringHashMapUnmanaged(void),
    defined_vars: *std.StringHashMapUnmanaged(void),
) std.mem.Allocator.Error!void {
    switch (expr.*) {
        .variable => |name| {
            if (!builtin_names.has(name) and !defined_vars.contains(name)) {
                try optional.put(allocator, name, {});
            }
        },
        .getattr => |ga| try extractRootVariableAsOptional(allocator, ga.object, optional, defined_vars),
        .getitem => |gi| try extractRootVariableAsOptional(allocator, gi.object, optional, defined_vars),
        .call => |c| try extractRootVariableAsOptional(allocator, c.func, optional, defined_vars),
        else => {},
    }
}

/// Extract the root variable from a chained expression (e.g., user.name.first -> user)
fn extractRootVariable(
    allocator: std.mem.Allocator,
    expr: *const Expr,
    required: *std.StringHashMapUnmanaged(void),
    defined_vars: *std.StringHashMapUnmanaged(void),
) std.mem.Allocator.Error!void {
    switch (expr.*) {
        .variable => |name| {
            if (!builtin_names.has(name) and !defined_vars.contains(name)) {
                try required.put(allocator, name, {});
            }
        },
        .getattr => |ga| try extractRootVariable(allocator, ga.object, required, defined_vars),
        .getitem => |gi| try extractRootVariable(allocator, gi.object, required, defined_vars),
        .call => |c| try extractRootVariable(allocator, c.func, required, defined_vars),
        else => {
            // For other expressions, we need optional set too but we don't have it here
            // This is OK - getattr/getitem chains cover the common case
        },
    }
}

/// JSON validation error with descriptive message.
pub const JsonError = error{
    InvalidJson,
    InvalidNumber,
    Overflow,
    DuplicateField,
    OutOfMemory,
    TemplateSyntaxError,
};

/// Validate template inputs from JSON string.
/// Parses JSON to extract keys and validates against template requirements.
pub fn validateJson(
    allocator: std.mem.Allocator,
    template: []const u8,
    json_vars: []const u8,
) JsonError!ValidationResult {
    // Parse JSON to extract keys
    const parsed_json = io.json.parseValue(allocator, json_vars, .{ .max_size_bytes = 1 * 1024 * 1024 }) catch |err| {
        return switch (err) {
            error.InputTooLarge => error.InvalidJson,
            error.InputTooDeep => error.InvalidJson,
            error.StringTooLong => error.InvalidJson,
            error.InvalidJson => error.InvalidJson,
            error.OutOfMemory => error.OutOfMemory,
        };
    };
    defer parsed_json.deinit();

    // Extract keys from JSON object
    var input_keys = std.ArrayListUnmanaged([]const u8){};
    defer input_keys.deinit(allocator);

    if (parsed_json.value == .object) {
        var object_iter = parsed_json.value.object.iterator();
        while (object_iter.next()) |entry| {
            input_keys.append(allocator, entry.key_ptr.*) catch continue;
        }
    }

    // Run validation
    return validate(allocator, template, input_keys.items) catch |err| {
        return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.TemplateSyntaxError, // LexError/ParseError -> template syntax error
        };
    };
}

/// Get descriptive error message for JSON validation error.
pub fn jsonErrorMessage(err: JsonError) []const u8 {
    return switch (err) {
        error.InvalidJson => "invalid JSON syntax",
        error.InvalidNumber => "invalid number format in JSON",
        error.Overflow => "numeric overflow in JSON",
        error.DuplicateField => "duplicate field in JSON object",
        error.OutOfMemory => "out of memory",
        error.TemplateSyntaxError => "invalid template syntax",
    };
}

/// Validate template inputs against required/optional variables.
/// Returns validation result with required, optional, and extra variables.
pub fn validate(
    allocator: std.mem.Allocator,
    template: []const u8,
    input_keys: []const []const u8,
) !ValidationResult {
    // Extract required and optional variables from template
    var extracted = try extractVariables(allocator, template);
    defer extracted.deinit(allocator);

    // Build set of provided inputs
    var provided = std.StringHashMapUnmanaged(void){};
    defer provided.deinit(allocator);
    for (input_keys) |key| {
        try provided.put(allocator, key, {});
    }

    // Find missing required: in required but not in provided
    var required_list = std.ArrayListUnmanaged([]const u8){};
    var req_iter = extracted.required.iterator();
    while (req_iter.next()) |entry| {
        if (!provided.contains(entry.key_ptr.*)) {
            try required_list.append(allocator, entry.key_ptr.*);
        }
    }

    // Find missing optional: in optional but not in provided
    // (and not already in required - a variable can be both if used in multiple places)
    var optional_list = std.ArrayListUnmanaged([]const u8){};
    var opt_iter = extracted.optional.iterator();
    while (opt_iter.next()) |entry| {
        if (!provided.contains(entry.key_ptr.*) and !extracted.required.contains(entry.key_ptr.*)) {
            try optional_list.append(allocator, entry.key_ptr.*);
        }
    }

    // Find extra: in provided but not in required or optional
    var extra_list = std.ArrayListUnmanaged([]const u8){};
    var provided_iter = provided.iterator();
    while (provided_iter.next()) |entry| {
        if (!extracted.required.contains(entry.key_ptr.*) and !extracted.optional.contains(entry.key_ptr.*)) {
            try extra_list.append(allocator, entry.key_ptr.*);
        }
    }

    const required_missing = try required_list.toOwnedSlice(allocator);
    const optional_missing = try optional_list.toOwnedSlice(allocator);
    const extra = try extra_list.toOwnedSlice(allocator);

    return .{
        .required = required_missing,
        .optional = optional_missing,
        .extra = extra,
        .valid = required_missing.len == 0,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "extractVariables simple" {
    const allocator = std.testing.allocator;
    var vars = try extractVariables(allocator, "Hello {{ name }}!");
    defer vars.deinit(allocator);

    try std.testing.expect(vars.required.contains("name"));
    try std.testing.expectEqual(@as(usize, 1), vars.required.count());
    try std.testing.expectEqual(@as(usize, 0), vars.optional.count());
}

test "extractVariables multiple" {
    const allocator = std.testing.allocator;
    var vars = try extractVariables(allocator, "{{ name }} is {{ age }} years old");
    defer vars.deinit(allocator);

    try std.testing.expect(vars.required.contains("name"));
    try std.testing.expect(vars.required.contains("age"));
    try std.testing.expectEqual(@as(usize, 2), vars.required.count());
}

test "extractVariables nested attribute" {
    const allocator = std.testing.allocator;
    var vars = try extractVariables(allocator, "{{ user.name }}");
    defer vars.deinit(allocator);

    // Should extract root variable "user", not "user.name"
    try std.testing.expect(vars.required.contains("user"));
    try std.testing.expectEqual(@as(usize, 1), vars.required.count());
}

test "extractVariables for loop" {
    const allocator = std.testing.allocator;
    var vars = try extractVariables(allocator, "{% for item in items %}{{ item }}{% endfor %}");
    defer vars.deinit(allocator);

    // "items" is an input, "item" is a loop variable (not input)
    try std.testing.expect(vars.required.contains("items"));
    try std.testing.expect(!vars.required.contains("item"));
    try std.testing.expectEqual(@as(usize, 1), vars.required.count());
}

test "extractVariables excludes loop builtin" {
    const allocator = std.testing.allocator;
    var vars = try extractVariables(allocator, "{% for x in items %}{{ loop.index }}{% endfor %}");
    defer vars.deinit(allocator);

    try std.testing.expect(vars.required.contains("items"));
    try std.testing.expect(!vars.required.contains("loop"));
    try std.testing.expectEqual(@as(usize, 1), vars.required.count());
}

test "extractVariables set statement" {
    const allocator = std.testing.allocator;
    var vars = try extractVariables(allocator, "{% set x = value %}{{ x }}");
    defer vars.deinit(allocator);

    // "value" is input, "x" is defined by set
    try std.testing.expect(vars.required.contains("value"));
    try std.testing.expect(!vars.required.contains("x"));
    try std.testing.expectEqual(@as(usize, 1), vars.required.count());
}

test "extractVariables if condition" {
    const allocator = std.testing.allocator;
    var vars = try extractVariables(allocator, "{% if show %}{{ content }}{% endif %}");
    defer vars.deinit(allocator);

    try std.testing.expect(vars.required.contains("show"));
    try std.testing.expect(vars.required.contains("content"));
    try std.testing.expectEqual(@as(usize, 2), vars.required.count());
}

test "extractVariables filter" {
    const allocator = std.testing.allocator;
    var vars = try extractVariables(allocator, "{{ name | upper }}");
    defer vars.deinit(allocator);

    try std.testing.expect(vars.required.contains("name"));
    try std.testing.expectEqual(@as(usize, 1), vars.required.count());
}

test "extractVariables with default filter is optional" {
    const allocator = std.testing.allocator;
    var vars = try extractVariables(allocator, "{{ context | default('') }}");
    defer vars.deinit(allocator);

    // Variable with default() filter should be optional, not required
    try std.testing.expect(!vars.required.contains("context"));
    try std.testing.expect(vars.optional.contains("context"));
    try std.testing.expectEqual(@as(usize, 0), vars.required.count());
    try std.testing.expectEqual(@as(usize, 1), vars.optional.count());
}

test "extractVariables with d filter alias is optional" {
    const allocator = std.testing.allocator;
    var vars = try extractVariables(allocator, "{{ value | d('fallback') }}");
    defer vars.deinit(allocator);

    // d() is alias for default()
    try std.testing.expect(!vars.required.contains("value"));
    try std.testing.expect(vars.optional.contains("value"));
}

test "extractVariables mixed required and optional" {
    const allocator = std.testing.allocator;
    var vars = try extractVariables(allocator, "{{ name }} - {{ context | default('N/A') }}");
    defer vars.deinit(allocator);

    // name is required (naked), context is optional (has default)
    try std.testing.expect(vars.required.contains("name"));
    try std.testing.expect(!vars.required.contains("context"));
    try std.testing.expect(vars.optional.contains("context"));
    try std.testing.expectEqual(@as(usize, 1), vars.required.count());
    try std.testing.expectEqual(@as(usize, 1), vars.optional.count());
}

test "extractVariables nested with default" {
    const allocator = std.testing.allocator;
    var vars = try extractVariables(allocator, "{{ user.profile | default({}) }}");
    defer vars.deinit(allocator);

    // user is optional because it's under default()
    try std.testing.expect(!vars.required.contains("user"));
    try std.testing.expect(vars.optional.contains("user"));
}

test "validate missing required variable" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{"name"};
    var result = try validate(allocator, "{{ name }} {{ age }}", &inputs);
    defer result.deinit(allocator);

    try std.testing.expect(!result.valid);
    try std.testing.expectEqual(@as(usize, 1), result.required.len);
    try std.testing.expectEqualStrings("age", result.required[0]);
}

test "validate extra variable" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{ "name", "unused" };
    var result = try validate(allocator, "{{ name }}", &inputs);
    defer result.deinit(allocator);

    try std.testing.expect(result.valid); // Extra vars don't fail validation
    try std.testing.expectEqual(@as(usize, 0), result.required.len);
    try std.testing.expectEqual(@as(usize, 1), result.extra.len);
    try std.testing.expectEqualStrings("unused", result.extra[0]);
}

test "validate all provided" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{ "name", "age" };
    var result = try validate(allocator, "{{ name }} {{ age }}", &inputs);
    defer result.deinit(allocator);

    try std.testing.expect(result.valid);
    try std.testing.expectEqual(@as(usize, 0), result.required.len);
    try std.testing.expectEqual(@as(usize, 0), result.extra.len);
}

test "validate optional variable missing is valid" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{"name"};
    var result = try validate(allocator, "{{ name }} {{ context | default('') }}", &inputs);
    defer result.deinit(allocator);

    // Missing optional variable doesn't fail validation
    try std.testing.expect(result.valid);
    try std.testing.expectEqual(@as(usize, 0), result.required.len);
    try std.testing.expectEqual(@as(usize, 1), result.optional.len);
    try std.testing.expectEqualStrings("context", result.optional[0]);
}

test "validate variable used both naked and with default counts as required" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{};
    // x is used naked first, then with default - should be required
    var result = try validate(allocator, "{{ x }} {{ x | default('') }}", &inputs);
    defer result.deinit(allocator);

    try std.testing.expect(!result.valid);
    try std.testing.expectEqual(@as(usize, 1), result.required.len);
    try std.testing.expectEqualStrings("x", result.required[0]);
    // x shouldn't appear in optional since it's required
    try std.testing.expectEqual(@as(usize, 0), result.optional.len);
}

// ============================================================================
// JSON Validation Tests
// ============================================================================

test "validateJson valid JSON" {
    const allocator = std.testing.allocator;
    var result = try validateJson(allocator, "{{ name }} {{ age }}", "{\"name\": \"Alice\", \"age\": 30}");
    defer result.deinit(allocator);

    try std.testing.expect(result.valid);
}

test "validateJson missing variable" {
    const allocator = std.testing.allocator;
    var result = try validateJson(allocator, "{{ name }} {{ age }}", "{\"name\": \"Alice\"}");
    defer result.deinit(allocator);

    try std.testing.expect(!result.valid);
    try std.testing.expectEqual(@as(usize, 1), result.required.len);
}

test "validateJson invalid JSON syntax" {
    const allocator = std.testing.allocator;
    const err = validateJson(allocator, "{{ name }}", "{invalid json}");
    try std.testing.expectError(error.InvalidJson, err);
}

test "validateJson unclosed brace" {
    const allocator = std.testing.allocator;
    const err = validateJson(allocator, "{{ name }}", "{\"name\": \"Alice\"");
    try std.testing.expectError(error.InvalidJson, err);
}

test "validateJson invalid number" {
    const allocator = std.testing.allocator;
    const err = validateJson(allocator, "{{ x }}", "{\"x\": 1.2.3}");
    try std.testing.expectError(error.InvalidJson, err);
}

test "validateJson empty object" {
    const allocator = std.testing.allocator;
    var result = try validateJson(allocator, "{{ name }}", "{}");
    defer result.deinit(allocator);

    try std.testing.expect(!result.valid);
    try std.testing.expectEqual(@as(usize, 1), result.required.len);
}

test "jsonErrorMessage returns descriptive messages" {
    try std.testing.expectEqualStrings("invalid JSON syntax", jsonErrorMessage(error.InvalidJson));
    try std.testing.expectEqualStrings("invalid number format in JSON", jsonErrorMessage(error.InvalidNumber));
    try std.testing.expectEqualStrings("invalid template syntax", jsonErrorMessage(error.TemplateSyntaxError));
}

test "ValidationResult.deinit frees allocated slices" {
    const allocator = std.testing.allocator;
    var result = ValidationResult{
        .required = try allocator.alloc([]const u8, 0),
        .optional = try allocator.alloc([]const u8, 0),
        .extra = try allocator.alloc([]const u8, 0),
        .valid = true,
    };
    result.deinit(allocator);
}

test "ExtractedVariables.deinit frees hash maps" {
    const allocator = std.testing.allocator;
    var vars = ExtractedVariables{
        .required = std.StringHashMapUnmanaged(void){},
        .optional = std.StringHashMapUnmanaged(void){},
    };
    try vars.required.put(allocator, "test_var", {});
    try vars.optional.put(allocator, "optional_var", {});
    vars.deinit(allocator);
}
