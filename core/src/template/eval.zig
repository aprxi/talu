//! Template Evaluation Engine
//!
//! Provides the core types for template parsing and rendering:
//!
//! - `TemplateParser`: Parse and render templates with your data
//! - `TemplateInput`: Represent dynamic data (strings, numbers, lists, objects)
//!
//! ## Example
//!
//! ```zig
//! var parser = TemplateParser.init(allocator);
//! defer parser.deinit();
//!
//! // Add your data
//! try parser.set("name", .{ .string = "Alice" });
//! try parser.set("items", .{ .array = &.{...} });
//!
//! // Render the template
//! const result = try render(allocator, "Hello {{ name }}!", &parser);
//! ```

const std = @import("std");
const ast = @import("ast.zig");
const filters = @import("filters/root.zig");
const predicates = @import("predicates.zig");
const methods = @import("methods.zig");
const functions = @import("functions.zig");
const lexer_mod = @import("lexer.zig");
const parser_mod = @import("parser.zig");
pub const input = @import("input.zig");

// Re-export types from input.zig (eval.zig is the main module entry point)
pub const TemplateInput = input.TemplateInput;
pub const LoopContext = input.LoopContext;
pub const MacroDef = input.MacroDef;
pub const JoinerState = input.JoinerState;
pub const CyclerState = input.CyclerState;

const Expr = ast.Expr;
const Node = ast.Node;
const BinOp = ast.BinOp;
const UnaryOp = ast.UnaryOp;

pub const EvalError = error{
    UndefinedVariable,
    TypeError,
    IndexOutOfBounds,
    KeyError,
    DivisionByZero,
    InvalidOperation,
    OutOfMemory,
    UnsupportedFilter,
    UnsupportedTest,
    UnsupportedMethod,
    LoopBreak,
    LoopContinue,
    CustomFilterError,
    RaiseException,
    IncludeError,
    IncludeTypeError, // include got non-string (likely undefined variable)
};

// ============================================================================
// Custom Filter Support
// ============================================================================

/// Callback function type for custom filters.
/// Called from Python/C to execute user-defined filter logic.
///
/// Parameters:
///   - value_json: JSON-encoded filter input value
///   - args_json: JSON array of filter arguments (e.g., '["arg1", 2]')
///   - user_data: Optional context pointer passed during registration
///
/// Returns:
///   - JSON-encoded result string (caller must free with c_allocator)
///   - null on error
pub const CustomFilterCallback = *const fn (
    [*:0]const u8, // value_json
    [*:0]const u8, // args_json
    ?*anyopaque, // user_data
) callconv(.c) ?[*:0]u8;

/// A registered custom filter with its callback and context.
pub const CustomFilter = struct {
    callback: CustomFilterCallback,
    user_data: ?*anyopaque,
};

/// Set of custom filters passed to the evaluator.
/// Filters are looked up by name during template rendering.
pub const CustomFilterSet = struct {
    filters: std.StringHashMapUnmanaged(CustomFilter),

    pub fn init() CustomFilterSet {
        return .{ .filters = .{} };
    }

    pub fn deinit(self: *CustomFilterSet, allocator: std.mem.Allocator) void {
        self.filters.deinit(allocator);
    }

    pub fn put(self: *CustomFilterSet, allocator: std.mem.Allocator, name: []const u8, filter: CustomFilter) !void {
        try self.filters.put(allocator, name, filter);
    }

    pub fn get(self: *const CustomFilterSet, name: []const u8) ?CustomFilter {
        return self.filters.get(name);
    }
};

// ============================================================================
// Debug Mode: Output Span Tracking
// ============================================================================

/// Source type for a span in the rendered output.
/// Tracks whether output came from static template text or dynamic expressions.
pub const SpanSource = union(enum) {
    /// Static text from the template (not from a variable)
    static_text,
    /// Variable substitution - stores the variable path (e.g., "name", "user.email")
    variable: []const u8,
    /// Complex expression (arithmetic, filters, etc.) - cannot attribute to single variable
    expression,
};

/// A span in the rendered output, tracking what produced it.
/// Used by debug mode to show which parts came from variables vs static text.
pub const OutputSpan = struct {
    /// Start position in the rendered output (inclusive)
    start: usize,
    /// End position in the rendered output (exclusive)
    end: usize,
    /// What produced this span
    source: SpanSource,
};

/// Parse and render templates with dynamic data.
///
/// `TemplateParser` holds the variables that templates can access. Create one,
/// add your data with `set()`, then pass it to `render()` to process a template.
///
/// ## Example
///
/// ```zig
/// var parser = TemplateParser.init(allocator);
/// defer parser.deinit();
///
/// // Add variables the template can use
/// try parser.set("user", .{ .string = "Alice" });
/// try parser.set("score", .{ .integer = 100 });
///
/// // Render a template
/// const result = try render(allocator, "{{ user }} scored {{ score }}!", &parser);
/// // result: "Alice scored 100!"
/// ```
///
/// ## Supported Template Syntax
///
/// - Variables: `{{ name }}`
/// - Conditionals: `{% if condition %}...{% endif %}`
/// - Loops: `{% for item in items %}...{% endfor %}`
/// - Filters: `{{ name | upper }}`
///
pub const TemplateParser = struct {
    allocator: std.mem.Allocator,
    variables: std.StringHashMap(TemplateInput),
    loop: ?LoopContext = null,

    /// Arena for intermediate allocations during evaluation
    arena: std.heap.ArenaAllocator,

    /// When true, accessing undefined variables raises UndefinedVariable error.
    /// When false (default), undefined variables silently return .none.
    strict: bool = false,

    /// Message from raise_exception() call (stored here so it survives Evaluator destruction)
    raise_exception_message: ?[]const u8 = null,

    /// Path to undefined variable/key (stored here so it survives Evaluator destruction)
    /// Format: "var.attr.key" for nested access paths
    undefined_path: ?[]const u8 = null,

    /// Context for parse errors (e.g., "if", "for" block name)
    parse_error_context: ?[]const u8 = null,

    pub fn init(allocator: std.mem.Allocator) TemplateParser {
        return .{
            .allocator = allocator,
            .variables = std.StringHashMap(TemplateInput).init(allocator),
            .arena = std.heap.ArenaAllocator.init(allocator),
        };
    }

    /// Create a TemplateParser with strict mode enabled.
    /// In strict mode, accessing undefined variables raises an error.
    pub fn initStrict(allocator: std.mem.Allocator) TemplateParser {
        var parser = init(allocator);
        parser.strict = true;
        return parser;
    }

    pub fn deinit(self: *TemplateParser) void {
        // Free undefined_path if allocated
        if (self.undefined_path) |path| {
            self.allocator.free(path);
        }
        self.variables.deinit();
        self.arena.deinit();
    }

    /// Set the undefined path for error reporting.
    /// The path is allocated in the allocator (not arena) so it survives deinit.
    pub fn setUndefinedPath(self: *TemplateParser, path: []const u8) void {
        // Free previous path if any
        if (self.undefined_path) |old| {
            self.allocator.free(old);
        }
        self.undefined_path = self.allocator.dupe(u8, path) catch null;
    }

    /// Set the undefined path with attribute appended.
    /// Format: "base.attr" or just "attr" if base is null.
    pub fn setUndefinedPathAttr(self: *TemplateParser, base: ?[]const u8, attr: []const u8) void {
        if (base) |b| {
            const path = std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ b, attr }) catch return;
            if (self.undefined_path) |old| self.allocator.free(old);
            self.undefined_path = path;
        } else {
            self.setUndefinedPath(attr);
        }
    }

    pub fn set(self: *TemplateParser, name: []const u8, value: TemplateInput) !void {
        try self.variables.put(name, value);
    }

    pub fn get(self: *TemplateParser, name: []const u8) ?TemplateInput {
        return self.variables.get(name);
    }

    /// Get the arena allocator for values that should be freed with the parser.
    pub fn arenaAllocator(self: *TemplateParser) std.mem.Allocator {
        return self.arena.allocator();
    }
};

/// Process escape sequences in a string (like \n, \", \\)
fn processEscapes(allocator: std.mem.Allocator, s: []const u8) ![]const u8 {
    // Quick check if there are any escapes
    if (std.mem.indexOf(u8, s, "\\") == null) {
        return s;
    }

    var result = std.ArrayListUnmanaged(u8){};
    errdefer result.deinit(allocator);

    var char_idx: usize = 0;
    while (char_idx < s.len) {
        if (s[char_idx] == '\\' and char_idx + 1 < s.len) {
            const next = s[char_idx + 1];
            switch (next) {
                'n' => try result.append(allocator, '\n'),
                't' => try result.append(allocator, '\t'),
                'r' => try result.append(allocator, '\r'),
                '\\' => try result.append(allocator, '\\'),
                '"' => try result.append(allocator, '"'),
                '\'' => try result.append(allocator, '\''),
                else => {
                    // Unknown escape - keep both characters
                    try result.append(allocator, '\\');
                    try result.append(allocator, next);
                },
            }
            char_idx += 2;
        } else {
            try result.append(allocator, s[char_idx]);
            char_idx += 1;
        }
    }

    return result.toOwnedSlice(allocator);
}

/// Local scope for variables set inside loops (Jinja2 scope isolation).
const LocalScope = std.StringHashMapUnmanaged(TemplateInput);

/// Template evaluator
pub const Evaluator = struct {
    allocator: std.mem.Allocator,
    ctx: *TemplateParser,
    output: std.ArrayListUnmanaged(u8),
    caller_body: ?[]const *const Node = null,

    /// Debug mode: track output spans (only populated when debug_mode=true)
    spans: std.ArrayListUnmanaged(OutputSpan) = .{},
    /// Enable span tracking for debug output
    debug_mode: bool = false,

    /// Optional custom filters (Python callbacks)
    custom_filters: ?*const CustomFilterSet = null,

    /// Stack of local scopes for loop variable isolation (Jinja2 spec)
    local_scopes: std.ArrayListUnmanaged(LocalScope) = .{},

    pub fn init(allocator: std.mem.Allocator, ctx: *TemplateParser) Evaluator {
        return .{
            .allocator = allocator,
            .ctx = ctx,
            .output = .{},
            .caller_body = null,
            .spans = .{},
            .debug_mode = false,
            .custom_filters = null,
            .local_scopes = .{},
        };
    }

    pub fn initDebug(allocator: std.mem.Allocator, ctx: *TemplateParser) Evaluator {
        return .{
            .allocator = allocator,
            .ctx = ctx,
            .output = .{},
            .caller_body = null,
            .spans = .{},
            .debug_mode = true,
            .custom_filters = null,
            .local_scopes = .{},
        };
    }

    pub fn initWithFilters(allocator: std.mem.Allocator, ctx: *TemplateParser, custom_filters: ?*const CustomFilterSet) Evaluator {
        return .{
            .allocator = allocator,
            .ctx = ctx,
            .output = .{},
            .caller_body = null,
            .spans = .{},
            .debug_mode = false,
            .custom_filters = custom_filters,
            .local_scopes = .{},
        };
    }

    pub fn deinit(self: *Evaluator) void {
        self.output.deinit(self.allocator);
        self.spans.deinit(self.allocator);
        for (self.local_scopes.items) |*scope| {
            scope.deinit(self.allocator);
        }
        self.local_scopes.deinit(self.allocator);
    }

    /// Push a new local scope for loop variable isolation
    fn pushScope(self: *Evaluator) EvalError!void {
        self.local_scopes.append(self.allocator, LocalScope{}) catch return EvalError.OutOfMemory;
    }

    /// Pop the current local scope
    fn popScope(self: *Evaluator) void {
        if (self.local_scopes.pop()) |scope| {
            var s = scope;
            s.deinit(self.allocator);
        }
    }

    /// Get a variable, checking local scopes first (innermost to outermost), then global context
    fn getVar(self: *Evaluator, name: []const u8) ?TemplateInput {
        // Check local scopes from innermost to outermost
        var i = self.local_scopes.items.len;
        while (i > 0) {
            i -= 1;
            if (self.local_scopes.items[i].get(name)) |val| {
                return val;
            }
        }
        // Fall back to global context
        return self.ctx.get(name);
    }

    /// Set a variable in the current scope (local if inside loop, global otherwise)
    fn setVar(self: *Evaluator, name: []const u8, value: TemplateInput) EvalError!void {
        if (self.local_scopes.items.len > 0) {
            // Inside a loop - set in innermost local scope
            self.local_scopes.items[self.local_scopes.items.len - 1].put(self.allocator, name, value) catch return EvalError.OutOfMemory;
        } else {
            // No loop - set in global context
            self.ctx.set(name, value) catch return EvalError.OutOfMemory;
        }
    }

    pub fn render(self: *Evaluator, nodes: []const *const Node) EvalError![]const u8 {
        try self.evalNodes(nodes);
        return self.output.toOwnedSlice(self.allocator) catch return EvalError.OutOfMemory;
    }

    /// Render and return spans (for debug mode)
    pub fn renderWithSpans(self: *Evaluator, nodes: []const *const Node) EvalError!struct { output: []const u8, spans: []const OutputSpan } {
        try self.evalNodes(nodes);
        return .{
            .output = self.output.toOwnedSlice(self.allocator) catch return EvalError.OutOfMemory,
            .spans = self.spans.toOwnedSlice(self.allocator) catch return EvalError.OutOfMemory,
        };
    }

    fn evalNodes(self: *Evaluator, nodes: []const *const Node) EvalError!void {
        for (nodes) |node| {
            try self.evalNode(node);
        }
    }

    /// Record a span if debug mode is enabled
    fn recordSpan(self: *Evaluator, start: usize, end: usize, source: SpanSource) EvalError!void {
        if (self.debug_mode and start < end) {
            self.spans.append(self.allocator, .{
                .start = start,
                .end = end,
                .source = source,
            }) catch return EvalError.OutOfMemory;
        }
    }

    fn appendValue(self: *Evaluator, value: TemplateInput) EvalError!void {
        const str = value.asString(self.ctx.arena.allocator()) catch return EvalError.OutOfMemory;
        self.output.appendSlice(self.allocator, str) catch return EvalError.OutOfMemory;
    }

    /// Append value and record span with source information
    fn appendValueWithSpan(self: *Evaluator, value: TemplateInput, source: SpanSource) EvalError!void {
        const start = self.output.items.len;
        try self.appendValue(value);
        const end = self.output.items.len;
        try self.recordSpan(start, end, source);
    }

    pub fn renderNodesToString(self: *Evaluator, nodes: []const *const Node) EvalError![]const u8 {
        const old_output = self.output;
        const old_spans = self.spans;
        const old_debug = self.debug_mode;
        self.output = std.ArrayListUnmanaged(u8){};
        self.spans = std.ArrayListUnmanaged(OutputSpan){};
        self.debug_mode = false; // Don't track spans in nested renders
        defer {
            self.output = old_output;
            self.spans = old_spans;
            self.debug_mode = old_debug;
        }
        errdefer self.output.deinit(self.allocator);

        try self.evalNodes(nodes);
        return self.output.toOwnedSlice(self.allocator) catch return EvalError.OutOfMemory;
    }

    /// Extract variable path from an expression (e.g., "name", "user.email", "items[0]")
    /// Returns null for complex expressions that can't be attributed to a single variable.
    fn getExprPath(self: *Evaluator, expr: *const Expr) ?[]const u8 {
        return switch (expr.*) {
            .variable => |name| name,
            .getattr => |ga| blk: {
                // Build path like "obj.attr"
                const obj_path = self.getExprPath(ga.object) orelse break :blk null;
                const path = std.fmt.allocPrint(self.ctx.arena.allocator(), "{s}.{s}", .{ obj_path, ga.attr }) catch break :blk null;
                break :blk path;
            },
            .getitem => |gi| blk: {
                // Build path like "obj[key]"
                const obj_path = self.getExprPath(gi.object) orelse break :blk null;
                const key_str = switch (gi.key.*) {
                    .integer => |i| std.fmt.allocPrint(self.ctx.arena.allocator(), "{d}", .{i}) catch break :blk null,
                    .string => |s| s,
                    else => break :blk null,
                };
                const path = std.fmt.allocPrint(self.ctx.arena.allocator(), "{s}[{s}]", .{ obj_path, key_str }) catch break :blk null;
                break :blk path;
            },
            // Filters on variables still attribute to the variable
            .filter => |f| self.getExprPath(f.value),
            // Complex expressions can't be attributed to a single variable
            else => null,
        };
    }

    pub fn evalNode(self: *Evaluator, node: *const Node) EvalError!void {
        switch (node.*) {
            .text => |t| {
                const start = self.output.items.len;
                self.output.appendSlice(self.allocator, t) catch return EvalError.OutOfMemory;
                const end = self.output.items.len;
                try self.recordSpan(start, end, .static_text);
            },
            .print => |expr| {
                const value = try self.evalExpr(expr);
                // Determine source: variable path or generic expression
                const source: SpanSource = if (self.getExprPath(expr)) |path|
                    .{ .variable = path }
                else
                    .expression;
                try self.appendValueWithSpan(value, source);
            },
            .if_stmt => |stmt| {
                for (stmt.branches) |branch| {
                    const cond = try self.evalExpr(branch.condition);
                    if (cond.isTruthy()) {
                        try self.evalNodes(branch.body);
                        return;
                    }
                }
                // No branch matched, try else
                try self.evalNodes(stmt.else_body);
            },
            .for_stmt => |stmt| {
                const iterable = try self.evalExpr(stmt.iterable);
                const items = try self.collectIterable(iterable);

                if (items.len == 0) {
                    // Empty - render else body
                    try self.evalNodes(stmt.else_body);
                    return;
                }

                // Save old loop context
                const old_loop = self.ctx.loop;
                defer self.ctx.loop = old_loop;

                // Push a new local scope for loop variable isolation (Jinja2 spec)
                try self.pushScope();
                defer self.popScope();

                // Get current depth from outer loop (if any)
                const current_depth = if (old_loop) |ol| ol.depth + 1 else 1;

                for (items, 0..) |item, item_idx| {
                    // Set loop context
                    self.setLoopContext(item_idx, items, current_depth, stmt.recursive, stmt.body, stmt.target, stmt.target2);
                    try self.setLoopTargets(stmt.target, stmt.target2, item);

                    // Check filter condition if present
                    if (stmt.filter) |filter_expr| {
                        const filter_val = try self.evalExpr(filter_expr);
                        if (!filter_val.isTruthy()) {
                            continue; // Skip this item
                        }
                    }

                    // Render body with break/continue handling
                    var should_break = false;
                    for (stmt.body) |child| {
                        self.evalNode(child) catch |err| {
                            if (err == EvalError.LoopBreak) {
                                should_break = true;
                                break;
                            } else if (err == EvalError.LoopContinue) {
                                break; // Continue to next iteration
                            } else {
                                return err;
                            }
                        };
                    }
                    if (should_break) break;
                }
            },
            .set_stmt => |stmt| {
                const value = try self.evalExpr(stmt.value);

                if (stmt.namespace) |ns_name| {
                    // Namespace assignment: ns.foo = val
                    // Use getVar to check local scopes first, then global context
                    if (self.getVar(ns_name)) |ns_val| {
                        if (ns_val == .namespace) {
                            ns_val.namespace.put(self.ctx.arena.allocator(), stmt.target, value) catch return EvalError.OutOfMemory;
                            return;
                        }
                    }
                    return EvalError.TypeError;
                } else {
                    // Use setVar for proper scope isolation
                    try self.setVar(stmt.target, value);
                }
            },
            .macro_def => |macro| {
                // Register macro in context
                self.ctx.set(macro.name, .{
                    .macro = .{
                        .name = macro.name,
                        .params = macro.params,
                        .body = macro.body,
                    },
                }) catch return EvalError.OutOfMemory;
            },
            .macro_call_stmt => |call| {
                // Evaluate macro call and append result to output
                const result = try self.callMacro(call.name, call.args, call.kwargs);
                try self.appendValue(result);
            },
            .break_stmt => {
                return EvalError.LoopBreak;
            },
            .continue_stmt => {
                return EvalError.LoopContinue;
            },
            .filter_block => |fb| {
                const content = try self.renderNodesToString(fb.body);

                // Apply each filter in the chain
                var value: TemplateInput = .{ .string = content };
                for (fb.filters) |filter_name| {
                    value = try filters.applyFilter(self, filter_name, value, &.{});
                }
                try self.appendValue(value);
            },
            .call_block => |cb| {
                // Save call body for caller() function
                const old_caller = self.caller_body;
                self.caller_body = cb.body;
                defer self.caller_body = old_caller;

                // Call the macro
                const result = try self.callMacro(cb.macro_name, cb.args, &.{});
                try self.appendValue(result);
            },
            .generation_block => |gb| {
                // HuggingFace extension: {% generation %}...{% endgeneration %}
                // Simply render the body content (acts as a pass-through marker)
                for (gb.body) |child| {
                    try self.evalNode(child);
                }
            },
            .include => |inc| {
                // {% include template_expr %}
                // Evaluate the expression to get the template string
                const template_value = try self.evalExpr(inc.template_expr);
                const template_str = switch (template_value) {
                    .string => |s| s,
                    .none => {
                        // Undefined variable - set path for better error message
                        if (inc.template_expr.* == .variable) {
                            self.ctx.setUndefinedPath(inc.template_expr.variable);
                        }
                        return EvalError.IncludeTypeError;
                    },
                    else => return EvalError.IncludeTypeError,
                };

                // Parse and render the included template in the current context
                // Use the same arena allocator for the sub-template
                const arena_alloc = self.ctx.arena.allocator();

                // Tokenize
                var lexer = lexer_mod.Lexer.init(arena_alloc, template_str);
                const tokens = lexer.tokenize() catch return EvalError.IncludeError;

                // Parse
                var parser = parser_mod.Parser.init(arena_alloc, tokens);
                const nodes = parser.parse() catch return EvalError.IncludeError;

                // Render the included nodes with the current context
                for (nodes) |child| {
                    try self.evalNode(child);
                }
            },
        }
    }

    /// Build a human-readable path string from an expression AST for error messages.
    /// Example: variable "doc" with getattr "page_content" -> "doc.page_content"
    fn buildExprPath(self: *Evaluator, expr: *const Expr) ?[]const u8 {
        return switch (expr.*) {
            .variable => |name| self.ctx.allocator.dupe(u8, name) catch null,
            .getattr => |ga| {
                const base = self.buildExprPath(ga.object) orelse return null;
                defer self.ctx.allocator.free(base);
                return std.fmt.allocPrint(self.ctx.allocator, "{s}.{s}", .{ base, ga.attr }) catch null;
            },
            .getitem => |gi| {
                const base = self.buildExprPath(gi.object) orelse return null;
                defer self.ctx.allocator.free(base);
                // Try to get key as string for better error messages
                const key_str = switch (gi.key.*) {
                    .string => |s| s,
                    .integer => |i| std.fmt.allocPrint(self.ctx.allocator, "{d}", .{i}) catch return null,
                    else => "?",
                };
                const is_string_key = gi.key.* == .string;
                const result = if (is_string_key)
                    std.fmt.allocPrint(self.ctx.allocator, "{s}['{s}']", .{ base, key_str }) catch null
                else
                    std.fmt.allocPrint(self.ctx.allocator, "{s}[{s}]", .{ base, key_str }) catch null;
                if (gi.key.* == .integer) self.ctx.allocator.free(key_str);
                return result;
            },
            else => null,
        };
    }

    pub fn evalExpr(self: *Evaluator, expr: *const Expr) EvalError!TemplateInput {
        switch (expr.*) {
            .string => |s| {
                // Process escape sequences (like \n, \", \\)
                const processed = processEscapes(self.ctx.arena.allocator(), s) catch return EvalError.OutOfMemory;
                return .{ .string = processed };
            },
            .integer => |int_val| return .{ .integer = int_val },
            .float => |f| return .{ .float = f },
            .boolean => |b| return .{ .boolean = b },
            .none => return .none,

            .variable => |name| {
                // Handle special 'loop' variable
                if (std.mem.eql(u8, name, "loop")) {
                    return self.getLoopValue();
                }
                // Check local scopes first, then global context
                if (self.getVar(name)) |value| {
                    return value;
                }
                // In strict mode, undefined variables raise an error
                if (self.ctx.strict) {
                    self.ctx.setUndefinedPath(name);
                    return EvalError.UndefinedVariable;
                }
                // In lenient mode, return none (like Jinja's Undefined)
                return .none;
            },

            .getattr => |ga| {
                const obj = try self.evalExpr(ga.object);
                return self.getAttributeWithPath(obj, ga.attr, expr);
            },

            .getitem => |gi| {
                const obj = try self.evalExpr(gi.object);
                const key = try self.evalExpr(gi.key);
                return self.getItemWithPath(obj, key, expr);
            },

            .slice => |sl| {
                const obj = try self.evalExpr(sl.object);
                return self.getSlice(obj, sl.start, sl.stop, sl.step);
            },

            .binop => |bo| {
                return self.evalBinOp(bo.op, bo.left, bo.right);
            },

            .unaryop => |uo| {
                const value = try self.evalExpr(uo.operand);
                return switch (uo.op) {
                    .not => .{ .boolean = !value.isTruthy() },
                    .neg => switch (value) {
                        .integer => |int_val| .{ .integer = -int_val },
                        .float => |f| .{ .float = -f },
                        else => EvalError.TypeError,
                    },
                    .pos => switch (value) {
                        .integer => value,
                        .float => value,
                        else => EvalError.TypeError,
                    },
                };
            },

            .call => |c| {
                return self.evalCall(c.func, c.args);
            },

            .filter => |f| {
                // For 'default' filter, catch UndefinedVariable and pass .none
                if (std.mem.eql(u8, f.name, "default") or std.mem.eql(u8, f.name, "d")) {
                    const value = self.evalExpr(f.value) catch |err| switch (err) {
                        EvalError.UndefinedVariable => .none,
                        else => return err,
                    };
                    return filters.applyFilter(self, f.name, value, f.args);
                }
                const value = try self.evalExpr(f.value);
                return filters.applyFilter(self, f.name, value, f.args);
            },

            .test_expr => |te| {
                // For 'defined'/'undefined' tests, catch UndefinedVariable/KeyError and pass .none
                // This allows patterns like: {% if item['key'] is defined %}
                // where the key may not exist in the dict
                if (std.mem.eql(u8, te.name, "defined") or std.mem.eql(u8, te.name, "undefined")) {
                    const value = self.evalExpr(te.value) catch |err| switch (err) {
                        EvalError.UndefinedVariable, EvalError.KeyError => .none,
                        else => return err,
                    };
                    const result = try predicates.applyTest(self, te.name, value, te.args);
                    return .{ .boolean = if (te.negated) !result else result };
                }
                const value = try self.evalExpr(te.value);
                const result = try predicates.applyTest(self, te.name, value, te.args);
                return .{ .boolean = if (te.negated) !result else result };
            },

            .conditional => |c| {
                const cond = try self.evalExpr(c.test_val);
                if (cond.isTruthy()) {
                    return self.evalExpr(c.true_val);
                } else {
                    return self.evalExpr(c.false_val);
                }
            },

            .list => |items| {
                const arena = self.ctx.arena.allocator();
                var arr = std.ArrayListUnmanaged(TemplateInput){};
                for (items) |item| {
                    const item_value = try self.evalExpr(item);
                    arr.append(arena, item_value) catch return EvalError.OutOfMemory;
                }
                return .{ .array = arr.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
            },

            .dict => |pairs| {
                var map = std.StringHashMapUnmanaged(TemplateInput){};
                for (pairs) |pair| {
                    const key_val = try self.evalExpr(pair.key);
                    const key_str = key_val.asString(self.ctx.arena.allocator()) catch return EvalError.OutOfMemory;
                    const value = try self.evalExpr(pair.value);
                    map.put(self.ctx.arena.allocator(), key_str, value) catch return EvalError.OutOfMemory;
                }
                return .{ .map = map };
            },

            .namespace_call => |args| {
                const ns = self.ctx.arena.allocator().create(std.StringHashMapUnmanaged(TemplateInput)) catch return EvalError.OutOfMemory;
                ns.* = .{};
                for (args) |arg| {
                    const value = try self.evalExpr(arg.value);
                    ns.put(self.ctx.arena.allocator(), arg.name, value) catch return EvalError.OutOfMemory;
                }
                return .{ .namespace = ns };
            },

            .macro_call => |call| {
                return self.callMacro(call.name, call.args, call.kwargs);
            },
        }
    }

    fn getLoopValue(self: *Evaluator) TemplateInput {
        // Return loop_ctx for loop.cycle() support
        if (self.ctx.loop) |*lp| {
            // Return pointer to the loop context
            return .{ .loop_ctx = lp };
        }
        return .none;
    }

    fn collectIterable(self: *Evaluator, iterable: TemplateInput) EvalError![]const TemplateInput {
        const arena = self.ctx.arena.allocator();
        return switch (iterable) {
            .array => |arr| arr,
            .string => |s| blk: {
                // Iterate over characters
                var chars = std.ArrayListUnmanaged(TemplateInput){};
                for (s) |c| {
                    const char_str = arena.alloc(u8, 1) catch return EvalError.OutOfMemory;
                    char_str[0] = c;
                    chars.append(arena, .{ .string = char_str }) catch return EvalError.OutOfMemory;
                }
                break :blk chars.toOwnedSlice(arena) catch return EvalError.OutOfMemory;
            },
            else => return EvalError.TypeError,
        };
    }

    fn setLoopTargets(self: *Evaluator, target: []const u8, target2: ?[]const u8, item: TemplateInput) EvalError!void {
        // Loop targets should be set in the local scope
        if (target2) |t2| {
            if (item == .array and item.array.len >= 2) {
                try self.setVar(target, item.array[0]);
                try self.setVar(t2, item.array[1]);
            } else {
                try self.setVar(target, item);
            }
        } else {
            try self.setVar(target, item);
        }
    }

    fn setLoopContext(
        self: *Evaluator,
        index: usize,
        items: []const TemplateInput,
        depth: usize,
        recursive: bool,
        body: []const *const Node,
        target: []const u8,
        target2: ?[]const u8,
    ) void {
        self.ctx.loop = .{
            .index0 = index,
            .index = index + 1,
            .first = index == 0,
            .last = index == items.len - 1,
            .length = items.len,
            .revindex = items.len - index,
            .revindex0 = items.len - index - 1,
            .previtem = if (index > 0) items[index - 1] else null,
            .nextitem = if (index + 1 < items.len) items[index + 1] else null,
            .depth = depth,
            .depth0 = depth - 1,
            .recursive_body = if (recursive) body else null,
            .recursive_target = if (recursive) target else null,
            .recursive_target2 = if (recursive) target2 else null,
        };
    }

    fn getAttribute(self: *Evaluator, obj: TemplateInput, attr: []const u8) EvalError!TemplateInput {
        switch (obj) {
            .none => {
                // Accessing attribute on undefined
                if (self.ctx.strict) return EvalError.UndefinedVariable;
                return .none;
            },
            .map => |m| {
                if (m.get(attr)) |value| return value;
                if (self.ctx.strict) return EvalError.KeyError;
                return .none;
            },
            .namespace => |ns| {
                if (ns.get(attr)) |value| return value;
                if (self.ctx.strict) return EvalError.KeyError;
                return .none;
            },
            .array => |arr| {
                // Array attributes
                if (std.mem.eql(u8, attr, "length")) {
                    return .{ .integer = @intCast(arr.len) };
                }
            },
            .string => {
                // String method access handled in call
            },
            .cycler => |c| {
                // Cycler properties
                if (std.mem.eql(u8, attr, "current")) {
                    if (c.items.len == 0) return .none;
                    return c.items[c.index];
                }
            },
            .loop_ctx => |lp| {
                // Loop context properties
                if (std.mem.eql(u8, attr, "index0")) {
                    return .{ .integer = @intCast(lp.index0) };
                }
                if (std.mem.eql(u8, attr, "index")) {
                    return .{ .integer = @intCast(lp.index) };
                }
                if (std.mem.eql(u8, attr, "first")) {
                    return .{ .boolean = lp.first };
                }
                if (std.mem.eql(u8, attr, "last")) {
                    return .{ .boolean = lp.last };
                }
                if (std.mem.eql(u8, attr, "length")) {
                    return .{ .integer = @intCast(lp.length) };
                }
                if (std.mem.eql(u8, attr, "revindex")) {
                    return .{ .integer = @intCast(lp.revindex) };
                }
                if (std.mem.eql(u8, attr, "revindex0")) {
                    return .{ .integer = @intCast(lp.revindex0) };
                }
                if (std.mem.eql(u8, attr, "previtem")) {
                    return lp.previtem orelse .none;
                }
                if (std.mem.eql(u8, attr, "nextitem")) {
                    return lp.nextitem orelse .none;
                }
                if (std.mem.eql(u8, attr, "depth")) {
                    return .{ .integer = @intCast(lp.depth) };
                }
                if (std.mem.eql(u8, attr, "depth0")) {
                    return .{ .integer = @intCast(lp.depth0) };
                }
                // cycle is a method, handled in callMethod
            },
            else => {},
        }
        return EvalError.TypeError;
    }

    fn normalizeIndex(len: usize, index: i64) EvalError!usize {
        if (index < 0) {
            const abs_i: usize = @intCast(-index);
            if (abs_i > len) return EvalError.IndexOutOfBounds;
            return len - abs_i;
        }
        const idx: usize = @intCast(index);
        if (idx >= len) return EvalError.IndexOutOfBounds;
        return idx;
    }

    fn getItem(self: *Evaluator, obj: TemplateInput, key: TemplateInput) EvalError!TemplateInput {
        switch (obj) {
            .none => {
                if (self.ctx.strict) return EvalError.UndefinedVariable;
                return .none;
            },
            .array => |arr| {
                switch (key) {
                    .integer => |int_val| {
                        const idx = try normalizeIndex(arr.len, int_val);
                        return arr[idx];
                    },
                    else => return EvalError.TypeError,
                }
            },
            .map => |m| {
                const key_str = switch (key) {
                    .string => |s| s,
                    else => return EvalError.TypeError,
                };
                if (m.get(key_str)) |value| return value;
                // In strict mode, missing keys raise KeyError
                if (self.ctx.strict) return EvalError.KeyError;
                // In lenient mode, return none for missing keys (Jinja2 behavior)
                return .none;
            },
            .string => |s| {
                switch (key) {
                    .integer => |int_val| {
                        const idx = try normalizeIndex(s.len, int_val);
                        return .{ .string = s[idx .. idx + 1] };
                    },
                    else => return EvalError.TypeError,
                }
            },
            else => return EvalError.TypeError,
        }
    }

    /// getAttribute with error path tracking for better error messages.
    fn getAttributeWithPath(self: *Evaluator, obj: TemplateInput, attr: []const u8, expr: *const Expr) EvalError!TemplateInput {
        return self.getAttribute(obj, attr) catch |err| {
            if (self.ctx.strict and (err == EvalError.UndefinedVariable or err == EvalError.KeyError)) {
                if (self.buildExprPath(expr)) |path| {
                    // Free old path if any, then set new one
                    if (self.ctx.undefined_path) |old| self.ctx.allocator.free(old);
                    self.ctx.undefined_path = path;
                }
            }
            return err;
        };
    }

    /// getItem with error path tracking for better error messages.
    fn getItemWithPath(self: *Evaluator, obj: TemplateInput, key: TemplateInput, expr: *const Expr) EvalError!TemplateInput {
        return self.getItem(obj, key) catch |err| {
            if (self.ctx.strict and (err == EvalError.UndefinedVariable or err == EvalError.KeyError)) {
                if (self.buildExprPath(expr)) |path| {
                    if (self.ctx.undefined_path) |old| self.ctx.allocator.free(old);
                    self.ctx.undefined_path = path;
                }
            }
            return err;
        };
    }

    const SliceSpec = struct {
        start: i64,
        stop: i64,
        step: i64,
    };

    fn parseSliceSpec(
        self: *Evaluator,
        len: i64,
        start_expr: ?*const Expr,
        stop_expr: ?*const Expr,
        step_expr: ?*const Expr,
    ) EvalError!SliceSpec {
        var start: i64 = 0;
        var stop: i64 = len;
        var step: i64 = 1;

        if (start_expr) |e| {
            const value = try self.evalExpr(e);
            start = if (value == .integer) value.integer else return EvalError.TypeError;
        }
        if (stop_expr) |e| {
            const value = try self.evalExpr(e);
            stop = if (value == .integer) value.integer else return EvalError.TypeError;
        }
        if (step_expr) |e| {
            const value = try self.evalExpr(e);
            step = if (value == .integer) value.integer else return EvalError.TypeError;
        }

        if (start < 0) start = @max(0, len + start);
        if (stop < 0) stop = @max(0, len + stop);
        start = @min(start, len);
        stop = @min(stop, len);

        if (step == 0) return EvalError.InvalidOperation;

        if (step < 0) {
            if (start_expr == null) start = len - 1;
            if (stop_expr == null) stop = -1;
        }

        return .{ .start = start, .stop = stop, .step = step };
    }

    fn getSlice(self: *Evaluator, obj: TemplateInput, start_expr: ?*const Expr, stop_expr: ?*const Expr, step_expr: ?*const Expr) EvalError!TemplateInput {
        switch (obj) {
            .array => |arr| {
                const len: i64 = @intCast(arr.len);
                const spec = try self.parseSliceSpec(len, start_expr, stop_expr, step_expr);

                const arena = self.ctx.arena.allocator();
                var result = std.ArrayListUnmanaged(TemplateInput){};

                if (spec.step > 0) {
                    var slice_idx = spec.start;
                    while (slice_idx < spec.stop) : (slice_idx += spec.step) {
                        result.append(arena, arr[@intCast(slice_idx)]) catch return EvalError.OutOfMemory;
                    }
                } else {
                    // Negative step - reverse iteration
                    // For [::-1], start defaults to len-1, stop defaults to -1
                    var slice_idx = spec.start;
                    while (slice_idx > spec.stop) : (slice_idx += spec.step) {
                        if (slice_idx >= 0 and slice_idx < len) {
                            result.append(arena, arr[@intCast(slice_idx)]) catch return EvalError.OutOfMemory;
                        }
                    }
                }

                return .{ .array = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
            },
            .string => |s| {
                const len: i64 = @intCast(s.len);
                const spec = try self.parseSliceSpec(len, start_expr, stop_expr, step_expr);

                const arena = self.ctx.arena.allocator();
                var result = std.ArrayListUnmanaged(u8){};

                if (spec.step > 0) {
                    var slice_idx = spec.start;
                    while (slice_idx < spec.stop) : (slice_idx += spec.step) {
                        result.append(arena, s[@intCast(slice_idx)]) catch return EvalError.OutOfMemory;
                    }
                } else {
                    var slice_idx = spec.start;
                    while (slice_idx > spec.stop) : (slice_idx += spec.step) {
                        if (slice_idx >= 0 and slice_idx < len) {
                            result.append(arena, s[@intCast(slice_idx)]) catch return EvalError.OutOfMemory;
                        }
                    }
                }

                return .{ .string = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
            },
            else => return EvalError.TypeError,
        }
    }

    fn evalBinOp(self: *Evaluator, op: BinOp, left_expr: *const Expr, right_expr: *const Expr) EvalError!TemplateInput {
        // Short-circuit evaluation for and/or (Python-style: return value, not boolean)
        // x and y: return x if x is falsy, else y
        // x or y: return x if x is truthy, else y
        if (op == .@"and") {
            const left = try self.evalExpr(left_expr);
            if (!left.isTruthy()) return left;
            return try self.evalExpr(right_expr);
        }
        if (op == .@"or") {
            const left = try self.evalExpr(left_expr);
            if (left.isTruthy()) return left;
            return try self.evalExpr(right_expr);
        }

        const left = try self.evalExpr(left_expr);
        const right = try self.evalExpr(right_expr);

        return switch (op) {
            .add => self.evalAdd(left, right),
            .sub => self.evalSub(left, right),
            .mul => self.evalMul(left, right),
            .div => self.evalDiv(left, right),
            .floordiv => self.evalFloorDiv(left, right),
            .mod => self.evalMod(left, right),
            .pow => self.evalPow(left, right),
            .eq => .{ .boolean = left.eql(right) },
            .ne => .{ .boolean = !left.eql(right) },
            .lt => self.evalCompare(left, right, .lt),
            .gt => self.evalCompare(left, right, .gt),
            .le => self.evalCompare(left, right, .le),
            .ge => self.evalCompare(left, right, .ge),
            .in => self.evalIn(left, right),
            .not_in => blk: {
                const result = try self.evalIn(left, right);
                break :blk .{ .boolean = !result.boolean };
            },
            .concat => self.evalConcat(left, right),
            else => EvalError.InvalidOperation,
        };
    }

    fn evalAdd(self: *Evaluator, left: TemplateInput, right: TemplateInput) EvalError!TemplateInput {
        switch (left) {
            .integer => |l| switch (right) {
                .integer => |r| return .{ .integer = l + r },
                .float => |r| return .{ .float = @as(f64, @floatFromInt(l)) + r },
                else => {},
            },
            .float => |l| switch (right) {
                .integer => |r| return .{ .float = l + @as(f64, @floatFromInt(r)) },
                .float => |r| return .{ .float = l + r },
                else => {},
            },
            .string => |l| switch (right) {
                .string => |r| {
                    const result = std.mem.concat(self.ctx.arena.allocator(), u8, &.{ l, r }) catch return EvalError.OutOfMemory;
                    return .{ .string = result };
                },
                else => {},
            },
            .array => |l| switch (right) {
                .array => |r| {
                    const result = std.mem.concat(self.ctx.arena.allocator(), TemplateInput, &.{ l, r }) catch return EvalError.OutOfMemory;
                    return .{ .array = result };
                },
                else => {},
            },
            else => {},
        }
        return EvalError.TypeError;
    }

    fn evalSub(self: *Evaluator, left: TemplateInput, right: TemplateInput) EvalError!TemplateInput {
        _ = self;
        switch (left) {
            .integer => |l| switch (right) {
                .integer => |r| return .{ .integer = l - r },
                .float => |r| return .{ .float = @as(f64, @floatFromInt(l)) - r },
                else => {},
            },
            .float => |l| switch (right) {
                .integer => |r| return .{ .float = l - @as(f64, @floatFromInt(r)) },
                .float => |r| return .{ .float = l - r },
                else => {},
            },
            else => {},
        }
        return EvalError.TypeError;
    }

    fn evalMul(self: *Evaluator, left: TemplateInput, right: TemplateInput) EvalError!TemplateInput {
        switch (left) {
            .integer => |l| switch (right) {
                .integer => |r| return .{ .integer = l * r },
                .float => |r| return .{ .float = @as(f64, @floatFromInt(l)) * r },
                else => {},
            },
            .float => |l| switch (right) {
                .integer => |r| return .{ .float = l * @as(f64, @floatFromInt(r)) },
                .float => |r| return .{ .float = l * r },
                else => {},
            },
            .string => |s| switch (right) {
                .integer => |n| {
                    // String repeat: "ab" * 3 = "ababab"
                    if (n <= 0) return .{ .string = "" };
                    const count: usize = @intCast(n);
                    const result = self.ctx.arena.allocator().alloc(u8, s.len * count) catch return EvalError.OutOfMemory;
                    for (0..count) |rep_idx| {
                        @memcpy(result[rep_idx * s.len .. (rep_idx + 1) * s.len], s);
                    }
                    return .{ .string = result };
                },
                else => {},
            },
            else => {},
        }
        return EvalError.TypeError;
    }

    fn evalDiv(self: *Evaluator, left: TemplateInput, right: TemplateInput) EvalError!TemplateInput {
        _ = self;
        const l_num: f64 = switch (left) {
            .integer => |int_val| @floatFromInt(int_val),
            .float => |f| f,
            else => return EvalError.TypeError,
        };
        const r_num: f64 = switch (right) {
            .integer => |int_val| @floatFromInt(int_val),
            .float => |f| f,
            else => return EvalError.TypeError,
        };
        if (r_num == 0) return EvalError.DivisionByZero;
        return .{ .float = l_num / r_num };
    }

    fn evalMod(self: *Evaluator, left: TemplateInput, right: TemplateInput) EvalError!TemplateInput {
        _ = self;
        switch (left) {
            .integer => |l| switch (right) {
                .integer => |r| {
                    if (r == 0) return EvalError.DivisionByZero;
                    return .{ .integer = @mod(l, r) };
                },
                else => {},
            },
            else => {},
        }
        return EvalError.TypeError;
    }

    fn evalFloorDiv(self: *Evaluator, left: TemplateInput, right: TemplateInput) EvalError!TemplateInput {
        _ = self;
        switch (left) {
            .integer => |l| switch (right) {
                .integer => |r| {
                    if (r == 0) return EvalError.DivisionByZero;
                    return .{ .integer = @divFloor(l, r) };
                },
                .float => |r| {
                    if (r == 0) return EvalError.DivisionByZero;
                    return .{ .integer = @intFromFloat(@floor(@as(f64, @floatFromInt(l)) / r)) };
                },
                else => {},
            },
            .float => |l| switch (right) {
                .integer => |r| {
                    if (r == 0) return EvalError.DivisionByZero;
                    return .{ .integer = @intFromFloat(@floor(l / @as(f64, @floatFromInt(r)))) };
                },
                .float => |r| {
                    if (r == 0) return EvalError.DivisionByZero;
                    return .{ .integer = @intFromFloat(@floor(l / r)) };
                },
                else => {},
            },
            else => {},
        }
        return EvalError.TypeError;
    }

    fn evalPow(self: *Evaluator, left: TemplateInput, right: TemplateInput) EvalError!TemplateInput {
        _ = self;
        const l_num: f64 = switch (left) {
            .integer => |int_val| @floatFromInt(int_val),
            .float => |f| f,
            else => return EvalError.TypeError,
        };
        const r_num: f64 = switch (right) {
            .integer => |int_val| @floatFromInt(int_val),
            .float => |f| f,
            else => return EvalError.TypeError,
        };
        const result = std.math.pow(f64, l_num, r_num);
        // Return integer if both operands were integers and result is whole
        if (left == .integer and right == .integer and right.integer >= 0) {
            if (@floor(result) == result and result <= @as(f64, @floatFromInt(std.math.maxInt(i64)))) {
                return .{ .integer = @intFromFloat(result) };
            }
        }
        return .{ .float = result };
    }

    fn evalCompare(self: *Evaluator, left: TemplateInput, right: TemplateInput, op: enum { lt, gt, le, ge }) EvalError!TemplateInput {
        _ = self;
        const l_num = left.asNumber() orelse return EvalError.TypeError;
        const r_num = right.asNumber() orelse return EvalError.TypeError;
        const result = switch (op) {
            .lt => l_num < r_num,
            .gt => l_num > r_num,
            .le => l_num <= r_num,
            .ge => l_num >= r_num,
        };
        return .{ .boolean = result };
    }

    fn evalIn(self: *Evaluator, needle: TemplateInput, haystack: TemplateInput) EvalError!TemplateInput {
        switch (haystack) {
            .string => |s| {
                const needle_str = switch (needle) {
                    .string => |ns| ns,
                    else => return EvalError.TypeError,
                };
                return .{ .boolean = std.mem.indexOf(u8, s, needle_str) != null };
            },
            .array => |arr| {
                for (arr) |item| {
                    if (needle.eql(item)) return .{ .boolean = true };
                }
                return .{ .boolean = false };
            },
            .map => |m| {
                const key = switch (needle) {
                    .string => |s| s,
                    else => return EvalError.TypeError,
                };
                return .{ .boolean = m.contains(key) };
            },
            .none => {
                // In strict mode, 'x in undefined' is an error
                if (self.ctx.strict) return EvalError.UndefinedVariable;
                return .{ .boolean = false };
            },
            else => return EvalError.TypeError,
        }
    }

    fn evalConcat(self: *Evaluator, left: TemplateInput, right: TemplateInput) EvalError!TemplateInput {
        const l_str = left.asString(self.ctx.arena.allocator()) catch return EvalError.OutOfMemory;
        const r_str = right.asString(self.ctx.arena.allocator()) catch return EvalError.OutOfMemory;
        const result = std.mem.concat(self.ctx.arena.allocator(), u8, &.{ l_str, r_str }) catch return EvalError.OutOfMemory;
        return .{ .string = result };
    }

    fn evalCall(self: *Evaluator, func_expr: *const Expr, args: []const *const Expr) EvalError!TemplateInput {
        // Method calls are getattr expressions
        switch (func_expr.*) {
            .getattr => |ga| {
                const obj = try self.evalExpr(ga.object);
                return methods.callMethod(self, obj, ga.attr, args);
            },
            .variable => |name| {
                // Check for loop() recursive call
                if (std.mem.eql(u8, name, "loop")) {
                    if (self.ctx.loop) |lp| {
                        if (lp.recursive_body) |body| {
                            // Recursive loop call - evaluate with new items
                            if (args.len < 1) return EvalError.TypeError;
                            const new_iterable = try self.evalExpr(args[0]);
                            return self.evalRecursiveLoop(new_iterable, body, lp.recursive_target.?, lp.recursive_target2, lp.depth);
                        }
                    }
                }
                // Check if it's a callable value first
                if (self.ctx.get(name)) |val| {
                    switch (val) {
                        .macro => return self.callMacro(name, args, &.{}),
                        .joiner => |joiner| {
                            // joiner() returns empty on first call, separator afterwards
                            if (joiner.called) {
                                return .{ .string = joiner.separator };
                            } else {
                                joiner.called = true;
                                return .{ .string = "" };
                            }
                        },
                        else => {},
                    }
                }
                // Built-in function call
                return functions.callFunction(self, name, args);
            },
            else => return EvalError.InvalidOperation,
        }
    }

    fn evalRecursiveLoop(self: *Evaluator, iterable: TemplateInput, body: []const *const ast.Node, target: []const u8, target2: ?[]const u8, parent_depth: usize) EvalError!TemplateInput {
        const items = try self.collectIterable(iterable);

        if (items.len == 0) return .{ .string = "" };

        // Save old loop context
        const old_loop = self.ctx.loop;
        defer self.ctx.loop = old_loop;

        // Push a new local scope for loop variable isolation (Jinja2 spec)
        try self.pushScope();
        defer self.popScope();

        const new_depth = parent_depth + 1;

        const old_output = self.output;
        self.output = std.ArrayListUnmanaged(u8){};
        errdefer self.output.deinit(self.allocator);
        defer self.output = old_output;

        for (items, 0..) |item, item_idx| {
            self.setLoopContext(item_idx, items, new_depth, true, body, target, target2);
            try self.setLoopTargets(target, target2, item);

            // Render body
            for (body) |child| {
                self.evalNode(child) catch |err| {
                    if (err == EvalError.LoopBreak) break;
                    if (err == EvalError.LoopContinue) break;
                    // Restore and return error
                    return err;
                };
            }
        }

        const result = self.output.toOwnedSlice(self.allocator) catch return EvalError.OutOfMemory;
        return .{ .string = result };
    }

    fn callMacro(self: *Evaluator, name: []const u8, args: []const *const Expr, kwargs: []const Expr.NamespaceArg) EvalError!TemplateInput {
        // Look up macro
        const macro_val = self.ctx.get(name) orelse return EvalError.UndefinedVariable;
        const macro = switch (macro_val) {
            .macro => |m| m,
            else => return EvalError.TypeError,
        };

        // Create a new scope for macro execution
        // Save old variable values
        var old_vars = std.StringHashMapUnmanaged(?TemplateInput){};
        defer old_vars.deinit(self.ctx.arena.allocator());

        // Bind parameters
        for (macro.params, 0..) |param, param_idx| {
            // Save old value if exists
            old_vars.put(self.ctx.arena.allocator(), param.name, self.ctx.get(param.name)) catch return EvalError.OutOfMemory;

            // Check kwargs first
            var found_kwarg = false;
            for (kwargs) |kwarg| {
                if (std.mem.eql(u8, kwarg.name, param.name)) {
                    const value = try self.evalExpr(kwarg.value);
                    self.ctx.set(param.name, value) catch return EvalError.OutOfMemory;
                    found_kwarg = true;
                    break;
                }
            }

            if (!found_kwarg) {
                // Use positional arg if available
                if (param_idx < args.len) {
                    const value = try self.evalExpr(args[param_idx]);
                    self.ctx.set(param.name, value) catch return EvalError.OutOfMemory;
                } else if (param.default) |default_expr| {
                    // Use default value
                    const value = try self.evalExpr(default_expr);
                    self.ctx.set(param.name, value) catch return EvalError.OutOfMemory;
                } else {
                    // Missing required parameter
                    self.ctx.set(param.name, .none) catch return EvalError.OutOfMemory;
                }
            }
        }

        const temp_result = try self.renderNodesToString(macro.body);
        defer self.allocator.free(temp_result);
        const result = self.ctx.arena.allocator().dupe(u8, temp_result) catch return EvalError.OutOfMemory;

        // Restore old variable values
        var it = old_vars.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.*) |old_val| {
                self.ctx.set(entry.key_ptr.*, old_val) catch return EvalError.OutOfMemory;
            }
        }

        return .{ .string = result };
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "TemplateParser - init and deinit" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    try std.testing.expect(parser.variables.count() == 0);
    try std.testing.expect(parser.loop == null);
    try std.testing.expectEqual(false, parser.strict);
}

test "TemplateParser - initStrict" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.initStrict(allocator);
    defer parser.deinit();

    try std.testing.expectEqual(true, parser.strict);
    try std.testing.expect(parser.variables.count() == 0);
}

test "TemplateParser - set and get string" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    try parser.set("name", .{ .string = "Alice" });
    const value = parser.get("name");
    try std.testing.expect(value != null);
    try std.testing.expectEqual(TemplateInput{ .string = "Alice" }, value.?);
}

test "TemplateParser - set and get integer" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    try parser.set("count", .{ .integer = 42 });
    const value = parser.get("count");
    try std.testing.expect(value != null);
    try std.testing.expectEqual(TemplateInput{ .integer = 42 }, value.?);
}

test "TemplateParser - set and get boolean" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    try parser.set("enabled", .{ .boolean = true });
    const value = parser.get("enabled");
    try std.testing.expect(value != null);
    try std.testing.expectEqual(TemplateInput{ .boolean = true }, value.?);
}

test "TemplateParser - get non-existent variable" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    const value = parser.get("nonexistent");
    try std.testing.expect(value == null);
}

test "set overwrite variable" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    try parser.set("value", .{ .integer = 10 });
    try parser.set("value", .{ .integer = 20 });
    const value = parser.get("value");
    try std.testing.expect(value != null);
    try std.testing.expectEqual(TemplateInput{ .integer = 20 }, value.?);
}

test "set multiple variables" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    try parser.set("name", .{ .string = "Alice" });
    try parser.set("age", .{ .integer = 30 });
    try parser.set("active", .{ .boolean = true });

    try std.testing.expect(parser.variables.count() == 3);

    const name = parser.get("name");
    try std.testing.expect(name != null);
    try std.testing.expectEqualStrings("Alice", name.?.string);

    const age = parser.get("age");
    try std.testing.expect(age != null);
    try std.testing.expectEqual(@as(i64, 30), age.?.integer);

    const active = parser.get("active");
    try std.testing.expect(active != null);
    try std.testing.expectEqual(true, active.?.boolean);
}

test "Evaluator - init and deinit" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    try std.testing.expect(evaluator.output.items.len == 0);
    try std.testing.expect(evaluator.caller_body == null);
}

// ============================================================================
// Tests for Evaluator.render
// ============================================================================

test "Evaluator.render - empty nodes" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const nodes: []const *const Node = &.{};
    const result = try evaluator.render(nodes);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("", result);
}

test "Evaluator.render - single text node" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const text_node = Node{ .text = "Hello, World!" };
    const nodes = [_]*const Node{&text_node};

    const result = try evaluator.render(&nodes);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello, World!", result);
}

test "Evaluator.render - multiple text nodes" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const node1 = Node{ .text = "Hello, " };
    const node2 = Node{ .text = "World!" };
    const nodes = [_]*const Node{ &node1, &node2 };

    const result = try evaluator.render(&nodes);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello, World!", result);
}

test "Evaluator.render - print node with variable" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    try parser.set("name", .{ .string = "Alice" });

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const var_expr = Expr{ .variable = "name" };
    const print_node = Node{ .print = &var_expr };
    const nodes = [_]*const Node{&print_node};

    const result = try evaluator.render(&nodes);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Alice", result);
}

// ============================================================================
// Tests for Evaluator.renderNodesToString
// ============================================================================

test "Evaluator.renderNodesToString - basic text" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const node = Node{ .text = "test" };
    const nodes = [_]*const Node{&node};

    const result = try evaluator.renderNodesToString(&nodes);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("test", result);
}

test "Evaluator.renderNodesToString - preserves original output" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    // Add some content to original output
    try evaluator.output.appendSlice(allocator, "original");

    const node = Node{ .text = "new" };
    const nodes = [_]*const Node{&node};

    const result = try evaluator.renderNodesToString(&nodes);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("new", result);
    try std.testing.expectEqualStrings("original", evaluator.output.items);
}

// ============================================================================
// Tests for Evaluator.evalNode
// ============================================================================

test "Evaluator.evalNode - text node" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const node = Node{ .text = "Hello" };
    try evaluator.evalNode(&node);

    try std.testing.expectEqualStrings("Hello", evaluator.output.items);
}

test "Evaluator.evalNode - print node" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    try parser.set("value", .{ .integer = 42 });

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const expr = Expr{ .variable = "value" };
    const node = Node{ .print = &expr };
    try evaluator.evalNode(&node);

    try std.testing.expectEqualStrings("42", evaluator.output.items);
}

test "Evaluator.evalNode - if statement true branch" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const cond_expr = Expr{ .boolean = true };
    const text_node = Node{ .text = "yes" };
    const body_nodes = [_]*const Node{&text_node};

    const branch = Node.IfBranch{
        .condition = &cond_expr,
        .body = &body_nodes,
    };
    const branches = [_]Node.IfBranch{branch};

    const node = Node{ .if_stmt = .{
        .branches = &branches,
        .else_body = &.{},
    } };

    try evaluator.evalNode(&node);
    try std.testing.expectEqualStrings("yes", evaluator.output.items);
}

test "Evaluator.evalNode - if statement false branch with else" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const cond_expr = Expr{ .boolean = false };
    const empty_body: []const *const Node = &.{};

    const branch = Node.IfBranch{
        .condition = &cond_expr,
        .body = empty_body,
    };
    const branches = [_]Node.IfBranch{branch};

    const else_text = Node{ .text = "no" };
    const else_body = [_]*const Node{&else_text};

    const node = Node{ .if_stmt = .{
        .branches = &branches,
        .else_body = &else_body,
    } };

    try evaluator.evalNode(&node);
    try std.testing.expectEqualStrings("no", evaluator.output.items);
}

test "Evaluator.evalNode - for statement basic" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    const items = [_]TemplateInput{
        .{ .string = "a" },
        .{ .string = "b" },
    };
    try parser.set("items", .{ .array = &items });

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const iterable_expr = Expr{ .variable = "items" };
    const var_expr = Expr{ .variable = "x" };
    const print_node = Node{ .print = &var_expr };
    const body = [_]*const Node{&print_node};

    const node = Node{ .for_stmt = .{
        .target = "x",
        .target2 = null,
        .iterable = &iterable_expr,
        .filter = null,
        .body = &body,
        .else_body = &.{},
        .recursive = false,
    } };

    try evaluator.evalNode(&node);
    try std.testing.expectEqualStrings("ab", evaluator.output.items);
}

test "Evaluator.evalNode - for statement with empty iterable renders else" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    const items: []const TemplateInput = &.{};
    try parser.set("items", .{ .array = items });

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const iterable_expr = Expr{ .variable = "items" };
    const var_expr = Expr{ .variable = "x" };
    const print_node = Node{ .print = &var_expr };
    const body = [_]*const Node{&print_node};

    const else_text = Node{ .text = "empty" };
    const else_body = [_]*const Node{&else_text};

    const node = Node{ .for_stmt = .{
        .target = "x",
        .target2 = null,
        .iterable = &iterable_expr,
        .filter = null,
        .body = &body,
        .else_body = &else_body,
        .recursive = false,
    } };

    try evaluator.evalNode(&node);
    try std.testing.expectEqualStrings("empty", evaluator.output.items);
}

test "Evaluator.evalNode - set statement" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const value_expr = Expr{ .integer = 42 };
    const node = Node{ .set_stmt = .{
        .target = "x",
        .namespace = null,
        .value = &value_expr,
    } };

    try evaluator.evalNode(&node);

    const result = parser.get("x");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(i64, 42), result.?.integer);
}

test "Evaluator.evalNode - break statement" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const node: Node = .break_stmt;
    const err = evaluator.evalNode(&node);

    try std.testing.expectError(EvalError.LoopBreak, err);
}

test "Evaluator.evalNode - continue statement" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const node: Node = .continue_stmt;
    const err = evaluator.evalNode(&node);

    try std.testing.expectError(EvalError.LoopContinue, err);
}

// ============================================================================
// Tests for Evaluator.evalExpr
// ============================================================================

test "Evaluator.evalExpr - string literal" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const expr = Expr{ .string = "hello" };
    const result = try evaluator.evalExpr(&expr);

    try std.testing.expectEqualStrings("hello", result.string);
}

test "Evaluator.evalExpr - string with escape sequences" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const expr = Expr{ .string = "hello\\nworld" };
    const result = try evaluator.evalExpr(&expr);

    try std.testing.expect(std.mem.indexOf(u8, result.string, "\n") != null);
}

test "Evaluator.evalExpr - integer literal" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const expr = Expr{ .integer = 42 };
    const result = try evaluator.evalExpr(&expr);

    try std.testing.expectEqual(@as(i64, 42), result.integer);
}

test "Evaluator.evalExpr - float literal" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const expr = Expr{ .float = 3.14 };
    const result = try evaluator.evalExpr(&expr);

    try std.testing.expectEqual(@as(f64, 3.14), result.float);
}

test "Evaluator.evalExpr - boolean literals" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const expr_true = Expr{ .boolean = true };
    const result_true = try evaluator.evalExpr(&expr_true);
    try std.testing.expectEqual(true, result_true.boolean);

    const expr_false = Expr{ .boolean = false };
    const result_false = try evaluator.evalExpr(&expr_false);
    try std.testing.expectEqual(false, result_false.boolean);
}

test "Evaluator.evalExpr - none literal" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const expr: Expr = .none;
    const result = try evaluator.evalExpr(&expr);

    try std.testing.expect(result == .none);
}

test "Evaluator.evalExpr - variable lookup" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    try parser.set("x", .{ .integer = 100 });

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const expr = Expr{ .variable = "x" };
    const result = try evaluator.evalExpr(&expr);

    try std.testing.expectEqual(@as(i64, 100), result.integer);
}

test "Evaluator.evalExpr - undefined variable in lenient mode" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const expr = Expr{ .variable = "undefined" };
    const result = try evaluator.evalExpr(&expr);

    try std.testing.expect(result == .none);
}

test "Evaluator.evalExpr - undefined variable in strict mode" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.initStrict(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const expr = Expr{ .variable = "undefined" };
    const err = evaluator.evalExpr(&expr);

    try std.testing.expectError(EvalError.UndefinedVariable, err);
}

test "Evaluator.evalExpr - binary operation addition" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const left = Expr{ .integer = 5 };
    const right = Expr{ .integer = 3 };
    const expr = Expr{ .binop = .{
        .op = .add,
        .left = &left,
        .right = &right,
    } };

    const result = try evaluator.evalExpr(&expr);
    try std.testing.expectEqual(@as(i64, 8), result.integer);
}

test "Evaluator.evalExpr - binary operation string concatenation" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const left = Expr{ .string = "hello" };
    const right = Expr{ .string = "world" };
    const expr = Expr{ .binop = .{
        .op = .add,
        .left = &left,
        .right = &right,
    } };

    const result = try evaluator.evalExpr(&expr);
    try std.testing.expectEqualStrings("helloworld", result.string);
}

test "Evaluator.evalExpr - unary operation not" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const operand = Expr{ .boolean = true };
    const expr = Expr{ .unaryop = .{
        .op = .not,
        .operand = &operand,
    } };

    const result = try evaluator.evalExpr(&expr);
    try std.testing.expectEqual(false, result.boolean);
}

test "Evaluator.evalExpr - unary operation negation" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const operand = Expr{ .integer = 42 };
    const expr = Expr{ .unaryop = .{
        .op = .neg,
        .operand = &operand,
    } };

    const result = try evaluator.evalExpr(&expr);
    try std.testing.expectEqual(@as(i64, -42), result.integer);
}

test "Evaluator.evalExpr - getattr on map" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var map = std.StringHashMapUnmanaged(TemplateInput){};
    try map.put(parser.arena.allocator(), "key", .{ .integer = 42 });
    try parser.set("obj", .{ .map = map });

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const obj_expr = Expr{ .variable = "obj" };
    const expr = Expr{ .getattr = .{
        .object = &obj_expr,
        .attr = "key",
    } };

    const result = try evaluator.evalExpr(&expr);
    try std.testing.expectEqual(@as(i64, 42), result.integer);
}

test "Evaluator.evalExpr - getitem on array" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 10 },
        .{ .integer = 20 },
        .{ .integer = 30 },
    };
    try parser.set("arr", .{ .array = &items });

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const arr_expr = Expr{ .variable = "arr" };
    const index_expr = Expr{ .integer = 1 };
    const expr = Expr{ .getitem = .{
        .object = &arr_expr,
        .key = &index_expr,
    } };

    const result = try evaluator.evalExpr(&expr);
    try std.testing.expectEqual(@as(i64, 20), result.integer);
}

test "Evaluator.evalExpr - slice array" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 2 },
        .{ .integer = 3 },
        .{ .integer = 4 },
    };
    try parser.set("arr", .{ .array = &items });

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const arr_expr = Expr{ .variable = "arr" };
    const start_expr = Expr{ .integer = 1 };
    const stop_expr = Expr{ .integer = 3 };
    const expr = Expr{ .slice = .{
        .object = &arr_expr,
        .start = &start_expr,
        .stop = &stop_expr,
        .step = null,
    } };

    const result = try evaluator.evalExpr(&expr);
    try std.testing.expectEqual(@as(usize, 2), result.array.len);
    try std.testing.expectEqual(@as(i64, 2), result.array[0].integer);
    try std.testing.expectEqual(@as(i64, 3), result.array[1].integer);
}

test "Evaluator.evalExpr - conditional ternary true" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const test_expr = Expr{ .boolean = true };
    const true_expr = Expr{ .string = "yes" };
    const false_expr = Expr{ .string = "no" };
    const expr = Expr{ .conditional = .{
        .test_val = &test_expr,
        .true_val = &true_expr,
        .false_val = &false_expr,
    } };

    const result = try evaluator.evalExpr(&expr);
    try std.testing.expectEqualStrings("yes", result.string);
}

test "Evaluator.evalExpr - conditional ternary false" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const test_expr = Expr{ .boolean = false };
    const true_expr = Expr{ .string = "yes" };
    const false_expr = Expr{ .string = "no" };
    const expr = Expr{ .conditional = .{
        .test_val = &test_expr,
        .true_val = &true_expr,
        .false_val = &false_expr,
    } };

    const result = try evaluator.evalExpr(&expr);
    try std.testing.expectEqualStrings("no", result.string);
}

test "Evaluator.evalExpr - list literal" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const item1 = Expr{ .integer = 1 };
    const item2 = Expr{ .integer = 2 };
    const items = [_]*const Expr{ &item1, &item2 };
    const expr = Expr{ .list = &items };

    const result = try evaluator.evalExpr(&expr);
    try std.testing.expectEqual(@as(usize, 2), result.array.len);
    try std.testing.expectEqual(@as(i64, 1), result.array[0].integer);
    try std.testing.expectEqual(@as(i64, 2), result.array[1].integer);
}

test "Evaluator.evalExpr - dict literal" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    const key_expr = Expr{ .string = "name" };
    const value_expr = Expr{ .string = "Alice" };
    const pair = Expr.DictPair{
        .key = &key_expr,
        .value = &value_expr,
    };
    const pairs = [_]Expr.DictPair{pair};
    const expr = Expr{ .dict = &pairs };

    const result = try evaluator.evalExpr(&expr);
    try std.testing.expect(result.map.contains("name"));
    const value = result.map.get("name").?;
    try std.testing.expectEqualStrings("Alice", value.string);
}

// ============================================================================
// Tests for CustomFilterSet
// ============================================================================

test "CustomFilterSet.put - add single filter" {
    const allocator = std.testing.allocator;
    var filter_set = CustomFilterSet.init();
    defer filter_set.deinit(allocator);

    const dummy_callback: CustomFilterCallback = struct {
        fn callback(_: [*:0]const u8, _: [*:0]const u8, _: ?*anyopaque) callconv(.c) ?[*:0]u8 {
            return null;
        }
    }.callback;

    const filter = CustomFilter{
        .callback = dummy_callback,
        .user_data = null,
    };

    try filter_set.put(allocator, "test_filter", filter);

    const retrieved = filter_set.get("test_filter");
    try std.testing.expect(retrieved != null);
    try std.testing.expect(retrieved.?.callback == dummy_callback);
    try std.testing.expect(retrieved.?.user_data == null);
}

test "CustomFilterSet.put - overwrite existing filter" {
    const allocator = std.testing.allocator;
    var filter_set = CustomFilterSet.init();
    defer filter_set.deinit(allocator);

    const callback1: CustomFilterCallback = struct {
        fn callback(_: [*:0]const u8, _: [*:0]const u8, _: ?*anyopaque) callconv(.c) ?[*:0]u8 {
            return null;
        }
    }.callback;

    const callback2: CustomFilterCallback = struct {
        fn callback(_: [*:0]const u8, _: [*:0]const u8, _: ?*anyopaque) callconv(.c) ?[*:0]u8 {
            return null;
        }
    }.callback;

    const filter1 = CustomFilter{ .callback = callback1, .user_data = null };
    const filter2 = CustomFilter{ .callback = callback2, .user_data = null };

    try filter_set.put(allocator, "my_filter", filter1);
    try filter_set.put(allocator, "my_filter", filter2);

    const retrieved = filter_set.get("my_filter");
    try std.testing.expect(retrieved != null);
    try std.testing.expect(retrieved.?.callback == callback2);
}

test "CustomFilterSet.put - multiple filters" {
    const allocator = std.testing.allocator;
    var filter_set = CustomFilterSet.init();
    defer filter_set.deinit(allocator);

    const callback1: CustomFilterCallback = struct {
        fn callback(_: [*:0]const u8, _: [*:0]const u8, _: ?*anyopaque) callconv(.c) ?[*:0]u8 {
            return null;
        }
    }.callback;

    const callback2: CustomFilterCallback = struct {
        fn callback(_: [*:0]const u8, _: [*:0]const u8, _: ?*anyopaque) callconv(.c) ?[*:0]u8 {
            return null;
        }
    }.callback;

    const filter1 = CustomFilter{ .callback = callback1, .user_data = null };
    const filter2 = CustomFilter{ .callback = callback2, .user_data = null };

    try filter_set.put(allocator, "filter1", filter1);
    try filter_set.put(allocator, "filter2", filter2);

    const retrieved1 = filter_set.get("filter1");
    const retrieved2 = filter_set.get("filter2");

    try std.testing.expect(retrieved1 != null);
    try std.testing.expect(retrieved2 != null);
    try std.testing.expect(retrieved1.?.callback == callback1);
    try std.testing.expect(retrieved2.?.callback == callback2);
}

test "CustomFilterSet.put - with user_data" {
    const allocator = std.testing.allocator;
    var filter_set = CustomFilterSet.init();
    defer filter_set.deinit(allocator);

    var user_data: i32 = 42;

    const callback: CustomFilterCallback = struct {
        fn callback(_: [*:0]const u8, _: [*:0]const u8, _: ?*anyopaque) callconv(.c) ?[*:0]u8 {
            return null;
        }
    }.callback;

    const filter = CustomFilter{
        .callback = callback,
        .user_data = @ptrCast(&user_data),
    };

    try filter_set.put(allocator, "data_filter", filter);

    const retrieved = filter_set.get("data_filter");
    try std.testing.expect(retrieved != null);
    try std.testing.expect(retrieved.?.user_data != null);

    const retrieved_data: *i32 = @ptrCast(@alignCast(retrieved.?.user_data.?));
    try std.testing.expectEqual(@as(i32, 42), retrieved_data.*);
}

// ============================================================================
// Tests for TemplateParser.arenaAllocator
// ============================================================================

test "TemplateParser.arenaAllocator - returns valid allocator" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    const arena_alloc = parser.arenaAllocator();

    // Test that the allocator can allocate
    const slice = try arena_alloc.alloc(u8, 10);
    try std.testing.expectEqual(@as(usize, 10), slice.len);

    // No need to free - arena will handle it
}

test "TemplateParser.arenaAllocator - allocations persist until deinit" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    const arena_alloc = parser.arenaAllocator();

    // Multiple allocations
    const slice1 = try arena_alloc.alloc(u8, 10);
    const slice2 = try arena_alloc.alloc(u8, 20);

    @memset(slice1, 'a');
    @memset(slice2, 'b');

    // Verify both allocations are still valid
    try std.testing.expectEqual(@as(u8, 'a'), slice1[0]);
    try std.testing.expectEqual(@as(u8, 'b'), slice2[0]);
}

test "TemplateParser.arenaAllocator - used for template strings" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    const arena_alloc = parser.arenaAllocator();

    // Simulate storing a duped string in the arena
    const original = "test string";
    const duped = try arena_alloc.dupe(u8, original);

    try std.testing.expectEqualStrings(original, duped);

    // Store it as a template input
    try parser.set("text", .{ .string = duped });

    const retrieved = parser.get("text");
    try std.testing.expect(retrieved != null);
    try std.testing.expectEqualStrings(original, retrieved.?.string);
}

// ============================================================================
// Tests for Evaluator.initDebug
// ============================================================================

test "Evaluator.initDebug - debug mode enabled" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.initDebug(allocator, &parser);
    defer evaluator.deinit();

    try std.testing.expect(evaluator.debug_mode);
    try std.testing.expectEqual(@as(usize, 0), evaluator.spans.items.len);
}

test "Evaluator.initDebug - spans tracked in debug mode" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    try parser.set("name", .{ .string = "Alice" });

    var evaluator = Evaluator.initDebug(allocator, &parser);
    defer evaluator.deinit();

    // Create a simple variable print node
    const var_expr = Expr{ .variable = "name" };
    const node = Node{ .print = &var_expr };
    try evaluator.evalNode(&node);

    // Debug mode should record a span
    try std.testing.expect(evaluator.spans.items.len > 0);
    try std.testing.expectEqualStrings("Alice", evaluator.output.items);
}

test "Evaluator.initDebug - vs regular init" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    // Regular evaluator
    var evaluator_normal = Evaluator.init(allocator, &parser);
    defer evaluator_normal.deinit();

    // Debug evaluator
    var evaluator_debug = Evaluator.initDebug(allocator, &parser);
    defer evaluator_debug.deinit();

    try std.testing.expect(!evaluator_normal.debug_mode);
    try std.testing.expect(evaluator_debug.debug_mode);
}

// ============================================================================
// Tests for Evaluator.initWithFilters
// ============================================================================

test "Evaluator.initWithFilters - null filters" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.initWithFilters(allocator, &parser, null);
    defer evaluator.deinit();

    try std.testing.expect(evaluator.custom_filters == null);
    try std.testing.expect(!evaluator.debug_mode);
}

test "Evaluator.initWithFilters - with custom filters" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var filter_set = CustomFilterSet.init();
    defer filter_set.deinit(allocator);

    const dummy_callback: CustomFilterCallback = struct {
        fn callback(_: [*:0]const u8, _: [*:0]const u8, _: ?*anyopaque) callconv(.c) ?[*:0]u8 {
            return null;
        }
    }.callback;

    const filter = CustomFilter{
        .callback = dummy_callback,
        .user_data = null,
    };

    try filter_set.put(allocator, "custom", filter);

    var evaluator = Evaluator.initWithFilters(allocator, &parser, &filter_set);
    defer evaluator.deinit();

    try std.testing.expect(evaluator.custom_filters != null);
    try std.testing.expect(!evaluator.debug_mode);

    // Verify we can access the filter through the evaluator
    const retrieved = evaluator.custom_filters.?.get("custom");
    try std.testing.expect(retrieved != null);
    try std.testing.expect(retrieved.?.callback == dummy_callback);
}

test "Evaluator.initWithFilters - filters not enabled in debug mode" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var filter_set = CustomFilterSet.init();
    defer filter_set.deinit(allocator);

    // initWithFilters should not enable debug mode
    var evaluator = Evaluator.initWithFilters(allocator, &parser, &filter_set);
    defer evaluator.deinit();

    try std.testing.expect(!evaluator.debug_mode);
    try std.testing.expect(evaluator.custom_filters != null);
}

// ============================================================================
// Tests for Evaluator.renderWithSpans
// ============================================================================

test "Evaluator.renderWithSpans - basic text" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.initDebug(allocator, &parser);
    defer evaluator.deinit();

    const text_node = Node{ .text = "hello" };
    const nodes = [_]*const Node{&text_node};

    const result = try evaluator.renderWithSpans(&nodes);
    defer allocator.free(result.output);
    defer allocator.free(result.spans);

    try std.testing.expectEqualStrings("hello", result.output);
    try std.testing.expectEqual(@as(usize, 1), result.spans.len);
    try std.testing.expectEqual(@as(usize, 0), result.spans[0].start);
    try std.testing.expectEqual(@as(usize, 5), result.spans[0].end);
}

test "Evaluator.renderWithSpans - variable span" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    try parser.set("name", .{ .string = "Bob" });

    var evaluator = Evaluator.initDebug(allocator, &parser);
    defer evaluator.deinit();

    const var_expr = Expr{ .variable = "name" };
    const node = Node{ .print = &var_expr };
    const nodes = [_]*const Node{&node};

    const result = try evaluator.renderWithSpans(&nodes);
    defer allocator.free(result.output);
    defer allocator.free(result.spans);

    try std.testing.expectEqualStrings("Bob", result.output);
    try std.testing.expectEqual(@as(usize, 1), result.spans.len);
    try std.testing.expectEqual(@as(usize, 0), result.spans[0].start);
    try std.testing.expectEqual(@as(usize, 3), result.spans[0].end);

    // Verify it's a variable span
    switch (result.spans[0].source) {
        .variable => |var_name| try std.testing.expectEqualStrings("name", var_name),
        else => try std.testing.expect(false),
    }
}

test "Evaluator.renderWithSpans - mixed static and variable" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    try parser.set("user", .{ .string = "Alice" });

    var evaluator = Evaluator.initDebug(allocator, &parser);
    defer evaluator.deinit();

    const text_node1 = Node{ .text = "Hello, " };
    const var_expr = Expr{ .variable = "user" };
    const var_node = Node{ .print = &var_expr };
    const text_node2 = Node{ .text = "!" };
    const nodes = [_]*const Node{ &text_node1, &var_node, &text_node2 };

    const result = try evaluator.renderWithSpans(&nodes);
    defer allocator.free(result.output);
    defer allocator.free(result.spans);

    try std.testing.expectEqualStrings("Hello, Alice!", result.output);
    try std.testing.expectEqual(@as(usize, 3), result.spans.len);

    // Verify span boundaries and sources
    try std.testing.expectEqual(@as(usize, 0), result.spans[0].start);
    try std.testing.expectEqual(@as(usize, 7), result.spans[0].end);
    switch (result.spans[0].source) {
        .static_text => {},
        else => try std.testing.expect(false),
    }

    try std.testing.expectEqual(@as(usize, 7), result.spans[1].start);
    try std.testing.expectEqual(@as(usize, 12), result.spans[1].end);
    switch (result.spans[1].source) {
        .variable => |var_name| try std.testing.expectEqualStrings("user", var_name),
        else => try std.testing.expect(false),
    }

    try std.testing.expectEqual(@as(usize, 12), result.spans[2].start);
    try std.testing.expectEqual(@as(usize, 13), result.spans[2].end);
    switch (result.spans[2].source) {
        .static_text => {},
        else => try std.testing.expect(false),
    }
}

test "Evaluator.renderWithSpans - empty template" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.initDebug(allocator, &parser);
    defer evaluator.deinit();

    const nodes: []const *const Node = &.{};

    const result = try evaluator.renderWithSpans(nodes);
    defer allocator.free(result.output);
    defer allocator.free(result.spans);

    try std.testing.expectEqualStrings("", result.output);
    try std.testing.expectEqual(@as(usize, 0), result.spans.len);
}

test "Evaluator.renderWithSpans - without debug mode returns empty spans" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    // Use regular init (not initDebug)
    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    try parser.set("name", .{ .string = "Charlie" });

    const var_expr = Expr{ .variable = "name" };
    const node = Node{ .print = &var_expr };
    const nodes = [_]*const Node{&node};

    const result = try evaluator.renderWithSpans(&nodes);
    defer allocator.free(result.output);
    defer allocator.free(result.spans);

    try std.testing.expectEqualStrings("Charlie", result.output);
    // Without debug mode, spans should be empty
    try std.testing.expectEqual(@as(usize, 0), result.spans.len);
}

test "evalExpr getItem returns none for missing dict key" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    // Create an empty map - accessing any key should return none
    const map_input = TemplateInput{ .map = .{} };

    // Access a missing key - should return none, not error
    const result = try evaluator.getItem(map_input, .{ .string = "missing" });
    try std.testing.expectEqual(TemplateInput.none, result);
}

test "evalExpr or operator returns right value when left is falsy" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    // none or "default" should return "default"
    const none_expr = parser.arena.allocator().create(Expr) catch unreachable;
    none_expr.* = .none;
    const default_expr = parser.arena.allocator().create(Expr) catch unreachable;
    default_expr.* = .{ .string = "default" };

    const result = try evaluator.evalBinOp(.@"or", none_expr, default_expr);
    try std.testing.expectEqual(TemplateInput{ .string = "default" }, result);
}

test "evalExpr and operator returns left value when left is falsy" {
    const allocator = std.testing.allocator;
    var parser = TemplateParser.init(allocator);
    defer parser.deinit();

    var evaluator = Evaluator.init(allocator, &parser);
    defer evaluator.deinit();

    // none and "value" should return none
    const none_expr = parser.arena.allocator().create(Expr) catch unreachable;
    none_expr.* = .none;
    const value_expr = parser.arena.allocator().create(Expr) catch unreachable;
    value_expr.* = .{ .string = "value" };

    const result = try evaluator.evalBinOp(.@"and", none_expr, value_expr);
    try std.testing.expectEqual(TemplateInput.none, result);
}
