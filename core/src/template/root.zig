//! Template Engine for LLM Chat Templates
//!
//! Parse and render templates with dynamic data. Supports the Jinja2 syntax
//! used by LLM chat templates (ChatML, Llama, etc.).
//!
//! ## Public API
//!
//! - `TemplateParser`: Parse templates with your data
//! - `TemplateInput`: Dynamic values (string, integer, boolean, array, map)
//! - `render()`: Render a template string
//!
//! ## Example
//!
//! ```zig
//! const template_engine = @import("template/root.zig");
//!
//! // Create a parser and add your data
//! var parser = template_engine.TemplateParser.init(allocator);
//! defer parser.deinit();
//! try parser.set("name", .{ .string = "Alice" });
//! try parser.set("score", .{ .integer = 100 });
//!
//! // Render the template
//! const result = try template_engine.render(allocator, "{{ name }} scored {{ score }}!", &parser);
//! defer allocator.free(result);
//! // result: "Alice scored 100!"
//! ```
//!
//! ## Strict Mode
//!
//! By default, undefined variables silently return empty values. Enable strict
//! mode to catch undefined variable errors:
//!
//! ```zig
//! var parser = template_engine.TemplateParser.initStrict(allocator);
//! // or: parser.strict = true;
//!
//! // This will error instead of returning ""
//! const result = template_engine.render(allocator, "{{ undefined_var }}", &parser);
//! // Returns EvalError.UndefinedVariable
//! ```

const std = @import("std");

// Internal imports (not re-exported)
const ast = @import("ast.zig");
const lexer_mod = @import("lexer.zig");
const parser_mod = @import("parser.zig");
const eval_mod = @import("eval.zig");
const validate_mod = @import("validate.zig");
const json_mod = @import("json.zig");

// Public API: only expose what external callers need
pub const TemplateInput = eval_mod.TemplateInput;
pub const TemplateParser = eval_mod.TemplateParser;
pub const OutputSpan = eval_mod.OutputSpan;
pub const SpanSource = eval_mod.SpanSource;
pub const ValidationResult = validate_mod.ValidationResult;
pub const JsonError = validate_mod.JsonError;
pub const validate = validate_mod.validate;
pub const validateJson = validate_mod.validateJson;
pub const jsonErrorMessage = validate_mod.jsonErrorMessage;
pub const extractVariables = validate_mod.extractVariables;

// JSON integration (high-level API for C/Python)
pub const jsonToTemplateInput = json_mod.jsonToTemplateInput;
pub const validationResultToJson = json_mod.validationResultToJson;
pub const RenderResult = json_mod.RenderResult;
pub const RenderDebugResult = json_mod.RenderDebugResult;
pub const renderFromJson = json_mod.renderFromJson;
pub const renderFromJsonWithFilters = json_mod.renderFromJsonWithFilters;
pub const renderFromJsonDebug = json_mod.renderFromJsonDebug;

/// Chat template rendering for LLM conversations.
pub const chat_template = @import("chat_template.zig");

// Custom filter types for Python/C integration
pub const CustomFilterCallback = eval_mod.CustomFilterCallback;
pub const CustomFilter = eval_mod.CustomFilter;
pub const CustomFilterSet = eval_mod.CustomFilterSet;

// Internal types used by render()
const Evaluator = eval_mod.Evaluator;
const Parser = parser_mod.Parser;
const Lexer = lexer_mod.Lexer;

// Re-export behavioral types so check_coverage.sh --integration can verify test coverage
pub const TemplateEvaluator = eval_mod.Evaluator;
pub const TemplateParserInternal = parser_mod.Parser;
pub const TemplateLexer = lexer_mod.Lexer;
pub const EvalError = eval_mod.EvalError;
pub const ParseError = parser_mod.ParseError;
pub const Token = lexer_mod.Token;
pub const TokenType = lexer_mod.TokenType;

pub const Error = error{
    LexError,
    ParseError,
    EvalError,
    UndefinedVariable, // Strict mode: undefined variable accessed
    RaiseException,
    IncludeTypeError,
    OutOfMemory,
};

fn toValueArray(allocator: std.mem.Allocator, data: anytype) !TemplateInput {
    var arr = std.ArrayList(TemplateInput).init(allocator);
    for (data) |item| {
        try arr.append(try toValue(allocator, item));
    }
    return .{ .array = try arr.toOwnedSlice() };
}

/// Result of rendering with span tracking enabled.
pub const RenderWithSpansResult = struct {
    output: []const u8,
    spans: []const OutputSpan,
};

// =============================================================================
// C-compatible span types (for FFI)
// =============================================================================

/// C-compatible span source type.
pub const CSpanSourceType = enum(c_int) {
    static_text = 0,
    variable = 1,
    expression = 2,
};

/// C-compatible output span with null-terminated strings.
pub const COutputSpan = extern struct {
    start: u32,
    end: u32,
    source_type: CSpanSourceType,
    variable_path: ?[*:0]u8,
};

/// List of C-compatible output spans.
pub const COutputSpanList = struct {
    spans: []COutputSpan,

    const Self = @This();

    /// Convert internal OutputSpan slice to C-compatible spans.
    /// All variable paths are copied with null terminators.
    /// Caller owns the result and must call deinit().
    pub fn fromSpans(allocator: std.mem.Allocator, spans: []const OutputSpan) !Self {
        if (spans.len == 0) {
            return Self{ .spans = &.{} };
        }

        const c_spans = try allocator.alloc(COutputSpan, spans.len);
        errdefer allocator.free(c_spans);

        var initialized: usize = 0;
        errdefer {
            for (0..initialized) |j| {
                if (c_spans[j].variable_path) |path| {
                    const slice = std.mem.span(path);
                    allocator.free(slice[0 .. slice.len + 1]);
                }
            }
        }

        for (spans, 0..) |span, i| {
            c_spans[i] = .{
                .start = @intCast(span.start),
                .end = @intCast(span.end),
                .source_type = switch (span.source) {
                    .static_text => .static_text,
                    .variable => .variable,
                    .expression => .expression,
                },
                .variable_path = switch (span.source) {
                    .variable => |path| blk: {
                        const cstr = try allocator.allocSentinel(u8, path.len, 0);
                        @memcpy(cstr, path);
                        break :blk cstr.ptr;
                    },
                    else => null,
                },
            };
            initialized = i + 1;
        }

        return Self{ .spans = c_spans };
    }

    /// Free all variable paths and the spans array.
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        for (self.spans) |span| {
            if (span.variable_path) |path| {
                const slice = std.mem.span(path);
                allocator.free(slice[0 .. slice.len + 1]);
            }
        }
        if (self.spans.len > 0) {
            allocator.free(self.spans);
        }
        self.spans = &.{};
    }

    /// Get the number of spans.
    pub fn count(self: *const Self) usize {
        return self.spans.len;
    }
};

/// Render a Jinja2 template with the given context.
/// Returns allocated string that must be freed by caller.
pub fn render(allocator: std.mem.Allocator, template: []const u8, ctx: *TemplateParser) Error![]const u8 {
    // Tokenize
    var lexer_ctx = Lexer.init(allocator, template);
    defer lexer_ctx.deinit();

    const tokens = lexer_ctx.tokenize() catch {
        return Error.LexError;
    };

    // Parse
    var parser_ctx = Parser.init(allocator, tokens);
    defer parser_ctx.deinit();

    const nodes = parser_ctx.parse() catch {
        // Capture parse error context (e.g., "if", "for" block name)
        ctx.parse_error_context = parser_ctx.error_context;
        return Error.ParseError;
    };
    defer allocator.free(nodes);

    // Evaluate
    var eval_ctx = Evaluator.init(allocator, ctx);
    defer eval_ctx.deinit();

    return eval_ctx.render(nodes) catch |err| {
        // Distinguish specific errors for different error codes
        return switch (err) {
            // Both UndefinedVariable and KeyError map to "undefined" for consistency
            // KeyError is returned when accessing missing dict keys in strict mode
            EvalError.UndefinedVariable, EvalError.KeyError => Error.UndefinedVariable,
            EvalError.RaiseException => Error.RaiseException,
            EvalError.IncludeTypeError => Error.IncludeTypeError,
            else => Error.EvalError,
        };
    };
}

/// Render with span tracking for debug visualization.
/// Returns rendered output plus spans showing which parts came from variables vs static text.
/// Caller must free both output and spans.
pub fn renderWithSpans(allocator: std.mem.Allocator, template: []const u8, ctx: *TemplateParser) Error!RenderWithSpansResult {
    // Tokenize
    var lexer_ctx = Lexer.init(allocator, template);
    defer lexer_ctx.deinit();

    const tokens = lexer_ctx.tokenize() catch {
        return Error.LexError;
    };

    // Parse
    var parser_ctx = Parser.init(allocator, tokens);
    defer parser_ctx.deinit();

    const nodes = parser_ctx.parse() catch {
        ctx.parse_error_context = parser_ctx.error_context;
        return Error.ParseError;
    };
    defer allocator.free(nodes);

    // Evaluate with span tracking
    var eval_ctx = Evaluator.initDebug(allocator, ctx);
    defer eval_ctx.deinit();

    const result = eval_ctx.renderWithSpans(nodes) catch |err| {
        return switch (err) {
            EvalError.UndefinedVariable, EvalError.KeyError => Error.UndefinedVariable,
            EvalError.RaiseException => Error.RaiseException,
            EvalError.IncludeTypeError => Error.IncludeTypeError,
            else => Error.EvalError,
        };
    };

    return .{
        .output = result.output,
        .spans = result.spans,
    };
}


/// Render a template with custom filters (Python callbacks).
/// Custom filters are checked before built-in filters.
pub fn renderWithFilters(
    allocator: std.mem.Allocator,
    template: []const u8,
    ctx: *TemplateParser,
    custom_filters: ?*const CustomFilterSet,
) Error![]const u8 {
    // Tokenize
    var lexer_ctx = Lexer.init(allocator, template);
    defer lexer_ctx.deinit();

    const tokens = lexer_ctx.tokenize() catch {
        return Error.LexError;
    };

    // Parse
    var parser_ctx = Parser.init(allocator, tokens);
    defer parser_ctx.deinit();

    const nodes = parser_ctx.parse() catch {
        ctx.parse_error_context = parser_ctx.error_context;
        return Error.ParseError;
    };
    defer allocator.free(nodes);

    // Evaluate with custom filters
    var eval_ctx = Evaluator.initWithFilters(allocator, ctx, custom_filters);
    defer eval_ctx.deinit();

    return eval_ctx.render(nodes) catch |err| {
        return switch (err) {
            EvalError.UndefinedVariable, EvalError.KeyError => Error.UndefinedVariable,
            EvalError.RaiseException => Error.RaiseException,
            EvalError.IncludeTypeError => Error.IncludeTypeError,
            else => Error.EvalError,
        };
    };
}

/// Convert a Zig struct/slice to a Jinja TemplateInput for template context.
fn toValue(allocator: std.mem.Allocator, data: anytype) !TemplateInput {
    const T = @TypeOf(data);
    const info = @typeInfo(T);

    return switch (info) {
        .pointer => |ptr| {
            if (ptr.size == .Slice) {
                if (ptr.child == u8) {
                    // String slice
                    return .{ .string = data };
                } else {
                    return toValueArray(allocator, data);
                }
            } else if (ptr.size == .One) {
                // Pointer to single item - dereference
                return toValue(allocator, data.*);
            }
            return .none;
        },
        .@"struct" => |st| {
            var map = std.StringHashMapUnmanaged(TemplateInput){};
            inline for (st.fields) |field| {
                const field_value = try toValue(allocator, @field(data, field.name));
                try map.put(allocator, field.name, field_value);
            }
            return .{ .map = map };
        },
        .optional => {
            if (data) |d| {
                return toValue(allocator, d);
            }
            return .none;
        },
        .int, .comptime_int => {
            return .{ .integer = @intCast(data) };
        },
        .float, .comptime_float => {
            return .{ .float = @floatCast(data) };
        },
        .bool => {
            return .{ .boolean = data };
        },
        .array => |arr_info| {
            if (arr_info.child == u8) {
                return .{ .string = &data };
            }
            return toValueArray(allocator, data);
        },
        else => .none,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "render simple variable" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();
    try ctx.set("name", .{ .string = "World" });

    const result = try render(allocator, "Hello {{ name }}!", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello World!", result);
}

test "render if statement" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();
    try ctx.set("show", .{ .boolean = true });

    const result = try render(allocator, "{% if show %}visible{% endif %}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("visible", result);
}

test "render for loop" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    const items = [_]TemplateInput{
        .{ .string = "a" },
        .{ .string = "b" },
        .{ .string = "c" },
    };
    try ctx.set("items", .{ .array = &items });

    const result = try render(allocator, "{% for x in items %}{{ x }}{% endfor %}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("abc", result);
}

test "render slice reverse" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 2 },
        .{ .integer = 3 },
    };
    try ctx.set("items", .{ .array = &items });

    const result = try render(allocator, "{% for x in items[::-1] %}{{ x }}{% endfor %}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("321", result);
}

test "render filter tojson" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    var map = std.StringHashMapUnmanaged(TemplateInput){};
    try map.put(allocator, "name", .{ .string = "test" });
    defer map.deinit(allocator);

    try ctx.set("obj", .{ .map = map });

    const result = try render(allocator, "{{ obj | tojson }}", &ctx);
    defer allocator.free(result);

    // JSON output
    try std.testing.expect(std.mem.indexOf(u8, result, "\"name\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"test\"") != null);
}

test "render string methods" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();
    try ctx.set("s", .{ .string = "  hello  " });

    const result = try render(allocator, "{{ s.strip() }}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("hello", result);
}

test "render loop context" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    const items = [_]TemplateInput{
        .{ .string = "a" },
        .{ .string = "b" },
    };
    try ctx.set("items", .{ .array = &items });

    const result = try render(allocator, "{% for x in items %}{% if loop.first %}F{% endif %}{{ x }}{% endfor %}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Fab", result);
}

test "render chatml template" {
    // Test a simplified ChatML template
    const allocator = std.testing.allocator;

    const template =
        \\{%- if messages[0].role == 'system' -%}
        \\<|im_start|>system
        \\{{ messages[0].content }}<|im_end|>
        \\{%- endif -%}
        \\{%- for message in messages -%}
        \\{%- if message.role == 'user' or (message.role == 'system' and not loop.first) -%}
        \\<|im_start|>{{ message.role }}
        \\{{ message.content }}<|im_end|>
        \\{%- elif message.role == 'assistant' -%}
        \\<|im_start|>assistant
        \\{{ message.content }}<|im_end|>
        \\{%- endif -%}
        \\{%- endfor -%}
        \\{%- if add_generation_prompt -%}
        \\<|im_start|>assistant
        \\{%- endif -%}
    ;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    // Build messages array with map values
    var msg1 = std.StringHashMapUnmanaged(TemplateInput){};
    try msg1.put(allocator, "role", .{ .string = "system" });
    try msg1.put(allocator, "content", .{ .string = "You are a helpful assistant." });
    defer msg1.deinit(allocator);

    var msg2 = std.StringHashMapUnmanaged(TemplateInput){};
    try msg2.put(allocator, "role", .{ .string = "user" });
    try msg2.put(allocator, "content", .{ .string = "Hello!" });
    defer msg2.deinit(allocator);

    const messages = [_]TemplateInput{
        .{ .map = msg1 },
        .{ .map = msg2 },
    };

    try ctx.set("messages", .{ .array = &messages });
    try ctx.set("add_generation_prompt", .{ .boolean = true });

    const result = try render(allocator, template, &ctx);
    defer allocator.free(result);

    // Verify output contains expected markers
    try std.testing.expect(std.mem.indexOf(u8, result, "<|im_start|>system") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "You are a helpful assistant.") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "<|im_start|>user") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Hello!") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "<|im_start|>assistant") != null);
}

test "render namespace and set" {
    const allocator = std.testing.allocator;

    const template =
        \\{% set ns = namespace(count=0) %}
        \\{% for x in items %}{% set ns.count = ns.count + 1 %}{% endfor %}
        \\{{ ns.count }}
    ;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 2 },
        .{ .integer = 3 },
    };
    try ctx.set("items", .{ .array = &items });

    const result = try render(allocator, template, &ctx);
    defer allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "3") != null);
}

test "render is string test" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    var msg = std.StringHashMapUnmanaged(TemplateInput){};
    try msg.put(allocator, "content", .{ .string = "hello" });
    defer msg.deinit(allocator);
    try ctx.set("message", .{ .map = msg });

    const result = try render(allocator, "{% if message.content is string %}yes{% endif %}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("yes", result);
}

test "render or operator" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();
    try ctx.set("a", .{ .boolean = false });
    try ctx.set("b", .{ .boolean = true });

    const result = try render(allocator, "{% if a or b %}yes{% else %}no{% endif %}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("yes", result);
}

test "render undefined variable in if" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();
    // Note: 'tools' is NOT set, should be treated as falsy

    const result = try render(allocator, "{% if tools %}has tools{% else %}no tools{% endif %}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("no tools", result);
}

test "render filter with arithmetic" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 2 },
        .{ .integer = 3 },
    };
    try ctx.set("items", .{ .array = &items });

    // Test: items|length - 1 should equal 2
    const result = try render(allocator, "{{ items|length - 1 }}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("2", result);
}

test "render set with filter arithmetic" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 2 },
        .{ .integer = 3 },
    };
    try ctx.set("items", .{ .array = &items });

    // Test: {% set x = items|length - 1 %}
    const result = try render(allocator, "{% set x = items|length - 1 %}{{ x }}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("2", result);
}

test "namespace with multiple args" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    const items = [_]TemplateInput{ .{ .integer = 1 }, .{ .integer = 2 }, .{ .integer = 3 } };
    try ctx.set("items", .{ .array = &items });

    // Test namespace with multiple keyword args
    const result = try render(allocator, "{% set ns = namespace(flag=true, last=items|length - 1) %}{{ ns.last }}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("2", result);
}

test "namespace created inside for loop" {
    // Test for namespace created inside a for loop - must be accessible via local scopes
    // Regression test for Granite 4.0 Hybrid chat template pattern
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    var msg1 = std.StringHashMapUnmanaged(TemplateInput){};
    try msg1.put(allocator, "role", .{ .string = "user" });
    try msg1.put(allocator, "content", .{ .string = "Hello" });
    defer msg1.deinit(allocator);

    var msg2 = std.StringHashMapUnmanaged(TemplateInput){};
    try msg2.put(allocator, "role", .{ .string = "assistant" });
    try msg2.put(allocator, "content", .{ .string = "Hi there" });
    defer msg2.deinit(allocator);

    const messages = [_]TemplateInput{
        .{ .map = msg1 },
        .{ .map = msg2 },
    };
    try ctx.set("messages", .{ .array = &messages });

    // This pattern is used in Granite 4.0 Hybrid template:
    // Create a namespace inside the for loop, then modify it
    const template =
        \\{%- for message in messages %}
        \\{%- set content = namespace(val='') %}
        \\{%- if message.content is string %}
        \\{%- set content.val = message.content %}
        \\{%- endif %}
        \\{{ message.role }}: {{ content.val }}
        \\{% endfor %}
    ;

    const result = try render(allocator, template, &ctx);
    defer allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "user: Hello") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "assistant: Hi there") != null);
}

test "not with parentheses" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    try ctx.set("a", .{ .boolean = true });
    try ctx.set("b", .{ .boolean = true });

    // Test not(expr) syntax (as function call, not just 'not expr')
    const result = try render(allocator, "{% if not(a and b) %}no{% else %}yes{% endif %}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("yes", result);
}

test "method chaining with subscript" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    try ctx.set("s", .{ .string = "hello</think>world" });

    // Test: s.split('</think>')[0] should give "hello"
    const result = try render(allocator, "{{ s.split('</think>')[0] }}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("hello", result);
}

test "ChatML first section" {
    // Test just the first section of a ChatML template (before tools)
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    var msg1 = std.StringHashMapUnmanaged(TemplateInput){};
    try msg1.put(allocator, "role", .{ .string = "system" });
    try msg1.put(allocator, "content", .{ .string = "You are helpful." });
    defer msg1.deinit(allocator);

    var msg2 = std.StringHashMapUnmanaged(TemplateInput){};
    try msg2.put(allocator, "role", .{ .string = "user" });
    try msg2.put(allocator, "content", .{ .string = "Hello" });
    defer msg2.deinit(allocator);

    const messages = [_]TemplateInput{ .{ .map = msg1 }, .{ .map = msg2 } };
    try ctx.set("messages", .{ .array = &messages });
    try ctx.set("add_generation_prompt", .{ .boolean = true });

    // Simplified ChatML template (no tools, no thinking)
    const template =
        \\{%- if messages[0].role == 'system' -%}
        \\<|im_start|>system
        \\{{ messages[0].content }}<|im_end|>
        \\{%- endif -%}
        \\{%- for message in messages -%}
        \\{%- if message.role == 'user' or (message.role == 'system' and not loop.first) -%}
        \\<|im_start|>{{ message.role }}
        \\{{ message.content }}<|im_end|>
        \\{%- elif message.role == 'assistant' -%}
        \\<|im_start|>assistant
        \\{{ message.content }}<|im_end|>
        \\{%- endif -%}
        \\{%- endfor -%}
        \\{%- if add_generation_prompt -%}
        \\<|im_start|>assistant
        \\{%- endif -%}
    ;

    const result = try render(allocator, template, &ctx);
    defer allocator.free(result);

    // Should contain system message and user message
    try std.testing.expect(std.mem.indexOf(u8, result, "<|im_start|>system") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "You are helpful.") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "<|im_start|>user") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Hello") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "<|im_start|>assistant") != null);
}

test "ChatML template constructs" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    // Setup messages for ChatML format
    var msg1 = std.StringHashMapUnmanaged(TemplateInput){};
    try msg1.put(allocator, "role", .{ .string = "system" });
    try msg1.put(allocator, "content", .{ .string = "You are helpful." });
    defer msg1.deinit(allocator);

    var msg2 = std.StringHashMapUnmanaged(TemplateInput){};
    try msg2.put(allocator, "role", .{ .string = "user" });
    try msg2.put(allocator, "content", .{ .string = "Hello" });
    defer msg2.deinit(allocator);

    const messages = [_]TemplateInput{
        .{ .map = msg1 },
        .{ .map = msg2 },
    };
    try ctx.set("messages", .{ .array = &messages });
    try ctx.set("add_generation_prompt", .{ .boolean = true });

    // Test 1: namespace with arithmetic
    {
        const result = try render(allocator, "{% set ns = namespace(last=messages|length - 1) %}{{ ns.last }}", &ctx);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("1", result);
    }

    // Test 2: reverse iteration
    {
        const result = try render(allocator, "{% for m in messages[::-1] %}{{ m.role }},{% endfor %}", &ctx);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("user,system,", result);
    }

    // Test 3: loop.index0 arithmetic
    {
        const result = try render(allocator, "{% for m in messages %}{{ (messages|length - 1) - loop.index0 }},{% endfor %}", &ctx);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("1,0,", result);
    }

    // Test 4: is defined test
    {
        const result = try render(allocator, "{% if tools is defined %}yes{% else %}no{% endif %}", &ctx);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("no", result);
    }

    // Test 5: string method .startswith()
    {
        const result = try render(allocator, "{% if messages[0].content.startswith('You') %}yes{% else %}no{% endif %}", &ctx);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("yes", result);
    }

    // Test 6: complex condition with and/or/not
    {
        const result = try render(allocator,
            \\{% if messages[0].role == 'user' or (messages[0].role == 'system' and not loop is defined) %}yes{% else %}no{% endif %}
        , &ctx);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("yes", result);
    }

    // Test 7: not loop.first (attribute access with not)
    {
        const result = try render(allocator,
            \\{% for m in messages %}{% if not loop.first %},{% endif %}{{ m.role }}{% endfor %}
        , &ctx);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("system,user", result);
    }

    // Test 8: 'x' in content (in operator with string)
    {
        try ctx.set("content", .{ .string = "hello world" });
        const result = try render(allocator, "{% if 'world' in content %}yes{% else %}no{% endif %}", &ctx);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("yes", result);
    }

    // Test 9: not(expr) function-style not
    {
        try ctx.set("a", .{ .boolean = true });
        try ctx.set("b", .{ .boolean = true });
        const result = try render(allocator, "{% if not(a and b) %}yes{% else %}no{% endif %}", &ctx);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("no", result);
    }
}

test "ChatML full template parse" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    // Setup messages
    var msg1 = std.StringHashMapUnmanaged(TemplateInput){};
    try msg1.put(allocator, "role", .{ .string = "user" });
    try msg1.put(allocator, "content", .{ .string = "Hello" });
    defer msg1.deinit(allocator);

    const messages = [_]TemplateInput{
        .{ .map = msg1 },
    };
    try ctx.set("messages", .{ .array = &messages });
    try ctx.set("add_generation_prompt", .{ .boolean = true });

    // Test namespace with for loop and complex condition
    const template =
        \\{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
        \\{%- for message in messages[::-1] %}
        \\{%- set index = (messages|length - 1) - loop.index0 %}
        \\{%- if ns.multi_step_tool and message.role == "user" and message.content is string %}
        \\{%- set ns.multi_step_tool = false %}
        \\{%- set ns.last_query_index = index %}
        \\{%- endif %}
        \\{%- endfor %}
        \\last={{ ns.last_query_index }}
    ;

    const result = try render(allocator, template, &ctx);
    defer allocator.free(result);

    // Should find the user message and set last_query_index to 0
    try std.testing.expect(std.mem.indexOf(u8, result, "last=0") != null);
}

test "escaped quotes in string" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    // Test with escaped quotes in template strings
    const template =
        \\{{- "{\"name\": \"test\"}" }}
    ;

    const result = try render(allocator, template, &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("{\"name\": \"test\"}", result);
}

test "bracket access is defined with missing key" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    // Create a map without 'tools' key
    var item_map = std.StringHashMapUnmanaged(TemplateInput){};
    try item_map.put(allocator, "role", .{ .string = "user" });
    defer item_map.deinit(allocator);

    try ctx.set("item", .{ .map = item_map });

    // Test 1: Key exists
    try item_map.put(allocator, "tools", .{ .array = &.{} });
    {
        const result = try render(allocator, "{% if item['tools'] is defined %}yes{% else %}no{% endif %}", &ctx);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("yes", result);
    }

    // Test 2: Key doesn't exist - should return 'no' not throw error
    _ = item_map.remove("tools");
    {
        const result = try render(allocator, "{% if item['tools'] is defined %}yes{% else %}no{% endif %}", &ctx);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("no", result);
    }

    // Test 3: is undefined with missing key
    {
        const result = try render(allocator, "{% if item['tools'] is undefined %}yes{% else %}no{% endif %}", &ctx);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("yes", result);
    }
}

test "render substitutes simple variables" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();
    try ctx.set("name", .{ .string = "World" });

    const result = try render(allocator, "Hello {{ name }}!", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello World!", result);
}

test "render handles if/else conditionals" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    // Test true branch
    try ctx.set("show", .{ .boolean = true });
    const result_true = try render(allocator, "{% if show %}yes{% else %}no{% endif %}", &ctx);
    defer allocator.free(result_true);
    try std.testing.expectEqualStrings("yes", result_true);

    // Test false branch
    try ctx.set("show", .{ .boolean = false });
    const result_false = try render(allocator, "{% if show %}yes{% else %}no{% endif %}", &ctx);
    defer allocator.free(result_false);
    try std.testing.expectEqualStrings("no", result_false);
}

test "render iterates for loops" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    const items = [_]TemplateInput{
        .{ .string = "a" },
        .{ .string = "b" },
        .{ .string = "c" },
    };
    try ctx.set("items", .{ .array = &items });

    const result = try render(allocator, "{% for x in items %}{{ x }}{% endfor %}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("abc", result);
}

test "render handles integer arithmetic" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    try ctx.set("x", .{ .integer = 10 });
    try ctx.set("y", .{ .integer = 3 });

    const result = try render(allocator, "{{ x + y }}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("13", result);
}

test "render applies filters" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    try ctx.set("text", .{ .string = "hello" });

    const result = try render(allocator, "{{ text | upper }}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("HELLO", result);
}

test "render handles nested map access" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    var inner = std.StringHashMapUnmanaged(TemplateInput){};
    try inner.put(allocator, "name", .{ .string = "test" });
    defer inner.deinit(allocator);

    try ctx.set("obj", .{ .map = inner });

    const result = try render(allocator, "{{ obj.name }}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("test", result);
}

test "render returns error for invalid syntax" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    // Unclosed variable tag - gets past lexer but fails in parser
    const result = render(allocator, "{{ unclosed", &ctx);
    try std.testing.expectError(Error.ParseError, result);
}

test "render handles complex template with messages" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    // Build a chat message
    var msg = std.StringHashMapUnmanaged(TemplateInput){};
    try msg.put(allocator, "role", .{ .string = "user" });
    try msg.put(allocator, "content", .{ .string = "Hello!" });
    defer msg.deinit(allocator);

    const messages = [_]TemplateInput{.{ .map = msg }};
    try ctx.set("messages", .{ .array = &messages });

    const tpl =
        \\{% for m in messages %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>{% endfor %}
    ;

    const result = try render(allocator, tpl, &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("<user>Hello!</user>", result);
}

test "renderWithSpans tracks variable substitutions" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();
    try ctx.set("name", .{ .string = "Alice" });
    try ctx.set("age", .{ .integer = 30 });

    const result = try renderWithSpans(allocator, "Hello {{ name }}, you are {{ age }} years old!", &ctx);
    defer allocator.free(result.output);
    defer allocator.free(result.spans);

    // Output should be correct
    try std.testing.expectEqualStrings("Hello Alice, you are 30 years old!", result.output);

    // Should have spans for: "Hello " (static), "Alice" (var), ", you are " (static), "30" (var), " years old!" (static)
    try std.testing.expectEqual(@as(usize, 5), result.spans.len);

    // Check first span: "Hello " (static)
    try std.testing.expectEqual(@as(usize, 0), result.spans[0].start);
    try std.testing.expectEqual(@as(usize, 6), result.spans[0].end);
    try std.testing.expectEqual(SpanSource.static_text, result.spans[0].source);

    // Check second span: "Alice" (variable: name)
    try std.testing.expectEqual(@as(usize, 6), result.spans[1].start);
    try std.testing.expectEqual(@as(usize, 11), result.spans[1].end);
    try std.testing.expectEqualStrings("name", result.spans[1].source.variable);

    // Check third span: ", you are " (static)
    try std.testing.expectEqual(@as(usize, 11), result.spans[2].start);
    try std.testing.expectEqual(@as(usize, 21), result.spans[2].end);
    try std.testing.expectEqual(SpanSource.static_text, result.spans[2].source);

    // Check fourth span: "30" (variable: age)
    try std.testing.expectEqual(@as(usize, 21), result.spans[3].start);
    try std.testing.expectEqual(@as(usize, 23), result.spans[3].end);
    try std.testing.expectEqualStrings("age", result.spans[3].source.variable);

    // Check fifth span: " years old!" (static)
    try std.testing.expectEqual(@as(usize, 23), result.spans[4].start);
    try std.testing.expectEqual(@as(usize, 34), result.spans[4].end);
    try std.testing.expectEqual(SpanSource.static_text, result.spans[4].source);
}

test "renderWithSpans tracks nested attribute access" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    var user = std.StringHashMapUnmanaged(TemplateInput){};
    try user.put(allocator, "name", .{ .string = "Bob" });
    defer user.deinit(allocator);
    try ctx.set("user", .{ .map = user });

    const result = try renderWithSpans(allocator, "User: {{ user.name }}", &ctx);
    defer allocator.free(result.output);
    defer allocator.free(result.spans);

    try std.testing.expectEqualStrings("User: Bob", result.output);
    try std.testing.expectEqual(@as(usize, 2), result.spans.len);

    // Variable span should have path "user.name"
    try std.testing.expectEqualStrings("user.name", result.spans[1].source.variable);
}

test "renderWithSpans tracks expressions" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();
    try ctx.set("a", .{ .integer = 10 });
    try ctx.set("b", .{ .integer = 5 });

    const result = try renderWithSpans(allocator, "Result: {{ a + b }}", &ctx);
    defer allocator.free(result.output);
    defer allocator.free(result.spans);

    try std.testing.expectEqualStrings("Result: 15", result.output);
    try std.testing.expectEqual(@as(usize, 2), result.spans.len);

    // Expression span (a + b) should be marked as expression, not variable
    try std.testing.expectEqual(SpanSource.expression, result.spans[1].source);
}

test "renderWithSpans tracks filters on variables" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();
    try ctx.set("name", .{ .string = "alice" });

    const result = try renderWithSpans(allocator, "Hello {{ name | upper }}!", &ctx);
    defer allocator.free(result.output);
    defer allocator.free(result.spans);

    try std.testing.expectEqualStrings("Hello ALICE!", result.output);

    // Filter on variable should still attribute to the variable
    try std.testing.expectEqualStrings("name", result.spans[1].source.variable);
}

test "include with string literal" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();
    try ctx.set("name", .{ .string = "World" });

    const result = try render(allocator, "Before{% include \"Hello {{ name }}!\" %}After", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("BeforeHello World!After", result);
}

test "include with variable template" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();
    try ctx.set("header", .{ .string = "=== {{ title }} ===" });
    try ctx.set("title", .{ .string = "My Page" });

    const result = try render(allocator, "{% include header %}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("=== My Page ===", result);
}

test "include accesses parent context" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();
    const items = [_]TemplateInput{ .{ .string = "a" }, .{ .string = "b" }, .{ .string = "c" } };
    try ctx.set("items", .{ .array = &items });
    try ctx.set("loop_tmpl", .{ .string = "{% for item in items %}{{ item }}{% endfor %}" });

    const result = try render(allocator, "{% include loop_tmpl %}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("abc", result);
}

test "include with macros" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();
    try ctx.set("utils", .{ .string = "{% macro greet(name) %}Hello {{ name }}!{% endmacro %}" });

    const result = try render(allocator, "{% include utils %}{{ greet('World') }}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello World!", result);
}
