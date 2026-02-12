//! Abstract Syntax Tree for Jinja2 templates
//!
//! Represents the parsed structure of a template. Nodes can be:
//! - Text: raw text to output
//! - Print: expression to evaluate and output
//! - If/For/Set: control flow statements
//!
//! Expressions handle the logic inside {{ }} and {% %}:
//! - Variables, literals, operators, method calls, filters

const std = @import("std");

/// A span in the source template for error reporting
pub const Span = struct {
    start: usize,
    end: usize,
};

/// Expression nodes - the logic inside {{ }} and {% %}
pub const Expr = union(enum) {
    /// Literal values
    string: []const u8,
    integer: i64,
    float: f64,
    boolean: bool,
    none,

    /// Variable reference: `foo`
    variable: []const u8,

    /// Property access: `foo.bar`
    getattr: struct {
        object: *const Expr,
        attr: []const u8,
    },

    /// Subscript: `foo[0]` or `foo['key']`
    getitem: struct {
        object: *const Expr,
        key: *const Expr,
    },

    /// Slice: `foo[start:stop:step]` - any part can be null
    slice: struct {
        object: *const Expr,
        start: ?*const Expr,
        stop: ?*const Expr,
        step: ?*const Expr,
    },

    /// Binary operations: +, -, *, /, ==, !=, <, >, <=, >=, and, or, in
    binop: struct {
        op: BinOp,
        left: *const Expr,
        right: *const Expr,
    },

    /// Unary operations: not, -
    unaryop: struct {
        op: UnaryOp,
        operand: *const Expr,
    },

    /// Method call: `foo.bar(args)`
    call: struct {
        func: *const Expr,
        args: []const *const Expr,
    },

    /// Filter: `foo | bar` or `foo | bar(args)`
    filter: struct {
        value: *const Expr,
        name: []const u8,
        args: []const *const Expr,
    },

    /// Test: `foo is bar` or `foo is bar(args)`
    test_expr: struct {
        value: *const Expr,
        name: []const u8,
        args: []const *const Expr,
        negated: bool, // `is not`
    },

    /// Conditional expression: `foo if cond else bar`
    conditional: struct {
        test_val: *const Expr,
        true_val: *const Expr,
        false_val: *const Expr,
    },

    /// List literal: `[a, b, c]`
    list: []const *const Expr,

    /// Dict literal: `{a: b, c: d}`
    dict: []const DictPair,

    /// Namespace call: `namespace(key=value, ...)`
    namespace_call: []const NamespaceArg,

    /// Macro call: `macro_name(args, kwarg=value)`
    macro_call: struct {
        name: []const u8,
        args: []const *const Expr,
        kwargs: []const NamespaceArg,
    },

    pub const DictPair = struct {
        key: *const Expr,
        value: *const Expr,
    };

    pub const NamespaceArg = struct {
        name: []const u8,
        value: *const Expr,
    };
};

pub const BinOp = enum {
    add, // +
    sub, // -
    mul, // *
    div, // /
    floordiv, // //
    mod, // %
    pow, // **
    eq, // ==
    ne, // !=
    lt, // <
    gt, // >
    le, // <=
    ge, // >=
    @"and", // and
    @"or", // or
    in, // in
    not_in, // not in
    concat, // ~
};

pub const UnaryOp = enum {
    not,
    neg,
    pos,
};

/// Statement/template nodes
pub const Node = union(enum) {
    /// Raw text to output
    text: []const u8,

    /// Print expression: {{ expr }}
    print: *const Expr,

    /// If statement: {% if %} ... {% elif %} ... {% else %} ... {% endif %}
    if_stmt: struct {
        branches: []const IfBranch,
        else_body: []const *const Node,
    },

    /// For loop: {% for x in items %} ... {% endfor %}
    for_stmt: struct {
        target: []const u8, // loop variable name
        target2: ?[]const u8, // second var for `for k, v in dict`
        iterable: *const Expr,
        filter: ?*const Expr, // optional `if` filter condition
        body: []const *const Node,
        else_body: []const *const Node, // {% else %} branch
        recursive: bool = false, // {% for ... recursive %}
    },

    /// Set statement: {% set x = expr %}
    set_stmt: struct {
        target: []const u8,
        namespace: ?[]const u8, // for `ns.foo = bar`
        value: *const Expr,
    },

    /// Macro definition: {% macro name(args) %} ... {% endmacro %}
    macro_def: struct {
        name: []const u8,
        params: []const MacroParam,
        body: []const *const Node,
    },

    /// Macro call statement: {{ macro_name(args) }}
    macro_call_stmt: struct {
        name: []const u8,
        args: []const *const Expr,
        kwargs: []const Expr.NamespaceArg,
    },

    /// Break statement: {% break %}
    break_stmt,

    /// Continue statement: {% continue %}
    continue_stmt,

    /// Filter block: {% filter name %} ... {% endfilter %}
    filter_block: struct {
        filters: []const []const u8, // ordered filter names
        body: []const *const Node,
    },

    /// Call block: {% call name() %} ... {% endcall %}
    call_block: struct {
        macro_name: []const u8,
        args: []const *const Expr,
        body: []const *const Node,
    },

    /// Generation block: {% generation %} ... {% endgeneration %}
    /// HuggingFace extension used by MiniMax models to mark generated content
    generation_block: struct {
        body: []const *const Node,
    },

    /// Include statement: {% include template_expr %}
    /// Includes another template by evaluating the expression as a template string.
    /// The included template has access to the parent template's context.
    include: struct {
        template_expr: *const Expr,
    },

    pub const IfBranch = struct {
        condition: *const Expr,
        body: []const *const Node,
    };

    pub const MacroParam = struct {
        name: []const u8,
        default: ?*const Expr, // optional default value
    };
};

/// Root of a parsed template
pub const Template = struct {
    nodes: []const *const Node,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Template) void {
        // All nodes are arena-allocated, just discard the allocator
        _ = self;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "Template.deinit is safe to call" {
    // deinit is a no-op since nodes are arena-allocated
    // This test ensures the function exists and can be called
    var template = Template{
        .nodes = &.{},
        .allocator = std.testing.allocator,
    };
    template.deinit();
}
