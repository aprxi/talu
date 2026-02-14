//! Jinja2 Template Parser
//!
//! Converts a token stream into an AST using a Pratt parser for expressions.
//! Uses arena allocation - all nodes are freed together when parser.deinit() is called.

const std = @import("std");
const ast = @import("ast.zig");
const lexer = @import("lexer.zig");

const Token = lexer.Token;
const TokenType = lexer.TokenType;
const Expr = ast.Expr;
const Node = ast.Node;
const BinOp = ast.BinOp;
const UnaryOp = ast.UnaryOp;

pub const ParseError = error{
    UnexpectedToken,
    UnexpectedEof,
    InvalidSyntax,
    OutOfMemory,
    UnclosedBlock,
    InvalidSlice,
};

pub const Parser = struct {
    allocator: std.mem.Allocator, // Parent allocator for result
    arena: std.heap.ArenaAllocator, // Internal arena for AST nodes
    tokens: []const Token,
    pos: usize,
    /// Context for error messages (what was being parsed when error occurred)
    error_context: ?[]const u8 = null,

    pub fn init(allocator: std.mem.Allocator, tokens: []const Token) Parser {
        return .{
            .allocator = allocator,
            .arena = std.heap.ArenaAllocator.init(allocator),
            .tokens = tokens,
            .pos = 0,
            .error_context = null,
        };
    }

    pub fn deinit(self: *Parser) void {
        self.arena.deinit();
    }

    fn alloc(self: *Parser) std.mem.Allocator {
        return self.arena.allocator();
    }

    /// Parse the entire template
    /// Returns a slice allocated from the parent allocator (must be freed by caller)
    pub fn parse(self: *Parser) ParseError![]const *const Node {
        var result = std.ArrayListUnmanaged(*const Node){};
        while (!self.isAtEnd()) {
            if (try self.parseNode()) |n| {
                result.append(self.alloc(), n) catch return ParseError.OutOfMemory;
            }
        }
        // Copy result to parent allocator so caller can free it independently
        const arena_slice = result.items;
        const final = self.allocator.alloc(*const Node, arena_slice.len) catch return ParseError.OutOfMemory;
        @memcpy(final, arena_slice);
        return final;
    }

    fn parseNode(self: *Parser) ParseError!?*const Node {
        return switch (self.peekToken().type) {
            .text => blk: {
                const token = self.peekToken();
                self.advance();
                break :blk try self.allocNode(.{ .text = token.value });
            },
            .print_open => try self.parsePrint(),
            .stmt_open => try self.parseStatement(),
            .eof => null,
            else => ParseError.UnexpectedToken,
        };
    }

    fn parsePrint(self: *Parser) ParseError!*const Node {
        _ = try self.expect(.print_open);
        const expr = try self.parseExpression(0);
        _ = try self.expect(.print_close);
        return try self.allocNode(.{ .print = expr });
    }

    fn parseStatement(self: *Parser) ParseError!*const Node {
        _ = try self.expect(.stmt_open);
        return switch (self.peekToken().type) {
            .kw_if => self.parseIf(),
            .kw_for => self.parseFor(),
            .kw_set => self.parseSet(),
            .kw_macro => self.parseMacro(),
            .kw_break => self.parseBreak(),
            .kw_continue => self.parseContinue(),
            .kw_filter => self.parseFilterBlock(),
            .kw_call => self.parseCallBlock(),
            .kw_generation => self.parseGenerationBlock(),
            .kw_include => self.parseInclude(),
            else => ParseError.UnexpectedToken,
        };
    }

    fn parseIf(self: *Parser) ParseError!*const Node {
        var branches = std.ArrayListUnmanaged(Node.IfBranch){};
        const allocator = self.alloc();

        _ = try self.expect(.kw_if);
        self.error_context = "if"; // Track block type for error messages
        const first_cond = try self.parseExpression(0);
        try self.expectStmtClose();
        const first_body = try self.parseBodyUntil(&.{ .kw_elif, .kw_else, .kw_endif });
        branches.append(allocator, .{ .condition = first_cond, .body = first_body }) catch return ParseError.OutOfMemory;

        while (self.checkStmtKeyword(.kw_elif)) {
            try self.expectStmt(.kw_elif);
            const cond = try self.parseExpression(0);
            try self.expectStmtClose();
            const body = try self.parseBodyUntil(&.{ .kw_elif, .kw_else, .kw_endif });
            branches.append(allocator, .{ .condition = cond, .body = body }) catch return ParseError.OutOfMemory;
        }

        const else_body = try self.parseOptionalElse(.kw_endif);

        try self.expectStmt(.kw_endif);
        try self.expectStmtClose();

        self.error_context = null; // Clear context on success
        return try self.allocNode(.{ .if_stmt = .{
            .branches = branches.toOwnedSlice(allocator) catch return ParseError.OutOfMemory,
            .else_body = else_body,
        } });
    }

    fn parseFor(self: *Parser) ParseError!*const Node {
        _ = try self.expect(.kw_for);
        self.error_context = "for"; // Track block type for error messages
        const target = try self.expectName();
        var target2: ?[]const u8 = null;
        if (self.match(.comma)) target2 = try self.expectName();

        _ = try self.expect(.kw_in);
        // Use no-ternary version so `if` is not consumed for ternary expression
        const iterable = try self.parseExpressionNoTernary(0);

        // Optional if filter: {% for x in items if condition %}
        var filter: ?*const Expr = null;
        if (self.match(.kw_if)) {
            filter = try self.parseExpression(0);
        }

        // Optional recursive: {% for x in items recursive %}
        const recursive = self.match(.kw_recursive);

        try self.expectStmtClose();

        const body = try self.parseBodyUntil(&.{ .kw_else, .kw_endfor });

        const else_body = try self.parseOptionalElse(.kw_endfor);

        try self.expectStmt(.kw_endfor);
        try self.expectStmtClose();

        self.error_context = null; // Clear context on success
        return try self.allocNode(.{ .for_stmt = .{
            .target = target,
            .target2 = target2,
            .iterable = iterable,
            .filter = filter,
            .body = body,
            .else_body = else_body,
            .recursive = recursive,
        } });
    }

    fn parseSet(self: *Parser) ParseError!*const Node {
        _ = try self.expect(.kw_set);
        const name = try self.expectName();
        var namespace: ?[]const u8 = null;
        var target = name;

        if (self.match(.dot)) {
            namespace = name;
            target = try self.expectName();
        }

        _ = try self.expect(.assign);
        const value = try self.parseExpression(0);
        try self.expectStmtClose();

        return try self.allocNode(.{ .set_stmt = .{
            .target = target,
            .namespace = namespace,
            .value = value,
        } });
    }

    fn parseMacro(self: *Parser) ParseError!*const Node {
        _ = try self.expect(.kw_macro);
        self.error_context = "macro"; // Track block type for error messages
        const name = try self.expectName();
        _ = try self.expect(.lparen);
        const allocator = self.alloc();

        var params = std.ArrayListUnmanaged(Node.MacroParam){};
        if (!self.check(.rparen)) {
            while (true) {
                const param_name = try self.expectName();
                var default_val: ?*const Expr = null;
                if (self.match(.assign)) default_val = try self.parseExpression(0);
                params.append(allocator, .{ .name = param_name, .default = default_val }) catch return ParseError.OutOfMemory;
                if (!self.match(.comma)) break;
            }
        }

        _ = try self.expect(.rparen);
        try self.expectStmtClose();
        const body = try self.parseBodyUntil(&.{.kw_endmacro});
        try self.expectStmt(.kw_endmacro);
        try self.expectStmtClose();

        self.error_context = null; // Clear context on success
        return try self.allocNode(.{ .macro_def = .{
            .name = name,
            .params = params.toOwnedSlice(allocator) catch return ParseError.OutOfMemory,
            .body = body,
        } });
    }

    fn parseBreak(self: *Parser) ParseError!*const Node {
        _ = try self.expect(.kw_break);
        try self.expectStmtClose();
        return try self.allocNode(.break_stmt);
    }

    fn parseContinue(self: *Parser) ParseError!*const Node {
        _ = try self.expect(.kw_continue);
        try self.expectStmtClose();
        return try self.allocNode(.continue_stmt);
    }

    fn parseFilterBlock(self: *Parser) ParseError!*const Node {
        _ = try self.expect(.kw_filter);
        self.error_context = "filter"; // Track block type for error messages
        const allocator = self.alloc();

        // Parse filter chain: {% filter upper | trim %}
        var filters = std.ArrayListUnmanaged([]const u8){};
        filters.append(allocator, try self.expectName()) catch return ParseError.OutOfMemory;
        while (self.match(.pipe)) {
            filters.append(allocator, try self.expectName()) catch return ParseError.OutOfMemory;
        }

        try self.expectStmtClose();
        const body = try self.parseBodyUntil(&.{.kw_endfilter});
        try self.expectStmt(.kw_endfilter);
        try self.expectStmtClose();

        self.error_context = null; // Clear context on success
        return try self.allocNode(.{ .filter_block = .{
            .filters = filters.toOwnedSlice(allocator) catch return ParseError.OutOfMemory,
            .body = body,
        } });
    }

    fn parseCallBlock(self: *Parser) ParseError!*const Node {
        _ = try self.expect(.kw_call);
        self.error_context = "call"; // Track block type for error messages
        const macro_name = try self.expectName();
        _ = try self.expect(.lparen);
        const allocator = self.alloc();

        var args = std.ArrayListUnmanaged(*const Expr){};
        if (!self.check(.rparen)) {
            while (true) {
                args.append(allocator, try self.parseExpression(0)) catch return ParseError.OutOfMemory;
                if (!self.match(.comma)) break;
            }
        }

        _ = try self.expect(.rparen);
        try self.expectStmtClose();
        const body = try self.parseBodyUntil(&.{.kw_endcall});
        try self.expectStmt(.kw_endcall);
        try self.expectStmtClose();

        self.error_context = null; // Clear context on success
        return try self.allocNode(.{ .call_block = .{
            .macro_name = macro_name,
            .args = args.toOwnedSlice(allocator) catch return ParseError.OutOfMemory,
            .body = body,
        } });
    }

    /// Parse {% generation %} ... {% endgeneration %}
    /// HuggingFace extension used by MiniMax models
    fn parseGenerationBlock(self: *Parser) ParseError!*const Node {
        _ = try self.expect(.kw_generation);
        self.error_context = "generation"; // Track block type for error messages
        try self.expectStmtClose();
        const body = try self.parseBodyUntil(&.{.kw_endgeneration});
        try self.expectStmt(.kw_endgeneration);
        try self.expectStmtClose();

        self.error_context = null; // Clear context on success
        return try self.allocNode(.{ .generation_block = .{
            .body = body,
        } });
    }

    /// Parse {% include template_expr %}
    /// The template_expr is evaluated at runtime to get the template string.
    /// Example: {% include header_template %}
    ///          {% include "literal template: {{ name }}" %}
    fn parseInclude(self: *Parser) ParseError!*const Node {
        _ = try self.expect(.kw_include);
        const template_expr = try self.parseExpression(0);
        try self.expectStmtClose();

        return try self.allocNode(.{ .include = .{
            .template_expr = template_expr,
        } });
    }

    fn parseBodyUntil(self: *Parser, end_keywords: []const TokenType) ParseError![]const *const Node {
        var nodes = std.ArrayListUnmanaged(*const Node){};
        const allocator = self.alloc();

        while (!self.isAtEnd()) {
            if (self.peekToken().type == .stmt_open and self.pos + 1 < self.tokens.len) {
                const next = self.tokens[self.pos + 1];
                for (end_keywords) |kw| {
                    if (next.type == kw) return nodes.toOwnedSlice(allocator) catch return ParseError.OutOfMemory;
                }
            }
            if (try self.parseNode()) |n| {
                nodes.append(allocator, n) catch return ParseError.OutOfMemory;
            }
        }
        return ParseError.UnclosedBlock;
    }

    // ==== Expression Parser (Pratt Parser) ====

    fn parseExpression(self: *Parser, min_prec: u8) ParseError!*const Expr {
        return self.parseExpressionImpl(min_prec, true);
    }

    fn parseExpressionNoTernary(self: *Parser, min_prec: u8) ParseError!*const Expr {
        return self.parseExpressionImpl(min_prec, false);
    }

    fn parseExpressionImpl(self: *Parser, min_prec: u8, allow_ternary: bool) ParseError!*const Expr {
        var left = try self.parseUnary();

        while (true) {
            const op = self.peekToken();
            const prec = binOpPrecedence(op.type) orelse break;
            if (prec < min_prec) break;
            self.advance();

            // Handle 'is' / 'is not' tests
            if (op.type == .kw_is) {
                const negated = self.match(.kw_not);
                const test_name = try self.expectTestName();
                var args = std.ArrayListUnmanaged(*const Expr){};
                const allocator = self.alloc();
                if (self.match(.lparen)) {
                    while (!self.check(.rparen)) {
                        args.append(allocator, try self.parseExpression(0)) catch return ParseError.OutOfMemory;
                        if (!self.match(.comma)) break;
                    }
                    _ = try self.expect(.rparen);
                }
                left = try self.allocExpr(.{ .test_expr = .{
                    .value = left,
                    .name = test_name,
                    .args = args.toOwnedSlice(allocator) catch return ParseError.OutOfMemory,
                    .negated = negated,
                } });
                continue;
            }

            // Handle 'not in'
            if (op.type == .kw_not) {
                if (self.match(.kw_in)) {
                    const right = try self.parseExpressionNoTernary(prec + 1);
                    left = try self.allocExpr(.{ .binop = .{ .op = .not_in, .left = left, .right = right } });
                    continue;
                }
                return ParseError.UnexpectedToken;
            }

            const bin_op = tokenToBinOp(op.type) orelse unreachable;
            // Power is right-associative: use prec instead of prec + 1
            const right_prec = if (op.type == .starstar) prec else prec + 1;
            // Don't allow ternary in RHS - ternary has lowest precedence
            // so `a + b if x else c` should parse as `(a + b) if x else c`
            const right = try self.parseExpressionNoTernary(right_prec);
            left = try self.allocExpr(.{ .binop = .{ .op = bin_op, .left = left, .right = right } });
        }

        // Handle ternary: `x if cond else y` or `x if cond` (else is optional)
        if (allow_ternary and self.match(.kw_if)) {
            const cond = try self.parseExpression(0);
            // else clause is optional - if missing, false_val is empty string
            const false_val = if (self.match(.kw_else))
                try self.parseExpression(0)
            else
                try self.allocExpr(.{ .string = "" });
            return try self.allocExpr(.{ .conditional = .{
                .test_val = cond,
                .true_val = left,
                .false_val = false_val,
            } });
        }

        return left;
    }

    fn parseUnary(self: *Parser) ParseError!*const Expr {
        if (self.match(.kw_not)) {
            // Parse operand at precedence 3 so that `is` tests (prec 3) bind
            // tighter than prefix `not`. This makes `not x is defined` parse
            // as `not (x is defined)`, matching Jinja2 semantics.
            return try self.allocExpr(.{ .unaryop = .{ .op = .not, .operand = try self.parseExpressionImpl(3, false) } });
        }
        if (self.match(.minus)) {
            return try self.allocExpr(.{ .unaryop = .{ .op = .neg, .operand = try self.parseUnary() } });
        }
        if (self.match(.plus)) {
            return try self.allocExpr(.{ .unaryop = .{ .op = .pos, .operand = try self.parseUnary() } });
        }
        return try self.parsePostfix();
    }

    fn parsePostfix(self: *Parser) ParseError!*const Expr {
        var expr = try self.parsePrimary();
        const allocator = self.alloc();

        while (true) {
            if (self.match(.dot)) {
                expr = try self.parseDotExpr(expr, allocator);
            } else if (self.match(.lbracket)) {
                expr = try self.parseSubscript(expr);
            } else if (self.match(.pipe)) {
                expr = try self.parseFilterExpr(expr, allocator);
            } else if (self.match(.lparen)) {
                expr = try self.parseCallExpr(expr, allocator);
            } else break;
        }
        return expr;
    }

    fn parseDotExpr(self: *Parser, expr: *const Expr, allocator: std.mem.Allocator) ParseError!*const Expr {
        const attr = try self.expectName();
        if (self.match(.lparen)) {
            var args = std.ArrayListUnmanaged(*const Expr){};
            if (!self.check(.rparen)) {
                args.append(allocator, try self.parseExpression(0)) catch return ParseError.OutOfMemory;
                while (self.match(.comma)) {
                    args.append(allocator, try self.parseExpression(0)) catch return ParseError.OutOfMemory;
                }
            }
            _ = try self.expect(.rparen);
            const method = try self.allocExpr(.{ .getattr = .{ .object = expr, .attr = attr } });
            return try self.allocExpr(.{ .call = .{ .func = method, .args = args.toOwnedSlice(allocator) catch return ParseError.OutOfMemory } });
        }
        return try self.allocExpr(.{ .getattr = .{ .object = expr, .attr = attr } });
    }

    fn parseFilterExpr(self: *Parser, expr: *const Expr, allocator: std.mem.Allocator) ParseError!*const Expr {
        const filter_name = try self.expectName();
        var args = std.ArrayListUnmanaged(*const Expr){};
        if (self.match(.lparen)) {
            if (!self.check(.rparen)) {
                try self.parseArgValue(&args);
                while (self.match(.comma)) try self.parseArgValue(&args);
            }
            _ = try self.expect(.rparen);
        }
        return try self.allocExpr(.{ .filter = .{ .value = expr, .name = filter_name, .args = args.toOwnedSlice(allocator) catch return ParseError.OutOfMemory } });
    }

    fn parseCallExpr(self: *Parser, expr: *const Expr, allocator: std.mem.Allocator) ParseError!*const Expr {
        // Check if this is dict() with kwargs
        if (expr.* == .variable and std.mem.eql(u8, expr.variable, "dict")) {
            // Parse like namespace_call but create dict literal
            var pairs = std.ArrayListUnmanaged(Expr.DictPair){};
            if (!self.check(.rparen)) {
                // Check if first arg is a kwarg
                if (self.check(.name) and self.pos + 1 < self.tokens.len and self.tokens[self.pos + 1].type == .assign) {
                    const name = try self.expectName();
                    _ = try self.expect(.assign);
                    const value_expr = try self.parseExpression(0);
                    const key_expr = try self.allocExpr(.{ .string = name });
                    pairs.append(allocator, .{ .key = key_expr, .value = value_expr }) catch return ParseError.OutOfMemory;
                    while (self.match(.comma)) {
                        const kwarg_name = try self.expectName();
                        _ = try self.expect(.assign);
                        const kwarg_value_expr = try self.parseExpression(0);
                        const kwarg_key_expr = try self.allocExpr(.{ .string = kwarg_name });
                        pairs.append(allocator, .{ .key = kwarg_key_expr, .value = kwarg_value_expr }) catch return ParseError.OutOfMemory;
                    }
                    _ = try self.expect(.rparen);
                    return try self.allocExpr(.{ .dict = pairs.toOwnedSlice(allocator) catch return ParseError.OutOfMemory });
                }
                // Regular call (empty dict)
                _ = try self.expect(.rparen);
                return try self.allocExpr(.{ .call = .{ .func = expr, .args = &.{} } });
            }
            _ = try self.expect(.rparen);
            return try self.allocExpr(.{ .call = .{ .func = expr, .args = &.{} } });
        }

        var args = std.ArrayListUnmanaged(*const Expr){};
        if (!self.check(.rparen)) {
            // Kwargs are ignored; parse only values.
            try self.parseArgValue(&args);
            while (self.match(.comma)) {
                try self.parseArgValue(&args);
            }
        }
        _ = try self.expect(.rparen);
        return try self.allocExpr(.{ .call = .{ .func = expr, .args = args.toOwnedSlice(allocator) catch return ParseError.OutOfMemory } });
    }

    fn parseSubscript(self: *Parser, obj: *const Expr) ParseError!*const Expr {
        var start: ?*const Expr = null;
        var stop: ?*const Expr = null;
        var step: ?*const Expr = null;
        var is_slice = false;

        if (!self.check(.colon) and !self.check(.rbracket)) start = try self.parseExpression(0);
        if (self.match(.colon)) {
            is_slice = true;
            if (!self.check(.colon) and !self.check(.rbracket)) stop = try self.parseExpression(0);
            if (self.match(.colon)) {
                if (!self.check(.rbracket)) step = try self.parseExpression(0);
            }
        }
        _ = try self.expect(.rbracket);

        if (is_slice) {
            return try self.allocExpr(.{ .slice = .{ .object = obj, .start = start, .stop = stop, .step = step } });
        }
        return try self.allocExpr(.{ .getitem = .{ .object = obj, .key = start orelse return ParseError.InvalidSlice } });
    }

    fn parsePrimary(self: *Parser) ParseError!*const Expr {
        const token = self.peekToken();
        switch (token.type) {
            .string => {
                self.advance();
                // Handle implicit string concatenation (Jinja2/Python style)
                // Adjacent string literals are concatenated: 'hello' 'world' -> 'helloworld'
                if (self.peekToken().type == .string) {
                    return try self.parseAdjacentStrings(token.value);
                }
                return try self.allocExpr(.{ .string = token.value });
            },
            .integer => {
                self.advance();
                return try self.allocExpr(.{ .integer = std.fmt.parseInt(i64, token.value, 10) catch return ParseError.InvalidSyntax });
            },
            .float => {
                self.advance();
                return try self.allocExpr(.{ .float = std.fmt.parseFloat(f64, token.value) catch return ParseError.InvalidSyntax });
            },
            .kw_true => {
                self.advance();
                return try self.allocExpr(.{ .boolean = true });
            },
            .kw_false => {
                self.advance();
                return try self.allocExpr(.{ .boolean = false });
            },
            .kw_none => {
                self.advance();
                return try self.allocExpr(.none);
            },
            .kw_namespace => {
                // If followed by '(', it's a namespace() call
                // Otherwise, treat as a variable name
                if (self.pos + 1 < self.tokens.len and self.tokens[self.pos + 1].type == .lparen) {
                    self.advance();
                    return try self.parseNamespaceCall();
                }
                // Fall through to treat as variable name
                const tok = self.peekToken();
                self.advance();
                return try self.allocExpr(.{ .variable = tok.value });
            },
            .name,
            .kw_defined,
            // Context-sensitive keywords - can be used as variable names
            .kw_if,
            .kw_elif,
            .kw_else,
            .kw_endif,
            .kw_for,
            .kw_endfor,
            .kw_in,
            .kw_and,
            .kw_or,
            .kw_set,
            .kw_is,
            .kw_macro,
            .kw_endmacro,
            .kw_raw,
            .kw_endraw,
            .kw_break,
            .kw_continue,
            .kw_filter,
            .kw_endfilter,
            .kw_call,
            .kw_endcall,
            .kw_recursive,
            .kw_generation,
            .kw_endgeneration,
            => {
                self.advance();
                return try self.allocExpr(.{ .variable = token.value });
            },
            .lparen => {
                self.advance();
                // Could be grouped expression (x) or tuple (x, y)
                if (self.check(.rparen)) {
                    // Empty tuple ()
                    self.advance();
                    return try self.allocExpr(.{ .list = &.{} });
                }
                const first = try self.parseExpression(0);
                if (self.match(.comma)) {
                    // It's a tuple - collect remaining items
                    const allocator = self.alloc();
                    var items = std.ArrayListUnmanaged(*const Expr){};
                    items.append(allocator, first) catch return ParseError.OutOfMemory;
                    if (!self.check(.rparen)) {
                        items.append(allocator, try self.parseExpression(0)) catch return ParseError.OutOfMemory;
                        while (self.match(.comma)) {
                            if (self.check(.rparen)) break;
                            items.append(allocator, try self.parseExpression(0)) catch return ParseError.OutOfMemory;
                        }
                    }
                    _ = try self.expect(.rparen);
                    return try self.allocExpr(.{ .list = items.toOwnedSlice(allocator) catch return ParseError.OutOfMemory });
                }
                // Just a grouped expression
                _ = try self.expect(.rparen);
                return first;
            },
            .lbracket => {
                self.advance();
                const allocator = self.alloc();
                var items = std.ArrayListUnmanaged(*const Expr){};
                if (!self.check(.rbracket)) {
                    items.append(allocator, try self.parseExpression(0)) catch return ParseError.OutOfMemory;
                    while (self.match(.comma)) {
                        if (self.check(.rbracket)) break;
                        items.append(allocator, try self.parseExpression(0)) catch return ParseError.OutOfMemory;
                    }
                }
                _ = try self.expect(.rbracket);
                return try self.allocExpr(.{ .list = items.toOwnedSlice(allocator) catch return ParseError.OutOfMemory });
            },
            .lbrace => {
                self.advance();
                const allocator = self.alloc();
                var pairs = std.ArrayListUnmanaged(Expr.DictPair){};
                if (!self.check(.rbrace)) {
                    const key_expr = try self.parseExpression(0);
                    _ = try self.expect(.colon);
                    const value_expr = try self.parseExpression(0);
                    pairs.append(allocator, .{ .key = key_expr, .value = value_expr }) catch return ParseError.OutOfMemory;
                    while (self.match(.comma)) {
                        if (self.check(.rbrace)) break;
                        const key_expr_local = try self.parseExpression(0);
                        _ = try self.expect(.colon);
                        const value_expr_local = try self.parseExpression(0);
                        pairs.append(allocator, .{ .key = key_expr_local, .value = value_expr_local }) catch return ParseError.OutOfMemory;
                    }
                }
                _ = try self.expect(.rbrace);
                return try self.allocExpr(.{ .dict = pairs.toOwnedSlice(allocator) catch return ParseError.OutOfMemory });
            },
            else => return ParseError.UnexpectedToken,
        }
    }

    /// Parse adjacent string literals and concatenate them (Jinja2/Python style)
    /// 'hello' 'world' -> 'helloworld'
    fn parseAdjacentStrings(self: *Parser, first: []const u8) ParseError!*const Expr {
        const allocator = self.alloc();

        // Collect all adjacent strings
        var parts = std.ArrayListUnmanaged([]const u8){};
        parts.append(allocator, first) catch return ParseError.OutOfMemory;

        while (self.peekToken().type == .string) {
            const str_token = self.peekToken();
            self.advance();
            parts.append(allocator, str_token.value) catch return ParseError.OutOfMemory;
        }

        // Calculate total length
        var total_len: usize = 0;
        for (parts.items) |part| {
            total_len += part.len;
        }

        // Concatenate all parts
        const result = allocator.alloc(u8, total_len) catch return ParseError.OutOfMemory;
        var offset: usize = 0;
        for (parts.items) |part| {
            @memcpy(result[offset .. offset + part.len], part);
            offset += part.len;
        }

        return try self.allocExpr(.{ .string = result });
    }

    fn parseNamespaceCall(self: *Parser) ParseError!*const Expr {
        _ = try self.expect(.lparen);
        const allocator = self.alloc();
        var args = std.ArrayListUnmanaged(Expr.NamespaceArg){};
        if (!self.check(.rparen)) {
            const name = try self.expectName();
            _ = try self.expect(.assign);
            const value_expr = try self.parseExpression(0);
            args.append(allocator, .{ .name = name, .value = value_expr }) catch return ParseError.OutOfMemory;
            while (self.match(.comma)) {
                const arg_name = try self.expectName();
                _ = try self.expect(.assign);
                const value_expr_local = try self.parseExpression(0);
                args.append(allocator, .{ .name = arg_name, .value = value_expr_local }) catch return ParseError.OutOfMemory;
            }
        }
        _ = try self.expect(.rparen);
        return try self.allocExpr(.{ .namespace_call = args.toOwnedSlice(allocator) catch return ParseError.OutOfMemory });
    }

    fn parseArgValue(self: *Parser, args: *std.ArrayListUnmanaged(*const Expr)) ParseError!void {
        if (self.check(.name) and self.pos + 1 < self.tokens.len and self.tokens[self.pos + 1].type == .assign) {
            self.advance();
            self.advance();
        }
        args.append(self.alloc(), try self.parseExpression(0)) catch return ParseError.OutOfMemory;
    }

    // ==== Helpers ====

    fn expectStmt(self: *Parser, kw: TokenType) ParseError!void {
        _ = try self.expect(.stmt_open);
        _ = try self.expect(kw);
    }

    fn expectStmtClose(self: *Parser) ParseError!void {
        _ = try self.expect(.stmt_close);
    }

    fn parseOptionalElse(self: *Parser, end_kw: TokenType) ParseError![]const *const Node {
        if (self.checkStmtKeyword(.kw_else)) {
            try self.expectStmt(.kw_else);
            try self.expectStmtClose();
            return self.parseBodyUntil(&.{end_kw});
        }
        return &.{};
    }

    fn allocNode(self: *Parser, node: Node) ParseError!*const Node {
        const ptr = self.alloc().create(Node) catch return ParseError.OutOfMemory;
        ptr.* = node;
        return ptr;
    }

    fn allocExpr(self: *Parser, expr: Expr) ParseError!*const Expr {
        const ptr = self.alloc().create(Expr) catch return ParseError.OutOfMemory;
        ptr.* = expr;
        return ptr;
    }

    fn peekToken(self: *Parser) Token {
        return if (self.pos < self.tokens.len) self.tokens[self.pos] else .{ .type = .eof, .value = "", .pos = 0 };
    }

    fn advance(self: *Parser) void {
        if (self.pos < self.tokens.len) self.pos += 1;
    }

    fn check(self: *Parser, t: TokenType) bool {
        return self.peekToken().type == t;
    }

    fn match(self: *Parser, t: TokenType) bool {
        if (self.check(t)) {
            self.advance();
            return true;
        }
        return false;
    }

    fn expect(self: *Parser, t: TokenType) ParseError!Token {
        if (self.check(t)) {
            const token = self.peekToken();
            self.advance();
            return token;
        }
        return ParseError.UnexpectedToken;
    }

    fn expectName(self: *Parser) ParseError![]const u8 {
        const token = self.peekToken();
        // Most keywords can be used as variable names in Jinja2 (context-sensitive)
        // Only constants (true/false/none) and 'not' (unary operator) are reserved
        if (isValidName(token.type)) {
            self.advance();
            return token.value;
        }
        return ParseError.UnexpectedToken;
    }

    /// Expect a test name (used after 'is' / 'is not')
    /// Test names include 'none', 'true', 'false' as they are valid Jinja2 tests
    fn expectTestName(self: *Parser) ParseError![]const u8 {
        const token = self.peekToken();
        // Test names can include constants like 'none', 'true', 'false'
        if (isValidName(token.type) or
            token.type == .kw_true or
            token.type == .kw_false or
            token.type == .kw_none)
        {
            self.advance();
            return token.value;
        }
        return ParseError.UnexpectedToken;
    }

    fn isValidName(tok_type: TokenType) bool {
        return switch (tok_type) {
            // Actual identifier
            .name => true,
            // Constants - can't be variable names
            .kw_true, .kw_false, .kw_none => false,
            // 'not' is a unary operator - can't be used as variable name
            .kw_not => false,
            // All other keywords can be used as variable names (context-sensitive)
            .kw_if, .kw_elif, .kw_else, .kw_endif => true,
            .kw_for, .kw_endfor, .kw_in => true,
            .kw_and, .kw_or => true,
            .kw_set => true,
            .kw_is => true,
            .kw_namespace, .kw_defined => true,
            .kw_macro, .kw_endmacro => true,
            .kw_raw, .kw_endraw => true,
            .kw_break, .kw_continue => true,
            .kw_filter, .kw_endfilter => true,
            .kw_call, .kw_endcall => true,
            .kw_recursive => true,
            .kw_generation, .kw_endgeneration => true,
            // Not a name token
            else => false,
        };
    }

    fn isAtEnd(self: *Parser) bool {
        return self.peekToken().type == .eof;
    }

    fn checkStmtKeyword(self: *Parser, kw: TokenType) bool {
        return self.peekToken().type == .stmt_open and self.pos + 1 < self.tokens.len and self.tokens[self.pos + 1].type == kw;
    }

    fn binOpPrecedence(t: TokenType) ?u8 {
        return switch (t) {
            .kw_or => 1,
            .kw_and => 2,
            .kw_not => 3,
            .kw_in => 3,
            .kw_is => 3,
            .eq, .ne, .lt, .gt, .le, .ge => 4,
            .pipe => 5,
            .tilde => 6,
            .plus, .minus => 7,
            .star, .slash, .slashslash, .percent => 8,
            .starstar => 9,
            else => null,
        };
    }

    fn tokenToBinOp(t: TokenType) ?BinOp {
        return switch (t) {
            .plus => .add,
            .minus => .sub,
            .star => .mul,
            .starstar => .pow,
            .slash => .div,
            .slashslash => .floordiv,
            .percent => .mod,
            .eq => .eq,
            .ne => .ne,
            .lt => .lt,
            .gt => .gt,
            .le => .le,
            .ge => .ge,
            .kw_and => .@"and",
            .kw_or => .@"or",
            .kw_in => .in,
            .tilde => .concat,
            else => null,
        };
    }
};

test "parse simple expression" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ name }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .print);
    try std.testing.expectEqualStrings("name", nodes[0].print.variable);
}

test "parse if statement" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% if x %}yes{% endif %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .if_stmt);
}

test "parse for loop" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% for x in items %}{{ x }}{% endfor %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .for_stmt);
    try std.testing.expectEqualStrings("x", nodes[0].for_stmt.target);
}

test "parse slice" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ items[::-1] }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .slice);
}

test "parse set statement" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% set x = 5 %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .set_stmt);
    try std.testing.expectEqualStrings("x", nodes[0].set_stmt.target);
    try std.testing.expect(nodes[0].set_stmt.value.* == .integer);
    try std.testing.expectEqual(@as(i64, 5), nodes[0].set_stmt.value.integer);
}

test "parse set statement with namespace" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% set ns.counter = 10 %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .set_stmt);
    try std.testing.expectEqualStrings("counter", nodes[0].set_stmt.target);
    try std.testing.expect(nodes[0].set_stmt.namespace != null);
    try std.testing.expectEqualStrings("ns", nodes[0].set_stmt.namespace.?);
    try std.testing.expect(nodes[0].set_stmt.value.* == .integer);
}

test "parse macro definition" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% macro foo(a, b) %}{{ a + b }}{% endmacro %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .macro_def);
    try std.testing.expectEqualStrings("foo", nodes[0].macro_def.name);
    try std.testing.expectEqual(@as(usize, 2), nodes[0].macro_def.params.len);
    try std.testing.expectEqualStrings("a", nodes[0].macro_def.params[0].name);
    try std.testing.expectEqualStrings("b", nodes[0].macro_def.params[1].name);
    try std.testing.expect(nodes[0].macro_def.params[0].default == null);
}

test "parse macro with default parameters" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% macro greet(name, greeting='Hello') %}{{ greeting }} {{ name }}{% endmacro %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .macro_def);
    try std.testing.expectEqual(@as(usize, 2), nodes[0].macro_def.params.len);
    try std.testing.expectEqualStrings("name", nodes[0].macro_def.params[0].name);
    try std.testing.expect(nodes[0].macro_def.params[0].default == null);
    try std.testing.expectEqualStrings("greeting", nodes[0].macro_def.params[1].name);
    try std.testing.expect(nodes[0].macro_def.params[1].default != null);
    try std.testing.expect(nodes[0].macro_def.params[1].default.?.* == .string);
}

test "parse break statement" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% break %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .break_stmt);
}

test "parse continue statement" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% continue %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .continue_stmt);
}

test "parse filter block" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% filter upper %}hello world{% endfilter %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .filter_block);
    try std.testing.expectEqual(@as(usize, 1), nodes[0].filter_block.filters.len);
    try std.testing.expectEqualStrings("upper", nodes[0].filter_block.filters[0]);
}

test "parse filter block with chain" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% filter upper | trim %}  text  {% endfilter %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .filter_block);
    try std.testing.expectEqual(@as(usize, 2), nodes[0].filter_block.filters.len);
    try std.testing.expectEqualStrings("upper", nodes[0].filter_block.filters[0]);
    try std.testing.expectEqualStrings("trim", nodes[0].filter_block.filters[1]);
}

test "parse call block" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% call my_macro(x, y) %}content{% endcall %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .call_block);
    try std.testing.expectEqualStrings("my_macro", nodes[0].call_block.macro_name);
    try std.testing.expectEqual(@as(usize, 2), nodes[0].call_block.args.len);
}

test "parse generation block" {
    // HuggingFace extension: {% generation %}...{% endgeneration %}
    // Used by MiniMaxAI models
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% generation %}hello{% endgeneration %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .generation_block);
    try std.testing.expectEqual(@as(usize, 1), nodes[0].generation_block.body.len);
    try std.testing.expect(nodes[0].generation_block.body[0].* == .text);
}

test "parse include with variable" {
    // {% include template_var %}
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% include header_template %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .include);
    try std.testing.expect(nodes[0].include.template_expr.* == .variable);
}

test "parse include with string literal" {
    // {% include "template string" %}
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% include \"Hello {{ name }}\" %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .include);
    try std.testing.expect(nodes[0].include.template_expr.* == .string);
}

test "parse implicit string concatenation" {
    // Jinja2/Python style: adjacent strings are concatenated
    // 'hello' 'world' -> 'helloworld'
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ 'hello' 'world' }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .print);
    try std.testing.expect(nodes[0].print.* == .string);
    try std.testing.expectEqualStrings("helloworld", nodes[0].print.string);
}

test "parse implicit string concatenation multiple" {
    // Multiple adjacent strings
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ 'a' 'b' 'c' }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .string);
    try std.testing.expectEqualStrings("abc", nodes[0].print.string);
}

test "parse tuple literal" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ (a, b) }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .list);
    try std.testing.expectEqual(@as(usize, 2), nodes[0].print.list.len);
    try std.testing.expect(nodes[0].print.list[0].* == .variable);
    try std.testing.expect(nodes[0].print.list[1].* == .variable);
}

test "parse empty tuple" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ () }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .list);
    try std.testing.expectEqual(@as(usize, 0), nodes[0].print.list.len);
}

test "parse list literal" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ [1, 2, 3] }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .list);
    try std.testing.expectEqual(@as(usize, 3), nodes[0].print.list.len);
    try std.testing.expect(nodes[0].print.list[0].* == .integer);
    try std.testing.expectEqual(@as(i64, 1), nodes[0].print.list[0].integer);
    try std.testing.expectEqual(@as(i64, 2), nodes[0].print.list[1].integer);
    try std.testing.expectEqual(@as(i64, 3), nodes[0].print.list[2].integer);
}

test "parse empty list" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ [] }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .list);
    try std.testing.expectEqual(@as(usize, 0), nodes[0].print.list.len);
}

test "parse dict literal" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ {'a': 1, 'b': 2} }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .dict);
    try std.testing.expectEqual(@as(usize, 2), nodes[0].print.dict.len);
    try std.testing.expect(nodes[0].print.dict[0].key.* == .string);
    try std.testing.expectEqualStrings("a", nodes[0].print.dict[0].key.string);
    try std.testing.expect(nodes[0].print.dict[0].value.* == .integer);
    try std.testing.expectEqual(@as(i64, 1), nodes[0].print.dict[0].value.integer);
}

test "parse empty dict" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ {} }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .dict);
    try std.testing.expectEqual(@as(usize, 0), nodes[0].print.dict.len);
}

test "parse ternary expression" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ x if cond else y }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .conditional);
    try std.testing.expect(nodes[0].print.conditional.true_val.* == .variable);
    try std.testing.expectEqualStrings("x", nodes[0].print.conditional.true_val.variable);
    try std.testing.expect(nodes[0].print.conditional.test_val.* == .variable);
    try std.testing.expectEqualStrings("cond", nodes[0].print.conditional.test_val.variable);
    try std.testing.expect(nodes[0].print.conditional.false_val.* == .variable);
    try std.testing.expectEqualStrings("y", nodes[0].print.conditional.false_val.variable);
}

test "parse ternary precedence with binary operators" {
    // Ternary should have lowest precedence: `a + b if cond else c` -> `(a + b) if cond else c`
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ a + b if cond else c }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    // Should parse as conditional with true_val = (a + b)
    try std.testing.expect(nodes[0].print.* == .conditional);
    // true_val should be a binop (a + b), not just 'b'
    try std.testing.expect(nodes[0].print.conditional.true_val.* == .binop);
    try std.testing.expect(nodes[0].print.conditional.true_val.binop.op == .add);
    try std.testing.expect(nodes[0].print.conditional.true_val.binop.left.* == .variable);
    try std.testing.expectEqualStrings("a", nodes[0].print.conditional.true_val.binop.left.variable);
    try std.testing.expect(nodes[0].print.conditional.true_val.binop.right.* == .variable);
    try std.testing.expectEqualStrings("b", nodes[0].print.conditional.true_val.binop.right.variable);
    // condition should be 'cond'
    try std.testing.expect(nodes[0].print.conditional.test_val.* == .variable);
    try std.testing.expectEqualStrings("cond", nodes[0].print.conditional.test_val.variable);
    // false_val should be 'c'
    try std.testing.expect(nodes[0].print.conditional.false_val.* == .variable);
    try std.testing.expectEqualStrings("c", nodes[0].print.conditional.false_val.variable);
}

test "parse ternary without else clause" {
    // Jinja2 allows `x if cond` without else - defaults to empty string
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ x if cond }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    // Should parse as conditional
    try std.testing.expect(nodes[0].print.* == .conditional);
    // true_val should be 'x'
    try std.testing.expect(nodes[0].print.conditional.true_val.* == .variable);
    try std.testing.expectEqualStrings("x", nodes[0].print.conditional.true_val.variable);
    // condition should be 'cond'
    try std.testing.expect(nodes[0].print.conditional.test_val.* == .variable);
    try std.testing.expectEqualStrings("cond", nodes[0].print.conditional.test_val.variable);
    // false_val should be empty string (default)
    try std.testing.expect(nodes[0].print.conditional.false_val.* == .string);
    try std.testing.expectEqualStrings("", nodes[0].print.conditional.false_val.string);
}

test "parse test expression is defined" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ x is defined }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .test_expr);
    try std.testing.expect(nodes[0].print.test_expr.value.* == .variable);
    try std.testing.expectEqualStrings("x", nodes[0].print.test_expr.value.variable);
    try std.testing.expectEqualStrings("defined", nodes[0].print.test_expr.name);
    try std.testing.expect(!nodes[0].print.test_expr.negated);
}

test "parse test expression is not none" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ x is not none }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .test_expr);
    try std.testing.expectEqualStrings("none", nodes[0].print.test_expr.name);
    try std.testing.expect(nodes[0].print.test_expr.negated);
}

test "parse filter chain" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ name | upper | trim }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .filter);
    try std.testing.expectEqualStrings("trim", nodes[0].print.filter.name);
    try std.testing.expect(nodes[0].print.filter.value.* == .filter);
    try std.testing.expectEqualStrings("upper", nodes[0].print.filter.value.filter.name);
    try std.testing.expect(nodes[0].print.filter.value.filter.value.* == .variable);
    try std.testing.expectEqualStrings("name", nodes[0].print.filter.value.filter.value.variable);
}

test "parse filter with arguments" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ items | join(', ') }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .filter);
    try std.testing.expectEqualStrings("join", nodes[0].print.filter.name);
    try std.testing.expectEqual(@as(usize, 1), nodes[0].print.filter.args.len);
    try std.testing.expect(nodes[0].print.filter.args[0].* == .string);
}

test "parse binary operator addition" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ a + b }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .binop);
    try std.testing.expect(nodes[0].print.binop.op == .add);
    try std.testing.expect(nodes[0].print.binop.left.* == .variable);
    try std.testing.expect(nodes[0].print.binop.right.* == .variable);
}

test "parse binary operator precedence" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ a + b * c }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    // Should parse as: a + (b * c)
    try std.testing.expect(nodes[0].print.* == .binop);
    try std.testing.expect(nodes[0].print.binop.op == .add);
    try std.testing.expect(nodes[0].print.binop.left.* == .variable);
    try std.testing.expectEqualStrings("a", nodes[0].print.binop.left.variable);
    try std.testing.expect(nodes[0].print.binop.right.* == .binop);
    try std.testing.expect(nodes[0].print.binop.right.binop.op == .mul);
}

test "parse binary operator power right associative" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ 2 ** 3 ** 2 }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    // Should parse as: 2 ** (3 ** 2)
    try std.testing.expect(nodes[0].print.* == .binop);
    try std.testing.expect(nodes[0].print.binop.op == .pow);
    try std.testing.expect(nodes[0].print.binop.left.* == .integer);
    try std.testing.expectEqual(@as(i64, 2), nodes[0].print.binop.left.integer);
    try std.testing.expect(nodes[0].print.binop.right.* == .binop);
    try std.testing.expect(nodes[0].print.binop.right.binop.op == .pow);
}

test "parse comparison operators" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ x == y }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .binop);
    try std.testing.expect(nodes[0].print.binop.op == .eq);
}

test "parse logical operators" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ x and y or z }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    // Should parse as: (x and y) or z
    try std.testing.expect(nodes[0].print.* == .binop);
    try std.testing.expect(nodes[0].print.binop.op == .@"or");
    try std.testing.expect(nodes[0].print.binop.left.* == .binop);
    try std.testing.expect(nodes[0].print.binop.left.binop.op == .@"and");
}

test "parse string concatenation" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ 'hello' ~ ' ' ~ 'world' }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .binop);
    try std.testing.expect(nodes[0].print.binop.op == .concat);
}

test "parse floor division" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ x // y }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .binop);
    try std.testing.expect(nodes[0].print.binop.op == .floordiv);
}

test "parse modulo operator" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ x % y }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .binop);
    try std.testing.expect(nodes[0].print.binop.op == .mod);
}

test "parse in operator" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ x in items }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .binop);
    try std.testing.expect(nodes[0].print.binop.op == .in);
}

test "parse not in operator" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ x not in items }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .binop);
    try std.testing.expect(nodes[0].print.binop.op == .not_in);
}

test "parse unary not operator" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ not x }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .unaryop);
    try std.testing.expect(nodes[0].print.unaryop.op == .not);
    try std.testing.expect(nodes[0].print.unaryop.operand.* == .variable);
}

test "parse unary minus operator" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ -x }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .unaryop);
    try std.testing.expect(nodes[0].print.unaryop.op == .neg);
}

test "parse unary plus operator" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ +x }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .unaryop);
    try std.testing.expect(nodes[0].print.unaryop.op == .pos);
}

test "parse getattr" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ obj.attr }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .getattr);
    try std.testing.expect(nodes[0].print.getattr.object.* == .variable);
    try std.testing.expectEqualStrings("obj", nodes[0].print.getattr.object.variable);
    try std.testing.expectEqualStrings("attr", nodes[0].print.getattr.attr);
}

test "parse chained getattr" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ obj.attr.nested }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .getattr);
    try std.testing.expectEqualStrings("nested", nodes[0].print.getattr.attr);
    try std.testing.expect(nodes[0].print.getattr.object.* == .getattr);
    try std.testing.expectEqualStrings("attr", nodes[0].print.getattr.object.getattr.attr);
}

test "parse getitem" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ items[0] }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .getitem);
    try std.testing.expect(nodes[0].print.getitem.object.* == .variable);
    try std.testing.expect(nodes[0].print.getitem.key.* == .integer);
    try std.testing.expectEqual(@as(i64, 0), nodes[0].print.getitem.key.integer);
}

test "parse slice with all parts" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ items[1:5:2] }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .slice);
    try std.testing.expect(nodes[0].print.slice.start != null);
    try std.testing.expect(nodes[0].print.slice.stop != null);
    try std.testing.expect(nodes[0].print.slice.step != null);
    try std.testing.expectEqual(@as(i64, 1), nodes[0].print.slice.start.?.integer);
    try std.testing.expectEqual(@as(i64, 5), nodes[0].print.slice.stop.?.integer);
    try std.testing.expectEqual(@as(i64, 2), nodes[0].print.slice.step.?.integer);
}

test "parse slice with only stop" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ items[:5] }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .slice);
    try std.testing.expect(nodes[0].print.slice.start == null);
    try std.testing.expect(nodes[0].print.slice.stop != null);
    try std.testing.expect(nodes[0].print.slice.step == null);
}

test "parse slice with only start" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ items[5:] }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .slice);
    try std.testing.expect(nodes[0].print.slice.start != null);
    try std.testing.expect(nodes[0].print.slice.stop == null);
}

test "parse method call" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ obj.method(arg) }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .call);
    try std.testing.expect(nodes[0].print.call.func.* == .getattr);
    try std.testing.expectEqual(@as(usize, 1), nodes[0].print.call.args.len);
}

test "parse function call" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ func(a, b, c) }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .call);
    try std.testing.expectEqual(@as(usize, 3), nodes[0].print.call.args.len);
}

test "parse namespace call" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ namespace(x=1, y=2) }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .namespace_call);
    try std.testing.expectEqual(@as(usize, 2), nodes[0].print.namespace_call.len);
    try std.testing.expectEqualStrings("x", nodes[0].print.namespace_call[0].name);
    try std.testing.expectEqualStrings("y", nodes[0].print.namespace_call[1].name);
}

test "parse namespace as variable name" {
    // 'namespace' without '(' should be treated as a variable, not a namespace() call
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ namespace }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .variable);
    try std.testing.expectEqualStrings("namespace", nodes[0].print.variable);
}

test "parse namespace as variable in expression" {
    // 'namespace' in string concatenation should be treated as variable
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ 'prefix' + namespace }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .binop);
    try std.testing.expect(nodes[0].print.binop.right.* == .variable);
    try std.testing.expectEqualStrings("namespace", nodes[0].print.binop.right.variable);
}

test "parse dict from dict call with kwargs" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ dict(a=1, b=2) }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .dict);
    try std.testing.expectEqual(@as(usize, 2), nodes[0].print.dict.len);
}

test "parse if elif else statement" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% if x %}a{% elif y %}b{% elif z %}c{% else %}d{% endif %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .if_stmt);
    try std.testing.expectEqual(@as(usize, 3), nodes[0].if_stmt.branches.len);
    try std.testing.expectEqual(@as(usize, 1), nodes[0].if_stmt.else_body.len);
}

test "parse for with tuple unpacking" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% for k, v in items %}{{ k }}: {{ v }}{% endfor %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .for_stmt);
    try std.testing.expectEqualStrings("k", nodes[0].for_stmt.target);
    try std.testing.expect(nodes[0].for_stmt.target2 != null);
    try std.testing.expectEqualStrings("v", nodes[0].for_stmt.target2.?);
}

test "parse for with filter" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% for x in items if x > 5 %}{{ x }}{% endfor %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .for_stmt);
    try std.testing.expect(nodes[0].for_stmt.filter != null);
    try std.testing.expect(nodes[0].for_stmt.filter.?.* == .binop);
}

test "parse for with else" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% for x in items %}{{ x }}{% else %}empty{% endfor %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .for_stmt);
    try std.testing.expectEqual(@as(usize, 1), nodes[0].for_stmt.else_body.len);
}

test "parse for with recursive" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% for item in items recursive %}{{ item }}{% endfor %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .for_stmt);
    try std.testing.expect(nodes[0].for_stmt.recursive);
}

test "parse nested if statements" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% if x %}{% if y %}nested{% endif %}{% endif %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .if_stmt);
    try std.testing.expectEqual(@as(usize, 1), nodes[0].if_stmt.branches.len);
    try std.testing.expectEqual(@as(usize, 1), nodes[0].if_stmt.branches[0].body.len);
    try std.testing.expect(nodes[0].if_stmt.branches[0].body[0].* == .if_stmt);
}

test "parse nested for loops" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% for x in xs %}{% for y in ys %}{{ x }}{{ y }}{% endfor %}{% endfor %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .for_stmt);
    try std.testing.expectEqual(@as(usize, 1), nodes[0].for_stmt.body.len);
    try std.testing.expect(nodes[0].for_stmt.body[0].* == .for_stmt);
}

test "parse literal values" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ 42 }}{{ 3.14 }}{{ 'text' }}{{ true }}{{ false }}{{ none }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 6), nodes.len);
    try std.testing.expect(nodes[0].print.* == .integer);
    try std.testing.expect(nodes[1].print.* == .float);
    try std.testing.expect(nodes[2].print.* == .string);
    try std.testing.expect(nodes[3].print.* == .boolean);
    try std.testing.expect(nodes[4].print.* == .boolean);
    try std.testing.expect(nodes[5].print.* == .none);
}

test "parse complex expression" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ (a + b) * c }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .binop);
    try std.testing.expect(nodes[0].print.binop.op == .mul);
    try std.testing.expect(nodes[0].print.binop.left.* == .binop);
    try std.testing.expect(nodes[0].print.binop.left.binop.op == .add);
}

test "parse all comparison operators" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ a < b <= c > d >= e != f }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .binop);
}

test "parse subtraction operator" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ a - b }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .binop);
    try std.testing.expect(nodes[0].print.binop.op == .sub);
}

test "parse division operator" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ a / b }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .binop);
    try std.testing.expect(nodes[0].print.binop.op == .div);
}

test "parse grouped expression" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ (x) }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .variable);
}

test "parse trailing comma in tuple" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ (a, b,) }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .list);
    try std.testing.expectEqual(@as(usize, 2), nodes[0].print.list.len);
}

test "parse trailing comma in list" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ [1, 2,] }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .list);
    try std.testing.expectEqual(@as(usize, 2), nodes[0].print.list.len);
}

test "parse trailing comma in dict" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ {'a': 1,} }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .dict);
    try std.testing.expectEqual(@as(usize, 1), nodes[0].print.dict.len);
}

test "parse multiple statements" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "Text{% set x = 1 %}{{ x }}{% if x %}yes{% endif %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 4), nodes.len);
    try std.testing.expect(nodes[0].* == .text);
    try std.testing.expect(nodes[1].* == .set_stmt);
    try std.testing.expect(nodes[2].* == .print);
    try std.testing.expect(nodes[3].* == .if_stmt);
}

test "parse test with arguments" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ x is divisibleby(3) }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .test_expr);
    try std.testing.expectEqualStrings("divisibleby", nodes[0].print.test_expr.name);
    try std.testing.expectEqual(@as(usize, 1), nodes[0].print.test_expr.args.len);
}

test "parse empty function call" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ func() }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .call);
    try std.testing.expectEqual(@as(usize, 0), nodes[0].print.call.args.len);
}

test "parse filter without arguments" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ text | upper }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .filter);
    try std.testing.expectEqualStrings("upper", nodes[0].print.filter.name);
    try std.testing.expectEqual(@as(usize, 0), nodes[0].print.filter.args.len);
}

test "parse error unexpected token" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();

    try std.testing.expectError(ParseError.UnexpectedToken, p.parse());
}

test "parse error unclosed block" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% if x %}content");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();

    try std.testing.expectError(ParseError.UnclosedBlock, p.parse());
}

test "parse keyword 'call' as variable name in for loop" {
    // 'call' is a keyword for {% call %} blocks but should be usable as a variable name
    // This is used in nvidia/NVIDIA-Nemotron templates: {% for call in message.tool_calls %}
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% for call in items %}{{ call }}{% endfor %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .for_stmt);
    try std.testing.expectEqualStrings("call", nodes[0].for_stmt.target);
}

test "parse keyword 'filter' as variable name" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% for filter in items %}{{ filter }}{% endfor %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .for_stmt);
    try std.testing.expectEqualStrings("filter", nodes[0].for_stmt.target);
}

test "parse keyword 'if' as variable name" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% for if in items %}{{ if }}{% endfor %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .for_stmt);
    try std.testing.expectEqualStrings("if", nodes[0].for_stmt.target);
}

test "parse keyword 'for' as variable name" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% for for in items %}{{ for }}{% endfor %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .for_stmt);
    try std.testing.expectEqualStrings("for", nodes[0].for_stmt.target);
}

test "parse keyword 'set' as variable name" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{% for set in items %}{{ set }}{% endfor %}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .for_stmt);
    try std.testing.expectEqualStrings("set", nodes[0].for_stmt.target);
}

test "parse keyword as variable in expression" {
    // Keywords should work as variable references in expressions too
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ call }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .variable);
    try std.testing.expectEqualStrings("call", nodes[0].print.variable);
}

test "parse test name 'none' after is" {
    // 'none' should be valid as a test name: x is none
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ x is none }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .test_expr);
    try std.testing.expectEqualStrings("none", nodes[0].print.test_expr.name);
}

test "parse test name 'true' after is" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ x is true }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .test_expr);
    try std.testing.expectEqualStrings("true", nodes[0].print.test_expr.name);
}

test "parse test name 'false' after is not" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ x is not false }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .test_expr);
    try std.testing.expectEqualStrings("false", nodes[0].print.test_expr.name);
    try std.testing.expect(nodes[0].print.test_expr.negated);
}

test "parse less than operator" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ a < b }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .binop);
    try std.testing.expect(nodes[0].print.binop.op == .lt);
}

test "parse greater than operator" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ a > b }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .binop);
    try std.testing.expect(nodes[0].print.binop.op == .gt);
}

test "parse less than or equal operator" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ a <= b }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .binop);
    try std.testing.expect(nodes[0].print.binop.op == .le);
}

test "parse greater than or equal operator" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ a >= b }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .binop);
    try std.testing.expect(nodes[0].print.binop.op == .ge);
}

test "parse not equal operator" {
    const allocator = std.testing.allocator;
    var lex = lexer.Lexer.init(allocator, "{{ a != b }}");
    defer lex.deinit();
    const tokens = try lex.tokenize();

    var p = Parser.init(allocator, tokens);
    defer p.deinit();
    const nodes = try p.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].print.* == .binop);
    try std.testing.expect(nodes[0].print.binop.op == .ne);
}

test "init creates parser with tokens" {
    const allocator = std.testing.allocator;

    // Create a simple synthetic token array
    const tokens = [_]Token{
        .{ .type = .text, .value = "hello", .pos = 0 },
        .{ .type = .eof, .value = "", .pos = 5 },
    };

    var p = Parser.init(allocator, &tokens);
    defer p.deinit();

    // Verify parser was initialized correctly
    try std.testing.expectEqual(@as(usize, 0), p.pos);
    try std.testing.expectEqual(@as(usize, 2), p.tokens.len);
    try std.testing.expect(p.tokens[0].type == .text);
    try std.testing.expectEqualStrings("hello", p.tokens[0].value);
    try std.testing.expect(p.tokens[1].type == .eof);
}

test "deinit cleans up parser" {
    const allocator = std.testing.allocator;

    // Create a simple synthetic token array
    const tokens = [_]Token{
        .{ .type = .print_open, .value = "{{", .pos = 0 },
        .{ .type = .name, .value = "x", .pos = 2 },
        .{ .type = .print_close, .value = "}}", .pos = 3 },
        .{ .type = .eof, .value = "", .pos = 5 },
    };

    var p = Parser.init(allocator, &tokens);

    // Parse some nodes to allocate memory in the arena
    const nodes = try p.parse();
    defer allocator.free(nodes);

    // Verify we got expected result
    try std.testing.expectEqual(@as(usize, 1), nodes.len);
    try std.testing.expect(nodes[0].* == .print);

    // deinit should clean up the arena (no memory leaks)
    p.deinit();
}
