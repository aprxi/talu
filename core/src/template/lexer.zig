//! Jinja2 Template Lexer
//!
//! Tokenizes a template string into a stream of tokens.
//! Handles the different Jinja2 delimiters:
//! - {{ ... }} for expressions (print)
//! - {% ... %} for statements
//! - {# ... #} for comments
//!
//! Also handles whitespace control with - suffix/prefix:
//! - {{- ... -}} strips whitespace

const std = @import("std");

pub const TokenType = enum {
    // Template structure
    text, // Raw text between tags
    print_open, // {{ or {{-
    print_close, // }} or -}}
    stmt_open, // {% or {%-
    stmt_close, // %} or -%}

    // Literals
    string, // 'hello' or "hello"
    integer, // 123
    float, // 1.23
    name, // variable name or keyword

    // Keywords (identified from name during lexing)
    kw_if,
    kw_elif,
    kw_else,
    kw_endif,
    kw_for,
    kw_endfor,
    kw_in,
    kw_not,
    kw_and,
    kw_or,
    kw_set,
    kw_true,
    kw_false,
    kw_none,
    kw_is,
    kw_namespace,
    kw_defined,
    kw_macro,
    kw_endmacro,
    kw_raw,
    kw_endraw,
    kw_break,
    kw_continue,
    kw_filter,
    kw_endfilter,
    kw_call,
    kw_endcall,
    kw_recursive,
    kw_generation,
    kw_endgeneration,
    kw_include,

    // Operators
    lparen, // (
    rparen, // )
    lbracket, // [
    rbracket, // ]
    lbrace, // {
    rbrace, // }
    dot, // .
    comma, // ,
    colon, // :
    pipe, // |
    tilde, // ~
    plus, // +
    minus, // -
    star, // *
    starstar, // **
    slash, // /
    slashslash, // //
    percent, // %
    eq, // ==
    ne, // !=
    lt, // <
    gt, // >
    le, // <=
    ge, // >=
    assign, // =

    // End of input
    eof,
};

pub const Token = struct {
    type: TokenType,
    value: []const u8,
    pos: usize,
    trim_left: bool = false, // For {{- or {%-
    trim_right: bool = false, // For -}} or -%}
};

pub const LexerError = error{
    UnterminatedString,
    UnterminatedTag,
    InvalidCharacter,
    UnterminatedComment,
};

pub const Lexer = struct {
    allocator: std.mem.Allocator,
    source: []const u8,
    pos: usize,
    tokens: std.ArrayListUnmanaged(Token),

    // State machine
    in_tag: bool,
    tag_type: enum { none, print, stmt },

    pub fn init(allocator: std.mem.Allocator, source: []const u8) Lexer {
        return .{
            .allocator = allocator,
            .source = source,
            .pos = 0,
            .tokens = .{},
            .in_tag = false,
            .tag_type = .none,
        };
    }

    pub fn deinit(self: *Lexer) void {
        self.tokens.deinit(self.allocator);
    }

    pub fn tokenize(self: *Lexer) LexerError![]const Token {
        while (self.pos < self.source.len) {
            if (self.in_tag) {
                try self.lexInsideTag();
            } else {
                try self.lexOutsideTag();
            }
        }

        self.tokens.append(self.allocator, .{
            .type = .eof,
            .value = "",
            .pos = self.pos,
        }) catch return LexerError.UnterminatedTag;

        return self.tokens.items;
    }

    fn lexOutsideTag(self: *Lexer) LexerError!void {
        const start = self.pos;

        while (self.pos < self.source.len) {
            // Check for tag opening
            if (self.pos + 1 < self.source.len) {
                const first_char = self.source[self.pos];
                const second_char = self.source[self.pos + 1];

                if (first_char == '{' and (second_char == '{' or second_char == '%' or second_char == '#')) {
                    // Emit any text before this tag
                    try self.emitText(start);

                    if (second_char == '#') {
                        // Comment - skip to #}
                        try self.skipComment();
                    } else {
                        // Start of print or statement tag
                        try self.lexTagOpen();
                    }
                    return;
                }
            }
            self.pos += 1;
        }

        // Emit remaining text
        try self.emitText(start);
    }

    fn skipComment(self: *Lexer) LexerError!void {
        const comment_start = self.pos;
        self.pos += 2; // Skip {#

        // Check for trim_left: {#-
        if (self.pos < self.source.len and self.source[self.pos] == '-') {
            self.pos += 1;
            // Trim trailing whitespace from previous text token
            self.trimPreviousText();
        } else {
            // lstrip_blocks: strip line-leading whitespace before {# tags
            self.lstripPreviousText(comment_start);
        }

        // Find end of comment
        var trim_right = false;
        while (self.pos + 1 < self.source.len) {
            if (self.source[self.pos] == '-' and
                self.pos + 2 < self.source.len and
                self.source[self.pos + 1] == '#' and
                self.source[self.pos + 2] == '}')
            {
                trim_right = true;
                self.pos += 3; // Skip -#}
                break;
            } else if (self.source[self.pos] == '#' and self.source[self.pos + 1] == '}') {
                self.pos += 2; // Skip #}
                break;
            }
            self.pos += 1;
        } else {
            return LexerError.UnterminatedComment;
        }

        // Handle whitespace after comment
        if (trim_right) {
            // Skip all whitespace after comment
            self.skipWhitespaceChars();
        } else {
            // Apply trim_blocks: skip single newline after comment
            self.skipSingleNewline();
        }
    }

    fn lexTagOpen(self: *Lexer) LexerError!void {
        const start = self.pos;
        const c1 = self.source[self.pos + 1];

        self.pos += 2; // Skip {{ or {%

        // Check for whitespace trim
        var trim_left = false;
        if (self.pos < self.source.len and self.source[self.pos] == '-') {
            trim_left = true;
            self.pos += 1;
            // Apply trim to previous text token if any
            self.trimPreviousText();
        } else if (c1 == '%') {
            // lstrip_blocks: strip line-leading whitespace before {% tags
            self.lstripPreviousText(start);
        }

        const tok_type: TokenType = if (c1 == '{') .print_open else .stmt_open;
        self.tokens.append(self.allocator, .{
            .type = tok_type,
            .value = self.source[start..self.pos],
            .pos = start,
            .trim_left = trim_left,
        }) catch return;

        self.in_tag = true;
        self.tag_type = if (c1 == '{') .print else .stmt;
    }

    fn lexInsideTag(self: *Lexer) LexerError!void {
        self.skipWhitespace();

        if (self.pos >= self.source.len) {
            return LexerError.UnterminatedTag;
        }

        const current_char = self.source[self.pos];

        // Check for tag close
        if (current_char == '-' or current_char == '%' or current_char == '}') {
            if (try self.tryLexTagClose()) {
                return;
            }
        }

        // String literal
        if (current_char == '\'' or current_char == '"') {
            try self.lexString();
            return;
        }

        // Number
        if (std.ascii.isDigit(current_char) or (current_char == '-' and self.pos + 1 < self.source.len and std.ascii.isDigit(self.source[self.pos + 1]))) {
            try self.lexNumber();
            return;
        }

        // Name or keyword
        if (std.ascii.isAlphabetic(current_char) or current_char == '_') {
            try self.lexName();
            return;
        }

        // Operators
        try self.lexOperator();
    }

    fn tryLexTagClose(self: *Lexer) LexerError!bool {
        const start = self.pos;
        var trim_right = false;

        // Check for -}} or -%}
        if (self.source[self.pos] == '-') {
            if (self.pos + 2 < self.source.len) {
                const c1 = self.source[self.pos + 1];
                const c2 = self.source[self.pos + 2];
                if ((self.tag_type == .print and c1 == '}' and c2 == '}') or
                    (self.tag_type == .stmt and c1 == '%' and c2 == '}'))
                {
                    trim_right = true;
                    self.pos += 3;
                } else {
                    return false;
                }
            } else {
                return false;
            }
        } else if (self.tag_type == .print and self.source[self.pos] == '}') {
            if (self.pos + 1 < self.source.len and self.source[self.pos + 1] == '}') {
                self.pos += 2;
            } else {
                return false;
            }
        } else if (self.tag_type == .stmt and self.source[self.pos] == '%') {
            if (self.pos + 1 < self.source.len and self.source[self.pos + 1] == '}') {
                self.pos += 2;
            } else {
                return false;
            }
        } else {
            return false;
        }

        const tok_type: TokenType = if (self.tag_type == .print) .print_close else .stmt_close;
        self.tokens.append(self.allocator, .{
            .type = tok_type,
            .value = self.source[start..self.pos],
            .pos = start,
            .trim_right = trim_right,
        }) catch return false;

        self.in_tag = false;
        self.tag_type = .none;

        // Handle whitespace trimming for next text
        if (trim_right) {
            self.skipWhitespaceChars();
        } else if (tok_type == .stmt_close) {
            // trim_blocks behavior: skip single newline after statement close
            // This matches transformers/Jinja2 default behavior
            self.skipSingleNewline();
        }

        // Check if we just closed a {% raw %} block - if so, capture until {% endraw %}
        // Look back for: stmt_open, kw_raw, stmt_close
        const items = self.tokens.items;
        if (items.len >= 3) {
            const idx = items.len - 1;
            if (items[idx].type == .stmt_close and
                items[idx - 1].type == .kw_raw and
                items[idx - 2].type == .stmt_open)
            {
                // Remove the raw tokens from output - raw block produces only text
                self.tokens.shrinkRetainingCapacity(items.len - 3);
                try self.lexRawBlock();
            }
        }

        return true;
    }

    fn lexString(self: *Lexer) LexerError!void {
        const quote = self.source[self.pos];
        const start = self.pos;
        self.pos += 1;

        while (self.pos < self.source.len) {
            const current_char = self.source[self.pos];
            if (current_char == quote) {
                self.pos += 1;
                self.tokens.append(self.allocator, .{
                    .type = .string,
                    .value = self.source[start + 1 .. self.pos - 1], // Exclude quotes
                    .pos = start,
                }) catch return LexerError.UnterminatedString;
                return;
            }
            if (current_char == '\\' and self.pos + 1 < self.source.len) {
                self.pos += 2; // Skip escape sequence
            } else {
                self.pos += 1;
            }
        }

        return LexerError.UnterminatedString;
    }

    fn lexNumber(self: *Lexer) LexerError!void {
        const start = self.pos;
        var is_float = false;

        if (self.source[self.pos] == '-') {
            self.pos += 1;
        }

        while (self.pos < self.source.len) {
            const current_char = self.source[self.pos];
            if (std.ascii.isDigit(current_char)) {
                self.pos += 1;
            } else if (current_char == '.' and !is_float) {
                is_float = true;
                self.pos += 1;
            } else {
                break;
            }
        }

        self.tokens.append(self.allocator, .{
            .type = if (is_float) .float else .integer,
            .value = self.source[start..self.pos],
            .pos = start,
        }) catch return;
    }

    fn lexName(self: *Lexer) LexerError!void {
        const start = self.pos;

        while (self.pos < self.source.len) {
            const current_char = self.source[self.pos];
            if (std.ascii.isAlphanumeric(current_char) or current_char == '_') {
                self.pos += 1;
            } else {
                break;
            }
        }

        const name = self.source[start..self.pos];
        const tok_type = keywordType(name) orelse .name;

        self.tokens.append(self.allocator, .{
            .type = tok_type,
            .value = name,
            .pos = start,
        }) catch return;
    }

    fn lexOperator(self: *Lexer) LexerError!void {
        const start = self.pos;
        const first_char = self.source[self.pos];

        // Two-character operators
        if (self.pos + 1 < self.source.len) {
            const second_char = self.source[self.pos + 1];
            const tok_type: ?TokenType = switch (first_char) {
                '=' => if (second_char == '=') .eq else null,
                '!' => if (second_char == '=') .ne else null,
                '<' => if (second_char == '=') .le else null,
                '>' => if (second_char == '=') .ge else null,
                '*' => if (second_char == '*') .starstar else null,
                '/' => if (second_char == '/') .slashslash else null,
                else => null,
            };
            if (tok_type) |tt| {
                self.pos += 2;
                self.tokens.append(self.allocator, .{
                    .type = tt,
                    .value = self.source[start..self.pos],
                    .pos = start,
                }) catch return LexerError.InvalidCharacter;
                return;
            }
        }

        // Single-character operators
        const tok_type: TokenType = switch (first_char) {
            '(' => .lparen,
            ')' => .rparen,
            '[' => .lbracket,
            ']' => .rbracket,
            '{' => .lbrace,
            '}' => .rbrace,
            '.' => .dot,
            ',' => .comma,
            ':' => .colon,
            '|' => .pipe,
            '~' => .tilde,
            '+' => .plus,
            '-' => .minus,
            '*' => .star,
            '/' => .slash,
            '%' => .percent,
            '<' => .lt,
            '>' => .gt,
            '=' => .assign,
            else => return LexerError.InvalidCharacter,
        };

        self.pos += 1;
        self.tokens.append(self.allocator, .{
            .type = tok_type,
            .value = self.source[start..self.pos],
            .pos = start,
        }) catch return LexerError.InvalidCharacter;
    }

    fn skipWhitespace(self: *Lexer) void {
        while (self.pos < self.source.len) {
            const current_char = self.source[self.pos];
            if (isWhitespace(current_char)) {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn skipWhitespaceChars(self: *Lexer) void {
        while (self.pos < self.source.len and isWhitespace(self.source[self.pos])) {
            self.pos += 1;
        }
    }

    fn skipSingleNewline(self: *Lexer) void {
        if (self.pos < self.source.len and self.source[self.pos] == '\n') {
            self.pos += 1;
        } else if (self.pos + 1 < self.source.len and
            self.source[self.pos] == '\r' and self.source[self.pos + 1] == '\n')
        {
            self.pos += 2;
        }
    }

    fn trimPreviousText(self: *Lexer) void {
        if (self.tokens.items.len == 0) return;
        const last = &self.tokens.items[self.tokens.items.len - 1];
        if (last.type == .text) {
            last.value = std.mem.trimRight(u8, last.value, " \t\n\r");
        }
    }

    /// lstrip_blocks: strip spaces/tabs from the current line in the previous
    /// text token, but only if the entire source line up to the block tag
    /// contains nothing but whitespace. Matches Jinja2 lstrip_blocks=True.
    fn lstripPreviousText(self: *Lexer, tag_start: usize) void {
        if (self.tokens.items.len == 0) return;
        const last = &self.tokens.items[self.tokens.items.len - 1];
        if (last.type != .text) return;
        const text = last.value;
        if (text.len == 0) return;

        // Find start of line in source (position after last newline, or 0)
        var line_start = tag_start;
        while (line_start > 0) {
            if (self.source[line_start - 1] == '\n') break;
            line_start -= 1;
        }

        // Only strip if the entire line prefix (source[line_start..tag_start])
        // is spaces/tabs — any non-whitespace means the tag is mid-line
        for (self.source[line_start..tag_start]) |c| {
            if (c != ' ' and c != '\t') return;
        }

        // Strip the trailing whitespace from the text token
        const ws_len = tag_start - line_start;
        if (ws_len > 0 and ws_len <= text.len) {
            last.value = text[0 .. text.len - ws_len];
        }
    }

    fn isWhitespace(c: u8) bool {
        return c == ' ' or c == '\t' or c == '\n' or c == '\r';
    }

    fn keywordType(name: []const u8) ?TokenType {
        const map = std.StaticStringMap(TokenType).initComptime(.{
            .{ "if", .kw_if },
            .{ "elif", .kw_elif },
            .{ "else", .kw_else },
            .{ "endif", .kw_endif },
            .{ "for", .kw_for },
            .{ "endfor", .kw_endfor },
            .{ "in", .kw_in },
            .{ "not", .kw_not },
            .{ "and", .kw_and },
            .{ "or", .kw_or },
            .{ "set", .kw_set },
            .{ "true", .kw_true },
            .{ "True", .kw_true },
            .{ "false", .kw_false },
            .{ "False", .kw_false },
            .{ "none", .kw_none },
            .{ "None", .kw_none },
            .{ "is", .kw_is },
            .{ "namespace", .kw_namespace },
            .{ "defined", .kw_defined },
            .{ "macro", .kw_macro },
            .{ "endmacro", .kw_endmacro },
            .{ "raw", .kw_raw },
            .{ "endraw", .kw_endraw },
            .{ "break", .kw_break },
            .{ "continue", .kw_continue },
            .{ "filter", .kw_filter },
            .{ "endfilter", .kw_endfilter },
            .{ "call", .kw_call },
            .{ "endcall", .kw_endcall },
            .{ "recursive", .kw_recursive },
            .{ "generation", .kw_generation },
            .{ "endgeneration", .kw_endgeneration },
            .{ "include", .kw_include },
        });
        return map.get(name);
    }

    fn emitText(self: *Lexer, start: usize) LexerError!void {
        if (self.pos > start) {
            self.tokens.append(self.allocator, .{
                .type = .text,
                .value = self.source[start..self.pos],
                .pos = start,
            }) catch return LexerError.UnterminatedTag;
        }
    }

    fn tryLexRawEnd(self: *Lexer, start: usize, pattern: []const u8, trim_right: bool) LexerError!bool {
        const remaining = self.source[self.pos..];
        if (remaining.len < pattern.len or !std.mem.startsWith(u8, remaining, pattern)) {
            return false;
        }

        try self.emitText(start);
        self.pos += pattern.len;

        if (trim_right) {
            self.skipWhitespaceChars();
        }

        return true;
    }

    /// Handle {% raw %} block - collect everything until {% endraw %} as text
    fn lexRawBlock(self: *Lexer) LexerError!void {
        const start = self.pos;
        const endraw = "{% endraw %}";
        const endraw_trim = "{%- endraw -%}";
        const endraw_trim_left = "{%- endraw %}";
        const endraw_trim_right = "{% endraw -%}";

        while (self.pos < self.source.len) {
            // Check for any form of {% endraw %}
            if (try self.tryLexRawEnd(start, endraw, false)) return;
            if (try self.tryLexRawEnd(start, endraw_trim, true)) return;
            if (try self.tryLexRawEnd(start, endraw_trim_left, false)) return;
            if (try self.tryLexRawEnd(start, endraw_trim_right, true)) return;

            self.pos += 1;
        }

        return LexerError.UnterminatedTag;
    }
};

test "tokenize basic" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "Hello {{ name }}!");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.text, tokens[0].type);
    try std.testing.expectEqualStrings("Hello ", tokens[0].value);

    try std.testing.expectEqual(TokenType.print_open, tokens[1].type);
    try std.testing.expectEqual(TokenType.name, tokens[2].type);
    try std.testing.expectEqualStrings("name", tokens[2].value);
    try std.testing.expectEqual(TokenType.print_close, tokens[3].type);

    try std.testing.expectEqual(TokenType.text, tokens[4].type);
    try std.testing.expectEqualStrings("!", tokens[4].value);
}

test "tokenize whitespace trim" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "Hello {{- name -}} World");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    // Text should be trimmed
    try std.testing.expectEqual(TokenType.text, tokens[0].type);
    try std.testing.expectEqualStrings("Hello", tokens[0].value);
}

test "tokenize statement" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{% if x == 1 %}yes{% endif %}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.stmt_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.kw_if, tokens[1].type);
    try std.testing.expectEqual(TokenType.name, tokens[2].type);
    try std.testing.expectEqual(TokenType.eq, tokens[3].type);
    try std.testing.expectEqual(TokenType.integer, tokens[4].type);
    try std.testing.expectEqual(TokenType.stmt_close, tokens[5].type);
}

test "tokenize raw block" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{% raw %}{{ not_a_var }}{% endraw %}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    // Raw block should produce just text
    try std.testing.expectEqual(TokenType.text, tokens[0].type);
    try std.testing.expectEqualStrings("{{ not_a_var }}", tokens[0].value);
    try std.testing.expectEqual(TokenType.eof, tokens[1].type);
}

test "tokenize comments" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "Before{# this is a comment #}After");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    // Comments should be stripped
    try std.testing.expectEqual(TokenType.text, tokens[0].type);
    try std.testing.expectEqualStrings("Before", tokens[0].value);
    try std.testing.expectEqual(TokenType.text, tokens[1].type);
    try std.testing.expectEqualStrings("After", tokens[1].value);
    try std.testing.expectEqual(TokenType.eof, tokens[2].type);
}

test "tokenize comments with whitespace trim" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "Before  {#- comment -#}  After");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    // Whitespace should be trimmed around comment
    try std.testing.expectEqual(TokenType.text, tokens[0].type);
    try std.testing.expectEqualStrings("Before", tokens[0].value);
    try std.testing.expectEqual(TokenType.text, tokens[1].type);
    try std.testing.expectEqualStrings("After", tokens[1].value);
}

test "tokenize string literals with escapes" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{{ 'hello\\nworld' }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.string, tokens[1].type);
    try std.testing.expectEqualStrings("hello\\nworld", tokens[1].value);
    try std.testing.expectEqual(TokenType.print_close, tokens[2].type);
}

test "tokenize string with escaped quotes" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{{ \"escaped \\\"quote\\\"\" }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.string, tokens[1].type);
    try std.testing.expectEqualStrings("escaped \\\"quote\\\"", tokens[1].value);
    try std.testing.expectEqual(TokenType.print_close, tokens[2].type);
}

test "tokenize arithmetic operators" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{{ a + b - c * d / e // f % g ** h }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.name, tokens[1].type);
    try std.testing.expectEqualStrings("a", tokens[1].value);
    try std.testing.expectEqual(TokenType.plus, tokens[2].type);
    try std.testing.expectEqual(TokenType.name, tokens[3].type);
    try std.testing.expectEqualStrings("b", tokens[3].value);
    try std.testing.expectEqual(TokenType.minus, tokens[4].type);
    try std.testing.expectEqual(TokenType.name, tokens[5].type);
    try std.testing.expectEqualStrings("c", tokens[5].value);
    try std.testing.expectEqual(TokenType.star, tokens[6].type);
    try std.testing.expectEqual(TokenType.name, tokens[7].type);
    try std.testing.expectEqualStrings("d", tokens[7].value);
    try std.testing.expectEqual(TokenType.slash, tokens[8].type);
    try std.testing.expectEqual(TokenType.name, tokens[9].type);
    try std.testing.expectEqualStrings("e", tokens[9].value);
    try std.testing.expectEqual(TokenType.slashslash, tokens[10].type);
    try std.testing.expectEqual(TokenType.name, tokens[11].type);
    try std.testing.expectEqualStrings("f", tokens[11].value);
    try std.testing.expectEqual(TokenType.percent, tokens[12].type);
    try std.testing.expectEqual(TokenType.name, tokens[13].type);
    try std.testing.expectEqualStrings("g", tokens[13].value);
    try std.testing.expectEqual(TokenType.starstar, tokens[14].type);
    try std.testing.expectEqual(TokenType.name, tokens[15].type);
    try std.testing.expectEqualStrings("h", tokens[15].value);
    try std.testing.expectEqual(TokenType.print_close, tokens[16].type);
}

test "tokenize comparison operators" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{{ x == y != z < a > b <= c >= d }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.name, tokens[1].type);
    try std.testing.expectEqual(TokenType.eq, tokens[2].type);
    try std.testing.expectEqual(TokenType.name, tokens[3].type);
    try std.testing.expectEqual(TokenType.ne, tokens[4].type);
    try std.testing.expectEqual(TokenType.name, tokens[5].type);
    try std.testing.expectEqual(TokenType.lt, tokens[6].type);
    try std.testing.expectEqual(TokenType.name, tokens[7].type);
    try std.testing.expectEqual(TokenType.gt, tokens[8].type);
    try std.testing.expectEqual(TokenType.name, tokens[9].type);
    try std.testing.expectEqual(TokenType.le, tokens[10].type);
    try std.testing.expectEqual(TokenType.name, tokens[11].type);
    try std.testing.expectEqual(TokenType.ge, tokens[12].type);
    try std.testing.expectEqual(TokenType.name, tokens[13].type);
    try std.testing.expectEqual(TokenType.print_close, tokens[14].type);
}

test "tokenize float numbers" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{{ 1.5 + -3.14 }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.float, tokens[1].type);
    try std.testing.expectEqualStrings("1.5", tokens[1].value);
    try std.testing.expectEqual(TokenType.plus, tokens[2].type);
    try std.testing.expectEqual(TokenType.float, tokens[3].type);
    try std.testing.expectEqualStrings("-3.14", tokens[3].value);
    try std.testing.expectEqual(TokenType.print_close, tokens[4].type);
}

test "tokenize if elif else endif keywords" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{% if x %}a{% elif y %}b{% else %}c{% endif %}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.stmt_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.kw_if, tokens[1].type);
    try std.testing.expectEqual(TokenType.name, tokens[2].type);
    try std.testing.expectEqual(TokenType.stmt_close, tokens[3].type);
    try std.testing.expectEqual(TokenType.text, tokens[4].type);
    try std.testing.expectEqual(TokenType.stmt_open, tokens[5].type);
    try std.testing.expectEqual(TokenType.kw_elif, tokens[6].type);
    try std.testing.expectEqual(TokenType.name, tokens[7].type);
    try std.testing.expectEqual(TokenType.stmt_close, tokens[8].type);
    try std.testing.expectEqual(TokenType.text, tokens[9].type);
    try std.testing.expectEqual(TokenType.stmt_open, tokens[10].type);
    try std.testing.expectEqual(TokenType.kw_else, tokens[11].type);
    try std.testing.expectEqual(TokenType.stmt_close, tokens[12].type);
    try std.testing.expectEqual(TokenType.text, tokens[13].type);
    try std.testing.expectEqual(TokenType.stmt_open, tokens[14].type);
    try std.testing.expectEqual(TokenType.kw_endif, tokens[15].type);
    try std.testing.expectEqual(TokenType.stmt_close, tokens[16].type);
}

test "tokenize for endfor in keywords" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{% for item in items %}{{ item }}{% endfor %}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.stmt_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.kw_for, tokens[1].type);
    try std.testing.expectEqual(TokenType.name, tokens[2].type);
    try std.testing.expectEqualStrings("item", tokens[2].value);
    try std.testing.expectEqual(TokenType.kw_in, tokens[3].type);
    try std.testing.expectEqual(TokenType.name, tokens[4].type);
    try std.testing.expectEqualStrings("items", tokens[4].value);
    try std.testing.expectEqual(TokenType.stmt_close, tokens[5].type);
}

test "tokenize boolean and logical keywords" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{% if not x and y or z %}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.stmt_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.kw_if, tokens[1].type);
    try std.testing.expectEqual(TokenType.kw_not, tokens[2].type);
    try std.testing.expectEqual(TokenType.name, tokens[3].type);
    try std.testing.expectEqual(TokenType.kw_and, tokens[4].type);
    try std.testing.expectEqual(TokenType.name, tokens[5].type);
    try std.testing.expectEqual(TokenType.kw_or, tokens[6].type);
    try std.testing.expectEqual(TokenType.name, tokens[7].type);
}

test "tokenize set keyword" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{% set x = 42 %}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.stmt_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.kw_set, tokens[1].type);
    try std.testing.expectEqual(TokenType.name, tokens[2].type);
    try std.testing.expectEqualStrings("x", tokens[2].value);
    try std.testing.expectEqual(TokenType.assign, tokens[3].type);
    try std.testing.expectEqual(TokenType.integer, tokens[4].type);
    try std.testing.expectEqualStrings("42", tokens[4].value);
    try std.testing.expectEqual(TokenType.stmt_close, tokens[5].type);
}

test "tokenize true false none keywords" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{{ true, false, none, True, False, None }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.kw_true, tokens[1].type);
    try std.testing.expectEqual(TokenType.comma, tokens[2].type);
    try std.testing.expectEqual(TokenType.kw_false, tokens[3].type);
    try std.testing.expectEqual(TokenType.comma, tokens[4].type);
    try std.testing.expectEqual(TokenType.kw_none, tokens[5].type);
    try std.testing.expectEqual(TokenType.comma, tokens[6].type);
    try std.testing.expectEqual(TokenType.kw_true, tokens[7].type); // True
    try std.testing.expectEqual(TokenType.comma, tokens[8].type);
    try std.testing.expectEqual(TokenType.kw_false, tokens[9].type); // False
    try std.testing.expectEqual(TokenType.comma, tokens[10].type);
    try std.testing.expectEqual(TokenType.kw_none, tokens[11].type); // None
    try std.testing.expectEqual(TokenType.print_close, tokens[12].type);
}

test "tokenize is defined namespace keywords" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{% if x is defined %}{% set ns = namespace() %}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.kw_if, tokens[1].type);
    try std.testing.expectEqual(TokenType.name, tokens[2].type);
    try std.testing.expectEqual(TokenType.kw_is, tokens[3].type);
    try std.testing.expectEqual(TokenType.kw_defined, tokens[4].type);
    try std.testing.expectEqual(TokenType.kw_set, tokens[7].type);
    try std.testing.expectEqual(TokenType.kw_namespace, tokens[10].type);
}

test "tokenize macro endmacro keywords" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{% macro foo() %}content{% endmacro %}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.stmt_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.kw_macro, tokens[1].type);
    try std.testing.expectEqual(TokenType.name, tokens[2].type);
    try std.testing.expectEqual(TokenType.lparen, tokens[3].type);
    try std.testing.expectEqual(TokenType.rparen, tokens[4].type);
    try std.testing.expectEqual(TokenType.stmt_close, tokens[5].type);
    try std.testing.expectEqual(TokenType.stmt_open, tokens[7].type);
    try std.testing.expectEqual(TokenType.kw_endmacro, tokens[8].type);
}

test "tokenize nested tags" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{% if x %}{{ y }}{% endif %}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.stmt_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.kw_if, tokens[1].type);
    try std.testing.expectEqual(TokenType.name, tokens[2].type);
    try std.testing.expectEqual(TokenType.stmt_close, tokens[3].type);
    try std.testing.expectEqual(TokenType.print_open, tokens[4].type);
    try std.testing.expectEqual(TokenType.name, tokens[5].type);
    try std.testing.expectEqual(TokenType.print_close, tokens[6].type);
    try std.testing.expectEqual(TokenType.stmt_open, tokens[7].type);
    try std.testing.expectEqual(TokenType.kw_endif, tokens[8].type);
    try std.testing.expectEqual(TokenType.stmt_close, tokens[9].type);
}

test "tokenize multiple statements" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{% set a = 1 %}{% set b = 2 %}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.stmt_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.kw_set, tokens[1].type);
    try std.testing.expectEqual(TokenType.name, tokens[2].type);
    try std.testing.expectEqualStrings("a", tokens[2].value);
    try std.testing.expectEqual(TokenType.assign, tokens[3].type);
    try std.testing.expectEqual(TokenType.integer, tokens[4].type);
    try std.testing.expectEqualStrings("1", tokens[4].value);
    try std.testing.expectEqual(TokenType.stmt_close, tokens[5].type);
    try std.testing.expectEqual(TokenType.stmt_open, tokens[6].type);
    try std.testing.expectEqual(TokenType.kw_set, tokens[7].type);
    try std.testing.expectEqual(TokenType.name, tokens[8].type);
    try std.testing.expectEqualStrings("b", tokens[8].value);
    try std.testing.expectEqual(TokenType.assign, tokens[9].type);
    try std.testing.expectEqual(TokenType.integer, tokens[10].type);
    try std.testing.expectEqualStrings("2", tokens[10].value);
    try std.testing.expectEqual(TokenType.stmt_close, tokens[11].type);
}

test "tokenize list syntax" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{{ [1, 2, 3] }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.lbracket, tokens[1].type);
    try std.testing.expectEqual(TokenType.integer, tokens[2].type);
    try std.testing.expectEqualStrings("1", tokens[2].value);
    try std.testing.expectEqual(TokenType.comma, tokens[3].type);
    try std.testing.expectEqual(TokenType.integer, tokens[4].type);
    try std.testing.expectEqualStrings("2", tokens[4].value);
    try std.testing.expectEqual(TokenType.comma, tokens[5].type);
    try std.testing.expectEqual(TokenType.integer, tokens[6].type);
    try std.testing.expectEqualStrings("3", tokens[6].value);
    try std.testing.expectEqual(TokenType.rbracket, tokens[7].type);
    try std.testing.expectEqual(TokenType.print_close, tokens[8].type);
}

test "tokenize dict syntax" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{{ {a: 1, b: 2} }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.lbrace, tokens[1].type);
    try std.testing.expectEqual(TokenType.name, tokens[2].type);
    try std.testing.expectEqualStrings("a", tokens[2].value);
    try std.testing.expectEqual(TokenType.colon, tokens[3].type);
    try std.testing.expectEqual(TokenType.integer, tokens[4].type);
    try std.testing.expectEqualStrings("1", tokens[4].value);
    try std.testing.expectEqual(TokenType.comma, tokens[5].type);
    try std.testing.expectEqual(TokenType.name, tokens[6].type);
    try std.testing.expectEqualStrings("b", tokens[6].value);
    try std.testing.expectEqual(TokenType.colon, tokens[7].type);
    try std.testing.expectEqual(TokenType.integer, tokens[8].type);
    try std.testing.expectEqualStrings("2", tokens[8].value);
    try std.testing.expectEqual(TokenType.rbrace, tokens[9].type);
    try std.testing.expectEqual(TokenType.print_close, tokens[10].type);
}

test "tokenize error unterminated string" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{{ 'hello }}");
    defer lexer.deinit();

    try std.testing.expectError(LexerError.UnterminatedString, lexer.tokenize());
}

test "tokenize error unterminated tag" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{{ name ");
    defer lexer.deinit();

    try std.testing.expectError(LexerError.UnterminatedTag, lexer.tokenize());
}

test "tokenize error unterminated comment" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{# this is a comment");
    defer lexer.deinit();

    try std.testing.expectError(LexerError.UnterminatedComment, lexer.tokenize());
}

test "tokenize negative integers" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{{ -42 }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.integer, tokens[1].type);
    try std.testing.expectEqualStrings("-42", tokens[1].value);
    try std.testing.expectEqual(TokenType.print_close, tokens[2].type);
}

test "tokenize pipe and tilde operators" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{{ x | filtername ~ y }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.name, tokens[1].type);
    try std.testing.expectEqual(TokenType.pipe, tokens[2].type);
    try std.testing.expectEqual(TokenType.name, tokens[3].type);
    try std.testing.expectEqual(TokenType.tilde, tokens[4].type);
    try std.testing.expectEqual(TokenType.name, tokens[5].type);
    try std.testing.expectEqual(TokenType.print_close, tokens[6].type);
}

test "tokenize parentheses and brackets" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{{ (a)[b] }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.lparen, tokens[1].type);
    try std.testing.expectEqual(TokenType.name, tokens[2].type);
    try std.testing.expectEqual(TokenType.rparen, tokens[3].type);
    try std.testing.expectEqual(TokenType.lbracket, tokens[4].type);
    try std.testing.expectEqual(TokenType.name, tokens[5].type);
    try std.testing.expectEqual(TokenType.rbracket, tokens[6].type);
    try std.testing.expectEqual(TokenType.print_close, tokens[7].type);
}

test "tokenize braces and dot" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{{ {}.attr }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.lbrace, tokens[1].type);
    try std.testing.expectEqual(TokenType.rbrace, tokens[2].type);
    try std.testing.expectEqual(TokenType.dot, tokens[3].type);
    try std.testing.expectEqual(TokenType.name, tokens[4].type);
    try std.testing.expectEqual(TokenType.print_close, tokens[5].type);
}

test "tokenize break and continue keywords" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{% break %}{% continue %}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.stmt_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.kw_break, tokens[1].type);
    try std.testing.expectEqual(TokenType.stmt_close, tokens[2].type);
    try std.testing.expectEqual(TokenType.stmt_open, tokens[3].type);
    try std.testing.expectEqual(TokenType.kw_continue, tokens[4].type);
    try std.testing.expectEqual(TokenType.stmt_close, tokens[5].type);
}

test "tokenize filter and endfilter keywords" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{% filter upper %}text{% endfilter %}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.stmt_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.kw_filter, tokens[1].type);
    try std.testing.expectEqual(TokenType.name, tokens[2].type);
    try std.testing.expectEqual(TokenType.stmt_close, tokens[3].type);
    try std.testing.expectEqual(TokenType.text, tokens[4].type);
    try std.testing.expectEqual(TokenType.stmt_open, tokens[5].type);
    try std.testing.expectEqual(TokenType.kw_endfilter, tokens[6].type);
}

test "tokenize call and endcall keywords" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{% call foo() %}{% endcall %}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.stmt_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.kw_call, tokens[1].type);
    try std.testing.expectEqual(TokenType.name, tokens[2].type);
    try std.testing.expectEqual(TokenType.lparen, tokens[3].type);
    try std.testing.expectEqual(TokenType.rparen, tokens[4].type);
    try std.testing.expectEqual(TokenType.stmt_close, tokens[5].type);
    try std.testing.expectEqual(TokenType.stmt_open, tokens[6].type);
    try std.testing.expectEqual(TokenType.kw_endcall, tokens[7].type);
}

test "tokenize recursive keyword" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{% for x in items recursive %}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.stmt_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.kw_for, tokens[1].type);
    try std.testing.expectEqual(TokenType.name, tokens[2].type);
    try std.testing.expectEqual(TokenType.kw_in, tokens[3].type);
    try std.testing.expectEqual(TokenType.name, tokens[4].type);
    try std.testing.expectEqual(TokenType.kw_recursive, tokens[5].type);
}

test "tokenize single quote string" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{{ 'single quotes' }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.string, tokens[1].type);
    try std.testing.expectEqualStrings("single quotes", tokens[1].value);
    try std.testing.expectEqual(TokenType.print_close, tokens[2].type);
}

test "tokenize double quote string" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{{ \"double quotes\" }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.string, tokens[1].type);
    try std.testing.expectEqualStrings("double quotes", tokens[1].value);
    try std.testing.expectEqual(TokenType.print_close, tokens[2].type);
}

test "tokenize raw block with trim" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{% raw %}{{ var }}{% endraw -%}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.text, tokens[0].type);
    try std.testing.expectEqualStrings("{{ var }}", tokens[0].value);
    try std.testing.expectEqual(TokenType.eof, tokens[1].type);
}

test "tokenize raw block with left trim" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{%- raw %}content{%- endraw %}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.text, tokens[0].type);
    try std.testing.expectEqualStrings("content", tokens[0].value);
    try std.testing.expectEqual(TokenType.eof, tokens[1].type);
}

test "tokenize raw block with both trim" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "  {%- raw -%}content{%- endraw -%}  ");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.text, tokens[0].type);
    try std.testing.expectEqualStrings("", tokens[0].value);
    try std.testing.expectEqual(TokenType.text, tokens[1].type);
    try std.testing.expectEqualStrings("content", tokens[1].value);
    try std.testing.expectEqual(TokenType.eof, tokens[2].type);
}

test "tokenize statement with left trim" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "  {%- if x %}yes{% endif %}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.text, tokens[0].type);
    try std.testing.expectEqualStrings("", tokens[0].value);
    try std.testing.expectEqual(TokenType.stmt_open, tokens[1].type);
    try std.testing.expect(tokens[1].trim_left);
}

test "tokenize statement with right trim" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{% if x -%}  yes{% endif %}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.stmt_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.kw_if, tokens[1].type);
    try std.testing.expectEqual(TokenType.name, tokens[2].type);
    try std.testing.expectEqual(TokenType.stmt_close, tokens[3].type);
    try std.testing.expect(tokens[3].trim_right);
    try std.testing.expectEqual(TokenType.text, tokens[4].type);
    try std.testing.expectEqualStrings("yes", tokens[4].value);
}

test "tokenize print with both trim" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "  {{- x -}}  ");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.text, tokens[0].type);
    try std.testing.expectEqualStrings("", tokens[0].value);
    try std.testing.expectEqual(TokenType.print_open, tokens[1].type);
    try std.testing.expect(tokens[1].trim_left);
    try std.testing.expectEqual(TokenType.name, tokens[2].type);
    try std.testing.expectEqual(TokenType.print_close, tokens[3].type);
    try std.testing.expect(tokens[3].trim_right);
    try std.testing.expectEqual(TokenType.eof, tokens[4].type);
}

test "tokenize empty template" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqual(TokenType.eof, tokens[0].type);
}

test "tokenize text only" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "Just plain text");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(@as(usize, 2), tokens.len);
    try std.testing.expectEqual(TokenType.text, tokens[0].type);
    try std.testing.expectEqualStrings("Just plain text", tokens[0].value);
    try std.testing.expectEqual(TokenType.eof, tokens[1].type);
}

test "tokenize multiple text segments" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "Hello {{ name }}, how are {{ you }}?");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.text, tokens[0].type);
    try std.testing.expectEqualStrings("Hello ", tokens[0].value);
    try std.testing.expectEqual(TokenType.print_open, tokens[1].type);
    try std.testing.expectEqual(TokenType.name, tokens[2].type);
    try std.testing.expectEqual(TokenType.print_close, tokens[3].type);
    try std.testing.expectEqual(TokenType.text, tokens[4].type);
    try std.testing.expectEqualStrings(", how are ", tokens[4].value);
    try std.testing.expectEqual(TokenType.print_open, tokens[5].type);
    try std.testing.expectEqual(TokenType.name, tokens[6].type);
    try std.testing.expectEqual(TokenType.print_close, tokens[7].type);
    try std.testing.expectEqual(TokenType.text, tokens[8].type);
    try std.testing.expectEqualStrings("?", tokens[8].value);
}

test "tokenize comment in middle of text" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "Start{# comment #}Middle{# another #}End");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.text, tokens[0].type);
    try std.testing.expectEqualStrings("Start", tokens[0].value);
    try std.testing.expectEqual(TokenType.text, tokens[1].type);
    try std.testing.expectEqualStrings("Middle", tokens[1].value);
    try std.testing.expectEqual(TokenType.text, tokens[2].type);
    try std.testing.expectEqualStrings("End", tokens[2].value);
}

test "tokenize nested braces in dict" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{{ {'nested': {'key': 'value'}} }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.lbrace, tokens[1].type);
    try std.testing.expectEqual(TokenType.string, tokens[2].type);
    try std.testing.expectEqual(TokenType.colon, tokens[3].type);
    try std.testing.expectEqual(TokenType.lbrace, tokens[4].type);
}

test "tokenize assignment operator" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{% set x = 5 %}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.stmt_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.kw_set, tokens[1].type);
    try std.testing.expectEqual(TokenType.name, tokens[2].type);
    try std.testing.expectEqual(TokenType.assign, tokens[3].type);
    try std.testing.expectEqual(TokenType.integer, tokens[4].type);
}

test "tokenize zero integer" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{{ 0 }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.integer, tokens[1].type);
    try std.testing.expectEqualStrings("0", tokens[1].value);
}

test "tokenize float with leading zero" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{{ 0.5 }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.float, tokens[1].type);
    try std.testing.expectEqualStrings("0.5", tokens[1].value);
}

test "tokenize identifiers with underscores" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{{ my_var_name }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.name, tokens[1].type);
    try std.testing.expectEqualStrings("my_var_name", tokens[1].value);
}

test "tokenize identifiers with numbers" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{{ var123name }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.name, tokens[1].type);
    try std.testing.expectEqualStrings("var123name", tokens[1].value);
}

test "tokenize comment with newline" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "Line1\n{# comment #}\nLine2");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.text, tokens[0].type);
    try std.testing.expectEqualStrings("Line1\n", tokens[0].value);
    try std.testing.expectEqual(TokenType.text, tokens[1].type);
}

test "tokenize single newline after statement" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{% if x %}\nContent");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.stmt_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.kw_if, tokens[1].type);
    try std.testing.expectEqual(TokenType.name, tokens[2].type);
    try std.testing.expectEqual(TokenType.stmt_close, tokens[3].type);
    try std.testing.expectEqual(TokenType.text, tokens[4].type);
    try std.testing.expectEqualStrings("Content", tokens[4].value);
}

test "tokenize error invalid character" {
    const allocator = std.testing.allocator;

    var lexer = Lexer.init(allocator, "{{ @ }}");
    defer lexer.deinit();

    try std.testing.expectError(LexerError.InvalidCharacter, lexer.tokenize());
}

test "tokenize closing braces in string" {
    const allocator = std.testing.allocator;

    // This is a critical test: {{ "}}" }} should output the string "}}"
    // The lexer must not treat the }} inside the string as the tag close
    var lexer = Lexer.init(allocator, "{{ \"}}\" }}"); // lint:ignore thread-global - local variable in test, not global
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.string, tokens[1].type);
    try std.testing.expectEqualStrings("}}", tokens[1].value);
    try std.testing.expectEqual(TokenType.print_close, tokens[2].type);
    try std.testing.expectEqual(TokenType.eof, tokens[3].type);
}

test "init creates lexer with correct initial state" {
    const allocator = std.testing.allocator;
    const source = "test template";
    var lexer = Lexer.init(allocator, source);
    defer lexer.deinit();

    try std.testing.expectEqual(@as(usize, 0), lexer.pos);
    try std.testing.expectEqualStrings(source, lexer.source);
    try std.testing.expect(!lexer.in_tag);
}

test "deinit cleans up tokens array" {
    const allocator = std.testing.allocator;
    var lexer = Lexer.init(allocator, "{{ x }}");
    _ = try lexer.tokenize();
    // deinit frees the tokens array
    lexer.deinit();
}
