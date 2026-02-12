//! Grammar AST types for JSON Schema representation.

const std = @import("std");

pub const RuleId = u32;
pub const INVALID_RULE: RuleId = std.math.maxInt(RuleId);

pub fn dupeRuleIds(allocator: std.mem.Allocator, ids: []const RuleId) ![]RuleId {
    return allocator.dupe(RuleId, ids);
}

pub const Rule = union(enum) {
    char: u8,
    char_range: struct {
        start: u8,
        end: u8,
    },
    literal: []const u8,
    sequence: []const RuleId,
    alternatives: []const RuleId,
    optional: RuleId,
    star: RuleId,
    plus: RuleId,
    reference: RuleId,
    end: void,
};

pub const Grammar = struct {
    /// Internal arena for all allocations - ensures clean deallocation.
    arena: std.heap.ArenaAllocator,
    rules: std.ArrayList(Rule),
    root_rule: RuleId,
    rule_names: std.StringHashMap(RuleId),
    ws_rule: RuleId,
    is_json: bool,

    /// Get allocator for this grammar's allocations.
    pub fn allocator(self: *Grammar) std.mem.Allocator {
        return self.arena.allocator();
    }

    pub fn init(backing_allocator: std.mem.Allocator) Grammar {
        var arena = std.heap.ArenaAllocator.init(backing_allocator);
        return .{
            .arena = arena,
            .rules = .empty,
            .root_rule = INVALID_RULE,
            .rule_names = std.StringHashMap(RuleId).init(arena.allocator()),
            .ws_rule = INVALID_RULE,
            .is_json = true,
        };
    }

    pub fn deinit(self: *Grammar) void {
        // Arena handles all internal allocations - just destroy it
        self.arena.deinit();
    }

    pub fn addRule(self: *Grammar, rule: Rule) !RuleId {
        const id: RuleId = @intCast(self.rules.items.len);
        try self.rules.append(self.allocator(), rule);
        return id;
    }

    pub fn getRule(self: *const Grammar, id: RuleId) ?Rule {
        if (id >= self.rules.items.len) return null;
        return self.rules.items[id];
    }

    pub fn isValidRule(self: *const Grammar, id: RuleId) bool {
        return id < self.rules.items.len;
    }
};

pub const JsonRules = struct {
    pub fn whitespace(grammar: *Grammar) !RuleId {
        const ws_ids = [_]RuleId{
            try grammar.addRule(.{ .char = ' ' }),
            try grammar.addRule(.{ .char = '\t' }),
            try grammar.addRule(.{ .char = '\n' }),
            try grammar.addRule(.{ .char = '\r' }),
        };
        const ws_char = try grammar.addRule(.{
            .alternatives = try dupeRuleIds(grammar.allocator(), &ws_ids),
        });
        return grammar.addRule(.{ .star = ws_char });
    }

    pub fn string(grammar: *Grammar) !RuleId {
        const quote = try grammar.addRule(.{ .char = '"' });

        const unescaped_1 = try grammar.addRule(.{
            .char_range = .{ .start = 0x20, .end = 0x21 },
        });
        const unescaped_2 = try grammar.addRule(.{
            .char_range = .{ .start = 0x23, .end = 0x5B },
        });
        const unescaped_3 = try grammar.addRule(.{
            .char_range = .{ .start = 0x5D, .end = 0x7E },
        });
        const utf8_cont = try grammar.addRule(.{
            .char_range = .{ .start = 0x80, .end = 0xFF },
        });
        const unescaped_ids = [_]RuleId{ unescaped_1, unescaped_2, unescaped_3, utf8_cont };
        const unescaped = try grammar.addRule(.{
            .alternatives = try dupeRuleIds(grammar.allocator(), &unescaped_ids),
        });

        const backslash = try grammar.addRule(.{ .char = '\\' });
        const simple_escape_ids = [_]RuleId{
            try grammar.addRule(.{ .char = '"' }),
            try grammar.addRule(.{ .char = '\\' }),
            try grammar.addRule(.{ .char = '/' }),
            try grammar.addRule(.{ .char = 'b' }),
            try grammar.addRule(.{ .char = 'f' }),
            try grammar.addRule(.{ .char = 'n' }),
            try grammar.addRule(.{ .char = 'r' }),
            try grammar.addRule(.{ .char = 't' }),
        };
        const simple_escape = try grammar.addRule(.{
            .alternatives = try dupeRuleIds(grammar.allocator(), &simple_escape_ids),
        });

        const hex_digit_ids = [_]RuleId{
            try grammar.addRule(.{ .char_range = .{ .start = '0', .end = '9' } }),
            try grammar.addRule(.{ .char_range = .{ .start = 'a', .end = 'f' } }),
            try grammar.addRule(.{ .char_range = .{ .start = 'A', .end = 'F' } }),
        };
        const hex_digit = try grammar.addRule(.{
            .alternatives = try dupeRuleIds(grammar.allocator(), &hex_digit_ids),
        });
        const u_char = try grammar.addRule(.{ .char = 'u' });
        const unicode_ids = [_]RuleId{ u_char, hex_digit, hex_digit, hex_digit, hex_digit };
        const unicode_escape = try grammar.addRule(.{
            .sequence = try dupeRuleIds(grammar.allocator(), &unicode_ids),
        });

        const escape_ids = [_]RuleId{ simple_escape, unicode_escape };
        const escape_content = try grammar.addRule(.{
            .alternatives = try dupeRuleIds(grammar.allocator(), &escape_ids),
        });
        const escaped_ids = [_]RuleId{ backslash, escape_content };
        const escaped = try grammar.addRule(.{
            .sequence = try dupeRuleIds(grammar.allocator(), &escaped_ids),
        });

        const string_char_ids = [_]RuleId{ unescaped, escaped };
        const string_char = try grammar.addRule(.{
            .alternatives = try dupeRuleIds(grammar.allocator(), &string_char_ids),
        });
        const chars = try grammar.addRule(.{ .star = string_char });
        const string_ids = [_]RuleId{ quote, chars, quote };
        return grammar.addRule(.{
            .sequence = try dupeRuleIds(grammar.allocator(), &string_ids),
        });
    }

    /// JSON integer: -?(0|[1-9][0-9]*)
    /// No decimal point or exponent allowed.
    pub fn integer(grammar: *Grammar) !RuleId {
        const digit = try grammar.addRule(.{
            .char_range = .{ .start = '0', .end = '9' },
        });

        const minus = try grammar.addRule(.{ .char = '-' });
        const opt_minus = try grammar.addRule(.{ .optional = minus });

        const zero = try grammar.addRule(.{ .char = '0' });
        const nonzero = try grammar.addRule(.{
            .char_range = .{ .start = '1', .end = '9' },
        });
        const nonzero_star = try grammar.addRule(.{ .star = digit });
        const nonzero_int_ids = [_]RuleId{ nonzero, nonzero_star };
        const nonzero_int = try grammar.addRule(.{
            .sequence = try dupeRuleIds(grammar.allocator(), &nonzero_int_ids),
        });
        const int_part_ids = [_]RuleId{ zero, nonzero_int };
        const int_part = try grammar.addRule(.{
            .alternatives = try dupeRuleIds(grammar.allocator(), &int_part_ids),
        });

        const integer_ids = [_]RuleId{ opt_minus, int_part };
        return grammar.addRule(.{
            .sequence = try dupeRuleIds(grammar.allocator(), &integer_ids),
        });
    }

    /// JSON number: -?(0|[1-9][0-9]*)(.[0-9]+)?([eE][+-]?[0-9]+)?
    pub fn number(grammar: *Grammar) !RuleId {
        const digit = try grammar.addRule(.{
            .char_range = .{ .start = '0', .end = '9' },
        });
        const digits = try grammar.addRule(.{ .plus = digit });

        const minus = try grammar.addRule(.{ .char = '-' });
        const opt_minus = try grammar.addRule(.{ .optional = minus });

        const zero = try grammar.addRule(.{ .char = '0' });
        const nonzero = try grammar.addRule(.{
            .char_range = .{ .start = '1', .end = '9' },
        });
        const nonzero_star = try grammar.addRule(.{ .star = digit });
        const nonzero_int_ids = [_]RuleId{ nonzero, nonzero_star };
        const nonzero_int = try grammar.addRule(.{
            .sequence = try dupeRuleIds(grammar.allocator(), &nonzero_int_ids),
        });
        const int_part_ids = [_]RuleId{ zero, nonzero_int };
        const int_part = try grammar.addRule(.{
            .alternatives = try dupeRuleIds(grammar.allocator(), &int_part_ids),
        });

        const dot = try grammar.addRule(.{ .char = '.' });
        const frac_ids = [_]RuleId{ dot, digits };
        const frac = try grammar.addRule(.{
            .sequence = try dupeRuleIds(grammar.allocator(), &frac_ids),
        });
        const opt_frac = try grammar.addRule(.{ .optional = frac });

        const e_lower = try grammar.addRule(.{ .char = 'e' });
        const e_upper = try grammar.addRule(.{ .char = 'E' });
        const e_char_ids = [_]RuleId{ e_lower, e_upper };
        const e_char = try grammar.addRule(.{
            .alternatives = try dupeRuleIds(grammar.allocator(), &e_char_ids),
        });
        const plus = try grammar.addRule(.{ .char = '+' });
        const sign_ids = [_]RuleId{ plus, minus };
        const sign = try grammar.addRule(.{
            .alternatives = try dupeRuleIds(grammar.allocator(), &sign_ids),
        });
        const opt_sign = try grammar.addRule(.{ .optional = sign });
        const exp_ids = [_]RuleId{ e_char, opt_sign, digits };
        const exp = try grammar.addRule(.{
            .sequence = try dupeRuleIds(grammar.allocator(), &exp_ids),
        });
        const opt_exp = try grammar.addRule(.{ .optional = exp });

        const number_ids = [_]RuleId{ opt_minus, int_part, opt_frac, opt_exp };
        return grammar.addRule(.{
            .sequence = try dupeRuleIds(grammar.allocator(), &number_ids),
        });
    }

    pub fn boolean(grammar: *Grammar) !RuleId {
        const bool_ids = [_]RuleId{
            try grammar.addRule(.{ .literal = "true" }),
            try grammar.addRule(.{ .literal = "false" }),
        };
        return grammar.addRule(.{
            .alternatives = try dupeRuleIds(grammar.allocator(), &bool_ids),
        });
    }

    pub fn null_(grammar: *Grammar) !RuleId {
        return grammar.addRule(.{ .literal = "null" });
    }

    /// Build a string_char rule (unescaped | escaped) for use in bounded strings.
    /// This is the same as the internal char rule in string(), extracted for reuse.
    pub fn stringChar(grammar: *Grammar) !RuleId {
        // Unescaped: printable ASCII except " and \ plus UTF-8 continuation bytes
        const unescaped_1 = try grammar.addRule(.{
            .char_range = .{ .start = 0x20, .end = 0x21 },
        });
        const unescaped_2 = try grammar.addRule(.{
            .char_range = .{ .start = 0x23, .end = 0x5B },
        });
        const unescaped_3 = try grammar.addRule(.{
            .char_range = .{ .start = 0x5D, .end = 0x7E },
        });
        const utf8_cont = try grammar.addRule(.{
            .char_range = .{ .start = 0x80, .end = 0xFF },
        });
        const unescaped_ids = [_]RuleId{ unescaped_1, unescaped_2, unescaped_3, utf8_cont };
        const unescaped = try grammar.addRule(.{
            .alternatives = try dupeRuleIds(grammar.allocator(), &unescaped_ids),
        });

        // Escaped: \X where X is special char or \uXXXX lint:ignore todo-issue
        const backslash = try grammar.addRule(.{ .char = '\\' });
        const simple_escape_ids = [_]RuleId{
            try grammar.addRule(.{ .char = '"' }),
            try grammar.addRule(.{ .char = '\\' }),
            try grammar.addRule(.{ .char = '/' }),
            try grammar.addRule(.{ .char = 'b' }),
            try grammar.addRule(.{ .char = 'f' }),
            try grammar.addRule(.{ .char = 'n' }),
            try grammar.addRule(.{ .char = 'r' }),
            try grammar.addRule(.{ .char = 't' }),
        };
        const simple_escape = try grammar.addRule(.{
            .alternatives = try dupeRuleIds(grammar.allocator(), &simple_escape_ids),
        });

        const hex_digit_ids = [_]RuleId{
            try grammar.addRule(.{ .char_range = .{ .start = '0', .end = '9' } }),
            try grammar.addRule(.{ .char_range = .{ .start = 'a', .end = 'f' } }),
            try grammar.addRule(.{ .char_range = .{ .start = 'A', .end = 'F' } }),
        };
        const hex_digit = try grammar.addRule(.{
            .alternatives = try dupeRuleIds(grammar.allocator(), &hex_digit_ids),
        });
        const u_char = try grammar.addRule(.{ .char = 'u' });
        const unicode_ids = [_]RuleId{ u_char, hex_digit, hex_digit, hex_digit, hex_digit };
        const unicode_escape = try grammar.addRule(.{
            .sequence = try dupeRuleIds(grammar.allocator(), &unicode_ids),
        });

        const escape_ids = [_]RuleId{ simple_escape, unicode_escape };
        const escape_content = try grammar.addRule(.{
            .alternatives = try dupeRuleIds(grammar.allocator(), &escape_ids),
        });
        const escaped_ids = [_]RuleId{ backslash, escape_content };
        const escaped = try grammar.addRule(.{
            .sequence = try dupeRuleIds(grammar.allocator(), &escaped_ids),
        });

        const string_char_ids = [_]RuleId{ unescaped, escaped };
        return grammar.addRule(.{
            .alternatives = try dupeRuleIds(grammar.allocator(), &string_char_ids),
        });
    }

    /// JSON string with bounded character count: " char{min,max} "
    /// min_len: minimum number of characters (0 means empty string allowed)
    /// max_len: maximum number of characters (null means unbounded)
    pub fn stringBounded(grammar: *Grammar, min_len: usize, max_len: ?usize) !RuleId {
        const quote = try grammar.addRule(.{ .char = '"' });
        const string_char = try stringChar(grammar);

        // Build: quote + min required chars + (max-min) optional chars + quote
        // Note: No defer deinit needed - arena allocator handles cleanup via toOwnedSlice
        var seq = std.ArrayList(RuleId).empty;

        try seq.append(grammar.allocator(), quote);

        // Required chars (min_len)
        for (0..min_len) |_| {
            try seq.append(grammar.allocator(), string_char);
        }

        // Optional chars (max_len - min_len), or unbounded if max_len is null
        if (max_len) |max| {
            for (0..(max - min_len)) |_| {
                const opt_char = try grammar.addRule(.{ .optional = string_char });
                try seq.append(grammar.allocator(), opt_char);
            }
        } else {
            // Unbounded: use star for remaining chars
            const more_chars = try grammar.addRule(.{ .star = string_char });
            try seq.append(grammar.allocator(), more_chars);
        }

        try seq.append(grammar.allocator(), quote);

        return grammar.addRule(.{
            .sequence = try seq.toOwnedSlice(grammar.allocator()),
        });
    }
};

test "json rules build basic primitives" {
    const allocator = std.testing.allocator;
    var grammar = Grammar.init(allocator);
    defer grammar.deinit();

    const ws = try JsonRules.whitespace(&grammar);
    const str = try JsonRules.string(&grammar);
    const num = try JsonRules.number(&grammar);
    const boolean = try JsonRules.boolean(&grammar);
    const null_rule = try JsonRules.null_(&grammar);

    try std.testing.expect(grammar.isValidRule(ws));
    try std.testing.expect(grammar.isValidRule(str));
    try std.testing.expect(grammar.isValidRule(num));
    try std.testing.expect(grammar.isValidRule(boolean));
    try std.testing.expect(grammar.isValidRule(null_rule));
}
