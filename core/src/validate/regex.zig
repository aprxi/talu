//! Pre-compiled regex patterns for JSON Schema format support.
//!
//! Maps format strings to grammar rules that enforce the pattern.
//! Supported formats: date-time, date, time, uuid, email, uri, ipv4, ipv6.

const std = @import("std");
const ast = @import("ast.zig");

const Grammar = ast.Grammar;
const RuleId = ast.RuleId;

pub const Format = enum {
    date_time,
    date,
    time,
    uuid,
    email,
    uri,
    ipv4,
    ipv6,

    pub fn fromString(s: []const u8) ?Format {
        const map = std.StaticStringMap(Format).initComptime(.{
            .{ "date-time", .date_time },
            .{ "date", .date },
            .{ "time", .time },
            .{ "uuid", .uuid },
            .{ "email", .email },
            .{ "uri", .uri },
            .{ "ipv4", .ipv4 },
            .{ "ipv6", .ipv6 },
        });
        return map.get(s);
    }
};

pub fn compileFormat(grammar: *Grammar, format: Format) !RuleId {
    return switch (format) {
        .date_time => compileDateTime(grammar),
        .date => compileDate(grammar),
        .time => compileTime(grammar),
        .uuid => compileUuid(grammar),
        .email => compileEmail(grammar),
        .uri => compileGenericString(grammar),
        .ipv4 => compileIpv4(grammar),
        .ipv6 => compileGenericString(grammar),
    };
}

fn compileDateTime(grammar: *Grammar) !RuleId {
    const digit = try grammar.addRule(.{ .char_range = .{ .start = '0', .end = '9' } });

    const date = try compileDate(grammar);
    const t = try grammar.addRule(.{ .char = 'T' });
    const time = try compileTimeNoTz(grammar);

    const dot = try grammar.addRule(.{ .char = '.' });
    const frac_digits = try grammar.addRule(.{ .plus = digit });
    const frac_ids = [_]RuleId{ dot, frac_digits };
    const frac = try grammar.addRule(.{ .sequence = try ast.dupeRuleIds(grammar.allocator(), &frac_ids) });
    const opt_frac = try grammar.addRule(.{ .optional = frac });

    const z = try grammar.addRule(.{ .char = 'Z' });
    const plus = try grammar.addRule(.{ .char = '+' });
    const minus = try grammar.addRule(.{ .char = '-' });
    const sign_ids = [_]RuleId{ plus, minus };
    const sign = try grammar.addRule(.{ .alternatives = try ast.dupeRuleIds(grammar.allocator(), &sign_ids) });
    const colon = try grammar.addRule(.{ .char = ':' });
    const two_digits_ids = [_]RuleId{ digit, digit };
    const two_digits = try grammar.addRule(.{ .sequence = try ast.dupeRuleIds(grammar.allocator(), &two_digits_ids) });
    const offset_ids = [_]RuleId{ sign, two_digits, colon, two_digits };
    const offset = try grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &offset_ids),
    });
    const tz_ids = [_]RuleId{ z, offset };
    const tz = try grammar.addRule(.{ .alternatives = try ast.dupeRuleIds(grammar.allocator(), &tz_ids) });
    const opt_tz = try grammar.addRule(.{ .optional = tz });

    const quote = try grammar.addRule(.{ .char = '"' });
    const inner_ids = [_]RuleId{ date, t, time, opt_frac, opt_tz };
    const inner = try grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &inner_ids),
    });

    return grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &[_]RuleId{ quote, inner, quote }),
    });
}

fn compileDate(grammar: *Grammar) !RuleId {
    const digit = try grammar.addRule(.{ .char_range = .{ .start = '0', .end = '9' } });
    const dash = try grammar.addRule(.{ .char = '-' });
    const four_digits = try grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &[_]RuleId{ digit, digit, digit, digit }),
    });
    const two_digits = try grammar.addRule(.{ .sequence = try ast.dupeRuleIds(grammar.allocator(), &[_]RuleId{ digit, digit }) });

    return grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &[_]RuleId{ four_digits, dash, two_digits, dash, two_digits }),
    });
}

fn compileTimeNoTz(grammar: *Grammar) !RuleId {
    const digit = try grammar.addRule(.{ .char_range = .{ .start = '0', .end = '9' } });
    const colon = try grammar.addRule(.{ .char = ':' });
    const two_digits = try grammar.addRule(.{ .sequence = try ast.dupeRuleIds(grammar.allocator(), &[_]RuleId{ digit, digit }) });

    return grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &[_]RuleId{ two_digits, colon, two_digits, colon, two_digits }),
    });
}

fn compileTime(grammar: *Grammar) !RuleId {
    return compileTimeNoTz(grammar);
}

fn compileUuid(grammar: *Grammar) !RuleId {
    const hex_lower = try grammar.addRule(.{ .char_range = .{ .start = 'a', .end = 'f' } });
    const hex_upper = try grammar.addRule(.{ .char_range = .{ .start = 'A', .end = 'F' } });
    const digit = try grammar.addRule(.{ .char_range = .{ .start = '0', .end = '9' } });
    const hex_ids = [_]RuleId{ digit, hex_lower, hex_upper };
    const hex = try grammar.addRule(.{
        .alternatives = try ast.dupeRuleIds(grammar.allocator(), &hex_ids),
    });
    const dash = try grammar.addRule(.{ .char = '-' });

    const hex8 = try buildHexGroup(grammar, hex, 8);
    const hex4 = try buildHexGroup(grammar, hex, 4);
    const hex12 = try buildHexGroup(grammar, hex, 12);

    const quote = try grammar.addRule(.{ .char = '"' });
    const inner_ids = [_]RuleId{ hex8, dash, hex4, dash, hex4, dash, hex4, dash, hex12 };
    const inner = try grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &inner_ids),
    });

    return grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &[_]RuleId{ quote, inner, quote }),
    });
}

fn buildHexGroup(grammar: *Grammar, hex: RuleId, count: usize) !RuleId {
    var seq = std.ArrayList(RuleId).empty;
    defer seq.deinit(grammar.allocator());

    for (0..count) |_| {
        try seq.append(grammar.allocator(), hex);
    }

    return grammar.addRule(.{ .sequence = try seq.toOwnedSlice(grammar.allocator()) });
}

fn compileEmail(grammar: *Grammar) !RuleId {
    const alpha_lower = try grammar.addRule(.{ .char_range = .{ .start = 'a', .end = 'z' } });
    const alpha_upper = try grammar.addRule(.{ .char_range = .{ .start = 'A', .end = 'Z' } });
    const digit = try grammar.addRule(.{ .char_range = .{ .start = '0', .end = '9' } });
    const alpha_ids = [_]RuleId{ alpha_lower, alpha_upper };
    const alpha = try grammar.addRule(.{ .alternatives = try ast.dupeRuleIds(grammar.allocator(), &alpha_ids) });
    const alnum_ids = [_]RuleId{ alpha, digit };
    const alnum = try grammar.addRule(.{ .alternatives = try ast.dupeRuleIds(grammar.allocator(), &alnum_ids) });

    const dot = try grammar.addRule(.{ .char = '.' });
    const at = try grammar.addRule(.{ .char = '@' });

    const local = try grammar.addRule(.{ .plus = alnum });

    const domain_part = try grammar.addRule(.{ .plus = alnum });
    const dot_domain = try grammar.addRule(.{ .sequence = try ast.dupeRuleIds(grammar.allocator(), &[_]RuleId{ dot, domain_part }) });
    const domain = try grammar.addRule(.{ .sequence = try ast.dupeRuleIds(grammar.allocator(), &[_]RuleId{ domain_part, dot_domain }) });

    const quote = try grammar.addRule(.{ .char = '"' });
    const inner = try grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &[_]RuleId{ local, at, domain }),
    });

    return grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &[_]RuleId{ quote, inner, quote }),
    });
}

fn compileIpv4(grammar: *Grammar) !RuleId {
    const digit = try grammar.addRule(.{ .char_range = .{ .start = '0', .end = '9' } });
    const dot = try grammar.addRule(.{ .char = '.' });

    const d = try grammar.addRule(.{ .plus = digit });

    const quote = try grammar.addRule(.{ .char = '"' });
    const inner = try grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &[_]RuleId{ d, dot, d, dot, d, dot, d }),
    });

    return grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &[_]RuleId{ quote, inner, quote }),
    });
}

fn compileGenericString(grammar: *Grammar) !RuleId {
    return ast.JsonRules.string(grammar);
}

test "compile date-time format" {
    const allocator = std.testing.allocator;
    var grammar = ast.Grammar.init(allocator);
    defer grammar.deinit();

    grammar.ws_rule = try ast.JsonRules.whitespace(&grammar);
    const rule = try compileFormat(&grammar, .date_time);
    try std.testing.expect(rule != ast.INVALID_RULE);
}

test "compile uuid format" {
    const allocator = std.testing.allocator;
    var grammar = ast.Grammar.init(allocator);
    defer grammar.deinit();

    grammar.ws_rule = try ast.JsonRules.whitespace(&grammar);
    const rule = try compileFormat(&grammar, .uuid);
    try std.testing.expect(rule != ast.INVALID_RULE);
}
