//! Generic JSON grammar for unconstrained json_object output.

const std = @import("std");
const ast = @import("ast.zig");

const Grammar = ast.Grammar;
const RuleId = ast.RuleId;
const CompileError = std.mem.Allocator.Error;

pub fn compileGeneric(grammar: *Grammar) CompileError!RuleId {
    const ws = grammar.ws_rule;
    const string_rule = try ast.JsonRules.string(grammar);
    const number_rule = try ast.JsonRules.number(grammar);
    const bool_rule = try ast.JsonRules.boolean(grammar);
    const null_rule = try ast.JsonRules.null_(grammar);

    const value_placeholder = try grammar.addRule(.{ .end = {} });
    const array_placeholder = try grammar.addRule(.{ .end = {} });
    const object_placeholder = try grammar.addRule(.{ .end = {} });

    const value_ref = try grammar.addRule(.{ .reference = value_placeholder });
    const array_ref = try grammar.addRule(.{ .reference = array_placeholder });
    const object_ref = try grammar.addRule(.{ .reference = object_placeholder });

    const array_rule = try compileArray(grammar, value_ref);
    const object_rule = try compileObject(grammar, value_ref);

    const alt_ids = [_]RuleId{
        string_rule,
        number_rule,
        bool_rule,
        null_rule,
        array_ref,
        object_ref,
    };
    const alt_rule = try grammar.addRule(.{
        .alternatives = try ast.dupeRuleIds(grammar.allocator(), &alt_ids),
    });
    const value_ids = [_]RuleId{ ws, alt_rule, ws };
    const value_rule = try grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &value_ids),
    });

    grammar.rules.items[value_placeholder] = grammar.rules.items[value_rule];
    grammar.rules.items[array_placeholder] = grammar.rules.items[array_rule];
    grammar.rules.items[object_placeholder] = grammar.rules.items[object_rule];

    return value_placeholder;
}

fn compileArray(grammar: *Grammar, value_rule: RuleId) CompileError!RuleId {
    const ws = grammar.ws_rule;
    const comma = try grammar.addRule(.{ .char = ',' });
    const open = try grammar.addRule(.{ .char = '[' });
    const close = try grammar.addRule(.{ .char = ']' });

    const tail_ids = [_]RuleId{ ws, comma, ws, value_rule };
    const tail = try grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &tail_ids),
    });
    const tail_star = try grammar.addRule(.{ .star = tail });
    const items_ids = [_]RuleId{ value_rule, tail_star };
    const items = try grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &items_ids),
    });
    const opt_items = try grammar.addRule(.{ .optional = items });

    const array_ids = [_]RuleId{ open, ws, opt_items, ws, close };
    return grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &array_ids),
    });
}

fn compileObject(grammar: *Grammar, value_rule: RuleId) CompileError!RuleId {
    const ws = grammar.ws_rule;
    const comma = try grammar.addRule(.{ .char = ',' });
    const colon = try grammar.addRule(.{ .char = ':' });
    const open = try grammar.addRule(.{ .char = '{' });
    const close = try grammar.addRule(.{ .char = '}' });

    const key = try ast.JsonRules.string(grammar);

    const pair_ids = [_]RuleId{ key, ws, colon, ws, value_rule };
    const pair = try grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &pair_ids),
    });
    const tail_ids = [_]RuleId{ ws, comma, ws, pair };
    const tail = try grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &tail_ids),
    });
    const tail_star = try grammar.addRule(.{ .star = tail });
    const pairs_ids = [_]RuleId{ pair, tail_star };
    const pairs = try grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &pairs_ids),
    });
    const opt_pairs = try grammar.addRule(.{ .optional = pairs });

    const object_ids = [_]RuleId{ open, ws, opt_pairs, ws, close };
    return grammar.addRule(.{
        .sequence = try ast.dupeRuleIds(grammar.allocator(), &object_ids),
    });
}

test "generic json accepts valid objects" {
    const allocator = std.testing.allocator;

    var grammar = ast.Grammar.init(allocator);
    defer grammar.deinit();

    grammar.ws_rule = try ast.JsonRules.whitespace(&grammar);
    grammar.root_rule = try compileGeneric(&grammar);

    try std.testing.expect(grammar.root_rule != ast.INVALID_RULE);
}

test "compileGeneric builds grammar that accepts object literals" {
    const allocator = std.testing.allocator;

    var grammar = ast.Grammar.init(allocator);
    defer grammar.deinit();

    grammar.ws_rule = try ast.JsonRules.whitespace(&grammar);
    grammar.root_rule = try compileGeneric(&grammar);

    var engine = try @import("engine.zig").Engine.init(allocator, &grammar);
    defer engine.deinit();

    try std.testing.expect(try engine.canAccept("{}"));
    try engine.reset();
    try std.testing.expect(try engine.canAccept("{\"a\":1}"));
}

test "compileGeneric builds grammar that accepts arrays" {
    const allocator = std.testing.allocator;

    var grammar = ast.Grammar.init(allocator);
    defer grammar.deinit();

    grammar.ws_rule = try ast.JsonRules.whitespace(&grammar);
    grammar.root_rule = try compileGeneric(&grammar);

    var engine = try @import("engine.zig").Engine.init(allocator, &grammar);
    defer engine.deinit();

    try std.testing.expect(try engine.canAccept("[]"));
    try engine.reset();
    try std.testing.expect(try engine.canAccept("[1, 2, 3]"));
}
