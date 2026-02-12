//! Schema -> GBNF compilation helpers (placeholder).

const std = @import("std");
const ast = @import("ast.zig");

pub const GbnfError = error{
    Unsupported,
};

pub fn toGbnf(_: *const ast.Grammar, _: std.mem.Allocator) GbnfError![]const u8 {
    return GbnfError.Unsupported;
}

test "toGbnf reports unsupported for now" {
    const allocator = std.testing.allocator;
    var grammar = ast.Grammar.init(allocator);
    defer grammar.deinit();

    try std.testing.expectError(GbnfError.Unsupported, toGbnf(&grammar, allocator));
}

test "toGbnf remains unsupported with populated grammar" {
    const allocator = std.testing.allocator;
    var grammar = ast.Grammar.init(allocator);
    defer grammar.deinit();

    const rule_id = try grammar.addRule(.{ .char = 'x' });
    grammar.root_rule = rule_id;

    try std.testing.expectError(GbnfError.Unsupported, toGbnf(&grammar, allocator));
}
