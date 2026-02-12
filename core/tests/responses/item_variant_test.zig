//! Integration tests for responses.ItemVariant
//!
//! ItemVariant is a tagged union containing the type-specific payload for an Item.

const std = @import("std");
const main = @import("main");
const ItemVariant = main.responses.ItemVariant;

test "ItemVariant is a tagged union" {
    const info = @typeInfo(ItemVariant);
    try std.testing.expect(info == .@"union");
}

test "ItemVariant has expected variants" {
    const info = @typeInfo(ItemVariant);
    const fields = info.@"union".fields;

    var has_message = false;
    var has_function_call = false;
    var has_function_call_output = false;
    var has_reasoning = false;
    var has_item_reference = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "message")) has_message = true;
        if (comptime std.mem.eql(u8, field.name, "function_call")) has_function_call = true;
        if (comptime std.mem.eql(u8, field.name, "function_call_output")) has_function_call_output = true;
        if (comptime std.mem.eql(u8, field.name, "reasoning")) has_reasoning = true;
        if (comptime std.mem.eql(u8, field.name, "item_reference")) has_item_reference = true;
    }

    try std.testing.expect(has_message);
    try std.testing.expect(has_function_call);
    try std.testing.expect(has_function_call_output);
    try std.testing.expect(has_reasoning);
    try std.testing.expect(has_item_reference);
}
