//! Integration tests for responses.ItemType
//!
//! ItemType identifies the kind of Item (message, function_call, etc.).

const std = @import("std");
const main = @import("main");
const ItemType = main.responses.ItemType;

test "ItemType enum has expected variants" {
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(ItemType.message));
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(ItemType.function_call));
    try std.testing.expectEqual(@as(u8, 2), @intFromEnum(ItemType.function_call_output));
    try std.testing.expectEqual(@as(u8, 3), @intFromEnum(ItemType.reasoning));
    try std.testing.expectEqual(@as(u8, 4), @intFromEnum(ItemType.item_reference));
}

test "ItemType values are distinct" {
    try std.testing.expect(ItemType.message != ItemType.function_call);
    try std.testing.expect(ItemType.function_call != ItemType.function_call_output);
    try std.testing.expect(ItemType.function_call_output != ItemType.reasoning);
    try std.testing.expect(ItemType.reasoning != ItemType.item_reference);
}
