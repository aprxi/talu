//! Integration tests for responses.Item
//!
//! Item is the atomic unit of the Open Responses architecture.
//! Each Item has a unique ID, type, status, and type-specific payload.

const std = @import("std");
const main = @import("main");

const Item = main.responses.Item;
const ItemType = main.responses.ItemType;
const ItemStatus = main.responses.ItemStatus;
const ItemVariant = main.responses.ItemVariant;
const MessageRole = main.responses.MessageRole;
const Conversation = main.responses.Conversation;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "Item type is accessible" {
    const T = Item;
    _ = T;
}

test "Item is a struct" {
    const info = @typeInfo(Item);
    try std.testing.expect(info == .@"struct");
}

test "Item has expected fields" {
    const info = @typeInfo(Item);
    const fields = info.@"struct".fields;

    var has_id = false;
    var has_item_type = false;
    var has_status = false;
    var has_variant = false;
    var has_created_at_ms = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "id")) has_id = true;
        if (comptime std.mem.eql(u8, field.name, "item_type")) has_item_type = true;
        if (comptime std.mem.eql(u8, field.name, "status")) has_status = true;
        if (comptime std.mem.eql(u8, field.name, "variant")) has_variant = true;
        if (comptime std.mem.eql(u8, field.name, "created_at_ms")) has_created_at_ms = true;
    }

    try std.testing.expect(has_id);
    try std.testing.expect(has_item_type);
    try std.testing.expect(has_status);
    try std.testing.expect(has_variant);
    try std.testing.expect(has_created_at_ms);
}

// =============================================================================
// Item via Conversation Tests
// =============================================================================

test "Item created via Conversation has unique ID" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item1 = try conv.appendUserMessage("Hello");
    const item2 = try conv.appendAssistantMessage();

    try std.testing.expect(item1.id != item2.id);
}

test "Item has correct type based on creation method" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const msg_item = try conv.appendUserMessage("Hello");
    try std.testing.expectEqual(ItemType.message, msg_item.item_type);

    const fc_item = try conv.appendFunctionCall("test_func", "call_1", "{}");
    try std.testing.expectEqual(ItemType.function_call, fc_item.item_type);

    const fco_item = try conv.appendFunctionCallOutput("call_1", "result");
    try std.testing.expectEqual(ItemType.function_call_output, fco_item.item_type);
}

test "Item.asMessage returns MessageData for message items" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendUserMessage("Hello");
    const msg = item.asMessage();

    try std.testing.expect(msg != null);
    try std.testing.expectEqual(MessageRole.user, msg.?.role);
}

test "Item.asMessage returns null for non-message items" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendFunctionCall("test", "id", "{}");
    const msg = item.asMessage();

    try std.testing.expect(msg == null);
}

test "Item.asFunctionCall returns FunctionCallData for function call items" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendFunctionCall("get_weather", "call_123", "{\"city\":\"NYC\"}");
    const fc = item.asFunctionCall();

    try std.testing.expect(fc != null);
    try std.testing.expectEqualStrings("get_weather", fc.?.name);
    try std.testing.expectEqualStrings("call_123", fc.?.call_id);
}

test "Item.asFunctionCallOutput returns FunctionCallOutputData" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendFunctionCallOutput("call_123", "Sunny");
    const fco = item.asFunctionCallOutput();

    try std.testing.expect(fco != null);
    try std.testing.expectEqualStrings("call_123", fco.?.call_id);
}

// =============================================================================
// Item Status Tests
// =============================================================================

test "Item starts with in_progress status" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendUserMessage("Test");
    try std.testing.expectEqual(ItemStatus.in_progress, item.status);
}

test "Item status changes to completed on finalize" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendUserMessage("Test");
    conv.finalizeItem(item);

    try std.testing.expectEqual(ItemStatus.completed, item.status);
}

// =============================================================================
// Item Text Content Tests
// =============================================================================

test "Item.getTextContent returns text for message items" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendUserMessage("Hello, world!");
    try std.testing.expectEqualStrings("Hello, world!", item.getTextContent());
}

test "Item.getTextContent returns empty for function call items" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendFunctionCall("test", "id", "{}");
    try std.testing.expectEqualStrings("", item.getTextContent());
}

// =============================================================================
// Item Timestamp Tests
// =============================================================================

test "Item has valid created_at_ms timestamp" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const before = std.time.milliTimestamp();
    const item = try conv.appendUserMessage("Test");
    const after = std.time.milliTimestamp();

    try std.testing.expect(item.created_at_ms >= before);
    try std.testing.expect(item.created_at_ms <= after);
}
