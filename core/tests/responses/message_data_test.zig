//! Integration tests for responses.MessageData
//!
//! MessageData is the payload for message-type Items.
//! Each message contains role, content parts, and status.

const std = @import("std");
const main = @import("main");

const Conversation = main.responses.Conversation;
const Item = main.responses.Item;
const MessageData = main.responses.MessageData;
const MessageRole = main.responses.MessageRole;
const ItemStatus = main.responses.ItemStatus;
const ContentType = main.responses.ContentType;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "MessageData type is accessible" {
    const T = MessageData;
    _ = T;
}

test "MessageData is a struct" {
    const info = @typeInfo(MessageData);
    try std.testing.expect(info == .@"struct");
}

test "MessageData has expected fields" {
    const info = @typeInfo(MessageData);
    const fields = info.@"struct".fields;

    var has_role = false;
    var has_status = false;
    var has_content = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "role")) has_role = true;
        if (comptime std.mem.eql(u8, field.name, "status")) has_status = true;
        if (comptime std.mem.eql(u8, field.name, "content")) has_content = true;
    }

    try std.testing.expect(has_role);
    try std.testing.expect(has_status);
    try std.testing.expect(has_content);
}

// =============================================================================
// MessageData via Conversation Tests
// =============================================================================

test "Conversation.appendUserMessage creates MessageData with user role" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendUserMessage("Hello");
    const msg = item.asMessage().?;

    try std.testing.expectEqual(MessageRole.user, msg.role);
    try std.testing.expectEqual(ItemStatus.in_progress, msg.status);
}

test "Conversation.appendSystemMessage creates MessageData with system role" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendSystemMessage("You are helpful");
    const msg = item.asMessage().?;

    try std.testing.expectEqual(MessageRole.system, msg.role);
}

test "Conversation.appendAssistantMessage creates MessageData with assistant role" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendAssistantMessage();
    const msg = item.asMessage().?;

    try std.testing.expectEqual(MessageRole.assistant, msg.role);
}

test "Conversation.appendDeveloperMessage creates MessageData with developer role" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendDeveloperMessage("Developer instruction");
    const msg = item.asMessage().?;

    try std.testing.expectEqual(MessageRole.developer, msg.role);
}

// =============================================================================
// Content Part Tests
// =============================================================================

test "MessageData content starts empty for assistant" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendAssistantMessage();
    const msg = item.asMessage().?;

    try std.testing.expectEqual(@as(usize, 0), msg.content.items.len);
}

test "MessageData content can be streamed" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendAssistantMessage();

    try conv.appendTextContent(item, "Hello");
    try conv.appendTextContent(item, ", world!");

    try std.testing.expectEqualStrings("Hello, world!", item.getTextContent());
}

// =============================================================================
// Status Tests
// =============================================================================

test "MessageData starts as in_progress" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendUserMessage("Test");
    const msg = item.asMessage().?;

    try std.testing.expectEqual(ItemStatus.in_progress, msg.status);
}

test "MessageData status changes to completed on finalize" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendUserMessage("Test");

    conv.finalizeItem(item);

    try std.testing.expectEqual(ItemStatus.completed, item.status);
}

// =============================================================================
// Role Tests
// =============================================================================

test "MessageRole has expected values" {
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(MessageRole.system));
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(MessageRole.user));
    try std.testing.expectEqual(@as(u8, 2), @intFromEnum(MessageRole.assistant));
    try std.testing.expectEqual(@as(u8, 3), @intFromEnum(MessageRole.developer));
}

test "MessageRole.toString returns correct strings" {
    try std.testing.expectEqualStrings("system", MessageRole.system.toString());
    try std.testing.expectEqualStrings("user", MessageRole.user.toString());
    try std.testing.expectEqualStrings("assistant", MessageRole.assistant.toString());
    try std.testing.expectEqualStrings("developer", MessageRole.developer.toString());
}

test "MessageRole.fromString parses valid roles" {
    try std.testing.expectEqual(MessageRole.system, MessageRole.fromString("system").?);
    try std.testing.expectEqual(MessageRole.user, MessageRole.fromString("user").?);
    try std.testing.expectEqual(MessageRole.assistant, MessageRole.fromString("assistant").?);
    try std.testing.expectEqual(MessageRole.developer, MessageRole.fromString("developer").?);
}

test "MessageRole.fromString returns null for invalid" {
    try std.testing.expect(MessageRole.fromString("invalid") == null);
    try std.testing.expect(MessageRole.fromString("") == null);
    try std.testing.expect(MessageRole.fromString("SYSTEM") == null);
}
