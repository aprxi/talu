//! Integration tests for Conversation
//!
//! Conversation is the container for Open Responses Items.
//! It provides typed append functions for each item type.

const std = @import("std");
const main = @import("main");
const Conversation = main.responses.Conversation;
const Item = main.responses.Item;
const ItemType = main.responses.ItemType;
const ItemStatus = main.responses.ItemStatus;
const MessageRole = main.responses.MessageRole;
const ContentType = main.responses.ContentType;
const MemoryBackend = main.responses.MemoryBackend;

// =============================================================================
// Lifecycle Tests
// =============================================================================

test "Conversation.init creates empty container" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    try std.testing.expectEqual(@as(usize, 0), conv.len());
}

test "Conversation.deinit frees all resources" {
    const conv = try Conversation.init(std.testing.allocator);

    // Add some items
    const user_item = try conv.appendUserMessage("Hello");
    conv.finalizeItem(user_item);

    const asst_item = try conv.appendAssistantMessage();
    try conv.appendTextContent(asst_item, "Hi there!");
    conv.finalizeItem(asst_item);

    // Should not leak memory
    conv.deinit();
}

// =============================================================================
// Item Creation Tests
// =============================================================================

test "Conversation.appendUserMessage creates user message item" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendUserMessage("Hello");

    try std.testing.expectEqual(ItemType.message, item.item_type);
    try std.testing.expectEqual(ItemStatus.in_progress, item.status);

    const msg = item.asMessage().?;
    try std.testing.expectEqual(MessageRole.user, msg.role);
}

test "Conversation.appendSystemMessage creates system message item" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendSystemMessage("You are helpful");

    const msg = item.asMessage().?;
    try std.testing.expectEqual(MessageRole.system, msg.role);
}

test "Conversation.appendAssistantMessage creates assistant message item" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendAssistantMessage();

    const msg = item.asMessage().?;
    try std.testing.expectEqual(MessageRole.assistant, msg.role);
}

test "Conversation item IDs are unique and monotonic" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item0 = try conv.appendSystemMessage("System");
    const item1 = try conv.appendUserMessage("User");
    const item2 = try conv.appendAssistantMessage();

    try std.testing.expect(item0.id < item1.id);
    try std.testing.expect(item1.id < item2.id);
}

// =============================================================================
// Content Streaming Tests
// =============================================================================

test "Conversation.appendTextContent builds content incrementally" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendAssistantMessage();

    try conv.appendTextContent(item, "Hello");
    try conv.appendTextContent(item, ", ");
    try conv.appendTextContent(item, "world!");

    const text = item.getTextContent();
    try std.testing.expectEqualStrings("Hello, world!", text);
}

test "Conversation.finalizeItem marks item as completed" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendUserMessage("Test");

    try std.testing.expectEqual(ItemStatus.in_progress, item.status);

    conv.finalizeItem(item);

    try std.testing.expectEqual(ItemStatus.completed, item.status);
}

// =============================================================================
// Access Tests
// =============================================================================

test "Conversation.len returns correct count" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    try std.testing.expectEqual(@as(usize, 0), conv.len());

    _ = try conv.appendUserMessage("First");
    try std.testing.expectEqual(@as(usize, 1), conv.len());

    _ = try conv.appendAssistantMessage();
    try std.testing.expectEqual(@as(usize, 2), conv.len());
}

test "Conversation.getItem returns item by index" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    _ = try conv.appendSystemMessage("System prompt");
    _ = try conv.appendUserMessage("User message");

    const item0 = conv.getItem(0).?;
    const item1 = conv.getItem(1).?;

    try std.testing.expectEqual(MessageRole.system, item0.asMessage().?.role);
    try std.testing.expectEqual(MessageRole.user, item1.asMessage().?.role);
}

test "Conversation.getItem returns null for out of bounds" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    _ = try conv.appendUserMessage("Test");

    try std.testing.expect(conv.getItem(0) != null);
    try std.testing.expect(conv.getItem(1) == null);
    try std.testing.expect(conv.getItem(100) == null);
}

test "Conversation.lastItem returns last item" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    try std.testing.expect(conv.lastItem() == null);

    _ = try conv.appendUserMessage("First");
    const last1 = conv.lastItem().?;
    try std.testing.expectEqual(MessageRole.user, last1.asMessage().?.role);

    _ = try conv.appendAssistantMessage();
    const last2 = conv.lastItem().?;
    try std.testing.expectEqual(MessageRole.assistant, last2.asMessage().?.role);
}

test "Conversation.findById returns item by ID" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendUserMessage("Find me");
    const id = item.id;

    const found = conv.findById(id).?;
    try std.testing.expectEqual(id, found.id);

    try std.testing.expect(conv.findById(99999) == null);
}

// =============================================================================
// Clear Tests
// =============================================================================

test "Conversation.clear removes all items" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    _ = try conv.appendSystemMessage("System");
    _ = try conv.appendUserMessage("User");
    _ = try conv.appendAssistantMessage();

    try std.testing.expectEqual(@as(usize, 3), conv.len());

    conv.clear();

    try std.testing.expectEqual(@as(usize, 0), conv.len());
}

test "Conversation.clearKeepingContext preserves system message" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const sys = try conv.appendSystemMessage("You are helpful");
    conv.finalizeItem(sys);

    _ = try conv.appendUserMessage("User");
    _ = try conv.appendAssistantMessage();

    try std.testing.expectEqual(@as(usize, 3), conv.len());

    conv.clearKeepingContext();

    try std.testing.expectEqual(@as(usize, 1), conv.len());
    const remaining = conv.getItem(0).?;
    try std.testing.expectEqual(MessageRole.system, remaining.asMessage().?.role);
}

// =============================================================================
// Storage Backend Integration Tests
// =============================================================================

test "Conversation.initWithStorage accepts backend" {
    var mem = MemoryBackend{ .debug_stats = true };
    const conv = try Conversation.initWithStorage(std.testing.allocator, null, mem.backend());
    defer conv.deinit();

    const item = try conv.appendUserMessage("Hello");
    conv.finalizeItem(item);

    // Backend should have been notified
    try std.testing.expectEqual(@as(usize, 1), mem._debug_persist_count);
}

// =============================================================================
// Function Call Tests
// =============================================================================

test "Conversation.appendFunctionCall creates function call item" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendFunctionCall("get_weather", "call_123", "{\"city\":\"NYC\"}");

    try std.testing.expectEqual(ItemType.function_call, item.item_type);
    const fc = item.asFunctionCall().?;
    try std.testing.expectEqualStrings("get_weather", fc.name);
    try std.testing.expectEqualStrings("call_123", fc.call_id);
}

test "Conversation.appendFunctionCallOutput creates function output item" {
    const conv = try Conversation.init(std.testing.allocator);
    defer conv.deinit();

    const item = try conv.appendFunctionCallOutput("call_123", "Sunny, 72F");

    try std.testing.expectEqual(ItemType.function_call_output, item.item_type);
    const fco = item.asFunctionCallOutput().?;
    try std.testing.expectEqualStrings("call_123", fco.call_id);
}
