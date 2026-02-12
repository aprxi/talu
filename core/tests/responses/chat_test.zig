//! Integration tests for responses.Chat
//!
//! Chat is the lightweight chat state that manages conversation history
//! and generation preferences. It wraps Conversation and provides high-level
//! conversation operations.

const std = @import("std");
const main = @import("main");

const Chat = main.responses.Chat;
const Conversation = main.responses.Conversation;
const MessageRole = main.responses.MessageRole;
const ItemStatus = main.responses.ItemStatus;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "Chat type is accessible" {
    const T = Chat;
    _ = T;
}

test "Chat is a struct" {
    const info = @typeInfo(Chat);
    try std.testing.expect(info == .@"struct");
}

// =============================================================================
// Lifecycle Tests
// =============================================================================

test "Chat.init creates empty chat" {
    var chat = try Chat.init(std.testing.allocator);
    defer chat.deinit();

    try std.testing.expectEqual(@as(usize, 0), chat.len());
}

test "Chat.initWithSystem creates chat with system prompt" {
    var chat = try Chat.initWithSystem(std.testing.allocator, "You are helpful.");
    defer chat.deinit();

    try std.testing.expectEqual(@as(usize, 1), chat.len());
    try std.testing.expectEqualStrings("You are helpful.", chat.getSystem().?);
}

test "Chat.deinit frees resources" {
    var chat = try Chat.initWithSystem(std.testing.allocator, "System");
    try chat.append(.user, "Hello");
    try chat.append(.assistant, "Hi");
    chat.deinit();
    // No leak = success
}

// =============================================================================
// Message Append Tests
// =============================================================================

test "Chat.append adds messages" {
    var chat = try Chat.init(std.testing.allocator);
    defer chat.deinit();

    try chat.append(.user, "Hello!");
    try std.testing.expectEqual(@as(usize, 1), chat.len());

    try chat.append(.assistant, "Hi there!");
    try std.testing.expectEqual(@as(usize, 2), chat.len());
}

test "Chat.appendRole uses MessageRole" {
    var chat = try Chat.init(std.testing.allocator);
    defer chat.deinit();

    try chat.appendRole(.user, "User message");
    try chat.appendRole(.assistant, "Assistant message");
    try chat.appendRole(.developer, "Developer message");

    try std.testing.expectEqual(@as(usize, 3), chat.len());
}

// =============================================================================
// Streaming Tests
// =============================================================================

test "Chat.startStreaming creates streaming item" {
    var chat = try Chat.init(std.testing.allocator);
    defer chat.deinit();

    const item = try chat.startStreaming(.assistant);
    try std.testing.expectEqual(ItemStatus.in_progress, item.status);
    try std.testing.expectEqual(MessageRole.assistant, item.asMessage().?.role);
}

test "Chat.appendToStreaming builds content" {
    var chat = try Chat.init(std.testing.allocator);
    defer chat.deinit();

    const item = try chat.startStreaming(.assistant);

    try chat.appendToStreaming(item, "Hello");
    try chat.appendToStreaming(item, ", world!");

    try std.testing.expectEqualStrings("Hello, world!", item.getTextContent());
}

test "Chat.finalizeStreaming marks item as completed" {
    var chat = try Chat.init(std.testing.allocator);
    defer chat.deinit();

    const item = try chat.startStreaming(.assistant);
    try chat.appendToStreaming(item, "Done");

    try std.testing.expectEqual(ItemStatus.in_progress, item.status);

    chat.finalizeStreaming(item);

    try std.testing.expectEqual(ItemStatus.completed, item.status);
}

// =============================================================================
// System Prompt Tests
// =============================================================================

test "Chat.setSystem creates system message" {
    var chat = try Chat.init(std.testing.allocator);
    defer chat.deinit();

    try chat.setSystem("Be helpful");

    try std.testing.expectEqual(@as(usize, 1), chat.len());
    try std.testing.expectEqualStrings("Be helpful", chat.getSystem().?);
}

test "Chat.setSystem updates existing system message" {
    var chat = try Chat.initWithSystem(std.testing.allocator, "Old prompt");
    defer chat.deinit();

    try chat.setSystem("New prompt");

    try std.testing.expectEqual(@as(usize, 1), chat.len());
    try std.testing.expectEqualStrings("New prompt", chat.getSystem().?);
}

test "Chat.clearSystem removes system message" {
    var chat = try Chat.initWithSystem(std.testing.allocator, "System");
    defer chat.deinit();

    try chat.append(.user, "Hello");

    chat.clearSystem();

    try std.testing.expectEqual(@as(usize, 1), chat.len());
    try std.testing.expect(chat.getSystem() == null);
}

// =============================================================================
// Clear and Reset Tests
// =============================================================================

test "Chat.clear keeps system prompt" {
    var chat = try Chat.initWithSystem(std.testing.allocator, "System");
    defer chat.deinit();

    try chat.append(.user, "Hello");
    try chat.append(.assistant, "Hi");

    chat.clear();

    try std.testing.expectEqual(@as(usize, 1), chat.len());
    try std.testing.expectEqualStrings("System", chat.getSystem().?);
}

test "Chat.reset clears everything including settings" {
    var chat = try Chat.initWithSystem(std.testing.allocator, "System");
    defer chat.deinit();

    chat.temperature = 0.5;
    chat.max_tokens = 100;

    chat.reset();

    try std.testing.expectEqual(@as(usize, 0), chat.len());
    try std.testing.expect(chat.getSystem() == null);
    try std.testing.expectEqual(@as(f32, 0.7), chat.temperature);
    try std.testing.expectEqual(@as(usize, 256), chat.max_tokens);
}

// =============================================================================
// Conversation Access Tests
// =============================================================================

test "Chat.getConversation returns Conversation pointer" {
    var chat = try Chat.initWithSystem(std.testing.allocator, "System");
    defer chat.deinit();

    const conv = chat.getConversation();
    try std.testing.expectEqual(@as(usize, 1), conv.len());
}

// =============================================================================
// Default Values Tests
// =============================================================================

test "Chat has correct default sampling parameters" {
    var chat = try Chat.init(std.testing.allocator);
    defer chat.deinit();

    try std.testing.expectEqual(@as(f32, 0.7), chat.temperature);
    try std.testing.expectEqual(@as(usize, 256), chat.max_tokens);
    try std.testing.expectEqual(@as(usize, 50), chat.top_k);
    try std.testing.expectEqual(@as(f32, 0.9), chat.top_p);
    try std.testing.expectEqual(@as(f32, 0.0), chat.min_p);
    try std.testing.expectEqual(@as(f32, 1.0), chat.repetition_penalty);
}
