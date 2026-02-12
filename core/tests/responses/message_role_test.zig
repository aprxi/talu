//! Integration tests for MessageRole
//!
//! MessageRole identifies the sender of a message in a conversation.

const std = @import("std");
const main = @import("main");
const MessageRole = main.responses.MessageRole;

// =============================================================================
// Enum Value Tests
// =============================================================================

test "MessageRole has expected values" {
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(MessageRole.system));
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(MessageRole.user));
    try std.testing.expectEqual(@as(u8, 2), @intFromEnum(MessageRole.assistant));
    try std.testing.expectEqual(@as(u8, 3), @intFromEnum(MessageRole.developer));
}

test "MessageRole can be created from integer" {
    try std.testing.expectEqual(MessageRole.system, @as(MessageRole, @enumFromInt(0)));
    try std.testing.expectEqual(MessageRole.user, @as(MessageRole, @enumFromInt(1)));
    try std.testing.expectEqual(MessageRole.assistant, @as(MessageRole, @enumFromInt(2)));
    try std.testing.expectEqual(MessageRole.developer, @as(MessageRole, @enumFromInt(3)));
}

// =============================================================================
// String Conversion Tests
// =============================================================================

test "MessageRole.toString returns correct strings" {
    try std.testing.expectEqualStrings("system", MessageRole.system.toString());
    try std.testing.expectEqualStrings("user", MessageRole.user.toString());
    try std.testing.expectEqualStrings("assistant", MessageRole.assistant.toString());
    try std.testing.expectEqualStrings("developer", MessageRole.developer.toString());
}

test "MessageRole.fromString parses valid strings" {
    try std.testing.expectEqual(MessageRole.system, MessageRole.fromString("system").?);
    try std.testing.expectEqual(MessageRole.user, MessageRole.fromString("user").?);
    try std.testing.expectEqual(MessageRole.assistant, MessageRole.fromString("assistant").?);
    try std.testing.expectEqual(MessageRole.developer, MessageRole.fromString("developer").?);
}

test "MessageRole.fromString returns null for invalid strings" {
    try std.testing.expect(MessageRole.fromString("invalid") == null);
    try std.testing.expect(MessageRole.fromString("") == null);
    try std.testing.expect(MessageRole.fromString("SYSTEM") == null); // case sensitive
    try std.testing.expect(MessageRole.fromString("User") == null);
    try std.testing.expect(MessageRole.fromString("admin") == null);
}

// =============================================================================
// Roundtrip Tests
// =============================================================================

test "MessageRole string roundtrip" {
    const roles = [_]MessageRole{ .system, .user, .assistant, .developer };

    for (roles) |role| {
        const str = role.toString();
        const parsed = MessageRole.fromString(str).?;
        try std.testing.expectEqual(role, parsed);
    }
}

// =============================================================================
// Usage Pattern Tests
// =============================================================================

test "MessageRole works in switch statements" {
    const roles = [_]MessageRole{ .system, .user, .assistant, .developer };
    var counts = [_]u32{ 0, 0, 0, 0 };

    for (roles) |role| {
        switch (role) {
            .system => counts[0] += 1,
            .user => counts[1] += 1,
            .assistant => counts[2] += 1,
            .developer => counts[3] += 1,
        }
    }

    try std.testing.expectEqual(@as(u32, 1), counts[0]);
    try std.testing.expectEqual(@as(u32, 1), counts[1]);
    try std.testing.expectEqual(@as(u32, 1), counts[2]);
    try std.testing.expectEqual(@as(u32, 1), counts[3]);
}

test "MessageRole can be compared" {
    try std.testing.expect(MessageRole.user == MessageRole.user);
    try std.testing.expect(MessageRole.user != MessageRole.assistant);
    try std.testing.expect(MessageRole.system != MessageRole.developer);
}

test "MessageRole can be stored in arrays" {
    const conversation_roles = [_]MessageRole{
        .system,
        .user,
        .assistant,
        .user,
        .assistant,
    };

    try std.testing.expectEqual(@as(usize, 5), conversation_roles.len);
    try std.testing.expectEqual(MessageRole.system, conversation_roles[0]);
    try std.testing.expectEqual(MessageRole.user, conversation_roles[1]);
    try std.testing.expectEqual(MessageRole.assistant, conversation_roles[2]);
}
