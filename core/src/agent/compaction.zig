//! Compaction - Turn-aware context window management for agents.
//!
//! Groups conversation items into logical "turns" and compacts by removing
//! the oldest unpinned turns first. This prevents orphaning tool call outputs,
//! splitting user-assistant exchanges mid-turn, and respects pinned items.
//!
//! A "turn" starts at each user or developer message and includes all
//! subsequent assistant messages, function calls, function call outputs, and
//! reasoning items until the next user/developer message.
//!
//! # Thread Safety
//!
//! NOT thread-safe. Caller must ensure exclusive access to the Conversation
//! during compaction.

const std = @import("std");
const Allocator = std.mem.Allocator;

const responses = @import("../responses/root.zig");
const Conversation = responses.Conversation;
const Item = responses.Item;
const ItemType = responses.ItemType;
const MessageRole = responses.MessageRole;

// =============================================================================
// Turn
// =============================================================================

/// A logical turn: a group of contiguous items compacted atomically.
///
/// Turn boundaries are defined by user/developer messages. A turn starts at
/// a user or developer message and includes all subsequent assistant responses,
/// function calls, function call outputs, and reasoning items until the next
/// user/developer message.
///
/// The system prompt (index 0) is never included in any turn.
pub const Turn = struct {
    /// Start index in the conversation (inclusive).
    start: usize,
    /// End index in the conversation (exclusive).
    end: usize,
    /// Sum of input_tokens + output_tokens for all items in this turn.
    total_tokens: u64,
    /// Whether any item in this turn has pinned=true.
    has_pinned: bool,
};

// =============================================================================
// Turn identification
// =============================================================================

/// Scan the conversation and group items into turns.
///
/// Rules:
///   - Index 0 is skipped if it's a system/developer message (system prompt).
///   - A new turn starts at each user or developer message.
///   - All items between two user/developer messages belong to the preceding turn.
///   - A turn is "pinned" if any item within it has pinned=true.
///
/// Caller owns the returned slice.
pub fn identifyTurns(allocator: Allocator, conv: *const Conversation) ![]Turn {
    var turns = std.ArrayListUnmanaged(Turn){};
    errdefer turns.deinit(allocator);

    const count = conv.len();
    if (count == 0) return turns.toOwnedSlice(allocator);

    // Determine where to start scanning. Skip the system prompt at index 0
    // if it's a system or developer message.
    var scan_start: usize = 0;
    if (conv.getItem(0)) |first| {
        if (first.asMessage()) |msg| {
            if (msg.role == .system or msg.role == .developer) {
                scan_start = 1;
            }
        }
    }

    if (scan_start >= count) return turns.toOwnedSlice(allocator);

    var current_start: ?usize = null;
    var current_tokens: u64 = 0;
    var current_pinned: bool = false;

    var i: usize = scan_start;
    while (i < count) : (i += 1) {
        const item = conv.getItem(i) orelse continue;

        const is_turn_boundary = isTurnBoundary(item);

        if (is_turn_boundary) {
            // Close the previous turn if one is open.
            if (current_start) |start| {
                try turns.append(allocator, .{
                    .start = start,
                    .end = i,
                    .total_tokens = current_tokens,
                    .has_pinned = current_pinned,
                });
            }
            // Start a new turn.
            current_start = i;
            current_tokens = itemTokens(item);
            current_pinned = item.pinned;
        } else {
            // Extend current turn.
            if (current_start == null) {
                // Items before the first user/developer message form their own turn.
                current_start = i;
                current_tokens = 0;
                current_pinned = false;
            }
            current_tokens += itemTokens(item);
            if (item.pinned) current_pinned = true;
        }
    }

    // Close the last open turn.
    if (current_start) |start| {
        try turns.append(allocator, .{
            .start = start,
            .end = count,
            .total_tokens = current_tokens,
            .has_pinned = current_pinned,
        });
    }

    return turns.toOwnedSlice(allocator);
}

/// Check whether an item is a turn boundary (starts a new turn).
/// User and developer messages start new turns.
fn isTurnBoundary(item: *const Item) bool {
    const msg = item.asMessage() orelse return false;
    return msg.role == .user or msg.role == .developer;
}

/// Sum of input + output tokens for an item.
fn itemTokens(item: *const Item) u64 {
    return @as(u64, item.input_tokens) + @as(u64, item.output_tokens);
}

// =============================================================================
// Compaction
// =============================================================================

/// Delete the oldest unpinned turns until total tokens <= target.
///
/// Strategy:
///   1. Delete complete unpinned turns from oldest to newest.
///   2. Never delete pinned turns or the system prompt (index 0).
///   3. Items within a turn are deleted in reverse index order to avoid
///      index shifting during deletion.
///
/// Returns the number of items deleted.
pub fn compactTurns(
    conv: *Conversation,
    turns: []const Turn,
    target_tokens: u64,
    current_tokens: u64,
) usize {
    if (current_tokens <= target_tokens) return 0;

    var remaining_tokens = current_tokens;
    var total_deleted: usize = 0;

    for (turns) |turn| {
        if (remaining_tokens <= target_tokens) break;
        if (turn.has_pinned) continue;

        // Delete items in this turn in reverse order to avoid index shifting.
        const turn_item_count = turn.end - turn.start;
        var j: usize = turn_item_count;
        while (j > 0) {
            j -= 1;
            // Indices shift left by total_deleted as earlier items are removed.
            const adjusted_idx = turn.start - total_deleted + j;
            _ = conv.deleteItem(adjusted_idx);
        }

        remaining_tokens -|= turn.total_tokens;
        total_deleted += turn_item_count;
    }

    return total_deleted;
}

// =============================================================================
// Tests
// =============================================================================

test "identifyTurns with empty conversation" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    const turns = try identifyTurns(allocator, conv);
    defer allocator.free(turns);

    try std.testing.expectEqual(@as(usize, 0), turns.len);
}

test "identifyTurns with system prompt only" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    _ = try conv.appendSystemMessage("You are an assistant.");

    const turns = try identifyTurns(allocator, conv);
    defer allocator.free(turns);

    try std.testing.expectEqual(@as(usize, 0), turns.len);
}

test "identifyTurns with single user turn" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    _ = try conv.appendSystemMessage("System prompt.");
    _ = try conv.appendUserMessage("Hello");

    const turns = try identifyTurns(allocator, conv);
    defer allocator.free(turns);

    try std.testing.expectEqual(@as(usize, 1), turns.len);
    try std.testing.expectEqual(@as(usize, 1), turns[0].start);
    try std.testing.expectEqual(@as(usize, 2), turns[0].end);
    try std.testing.expect(!turns[0].has_pinned);
}

test "identifyTurns groups user + assistant + tool calls into one turn" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    _ = try conv.appendSystemMessage("System.");
    _ = try conv.appendUserMessage("Do something");
    _ = try conv.appendAssistantMessage();
    _ = try conv.appendFunctionCall("call_1", "search");
    _ = try conv.appendFunctionCallOutput("call_1", "results");

    const turns = try identifyTurns(allocator, conv);
    defer allocator.free(turns);

    try std.testing.expectEqual(@as(usize, 1), turns.len);
    try std.testing.expectEqual(@as(usize, 1), turns[0].start);
    try std.testing.expectEqual(@as(usize, 5), turns[0].end);
}

test "identifyTurns with multiple turns" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    _ = try conv.appendSystemMessage("System.");
    // Turn 1: user + assistant
    _ = try conv.appendUserMessage("Hello");
    _ = try conv.appendAssistantMessage();
    // Turn 2: user + assistant
    _ = try conv.appendUserMessage("Next question");
    _ = try conv.appendAssistantMessage();

    const turns = try identifyTurns(allocator, conv);
    defer allocator.free(turns);

    try std.testing.expectEqual(@as(usize, 2), turns.len);
    try std.testing.expectEqual(@as(usize, 1), turns[0].start);
    try std.testing.expectEqual(@as(usize, 3), turns[0].end);
    try std.testing.expectEqual(@as(usize, 3), turns[1].start);
    try std.testing.expectEqual(@as(usize, 5), turns[1].end);
}

test "identifyTurns detects pinned items" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    _ = try conv.appendSystemMessage("System.");
    // Turn 1: unpinned
    _ = try conv.appendUserMessage("Hello");
    _ = try conv.appendAssistantMessage();
    // Turn 2: pinned
    _ = try conv.appendUserMessage("Important");
    const pinned_item = try conv.appendAssistantMessage();
    pinned_item.pinned = true;

    const turns = try identifyTurns(allocator, conv);
    defer allocator.free(turns);

    try std.testing.expectEqual(@as(usize, 2), turns.len);
    try std.testing.expect(!turns[0].has_pinned);
    try std.testing.expect(turns[1].has_pinned);
}

test "compactTurns deletes oldest unpinned turn first" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    _ = try conv.appendSystemMessage("System.");
    // Turn 1: user + assistant (set token counts)
    const msg_u1 = try conv.appendUserMessage("Turn 1");
    msg_u1.input_tokens = 100;
    const msg_a1 = try conv.appendAssistantMessage();
    msg_a1.output_tokens = 100;
    // Turn 2: user + assistant
    const msg_u2 = try conv.appendUserMessage("Turn 2");
    msg_u2.input_tokens = 100;
    const msg_a2 = try conv.appendAssistantMessage();
    msg_a2.output_tokens = 100;

    const turns = try identifyTurns(allocator, conv);
    defer allocator.free(turns);

    // Total tokens = 400, target = 250
    const deleted = compactTurns(conv, turns, 250, 400);
    try std.testing.expectEqual(@as(usize, 2), deleted);
    // System + Turn 2 remain
    try std.testing.expectEqual(@as(usize, 3), conv.len());
    // First remaining non-system item should be Turn 2's user message
    if (conv.getItem(1)) |item| {
        const msg = item.asMessage() orelse return error.TestUnexpectedResult;
        try std.testing.expectEqualStrings("Turn 2", msg.getFirstText());
    } else return error.TestUnexpectedResult;
}

test "compactTurns skips pinned turns" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    _ = try conv.appendSystemMessage("System.");
    // Turn 1: pinned
    const pin_msg = try conv.appendUserMessage("Pinned turn");
    pin_msg.input_tokens = 100;
    pin_msg.pinned = true;
    const pin_asst = try conv.appendAssistantMessage();
    pin_asst.output_tokens = 100;
    // Turn 2: unpinned
    const unpin_msg = try conv.appendUserMessage("Unpinned turn");
    unpin_msg.input_tokens = 100;
    const unpin_asst = try conv.appendAssistantMessage();
    unpin_asst.output_tokens = 100;

    const turns = try identifyTurns(allocator, conv);
    defer allocator.free(turns);

    // Total = 400, target = 250. Turn 1 is pinned so Turn 2 gets deleted.
    const deleted = compactTurns(conv, turns, 250, 400);
    try std.testing.expectEqual(@as(usize, 2), deleted);
    // System + Turn 1 remain
    try std.testing.expectEqual(@as(usize, 3), conv.len());
    if (conv.getItem(1)) |item| {
        const msg = item.asMessage() orelse return error.TestUnexpectedResult;
        try std.testing.expectEqualStrings("Pinned turn", msg.getFirstText());
    } else return error.TestUnexpectedResult;
}

test "compactTurns preserves system prompt" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    _ = try conv.appendSystemMessage("System.");
    const sp_user = try conv.appendUserMessage("Hello");
    sp_user.input_tokens = 100;
    const sp_asst = try conv.appendAssistantMessage();
    sp_asst.output_tokens = 100;

    const turns = try identifyTurns(allocator, conv);
    defer allocator.free(turns);

    const deleted = compactTurns(conv, turns, 50, 200);
    try std.testing.expectEqual(@as(usize, 2), deleted);
    // System prompt survives
    try std.testing.expectEqual(@as(usize, 1), conv.len());
    if (conv.getItem(0)) |item| {
        const msg = item.asMessage() orelse return error.TestUnexpectedResult;
        try std.testing.expectEqual(MessageRole.system, msg.role);
    } else return error.TestUnexpectedResult;
}

test "compactTurns with all pinned turns deletes nothing" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    _ = try conv.appendSystemMessage("System.");
    const pin1 = try conv.appendUserMessage("Pinned 1");
    pin1.input_tokens = 100;
    pin1.pinned = true;
    const pin2 = try conv.appendUserMessage("Pinned 2");
    pin2.input_tokens = 100;
    pin2.pinned = true;

    const turns = try identifyTurns(allocator, conv);
    defer allocator.free(turns);

    const deleted = compactTurns(conv, turns, 50, 200);
    try std.testing.expectEqual(@as(usize, 0), deleted);
    try std.testing.expectEqual(@as(usize, 3), conv.len());
}

test "compactTurns no-op when under target" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    _ = try conv.appendSystemMessage("System.");
    _ = try conv.appendUserMessage("Hello");

    const turns = try identifyTurns(allocator, conv);
    defer allocator.free(turns);

    const deleted = compactTurns(conv, turns, 1000, 50);
    try std.testing.expectEqual(@as(usize, 0), deleted);
    try std.testing.expectEqual(@as(usize, 2), conv.len());
}

test "identifyTurns accumulates token counts" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    _ = try conv.appendSystemMessage("System.");
    const tok_user = try conv.appendUserMessage("Hello");
    tok_user.input_tokens = 50;
    const tok_asst = try conv.appendAssistantMessage();
    tok_asst.output_tokens = 150;

    const turns = try identifyTurns(allocator, conv);
    defer allocator.free(turns);

    try std.testing.expectEqual(@as(usize, 1), turns.len);
    try std.testing.expectEqual(@as(u64, 200), turns[0].total_tokens);
}
