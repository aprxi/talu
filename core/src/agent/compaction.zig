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
const ContentPart = responses.ContentPart;
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
// Intra-turn truncation fallback
// =============================================================================

const truncation_marker = "\n...(content truncated by compaction)";

/// Truncate the text in an ArrayListUnmanaged(u8) buffer.
/// Removes at least `bytes_to_remove` bytes (or the marker length, whichever
/// is larger) and appends the truncation marker.
/// Returns the new text length, or null if the buffer was empty.
fn truncateBuffer(buf: *std.ArrayListUnmanaged(u8), bytes_to_remove: usize) ?usize {
    const text_len = buf.items.len;
    if (text_len == 0) return null;

    // Ensure we remove at least enough space for the marker itself.
    const min_remove = @max(bytes_to_remove, truncation_marker.len);
    const keep = if (min_remove >= text_len) 0 else text_len - min_remove;

    buf.shrinkRetainingCapacity(keep);
    buf.appendSliceAssumeCapacity(truncation_marker);

    return buf.items.len;
}

/// Try to truncate the first text buffer found in a ContentPart.
fn truncateContentPart(part: *ContentPart, bytes_to_remove: usize) ?usize {
    switch (part.variant) {
        .input_text => |*v| return truncateBuffer(&v.text, bytes_to_remove),
        .output_text => |*v| return truncateBuffer(&v.text, bytes_to_remove),
        .text => |*v| return truncateBuffer(&v.text, bytes_to_remove),
        else => return null,
    }
}

/// Try to truncate the first text buffer found in an item's payload.
fn truncateItemText(item: *Item, bytes_to_remove: usize) ?usize {
    switch (item.data) {
        .message => |*msg| {
            for (msg.content.items) |*part| {
                if (truncateContentPart(part, bytes_to_remove)) |len| return len;
            }
        },
        .function_call_output => |*fco| {
            switch (fco.output) {
                .text => |*buf| return truncateBuffer(buf, bytes_to_remove),
                .parts => |*parts| {
                    for (parts.items) |*part| {
                        if (truncateContentPart(part, bytes_to_remove)) |len| return len;
                    }
                },
            }
        },
        else => {},
    }
    return null;
}

/// Fallback: truncate text content of the largest non-pinned item when
/// turn-level compaction can't reach the target.
///
/// Estimates bytes-to-tokens at 4:1. Updates item.input_tokens or
/// item.output_tokens to the estimated value after truncation.
/// Appends a truncation marker to the content.
///
/// Returns the estimated remaining tokens after truncation.
pub fn truncateOversizedItem(
    conv: *Conversation,
    target_tokens: u64,
    current_tokens: u64,
) u64 {
    if (current_tokens <= target_tokens) return current_tokens;

    const count = conv.len();
    if (count <= 1) return current_tokens; // only system prompt

    // Find the non-pinned item (index > 0) with the highest token count.
    var best_idx: ?usize = null;
    var best_tokens: u64 = 0;

    var i: usize = 1;
    while (i < count) : (i += 1) {
        const item = conv.getItem(i) orelse continue;
        if (item.pinned) continue;
        const tokens = itemTokens(item);
        if (tokens > best_tokens) {
            best_tokens = tokens;
            best_idx = i;
        }
    }

    const idx = best_idx orelse return current_tokens; // all pinned
    const item = conv.getItemMut(idx) orelse return current_tokens;

    const excess = current_tokens - target_tokens;
    const bytes_to_remove: usize = @intCast(excess * 4);

    const new_text_len = truncateItemText(item, bytes_to_remove) orelse
        return current_tokens; // no text to truncate

    // Update token estimate (4 bytes per token).
    const estimated_tokens: u32 = @intCast(new_text_len / 4);
    if (item.input_tokens > 0) {
        item.input_tokens = estimated_tokens;
    } else {
        item.output_tokens = estimated_tokens;
    }

    const new_item_tokens: u64 = @as(u64, item.input_tokens) + @as(u64, item.output_tokens);
    return current_tokens -| (best_tokens -| new_item_tokens);
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

test "truncateOversizedItem truncates massive user message" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    _ = try conv.appendSystemMessage("System.");
    // Create a message with large text content and high token count.
    const big_msg = try conv.appendUserMessage("x" ** 2000);
    big_msg.input_tokens = 500;

    // current=500, target=100 → excess=400 → bytes_to_remove=1600
    const remaining = truncateOversizedItem(conv, 100, 500);

    // Item should now have truncated text with marker appended.
    const item = conv.getItem(1).?;
    const msg = item.asMessage().?;
    const text = msg.getFirstText();
    try std.testing.expect(text.len < 2000);
    try std.testing.expect(std.mem.endsWith(u8, text, truncation_marker));
    // Token estimate should be reduced.
    try std.testing.expect(item.input_tokens < 500);
    try std.testing.expect(remaining < 500);
}

test "truncateOversizedItem skips pinned items" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    _ = try conv.appendSystemMessage("System.");
    const pinned = try conv.appendUserMessage("x" ** 2000);
    pinned.input_tokens = 500;
    pinned.pinned = true;

    // Only item is pinned — nothing to truncate.
    const remaining = truncateOversizedItem(conv, 100, 500);
    try std.testing.expectEqual(@as(u64, 500), remaining);

    // Text unchanged.
    const item = conv.getItem(1).?;
    const msg = item.asMessage().?;
    try std.testing.expectEqual(@as(usize, 2000), msg.getFirstText().len);
}

test "truncateOversizedItem no-op when under target" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    _ = try conv.appendSystemMessage("System.");
    const msg = try conv.appendUserMessage("Hello");
    msg.input_tokens = 10;

    const remaining = truncateOversizedItem(conv, 100, 50);
    try std.testing.expectEqual(@as(u64, 50), remaining);
}

test "truncateOversizedItem truncates function_call_output" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    _ = try conv.appendSystemMessage("System.");
    const fco = try conv.appendFunctionCallOutput("call_1", "y" ** 2000);
    fco.input_tokens = 500;

    const remaining = truncateOversizedItem(conv, 100, 500);

    const item = conv.getItem(1).?;
    const output = item.asFunctionCallOutput().?;
    const text = output.getOutputText();
    try std.testing.expect(text.len < 2000);
    try std.testing.expect(std.mem.endsWith(u8, text, truncation_marker));
    try std.testing.expect(item.input_tokens < 500);
    try std.testing.expect(remaining < 500);
}

test "truncateOversizedItem picks highest-token item" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    _ = try conv.appendSystemMessage("System.");
    const small = try conv.appendUserMessage("small");
    small.input_tokens = 50;
    const big = try conv.appendUserMessage("z" ** 2000);
    big.input_tokens = 400;

    // current=450, target=200 → should truncate the big item (400 tokens)
    const remaining = truncateOversizedItem(conv, 200, 450);

    // Small item unchanged.
    const item1 = conv.getItem(1).?;
    try std.testing.expectEqual(@as(u32, 50), item1.input_tokens);
    // Big item truncated.
    const item2 = conv.getItem(2).?;
    try std.testing.expect(item2.input_tokens < 400);
    try std.testing.expect(remaining < 450);
}
