//! Integration tests for FenceTracker.
//!
//! FenceTracker is a byte-wise state machine for markdown fenced code blocks.

const std = @import("std");
const main = @import("main");
const code = main.validate.code;
const FenceTracker = code.FenceTracker;
const FenceState = code.FenceState;
const CodeBlock = code.CodeBlock;

fn feedAll(tracker: *FenceTracker, text: []const u8) ?CodeBlock {
    var completed: ?CodeBlock = null;
    for (text, 0..) |byte, pos| {
        if (tracker.feed(byte, @intCast(pos))) |block| {
            completed = block;
        }
    }
    return completed;
}

test "FenceTracker init creates default state" {
    const tracker = FenceTracker.init();
    try std.testing.expectEqual(FenceState.text, tracker.state);
}

test "FenceTracker.feed detects complete fenced block" {
    const text = "```python\nprint('hi')\n```\n";
    var tracker = FenceTracker.init();

    const completed = feedAll(&tracker, text);
    try std.testing.expect(completed != null);
    try std.testing.expect(completed.?.complete);
    try std.testing.expectEqualStrings("python", completed.?.getLanguage(text));
    try std.testing.expectEqual(FenceState.line_start, tracker.state);
}

test "FenceTracker.feed detects complete block without language" {
    const text = "```\ncode\n```\n";
    var tracker = FenceTracker.init();

    const completed = feedAll(&tracker, text);
    try std.testing.expect(completed != null);
    try std.testing.expect(completed.?.complete);
    try std.testing.expectEqualStrings("", completed.?.getLanguage(text));
}

test "FenceTracker.finalize returns incomplete block when unclosed" {
    const text = "```python\nprint('hi')";
    var tracker = FenceTracker.init();

    _ = feedAll(&tracker, text);

    const incomplete = tracker.finalize(@intCast(text.len));
    if (incomplete) |block| {
        try std.testing.expect(!block.complete);
    }
}
