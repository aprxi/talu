//! Integration tests for FenceTracker.
//!
//! Tests the FenceTracker type exported from core/src/validate/code/root.zig.
//! FenceTracker is a state machine for parsing markdown code fences.

const std = @import("std");
const main = @import("main");
const code = main.validate.code;
const FenceTracker = code.FenceTracker;
const FenceState = code.FenceState;

// ============================================================================
// Lifecycle Tests
// ============================================================================

test "FenceTracker init creates default state" {
    var tracker = FenceTracker.init();
    try std.testing.expectEqual(FenceState.normal, tracker.state);
}

// ============================================================================
// Feed Tests - Normal Text
// ============================================================================

test "FenceTracker.feed normal text stays in normal state" {
    var tracker = FenceTracker.init();
    tracker.feed("Hello world");
    try std.testing.expectEqual(FenceState.normal, tracker.state);
}

test "FenceTracker.feed empty string stays in normal state" {
    var tracker = FenceTracker.init();
    tracker.feed("");
    try std.testing.expectEqual(FenceState.normal, tracker.state);
}

// ============================================================================
// Feed Tests - Opening Fence
// ============================================================================

test "FenceTracker.feed backtick fence opens code block" {
    var tracker = FenceTracker.init();
    tracker.feed("```python\n");
    try std.testing.expectEqual(FenceState.in_code, tracker.state);
}

test "FenceTracker.feed tilde fence opens code block" {
    var tracker = FenceTracker.init();
    tracker.feed("~~~rust\n");
    try std.testing.expectEqual(FenceState.in_code, tracker.state);
}

test "FenceTracker.feed fence without language opens code block" {
    var tracker = FenceTracker.init();
    tracker.feed("```\n");
    try std.testing.expectEqual(FenceState.in_code, tracker.state);
}

// ============================================================================
// Feed Tests - Closing Fence
// ============================================================================

test "FenceTracker.feed closing fence returns to normal" {
    var tracker = FenceTracker.init();
    tracker.feed("```python\ncode\n```\n");
    try std.testing.expectEqual(FenceState.normal, tracker.state);
}

test "FenceTracker.feed mismatched fence type does not close" {
    var tracker = FenceTracker.init();
    tracker.feed("```python\ncode\n~~~\n");
    // Backtick fence should not be closed by tilde fence
    try std.testing.expectEqual(FenceState.in_code, tracker.state);
}

// ============================================================================
// Feed Tests - Multiple Blocks
// ============================================================================

test "FenceTracker.feed handles multiple code blocks" {
    var tracker = FenceTracker.init();

    tracker.feed("Text before\n```python\ncode1\n```\nText between\n```rust\ncode2\n```\nText after");
    try std.testing.expectEqual(FenceState.normal, tracker.state);
}

// ============================================================================
// Feed Tests - Incremental Input
// ============================================================================

test "FenceTracker.feed handles incremental input" {
    var tracker = FenceTracker.init();

    tracker.feed("``");
    try std.testing.expectEqual(FenceState.normal, tracker.state);

    tracker.feed("`python\n");
    try std.testing.expectEqual(FenceState.in_code, tracker.state);

    tracker.feed("print('hi')\n");
    try std.testing.expectEqual(FenceState.in_code, tracker.state);

    tracker.feed("```\n");
    try std.testing.expectEqual(FenceState.normal, tracker.state);
}

// ============================================================================
// Finalize Tests
// ============================================================================

test "FenceTracker.finalize returns blocks" {
    var tracker = FenceTracker.init();
    tracker.feed("```python\ncode\n```\n");

    var list = tracker.finalize(std.testing.allocator);
    defer list.deinit();

    try std.testing.expectEqual(@as(usize, 1), list.count());
}

test "FenceTracker.finalize handles unclosed block" {
    var tracker = FenceTracker.init();
    tracker.feed("```python\nunclosed code");

    var list = tracker.finalize(std.testing.allocator);
    defer list.deinit();

    // Unclosed block should still be captured
    try std.testing.expectEqual(@as(usize, 1), list.count());
    const block = list.get(0).?;
    try std.testing.expect(!block.complete);
}

test "FenceTracker.finalize empty input returns empty list" {
    var tracker = FenceTracker.init();
    tracker.feed("");

    var list = tracker.finalize(std.testing.allocator);
    defer list.deinit();

    try std.testing.expectEqual(@as(usize, 0), list.count());
}
