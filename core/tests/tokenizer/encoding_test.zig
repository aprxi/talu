//! Integration tests for tokenizer.Encoding
//!
//! Encoding is a combined result struct holding token IDs, byte offsets,
//! attention mask, and special tokens mask. Produced by encode().
//! Caller owns all slices; call deinit() to free.

const std = @import("std");
const main = @import("main");

const Encoding = main.tokenizer.Encoding;
const TokenOffset = main.tokenizer.TokenOffset;

// =============================================================================
// deinit Tests
// =============================================================================

test "Encoding.deinit frees all allocated slices" {
    const allocator = std.testing.allocator;

    const ids = try allocator.alloc(u32, 3);
    const offsets = try allocator.alloc(TokenOffset, 3);
    const attention_mask = try allocator.alloc(u32, 3);
    const special_tokens_mask = try allocator.alloc(u32, 3);

    ids[0] = 10;
    ids[1] = 20;
    ids[2] = 30;
    offsets[0] = .{ .start = 0, .end = 3 };
    offsets[1] = .{ .start = 3, .end = 7 };
    offsets[2] = .{ .start = 7, .end = 10 };
    attention_mask[0] = 1;
    attention_mask[1] = 1;
    attention_mask[2] = 1;
    special_tokens_mask[0] = 1;
    special_tokens_mask[1] = 0;
    special_tokens_mask[2] = 1;

    var enc = Encoding{
        .ids = ids,
        .offsets = offsets,
        .attention_mask = attention_mask,
        .special_tokens_mask = special_tokens_mask,
        .allocator = allocator,
    };
    enc.deinit();
    // std.testing.allocator detects leaks; reaching here without error means all freed.
}

test "Encoding.deinit is safe on empty slices" {
    const allocator = std.testing.allocator;
    var enc = Encoding{
        .ids = &.{},
        .offsets = &.{},
        .attention_mask = &.{},
        .special_tokens_mask = &.{},
        .allocator = allocator,
    };
    enc.deinit();
}
