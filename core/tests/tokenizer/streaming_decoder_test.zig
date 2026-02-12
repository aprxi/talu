//! Integration tests for StreamingDecoder
//!
//! StreamingDecoder provides token-by-token text decoding for streaming output.
//! It wraps a Tokenizer and iterates through token IDs one at a time.

const std = @import("std");
const main = @import("main");
const StreamingDecoder = main.tokenizer.StreamingDecoder;
const Tokenizer = main.tokenizer.Tokenizer;

// =============================================================================
// StreamingDecoder Tests
// =============================================================================

test "StreamingDecoder iterates through tokens" {
    // StreamingDecoder requires a Tokenizer which requires loading tokenizer.json
    // This is an integration test that needs a real tokenizer file
    // Skip if no tokenizer available
    var tokenizer = Tokenizer.initFromPath(std.testing.allocator, "tests/fixtures/tokenizer.json") catch |err| {
        if (err == error.InitFailed) return; // No fixture available
        return err;
    };
    defer tokenizer.deinit();

    // Encode some text
    const ids = try tokenizer.encode("Hello world");
    defer std.testing.allocator.free(ids);

    if (ids.len == 0) return; // Skip if empty

    // Create streaming decoder
    var decoder = tokenizer.streamingDecoder(ids);

    // Should be able to iterate
    var count: usize = 0;
    while (try decoder.next()) |text| {
        defer std.testing.allocator.free(text);
        count += 1;
    }

    try std.testing.expectEqual(ids.len, count);
}

test "StreamingDecoder next returns null when exhausted" {
    var tokenizer = Tokenizer.initFromPath(std.testing.allocator, "tests/fixtures/tokenizer.json") catch |err| {
        if (err == error.InitFailed) return;
        return err;
    };
    defer tokenizer.deinit();

    // Empty token list
    const empty_ids: []const u32 = &.{};
    var decoder = tokenizer.streamingDecoder(empty_ids);

    // Should immediately return null
    const result = try decoder.next();
    try std.testing.expect(result == null);
}

test "StreamingDecoder reset allows re-iteration" {
    var tokenizer = Tokenizer.initFromPath(std.testing.allocator, "tests/fixtures/tokenizer.json") catch |err| {
        if (err == error.InitFailed) return;
        return err;
    };
    defer tokenizer.deinit();

    const ids = try tokenizer.encode("test");
    defer std.testing.allocator.free(ids);

    if (ids.len == 0) return;

    var decoder = tokenizer.streamingDecoder(ids);

    // Consume all tokens
    while (try decoder.next()) |text| {
        std.testing.allocator.free(text);
    }

    // Reset
    decoder.reset();

    // Should be able to iterate again
    var count: usize = 0;
    while (try decoder.next()) |text| {
        defer std.testing.allocator.free(text);
        count += 1;
    }

    try std.testing.expectEqual(ids.len, count);
}
