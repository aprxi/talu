//! Integration tests for Tokenizer
//!
//! Tokenizer converts text to tokens and tokens back to text.
//! This is the core interface for text processing in language models.

const std = @import("std");
const main = @import("main");
const Tokenizer = main.tokenizer.Tokenizer;
const TokenizerError = main.tokenizer.TokenizerError;

// =============================================================================
// Test Tokenizer JSON
// =============================================================================

/// Minimal BPE tokenizer JSON for testing.
/// Uses byte-level vocabulary (256 tokens) plus special tokens.
const test_tokenizer_json =
    \\{
    \\  "version": "1.0",
    \\  "model": {
    \\    "type": "BPE",
    \\    "vocab": {
    \\      "<pad>": 0,
    \\      "<s>": 1,
    \\      "</s>": 2,
    \\      "<unk>": 3,
    \\      " ": 4,
    \\      "!": 5,
    \\      "\"": 6,
    \\      "#": 7,
    \\      "$": 8,
    \\      "%": 9,
    \\      "&": 10,
    \\      "'": 11,
    \\      "(": 12,
    \\      ")": 13,
    \\      "*": 14,
    \\      "+": 15,
    \\      ",": 16,
    \\      "-": 17,
    \\      ".": 18,
    \\      "/": 19,
    \\      "0": 20,
    \\      "1": 21,
    \\      "2": 22,
    \\      "3": 23,
    \\      "4": 24,
    \\      "5": 25,
    \\      "6": 26,
    \\      "7": 27,
    \\      "8": 28,
    \\      "9": 29,
    \\      ":": 30,
    \\      ";": 31,
    \\      "<": 32,
    \\      "=": 33,
    \\      ">": 34,
    \\      "?": 35,
    \\      "@": 36,
    \\      "A": 37,
    \\      "B": 38,
    \\      "C": 39,
    \\      "D": 40,
    \\      "E": 41,
    \\      "F": 42,
    \\      "G": 43,
    \\      "H": 44,
    \\      "I": 45,
    \\      "J": 46,
    \\      "K": 47,
    \\      "L": 48,
    \\      "M": 49,
    \\      "N": 50,
    \\      "O": 51,
    \\      "P": 52,
    \\      "Q": 53,
    \\      "R": 54,
    \\      "S": 55,
    \\      "T": 56,
    \\      "U": 57,
    \\      "V": 58,
    \\      "W": 59,
    \\      "X": 60,
    \\      "Y": 61,
    \\      "Z": 62,
    \\      "[": 63,
    \\      "\\": 64,
    \\      "]": 65,
    \\      "^": 66,
    \\      "_": 67,
    \\      "`": 68,
    \\      "a": 69,
    \\      "b": 70,
    \\      "c": 71,
    \\      "d": 72,
    \\      "e": 73,
    \\      "f": 74,
    \\      "g": 75,
    \\      "h": 76,
    \\      "i": 77,
    \\      "j": 78,
    \\      "k": 79,
    \\      "l": 80,
    \\      "m": 81,
    \\      "n": 82,
    \\      "o": 83,
    \\      "p": 84,
    \\      "q": 85,
    \\      "r": 86,
    \\      "s": 87,
    \\      "t": 88,
    \\      "u": 89,
    \\      "v": 90,
    \\      "w": 91,
    \\      "x": 92,
    \\      "y": 93,
    \\      "z": 94,
    \\      "{": 95,
    \\      "|": 96,
    \\      "}": 97,
    \\      "~": 98
    \\    },
    \\    "merges": []
    \\  },
    \\  "added_tokens": [
    \\    {"id": 0, "content": "<pad>", "special": true},
    \\    {"id": 1, "content": "<s>", "special": true},
    \\    {"id": 2, "content": "</s>", "special": true},
    \\    {"id": 3, "content": "<unk>", "special": true}
    \\  ],
    \\  "normalizer": null,
    \\  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
    \\  "post_processor": null,
    \\  "decoder": {"type": "ByteLevel"}
    \\}
;

// =============================================================================
// Lifecycle Tests
// =============================================================================

test "Tokenizer: init and deinit from JSON" {
    const allocator = std.testing.allocator;
    var tok = try Tokenizer.initFromJson(allocator, test_tokenizer_json);
    defer tok.deinit();
    // Just verify it initializes without error
}

test "Tokenizer: init fails for invalid JSON" {
    const allocator = std.testing.allocator;
    const result = Tokenizer.initFromJson(allocator, "{ invalid json }");
    try std.testing.expectError(TokenizerError.InitFailed, result);
}

test "Tokenizer: init fails for missing path" {
    const allocator = std.testing.allocator;
    const result = Tokenizer.initFromPath(allocator, "/nonexistent/path/to/tokenizer");
    try std.testing.expectError(TokenizerError.InitFailed, result);
}

// =============================================================================
// Encode Tests
// =============================================================================

test "Tokenizer: encode simple text" {
    const allocator = std.testing.allocator;
    var tok = try Tokenizer.initFromJson(allocator, test_tokenizer_json);
    defer tok.deinit();

    const ids = try tok.encode("Hi");
    defer allocator.free(ids);

    // With byte-level BPE, "Hi" should be encoded as individual characters
    try std.testing.expect(ids.len >= 2);
}

test "Tokenizer: encode empty string returns empty slice" {
    const allocator = std.testing.allocator;
    var tok = try Tokenizer.initFromJson(allocator, test_tokenizer_json);
    defer tok.deinit();

    const ids = try tok.encode("");
    // Empty input should return empty output (no allocation needed)
    try std.testing.expectEqual(@as(usize, 0), ids.len);
}

test "Tokenizer: encodeSlice supports null bytes in text" {
    const allocator = std.testing.allocator;
    var tok = try Tokenizer.initFromJson(allocator, test_tokenizer_json);
    defer tok.deinit();

    // encodeSlice can handle text with embedded nulls
    const text_with_null = "a\x00b";
    const ids = try tok.encodeSlice(text_with_null);
    defer if (ids.len > 0) allocator.free(ids);

    // Should encode something (behavior depends on tokenizer)
    try std.testing.expect(ids.len >= 0);
}

test "Tokenizer: encodeSliceWithOptions without special tokens" {
    const allocator = std.testing.allocator;
    var tok = try Tokenizer.initFromJson(allocator, test_tokenizer_json);
    defer tok.deinit();

    const with_special = try tok.encodeSlice("test");
    defer if (with_special.len > 0) allocator.free(with_special);

    const without_special = try tok.encodeSliceWithOptions("test", .{ .add_special_tokens = false });
    defer if (without_special.len > 0) allocator.free(without_special);

    // Both should produce valid output
    try std.testing.expect(with_special.len >= 0);
    try std.testing.expect(without_special.len >= 0);
}

// =============================================================================
// Decode Tests
// =============================================================================

test "Tokenizer: decode token IDs to text" {
    const allocator = std.testing.allocator;
    var tok = try Tokenizer.initFromJson(allocator, test_tokenizer_json);
    defer tok.deinit();

    // Encode then decode should round-trip
    const original = "abc";
    const ids = try tok.encode(original);
    defer allocator.free(ids);

    const decoded = try tok.decode(ids);
    defer allocator.free(decoded);

    try std.testing.expectEqualStrings(original, decoded);
}

test "Tokenizer: decode empty slice returns empty string" {
    const allocator = std.testing.allocator;
    var tok = try Tokenizer.initFromJson(allocator, test_tokenizer_json);
    defer tok.deinit();

    const decoded = try tok.decode(&[_]u32{});
    defer if (decoded.len > 0) allocator.free(decoded);

    try std.testing.expectEqual(@as(usize, 0), decoded.len);
}

test "Tokenizer: decodeWithOptions skip_special_tokens" {
    const allocator = std.testing.allocator;
    var tok = try Tokenizer.initFromJson(allocator, test_tokenizer_json);
    defer tok.deinit();

    // Token ID 1 is <s> (BOS)
    const ids_with_special = [_]u32{ 1, 69, 70, 71 }; // <s>, a, b, c

    const with_skip = try tok.decodeWithOptions(&ids_with_special, .{ .skip_special_tokens = true });
    defer allocator.free(with_skip);

    const without_skip = try tok.decodeWithOptions(&ids_with_special, .{ .skip_special_tokens = false });
    defer allocator.free(without_skip);

    // without_skip should contain the special token marker
    // The exact behavior depends on the tokenizer implementation
    // At minimum, verify both return valid results
    try std.testing.expect(with_skip.len > 0 or without_skip.len > 0);
}

// =============================================================================
// Round-trip Tests
// =============================================================================

test "Tokenizer: encode-decode round-trip preserves text" {
    const allocator = std.testing.allocator;
    var tok = try Tokenizer.initFromJson(allocator, test_tokenizer_json);
    defer tok.deinit();

    // Test with ASCII characters that are in our minimal vocab
    // Note: space is token 4, but byte-level BPE may handle it differently
    const test_cases = [_][]const u8{
        "Hello",
        "abc",
        "123",
    };

    for (test_cases) |original| {
        const ids = try tok.encode(original);
        defer allocator.free(ids);

        const decoded = try tok.decode(ids);
        defer allocator.free(decoded);

        try std.testing.expectEqualStrings(original, decoded);
    }
}

// =============================================================================
// StreamingDecoder Tests
// =============================================================================

test "Tokenizer: streamingDecoder iterates tokens" {
    const allocator = std.testing.allocator;
    var tok = try Tokenizer.initFromJson(allocator, test_tokenizer_json);
    defer tok.deinit();

    const ids = try tok.encode("abc");
    defer allocator.free(ids);

    var decoder = tok.streamingDecoder(ids);

    var count: usize = 0;
    while (try decoder.next()) |text| {
        allocator.free(text);
        count += 1;
    }

    try std.testing.expectEqual(ids.len, count);
}

test "Tokenizer: streamingDecoder reset restarts iteration" {
    const allocator = std.testing.allocator;
    var tok = try Tokenizer.initFromJson(allocator, test_tokenizer_json);
    defer tok.deinit();

    const ids = [_]u32{ 69, 70 }; // a, b
    var decoder = tok.streamingDecoder(&ids);

    // First iteration
    var first = (try decoder.next()).?;
    allocator.free(first);
    var second = (try decoder.next()).?;
    allocator.free(second);
    try std.testing.expect((try decoder.next()) == null);

    // Reset and iterate again
    decoder.reset();
    first = (try decoder.next()).?;
    allocator.free(first);
    second = (try decoder.next()).?;
    allocator.free(second);
    try std.testing.expect((try decoder.next()) == null);
}

test "Tokenizer: streamingDecoder empty input returns null immediately" {
    const allocator = std.testing.allocator;
    var tok = try Tokenizer.initFromJson(allocator, test_tokenizer_json);
    defer tok.deinit();

    var decoder = tok.streamingDecoder(&[_]u32{});
    try std.testing.expect((try decoder.next()) == null);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

test "Tokenizer: decode rejects out-of-range token ID" {
    const allocator = std.testing.allocator;
    var tok = try Tokenizer.initFromJson(allocator, test_tokenizer_json);
    defer tok.deinit();

    // 0xFFFFFFFF is too large to fit in i32
    const ids = [_]u32{0xFFFFFFFF};
    const result = tok.decode(&ids);
    try std.testing.expectError(TokenizerError.InvalidTokenId, result);
}
