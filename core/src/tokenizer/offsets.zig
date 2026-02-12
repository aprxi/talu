//! Token Offset Computation
//!
//! Utilities for computing byte offsets that map tokens back to source text.
//! Used by the C API for token-to-text alignment.

const std = @import("std");
const ct = @import("c_types.zig");
const tok_encode = @import("encode.zig");

/// Token offset in source text (UTF-8 byte indices).
pub const TokenOffset = extern struct {
    start: u32,
    end: u32,
};

/// Result of offset computation.
pub const OffsetsResult = struct {
    offsets: []TokenOffset,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *OffsetsResult) void {
        if (self.offsets.len > 0) {
            self.allocator.free(self.offsets);
        }
        self.* = undefined;
    }
};

/// Compute byte offsets for tokens in the source text.
/// Returns offsets mapping each token to its position in the original text.
pub fn computeOffsets(
    allocator: std.mem.Allocator,
    tokenizer_handle: *ct.Tokenizer,
    text: []const u8,
) !OffsetsResult {
    var token_encoding = std.mem.zeroes(ct.TokenizerEncoding);
    if (tok_encode.tokenizer_encode_struct(tokenizer_handle, text, &token_encoding) != 0) {
        return error.TokenizationFailed;
    }
    defer tok_encode.tokenizer_encoding_free_struct(&token_encoding);

    const token_count = token_encoding.ids_len;
    if (token_count == 0) {
        return OffsetsResult{
            .offsets = &.{},
            .allocator = allocator,
        };
    }

    const offsets = try allocator.alloc(TokenOffset, token_count);
    errdefer allocator.free(offsets);

    var text_offset: u32 = 0;

    for (0..token_count) |token_idx| {
        const token_cstrs: [*][*c]u8 = if (token_encoding.tokens) |toks|
            @ptrCast(toks)
        else {
            offsets[token_idx] = .{ .start = 0, .end = 0 };
            continue;
        };

        const token_ptr: [*c]u8 = token_cstrs[token_idx];
        if (token_ptr == null) {
            offsets[token_idx] = .{ .start = 0, .end = 0 };
            continue;
        }

        const token_text = std.mem.sliceTo(token_ptr, 0);
        const token_byte_sequence = decodeTokenToBytes(allocator, token_text, tokenizer_handle) catch {
            offsets[token_idx] = .{ .start = 0, .end = 0 };
            continue;
        };
        defer if (token_byte_sequence.ptr != token_text.ptr) allocator.free(token_byte_sequence);

        const remaining_text = text[text_offset..];
        if (findSubsequence(remaining_text, token_byte_sequence)) |match_offset| {
            const start_offset = text_offset + @as(u32, @intCast(match_offset));
            const end_offset = start_offset + @as(u32, @intCast(token_byte_sequence.len));
            offsets[token_idx] = .{ .start = start_offset, .end = end_offset };
            text_offset = end_offset;
        } else {
            offsets[token_idx] = .{ .start = 0, .end = 0 };
        }
    }

    return OffsetsResult{
        .offsets = offsets,
        .allocator = allocator,
    };
}

/// Decode a token string to its raw byte representation.
/// Handles byte-level tokenization and sentencepiece underscore character.
pub fn decodeTokenToBytes(allocator: std.mem.Allocator, token_text: []const u8, tokenizer_handle: *ct.Tokenizer) ![]const u8 {
    const is_byte_level = tokenizer_handle.pretokenizer.byte_level != 0;

    if (!is_byte_level) {
        // SentencePiece: replace leading underscore (▁ = U+2581) with space
        if (std.mem.startsWith(u8, token_text, "\xE2\x96\x81")) {
            const decoded_bytes = try allocator.alloc(u8, token_text.len - 2);
            decoded_bytes[0] = ' ';
            @memcpy(decoded_bytes[1..], token_text[3..]);
            return decoded_bytes;
        }
        return token_text;
    }

    // Byte-level: decode unicode codepoints to raw bytes
    var decoded_bytes = std.ArrayListUnmanaged(u8){};
    errdefer decoded_bytes.deinit(allocator);

    var byte_idx: usize = 0;
    while (byte_idx < token_text.len) {
        const char_len = utf8CharLen(token_text[byte_idx]);
        if (byte_idx + char_len > token_text.len) break;

        const codepoint = utf8Decode(token_text[byte_idx..][0..char_len]);
        const raw_byte = unicodeToRawByte(codepoint);
        try decoded_bytes.append(allocator, raw_byte);
        byte_idx += char_len;
    }

    return try decoded_bytes.toOwnedSlice(allocator);
}

/// Map a unicode codepoint to its raw byte value.
/// Handles the GPT-2 byte-level BPE encoding where bytes 0-255 are mapped
/// to unicode codepoints, with some gaps filled by codepoints >= 256.
pub fn unicodeToRawByte(codepoint: u21) u8 {
    if (codepoint < 256) {
        return @intCast(codepoint);
    }
    // Gap bytes: bytes that were mapped to codepoints >= 256 in GPT-2 encoding
    const gap_index = codepoint - 256;
    const gap_bytes = [_]u8{
        0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,
        16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
        32,  127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
        142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
        158, 159, 160, 173,
    };
    if (gap_index < gap_bytes.len) {
        return gap_bytes[gap_index];
    }
    return 0;
}

/// Decode a UTF-8 byte sequence to a unicode codepoint.
pub fn utf8Decode(utf8_bytes: []const u8) u21 {
    if (utf8_bytes.len == 0) return 0;
    const first_byte = utf8_bytes[0];
    if (first_byte < 0x80) return first_byte;
    if (first_byte < 0xE0) {
        if (utf8_bytes.len < 2) return 0;
        return (@as(u21, first_byte & 0x1F) << 6) | (utf8_bytes[1] & 0x3F);
    }
    if (first_byte < 0xF0) {
        if (utf8_bytes.len < 3) return 0;
        return (@as(u21, first_byte & 0x0F) << 12) | (@as(u21, utf8_bytes[1] & 0x3F) << 6) | (utf8_bytes[2] & 0x3F);
    }
    if (utf8_bytes.len < 4) return 0;
    return (@as(u21, first_byte & 0x07) << 18) | (@as(u21, utf8_bytes[1] & 0x3F) << 12) | (@as(u21, utf8_bytes[2] & 0x3F) << 6) | (utf8_bytes[3] & 0x3F);
}

/// Get the length of a UTF-8 character from its first byte.
pub fn utf8CharLen(first_byte: u8) usize {
    if (first_byte < 0x80) return 1;
    if (first_byte < 0xE0) return 2;
    if (first_byte < 0xF0) return 3;
    return 4;
}

/// Find a subsequence in a source slice. Returns the offset if found, null otherwise.
pub fn findSubsequence(source: []const u8, pattern: []const u8) ?usize {
    if (pattern.len == 0) return 0;
    if (pattern.len > source.len) return null;

    var index: usize = 0;
    while (index <= source.len - pattern.len) : (index += 1) {
        if (std.mem.eql(u8, source[index..][0..pattern.len], pattern)) {
            return index;
        }
    }
    return null;
}

// =============================================================================
// Tests
// =============================================================================

test "utf8CharLen returns correct lengths" {
    // ASCII
    try std.testing.expectEqual(@as(usize, 1), utf8CharLen('A'));
    try std.testing.expectEqual(@as(usize, 1), utf8CharLen(0x7F));

    // 2-byte sequences
    try std.testing.expectEqual(@as(usize, 2), utf8CharLen(0xC2));
    try std.testing.expectEqual(@as(usize, 2), utf8CharLen(0xDF));

    // 3-byte sequences
    try std.testing.expectEqual(@as(usize, 3), utf8CharLen(0xE0));
    try std.testing.expectEqual(@as(usize, 3), utf8CharLen(0xEF));

    // 4-byte sequences
    try std.testing.expectEqual(@as(usize, 4), utf8CharLen(0xF0));
    try std.testing.expectEqual(@as(usize, 4), utf8CharLen(0xF4));
}

test "utf8Decode decodes ASCII" {
    try std.testing.expectEqual(@as(u21, 'A'), utf8Decode("A"));
    try std.testing.expectEqual(@as(u21, ' '), utf8Decode(" "));
    try std.testing.expectEqual(@as(u21, 0x7F), utf8Decode("\x7F"));
}

test "utf8Decode decodes 2-byte sequences" {
    // U+00E9 (e with acute) = 0xC3 0xA9
    try std.testing.expectEqual(@as(u21, 0xE9), utf8Decode("\xC3\xA9"));
}

test "utf8Decode decodes 3-byte sequences" {
    // U+2581 (lower one eighth block) = 0xE2 0x96 0x81
    try std.testing.expectEqual(@as(u21, 0x2581), utf8Decode("\xE2\x96\x81"));
}

test "utf8Decode handles empty input" {
    try std.testing.expectEqual(@as(u21, 0), utf8Decode(""));
}

test "utf8Decode handles truncated sequences" {
    // 2-byte sequence with only 1 byte
    try std.testing.expectEqual(@as(u21, 0), utf8Decode("\xC3"));
    // 3-byte sequence with only 2 bytes
    try std.testing.expectEqual(@as(u21, 0), utf8Decode("\xE2\x96"));
}

test "unicodeToRawByte maps ASCII correctly" {
    try std.testing.expectEqual(@as(u8, 'A'), unicodeToRawByte('A'));
    try std.testing.expectEqual(@as(u8, 255), unicodeToRawByte(255));
}

test "unicodeToRawByte maps gap bytes" {
    // Codepoint 256 should map to byte 0
    try std.testing.expectEqual(@as(u8, 0), unicodeToRawByte(256));
    // Codepoint 257 should map to byte 1
    try std.testing.expectEqual(@as(u8, 1), unicodeToRawByte(257));
}

test "unicodeToRawByte returns 0 for out-of-range codepoints" {
    try std.testing.expectEqual(@as(u8, 0), unicodeToRawByte(10000));
}

test "findSubsequence finds pattern at start" {
    const result = findSubsequence("hello world", "hello");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(usize, 0), result.?);
}

test "findSubsequence finds pattern in middle" {
    const result = findSubsequence("hello world", "world");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(usize, 6), result.?);
}

test "findSubsequence returns null for missing pattern" {
    const result = findSubsequence("hello world", "xyz");
    try std.testing.expect(result == null);
}

test "findSubsequence handles empty pattern" {
    const result = findSubsequence("hello", "");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(usize, 0), result.?);
}

test "findSubsequence handles pattern longer than source" {
    const result = findSubsequence("hi", "hello");
    try std.testing.expect(result == null);
}

test "findSubsequence finds single character" {
    const result = findSubsequence("hello", "l");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(usize, 2), result.?);
}
