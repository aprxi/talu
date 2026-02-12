//! Token Decoders
//!
//! Unified decoder implementation using tagged union dispatch.
//! Eliminates C-style vtable pattern for better optimization.

const std = @import("std");

/// Decoder type for dispatch
pub const DecoderType = enum {
    wordpiece,
    bpe,
    byte_level,
};

pub const DecodeError = error{
    OutOfMemory,
    /// byte_level decoder requires i32 ids, use decodeByteLevel instead
    ByteLevelRequiresIds,
};

/// Unified decode function with internal dispatch
pub fn decode(
    decoder_type: DecoderType,
    allocator: std.mem.Allocator,
    tokens: []const []const u8,
) DecodeError![]u8 {
    return switch (decoder_type) {
        .wordpiece => decodeWordPiece(allocator, tokens),
        .bpe => decodeBpe(allocator, tokens),
        .byte_level => error.ByteLevelRequiresIds,
    };
}

/// Decode WordPiece tokens (## prefix indicates subword)
fn decodeWordPiece(allocator: std.mem.Allocator, tokens: []const []const u8) ![]u8 {
    var output_bytes = std.ArrayListUnmanaged(u8){};
    errdefer output_bytes.deinit(allocator);

    for (tokens, 0..) |token_bytes, token_index| {
        const is_subword = token_bytes.len >= 2 and token_bytes[0] == '#' and token_bytes[1] == '#';
        if (!is_subword and token_index > 0) {
            try output_bytes.append(allocator, ' ');
        }
        const token_content = if (is_subword) token_bytes[2..] else token_bytes;
        try output_bytes.appendSlice(allocator, token_content);
    }

    return output_bytes.toOwnedSlice(allocator);
}

/// Decode BPE tokens (simple concatenation)
fn decodeBpe(allocator: std.mem.Allocator, tokens: []const []const u8) ![]u8 {
    var output_bytes = std.ArrayListUnmanaged(u8){};
    errdefer output_bytes.deinit(allocator);

    for (tokens) |token_bytes| {
        try output_bytes.appendSlice(allocator, token_bytes);
    }

    return output_bytes.toOwnedSlice(allocator);
}

/// Decode byte-level token IDs to bytes
fn decodeByteLevel(allocator: std.mem.Allocator, ids: []const i32) ![]u8 {
    const output_bytes = try allocator.alloc(u8, ids.len);
    for (output_bytes, ids) |*out_byte, token_id| {
        out_byte.* = @intCast(token_id & 0xFF);
    }
    return output_bytes;
}

// =============================================================================
// C API Exports
// =============================================================================

pub fn decoder_wordpiece(
    token_ptrs: [*]const [*:0]const u8,
    token_count: usize,
    out: *[*c]u8,
    out_len: *usize,
) c_int {
    out.* = null;
    const allocator = std.heap.c_allocator;

    // Convert C strings to slices
    var token_slices = allocator.alloc([]const u8, token_count) catch return -1;
    defer allocator.free(token_slices);
    for (token_ptrs[0..token_count], 0..) |token_ptr, token_index| {
        token_slices[token_index] = std.mem.sliceTo(token_ptr, 0);
    }

    const decoded = decodeWordPiece(allocator, token_slices) catch return -1;
    // Return actual length before null terminator
    out_len.* = decoded.len;
    // Add null terminator for C string convention
    const decoded_with_null = allocator.realloc(decoded, decoded.len + 1) catch {
        allocator.free(decoded);
        return -1;
    };
    decoded_with_null[decoded.len] = 0;
    out.* = decoded_with_null.ptr;
    return 0;
}

// =============================================================================
// Tests
// =============================================================================

test "decode dispatches to correct decoder" {
    const allocator = std.testing.allocator;

    // Test BPE decoder
    const bpe_tokens = [_][]const u8{ "hello", "world" };
    const bpe_result = try decode(.bpe, allocator, &bpe_tokens);
    defer allocator.free(bpe_result);
    try std.testing.expectEqualStrings("helloworld", bpe_result);

    // Test byte_level requires ids
    try std.testing.expectError(error.ByteLevelRequiresIds, decode(.byte_level, allocator, &bpe_tokens));
}

test "decodeWordPiece handles subword tokens" {
    const allocator = std.testing.allocator;

    // WordPiece: ## prefix means subword (no space before), strips prefix
    // "un" + "##break" + "##able" -> "unbreakable"
    const tokens = [_][]const u8{ "un", "##break", "##able" };
    const result = try decodeWordPiece(allocator, &tokens);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("unbreakable", result);
}

test "decodeWordPiece adds spaces between words" {
    const allocator = std.testing.allocator;

    const tokens = [_][]const u8{ "hello", "world" };
    const result = try decodeWordPiece(allocator, &tokens);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("hello world", result);
}

test "decodeBpe concatenates tokens" {
    const allocator = std.testing.allocator;

    const tokens = [_][]const u8{ "hello", " ", "world" };
    const result = try decodeBpe(allocator, &tokens);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("hello world", result);
}

test "decodeByteLevel converts ids to bytes" {
    const allocator = std.testing.allocator;

    const ids = [_]i32{ 72, 101, 108, 108, 111 }; // "Hello"
    const result = try decodeByteLevel(allocator, &ids);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello", result);
}

test "decoder_wordpiece returns valid output" {
    const tokens = [_][*:0]const u8{ "hello", "##world" };
    var out: [*c]u8 = null;
    var out_len: usize = 0;

    const result = decoder_wordpiece(&tokens, tokens.len, &out, &out_len);

    try std.testing.expectEqual(@as(c_int, 0), result);
    try std.testing.expect(out != null);
    defer std.heap.c_allocator.free(out[0 .. out_len + 1]); // +1 for null terminator

    const output_slice = out[0..out_len];
    try std.testing.expectEqualStrings("helloworld", output_slice);
}
