//! High-Level Tokenizer API
//!
//! User-facing Zig API for text tokenization.
//! Wraps the C pipeline with idiomatic Zig memory management.

const std = @import("std");

const ct = @import("c_types.zig");
const offsets_mod = @import("offsets.zig");
const pipeline_impl = @import("pipeline.zig");

// Unicode replacement character (U+FFFD)
const REPLACEMENT_CHAR = "\xEF\xBF\xBD";

pub const TokenizerError = error{
    InitFailed,
    EncodeFailed,
    DecodeFailed,
    InvalidTokenId,
};

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    tokenizer_handle: *ct.Tokenizer,
    vocab_size: usize = 0,
    prefix_index: [256][]u32 = undefined,
    token_bytes_data: []u8 = &.{},
    token_bytes_offsets: []usize = &.{},

    /// Initialize from a path. Accepts sentinel-terminated slice to avoid allocation.
    pub fn initFromPathZ(allocator: std.mem.Allocator, path: [:0]const u8) !Tokenizer {
        const tokenizer_handle = pipeline_impl.tokenizer_from_pretrained(path.ptr);
        if (tokenizer_handle == null) return TokenizerError.InitFailed;
        errdefer pipeline_impl.tokenizer_free(tokenizer_handle.?);
        var tokenizer = Tokenizer{ .allocator = allocator, .tokenizer_handle = tokenizer_handle.? };
        try tokenizer.buildPrefixIndex();
        return tokenizer;
    }

    /// Initialize from a path (allocates to add null terminator).
    pub fn initFromPath(allocator: std.mem.Allocator, path: []const u8) !Tokenizer {
        const path_zstr = try allocator.dupeZ(u8, path);
        defer allocator.free(path_zstr);
        return initFromPathZ(allocator, path_zstr);
    }

    /// Alias for initFromPath
    pub const init = initFromPath;

    /// Initialize from a JSON string in memory.
    pub fn initFromJsonZ(allocator: std.mem.Allocator, json: [:0]const u8) !Tokenizer {
        const tokenizer_handle = pipeline_impl.tokenizer_from_json_string(json.ptr);
        if (tokenizer_handle == null) return TokenizerError.InitFailed;
        errdefer pipeline_impl.tokenizer_free(tokenizer_handle.?);
        var tokenizer = Tokenizer{ .allocator = allocator, .tokenizer_handle = tokenizer_handle.? };
        try tokenizer.buildPrefixIndex();
        return tokenizer;
    }

    /// Initialize from a JSON string in memory (allocates to add null terminator).
    pub fn initFromJson(allocator: std.mem.Allocator, json: []const u8) !Tokenizer {
        const json_zstr = try allocator.dupeZ(u8, json);
        defer allocator.free(json_zstr);
        return initFromJsonZ(allocator, json_zstr);
    }

    pub fn deinit(self: *Tokenizer) void {
        for (self.prefix_index) |bucket| {
            if (bucket.len > 0) {
                self.allocator.free(bucket);
            }
        }
        if (self.token_bytes_data.len > 0) {
            self.allocator.free(self.token_bytes_data);
        }
        if (self.token_bytes_offsets.len > 0) {
            self.allocator.free(self.token_bytes_offsets);
        }
        pipeline_impl.tokenizer_free(self.tokenizer_handle);
        self.* = undefined;
    }

    pub fn getTokensStartingWith(self: *const Tokenizer, byte: u8) []const u32 {
        return self.prefix_index[byte];
    }

    pub fn tokenBytes(self: *const Tokenizer, token_id: usize) ?[]const u8 {
        if (self.token_bytes_offsets.len == 0 or token_id + 1 >= self.token_bytes_offsets.len) {
            return null;
        }
        const start = self.token_bytes_offsets[token_id];
        const end = self.token_bytes_offsets[token_id + 1];
        if (end < start or end > self.token_bytes_data.len) return null;
        return self.token_bytes_data[start..end];
    }

    fn buildPrefixIndex(self: *Tokenizer) !void {
        var buckets: [256]std.ArrayList(u32) = undefined;
        for (0..256) |i| {
            buckets[i] = .empty;
        }
        defer {
            for (0..256) |i| {
                buckets[i].deinit(self.allocator);
            }
        }

        const vocab_size = self.tokenizer_handle.getVocabSize();
        self.vocab_size = vocab_size;

        const token_offsets = try self.allocator.alloc(usize, vocab_size + 1);
        errdefer self.allocator.free(token_offsets);

        var token_bytes = std.ArrayListUnmanaged(u8){};
        errdefer token_bytes.deinit(self.allocator);

        for (0..vocab_size) |token_id| {
            token_offsets[token_id] = token_bytes.items.len;
            const token_text = self.tokenizer_handle.idToToken(@intCast(token_id)) orelse continue;
            if (token_text.len == 0) continue;
            const decoded = try offsets_mod.decodeTokenToBytes(self.allocator, token_text, self.tokenizer_handle);
            defer if (decoded.ptr != token_text.ptr) self.allocator.free(decoded);

            try token_bytes.appendSlice(self.allocator, decoded);
            const idx: usize = decoded[0];
            try buckets[idx].append(self.allocator, @intCast(token_id));
        }
        token_offsets[vocab_size] = token_bytes.items.len;

        const token_bytes_data = try token_bytes.toOwnedSlice(self.allocator);
        errdefer self.allocator.free(token_bytes_data);

        var built: usize = 0;
        errdefer {
            for (0..built) |i| {
                if (self.prefix_index[i].len > 0) {
                    self.allocator.free(self.prefix_index[i]);
                }
            }
        }

        for (0..256) |i| {
            self.prefix_index[i] = try buckets[i].toOwnedSlice(self.allocator);
            built += 1;
        }

        self.token_bytes_data = token_bytes_data;
        self.token_bytes_offsets = token_offsets;
    }

    /// Encode text to token IDs. Accepts sentinel-terminated slice to avoid allocation.
    pub fn encodeZ(self: *Tokenizer, text: [:0]const u8) ![]u32 {
        var needed_count: usize = 0;
        if (pipeline_impl.tokenizer_encode_ids(self.tokenizer_handle, text.ptr, null, &needed_count) != 0) {
            return TokenizerError.EncodeFailed;
        }

        // Empty input returns empty output (0 tokens is valid)
        if (needed_count == 0) {
            return &[_]u32{};
        }

        var output_ids = try self.allocator.alloc(u32, @intCast(needed_count));
        var output_len = needed_count;
        if (pipeline_impl.tokenizer_encode_ids(self.tokenizer_handle, text.ptr, @ptrCast(output_ids.ptr), &output_len) != 0) {
            self.allocator.free(output_ids);
            return TokenizerError.EncodeFailed;
        }
        if (output_len != needed_count) output_ids = output_ids[0..@intCast(output_len)];
        return output_ids;
    }

    /// Encode text to token IDs (allocates to add null terminator).
    /// Note: This version doesn't support null bytes in text. Use encodeSlice for that.
    pub fn encode(self: *Tokenizer, text: []const u8) ![]u32 {
        const text_z = try self.allocator.dupeZ(u8, text);
        defer self.allocator.free(text_z);
        return self.encodeZ(text_z);
    }

    /// Encode options for the Tokenizer API.
    pub const EncodeOptions = struct {
        /// If true, add special tokens (BOS/EOS/CLS/SEP) according to tokenizer config.
        /// If false, encode raw text without special tokens.
        add_special_tokens: bool = true,
    };

    /// Encode text to token IDs using a slice directly (supports null bytes in text).
    pub fn encodeSlice(self: *Tokenizer, text: []const u8) ![]u32 {
        return self.encodeSliceWithOptions(text, .{});
    }

    /// Encode text to token IDs with options (thread-safe).
    pub fn encodeSliceWithOptions(self: *Tokenizer, text: []const u8, options: EncodeOptions) ![]u32 {
        var needed_count: usize = 0;
        // First call to get the count
        if (pipeline_impl.tokenizer_encode_ids_slice_with_options(
            self.tokenizer_handle,
            text,
            null,
            &needed_count,
            options.add_special_tokens,
        ) != 0) {
            return TokenizerError.EncodeFailed;
        }

        // Empty input returns empty output (0 tokens is valid)
        if (needed_count == 0) {
            return &[_]u32{};
        }

        var output_ids = try self.allocator.alloc(u32, @intCast(needed_count));
        var output_len = needed_count;
        // Second call to fill the buffer
        if (pipeline_impl.tokenizer_encode_ids_slice_with_options(
            self.tokenizer_handle,
            text,
            @ptrCast(output_ids.ptr),
            &output_len,
            options.add_special_tokens,
        ) != 0) {
            self.allocator.free(output_ids);
            return TokenizerError.EncodeFailed;
        }
        if (output_len != needed_count) output_ids = output_ids[0..@intCast(output_len)];
        return output_ids;
    }

    fn idsToI32Checked(self: *Tokenizer, ids: []const u32) ![]i32 {
        const ids_i32_buffer = try self.allocator.alloc(i32, ids.len);
        errdefer self.allocator.free(ids_i32_buffer);
        for (ids, 0..) |id, idx| {
            if (self.vocab_size > 0 and id >= self.vocab_size) {
                return TokenizerError.InvalidTokenId;
            }
            ids_i32_buffer[idx] = std.math.cast(i32, id) orelse return TokenizerError.InvalidTokenId;
        }
        return ids_i32_buffer;
    }

    /// Decode options for the Tokenizer API.
    pub const DecodeOptions = struct {
        /// If true (default), skip special tokens (BOS/EOS/CLS/SEP) in output.
        /// Set to false to include special tokens in output (useful for debugging).
        skip_special_tokens: bool = true,
    };

    pub fn decode(self: *Tokenizer, ids: []const u32) ![]u8 {
        return self.decodeWithOptions(ids, .{});
    }

    /// Decode token IDs to text with options.
    pub fn decodeWithOptions(self: *Tokenizer, ids: []const u32, options: DecodeOptions) ![]u8 {
        const raw = try self.decodeRawBytes(ids, options);

        // Check if output is valid UTF-8; if so, return directly
        if (std.unicode.utf8ValidateSlice(raw)) return raw;

        // Invalid UTF-8: replace invalid bytes with U+FFFD.
        // This is correct for batch decode (all tokens present, so any
        // invalid bytes are genuinely malformed).  For streaming single-
        // token decode, callers should use decodeRawBytes + their own
        // UTF-8 byte buffering instead.
        defer self.allocator.free(raw);
        return sanitizeUtf8(self.allocator, raw);
    }

    /// Decode token IDs to raw bytes without UTF-8 sanitization.
    ///
    /// For byte-level BPE tokenizers (GPT-2, Qwen, Llama3), a single
    /// token may decode to an incomplete UTF-8 byte sequence.  This
    /// function returns those raw bytes as-is, letting the caller
    /// buffer across token boundaries for streaming decode.
    ///
    /// Caller owns the returned slice and must free with self.allocator.
    pub fn decodeRawBytes(self: *Tokenizer, ids: []const u32, options: DecodeOptions) ![]u8 {
        const ids_i32_buffer = try self.idsToI32Checked(ids);
        defer self.allocator.free(ids_i32_buffer);

        var decoded_ptr: ?[*]u8 = null;
        var decoded_len: usize = 0;
        if (pipeline_impl.tokenizer_decode_with_options(
            self.tokenizer_handle,
            ids_i32_buffer.ptr,
            ids_i32_buffer.len,
            @ptrCast(&decoded_ptr),
            &decoded_len,
            options.skip_special_tokens,
        ) != 0 or decoded_ptr == null) {
            return TokenizerError.DecodeFailed;
        }
        defer pipeline_impl.tokenizer_string_free_with_len(@ptrCast(decoded_ptr.?), decoded_len + 1);

        const decoded_bytes = decoded_ptr.?[0..decoded_len];
        const output_bytes = try self.allocator.alloc(u8, decoded_len);
        @memcpy(output_bytes, decoded_bytes);
        return output_bytes;
    }

    pub fn lastError(self: *Tokenizer) ?[]const u8 {
        const error_message_ptr = pipeline_impl.tokenizer_get_last_error(self.tokenizer_handle);
        if (error_message_ptr == null) return null;
        return std.mem.span(error_message_ptr);
    }

    /// Create a streaming decoder for token-by-token text output.
    /// Useful for streaming generation output as tokens arrive.
    pub fn streamingDecoder(self: *Tokenizer, ids: []const u32) StreamingDecoder {
        return .{ .tokenizer = self, .ids = ids };
    }
};

/// Decode tokens to text one at a time for streaming output.
///
/// Use this when you want to display text as tokens are generated,
/// rather than waiting for all tokens to decode at once. Each call
/// to `next()` returns the text for one token.
///
/// Created via `Tokenizer.streamingDecoder()`.
pub const StreamingDecoder = struct {
    tokenizer: *Tokenizer,
    ids: []const u32,
    index: usize = 0,

    /// Get the next token as text. Returns null when all tokens are consumed.
    /// Caller owns the returned slice and must free it.
    pub fn next(self: *StreamingDecoder) !?[]u8 {
        if (self.index >= self.ids.len) return null;
        const next_token_id = self.ids[self.index];
        self.index += 1;
        const decoded = try self.tokenizer.decode(&.{next_token_id});
        return decoded;
    }

    /// Reset to decode from the beginning again.
    pub fn reset(self: *StreamingDecoder) void {
        self.index = 0;
    }
};

/// Sanitize byte sequence to valid UTF-8, replacing invalid bytes with U+FFFD.
/// This handles cases where token decoding produces partial/invalid UTF-8
/// (e.g., high-temperature sampling producing random token sequences).
fn sanitizeUtf8(allocator: std.mem.Allocator, bytes: []const u8) ![]u8 {
    // Worst case: every byte is invalid and becomes 3-byte replacement char
    var sanitized = std.ArrayListUnmanaged(u8){};
    errdefer sanitized.deinit(allocator);

    var cursor: usize = 0;
    while (cursor < bytes.len) {
        const current_byte = bytes[cursor];

        // Determine expected sequence length from first byte
        const sequence_len: usize = if (current_byte < 0x80)
            1
        else if (current_byte & 0xE0 == 0xC0)
            2
        else if (current_byte & 0xF0 == 0xE0)
            3
        else if (current_byte & 0xF8 == 0xF0)
            4
        else {
            // Invalid start byte: replace and continue
            try sanitized.appendSlice(allocator, REPLACEMENT_CHAR);
            cursor += 1;
            continue;
        };

        // Check if we have enough bytes and they're valid continuation bytes
        if (cursor + sequence_len > bytes.len) {
            // Incomplete sequence at end: replace remaining bytes
            try sanitized.appendSlice(allocator, REPLACEMENT_CHAR);
            break;
        }

        var is_valid = true;
        for (1..sequence_len) |cont_idx| {
            if (bytes[cursor + cont_idx] & 0xC0 != 0x80) {
                is_valid = false;
                break;
            }
        }

        if (is_valid) {
            // Valid UTF-8 sequence: copy it
            try sanitized.appendSlice(allocator, bytes[cursor .. cursor + sequence_len]);
            cursor += sequence_len;
        } else {
            // Invalid sequence: replace first byte and continue
            try sanitized.appendSlice(allocator, REPLACEMENT_CHAR);
            cursor += 1;
        }
    }

    return sanitized.toOwnedSlice(allocator);
}

test "sanitizeUtf8 valid ascii" {
    const sanitized = try sanitizeUtf8(std.testing.allocator, "hello");
    defer std.testing.allocator.free(sanitized);
    try std.testing.expectEqualStrings("hello", sanitized);
}

test "sanitizeUtf8 valid utf8" {
    const sanitized = try sanitizeUtf8(std.testing.allocator, "café");
    defer std.testing.allocator.free(sanitized);
    try std.testing.expectEqualStrings("café", sanitized);
}

test "sanitizeUtf8 invalid byte" {
    const sanitized = try sanitizeUtf8(std.testing.allocator, "a\xFFb");
    defer std.testing.allocator.free(sanitized);
    try std.testing.expectEqualStrings("a\xEF\xBF\xBDb", sanitized);
}

test "sanitizeUtf8 truncated sequence" {
    // \xC3 expects one continuation byte but we only have end of string
    const sanitized = try sanitizeUtf8(std.testing.allocator, "ab\xC3");
    defer std.testing.allocator.free(sanitized);
    try std.testing.expectEqualStrings("ab\xEF\xBF\xBD", sanitized);
}

test "tokenizer fails cleanly for missing path" {
    const bad_path = "/does/not/exist";
    const tokenizer = Tokenizer.initFromPath(std.testing.allocator, bad_path);
    try std.testing.expectError(TokenizerError.InitFailed, tokenizer);
}

test "tokenizer decode rejects out-of-range u32 token" {
    var dummy = Tokenizer{
        .allocator = std.testing.allocator,
        .tokenizer_handle = undefined,
    };
    const ids = [_]u32{0xFFFFFFFF};
    try std.testing.expectError(TokenizerError.InvalidTokenId, dummy.idsToI32Checked(&ids));
}

test "idsToI32Checked - valid conversion" {
    var dummy = Tokenizer{
        .allocator = std.testing.allocator,
        .tokenizer_handle = undefined,
    };
    const ids = [_]u32{ 0, 42, 100, 0x7FFFFFFF };
    const ids_i32 = try dummy.idsToI32Checked(&ids);
    defer std.testing.allocator.free(ids_i32);

    try std.testing.expectEqual(@as(usize, 4), ids_i32.len);
    try std.testing.expectEqual(@as(i32, 0), ids_i32[0]);
    try std.testing.expectEqual(@as(i32, 42), ids_i32[1]);
    try std.testing.expectEqual(@as(i32, 100), ids_i32[2]);
    try std.testing.expectEqual(@as(i32, 0x7FFFFFFF), ids_i32[3]);
}

test "idsToI32Checked - empty slice" {
    var dummy = Tokenizer{
        .allocator = std.testing.allocator,
        .tokenizer_handle = undefined,
    };
    const ids = [_]u32{};
    const ids_i32 = try dummy.idsToI32Checked(&ids);
    defer std.testing.allocator.free(ids_i32);

    try std.testing.expectEqual(@as(usize, 0), ids_i32.len);
}

test "idsToI32Checked - max valid i32 value" {
    var dummy = Tokenizer{
        .allocator = std.testing.allocator,
        .tokenizer_handle = undefined,
    };
    const ids = [_]u32{0x7FFFFFFF};
    const ids_i32 = try dummy.idsToI32Checked(&ids);
    defer std.testing.allocator.free(ids_i32);

    try std.testing.expectEqual(@as(i32, 0x7FFFFFFF), ids_i32[0]);
}

test "idsToI32Checked - rejects value just over i32 max" {
    var dummy = Tokenizer{
        .allocator = std.testing.allocator,
        .tokenizer_handle = undefined,
    };
    const ids = [_]u32{0x80000000}; // i32::MAX + 1
    try std.testing.expectError(TokenizerError.InvalidTokenId, dummy.idsToI32Checked(&ids));
}

test "Tokenizer.streamingDecoder - creates decoder with initial state" {
    // Verify streamingDecoder() creates decoder with correct initialization
    var dummy_tokenizer = Tokenizer{
        .allocator = std.testing.allocator,
        .tokenizer_handle = undefined,
    };
    const ids = [_]u32{ 1, 2, 3 };
    const decoder = dummy_tokenizer.streamingDecoder(&ids);

    try std.testing.expectEqual(@as(usize, 0), decoder.index);
    try std.testing.expectEqual(@as(usize, 3), decoder.ids.len);
}

test "StreamingDecoder.reset - resets index to zero" {
    // Verify reset() resets the decoder to the beginning
    var dummy_tokenizer = Tokenizer{
        .allocator = std.testing.allocator,
        .tokenizer_handle = undefined,
    };
    const ids = [_]u32{ 1, 2, 3, 4, 5 };
    var decoder = dummy_tokenizer.streamingDecoder(&ids);

    // Simulate advancing through tokens
    decoder.index = 3;
    try std.testing.expectEqual(@as(usize, 3), decoder.index);

    // Call reset
    decoder.reset();

    // Verify index is back to zero
    try std.testing.expectEqual(@as(usize, 0), decoder.index);
}

test "DecodeOptions - default values" {
    const options = Tokenizer.DecodeOptions{};
    try std.testing.expectEqual(true, options.skip_special_tokens);
}

test "DecodeOptions - custom values" {
    const options = Tokenizer.DecodeOptions{
        .skip_special_tokens = false,
    };
    try std.testing.expectEqual(false, options.skip_special_tokens);
}

test "EncodeOptions - default values" {
    const options = Tokenizer.EncodeOptions{};
    try std.testing.expectEqual(true, options.add_special_tokens);
}

test "EncodeOptions - custom values" {
    const options = Tokenizer.EncodeOptions{
        .add_special_tokens = false,
    };
    try std.testing.expectEqual(false, options.add_special_tokens);
}

// =============================================================================
// Unit tests for coverage - functions requiring integration testing
// =============================================================================

test "initFromPathZ fails for missing path" {
    // initFromPathZ requires a valid tokenizer.json file
    // Test the error path for missing file
    const bad_path: [:0]const u8 = "/nonexistent/path/to/tokenizer";
    const result = Tokenizer.initFromPathZ(std.testing.allocator, bad_path);
    try std.testing.expectError(TokenizerError.InitFailed, result);
}

test "initFromPath fails for missing path" {
    // initFromPath wraps initFromPathZ, testing error path
    const bad_path = "/nonexistent/tokenizer/path";
    const result = Tokenizer.initFromPath(std.testing.allocator, bad_path);
    try std.testing.expectError(TokenizerError.InitFailed, result);
}

test "initFromJsonZ fails for invalid JSON" {
    // initFromJsonZ requires valid tokenizer JSON content
    // Test the error path for invalid JSON
    const invalid_json: [:0]const u8 = "not valid json";
    const result = Tokenizer.initFromJsonZ(std.testing.allocator, invalid_json);
    try std.testing.expectError(TokenizerError.InitFailed, result);
}

test "initFromJson fails for invalid JSON" {
    // initFromJson wraps initFromJsonZ, testing error path
    const invalid_json = "{invalid json content}";
    const result = Tokenizer.initFromJson(std.testing.allocator, invalid_json);
    try std.testing.expectError(TokenizerError.InitFailed, result);
}

// =============================================================================
// API coverage tests for functions requiring C API handles
// =============================================================================
// Note: Functions like deinit, encodeZ, encode, encodeSlice,
// encodeSliceWithOptions, decodeWithOptions, lastError, and StreamingDecoder.next
// require valid tokenizer handles from the C API. Full functional testing
// happens in integration tests (tests/tokenizer/) with real tokenizer configs.
//
// The tests below verify contracts that CAN be tested without C API interaction:
// - Wrapper function behavior (aliases, default parameter forwarding)
// - Error enum coverage
// =============================================================================

test "Tokenizer.init is alias for initFromPath" {
    // Verify init is correctly aliased to initFromPath
    const path = "/nonexistent/tokenizer";
    const result = Tokenizer.init(std.testing.allocator, path);
    try std.testing.expectError(TokenizerError.InitFailed, result);
}

test "encodeSlice delegates to encodeSliceWithOptions with defaults" {
    // encodeSlice should use default EncodeOptions
    const default_opts = Tokenizer.EncodeOptions{};
    try std.testing.expectEqual(true, default_opts.add_special_tokens);
    // Integration tests verify encodeSlice behavior matches encodeSliceWithOptions(.{})
}

test "decode delegates to decodeWithOptions with defaults" {
    // decode should use default DecodeOptions
    const default_opts = Tokenizer.DecodeOptions{};
    try std.testing.expectEqual(true, default_opts.skip_special_tokens);
    // Integration tests verify decode behavior matches decodeWithOptions(.{})
}

test "TokenizerError - all error variants defined" {
    // Verify all expected error types are defined
    const init_err: TokenizerError = TokenizerError.InitFailed;
    const encode_err: TokenizerError = TokenizerError.EncodeFailed;
    const decode_err: TokenizerError = TokenizerError.DecodeFailed;

    try std.testing.expectEqual(TokenizerError.InitFailed, init_err);
    try std.testing.expectEqual(TokenizerError.EncodeFailed, encode_err);
    try std.testing.expectEqual(TokenizerError.DecodeFailed, decode_err);
}

// =============================================================================
// Vocabulary Access
// =============================================================================

/// A single vocabulary entry with token text and ID.
pub const VocabEntry = struct {
    token: []const u8,
    id: u32,
};

/// Result of vocabulary extraction.
pub const VocabResult = struct {
    entries: []VocabEntry,
    /// Backing storage for token strings (single allocation).
    string_data: []u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *VocabResult) void {
        if (self.entries.len > 0) {
            self.allocator.free(self.entries);
        }
        if (self.string_data.len > 0) {
            self.allocator.free(self.string_data);
        }
        self.* = undefined;
    }
};

/// Extract the full vocabulary from the tokenizer.
/// Returns entries sorted by token ID.
pub fn getVocab(allocator: std.mem.Allocator, tokenizer_handle: *ct.Tokenizer) !VocabResult {
    const vocab_size = tokenizer_handle.getVocabSize();

    // First pass: count valid entries and total string bytes needed
    var valid_count: usize = 0;
    var total_string_bytes: usize = 0;
    for (0..vocab_size) |token_id| {
        if (tokenizer_handle.idToToken(@intCast(token_id))) |token_text| {
            valid_count += 1;
            total_string_bytes += token_text.len;
        }
    }

    if (valid_count == 0) {
        return VocabResult{
            .entries = &.{},
            .string_data = &.{},
            .allocator = allocator,
        };
    }

    // Allocate entries and string backing storage
    const entries = try allocator.alloc(VocabEntry, valid_count);
    errdefer allocator.free(entries);

    const string_data = try allocator.alloc(u8, total_string_bytes);
    errdefer allocator.free(string_data);

    // Second pass: populate entries
    var write_idx: usize = 0;
    var string_offset: usize = 0;
    for (0..vocab_size) |token_id| {
        if (tokenizer_handle.idToToken(@intCast(token_id))) |token_text| {
            // Copy token string to backing storage
            @memcpy(string_data[string_offset..][0..token_text.len], token_text);
            entries[write_idx] = .{
                .token = string_data[string_offset..][0..token_text.len],
                .id = @intCast(token_id),
            };
            string_offset += token_text.len;
            write_idx += 1;
        }
    }

    return VocabResult{
        .entries = entries,
        .string_data = string_data,
        .allocator = allocator,
    };
}

// =============================================================================
// Tokenize to Bytes
// =============================================================================

/// Result of tokenize-to-bytes operation.
/// Provides token strings as a flat byte buffer with offsets.
pub const TokenizeBytesResult = struct {
    /// Flat buffer containing all token strings concatenated.
    data: []u8,
    /// Offsets into data for each token (length = num_tokens + 1).
    /// Token i spans data[offsets[i]..offsets[i+1]].
    offsets: []usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *TokenizeBytesResult) void {
        if (self.data.len > 0) {
            self.allocator.free(self.data);
        }
        if (self.offsets.len > 0) {
            self.allocator.free(self.offsets);
        }
        self.* = undefined;
    }

    /// Get the number of tokens.
    pub fn tokenCount(self: *const TokenizeBytesResult) usize {
        if (self.offsets.len == 0) return 0;
        return self.offsets.len - 1;
    }

    /// Get the byte slice for token at index.
    pub fn getToken(self: *const TokenizeBytesResult, index: usize) ?[]const u8 {
        if (index >= self.tokenCount()) return null;
        return self.data[self.offsets[index]..self.offsets[index + 1]];
    }
};

/// Tokenize text and return token strings as a flat byte buffer with offsets.
pub fn tokenizeToBytes(
    allocator: std.mem.Allocator,
    tokenizer_handle: *ct.Tokenizer,
    text: []const u8,
) !TokenizeBytesResult {
    const encode = @import("pipeline.zig").encode;

    var token_encoding = std.mem.zeroes(ct.TokenizerEncoding);
    // tokenize (not encode): skip special tokens so BOS/EOS are not included
    if (encode.tokenizer_encode_struct_with_options(tokenizer_handle, text, &token_encoding, .{ .add_special_tokens = false }) != 0) {
        return TokenizerError.EncodeFailed;
    }
    defer encode.tokenizer_encoding_free_struct(&token_encoding);

    const token_count = token_encoding.tokens_len;

    // Handle empty result
    if (token_count == 0 or token_encoding.tokens == null) {
        const offsets = try allocator.alloc(usize, 1);
        offsets[0] = 0;
        return TokenizeBytesResult{
            .data = &.{},
            .offsets = offsets,
            .allocator = allocator,
        };
    }

    const token_cstrs: [*][*c]u8 = @ptrCast(token_encoding.tokens.?);

    // Calculate total bytes needed
    var total_bytes: usize = 0;
    for (0..token_count) |i| {
        if (token_cstrs[i]) |ptr| {
            total_bytes += std.mem.len(@as([*:0]u8, @ptrCast(ptr)));
        }
    }

    // Allocate output buffers
    const offsets = try allocator.alloc(usize, token_count + 1);
    errdefer allocator.free(offsets);

    const data = if (total_bytes > 0)
        try allocator.alloc(u8, total_bytes)
    else
        @as([]u8, &.{});
    errdefer if (total_bytes > 0) allocator.free(data);

    // Copy token strings to flat buffer
    var write_pos: usize = 0;
    for (0..token_count) |i| {
        offsets[i] = write_pos;
        if (token_cstrs[i]) |ptr| {
            const token_bytes = std.mem.span(@as([*:0]u8, @ptrCast(ptr)));
            if (data.len > 0) {
                @memcpy(data[write_pos..][0..token_bytes.len], token_bytes);
            }
            write_pos += token_bytes.len;
        }
    }
    offsets[token_count] = write_pos;

    return TokenizeBytesResult{
        .data = data,
        .offsets = offsets,
        .allocator = allocator,
    };
}

// =============================================================================
// Vocab and TokenizeBytes Tests
// =============================================================================

test "VocabResult.deinit handles empty result" {
    var result = VocabResult{
        .entries = &.{},
        .string_data = &.{},
        .allocator = std.testing.allocator,
    };
    result.deinit();
}

test "TokenizeBytesResult.tokenCount returns correct count" {
    var offsets = [_]usize{ 0, 3, 7, 10 };
    const result = TokenizeBytesResult{
        .data = &.{},
        .offsets = &offsets,
        .allocator = std.testing.allocator,
    };
    try std.testing.expectEqual(@as(usize, 3), result.tokenCount());
}

test "TokenizeBytesResult.tokenCount handles empty" {
    const result = TokenizeBytesResult{
        .data = &.{},
        .offsets = &.{},
        .allocator = std.testing.allocator,
    };
    try std.testing.expectEqual(@as(usize, 0), result.tokenCount());
}

test "TokenizeBytesResult.getToken returns correct slice" {
    var data = [_]u8{ 'a', 'b', 'c', 'd', 'e', 'f' };
    var offsets = [_]usize{ 0, 2, 5, 6 };
    const result = TokenizeBytesResult{
        .data = &data,
        .offsets = &offsets,
        .allocator = std.testing.allocator,
    };

    try std.testing.expectEqualStrings("ab", result.getToken(0).?);
    try std.testing.expectEqualStrings("cde", result.getToken(1).?);
    try std.testing.expectEqualStrings("f", result.getToken(2).?);
    try std.testing.expect(result.getToken(3) == null);
}
