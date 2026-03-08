//! Token Offset Computation & Rich Encoding
//!
//! Utilities for computing byte offsets that map tokens back to source text,
//! and the combined "rich encode" operation that produces IDs, offsets, and masks
//! in a single pass through the encoding pipeline.

const std = @import("std");
const pipeline = @import("pipeline.zig");
const ct = @import("c_types.zig");
const tok_encode = @import("encode.zig");
const normalize = @import("normalize.zig");
const unigram = @import("unigram.zig");
const utils = @import("utils.zig");
const wordpiece = @import("wordpiece.zig");
const c = @cImport({
    @cInclude("utf8proc.h");
});

/// Token offset in source text (UTF-8 byte indices).
pub const TokenOffset = extern struct {
    start: u32,
    end: u32,
};

fn findAddedTokenById(tokenizer_handle: *const ct.Tokenizer, token_id: i32) ?*const ct.AddedToken {
    var added_iter = tokenizer_handle.added;
    while (added_iter) |added| : (added_iter = added.next) {
        if (added.id == token_id) return added;
    }
    return null;
}

fn isUtf8ContinuationByte(byte_value: u8) bool {
    return (byte_value & 0xC0) == 0x80;
}

fn isUnicodeWhitespaceCodepoint(codepoint: c.utf8proc_int32_t) bool {
    return switch (c.utf8proc_category(codepoint)) {
        c.UTF8PROC_CATEGORY_ZS, c.UTF8PROC_CATEGORY_ZL, c.UTF8PROC_CATEGORY_ZP => true,
        else => false,
    };
}

fn whitespaceLenAt(input_bytes: []const u8, position: usize) usize {
    if (position >= input_bytes.len) return 0;
    const byte_value = input_bytes[position];
    if (byte_value < 0x80) {
        return if (std.ascii.isWhitespace(byte_value)) 1 else 0;
    }
    var codepoint: c.utf8proc_int32_t = 0;
    const consumed = c.utf8proc_iterate(@ptrCast(input_bytes.ptr + position), @intCast(input_bytes.len - position), &codepoint);
    if (consumed <= 0) return 0;
    return if (isUnicodeWhitespaceCodepoint(codepoint)) @intCast(consumed) else 0;
}

fn allWhitespace(input_bytes: []const u8) bool {
    var cursor: usize = 0;
    while (cursor < input_bytes.len) {
        const ws_len = whitespaceLenAt(input_bytes, cursor);
        if (ws_len == 0) return false;
        cursor += ws_len;
    }
    return true;
}

fn consumeWhitespaceForward(input_bytes: []const u8, start: usize) usize {
    var cursor = start;
    while (cursor < input_bytes.len) {
        const ws_len = whitespaceLenAt(input_bytes, cursor);
        if (ws_len == 0) break;
        cursor += ws_len;
    }
    return cursor;
}

fn adjustAddedTokenSpan(
    tokenizer_handle: *const ct.Tokenizer,
    token_id: i32,
    normalized_text: []const u8,
    normalized_cursor: usize,
    start_offset: usize,
    end_offset: usize,
) struct { start: usize, end: usize } {
    var span_start = start_offset;
    var span_end = end_offset;

    const added = findAddedTokenById(tokenizer_handle, token_id) orelse {
        return .{ .start = span_start, .end = span_end };
    };

    if (added.lstrip != 0 and normalized_cursor <= span_start and allWhitespace(normalized_text[normalized_cursor..span_start])) {
        span_start = normalized_cursor;
    }
    if (added.rstrip != 0) {
        span_end = consumeWhitespaceForward(normalized_text, span_end);
    }
    return .{ .start = span_start, .end = span_end };
}

/// Compute byte offsets from an already-computed encoding's token strings.
/// Matches each token's decoded byte representation against normalized text and
/// maps the normalized byte span back to the original source byte span.
/// Caller owns the returned slice.
pub fn computeOffsetsFromEncoding(
    alloc: std.mem.Allocator,
    encoding: *const ct.TokenizerEncoding,
    tokenizer_handle: *ct.Tokenizer,
    text: []const u8,
) ![]TokenOffset {
    const token_count = encoding.ids_len;
    if (token_count == 0) return &.{};

    const offsets = try alloc.alloc(TokenOffset, token_count);
    errdefer alloc.free(offsets);

    var normalized = try normalize.normalize_text(&tokenizer_handle.normalizer, text);
    defer normalized.deinit();
    if (tokenizer_handle.pretokenizer.add_prefix_space != 0 and tokenizer_handle.pretokenizer.metaspace == 0 and
        (normalized.text.len == 0 or normalized.text[0] != ' '))
    {
        try normalize.addPrefixSpace(&normalized);
    }

    var text_offset: u32 = 0;
    var normalized_offset: usize = 0;

    const enc_ids: [*]i32 = if (encoding.ids) |ids| @ptrCast(ids) else {
        @memset(offsets, .{ .start = 0, .end = 0 });
        return offsets;
    };
    const token_cstrs: ?[*][*c]u8 = if (encoding.tokens) |toks| @ptrCast(toks) else null;
    const is_byte_level = tokenizer_handle.pretokenizer.byte_level != 0;
    const byte_level_uses_source_map = is_byte_level and byteLevelRequiresSourceMap(tokenizer_handle);
    const unk_id = tokenizer_handle.getUnkId();

    for (0..token_count) |token_idx| {
        const token_id = enc_ids[token_idx];

        // Resolve token text: from encoding.tokens if available, otherwise from vocab by ID.
        var token_text: []const u8 = blk: {
            // Byte-level offsets are defined in terms of model byte tokens, not
            // the higher-level token surfaces that encode() may expose for the
            // original UTF-8 text. Use the vocab/add-token text by ID here so a
            // single raw byte never expands to an entire emoji or accented scalar.
            if (is_byte_level and token_id != unk_id) {
                break :blk tokenizer_handle.idToToken(token_id) orelse
                    tok_encode.findAddedTokenContentById(tokenizer_handle, token_id) orelse "";
            }
            if (token_cstrs) |ts| {
                const token_ptr: [*c]u8 = ts[token_idx];
                if (token_ptr != null) break :blk std.mem.sliceTo(token_ptr, 0);
            }
            // Fall back to vocabulary lookup by ID.
            // For unk tokens, idToToken returns "<unk>" which won't match source text.
            // In byte-level mode each unk represents exactly 1 source byte, handled below.
            if (is_byte_level and token_id == unk_id) break :blk "";
            break :blk tokenizer_handle.idToToken(token_id) orelse
                tok_encode.findAddedTokenContentById(tokenizer_handle, token_id) orelse "";
        };

        if (is_byte_level and token_id == unk_id) {
            token_text = "";
        }

        if (is_byte_level and token_text.len > 0) {
            const token_byte_sequence = decodeTokenToBytes(alloc, token_text, tokenizer_handle) catch {
                offsets[token_idx] = .{ .start = 0, .end = 0 };
                continue;
            };
            defer if (token_byte_sequence.ptr != token_text.ptr) alloc.free(token_byte_sequence);

            const remaining_text = normalized.text[normalized_offset..];
            const match_offset = if (remaining_text.len >= token_byte_sequence.len and
                std.mem.eql(u8, remaining_text[0..token_byte_sequence.len], token_byte_sequence))
                @as(usize, 0)
            else
                findSubsequence(remaining_text, token_byte_sequence) orelse {
                    offsets[token_idx] = .{ .start = 0, .end = 0 };
                    continue;
                };

            const base_start_offset = normalized_offset + match_offset;
            const base_end_offset = base_start_offset + token_byte_sequence.len;
            const span = adjustAddedTokenSpan(
                tokenizer_handle,
                token_id,
                normalized.text,
                normalized_offset,
                base_start_offset,
                base_end_offset,
            );
            const start_offset = span.start;
            const end_offset = span.end;
            offsets[token_idx] = if (normalizedSpanIsSynthetic(&normalized, start_offset, end_offset))
                .{ .start = 0, .end = 0 }
            else if (byte_level_uses_source_map) blk: {
                const source_offset = normalizedSpanToSourceOffset(&normalized, start_offset, end_offset);
                text_offset = source_offset.end;
                break :blk source_offset;
            } else blk: {
                const start_raw: u32 = @intCast(start_offset);
                const end_raw: u32 = @intCast(end_offset);
                text_offset = end_raw;
                break :blk .{ .start = start_raw, .end = end_raw };
            };
            normalized_offset = end_offset;
            continue;
        }

        // Byte-level unk: each unk token represents 1 source byte
        if (token_text.len == 0 and is_byte_level and token_id == unk_id) {
            if (byte_level_uses_source_map) {
                if (normalized_offset < normalized.text.len) {
                    const source_offset = normalizedSpanToSourceOffset(&normalized, normalized_offset, normalized_offset + 1);
                    offsets[token_idx] = source_offset;
                    text_offset = source_offset.end;
                    normalized_offset += 1;
                } else {
                    offsets[token_idx] = .{ .start = 0, .end = 0 };
                }
            } else if (text_offset < text.len) {
                offsets[token_idx] = .{ .start = text_offset, .end = text_offset + 1 };
                text_offset += 1;
                normalized_offset += 1;
            } else {
                offsets[token_idx] = .{ .start = 0, .end = 0 };
            }
            continue;
        }

        if (!is_byte_level and token_id == unk_id and normalized_offset < normalized.text.len) {
            if (tokenizer_handle.type == .wordpiece) {
                const span = nextWordPieceUnknownSpan(normalized.text, normalized_offset);
                offsets[token_idx] = normalizedSpanToSourceOffset(&normalized, span.start, span.end);
                normalized_offset = span.end;
            } else if (tokenizer_handle.type == .unigram) {
                const start_offset = normalized_offset;
                const token_is_placeholder_unk = token_text.len == 0 or std.mem.eql(u8, token_text, "<unk>");
                const default_span_len = if (token_is_placeholder_unk)
                    @as(usize, 0)
                else
                    token_text.len;
                var end_offset = @min(start_offset + default_span_len, normalized.text.len);

                // Fused unigram unknown runs should own bytes up to the next
                // resolvable non-unk token boundary.
                var lookahead = token_idx + 1;
                while (lookahead < token_count) : (lookahead += 1) {
                    const next_id = enc_ids[lookahead];
                    if (next_id == unk_id) continue;

                    var next_text: []const u8 = "";
                    if (token_cstrs) |ts| {
                        const ptr: [*c]u8 = ts[lookahead];
                        if (ptr != null) next_text = std.mem.sliceTo(ptr, 0);
                    }
                    if (next_text.len == 0) {
                        next_text = tokenizer_handle.idToToken(next_id) orelse
                            tok_encode.findAddedTokenContentById(tokenizer_handle, next_id) orelse "";
                    }
                    if (next_text.len == 0) continue;

                    const next_bytes = decodeTokenToBytes(alloc, next_text, tokenizer_handle) catch continue;
                    defer if (next_bytes.ptr != next_text.ptr) alloc.free(next_bytes);

                    if (findSubsequence(normalized.text[start_offset..], next_bytes)) |match_offset| {
                        end_offset = start_offset + match_offset;
                        break;
                    }
                }

                if (end_offset <= start_offset and start_offset < normalized.text.len) {
                    if (token_is_placeholder_unk) {
                        const span = nextWordPieceUnknownSpan(normalized.text, start_offset);
                        if (span.end > span.start) {
                            end_offset = span.end;
                        } else {
                            end_offset = start_offset + 1;
                        }
                    } else {
                        end_offset = start_offset + 1;
                    }
                }
                if (end_offset > normalized.text.len) end_offset = normalized.text.len;
                offsets[token_idx] = normalizedSpanToSourceOffset(&normalized, start_offset, end_offset);
                normalized_offset = end_offset;
            } else {
                offsets[token_idx] = normalizedSpanToSourceOffset(&normalized, normalized_offset, normalized_offset + 1);
                normalized_offset += 1;
            }
            continue;
        }

        if (token_text.len == 0) {
            offsets[token_idx] = .{ .start = 0, .end = 0 };
            continue;
        }
        const token_byte_sequence = decodeTokenToBytes(alloc, token_text, tokenizer_handle) catch {
            offsets[token_idx] = .{ .start = 0, .end = 0 };
            continue;
        };
        defer if (token_byte_sequence.ptr != token_text.ptr) alloc.free(token_byte_sequence);

        const remaining_text = normalized.text[normalized_offset..];
        if (findSubsequence(remaining_text, token_byte_sequence)) |match_offset| {
            const base_start_offset = normalized_offset + match_offset;
            const base_end_offset = base_start_offset + token_byte_sequence.len;
            const span = adjustAddedTokenSpan(
                tokenizer_handle,
                token_id,
                normalized.text,
                normalized_offset,
                base_start_offset,
                base_end_offset,
            );
            const start_offset = span.start;
            const end_offset = span.end;
            offsets[token_idx] = normalizedSpanToSourceOffset(&normalized, start_offset, end_offset);
            normalized_offset = end_offset;
        } else {
            offsets[token_idx] = .{ .start = 0, .end = 0 };
        }
    }

    return offsets;
}

fn byteLevelRequiresSourceMap(tokenizer_handle: *const ct.Tokenizer) bool {
    const normalizer = tokenizer_handle.normalizer;
    return normalizer.lowercase != 0 or
        normalizer.nfc != 0 or
        normalizer.nfd != 0 or
        normalizer.nfkc != 0 or
        normalizer.nfkd != 0 or
        normalizer.strip_accents != 0 or
        normalizer.strip_left != 0 or
        normalizer.strip_right != 0 or
        normalizer.clean_text != 0 or
        normalizer.handle_chinese_chars != 0 or
        normalizer.prepend != null or
        normalizer.replace_pattern != null or
        tokenizer_handle.pretokenizer.add_prefix_space != 0;
}

fn normalizedSpanIsSynthetic(normalized: *const @import("types.zig").Normalized, start: usize, end: usize) bool {
    if (start >= end or end > normalized.map.len) return false;
    for (start..end) |idx| {
        if (normalized.map[idx] >= 0) return false;
    }
    return true;
}

fn normalizedSpanToSourceOffset(normalized: *const @import("types.zig").Normalized, start: usize, end: usize) TokenOffset {
    if (start >= end or start >= normalized.map.len or end > normalized.map.len) {
        return .{ .start = 0, .end = 0 };
    }

    var source_start: i32 = -1;
    var source_end: i32 = -1;
    for (start..end) |idx| {
        const mapped_start = normalized.map[idx];
        const mapped_end = normalized.map_end[idx];
        if (mapped_start >= 0 and (source_start < 0 or mapped_start < source_start)) {
            source_start = mapped_start;
        }
        if (mapped_end > source_end) {
            source_end = mapped_end;
        }
    }
    if (source_start < 0 or source_end < 0) return .{ .start = 0, .end = 0 };
    return .{
        .start = @intCast(source_start),
        .end = @intCast(source_end),
    };
}

/// Combined encoding result with IDs, offsets, and masks.
/// Produced by encode(). Caller owns all slices; call deinit() to free.
pub const Encoding = struct {
    ids: []u32,
    offsets: []TokenOffset,
    attention_mask: []u32,
    special_tokens_mask: []u32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Encoding) void {
        if (self.ids.len > 0) self.allocator.free(self.ids);
        if (self.offsets.len > 0) self.allocator.free(self.offsets);
        if (self.attention_mask.len > 0) self.allocator.free(self.attention_mask);
        if (self.special_tokens_mask.len > 0) self.allocator.free(self.special_tokens_mask);
        self.* = undefined;
    }

    /// Truncate the encoding to the window [start..start+count].
    /// Allocates new arrays for the window and frees the originals.
    /// If start == 0 and count == ids.len, this is a no-op.
    pub fn truncate(self: *Encoding, start: usize, count: usize) !void {
        if (count == 0) {
            self.deinit();
            return;
        }
        if (start == 0 and count == self.ids.len) return;

        const alloc = self.allocator;
        const out_ids = try alloc.alloc(u32, count);
        errdefer alloc.free(out_ids);
        const out_offsets = try alloc.alloc(TokenOffset, count);
        errdefer alloc.free(out_offsets);
        const out_mask = try alloc.alloc(u32, count);
        errdefer alloc.free(out_mask);
        const out_special = try alloc.alloc(u32, count);
        errdefer alloc.free(out_special);

        @memcpy(out_ids, self.ids[start..][0..count]);
        @memcpy(out_offsets, self.offsets[start..][0..count]);
        @memcpy(out_mask, self.attention_mask[start..][0..count]);
        @memcpy(out_special, self.special_tokens_mask[start..][0..count]);

        const saved_alloc = self.allocator;
        self.deinit();
        self.* = .{
            .ids = out_ids,
            .offsets = out_offsets,
            .attention_mask = out_mask,
            .special_tokens_mask = out_special,
            .allocator = saved_alloc,
        };
    }

    pub fn applySelectiveSpecialTokens(
        self: *Encoding,
        tokenizer_handle: *ct.Tokenizer,
        add_bos: bool,
        add_eos: bool,
    ) !void {
        const include_bos = add_bos and tokenizer_handle.postproc.cls_id >= 0;
        const include_eos = add_eos and tokenizer_handle.postproc.sep_id >= 0;
        if (!include_bos and !include_eos) return;

        const extra: usize = @intFromBool(include_bos) + @intFromBool(include_eos);
        const total_len = self.ids.len + extra;

        const alloc = self.allocator;
        const out_ids = try alloc.alloc(u32, total_len);
        errdefer alloc.free(out_ids);
        const out_offsets = try alloc.alloc(TokenOffset, total_len);
        errdefer alloc.free(out_offsets);
        const out_mask = try alloc.alloc(u32, total_len);
        errdefer alloc.free(out_mask);
        const out_special = try alloc.alloc(u32, total_len);
        errdefer alloc.free(out_special);

        var write_idx: usize = 0;
        if (include_bos) {
            out_ids[write_idx] = @intCast(tokenizer_handle.postproc.cls_id);
            out_offsets[write_idx] = .{ .start = 0, .end = 0 };
            out_mask[write_idx] = 1;
            out_special[write_idx] = 1;
            write_idx += 1;
        }

        if (self.ids.len > 0) {
            @memcpy(out_ids[write_idx..][0..self.ids.len], self.ids);
            @memcpy(out_offsets[write_idx..][0..self.offsets.len], self.offsets);
            @memcpy(out_mask[write_idx..][0..self.attention_mask.len], self.attention_mask);
            @memcpy(out_special[write_idx..][0..self.special_tokens_mask.len], self.special_tokens_mask);
            write_idx += self.ids.len;
        }

        if (include_eos) {
            out_ids[write_idx] = @intCast(tokenizer_handle.postproc.sep_id);
            out_offsets[write_idx] = .{ .start = 0, .end = 0 };
            out_mask[write_idx] = 1;
            out_special[write_idx] = 1;
        }

        const saved_alloc = self.allocator;
        self.deinit();
        self.* = .{
            .ids = out_ids,
            .offsets = out_offsets,
            .attention_mask = out_mask,
            .special_tokens_mask = out_special,
            .allocator = saved_alloc,
        };
    }
};

/// Single-pass encode producing IDs, offsets, attention mask, and special tokens mask.
///
/// Runs the encoding pipeline once and computes source-text byte offsets from the
/// resulting token strings. This avoids the O(2N) cost of calling encode() and
/// computeOffsets() separately (each of which runs the full pipeline independently).
/// Caller owns the result; call deinit() to free all buffers.
pub fn encode(
    alloc: std.mem.Allocator,
    tokenizer_handle: *ct.Tokenizer,
    text: []const u8,
    add_special_tokens: bool,
) !Encoding {
    // Full encode to get IDs, token strings, attention mask, and special tokens mask.
    var encoding = std.mem.zeroes(ct.TokenizerEncoding);
    const options = tok_encode.EncodeOptions{ .add_special_tokens = add_special_tokens };
    if (tok_encode.tokenizer_encode_struct_with_options(tokenizer_handle, text, &encoding, options) != 0) {
        return error.TokenizationFailed;
    }
    defer tok_encode.tokenizer_encoding_free_struct(&encoding);

    const token_count = encoding.ids_len;
    if (token_count == 0) {
        return Encoding{
            .ids = &.{},
            .offsets = &.{},
            .attention_mask = &.{},
            .special_tokens_mask = &.{},
            .allocator = alloc,
        };
    }

    // Compute source-text byte offsets from the encoding's token strings.
    const offsets = try computeOffsetsFromEncoding(alloc, &encoding, tokenizer_handle, text);
    errdefer alloc.free(offsets);

    // Convert i32 arrays from the encoding to u32 for the C-API contract.
    const ids = try alloc.alloc(u32, token_count);
    errdefer alloc.free(ids);

    const attention_mask = try alloc.alloc(u32, token_count);
    errdefer alloc.free(attention_mask);

    const special_tokens_mask = try alloc.alloc(u32, token_count);
    errdefer alloc.free(special_tokens_mask);

    const enc_ids: [*]i32 = @ptrCast(encoding.ids orelse return error.TokenizationFailed);
    const enc_mask: ?[*]i32 = if (encoding.attention_mask) |m| @ptrCast(m) else null;
    const enc_special: ?[*]i32 = if (encoding.special_tokens_mask) |s| @ptrCast(s) else null;

    for (0..token_count) |i| {
        ids[i] = @intCast(enc_ids[i]);
        attention_mask[i] = if (enc_mask) |m| @intCast(m[i]) else 1;
        special_tokens_mask[i] = if (enc_special) |s| @intCast(s[i]) else 0;
    }

    return Encoding{
        .ids = ids,
        .offsets = offsets,
        .attention_mask = attention_mask,
        .special_tokens_mask = special_tokens_mask,
        .allocator = alloc,
    };
}

/// Decode a token string to its raw byte representation.
/// Handles byte-level tokenization and sentencepiece underscore character.
pub fn decodeTokenToBytes(allocator: std.mem.Allocator, token_text: []const u8, tokenizer_handle: *ct.Tokenizer) ![]const u8 {
    const is_byte_level = tokenizer_handle.pretokenizer.byte_level != 0;

    if (!is_byte_level) {
        if (tokenizer_handle.type == .unigram and std.mem.startsWith(u8, token_text, "\xE2\x96\x81")) {
            const suffix = token_text[3..];
            if (suffix.len == 0) return token_text[0..0];
            return allocator.dupe(u8, suffix);
        }
        if (tokenizer_handle.type == .wordpiece and std.mem.startsWith(u8, token_text, "##")) {
            return allocator.dupe(u8, token_text[2..]);
        }
        if (token_text.len == 6 and std.mem.startsWith(u8, token_text, "<0x") and token_text[5] == '>') {
            const value = std.fmt.parseUnsigned(u8, token_text[3..5], 16) catch return token_text;
            const decoded = try allocator.alloc(u8, 1);
            decoded[0] = value;
            return decoded;
        }
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

const NormalizedSpan = struct {
    start: usize,
    end: usize,
};

fn nextWordPieceUnknownSpan(text: []const u8, offset: usize) NormalizedSpan {
    var start = offset;
    while (start < text.len and std.ascii.isWhitespace(text[start])) : (start += 1) {}
    if (start >= text.len) return .{ .start = offset, .end = offset };

    if (isAsciiPunctuation(text[start])) {
        const punct_len = utf8CharLen(text[start]);
        return .{ .start = start, .end = @min(text.len, start + punct_len) };
    }

    var end = start;
    while (end < text.len) {
        const byte = text[end];
        if (std.ascii.isWhitespace(byte) or isAsciiPunctuation(byte)) break;
        end += utf8CharLen(byte);
    }
    return .{ .start = start, .end = end };
}

fn isAsciiPunctuation(ch: u8) bool {
    return switch (ch) {
        '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~' => true,
        else => false,
    };
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

test "decodeTokenToBytes returns borrowed empty slice for standalone unigram metaspace token" {
    var tokenizer = std.mem.zeroes(ct.Tokenizer);
    tokenizer.type = .unigram;

    const token = "\xE2\x96\x81";
    const decoded = try decodeTokenToBytes(std.testing.allocator, token, &tokenizer);

    try std.testing.expectEqual(@as(usize, 0), decoded.len);
    try std.testing.expectEqual(token.ptr, decoded.ptr);
}

test "computeOffsetsFromEncoding requires integration testing" {
    // Requires fully initialized tokenizer with vocab, model, normalizer.
    // Integration tests: tests/tokenizer/test_*.py
}

fn addWordPieceTestToken(model: *wordpiece.WordPieceModel, token: []const u8, id: usize) !void {
    const allocator = model.allocator;
    const dup = try allocator.dupeZ(u8, token);
    errdefer allocator.free(dup);
    try model.vocab_strings.append(allocator, dup);
    try model.vocab.put(allocator, dup[0..dup.len], @intCast(id));
    model.id_to_token[id] = dup.ptr;
}

fn initWordPieceTokenizerForOffsets(allocator: std.mem.Allocator, vocab_size: usize) !struct {
    tokenizer: *ct.Tokenizer,
    model: *wordpiece.WordPieceModel,
} {
    const tokenizer = try allocator.create(ct.Tokenizer);
    errdefer allocator.destroy(tokenizer);
    tokenizer.* = std.mem.zeroes(ct.Tokenizer);
    tokenizer.type = .wordpiece;

    const model = try allocator.create(wordpiece.WordPieceModel);
    errdefer allocator.destroy(model);
    model.* = .{
        .allocator = allocator,
        .vocab = .{},
        .id_to_token = try allocator.alloc(?[*:0]u8, vocab_size),
        .vocab_strings = .{},
        .vocab_size = vocab_size,
        .unk_id = 0,
        .unk_token = std.mem.zeroes([16]u8),
        .max_input_chars_per_word = 200,
        .owner = tokenizer,
    };
    @memset(model.id_to_token, null);
    @memcpy(model.unk_token[0.."[UNK]".len], "[UNK]");

    tokenizer.model = model;
    return .{ .tokenizer = tokenizer, .model = model };
}

fn deinitWordPieceTokenizerForOffsets(allocator: std.mem.Allocator, tokenizer: *ct.Tokenizer, model: *wordpiece.WordPieceModel) void {
    tokenizer.model = null;
    for (model.vocab_strings.items) |s| allocator.free(s);
    model.vocab_strings.deinit(allocator);
    model.vocab.deinit(allocator);
    allocator.free(model.id_to_token);
    allocator.destroy(model);
    allocator.destroy(tokenizer);
}

fn addUnigramTestToken(model: *unigram.UnigramModel, token: []const u8, score: f32, id: usize) !void {
    const allocator = model.allocator;
    const dup = try allocator.dupeZ(u8, token);
    errdefer allocator.free(dup);
    try model.vocab.append(allocator, .{
        .token = dup,
        .score = score,
        .id = @intCast(id),
    });
    model.id_to_token[id] = dup.ptr;
}

fn initUnigramTokenizerForOffsets(allocator: std.mem.Allocator, vocab_size: usize) !struct {
    tokenizer: *ct.Tokenizer,
    model: *unigram.UnigramModel,
} {
    const tokenizer = try allocator.create(ct.Tokenizer);
    errdefer allocator.destroy(tokenizer);
    tokenizer.* = std.mem.zeroes(ct.Tokenizer);
    tokenizer.type = .unigram;
    tokenizer.pretokenizer.metaspace = 1;
    tokenizer.pretokenizer.add_prefix_space = 1;
    tokenizer.decoder.metaspace = 1;
    tokenizer.decoder.add_prefix_space = 1;

    const model = try allocator.create(unigram.UnigramModel);
    errdefer allocator.destroy(model);
    model.* = .{
        .allocator = allocator,
        .vocab = .{},
        .id_to_token = try allocator.alloc(?[*:0]u8, vocab_size),
        .vocab_size = vocab_size,
        .unk_id = 0,
        .bos_id = -1,
        .eos_id = -1,
        .unk_token = std.mem.zeroes([16]u8),
        .unk_entry = null,
        .owner = tokenizer,
    };
    @memset(model.id_to_token, null);
    @memcpy(model.unk_token[0.."<unk>".len], "<unk>");

    tokenizer.model = model;
    return .{ .tokenizer = tokenizer, .model = model };
}

fn deinitUnigramTokenizerForOffsets(allocator: std.mem.Allocator, tokenizer: *ct.Tokenizer, model: *unigram.UnigramModel) void {
    tokenizer.model = null;
    for (model.vocab.items) |entry| allocator.free(entry.token);
    model.vocab.deinit(allocator);
    allocator.free(model.id_to_token);
    allocator.destroy(model);
    allocator.destroy(tokenizer);
}

test "computeOffsetsFromEncoding strips wordpiece continuation prefix" {
    const allocator = std.testing.allocator;
    const setup = try initWordPieceTokenizerForOffsets(allocator, 3);
    defer deinitWordPieceTokenizerForOffsets(allocator, setup.tokenizer, setup.model);

    try addWordPieceTestToken(setup.model, "[UNK]", 0);
    try addWordPieceTestToken(setup.model, "go", 1);
    try addWordPieceTestToken(setup.model, "##ing", 2);

    const ids = try allocator.alloc(i32, 2);
    defer allocator.free(ids);
    ids[0] = 1;
    ids[1] = 2;

    const encoding = ct.TokenizerEncoding{
        .ids = @ptrCast(ids.ptr),
        .ids_len = 2,
        .tokens = null,
        .tokens_len = 0,
        .attention_mask = null,
        .type_ids = null,
        .special_tokens_mask = null,
        .offsets = null,
        .overflows = null,
        .overflow_count = 0,
    };

    const offsets = try computeOffsetsFromEncoding(allocator, &encoding, setup.tokenizer, "going");
    defer allocator.free(offsets);

    try std.testing.expectEqual(@as(u32, 0), offsets[0].start);
    try std.testing.expectEqual(@as(u32, 2), offsets[0].end);
    try std.testing.expectEqual(@as(u32, 2), offsets[1].start);
    try std.testing.expectEqual(@as(u32, 5), offsets[1].end);
}

test "computeOffsetsFromEncoding maps wordpiece unk to full word span" {
    const allocator = std.testing.allocator;
    const setup = try initWordPieceTokenizerForOffsets(allocator, 3);
    defer deinitWordPieceTokenizerForOffsets(allocator, setup.tokenizer, setup.model);

    try addWordPieceTestToken(setup.model, "[UNK]", 0);
    try addWordPieceTestToken(setup.model, "hello", 1);
    try addWordPieceTestToken(setup.model, "world", 2);

    const ids = try allocator.alloc(i32, 3);
    defer allocator.free(ids);
    ids[0] = 1;
    ids[1] = 0;
    ids[2] = 2;

    const encoding = ct.TokenizerEncoding{
        .ids = @ptrCast(ids.ptr),
        .ids_len = 3,
        .tokens = null,
        .tokens_len = 0,
        .attention_mask = null,
        .type_ids = null,
        .special_tokens_mask = null,
        .offsets = null,
        .overflows = null,
        .overflow_count = 0,
    };

    const offsets = try computeOffsetsFromEncoding(allocator, &encoding, setup.tokenizer, "hello xyz world");
    defer allocator.free(offsets);

    try std.testing.expectEqual(@as(u32, 0), offsets[0].start);
    try std.testing.expectEqual(@as(u32, 5), offsets[0].end);
    try std.testing.expectEqual(@as(u32, 6), offsets[1].start);
    try std.testing.expectEqual(@as(u32, 9), offsets[1].end);
    try std.testing.expectEqual(@as(u32, 10), offsets[2].start);
    try std.testing.expectEqual(@as(u32, 15), offsets[2].end);
}

test "computeOffsetsFromEncoding maps unigram whole-word metaspace token to source span" {
    const allocator = std.testing.allocator;
    const setup = try initUnigramTokenizerForOffsets(allocator, 2);
    defer deinitUnigramTokenizerForOffsets(allocator, setup.tokenizer, setup.model);

    try addUnigramTestToken(setup.model, "<unk>", 0.0, 0);
    try addUnigramTestToken(setup.model, "\xE2\x96\x81hello", -1.0, 1);

    const ids = try allocator.alloc(i32, 1);
    defer allocator.free(ids);
    ids[0] = 1;

    const encoding = ct.TokenizerEncoding{
        .ids = @ptrCast(ids.ptr),
        .ids_len = 1,
        .tokens = null,
        .tokens_len = 0,
        .attention_mask = null,
        .type_ids = null,
        .special_tokens_mask = null,
        .offsets = null,
        .overflows = null,
        .overflow_count = 0,
    };

    const offsets = try computeOffsetsFromEncoding(allocator, &encoding, setup.tokenizer, "hello");
    defer allocator.free(offsets);

    try std.testing.expectEqual(@as(u32, 0), offsets[0].start);
    try std.testing.expectEqual(@as(u32, 5), offsets[0].end);
}

test "computeOffsetsFromEncoding maps unigram multiword metaspace tokens to each word span" {
    const allocator = std.testing.allocator;
    const setup = try initUnigramTokenizerForOffsets(allocator, 3);
    defer deinitUnigramTokenizerForOffsets(allocator, setup.tokenizer, setup.model);

    try addUnigramTestToken(setup.model, "<unk>", 0.0, 0);
    try addUnigramTestToken(setup.model, "\xE2\x96\x81hello", -1.0, 1);
    try addUnigramTestToken(setup.model, "\xE2\x96\x81world", -1.5, 2);

    const ids = try allocator.alloc(i32, 2);
    defer allocator.free(ids);
    ids[0] = 1;
    ids[1] = 2;

    const encoding = ct.TokenizerEncoding{
        .ids = @ptrCast(ids.ptr),
        .ids_len = 2,
        .tokens = null,
        .tokens_len = 0,
        .attention_mask = null,
        .type_ids = null,
        .special_tokens_mask = null,
        .offsets = null,
        .overflows = null,
        .overflow_count = 0,
    };

    const offsets = try computeOffsetsFromEncoding(allocator, &encoding, setup.tokenizer, "hello world");
    defer allocator.free(offsets);

    try std.testing.expectEqual(@as(u32, 0), offsets[0].start);
    try std.testing.expectEqual(@as(u32, 5), offsets[0].end);
    try std.testing.expectEqual(@as(u32, 6), offsets[1].start);
    try std.testing.expectEqual(@as(u32, 11), offsets[1].end);
}

test "computeOffsetsFromEncoding maps byte-level unk tokens with token surfaces to single-byte spans" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "version": "1.0",
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {
        \\      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
        \\      "H": 44, "i": 77, "b": 70, "y": 93, "e": 73
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
    const json_z = try allocator.dupeZ(u8, json);
    defer allocator.free(json_z);

    const tokenizer = pipeline.tokenizer_from_json_string(json_z.ptr) orelse return error.OutOfMemory;
    defer {
        tokenizer.destroy();
        allocator.destroy(tokenizer);
    }

    var encoding = std.mem.zeroes(ct.TokenizerEncoding);
    defer tok_encode.tokenizer_encoding_free_struct(&encoding);
    try std.testing.expectEqual(@as(c_int, 0), tok_encode.tokenizer_encode_struct_with_options(tokenizer, "Hi🎉bye", &encoding, .{}));

    const offsets = try computeOffsetsFromEncoding(allocator, &encoding, tokenizer, "Hi🎉bye");
    defer allocator.free(offsets);

    try std.testing.expectEqual(@as(usize, 9), offsets.len);
    for (offsets, 0..) |off, idx| {
        try std.testing.expectEqual(@as(u32, @intCast(idx)), off.start);
        try std.testing.expectEqual(@as(u32, @intCast(idx + 1)), off.end);
    }
}

test "computeOffsetsFromEncoding maps large nfkc expansion bytes to original scalar span" {
    const allocator = std.testing.allocator;
    var normalizer = std.mem.zeroes(ct.Normalizer);
    normalizer.nfkc = 1;
    var normalized = try normalize.normalize_text(&normalizer, "ﷺ");
    defer normalized.deinit();

    var tokenizer = std.mem.zeroes(ct.Tokenizer);
    tokenizer.pretokenizer.byte_level = 1;

    const token_count = normalized.text.len;
    const ids = try allocator.alloc(i32, token_count);
    defer allocator.free(ids);
    const token_ptrs = try allocator.alloc([*c]u8, token_count);
    defer {
        for (token_ptrs) |token_ptr| {
            if (token_ptr != null) allocator.free(std.mem.sliceTo(@as([*:0]u8, @ptrCast(token_ptr)), 0));
        }
        allocator.free(token_ptrs);
    }

    for (normalized.text, 0..) |byte_value, idx| {
        ids[idx] = @intCast(idx + 10);

        const codepoint: u21 = @intCast(utils.byteToUnicodeCodepoint(byte_value));
        var utf8_buf: [4]u8 = undefined;
        const utf8_len = try std.unicode.utf8Encode(codepoint, &utf8_buf);
        const token = try allocator.allocSentinel(u8, utf8_len, 0);
        @memcpy(token[0..utf8_len], utf8_buf[0..utf8_len]);
        token_ptrs[idx] = token.ptr;
    }

    var encoding = std.mem.zeroes(ct.TokenizerEncoding);
    encoding.ids = @ptrCast(ids.ptr);
    encoding.ids_len = token_count;
    encoding.tokens = @ptrCast(token_ptrs.ptr);
    encoding.tokens_len = token_count;

    const offsets = try computeOffsetsFromEncoding(allocator, &encoding, &tokenizer, "ﷺ");
    defer allocator.free(offsets);

    try std.testing.expect(offsets.len > 10);
    for (offsets, 0..) |off, idx| {
        try std.testing.expectEqual(@as(u32, 0), off.start);
        try std.testing.expectEqual(@as(u32, 3), off.end);
        _ = idx;
    }
}

test "encode requires integration testing" {
    // Requires fully initialized tokenizer with vocab, model, normalizer.
    // Integration tests: tests/tokenizer/test_*.py
}
