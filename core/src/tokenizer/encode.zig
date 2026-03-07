//! Tokenizer Encoding Pipeline
//!
//! Orchestrates the full encoding flow: normalization, pretokenization,
//! model encoding (BPE/WordPiece/Unigram), and post-processing.

const std = @import("std");
const ct = @import("c_types.zig");
const bpe = @import("bpe.zig");
const pipeline = @import("pipeline.zig");
const pretokenize = @import("pretokenize.zig");
const normalize = @import("normalize.zig");
const postprocess = @import("postprocess.zig");
const buffers = @import("encoding_buffers.zig");
const errors = @import("errors.zig");
const types = @import("types.zig");
const log = @import("../log.zig");
const parallel = @import("../system/parallel.zig");
const c = @cImport({
    @cInclude("utf8proc.h");
});

const Allocator = types.Allocator;
const Token = types.Token;
const Normalized = types.Normalized;
const Range = types.Range;

// Added token span
const AddedSpan = struct {
    start: usize,
    end: usize,
    at: *const ct.AddedToken,
};

fn isAddedTokenWordByte(byte_value: u8) bool {
    return std.ascii.isAlphabetic(byte_value) or std.ascii.isDigit(byte_value) or byte_value == '_';
}

fn isUtf8ContinuationByte(byte_value: u8) bool {
    return (byte_value & 0xC0) == 0x80;
}

fn isAddedTokenWordCodepoint(codepoint: c.utf8proc_int32_t) bool {
    return switch (c.utf8proc_category(codepoint)) {
        c.UTF8PROC_CATEGORY_LU,
        c.UTF8PROC_CATEGORY_LL,
        c.UTF8PROC_CATEGORY_LT,
        c.UTF8PROC_CATEGORY_LM,
        c.UTF8PROC_CATEGORY_LO,
        c.UTF8PROC_CATEGORY_MN,
        c.UTF8PROC_CATEGORY_MC,
        c.UTF8PROC_CATEGORY_ME,
        c.UTF8PROC_CATEGORY_ND,
        c.UTF8PROC_CATEGORY_NL,
        c.UTF8PROC_CATEGORY_NO,
        => true,
        else => false,
    };
}

fn isAddedTokenWordAt(input_bytes: []const u8, position: usize) bool {
    if (position >= input_bytes.len) return false;
    const first_byte = input_bytes[position];
    if (first_byte < 0x80) return isAddedTokenWordByte(first_byte);

    var codepoint: c.utf8proc_int32_t = 0;
    const consumed = c.utf8proc_iterate(@ptrCast(input_bytes.ptr + position), @intCast(input_bytes.len - position), &codepoint);
    if (consumed <= 0) return false;
    return isAddedTokenWordCodepoint(codepoint);
}

fn isAddedTokenWordBefore(input_bytes: []const u8, position: usize) bool {
    if (position == 0) return false;

    var start = position - 1;
    while (start > 0 and isUtf8ContinuationByte(input_bytes[start])) : (start -= 1) {}

    if (input_bytes[start] < 0x80) {
        return isAddedTokenWordByte(input_bytes[start]);
    }

    var codepoint: c.utf8proc_int32_t = 0;
    const consumed = c.utf8proc_iterate(@ptrCast(input_bytes.ptr + start), @intCast(input_bytes.len - start), &codepoint);
    if (consumed <= 0 or start + @as(usize, @intCast(consumed)) != position) {
        return false;
    }
    return isAddedTokenWordCodepoint(codepoint);
}

fn hasAddedTokenBoundary(input_bytes: []const u8, position: usize, content_len: usize) bool {
    const left_boundary_ok = position == 0 or !isAddedTokenWordBefore(input_bytes, position);
    const right_pos = position + content_len;
    const right_boundary_ok = right_pos == input_bytes.len or !isAddedTokenWordAt(input_bytes, right_pos);
    return left_boundary_ok and right_boundary_ok;
}

fn consumeAddedWhitespaceForward(input_bytes: []const u8, start: usize) usize {
    var cursor = start;
    while (cursor < input_bytes.len) {
        if (std.ascii.isWhitespace(input_bytes[cursor])) {
            cursor += 1;
            continue;
        }
        if (cursor + 3 <= input_bytes.len and
            input_bytes[cursor] == 0xE2 and
            input_bytes[cursor + 1] == 0x96 and
            input_bytes[cursor + 2] == 0x81)
        {
            cursor += 3;
            continue;
        }
        break;
    }
    return cursor;
}

fn leadingAsciiWhitespaceLen(input_bytes: []const u8) usize {
    var cursor: usize = 0;
    while (cursor < input_bytes.len and std.ascii.isWhitespace(input_bytes[cursor])) : (cursor += 1) {}
    return cursor;
}

fn trailingAsciiWhitespaceStart(input_bytes: []const u8) usize {
    var cursor = input_bytes.len;
    while (cursor > 0 and std.ascii.isWhitespace(input_bytes[cursor - 1])) : (cursor -= 1) {}
    return cursor;
}

fn findNormalizedCursorForSourceEnd(normalized: *const Normalized, start_cursor: usize, source_end: usize) usize {
    if (source_end == 0) return start_cursor;
    var cursor = start_cursor;
    while (cursor < normalized.text.len and cursor < normalized.map_end.len) : (cursor += 1) {
        const mapped_end = normalized.map_end[cursor];
        if (mapped_end >= @as(i32, @intCast(source_end))) {
            return cursor + 1;
        }
    }
    return normalized.text.len;
}

// ============================================================================
// ADDED TOKENS COLLECTION
// ============================================================================

fn matches_added_token_boundaries(added_token: *const ct.AddedToken, input_bytes: []const u8, position: usize) bool {
    if (added_token.content == null) return false;
    const content = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(added_token.content.?)), 0);
    if (position + content.len > input_bytes.len) return false;
    if (!std.mem.eql(u8, input_bytes[position..][0..content.len], content)) return false;

    if (added_token.single_word != 0) {
        if (!hasAddedTokenBoundary(input_bytes, position, content.len)) return false;
    }
    return true;
}

fn collect_added_spans(tokenizer: *ct.Tokenizer, normalized: *const Normalized, original_input: []const u8) ?std.ArrayListUnmanaged(AddedSpan) {
    // Fast path: no added tokens → no spans possible.
    // Skips the O(normalized.len × n_added) byte-by-byte walk.
    if (tokenizer.added == null) {
        return std.ArrayListUnmanaged(AddedSpan){};
    }

    var spans = std.ArrayListUnmanaged(AddedSpan){};

    log.trace("tokenizer", "collect_added_spans", .{
        .normalized_len = normalized.text.len,
        .has_added_tokens = tokenizer.added != null,
    }, @src());

    var cursor: usize = 0;
    while (cursor < normalized.text.len) {
        var best_match: ?AddedSpan = null;
        var best_span_len: usize = 0;

        var added_iter = tokenizer.added;
        while (added_iter) |added_token| {
            if (added_token.content == null) {
                added_iter = added_token.next;
                continue;
            }
            const content = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(added_token.content.?)), 0);
            if (content.len == 0) {
                added_iter = added_token.next;
                continue;
            }

            const text = if (added_token.normalized != 0) normalized.text else original_input;
            const text_pos_opt: ?usize = if (added_token.normalized != 0) cursor else blk: {
                // For non-normalized tokens, map back to original position
                if (cursor < normalized.map.len and normalized.map[cursor] >= 0) {
                    break :blk @as(usize, @intCast(normalized.map[cursor]));
                }
                // Position doesn't map to original (e.g., prepended chars) - skip
                break :blk null;
            };
            if (text_pos_opt == null) {
                added_iter = added_token.next;
                continue;
            }
            const text_len = text.len;
            const text_pos = text_pos_opt.?;
            const content_pos = if (added_token.lstrip != 0) consumeAddedWhitespaceForward(text, text_pos) else text_pos;

            if (content_pos + content.len > text_len) {
                added_iter = added_token.next;
                continue;
            }
            if (!std.mem.eql(u8, text[content_pos..][0..content.len], content)) {
                added_iter = added_token.next;
                continue;
            }
            if (!matches_added_token_boundaries(added_token, text, content_pos)) {
                added_iter = added_token.next;
                continue;
            }
            var text_end = content_pos + content.len;
            if (added_token.rstrip != 0) {
                text_end = consumeAddedWhitespaceForward(text, text_end);
            }

            const span_end = if (added_token.normalized != 0)
                cursor + (text_end - text_pos)
            else
                findNormalizedCursorForSourceEnd(normalized, cursor, text_end);

            const span = AddedSpan{
                .start = cursor,
                .end = span_end,
                .at = added_token,
            };
            const span_len = span.end - span.start;
            if (span_len > best_span_len) {
                best_match = span;
                best_span_len = span_len;
            }
            added_iter = added_token.next;
        }

        if (best_match) |matched| {
            spans.append(Allocator, matched) catch {
                spans.deinit(Allocator);
                return null;
            };
            cursor = matched.end;
        } else {
            cursor += 1;
        }
    }

    return spans;
}

// ============================================================================
// ENCODING
// ============================================================================

fn truncateEncoding(encoding: *ct.TokenizerEncoding, new_length: usize) void {
    if (new_length >= encoding.ids_len) return;
    if (encoding.tokens) |tokens_ptr| {
        const tokens_slice: [*][*c]u8 = @ptrCast(tokens_ptr);
        for (new_length..encoding.tokens_len) |token_idx| {
            if (tokens_slice[token_idx]) |token_ptr| Allocator.free(std.mem.span(@as([*:0]u8, @ptrCast(token_ptr))));
        }
    }
    encoding.ids_len = new_length;
    encoding.tokens_len = new_length;
}

fn pairSpecialTokenCount(tokenizer: *const ct.Tokenizer) usize {
    if (tokenizer.postproc.add_special == 0) return 0;
    const cls: usize = if (tokenizer.postproc.cls_id >= 0) 1 else 0;
    const sep: usize = if (tokenizer.postproc.sep_id >= 0) 1 else 0;
    // Pair: CLS? + SEP? after A + SEP? pair-between + content B + SEP? after B
    // Non-pair: CLS? + content A + SEP? after A + content B + SEP? after B
    return cls + sep * (if (tokenizer.postproc.pair != 0) @as(usize, 3) else 2);
}

fn truncationSpecialTokenCount(tokenizer: *const ct.Tokenizer) usize {
    if (tokenizer.postproc.add_special == 0) return 0;
    const cls: usize = if (tokenizer.postproc.cls_id >= 0) 1 else 0;
    const sep: usize = if (tokenizer.postproc.sep_id >= 0) 1 else 0;
    return cls + sep * (if (tokenizer.postproc.pair != 0) @as(usize, 2) else 1);
}

fn appendSpecial(
    buffers_out: *buffers.Buffers,
    write_index: *usize,
    token_id: i32,
    token_bytes: []const u8,
    type_id: i32,
    offset: ct.Offset,
) bool {
    return postprocess.appendSpecialToken(
        buffers_out.ids,
        buffers_out.tokens,
        buffers_out.attention_mask,
        buffers_out.type_ids,
        buffers_out.special,
        buffers_out.offsets,
        write_index,
        token_id,
        token_bytes,
        type_id,
        offset.start,
        offset.end,
    );
}

fn appendCls(tokenizer: *const ct.Tokenizer, buffers_out: *buffers.Buffers, write_index: *usize, offset: ct.Offset) bool {
    if (tokenizer.postproc.cls_id < 0) return true;
    const cls_str = std.mem.sliceTo(&tokenizer.postproc.cls_token, 0);
    return appendSpecial(buffers_out, write_index, tokenizer.postproc.cls_id, cls_str, 0, offset);
}

fn appendSep(tokenizer: *const ct.Tokenizer, buffers_out: *buffers.Buffers, write_index: *usize, type_id: i32, offset: ct.Offset) bool {
    if (tokenizer.postproc.sep_id < 0) return true;
    const sep_str = std.mem.sliceTo(&tokenizer.postproc.sep_token, 0);
    return appendSpecial(buffers_out, write_index, tokenizer.postproc.sep_id, sep_str, type_id, offset);
}

fn applyPairTruncation(tokenizer: *const ct.Tokenizer, first_encoding: *ct.TokenizerEncoding, second_encoding: *ct.TokenizerEncoding) void {
    const special_token_count = truncationSpecialTokenCount(tokenizer);
    const max_length_signed: i64 = @as(i64, tokenizer.truncation.max_length) - @as(i64, @intCast(special_token_count));
    const max_total_len: usize = if (max_length_signed < 0) 0 else @intCast(max_length_signed);

    var first_len = first_encoding.ids_len;
    var second_len = second_encoding.ids_len;

    while (first_len + second_len > max_total_len) {
        if (tokenizer.truncation.strategy == ct.TruncationStrategy.only_first or second_len == 0 or first_len >= second_len) {
            if (first_len == 0) break;
            first_len -= 1;
        } else {
            if (second_len == 0) break;
            second_len -= 1;
        }
    }

    truncateEncoding(first_encoding, first_len);
    truncateEncoding(second_encoding, second_len);
}

fn assemblePairEncoding(tokenizer: *const ct.Tokenizer, first_encoding: *ct.TokenizerEncoding, second_encoding: *ct.TokenizerEncoding, out_encoding: *ct.TokenizerEncoding) c_int {
    const default_offset: ct.Offset = .{ .start = 0, .end = 0 };
    const total_len = first_encoding.ids_len + second_encoding.ids_len + pairSpecialTokenCount(tokenizer);

    var buffers_out = buffers.allocBuffers(total_len) catch return -1;
    errdefer buffers_out.deinit();

    var write_index: usize = 0;

    // CLS
    if (tokenizer.postproc.add_special != 0) {
        if (!appendCls(tokenizer, &buffers_out, &write_index, default_offset)) return -1;
    }

    buffers.fillFromEncoding(&buffers_out, &write_index, first_encoding, 0, default_offset);

    // SEP after A
    if (tokenizer.postproc.add_special != 0) {
        if (!appendSep(tokenizer, &buffers_out, &write_index, 0, default_offset)) return -1;
        if (tokenizer.postproc.pair != 0) {
            if (!appendSep(tokenizer, &buffers_out, &write_index, 0, default_offset)) return -1;
        }
    }

    buffers.fillFromEncoding(&buffers_out, &write_index, second_encoding, 1, default_offset);

    // SEP after B
    if (tokenizer.postproc.add_special != 0) {
        if (!appendSep(tokenizer, &buffers_out, &write_index, 1, default_offset)) return -1;
    }

    buffers.initEncoding(out_encoding, &buffers_out, write_index, null, 0);
    return 0;
}

pub fn tokenizer_encoding_free_struct(encoding: *ct.TokenizerEncoding) void {
    if (encoding.overflows) |overflow_ptr| {
        const overflow_slice: [*]ct.TokenizerEncoding = @ptrCast(overflow_ptr);
        for (0..encoding.overflow_count) |overflow_index| {
            tokenizer_encoding_free_struct(&overflow_slice[overflow_index]);
        }
        Allocator.free(overflow_slice[0..encoding.overflow_count]);
    }
    if (encoding.tokens) |tokens_ptr| {
        const tokens_slice: [*][*c]u8 = @ptrCast(tokens_ptr);
        for (0..encoding.tokens_len) |token_idx| {
            if (tokens_slice[token_idx]) |token_ptr| Allocator.free(std.mem.span(@as([*:0]u8, @ptrCast(token_ptr))));
        }
        Allocator.free(tokens_slice[0..encoding.tokens_len]);
    }
    if (encoding.attention_mask) |mask_ptr| {
        const slice: [*]i32 = @ptrCast(mask_ptr);
        Allocator.free(slice[0..encoding.ids_len]);
    }
    if (encoding.type_ids) |type_ids_ptr| {
        const slice: [*]i32 = @ptrCast(type_ids_ptr);
        Allocator.free(slice[0..encoding.ids_len]);
    }
    if (encoding.special_tokens_mask) |special_ptr| {
        const slice: [*]i32 = @ptrCast(special_ptr);
        Allocator.free(slice[0..encoding.ids_len]);
    }
    if (encoding.offsets) |offsets_ptr| {
        const slice: [*]ct.Offset = @ptrCast(offsets_ptr);
        Allocator.free(slice[0..encoding.ids_len]);
    }
    if (encoding.ids) |ids_ptr| {
        const slice: [*]i32 = @ptrCast(ids_ptr);
        Allocator.free(slice[0..encoding.ids_len]);
    }
    encoding.* = std.mem.zeroes(ct.TokenizerEncoding);
}

/// Look up an added token's content string by ID.
pub fn findAddedTokenContentById(tokenizer: *ct.Tokenizer, id: i32) ?[]const u8 {
    var added_iter = tokenizer.added;
    while (added_iter) |at| : (added_iter = at.next) {
        if (at.id == id) {
            if (at.content) |content_ptr| {
                return std.mem.sliceTo(@as([*:0]const u8, @ptrCast(content_ptr)), 0);
            }
        }
    }
    return null;
}

/// Accumulator for building token encoding results.
/// Token strings are not tracked here — they are resolved lazily from IDs
/// by consumers that need them (e.g. offset computation via idToToken).
///
/// Special token flags use a sparse representation: only positions where
/// special == 1 are stored (typically 0-10 per input). This eliminates
/// per-word zero-append overhead (161K appendNTimes calls for 1MB input).
const EncodeAccum = struct {
    ids: std.ArrayListUnmanaged(i32) = .{},
    /// Sparse positions where special == 1 (added tokens only).
    special_positions: std.ArrayListUnmanaged(u32) = .{},

    fn deinit(self: *EncodeAccum) void {
        self.special_positions.deinit(Allocator);
        self.ids.deinit(Allocator);
    }

    fn appendAdded(self: *EncodeAccum, added_token: *const ct.AddedToken) !void {
        if (added_token.special != 0) {
            try self.special_positions.append(Allocator, @intCast(self.ids.items.len));
        }
        try self.ids.append(Allocator, added_token.id);
    }

    /// Append tokens from cached IDs (word cache hit path).
    /// BPE-produced tokens are never special — no tracking needed.
    fn appendCachedIds(self: *EncodeAccum, cached_ids: []const i32) !void {
        try self.ids.appendSlice(Allocator, cached_ids);
    }

    fn appendEncoding(self: *EncodeAccum, encoding: *const ct.TokenizerEncoding, added_head: ?*ct.AddedToken) !void {
        if (encoding.ids == null) return;
        const ids_ptr: [*]i32 = @ptrCast(encoding.ids.?);

        for (0..encoding.ids_len) |id_index| {
            if (checkSpecial(added_head, ids_ptr[id_index]) != 0) {
                try self.special_positions.append(Allocator, @intCast(self.ids.items.len));
            }
            try self.ids.append(Allocator, ids_ptr[id_index]);
        }
    }

    fn checkSpecial(added_head: ?*ct.AddedToken, id: i32) i32 {
        var added_iter = added_head;
        while (added_iter) |token| : (added_iter = token.next) {
            if (token.special != 0 and token.id == id) return 1;
        }
        return 0;
    }

    fn buildOutput(self: *EncodeAccum, out_encoding: *ct.TokenizerEncoding) !void {
        const token_count = self.ids.items.len;
        if (token_count == 0) {
            out_encoding.* = std.mem.zeroes(ct.TokenizerEncoding);
            return;
        }

        // Transfer ids ownership directly (avoids alloc + memcpy for the
        // largest buffer). Allocate remaining buffers individually.
        var buffers_out = buffers.Buffers{
            .ids = try self.ids.toOwnedSlice(Allocator),
            .tokens = &.{},
            .attention_mask = &.{},
            .type_ids = &.{},
            .special = &.{},
            .offsets = &.{},
        };
        errdefer buffers_out.deinit();

        buffers_out.tokens = try Allocator.alloc([*c]u8, token_count);
        @memset(buffers_out.tokens, null);
        buffers_out.attention_mask = try Allocator.alloc(i32, token_count);
        @memset(buffers_out.attention_mask, 1);
        buffers_out.type_ids = try Allocator.alloc(i32, token_count);
        @memset(buffers_out.type_ids, 0);
        buffers_out.special = try Allocator.alloc(i32, token_count);
        @memset(buffers_out.special, 0);
        for (self.special_positions.items) |pos| {
            buffers_out.special[pos] = 1;
        }
        buffers_out.offsets = try Allocator.alloc(ct.Offset, token_count);
        @memset(std.mem.sliceAsBytes(buffers_out.offsets), 0);

        buffers.initEncoding(out_encoding, &buffers_out, token_count, null, 0);

        // ids ownership transferred via toOwnedSlice; just free sparse positions
        self.special_positions.deinit(Allocator);
        self.* = .{};
    }
};

fn encodeLiteralBpeBytes(tokenizer: *ct.Tokenizer, bytes: []const u8, accumulator: *EncodeAccum) !void {
    if (bytes.len == 0 or tokenizer.type != .bpe) return;
    const model_ptr = tokenizer.model orelse return error.OutOfMemory;
    const model: *bpe.BpeModel = @ptrCast(@alignCast(model_ptr));
    try model.encodeWordDirect(tokenizer, bytes, &accumulator.ids);
}

/// Check if input exactly matches an added token
fn findExactAddedToken(tokenizer: *ct.Tokenizer, input: []const u8) ?*const ct.AddedToken {
    var added_iter = tokenizer.added;
    while (added_iter) |added_token| {
        if (added_token.content) |content_ptr| {
            const content = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(content_ptr)), 0);
            if (std.mem.eql(u8, content, input)) {
                return added_token;
            }
        }
        added_iter = added_token.next;
    }
    return null;
}

fn encode_internal(tokenizer: *ct.Tokenizer, input: []const u8, out: *ct.TokenizerEncoding, apply_postprocess: bool) c_int {
    return encode_internal_impl(tokenizer, input, out, apply_postprocess) catch |err| {
        errors.tokenizer_set_error_internal(tokenizer, "Encoding failed: {}", .{err});
        return -1;
    };
}

fn encode_internal_impl(tokenizer: *ct.Tokenizer, input: []const u8, out: *ct.TokenizerEncoding, apply_postprocess: bool) !c_int {
    // Check if input exactly matches an added token - if so, skip normalization
    // Empty input returns empty output (no tokens), but still apply post-processor
    // so that models with BOS/EOS (BERT, Llama) produce [CLS]+[SEP] or <s>+</s>.
    if (input.len == 0 and tokenizer.normalizer.prepend == null and tokenizer.normalizer.replace_pattern == null) {
        out.* = std.mem.zeroes(ct.TokenizerEncoding);
        if (apply_postprocess and tokenizer.postproc.add_special != 0) {
            if (postprocess.postprocess_single(&tokenizer.postproc, out) != 0) {
                return error.OutOfMemory;
            }
        }
        return 0;
    }

    // SentencePiece-style prepend/replace can synthesize text from empty input.
    // There is no original source span to preserve here, so bypass added-token
    // span collection and encode the normalized segment directly.
    if (input.len == 0) {
        var normalized = try normalize.normalize_text(&tokenizer.normalizer, input);
        defer normalized.deinit();

        if (normalized.text.len == 0) {
            out.* = std.mem.zeroes(ct.TokenizerEncoding);
        } else {
            var accum = EncodeAccum{};
            errdefer accum.deinit();
            try encodeSegment(tokenizer, normalized.text, 0, &accum);
            try accum.buildOutput(out);
        }

        if (apply_postprocess and tokenizer.postproc.add_special != 0) {
            if (postprocess.postprocess_single(&tokenizer.postproc, out) != 0) {
                tokenizer_encoding_free_struct(out);
                return error.OutOfMemory;
            }
        }
        return 0;
    }

    // This prevents SentencePiece prepend from affecting special tokens like </s>
    if (findExactAddedToken(tokenizer, input)) |at| {
        var accum = EncodeAccum{};
        errdefer accum.deinit();
        try accum.appendAdded(at);
        try accum.buildOutput(out);
        // Still apply post-processing (CLS/SEP) for non-special added tokens
        if (apply_postprocess and tokenizer.postproc.add_special != 0) {
            if (postprocess.postprocess_single(&tokenizer.postproc, out) != 0) {
                tokenizer_encoding_free_struct(out);
                return error.OutOfMemory;
            }
        }
        return 0;
    }

    // Step 1: Normalize
    var normalized = try normalize.normalize_text(&tokenizer.normalizer, input);
    defer normalized.deinit();

    // Add prefix space if needed.
    // Skip for Metaspace pretokenizer — Metaspace adds ▁ (not space) in encodeSegment.
    if (tokenizer.pretokenizer.add_prefix_space != 0 and tokenizer.pretokenizer.metaspace == 0 and (normalized.text.len == 0 or normalized.text[0] != ' ')) {
        try normalize.addPrefixSpace(&normalized);
    }

    try encodeNormalized(tokenizer, input, &normalized, out);

    // Step 4: Post-processing
    if (apply_postprocess and tokenizer.postproc.add_special != 0) {
        if (postprocess.postprocess_single(&tokenizer.postproc, out) != 0) {
            tokenizer_encoding_free_struct(out);
            return error.OutOfMemory;
        }
    }

    return 0;
}

fn encodeNormalized(tokenizer: *ct.Tokenizer, input: []const u8, normalized: *const Normalized, out: *ct.TokenizerEncoding) !void {
    // Step 2: Collect added token spans
    var spans = collect_added_spans(tokenizer, normalized, input) orelse return error.OutOfMemory;
    defer spans.deinit(Allocator);

    // Step 3: Encode segments
    var accum = EncodeAccum{};
    errdefer accum.deinit();

    // Pre-allocate: ~4 bytes per token is typical for BPE
    const estimated_tokens = normalized.text.len / 4;
    try accum.ids.ensureTotalCapacity(Allocator, estimated_tokens);

    var cursor: usize = 0;
    var span_index: usize = 0;

    // Check if SentencePiece prepend is enabled
    const has_sp_prepend = tokenizer.normalizer.prepend != null;
    const prepend_bytes = if (tokenizer.normalizer.prepend) |p| std.mem.sliceTo(p, 0) else "";

    // Track whether the initial prepend ▁ was skipped and needs re-attaching.
    // This is a one-shot flag: once consumed by the first non-special segment, it stays false.
    var needs_prepend = false;
    const preserve_literal_gap_whitespace = tokenizer.type == .bpe and
        tokenizer.pretokenizer.metaspace == 0 and
        tokenizer.normalizer.prepend == null;

    // Skip the prepend if input starts with an added token
    // (the prepend was added by normalization but shouldn't apply to added tokens)
    if (has_sp_prepend and spans.items.len > 0 and spans.items[0].start == prepend_bytes.len) {
        // First added token is right after the prepend - skip the prepend
        cursor = prepend_bytes.len;
        needs_prepend = true; // Re-attach to first non-special segment
    }

    while (cursor < normalized.text.len) {
        const previous_was_added = span_index > 0 and cursor == spans.items[span_index - 1].end;
        // Handle added token spans
        if (span_index < spans.items.len and cursor == spans.items[span_index].start) {
            const added_span = spans.items[span_index];
            try accum.appendAdded(added_span.at);
            cursor = added_span.end;
            span_index += 1;
            // Re-arm prepend for the next non-special segment: SentencePiece
            // treats each span after a special token as a new "sentence start".
            if (has_sp_prepend) needs_prepend = true;
            continue;
        }

        // Encode segment up to next span
        const next_span_start = if (span_index < spans.items.len) spans.items[span_index].start else normalized.text.len;
        if (next_span_start <= cursor) {
            cursor += 1;
            continue;
        }

        const segment = normalized.text[cursor..next_span_start];
        var segment_start: usize = 0;
        var segment_end: usize = segment.len;

        if (preserve_literal_gap_whitespace and previous_was_added) {
            const leading_ws_len = leadingAsciiWhitespaceLen(segment);
            if (leading_ws_len > 0) {
                try encodeLiteralBpeBytes(tokenizer, segment[0..leading_ws_len], &accum);
                segment_start = leading_ws_len;
            }
        }

        if (preserve_literal_gap_whitespace and span_index < spans.items.len) {
            const trailing_ws_start = trailingAsciiWhitespaceStart(segment[segment_start..]) + segment_start;
            if (trailing_ws_start < segment_end) {
                segment_end = trailing_ws_start;
            }
        }

        const interior = segment[segment_start..segment_end];
        const should_prepend = needs_prepend and prepend_bytes.len > 0;
        if (interior.len > 0) {
            needs_prepend = false;
            try encodeSegmentMaybePrepended(tokenizer, interior, cursor + segment_start, should_prepend, prepend_bytes, &accum);
        }
        if (segment_end < segment.len and preserve_literal_gap_whitespace) {
            try encodeLiteralBpeBytes(tokenizer, segment[segment_end..], &accum);
            needs_prepend = false;
        } else if (interior.len == 0 and should_prepend) {
            needs_prepend = false;
        }
        cursor = next_span_start;
    }

    // Build output
    try accum.buildOutput(out);
}

fn encodeSegmentMaybePrepended(
    tokenizer: *ct.Tokenizer,
    segment: []const u8,
    base_offset: usize,
    should_prepend: bool,
    prepend_bytes: []const u8,
    accumulator: *EncodeAccum,
) !void {
    if (!should_prepend) {
        try encodeSegment(tokenizer, segment, base_offset, accumulator);
        return;
    }

    // Prepend ▁ to the segment and encode together.
    const combined_len = prepend_bytes.len + segment.len;
    var combined_bytes = try Allocator.alloc(u8, combined_len);
    defer Allocator.free(combined_bytes);
    @memcpy(combined_bytes[0..prepend_bytes.len], prepend_bytes);
    @memcpy(combined_bytes[prepend_bytes.len..], segment);
    try encodeSegment(tokenizer, combined_bytes, base_offset, accumulator);
}

fn encodeSegment(tokenizer: *ct.Tokenizer, segment: []const u8, base_offset: usize, accumulator: *EncodeAccum) !void {
    const is_sentencepiece_bpe = tokenizer.type == ct.ModelType.bpe and
        tokenizer.pretokenizer.regex_split != 0 and
        tokenizer.pretokenizer.byte_level == 0 and
        tokenizer.normalizer.prepend != null;
    const is_byte_level_bpe = tokenizer.type == ct.ModelType.bpe and tokenizer.pretokenizer.byte_level != 0;
    const is_metaspace = tokenizer.pretokenizer.metaspace != 0;
    const split_bpe_encode = tokenizer.type == ct.ModelType.bpe and
        tokenizer.pretokenizer.byte_level == 0 and
        tokenizer.pretokenizer.metaspace == 0 and
        (tokenizer.pretokenizer.whitespace != 0 or tokenizer.pretokenizer.punctuation != 0);
    const per_word_encode = is_sentencepiece_bpe or is_byte_level_bpe or is_metaspace or split_bpe_encode;

    // Skip byte-level encoding in pretokenize when the BPE model can handle
    // raw bytes directly via orig_byte_vocab_ids. This eliminates the
    // applyByteLevel pass (arena allocs + UTF-8 encoding per byte).
    const skip_byte_level = is_byte_level_bpe and tokenizer.pretokenizer.is_sequence == 0 and
        if (tokenizer.model) |model_ptr| blk: {
        const model: *bpe.BpeModel = @ptrCast(@alignCast(model_ptr));
        break :blk model.use_raw_byte_init;
    } else false;

    // Overlapping-chunk parallelism: split text, parallelize both regex + BPE,
    // deduplicate at merge using word offsets vs midpoints.
    // Skip parallelism when the segment has no whitespace: without word
    // boundaries, the pretokenizer produces one word per chunk and the
    // overlap-based ownership deduplication silently drops all non-first chunks.
    if (per_word_encode and segment.len >= chunk_parallel_threshold and
        std.mem.indexOfScalar(u8, segment, ' ') != null)
    {
        const pool = parallel.global();
        if (pool.n_threads > 1) {
            return encodeSegmentOverlapped(tokenizer, segment, base_offset, accumulator, pool, is_sentencepiece_bpe, is_metaspace, skip_byte_level);
        }
    }

    // Sequential path
    var pretokenized = try pretokenize.pretokenize(&tokenizer.pretokenizer, segment, .{ .start = base_offset, .end = base_offset + segment.len }, skip_byte_level);
    defer pretokenized.deinit();

    if (per_word_encode) {
        // Per-segment word cache (avoids redundant BPE/model merges for repeated words)
        var word_cache = WordCache.init();
        defer word_cache.deinit();

        // Byte-level BPE fast path: hoist model pointer, skip type checks,
        // and add direct-lookup paths for 1-2 byte words.
        if (is_byte_level_bpe) {
            if (tokenizer.model) |model_ptr| {
                const model: *bpe.BpeModel = @ptrCast(@alignCast(model_ptr));
                for (pretokenized.tokens.items) |token_item| {
                    try encodeWordBpe(model, tokenizer, token_item.sliceConst(), accumulator, &word_cache);
                }
                return;
            }
        }

        for (pretokenized.tokens.items, 0..) |token_item, token_index| {
            const add_sp_prefix = if (is_metaspace)
                false
            else
                (is_sentencepiece_bpe and token_index > 0);
            try encodeWord(tokenizer, token_item.sliceConst(), add_sp_prefix, accumulator, &word_cache);
        }
    } else {
        try encodeCombined(tokenizer, segment, base_offset, pretokenized.tokens.items, pretokenized.ranges.items, accumulator);
    }
}

fn encodeBpePretokenizedWithGaps(
    tokenizer: *ct.Tokenizer,
    segment: []const u8,
    base_offset: usize,
    token_items: []Token,
    token_ranges: []const Range,
    accumulator: *EncodeAccum,
) !void {
    if (token_items.len == 0) return;

    var cursor: usize = 0;
    var word_cache = WordCache.init();
    defer word_cache.deinit();

    for (token_items, token_ranges) |token_item, token_range| {
        const rel_start = token_range.start - base_offset;
        const rel_end = token_range.end - base_offset;

        if (rel_start > cursor) {
            try encodeLiteralBpeBytes(tokenizer, segment[cursor..rel_start], accumulator);
        }

        try encodeWord(tokenizer, token_item.sliceConst(), false, accumulator, &word_cache);
        cursor = rel_end;
    }

    if (cursor < segment.len) {
        try encodeLiteralBpeBytes(tokenizer, segment[cursor..], accumulator);
    }
}

// ============================================================================
// WORD BPE CACHE
// ============================================================================

/// Per-chunk word BPE cache. Maps pretokenized word bytes to their BPE token IDs.
/// Thread-safe by construction: each chunk/segment has its own cache instance.
const WordCache = struct {
    map: std.StringHashMapUnmanaged([]const i32) = .{},
    arena: std.heap.ArenaAllocator,

    fn init() WordCache {
        return .{
            .map = .{},
            .arena = std.heap.ArenaAllocator.init(Allocator),
        };
    }

    fn deinit(self: *WordCache) void {
        self.map.deinit(Allocator);
        self.arena.deinit();
    }

    fn get(self: *const WordCache, word: []const u8) ?[]const i32 {
        return self.map.get(word);
    }

    fn put(self: *WordCache, word: []const u8, ids: []const i32) void {
        if (self.map.contains(word)) return;
        const arena_alloc = self.arena.allocator();
        const key_copy = arena_alloc.dupe(u8, word) catch return;
        const ids_copy = arena_alloc.dupe(i32, ids) catch return;
        self.map.put(Allocator, key_copy, ids_copy) catch return;
    }

    /// Fast insert for BPE path: key slice outlives cache, skip key dupe.
    /// Caller guarantees word slice remains valid until cache deinit.
    fn putBorrowed(self: *WordCache, word: []const u8, ids: []const i32) void {
        const arena_alloc = self.arena.allocator();
        const ids_copy = arena_alloc.dupe(i32, ids) catch return;
        self.map.put(Allocator, word, ids_copy) catch return;
    }
};

// ============================================================================
// OVERLAPPING-CHUNK PARALLEL ENCODING
// ============================================================================

// Minimum segment size (bytes) to justify parallel dispatch.
const chunk_parallel_threshold: usize = 4096;
// Overlap per side between adjacent chunks. Must exceed the longest possible
// regex match so that words spanning a split point are fully captured by at
// least one chunk. 256 bytes is generous for all common tokenizer patterns.
const overlap_bytes: usize = 256;
// Minimum useful chunk size (excluding overlap).
const min_chunk_bytes: usize = 512;
// Must match parallel.zig FLOATS_PER_CACHE_LINE.
const CACHE_LINE_ITEMS: usize = 16;
// Stack limit for chunk metadata arrays (supports up to 64 threads).
const MAX_CHUNKS: usize = 64;

/// Per-chunk encoding result with word boundary tracking for selective merge.
const ChunkResult = struct {
    accum: EncodeAccum = .{},
    /// Number of tokens produced by each word (parallel to word_offsets).
    word_token_counts: std.ArrayListUnmanaged(u32) = .{},
    /// Start offset of each word in segment-relative coordinates.
    word_offsets: std.ArrayListUnmanaged(usize) = .{},

    fn deinit(self: *ChunkResult) void {
        self.accum.deinit();
        self.word_token_counts.deinit(Allocator);
        self.word_offsets.deinit(Allocator);
    }
};

const ChunkContext = struct {
    tokenizer: *ct.Tokenizer,
    segment: []const u8,
    base_offset: usize,
    // Per-chunk byte ranges (segment-relative). Length = n_chunks.
    chunk_starts: []const usize,
    chunk_ends: []const usize,
    // Per-chunk owned ranges for midpoint deduplication. Length = n_chunks.
    own_starts: []const usize,
    own_ends: []const usize,
    results: []ChunkResult,
    n_chunks: usize,
    is_sentencepiece_bpe: bool,
    is_metaspace: bool,
    skip_byte_level: bool,
    had_error: std.atomic.Value(bool),
};

fn chunkWorker(start: usize, end: usize, ctx: *ChunkContext) void {
    // Map virtual items to chunk indices (CACHE_LINE_ITEMS virtual items per chunk)
    const first_chunk = start / CACHE_LINE_ITEMS;
    const last_chunk = @min((end + CACHE_LINE_ITEMS - 1) / CACHE_LINE_ITEMS, ctx.n_chunks);

    for (first_chunk..last_chunk) |chunk_idx| {
        if (ctx.had_error.load(.monotonic)) return;
        processChunk(ctx, chunk_idx);
    }
}

fn processChunk(ctx: *ChunkContext, chunk_idx: usize) void {
    const chunk_start = ctx.chunk_starts[chunk_idx];
    const chunk_end = ctx.chunk_ends[chunk_idx];
    const chunk = ctx.segment[chunk_start..chunk_end];
    if (chunk.len == 0) return;

    const chunk_base = ctx.base_offset + chunk_start;
    var pretokenized = pretokenize.pretokenize(
        &ctx.tokenizer.pretokenizer,
        chunk,
        .{ .start = chunk_base, .end = chunk_base + chunk.len },
        ctx.skip_byte_level,
    ) catch {
        ctx.had_error.store(true, .release);
        return;
    };
    defer pretokenized.deinit();

    const result = &ctx.results[chunk_idx];
    const words = pretokenized.tokens.items;
    const ranges = pretokenized.ranges.items;

    var word_cache = WordCache.init();
    defer word_cache.deinit();

    for (words, ranges, 0..) |token_item, range, token_index| {
        const tokens_before: u32 = @intCast(result.accum.ids.items.len);

        const add_sp_prefix = if (ctx.is_metaspace)
            false
        else
            (ctx.is_sentencepiece_bpe and token_index > 0);

        encodeWord(ctx.tokenizer, token_item.sliceConst(), add_sp_prefix, &result.accum, &word_cache) catch {
            ctx.had_error.store(true, .release);
            return;
        };

        const tokens_after: u32 = @intCast(result.accum.ids.items.len);
        // Segment-relative word offset
        const word_offset = range.start - ctx.base_offset;

        result.word_token_counts.append(Allocator, tokens_after - tokens_before) catch {
            ctx.had_error.store(true, .release);
            return;
        };
        result.word_offsets.append(Allocator, word_offset) catch {
            ctx.had_error.store(true, .release);
            return;
        };
    }
}

fn encodeSegmentOverlapped(
    tokenizer: *ct.Tokenizer,
    segment: []const u8,
    base_offset: usize,
    accumulator: *EncodeAccum,
    pool: *parallel.ThreadPool,
    is_sentencepiece_bpe: bool,
    is_metaspace: bool,
    skip_byte_level: bool,
) !void {
    const n_threads = pool.n_threads;
    const n_chunks = @min(n_threads, @max(1, segment.len / min_chunk_bytes));

    if (n_chunks <= 1) {
        // Fall back to sequential
        var pretokenized = try pretokenize.pretokenize(&tokenizer.pretokenizer, segment, .{ .start = base_offset, .end = base_offset + segment.len }, skip_byte_level);
        defer pretokenized.deinit();
        var word_cache = WordCache.init();
        defer word_cache.deinit();
        for (pretokenized.tokens.items, 0..) |token_item, token_index| {
            const add_sp_prefix = if (is_metaspace) false else (is_sentencepiece_bpe and token_index > 0);
            try encodeWord(tokenizer, token_item.sliceConst(), add_sp_prefix, accumulator, &word_cache);
        }
        return;
    }

    // Compute split points, chunk byte ranges (with overlap), and owned ranges
    var chunk_starts_buf: [MAX_CHUNKS]usize = undefined;
    var chunk_ends_buf: [MAX_CHUNKS]usize = undefined;
    var own_starts_buf: [MAX_CHUNKS]usize = undefined;
    var own_ends_buf: [MAX_CHUNKS]usize = undefined;

    for (0..n_chunks) |i| {
        // Owned range: [split[i], split[i+1])
        const own_start = segment.len * i / n_chunks;
        const own_end = segment.len * (i + 1) / n_chunks;
        own_starts_buf[i] = own_start;
        own_ends_buf[i] = own_end;
        // Byte range with overlap on both sides
        chunk_starts_buf[i] = if (own_start > overlap_bytes) own_start - overlap_bytes else 0;
        chunk_ends_buf[i] = @min(own_end + overlap_bytes, segment.len);
    }

    const results = try Allocator.alloc(ChunkResult, n_chunks);
    defer Allocator.free(results);
    for (results) |*r| r.* = .{};
    errdefer for (results) |*r| r.deinit();

    var ctx = ChunkContext{
        .tokenizer = tokenizer,
        .segment = segment,
        .base_offset = base_offset,
        .chunk_starts = chunk_starts_buf[0..n_chunks],
        .chunk_ends = chunk_ends_buf[0..n_chunks],
        .own_starts = own_starts_buf[0..n_chunks],
        .own_ends = own_ends_buf[0..n_chunks],
        .results = results,
        .n_chunks = n_chunks,
        .is_sentencepiece_bpe = is_sentencepiece_bpe,
        .is_metaspace = is_metaspace,
        .skip_byte_level = skip_byte_level,
        .had_error = std.atomic.Value(bool).init(false),
    };

    pool.parallelFor(n_chunks * CACHE_LINE_ITEMS, chunkWorker, &ctx);

    if (ctx.had_error.load(.acquire)) return error.OutOfMemory;

    // Selective merge: only include tokens from words owned by each chunk
    try mergeChunkResults(results, own_starts_buf[0..n_chunks], own_ends_buf[0..n_chunks], accumulator);
}

fn mergeChunkResults(
    results: []ChunkResult,
    own_starts: []const usize,
    own_ends: []const usize,
    accumulator: *EncodeAccum,
) !void {
    // Count total owned tokens across all chunks
    var total_owned: usize = 0;
    for (results, own_starts, own_ends) |*r, own_start, own_end| {
        var token_pos: usize = 0;
        for (r.word_offsets.items, r.word_token_counts.items) |offset, count| {
            if (offset >= own_start and offset < own_end) {
                total_owned += count;
            }
            token_pos += count;
        }
    }

    try accumulator.ids.ensureTotalCapacity(Allocator, accumulator.ids.items.len + total_owned);

    // Merge owned tokens (special flags not needed — BPE words are never special)
    for (results, own_starts, own_ends) |*r, own_start, own_end| {
        var token_pos: usize = 0;
        for (r.word_offsets.items, r.word_token_counts.items) |offset, count| {
            if (offset >= own_start and offset < own_end) {
                // Owned — transfer IDs to accumulator
                const start_idx = token_pos;
                const end_idx = token_pos + count;
                accumulator.ids.appendSliceAssumeCapacity(r.accum.ids.items[start_idx..end_idx]);
            }
            token_pos += count;
        }
        // Free containers
        r.accum.ids.deinit(Allocator);
        r.accum.special_positions.deinit(Allocator);
        r.accum = .{};
        r.word_token_counts.deinit(Allocator);
        r.word_offsets.deinit(Allocator);
        r.* = .{};
    }
}

/// Byte-level BPE fast path: model pointer pre-hoisted, no prefix construction,
/// no type/null checks per word. Adds direct-lookup paths for 1-2 byte words
/// that bypass both cache and BPE entirely.
fn encodeWordBpe(model: *bpe.BpeModel, tokenizer: *ct.Tokenizer, word: []const u8, accumulator: *EncodeAccum, cache: *WordCache) !void {
    // 1-byte: direct vocab ID lookup (no cache, no BPE)
    if (word.len == 1 and model.use_raw_byte_init) {
        const id = model.orig_byte_vocab_ids[word[0]];
        if (id >= 0) {
            try accumulator.ids.append(Allocator, id);
            return;
        }
    }

    // 2-byte: two vocab IDs + single merge check (no cache, no BPE loop)
    if (word.len == 2 and model.use_raw_byte_init) {
        const id_a = model.orig_byte_vocab_ids[word[0]];
        const id_b = model.orig_byte_vocab_ids[word[1]];
        if (id_a >= 0 and id_b >= 0) {
            if (model.pair_merges.get(.{ .left = id_a, .right = id_b })) |info| {
                try accumulator.ids.append(Allocator, info.new_id);
            } else {
                try accumulator.ids.appendSlice(Allocator, &.{ id_a, id_b });
            }
            return;
        }
    }

    // Word cache lookup
    if (cache.get(word)) |cached_ids| {
        try accumulator.appendCachedIds(cached_ids);
        return;
    }

    // BPE encode
    const old_len = accumulator.ids.items.len;
    try model.encodeWordDirect(tokenizer, word, &accumulator.ids);
    // Word slices from pretokenize outlive the cache — skip key dupe.
    cache.putBorrowed(word, accumulator.ids.items[old_len..]);
}

fn encodeWord(tokenizer: *ct.Tokenizer, word_bytes: []const u8, add_sentencepiece_prefix: bool, accumulator: *EncodeAccum, cache: ?*WordCache) !void {
    // Fast path: no prefix needed — use word_bytes directly (no copy).
    // Slow path: prepend ▁ (3 bytes) using stack or heap buffer.
    var stack_buf: [1024]u8 = undefined;
    var heap_buf: ?[]u8 = null;
    defer if (heap_buf) |hb| Allocator.free(hb);

    const word = if (!add_sentencepiece_prefix)
        word_bytes
    else blk: {
        const total_len = 3 + word_bytes.len;
        const buf = if (total_len <= stack_buf.len)
            stack_buf[0..total_len]
        else inner: {
            heap_buf = Allocator.alloc(u8, total_len) catch return error.OutOfMemory;
            break :inner heap_buf.?;
        };
        @memcpy(buf[0..3], "\xE2\x96\x81");
        @memcpy(buf[3..], word_bytes);
        break :blk buf;
    };

    // Word cache lookup: skip BPE on repeated words
    if (cache) |wc| {
        if (wc.get(word)) |cached_ids| {
            try accumulator.appendCachedIds(cached_ids);
            return;
        }
    }

    // Direct BPE path: append IDs to accumulator without per-word allocation.
    // This bypasses the TokenizerEncoding intermediary entirely.
    if (tokenizer.type == .bpe) {
        if (tokenizer.model) |model_ptr| {
            const model: *bpe.BpeModel = @ptrCast(@alignCast(model_ptr));
            const old_len = accumulator.ids.items.len;
            try model.encodeWordDirect(tokenizer, word, &accumulator.ids);
            // BPE-produced tokens are never special — sparse tracking
            // in EncodeAccum handles this without per-word overhead.

            // Cache miss: store IDs for future hits
            if (cache) |wc| {
                wc.put(word, accumulator.ids.items[old_len..]);
            }
            return;
        }
    }

    // Fallback for non-BPE models: go through TokenizerEncoding
    var encoding = std.mem.zeroes(ct.TokenizerEncoding);
    defer tokenizer_encoding_free_struct(&encoding);

    const rc = tokenizer.encodeSlice(word, &encoding);
    if (rc != 0) return error.OutOfMemory;

    try accumulator.appendEncoding(&encoding, tokenizer.added);

    // Cache miss: store IDs for future hits
    if (cache) |wc| {
        if (encoding.ids) |ids_ptr| {
            const ids_slice: [*]const i32 = @ptrCast(ids_ptr);
            wc.put(word, ids_slice[0..encoding.ids_len]);
        }
    }
}

fn encodeCombined(
    tokenizer: *ct.Tokenizer,
    segment: []const u8,
    base_offset: usize,
    token_items: []Token,
    token_ranges: []const Range,
    accumulator: *EncodeAccum,
) !void {
    if (token_items.len == 0) return;

    if (tokenizer.type == .bpe and tokenizer.pretokenizer.byte_level == 0 and tokenizer.pretokenizer.metaspace == 0 and
        (tokenizer.pretokenizer.whitespace != 0 or tokenizer.pretokenizer.punctuation != 0))
    {
        try encodeCombinedBpePreservingGaps(tokenizer, segment, base_offset, token_items, token_ranges, accumulator);
        return;
    }

    // Compute total size needed
    var total_len: usize = 0;
    for (token_items, 0..) |token_item, token_index| {
        if (tokenizer.type != ct.ModelType.bpe and token_index > 0) total_len += 1; // space
        total_len += token_item.len;
    }

    // Stack buffer with heap fallback
    var stack_buf: [4096]u8 = undefined;
    var heap_buf: ?[]u8 = null;
    defer if (heap_buf) |hb| Allocator.free(hb);

    const buf = if (total_len <= stack_buf.len)
        stack_buf[0..total_len]
    else blk: {
        heap_buf = Allocator.alloc(u8, total_len) catch return error.OutOfMemory;
        break :blk heap_buf.?;
    };

    var pos: usize = 0;
    for (token_items, 0..) |token_item, token_index| {
        if (tokenizer.type != ct.ModelType.bpe and token_index > 0) {
            buf[pos] = ' ';
            pos += 1;
        }
        const slice = token_item.sliceConst();
        @memcpy(buf[pos..][0..slice.len], slice);
        pos += slice.len;
    }

    if (tokenizer.type == .bpe and tokenizer.added != null) {
        try encodeLiteralBpeBytes(tokenizer, buf[0..pos], accumulator);
        return;
    }

    var encoding = std.mem.zeroes(ct.TokenizerEncoding);
    defer tokenizer_encoding_free_struct(&encoding);

    const rc = tokenizer.encodeSlice(buf[0..pos], &encoding);
    if (rc != 0) return error.OutOfMemory;

    try accumulator.appendEncoding(&encoding, tokenizer.added);
}

fn encodeCombinedBpePreservingGaps(
    tokenizer: *ct.Tokenizer,
    segment: []const u8,
    base_offset: usize,
    token_items: []Token,
    token_ranges: []const Range,
    accumulator: *EncodeAccum,
) !void {
    if (token_items.len == 0) return;

    var total_len: usize = 0;
    var cursor: usize = 0;
    for (token_ranges, 0..) |token_range, token_index| {
        const rel_start = token_range.start - base_offset;
        const rel_end = token_range.end - base_offset;
        if (rel_start > cursor) total_len += rel_start - cursor;
        total_len += token_items[token_index].len;
        cursor = rel_end;
    }
    if (cursor < segment.len) total_len += segment.len - cursor;

    var stack_buf: [4096]u8 = undefined;
    var heap_buf: ?[]u8 = null;
    defer if (heap_buf) |hb| Allocator.free(hb);

    const buf = if (total_len <= stack_buf.len)
        stack_buf[0..total_len]
    else blk: {
        heap_buf = Allocator.alloc(u8, total_len) catch return error.OutOfMemory;
        break :blk heap_buf.?;
    };

    var write_pos: usize = 0;
    cursor = 0;
    for (token_ranges, 0..) |token_range, token_index| {
        const rel_start = token_range.start - base_offset;
        const rel_end = token_range.end - base_offset;

        if (rel_start > cursor) {
            const gap = segment[cursor..rel_start];
            @memcpy(buf[write_pos..][0..gap.len], gap);
            write_pos += gap.len;
        }

        const slice = token_items[token_index].sliceConst();
        @memcpy(buf[write_pos..][0..slice.len], slice);
        write_pos += slice.len;
        cursor = rel_end;
    }
    if (cursor < segment.len) {
        const gap = segment[cursor..];
        @memcpy(buf[write_pos..][0..gap.len], gap);
        write_pos += gap.len;
    }

    try encodeLiteralBpeBytes(tokenizer, buf[0..write_pos], accumulator);
}

/// Encode options that can be passed as runtime arguments (thread-safe).
pub const EncodeOptions = struct {
    /// Override for add_special_tokens. If null, uses tokenizer's configured value.
    add_special_tokens: ?bool = null,
};

pub fn tokenizer_encode_struct(tokenizer_handle: *ct.Tokenizer, input: []const u8, out: *ct.TokenizerEncoding) c_int {
    return tokenizer_encode_struct_with_options(tokenizer_handle, input, out, .{});
}

/// Encode with options (thread-safe - does not mutate tokenizer state).
pub fn tokenizer_encode_struct_with_options(
    tokenizer_handle: *ct.Tokenizer,
    input: []const u8,
    out: *ct.TokenizerEncoding,
    options: EncodeOptions,
) c_int {
    out.* = std.mem.zeroes(ct.TokenizerEncoding);
    return tokenizer_encode_pair_struct_with_options(tokenizer_handle, input, null, out, options);
}

fn tokenizer_encode_pair_struct(tokenizer_handle: *ct.Tokenizer, text_a: []const u8, text_b: ?[]const u8, out: *ct.TokenizerEncoding) c_int {
    return tokenizer_encode_pair_struct_with_options(tokenizer_handle, text_a, text_b, out, .{});
}

fn tokenizer_encode_pair_struct_with_options(
    tokenizer_handle: *ct.Tokenizer,
    text_a: []const u8,
    text_b: ?[]const u8,
    out: *ct.TokenizerEncoding,
    options: EncodeOptions,
) c_int {
    const tokenizer = tokenizer_handle;

    // Determine whether to apply postprocessing (add special tokens).
    // If options.add_special_tokens is set, use that value.
    // Otherwise, use the tokenizer's configured value.
    const apply_postprocess = options.add_special_tokens orelse (tokenizer.postproc.add_special != 0);

    if (text_b == null) {
        return encode_internal(tokenizer, text_a, out, apply_postprocess);
    }

    // Encode both sequences without post-processing
    var first_encoding = std.mem.zeroes(ct.TokenizerEncoding);
    var second_encoding = std.mem.zeroes(ct.TokenizerEncoding);

    if (encode_internal(tokenizer, text_a, &first_encoding, false) != 0) {
        tokenizer_encoding_free_struct(&first_encoding);
        return -1;
    }

    if (encode_internal(tokenizer, text_b.?, &second_encoding, false) != 0) {
        tokenizer_encoding_free_struct(&first_encoding);
        tokenizer_encoding_free_struct(&second_encoding);
        return -1;
    }

    if (tokenizer.truncation.enabled != 0) {
        applyPairTruncation(tokenizer, &first_encoding, &second_encoding);
    }

    if (assemblePairEncoding(tokenizer, &first_encoding, &second_encoding, out) != 0) {
        tokenizer_encoding_free_struct(&first_encoding);
        tokenizer_encoding_free_struct(&second_encoding);
        return -1;
    }

    // Clear ownership from first/second encodings before freeing
    first_encoding.tokens = null;
    first_encoding.tokens_len = 0;
    second_encoding.tokens = null;
    second_encoding.tokens_len = 0;
    tokenizer_encoding_free_struct(&first_encoding);
    tokenizer_encoding_free_struct(&second_encoding);

    return 0;
}

// =============================================================================
// Tests
// =============================================================================

// Note: Most encoding functions (encode_internal, encodeNormalized, encodeSegment, etc.)
// require full tokenizer context with vocab, added tokens, and model state.
// They are tested via integration tests in tests/tokenizer/.

test "EncodeAccum appendAdded adds token" {
    var accum = EncodeAccum{};
    defer accum.deinit();

    // Use a mutable buffer since AddedToken.content expects ?*u8
    var content_buf = [_]u8{ 't', 'e', 's', 't' };
    var added = ct.AddedToken{
        .content = @ptrCast(&content_buf),
        .id = 123,
        .special = 1,
        .single_word = 0,
        .lstrip = 0,
        .rstrip = 0,
        .normalized = 0,
        .next = null,
    };

    try accum.appendAdded(&added);

    try std.testing.expectEqual(@as(usize, 1), accum.ids.items.len);
    try std.testing.expectEqual(@as(i32, 123), accum.ids.items[0]);
    try std.testing.expectEqual(@as(usize, 1), accum.special_positions.items.len);
    try std.testing.expectEqual(@as(u32, 0), accum.special_positions.items[0]);
}

test "tokenizer collect_added_spans matches single_word token at punctuation boundary" {
    var normalized = Normalized{
        .text = try Allocator.dupe(u8, "cat."),
        .map = try Allocator.dupe(i32, &[_]i32{ 0, 1, 2, 3 }),
        .map_end = try Allocator.dupe(i32, &[_]i32{ 1, 2, 3, 4 }),
    };
    defer normalized.deinit();

    const content: [:0]const u8 = "cat";
    var added = ct.AddedToken{
        .content = @constCast(@ptrCast(content.ptr)),
        .id = 100,
        .special = 0,
        .single_word = 1,
        .lstrip = 0,
        .rstrip = 0,
        .normalized = 1,
        .next = null,
    };
    var tokenizer = std.mem.zeroes(ct.Tokenizer);
    tokenizer.added = &added;

    var spans = collect_added_spans(&tokenizer, &normalized, "cat.") orelse return error.OutOfMemory;
    defer spans.deinit(Allocator);

    try std.testing.expectEqual(@as(usize, 1), spans.items.len);
    try std.testing.expectEqual(@as(usize, 0), spans.items[0].start);
    try std.testing.expectEqual(@as(usize, 3), spans.items[0].end);
}

test "tokenizer collect_added_spans expands lstrip to include leading whitespace" {
    var normalized = Normalized{
        .text = try Allocator.dupe(u8, " \t\n[MID]x"),
        .map = try Allocator.dupe(i32, &[_]i32{ 0, 1, 2, 3, 4, 5, 6, 7, 8 }),
        .map_end = try Allocator.dupe(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }),
    };
    defer normalized.deinit();

    const content: [:0]const u8 = "[MID]";
    var added = ct.AddedToken{
        .content = @constCast(@ptrCast(content.ptr)),
        .id = 100,
        .special = 0,
        .single_word = 0,
        .lstrip = 1,
        .rstrip = 0,
        .normalized = 1,
        .next = null,
    };
    var tokenizer = std.mem.zeroes(ct.Tokenizer);
    tokenizer.added = &added;

    var spans = collect_added_spans(&tokenizer, &normalized, " \t\n[MID]x") orelse return error.OutOfMemory;
    defer spans.deinit(Allocator);

    try std.testing.expectEqual(@as(usize, 1), spans.items.len);
    try std.testing.expectEqual(@as(usize, 0), spans.items[0].start);
    try std.testing.expectEqual(@as(usize, 8), spans.items[0].end);
}

test "tokenizer collect_added_spans does not match normalized false token on normalized-only text" {
    var normalized = Normalized{
        .text = try Allocator.dupe(u8, "hello"),
        .map = try Allocator.dupe(i32, &[_]i32{ 0, 1, 2, 3, 4 }),
        .map_end = try Allocator.dupe(i32, &[_]i32{ 1, 2, 3, 4, 5 }),
    };
    defer normalized.deinit();

    const content: [:0]const u8 = "hello";
    var added = ct.AddedToken{
        .content = @constCast(@ptrCast(content.ptr)),
        .id = 100,
        .special = 0,
        .single_word = 0,
        .lstrip = 0,
        .rstrip = 0,
        .normalized = 0,
        .next = null,
    };
    var tokenizer = std.mem.zeroes(ct.Tokenizer);
    tokenizer.added = &added;

    var spans = collect_added_spans(&tokenizer, &normalized, "HELLO") orelse return error.OutOfMemory;
    defer spans.deinit(Allocator);

    try std.testing.expectEqual(@as(usize, 0), spans.items.len);
}

test "tokenizer collect_added_spans does not match single_word token inside unicode letters" {
    const allocator = std.testing.allocator;
    const content: [:0]const u8 = "cat";

    var added = ct.AddedToken{
        .content = @constCast(@ptrCast(content.ptr)),
        .id = 100,
        .special = 0,
        .single_word = 1,
        .lstrip = 0,
        .rstrip = 0,
        .normalized = 1,
        .next = null,
    };
    var tokenizer = std.mem.zeroes(ct.Tokenizer);
    tokenizer.added = &added;

    inline for ([_][]const u8{ "écat", "cat日" }) |input| {
        var normalizer = std.mem.zeroes(ct.Normalizer);
        var normalized = try normalize.normalize_text(&normalizer, input);
        defer normalized.deinit();

        var spans = collect_added_spans(&tokenizer, &normalized, input) orelse return error.OutOfMemory;
        defer spans.deinit(allocator);

        try std.testing.expectEqual(@as(usize, 0), spans.items.len);
    }
}

test "encodeCombined preserves whitespace gap boundaries for non-byte-level bpe" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "version": "1.0",
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {
        \\      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
        \\      "a": 4, "b": 5, "ab": 6
        \\    },
        \\    "merges": ["a b"]
        \\  },
        \\  "added_tokens": [
        \\    {"id": 0, "content": "<pad>", "special": true},
        \\    {"id": 1, "content": "<s>", "special": true},
        \\    {"id": 2, "content": "</s>", "special": true},
        \\    {"id": 3, "content": "<unk>", "special": true}
        \\  ],
        \\  "normalizer": null,
        \\  "pre_tokenizer": {"type": "Whitespace"},
        \\  "post_processor": null,
        \\  "decoder": null
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
    defer tokenizer_encoding_free_struct(&encoding);

    try std.testing.expectEqual(@as(c_int, 0), tokenizer_encode_struct_with_options(tokenizer, "a b", &encoding, .{}));
    try std.testing.expectEqual(@as(usize, 3), encoding.ids_len);

    const ids: [*]i32 = @ptrCast(encoding.ids.?);
    try std.testing.expectEqual(@as(i32, 4), ids[0]);
    try std.testing.expectEqual(@as(i32, 3), ids[1]);
    try std.testing.expectEqual(@as(i32, 5), ids[2]);
}

test "tokenizer_encoding_free_struct handles empty encoding" {
    var encoding = std.mem.zeroes(ct.TokenizerEncoding);
    tokenizer_encoding_free_struct(&encoding);
    // Should not crash
}

test "truncateEncoding reduces encoding length" {
    const ids = try Allocator.alloc(i32, 5);
    defer Allocator.free(ids);
    @memset(ids, 0);

    var encoding = ct.TokenizerEncoding{
        .ids = @ptrCast(ids.ptr),
        .tokens = null,
        .ids_len = 5,
        .tokens_len = 0,
        .attention_mask = null,
        .type_ids = null,
        .special_tokens_mask = null,
        .offsets = null,
        .overflows = null,
        .overflow_count = 0,
    };

    truncateEncoding(&encoding, 3);

    try std.testing.expectEqual(@as(usize, 3), encoding.ids_len);
}

test "tokenizer_encode_struct requires integration testing" {
    // This function requires a fully initialized tokenizer with:
    // - Loaded vocabulary and model (BPE/WordPiece/Unigram)
    // - Normalizer configuration
    // - Pretokenizer with compiled regex patterns
    // - Post-processor for special token handling
    // Integration tests: tests/tokenizer/test_*.py
}

test "tokenizer_encode_struct_with_options requires integration testing" {
    // This function requires a fully initialized tokenizer with:
    // - Complete tokenizer context (vocab, model, normalizer, pretokenizer)
    // - Support for encoding options (add_special_tokens override)
    // - Thread-safe operation
    // Integration tests: tests/tokenizer/test_*.py
}
