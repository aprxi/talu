//! Tokenizer Encoding Pipeline
//!
//! Orchestrates the full encoding flow: normalization, pretokenization,
//! model encoding (BPE/WordPiece/Unigram), and post-processing.

const std = @import("std");
const ct = @import("c_types.zig");
const pretokenize = @import("pretokenize.zig");
const normalize = @import("normalize.zig");
const postprocess = @import("postprocess.zig");
const buffers = @import("encoding_buffers.zig");
const strings = @import("strings.zig");
const errors = @import("errors.zig");
const types = @import("types.zig");
const log = @import("../log.zig");

const Allocator = types.Allocator;
const Token = types.Token;
const Normalized = types.Normalized;

// Added token span
const AddedSpan = struct {
    start: usize,
    end: usize,
    at: *const ct.AddedToken,
};

// ============================================================================
// ADDED TOKENS COLLECTION
// ============================================================================

fn matches_added_token_boundaries(added_token: *const ct.AddedToken, input_bytes: []const u8, position: usize) bool {
    if (added_token.content == null) return false;
    const content = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(added_token.content.?)), 0);
    if (position + content.len > input_bytes.len) return false;
    if (!std.mem.eql(u8, input_bytes[position..][0..content.len], content)) return false;

    if (added_token.single_word != 0) {
        const left_boundary_ok = (position == 0) or std.ascii.isWhitespace(input_bytes[position - 1]);
        const right_boundary_ok = (position + content.len == input_bytes.len) or std.ascii.isWhitespace(input_bytes[position + content.len]);
        if (!left_boundary_ok or !right_boundary_ok) return false;
    }
    // lstrip: if true, skip leading whitespace before matching (handled elsewhere)
    // rstrip: if true, consume trailing whitespace after token (handled elsewhere)
    // These do NOT restrict where the token can appear
    _ = added_token.lstrip;
    _ = added_token.rstrip;
    return true;
}

fn collect_added_spans(tokenizer: *ct.Tokenizer, normalized: []const u8, normalized_map: []const i32, original_input: []const u8) ?std.ArrayListUnmanaged(AddedSpan) {
    var spans = std.ArrayListUnmanaged(AddedSpan){};

    log.trace("tokenizer", "collect_added_spans", .{
        .normalized_len = normalized.len,
        .has_added_tokens = tokenizer.added != null,
    }, @src());

    var cursor: usize = 0;
    while (cursor < normalized.len) {
        var best_match: ?*const ct.AddedToken = null;
        var best_match_len: usize = 0;

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

            const text = if (added_token.normalized != 0) normalized else original_input;
            const text_pos_opt: ?usize = if (added_token.normalized != 0) cursor else blk: {
                // For non-normalized tokens, map back to original position
                if (cursor < normalized_map.len and normalized_map[cursor] >= 0) {
                    break :blk @as(usize, @intCast(normalized_map[cursor]));
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

            if (text_pos + content.len > text_len) {
                added_iter = added_token.next;
                continue;
            }
            if (!std.mem.eql(u8, text[text_pos..][0..content.len], content)) {
                added_iter = added_token.next;
                continue;
            }
            if (!matches_added_token_boundaries(added_token, text, text_pos)) {
                added_iter = added_token.next;
                continue;
            }
            if (content.len > best_match_len) {
                best_match = added_token;
                best_match_len = content.len;
            }
            added_iter = added_token.next;
        }

        if (best_match) |matched| {
            spans.append(Allocator, .{ .start = cursor, .end = cursor + best_match_len, .at = matched }) catch {
                spans.deinit(Allocator);
                return null;
            };
            cursor += best_match_len;
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
    return if (tokenizer.postproc.pair != 0) 4 else 3;
}

fn truncationSpecialTokenCount(tokenizer: *const ct.Tokenizer) usize {
    if (tokenizer.postproc.add_special == 0) return 0;
    return if (tokenizer.postproc.pair != 0) 3 else 2;
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
    const cls_str = std.mem.sliceTo(&tokenizer.postproc.cls_token, 0);
    return appendSpecial(buffers_out, write_index, tokenizer.postproc.cls_id, cls_str, 0, offset);
}

fn appendSep(tokenizer: *const ct.Tokenizer, buffers_out: *buffers.Buffers, write_index: *usize, type_id: i32, offset: ct.Offset) bool {
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

/// Free null-terminated token list (for EncodeAccum)
fn freeNullTerminatedTokenList(token_list: *std.ArrayListUnmanaged([*:0]u8)) void {
    for (token_list.items) |token_ptr| {
        Allocator.free(std.mem.span(token_ptr));
    }
    token_list.deinit(Allocator);
}

/// Accumulator for building token encoding results
const EncodeAccum = struct {
    ids: std.ArrayListUnmanaged(i32) = .{},
    tokens: std.ArrayListUnmanaged([*:0]u8) = .{},
    special: std.ArrayListUnmanaged(i32) = .{},

    fn deinit(self: *EncodeAccum) void {
        freeNullTerminatedTokenList(&self.tokens);
        self.ids.deinit(Allocator);
        self.special.deinit(Allocator);
    }

    fn appendAdded(self: *EncodeAccum, added_token: *const ct.AddedToken) !void {
        try self.ids.append(Allocator, added_token.id);
        const token_copy = strings.tokenizer_strdup(@ptrCast(added_token.content.?)) orelse return error.OutOfMemory;
        errdefer Allocator.free(std.mem.span(token_copy));
        try self.tokens.append(Allocator, token_copy);
        try self.special.append(Allocator, added_token.special);
    }

    fn appendEncoding(self: *EncodeAccum, encoding: *const ct.TokenizerEncoding, added_head: ?*ct.AddedToken) !void {
        if (encoding.ids == null) return;
        const ids_ptr: [*]i32 = @ptrCast(encoding.ids.?);
        const tokens_ptrs: ?[*][*c]u8 = if (encoding.tokens) |t| @ptrCast(t) else null;

        for (0..encoding.ids_len) |id_index| {
            try self.ids.append(Allocator, ids_ptr[id_index]);

            if (tokens_ptrs) |ts| {
                if (ts[id_index]) |token_ptr| {
                    const token_copy = strings.tokenizer_strdup(@ptrCast(token_ptr)) orelse return error.OutOfMemory;
                    errdefer Allocator.free(std.mem.span(token_copy));
                    try self.tokens.append(Allocator, token_copy);
                } else {
                    // Allocate sentinel-terminated empty string so free() accounts for the sentinel byte.
                    const empty_token = Allocator.allocSentinel(u8, 0, 0) catch return error.OutOfMemory;
                    try self.tokens.append(Allocator, @ptrCast(empty_token.ptr));
                }
            }

            try self.special.append(Allocator, checkSpecial(added_head, ids_ptr[id_index]));
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

        var buffers_out = try buffers.allocBuffers(token_count);
        errdefer buffers_out.deinit();

        @memcpy(buffers_out.ids, self.ids.items);
        @memcpy(buffers_out.special, self.special.items);
        for (0..token_count) |token_idx| {
            buffers_out.tokens[token_idx] = @ptrCast(self.tokens.items[token_idx]);
            buffers_out.attention_mask[token_idx] = 1;
            buffers_out.type_ids[token_idx] = 0;
            buffers_out.offsets[token_idx] = .{ .start = 0, .end = 0 };
        }

        buffers.initEncoding(out_encoding, &buffers_out, token_count, null, 0);

        // Ownership transferred - just free containers
        self.ids.deinit(Allocator);
        self.tokens.deinit(Allocator);
        self.special.deinit(Allocator);
        self.* = .{};
    }
};

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
    if (input.len == 0) {
        out.* = std.mem.zeroes(ct.TokenizerEncoding);
        if (apply_postprocess and tokenizer.postproc.add_special != 0) {
            if (postprocess.postprocess_single(&tokenizer.postproc, out) != 0) {
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
    var spans = collect_added_spans(tokenizer, normalized.text, normalized.map, input) orelse return error.OutOfMemory;
    defer spans.deinit(Allocator);

    // Step 3: Encode segments
    var accum = EncodeAccum{};
    errdefer accum.deinit();

    var cursor: usize = 0;
    var span_index: usize = 0;
    var after_added_token = false;

    // Check if SentencePiece prepend is enabled
    const has_sp_prepend = tokenizer.normalizer.prepend != null;
    const prepend_bytes = if (tokenizer.normalizer.prepend) |p| std.mem.sliceTo(p, 0) else "";

    // Skip the prepend if input starts with an added token
    // (the prepend was added by normalization but shouldn't apply to added tokens)
    if (has_sp_prepend and spans.items.len > 0 and spans.items[0].start == prepend_bytes.len) {
        // First added token is right after the prepend - skip the prepend
        cursor = prepend_bytes.len;
    }

    while (cursor < normalized.text.len) {
        // Handle added token spans
        if (span_index < spans.items.len and cursor == spans.items[span_index].start) {
            const added_span = spans.items[span_index];
            try accum.appendAdded(added_span.at);
            cursor = added_span.end;
            span_index += 1;
            after_added_token = true; // Next segment should get prepend
            continue;
        }

        // Encode segment up to next span
        const next_span_start = if (span_index < spans.items.len) spans.items[span_index].start else normalized.text.len;
        if (next_span_start <= cursor) {
            cursor += 1;
            continue;
        }

        const segment = normalized.text[cursor..next_span_start];
        const should_prepend = after_added_token and has_sp_prepend and prepend_bytes.len > 0;
        after_added_token = false;
        try encodeSegmentMaybePrepended(tokenizer, segment, cursor, should_prepend, prepend_bytes, &accum);
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
    var pretokenized = try pretokenize.pretokenize(&tokenizer.pretokenizer, segment, .{ .start = base_offset, .end = base_offset + segment.len });
    defer pretokenized.deinit();

    const is_sentencepiece_bpe = tokenizer.type == ct.ModelType.bpe and tokenizer.pretokenizer.regex_split != 0 and tokenizer.pretokenizer.byte_level == 0;
    const is_byte_level_bpe = tokenizer.type == ct.ModelType.bpe and tokenizer.pretokenizer.byte_level != 0;
    const is_metaspace_bpe = tokenizer.type == ct.ModelType.bpe and tokenizer.pretokenizer.metaspace != 0;

    if (is_sentencepiece_bpe or is_byte_level_bpe or is_metaspace_bpe) {
        // Encode words separately
        for (pretokenized.tokens.items, 0..) |token_item, token_index| {
            // SentencePiece: ▁ prefix on non-first words (first gets it from normalizer prepend).
            // Metaspace: tokens from splitMetaspace already have ▁ embedded
            // (from space→▁ replacement). Only add ▁ if the token doesn't
            // already start with ▁.
            const add_sp_prefix = if (is_metaspace_bpe) blk: {
                const tok_bytes = token_item.sliceConst();
                const starts_with_sp = tok_bytes.len >= 3 and
                    tok_bytes[0] == 0xE2 and tok_bytes[1] == 0x96 and tok_bytes[2] == 0x81;
                break :blk !starts_with_sp and
                    (tokenizer.pretokenizer.add_prefix_space != 0 or token_index > 0);
            } else
                (is_sentencepiece_bpe and token_index > 0);
            try encodeWord(tokenizer, token_item.sliceConst(), add_sp_prefix, accumulator);
        }
    } else {
        // Combine and encode together
        try encodeCombined(tokenizer, pretokenized.tokens.items, accumulator);
    }
}

fn encodeWord(tokenizer: *ct.Tokenizer, word_bytes: []const u8, add_sentencepiece_prefix: bool, accumulator: *EncodeAccum) !void {
    var token_bytes = std.ArrayListUnmanaged(u8){};
    defer token_bytes.deinit(Allocator);

    if (add_sentencepiece_prefix) try token_bytes.appendSlice(Allocator, "\xE2\x96\x81"); // ▁
    try token_bytes.appendSlice(Allocator, word_bytes);
    // Don't add null terminator - use slice-based encoding to support embedded nulls

    var encoding = std.mem.zeroes(ct.TokenizerEncoding);
    defer tokenizer_encoding_free_struct(&encoding);

    const rc = tokenizer.encodeSlice(token_bytes.items, &encoding);
    if (rc != 0) return error.OutOfMemory;

    try accumulator.appendEncoding(&encoding, tokenizer.added);
}

fn encodeCombined(tokenizer: *ct.Tokenizer, token_items: []Token, accumulator: *EncodeAccum) !void {
    if (token_items.len == 0) return;

    var token_bytes = std.ArrayListUnmanaged(u8){};
    defer token_bytes.deinit(Allocator);

    for (token_items, 0..) |token_item, token_index| {
        if (tokenizer.type != ct.ModelType.bpe and token_index > 0) try token_bytes.append(Allocator, ' ');
        try token_bytes.appendSlice(Allocator, token_item.sliceConst());
    }
    // Don't add null terminator - use slice-based encoding to support embedded nulls

    var encoding = std.mem.zeroes(ct.TokenizerEncoding);
    defer tokenizer_encoding_free_struct(&encoding);

    const rc = tokenizer.encodeSlice(token_bytes.items, &encoding);
    if (rc != 0) return error.OutOfMemory;

    try accumulator.appendEncoding(&encoding, tokenizer.added);
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
    try std.testing.expectEqual(@as(i32, 1), accum.special.items[0]);
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
