//! BPE (Byte-Pair Encoding) Model
//!
//! Implements BPE tokenization with lazy vocab/merges parsing.
//! Supports both GPT-2 style (byte-level) and SentencePiece BPE.

const std = @import("std");
const ct = @import("c_types.zig");
const json_utils = @import("json_utils.zig");
const utils = @import("utils.zig");
const log = @import("log_pkg");

const tok_fns = @import("pipeline.zig");

// Internal helpers from utils
const utf8Encode = utils.utf8Encode;
const utf8Decode = utils.utf8Decode;
const utf8CharLen = utils.utf8CharLen;

const DEFAULT_PATTERN: [:0]const u8 = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
const DEFAULT_UNK: [:0]const u8 = "<unk>";

// ---------------------------------------------------------------------------
// Integer-pair merge table types (fast encode path)
// ---------------------------------------------------------------------------

const MergePair = struct { left: i32, right: i32 };
const MergeInfo = struct { rank: i32, new_id: i32 };

const PairContext = struct {
    pub fn hash(_: PairContext, key: MergePair) u64 {
        const a: u64 = @bitCast(@as(i64, key.left));
        const b: u64 = @bitCast(@as(i64, key.right));
        return (a *% 0x517cc1b727220a95) ^ b;
    }
    pub fn eql(_: PairContext, a: MergePair, b: MergePair) bool {
        return a.left == b.left and a.right == b.right;
    }
};

const PairMergeMap = std.HashMapUnmanaged(MergePair, MergeInfo, PairContext, std.hash_map.default_max_load_percentage);

/// Symbol for the linked-list BPE merge loop.
/// Uses i32 indices to support words with more than 32K characters.
const Symbol = struct {
    id: i32, // vocab ID (-1 = unknown)
    start: u32, // byte offset in word
    len: u32, // byte length
    prev: i32, // previous symbol index (-1 = head)
    next: i32, // next symbol index (-1 = tail)
};

const MAX_WORD_SYMBOLS = 512;
const MAX_MODEL_VOCAB_SIZE: usize = 1_000_000;

fn jsonUnescapeAllocated(raw: []const u8, decoded: []const u8) bool {
    return decoded.ptr != raw.ptr;
}

fn freeJsonUnescapeScratch(allocator: std.mem.Allocator, raw: []const u8, decoded: []const u8) void {
    if (jsonUnescapeAllocated(raw, decoded)) allocator.free(@constCast(decoded));
}

fn rememberOwnedVocabToken(model: *BpeModel, raw: []const u8, decoded: []const u8) !void {
    if (jsonUnescapeAllocated(raw, decoded)) {
        model.owned_vocab_tokens.append(model.allocator, @constCast(decoded)) catch |err| {
            model.allocator.free(@constCast(decoded));
            return err;
        };
    }
}

fn deinitBpeModelStorage(model: *BpeModel, free_json_buffer: bool) void {
    for (model.merge_added_tokens.items) |token| model.allocator.free(token);
    for (model.owned_vocab_tokens.items) |token| model.allocator.free(token);
    model.merge_added_tokens.deinit(model.allocator);
    model.owned_vocab_tokens.deinit(model.allocator);

    for (model.merge_strings.items) |merge_storage| model.allocator.free(@constCast(merge_storage));
    model.merge_strings.deinit(model.allocator);
    model.merges.deinit(model.allocator);
    model.pair_merges.deinit(model.allocator);
    model.vocab_hash.deinit(model.allocator);
    model.allocator.free(model.id_to_token);

    for (model.byte_to_unicode) |mapped_bytes| {
        if (mapped_bytes.len > 0) model.allocator.free(@constCast(mapped_bytes));
    }

    if (free_json_buffer and model.json_owned and model.json_buffer.len > 0) {
        model.allocator.free(@constCast(model.json_buffer));
    }
}

/// BPE model. Vocab and merges are parsed synchronously during creation.
pub const BpeModel = struct {
    allocator: std.mem.Allocator,

    // Raw JSON buffer
    json_buffer: []const u8,
    json_owned: bool,

    // Vocab: array indexed by ID. JSON-path entries usually borrow from
    // json_buffer; escaped JSON strings are unescaped into owned_vocab_tokens.
    id_to_token: []?[]const u8,
    owned_vocab_tokens: std.ArrayListUnmanaged([]u8),
    vocab_hash: std.StringHashMapUnmanaged(i32),

    // Merges: string-based hash map (used during init, kept for edge cases)
    merges: std.StringHashMapUnmanaged(i32),
    merge_strings: std.ArrayListUnmanaged([]const u8),

    // Fast merge table: (left_id, right_id) → { rank, new_id }
    pair_merges: PairMergeMap,

    // Tokens auto-added from merge rules not present in vocab.
    // Separately allocated; id_to_token stores borrowed slices for normal JSON
    // vocab entries and these owned slices for implicit merge tokens.
    merge_added_tokens: std.ArrayListUnmanaged([]u8),

    // Byte mapping (small, always built)
    byte_to_unicode: [256][]const u8,
    unicode_to_byte: [65536]i32,

    // SentencePiece byte fallback: maps byte value -> token ID for <0xNN> tokens
    // Used when a character can't be found in vocab (e.g., control chars like \n)
    byte_fallback_ids: [256]i32,

    // Direct lookup table for single-byte vocab tokens.
    // byte_vocab_ids[b] = vocab ID for the 1-byte token {b}, or -1 if not in vocab.
    // Avoids hash lookup in symbol init for the 95%+ of chars that are single-byte.
    byte_vocab_ids: [256]i32,

    // Direct byte→vocab-ID for byte-level BPE (pre-byte-level-encoding).
    // Maps original byte value → vocab ID of its byte_to_unicode representation.
    // Eliminates both applyByteLevel and all hash lookups in symbol init.
    orig_byte_vocab_ids: [256]i32,

    // True when all 256 orig_byte_vocab_ids entries are valid (>= 0).
    // When set, callers can skip applyByteLevel and pass raw bytes to BPE,
    // which uses orig_byte_vocab_ids for O(1) symbol init per byte.
    use_raw_byte_init: bool,

    // Mirrors HuggingFace BPE `ignore_merges` option. When enabled, encoding
    // prefers a direct full-token vocab match before applying merge rules.
    ignore_merges: bool,

    // Config
    unk_token: [16]u8,
    unk_id: i32,
    bos_id: i32,
    eos_id: i32,
    vocab_size: usize,

    owner: ?*ct.Tokenizer,

    // =============================================================================
    // Public Methods (Dispatch Entry Points)
    // =============================================================================

    /// Encode text to tokens with explicit length (supports embedded null bytes)
    pub fn encodeSlice(self: *BpeModel, tok: *ct.Tokenizer, input: []const u8, enc: *ct.TokenizerEncoding) c_int {
        return bpe_encode_slice_impl(self, tok, input, enc);
    }

    /// Encode a word and append IDs directly to the caller's list.
    /// Bypasses the TokenizerEncoding intermediary — no per-word allocation.
    /// Skips the added-token check (caller must pre-separate added tokens).
    pub fn encodeWordDirect(self: *BpeModel, tok: *ct.Tokenizer, word: []const u8, out_ids: *std.ArrayListUnmanaged(i32)) !void {
        return encodeWordCore(self, tok, word, out_ids);
    }

    /// Decode token IDs to text with options (skip special tokens etc.)
    pub fn decodeWithOptions(self: *BpeModel, tok: *ct.Tokenizer, ids: [*c]const i32, ids_len: usize, out: *[*c]u8, out_len: *usize, options: ct.DecodeOptions) c_int {
        return bpe_decode_with_options_impl(self, tok, ids, ids_len, out, out_len, options);
    }

    /// Free the model resources
    pub fn destroy(self: *BpeModel, tok: *ct.Tokenizer) void {
        bpe_destroy_impl(self, tok);
    }
};

/// Create a BPE model from a JSON buffer, parsing vocab and merges eagerly.
fn create(allocator: std.mem.Allocator, json_buffer: []const u8, json_owned: bool) !*BpeModel {
    var model = try allocator.create(BpeModel);
    errdefer allocator.destroy(model);

    // Determine vocab size by finding max ID in JSON (scan for largest number after ":")
    const vocab_size = try findVocabSize(json_buffer);
    if (vocab_size == 0 or vocab_size > MAX_MODEL_VOCAB_SIZE) return error.InvalidArgument;

    var model_storage_initialized = false;
    errdefer if (model_storage_initialized) {
        deinitBpeModelStorage(model, false);
    };

    model.* = .{
        .allocator = allocator,
        .json_buffer = json_buffer,
        .json_owned = json_owned,
        .id_to_token = try allocator.alloc(?[]const u8, vocab_size),
        .owned_vocab_tokens = .{},
        .vocab_hash = .{},
        .merges = .{},
        .merge_strings = .{},
        .pair_merges = .{},
        .merge_added_tokens = .{},
        .byte_to_unicode = undefined,
        .unicode_to_byte = [_]i32{-1} ** 65536,
        .byte_fallback_ids = [_]i32{-1} ** 256,
        .byte_vocab_ids = [_]i32{-1} ** 256,
        .orig_byte_vocab_ids = [_]i32{-1} ** 256,
        .use_raw_byte_init = false,
        .ignore_merges = parseIgnoreMergesOption(json_buffer),
        .unk_token = undefined,
        .unk_id = 0,
        .bos_id = -1,
        .eos_id = -1,
        .vocab_size = vocab_size,
        .owner = null,
    };
    model_storage_initialized = true;

    // Initialize id_to_token to null
    @memset(model.id_to_token, null);

    // Initialize byte_to_unicode
    for (&model.byte_to_unicode) |*slot| {
        slot.* = "";
    }

    // Set default unk token
    @memset(model.unk_token[0..], 0);
    @memcpy(model.unk_token[0..DEFAULT_UNK.len], DEFAULT_UNK);

    // Build byte map (fast, ~1ms)
    try initByteMap(model);

    try parseVocabAndMerges(model);

    return model;
}

fn skipJsonWhitespace(bytes: []const u8, cursor: *usize) void {
    while (cursor.* < bytes.len and std.ascii.isWhitespace(bytes[cursor.*])) : (cursor.* += 1) {}
}

fn parseVocabEntryId(vocab_json: []const u8, cursor: *usize) !usize {
    skipJsonWhitespace(vocab_json, cursor);
    if (cursor.* >= vocab_json.len or vocab_json[cursor.*] != ':') return error.InvalidVocab;
    cursor.* += 1;
    skipJsonWhitespace(vocab_json, cursor);

    if (cursor.* >= vocab_json.len) return error.InvalidVocab;
    const first = vocab_json[cursor.*];
    if (first == '-' or first == '+') return error.InvalidVocab;
    if (first < '0' or first > '9') return error.InvalidVocab;

    var id: usize = 0;
    while (cursor.* < vocab_json.len and vocab_json[cursor.*] >= '0' and vocab_json[cursor.*] <= '9') : (cursor.* += 1) {
        id = std.math.mul(usize, id, 10) catch return error.InvalidVocab;
        id = std.math.add(usize, id, @as(usize, vocab_json[cursor.*] - '0')) catch return error.InvalidVocab;
    }

    skipJsonWhitespace(vocab_json, cursor);
    if (cursor.* >= vocab_json.len) return error.InvalidVocab;
    if (vocab_json[cursor.*] != ',' and vocab_json[cursor.*] != '}') return error.InvalidVocab;
    return id;
}

fn vocabObjectSlice(json: []const u8) ![]const u8 {
    const vocab_start = findSectionStart(json, "\"vocab\"") orelse return error.NoVocab;
    const vocab_json = json[vocab_start..];
    if (vocab_json.len == 0 or vocab_json[0] != '{') return error.InvalidVocab;
    const vocab_end = utils.findMatchingBrace(vocab_json, '{', '}') orelse return error.InvalidVocab;
    return vocab_json[0..vocab_end];
}

/// Find vocab size by scanning for the max ID inside the vocab object.
fn findVocabSize(json: []const u8) !usize {
    const vocab_json = try vocabObjectSlice(json);
    var max_id: usize = 0;
    var found_entry = false;
    var scan_idx: usize = 0;
    var depth: usize = 0;

    while (scan_idx < vocab_json.len) {
        const ch = vocab_json[scan_idx];
        if (ch == '{') {
            depth += 1;
            scan_idx += 1;
        } else if (ch == '}') {
            if (depth == 0) return error.InvalidVocab;
            depth -= 1;
            if (depth == 0) break;
            scan_idx += 1;
        } else if (ch == '"' and depth == 1) {
            scan_idx += 1;
            while (scan_idx < vocab_json.len and vocab_json[scan_idx] != '"') {
                if (vocab_json[scan_idx] == '\\') scan_idx += 2 else scan_idx += 1;
            }
            if (scan_idx >= vocab_json.len) return error.InvalidVocab;
            scan_idx += 1;

            const id = try parseVocabEntryId(vocab_json, &scan_idx);
            if (id > max_id) max_id = id;
            found_entry = true;
        } else {
            scan_idx += 1;
        }
    }

    if (depth != 0) return error.InvalidVocab;
    if (!found_entry) return 0;
    return std.math.add(usize, max_id, 1) catch error.InvalidArgument;
}

/// Parse vocab and merges from JSON - single pass, no pre-scanning
fn parseVocabAndMerges(model: *BpeModel) !void {
    const json = model.json_buffer;
    const vocab_start = findSectionStart(json, "\"vocab\"") orelse return error.NoVocab;
    const merges_start: ?usize = findSectionStart(json, "\"merges\"");
    const vocab_end = merges_start orelse json.len;

    const vocab_count = try parseVocabSection(model, json[vocab_start..vocab_end]);
    try buildVocabHash(model, vocab_count);
    initByteFallback(model);
    initByteVocabIds(model);
    initOrigByteVocabIds(model);

    if (merges_start) |m_start| {
        try parseMergesSection(model, json[m_start..]);
    }
    try buildPairMergeTable(model);
    finalizeSpecialTokenIds(model);
}

fn parseVocabSection(model: *BpeModel, vocab_json: []const u8) !usize {
    const allocator = model.allocator;
    var scan_idx: usize = 0;
    var count: usize = 0;
    var depth: usize = 0;

    while (scan_idx < vocab_json.len) {
        const ch = vocab_json[scan_idx];

        if (ch == '{') {
            depth += 1;
            scan_idx += 1;
        } else if (ch == '}') {
            if (depth == 0) return error.InvalidVocab;
            depth -= 1;
            if (depth == 0) break; // End of vocab object
            scan_idx += 1;
        } else if (ch == '"' and depth == 1) {
            // Parse key-value pair
            scan_idx += 1;
            const key_start = scan_idx;

            // Find closing quote (handle escapes)
            while (scan_idx < vocab_json.len and vocab_json[scan_idx] != '"') {
                if (vocab_json[scan_idx] == '\\') scan_idx += 2 else scan_idx += 1;
            }
            if (scan_idx >= vocab_json.len) return error.InvalidVocab;
            const key_end = scan_idx;
            scan_idx += 1;

            const id = try parseVocabEntryId(vocab_json, &scan_idx);
            if (id >= model.id_to_token.len) return error.InvalidVocab;
            if (model.id_to_token[id] != null) return error.InvalidVocab;

            // Store token - unescape JSON if needed
            const raw_token = vocab_json[key_start..key_end];
            const token = try json_utils.unescapeJsonString(allocator, raw_token);
            try rememberOwnedVocabToken(model, raw_token, token);
            model.id_to_token[id] = token;
            count += 1;
        } else {
            scan_idx += 1;
        }
    }
    if (depth != 0) return error.InvalidVocab;
    return count;
}

fn buildVocabHash(model: *BpeModel, vocab_count: usize) !void {
    try model.vocab_hash.ensureTotalCapacity(model.allocator, @intCast(vocab_count));
    for (model.id_to_token, 0..) |maybe_token, id| {
        if (maybe_token) |token| {
            const entry = try model.vocab_hash.getOrPut(model.allocator, token);
            if (entry.found_existing) return error.InvalidVocab;
            entry.value_ptr.* = @intCast(id);
        }
    }
}

fn parseMergesSection(model: *BpeModel, merges_json: []const u8) !void {
    const allocator = model.allocator;
    const merge_buffer = try allocator.alloc(u8, 8 * 1024 * 1024);
    var merge_buffer_owned_by_model = false;
    errdefer if (!merge_buffer_owned_by_model) allocator.free(merge_buffer);

    var merge_buf_pos: usize = 0;
    var rank: i32 = 0;
    var merge_idx: usize = 0;
    var depth: usize = 0;

    while (merge_idx < merges_json.len) {
        const ch = merges_json[merge_idx];

        if (ch == '[') {
            depth += 1;
            if (depth == 2) {
                try parseArrayMerge(model, merges_json, &merge_idx, merge_buffer, &merge_buf_pos, &rank);
            } else {
                merge_idx += 1;
            }
        } else if (ch == ']') {
            if (depth == 0) return error.InvalidMerges;
            depth -= 1;
            if (depth == 0) break;
            merge_idx += 1;
        } else if (ch == '"' and depth == 1) {
            try parseStringMerge(model, merges_json, &merge_idx, merge_buffer, &merge_buf_pos, &rank);
        } else {
            merge_idx += 1;
        }
    }
    if (depth != 0) return error.InvalidMerges;

    try model.merge_strings.append(allocator, merge_buffer);
    merge_buffer_owned_by_model = true;
}

fn parseArrayMerge(
    model: *BpeModel,
    merges_json: []const u8,
    merge_idx: *usize,
    merge_buffer: []u8,
    merge_buf_pos: *usize,
    rank: *i32,
) !void {
    const allocator = model.allocator;
    merge_idx.* += 1;

    const raw_lhs = try nextJsonString(merges_json, merge_idx);
    const raw_rhs = try nextJsonString(merges_json, merge_idx);
    const lhs = try json_utils.unescapeJsonString(allocator, raw_lhs);
    defer freeJsonUnescapeScratch(allocator, raw_lhs, lhs);
    const rhs = try json_utils.unescapeJsonString(allocator, raw_rhs);
    defer freeJsonUnescapeScratch(allocator, raw_rhs, rhs);

    const key_len = lhs.len + 1 + rhs.len;
    const merge_key = try reserveMergeKey(merge_buffer, merge_buf_pos, key_len);
    @memcpy(merge_key[0..lhs.len], lhs);
    merge_key[lhs.len] = ' ';
    @memcpy(merge_key[lhs.len + 1 ..], rhs);
    try putMerge(model, merge_key, rank.*);
    rank.* += 1;
}

fn parseStringMerge(
    model: *BpeModel,
    merges_json: []const u8,
    merge_idx: *usize,
    merge_buffer: []u8,
    merge_buf_pos: *usize,
    rank: *i32,
) !void {
    const allocator = model.allocator;
    merge_idx.* += 1;
    const str_start = merge_idx.*;
    while (merge_idx.* < merges_json.len and merges_json[merge_idx.*] != '"') {
        if (merges_json[merge_idx.*] == '\\') merge_idx.* += 2 else merge_idx.* += 1;
    }
    if (merge_idx.* >= merges_json.len) return error.InvalidMerges;
    const raw_merge = merges_json[str_start..merge_idx.*];
    merge_idx.* += 1;

    const merge_str = try json_utils.unescapeJsonString(allocator, raw_merge);
    defer freeJsonUnescapeScratch(allocator, raw_merge, merge_str);
    const merge_key = try reserveMergeKey(merge_buffer, merge_buf_pos, merge_str.len);
    @memcpy(merge_key, merge_str);
    try putMerge(model, merge_key, rank.*);
    rank.* += 1;
}

fn putMerge(model: *BpeModel, merge_key: []const u8, rank: i32) !void {
    const entry = try model.merges.getOrPut(model.allocator, merge_key);
    if (entry.found_existing) return error.InvalidMerges;
    entry.value_ptr.* = rank;
}

fn nextJsonString(json: []const u8, cursor: *usize) ![]const u8 {
    while (cursor.* < json.len and json[cursor.*] != '"') : (cursor.* += 1) {}
    if (cursor.* >= json.len) return error.InvalidMerges;
    cursor.* += 1;
    const start = cursor.*;
    while (cursor.* < json.len and json[cursor.*] != '"') {
        if (json[cursor.*] == '\\') cursor.* += 2 else cursor.* += 1;
    }
    if (cursor.* >= json.len) return error.InvalidMerges;
    const end = cursor.*;
    cursor.* += 1;
    return json[start..end];
}

fn reserveMergeKey(merge_buffer: []u8, merge_buf_pos: *usize, key_len: usize) ![]u8 {
    if (key_len == 0 or merge_buf_pos.* + key_len > merge_buffer.len) return error.InvalidMerges;
    const merge_key = merge_buffer[merge_buf_pos.* .. merge_buf_pos.* + key_len];
    merge_buf_pos.* += key_len;
    return merge_key;
}

fn buildPairMergeTable(model: *BpeModel) !void {
    const allocator = model.allocator;
    const merge_count = model.merges.count();
    try model.pair_merges.ensureTotalCapacity(allocator, @intCast(merge_count));
    const ranked_merges = try allocator.alloc(?[]const u8, merge_count);
    defer allocator.free(ranked_merges);
    @memset(ranked_merges, null);

    var merge_iter = model.merges.iterator();
    while (merge_iter.next()) |entry| {
        const rank: usize = @intCast(entry.value_ptr.*);
        if (rank >= ranked_merges.len or ranked_merges[rank] != null) return error.InvalidMerges;
        ranked_merges[rank] = entry.key_ptr.*;
    }

    var concat_buf = std.ArrayListUnmanaged(u8){};
    defer concat_buf.deinit(allocator);

    for (ranked_merges, 0..) |maybe_key, rank| {
        const key = maybe_key orelse return error.InvalidMerges;
        const space_pos = std.mem.indexOfScalar(u8, key, ' ') orelse return error.InvalidMerges;
        if (space_pos == 0 or space_pos + 1 >= key.len) return error.InvalidMerges;
        if (std.mem.indexOfScalarPos(u8, key, space_pos + 1, ' ') != null) return error.InvalidMerges;

        const left_str = key[0..space_pos];
        const right_str = key[space_pos + 1 ..];
        const left_id = model.vocab_hash.get(left_str) orelse return error.InvalidMerges;
        const right_id = model.vocab_hash.get(right_str) orelse return error.InvalidMerges;

        concat_buf.clearRetainingCapacity();
        try concat_buf.appendSlice(allocator, left_str);
        try concat_buf.appendSlice(allocator, right_str);
        const new_id = model.vocab_hash.get(concat_buf.items) orelse try addImplicitMergeToken(model, concat_buf.items);

        const pair_entry = try model.pair_merges.getOrPut(allocator, .{ .left = left_id, .right = right_id });
        if (pair_entry.found_existing) return error.InvalidMerges;
        pair_entry.value_ptr.* = .{ .rank = @intCast(rank), .new_id = new_id };
    }
}

fn addImplicitMergeToken(model: *BpeModel, token: []const u8) !i32 {
    const next_id: i32 = @intCast(model.vocab_size);
    const token_copy = try model.allocator.dupe(u8, token);
    errdefer model.allocator.free(token_copy);

    if (model.vocab_size >= model.id_to_token.len) {
        model.id_to_token = try model.allocator.realloc(model.id_to_token, model.vocab_size + 1);
    }
    model.id_to_token[model.vocab_size] = token_copy;
    try model.vocab_hash.put(model.allocator, token_copy, next_id);
    try model.merge_added_tokens.append(model.allocator, token_copy);
    model.vocab_size += 1;
    return next_id;
}

fn finalizeSpecialTokenIds(model: *BpeModel) void {
    const unk_slice = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&model.unk_token)), 0);
    if (model.vocab_hash.get(unk_slice)) |id| {
        model.unk_id = id;
    }

    if (model.vocab_hash.get("<bos>")) |id| {
        model.bos_id = id;
    } else if (model.vocab_hash.get("<s>")) |id| {
        model.bos_id = id;
    }

    if (model.vocab_hash.get("<eos>")) |id| {
        model.eos_id = id;
    } else if (model.vocab_hash.get("</s>")) |id| {
        model.eos_id = id;
    }
}

/// Find section start index (uses utils.findJsonSection internally)
fn findSectionStart(json: []const u8, key: []const u8) ?usize {
    const section = utils.findJsonSection(json, key) orelse return null;
    // Calculate offset from original json
    return @intFromPtr(section.ptr) - @intFromPtr(json.ptr);
}

fn parseIgnoreMergesOption(json: []const u8) bool {
    var search_scope = json;
    if (findSectionStart(json, "\"model\"")) |model_start| {
        const model_section = json[model_start..];
        if (model_section.len > 0 and model_section[0] == '{') {
            if (utils.findMatchingBrace(model_section, '{', '}')) |model_len| {
                search_scope = model_section[0..model_len];
            }
        }
    }

    if (json_utils.findJsonFieldValue(search_scope, "\"ignore_merges\"")) |value| {
        return std.mem.eql(u8, value, "true");
    }
    return false;
}

/// Build byte fallback lookup table for SentencePiece models
/// Looks for tokens like "<0x00>", "<0x01>", ..., "<0xFF>" in vocab
fn initByteFallback(model: *BpeModel) void {
    var found_count: usize = 0;

    // Look for <0xNN> pattern in vocab
    for (0..256) |byte_val| {
        // Build the token string "<0xNN>"
        var token_buf: [6]u8 = undefined;
        token_buf[0] = '<';
        token_buf[1] = '0';
        token_buf[2] = 'x';

        const hex_chars = "0123456789ABCDEF";
        token_buf[3] = hex_chars[byte_val >> 4];
        token_buf[4] = hex_chars[byte_val & 0x0F];
        token_buf[5] = '>';

        // Look up in vocab
        if (model.vocab_hash.get(token_buf[0..6])) |id| {
            model.byte_fallback_ids[byte_val] = id;
            found_count += 1;
        }
    }

    log.trace("tokenizer", "Byte fallback init", .{ .found_count = found_count, .vocab_size = model.vocab_hash.count() }, @src());
}

/// Build direct byte→vocab-ID lookup for single-byte tokens.
/// Avoids hash lookup in symbol init for the common case (ASCII).
fn initByteVocabIds(model: *BpeModel) void {
    for (0..256) |byte_val| {
        const single = [1]u8{@intCast(byte_val)};
        if (model.vocab_hash.get(&single)) |id| {
            model.byte_vocab_ids[byte_val] = id;
        }
    }
}

/// Build direct byte→vocab-ID for byte-level BPE (raw byte → vocab ID).
/// Maps each original byte value to the vocab ID of its byte_to_unicode
/// representation. This enables symbol init to use a single array lookup
/// per byte instead of UTF-8 parsing + hash lookup.
/// Sets use_raw_byte_init=true when all 256 byte_to_unicode mappings exist.
/// Missing vocab entries (id=-1) are safe: the byte-level collect path emits
/// unk_id for id=-1 symbols — same behavior as applyByteLevel + hash lookup.
fn initOrigByteVocabIds(model: *BpeModel) void {
    var all_mapped = true;
    for (0..256) |byte_val| {
        const encoded = model.byte_to_unicode[byte_val];
        if (encoded.len > 0) {
            model.orig_byte_vocab_ids[byte_val] = model.vocab_hash.get(encoded) orelse -1;
        } else {
            model.orig_byte_vocab_ids[byte_val] = -1;
            all_mapped = false;
        }
    }
    model.use_raw_byte_init = all_mapped;
}

fn initByteMap(model: *BpeModel) !void {
    var byte_values = [_]i32{0} ** 512;
    var codepoints = [_]i32{0} ** 512;
    var map_len: usize = 0;
    for (33..127) |b| {
        byte_values[map_len] = @intCast(b);
        codepoints[map_len] = @intCast(b);
        map_len += 1;
    }
    for (161..173) |b| {
        byte_values[map_len] = @intCast(b);
        codepoints[map_len] = @intCast(b);
        map_len += 1;
    }
    for (174..256) |b| {
        byte_values[map_len] = @intCast(b);
        codepoints[map_len] = @intCast(b);
        map_len += 1;
    }
    var offset_idx: usize = 0;
    for (0..256) |b| {
        var present = false;
        for (0..map_len) |map_idx| {
            if (byte_values[map_idx] == b) {
                present = true;
                break;
            }
        }
        if (!present) {
            byte_values[map_len] = @intCast(b);
            codepoints[map_len] = 256 + @as(i32, @intCast(offset_idx));
            map_len += 1;
            offset_idx += 1;
        }
    }

    for (0..map_len) |map_idx| {
        const codepoint = codepoints[map_idx];
        var utf8_buf: [4]u8 = undefined;
        const byte_len = utf8Encode(codepoint, &utf8_buf);
        const dup = try model.allocator.alloc(u8, @as(usize, byte_len));
        @memcpy(dup, utf8_buf[0..@as(usize, byte_len)]);
        model.byte_to_unicode[@as(usize, @intCast(byte_values[map_idx]))] = dup;
        if (codepoint >= 0 and codepoint < model.unicode_to_byte.len) {
            model.unicode_to_byte[@as(usize, @intCast(codepoint))] = byte_values[map_idx];
        }
    }
}

const IdList = std.ArrayListUnmanaged(i32);

/// Encode a word with added-token check (for C API vtable path).
fn encodeWordAppend(model: *BpeModel, tok: *ct.Tokenizer, word: []const u8, out_ids: *IdList) !void {
    // Check if entire word is an added token
    if (word.len < 128) {
        var word_z_buf: [128:0]u8 = undefined;
        @memcpy(word_z_buf[0..word.len], word);
        word_z_buf[word.len] = 0;
        if (tok_fns.tokenizer_added_token_find(tok, &word_z_buf)) |added| {
            try out_ids.append(model.allocator, added.*.id);
            return;
        }
    } else {
        const word_z = try model.allocator.dupeZ(u8, word);
        defer model.allocator.free(word_z);
        if (tok_fns.tokenizer_added_token_find(tok, word_z.ptr)) |added| {
            try out_ids.append(model.allocator, added.*.id);
            return;
        }
    }
    return encodeWordCore(model, tok, word, out_ids);
}

fn lookupWholeWordId(model: *BpeModel, word: []const u8, use_raw_byte_init: bool) ?i32 {
    if (use_raw_byte_init) {
        // Raw-byte fast path stores symbols as original bytes, but vocab keys
        // are byte-level encoded strings. Re-encode through byte_to_unicode.
        var enc_buf: [MAX_WORD_SYMBOLS * 4]u8 = undefined;
        var enc_len: usize = 0;
        for (word) |byte| {
            const encoded = model.byte_to_unicode[byte];
            if (encoded.len == 0 or enc_len + encoded.len > enc_buf.len) return null;
            @memcpy(enc_buf[enc_len..][0..encoded.len], encoded);
            enc_len += encoded.len;
        }
        return model.vocab_hash.get(enc_buf[0..enc_len]);
    }
    return model.vocab_hash.get(word);
}

const SymbolInitResult = struct {
    syms: []Symbol,
    n_syms: usize,
    used_raw_byte_init: bool,
    heap_syms: ?[]Symbol,
};

const CachedPair = struct { pos: i32, rank: i32, new_id: i32 };

fn initRawByteSymbols(model: *BpeModel, word: []const u8, syms: []Symbol) usize {
    var n_syms: usize = 0;
    for (word, 0..) |byte, i| {
        syms[n_syms] = .{
            .id = model.orig_byte_vocab_ids[byte],
            .start = @intCast(i),
            .len = 1,
            .prev = if (n_syms > 0) @as(i32, @intCast(n_syms - 1)) else -1,
            .next = -1,
        };
        if (n_syms > 0) syms[n_syms - 1].next = @intCast(n_syms);
        n_syms += 1;
    }
    return n_syms;
}

fn countUtf8Symbols(word: []const u8) !usize {
    var n_chars: usize = 0;
    var byte_idx: usize = 0;
    while (byte_idx < word.len) {
        const char_len = utf8CharLen(word[byte_idx]);
        if (byte_idx + char_len > word.len) return error.InvalidUtf8;
        byte_idx += char_len;
        n_chars += 1;
    }
    return n_chars;
}

fn initUtf8Symbols(model: *BpeModel, word: []const u8, syms: []Symbol) !usize {
    var n_syms: usize = 0;
    var byte_idx: usize = 0;
    while (byte_idx < word.len) {
        const char_len = utf8CharLen(word[byte_idx]);
        if (byte_idx + char_len > word.len) return error.InvalidUtf8;

        const id: i32 = if (char_len == 1)
            model.byte_vocab_ids[word[byte_idx]]
        else
            model.vocab_hash.get(word[byte_idx .. byte_idx + char_len]) orelse -1;

        syms[n_syms] = .{
            .id = id,
            .start = @intCast(byte_idx),
            .len = @intCast(char_len),
            .prev = if (n_syms > 0) @as(i32, @intCast(n_syms - 1)) else -1,
            .next = -1,
        };
        if (n_syms > 0) syms[n_syms - 1].next = @intCast(n_syms);
        byte_idx += char_len;
        n_syms += 1;
    }
    return n_syms;
}

fn initWordSymbols(
    model: *BpeModel,
    word: []const u8,
    use_raw_byte_init: bool,
    stack_syms: *[MAX_WORD_SYMBOLS]Symbol,
) !SymbolInitResult {
    var heap_syms: ?[]Symbol = null;
    var syms: []Symbol = stack_syms[0..];

    if (use_raw_byte_init) {
        if (word.len > MAX_WORD_SYMBOLS) {
            heap_syms = try model.allocator.alloc(Symbol, word.len);
            syms = heap_syms.?;
        }
        return .{
            .syms = syms,
            .n_syms = initRawByteSymbols(model, word, syms),
            .used_raw_byte_init = true,
            .heap_syms = heap_syms,
        };
    }

    if (word.len > MAX_WORD_SYMBOLS) {
        const n_chars = try countUtf8Symbols(word);
        if (n_chars > MAX_WORD_SYMBOLS) {
            heap_syms = try model.allocator.alloc(Symbol, n_chars);
            syms = heap_syms.?;
        }
    }
    return .{
        .syms = syms,
        .n_syms = try initUtf8Symbols(model, word, syms),
        .used_raw_byte_init = false,
        .heap_syms = heap_syms,
    };
}

fn appendSymbolIds(model: *BpeModel, is_byte_level: bool, word: []const u8, sym: *const Symbol, out_ids: *IdList) !void {
    if (sym.id >= 0) {
        try out_ids.append(model.allocator, sym.id);
    } else if (!is_byte_level) {
        const token_bytes = word[sym.start .. sym.start + sym.len];
        for (token_bytes) |byte_val| {
            const fallback_id = model.byte_fallback_ids[byte_val];
            try out_ids.append(model.allocator, if (fallback_id >= 0) fallback_id else model.unk_id);
        }
    } else {
        try out_ids.append(model.allocator, model.unk_id);
    }
}

fn appendSymbolIdsAssumeCapacity(model: *BpeModel, is_byte_level: bool, word: []const u8, sym: *const Symbol, out_ids: *IdList) void {
    if (sym.id >= 0) {
        out_ids.appendAssumeCapacity(sym.id);
    } else if (!is_byte_level) {
        const token_bytes = word[sym.start .. sym.start + sym.len];
        for (token_bytes) |byte_val| {
            const fallback_id = model.byte_fallback_ids[byte_val];
            out_ids.appendAssumeCapacity(if (fallback_id >= 0) fallback_id else model.unk_id);
        }
    } else {
        out_ids.appendAssumeCapacity(model.unk_id);
    }
}

fn applySymbolMerge(syms: []Symbol, left_pos: i32, new_id: i32) void {
    const left = &syms[@intCast(left_pos)];
    const right_idx = left.next;
    const right = &syms[@intCast(right_idx)];
    left.id = new_id;
    left.len += right.len;
    left.next = right.next;
    if (right.next >= 0) syms[@intCast(right.next)].prev = left_pos;
    right.id = -2;
}

fn encodeTwoSymbolWord(model: *BpeModel, is_byte_level: bool, word: []const u8, syms: []Symbol, out_ids: *IdList) !void {
    const s0 = &syms[0];
    const s1 = &syms[1];
    if (s0.id >= 0 and s1.id >= 0) {
        if (model.pair_merges.get(.{ .left = s0.id, .right = s1.id })) |info| {
            try out_ids.append(model.allocator, info.new_id);
        } else {
            try out_ids.ensureUnusedCapacity(model.allocator, 2);
            out_ids.appendAssumeCapacity(s0.id);
            out_ids.appendAssumeCapacity(s1.id);
        }
        return;
    }

    try appendSymbolIds(model, is_byte_level, word, s0, out_ids);
    try appendSymbolIds(model, is_byte_level, word, s1, out_ids);
}

fn mergeShortWord(model: *BpeModel, syms: []Symbol) void {
    while (true) {
        var best_rank: i32 = std.math.maxInt(i32);
        var best_pos: i32 = -1;
        var best_new_id: i32 = -1;

        var si: i32 = 0;
        while (si >= 0) {
            const sym = &syms[@intCast(si)];
            if (sym.next >= 0 and sym.id >= 0) {
                const right = &syms[@intCast(sym.next)];
                if (right.id >= 0) {
                    if (model.pair_merges.get(.{ .left = sym.id, .right = right.id })) |info| {
                        if (info.rank < best_rank or
                            (info.rank == best_rank and si < best_pos))
                        {
                            best_rank = info.rank;
                            best_pos = si;
                            best_new_id = info.new_id;
                        }
                    }
                }
            }
            si = sym.next;
        }
        if (best_pos < 0) break;
        applySymbolMerge(syms, best_pos, best_new_id);
    }
}

fn findBestCachedPairIndex(pair_cache: []const CachedPair, n_cached: usize) usize {
    var best_idx: usize = 0;
    for (1..n_cached) |i| {
        if (pair_cache[i].rank < pair_cache[best_idx].rank or
            (pair_cache[i].rank == pair_cache[best_idx].rank and
                pair_cache[i].pos < pair_cache[best_idx].pos))
        {
            best_idx = i;
        }
    }
    return best_idx;
}

fn removeCachedPairAt(pair_cache: []CachedPair, pos_to_cache: []i32, n_cached: *usize, stale_pos: i32) void {
    if (stale_pos < 0) return;
    const cache_idx = pos_to_cache[@intCast(stale_pos)];
    if (cache_idx < 0 or @as(usize, @intCast(cache_idx)) >= n_cached.*) return;

    const removed_idx: usize = @intCast(cache_idx);
    n_cached.* -= 1;
    if (removed_idx < n_cached.*) {
        pair_cache[removed_idx] = pair_cache[n_cached.*];
        pos_to_cache[@intCast(pair_cache[removed_idx].pos)] = @intCast(removed_idx);
    }
    pos_to_cache[@intCast(stale_pos)] = -1;
}

fn addCachedPairAt(model: *BpeModel, syms: []Symbol, pair_cache: []CachedPair, pos_to_cache: []i32, n_cached: *usize, pos: i32) void {
    if (pos < 0) return;
    const left = &syms[@intCast(pos)];
    if (left.id < 0 or left.next < 0) return;
    const right = &syms[@intCast(left.next)];
    if (right.id < 0) return;

    if (model.pair_merges.get(.{ .left = left.id, .right = right.id })) |info| {
        std.debug.assert(n_cached.* < pair_cache.len);
        pos_to_cache[@intCast(pos)] = @intCast(n_cached.*);
        pair_cache[n_cached.*] = .{ .pos = pos, .rank = info.rank, .new_id = info.new_id };
        n_cached.* += 1;
    }
}

fn mergeCachedPairs(model: *BpeModel, syms: []Symbol, n_syms: usize) !void {
    var pair_cache_stack: [MAX_WORD_SYMBOLS]CachedPair = undefined;
    var pair_cache_heap: ?[]CachedPair = null;
    defer if (pair_cache_heap) |cache| model.allocator.free(cache);
    var pair_cache: []CachedPair = &pair_cache_stack;

    var pos_to_cache_stack: [MAX_WORD_SYMBOLS]i32 = undefined;
    var pos_to_cache_heap: ?[]i32 = null;
    defer if (pos_to_cache_heap) |cache| model.allocator.free(cache);
    var pos_to_cache: []i32 = &pos_to_cache_stack;

    if (n_syms > MAX_WORD_SYMBOLS) {
        pair_cache_heap = try model.allocator.alloc(CachedPair, n_syms);
        pair_cache = pair_cache_heap.?;
        pos_to_cache_heap = try model.allocator.alloc(i32, n_syms);
        pos_to_cache = pos_to_cache_heap.?;
    }
    @memset(pos_to_cache[0..n_syms], -1);

    var n_cached: usize = 0;
    var si: i32 = 0;
    while (si >= 0) {
        addCachedPairAt(model, syms, pair_cache, pos_to_cache, &n_cached, si);
        si = syms[@intCast(si)].next;
    }

    while (n_cached > 0) {
        const best_idx = findBestCachedPairIndex(pair_cache, n_cached);
        const best = pair_cache[best_idx];

        n_cached -= 1;
        if (best_idx < n_cached) {
            pair_cache[best_idx] = pair_cache[n_cached];
            pos_to_cache[@intCast(pair_cache[best_idx].pos)] = @intCast(best_idx);
        }
        pos_to_cache[@intCast(best.pos)] = -1;

        const left = &syms[@intCast(best.pos)];
        if (left.id < 0 or left.next < 0) continue;
        const right_idx = left.next;
        const right = &syms[@intCast(right_idx)];
        if (right.id < 0) continue;

        applySymbolMerge(syms, best.pos, best.new_id);
        removeCachedPairAt(pair_cache, pos_to_cache, &n_cached, right_idx);
        removeCachedPairAt(pair_cache, pos_to_cache, &n_cached, left.prev);
        addCachedPairAt(model, syms, pair_cache, pos_to_cache, &n_cached, left.prev);
        addCachedPairAt(model, syms, pair_cache, pos_to_cache, &n_cached, best.pos);
    }
}

fn mergeWordSymbols(model: *BpeModel, syms: []Symbol, n_syms: usize) !void {
    if (model.pair_merges.count() == 0) return;
    if (n_syms >= 3 and n_syms <= 5) {
        mergeShortWord(model, syms);
    } else if (n_syms >= 6) {
        try mergeCachedPairs(model, syms, n_syms);
    }
}

fn containsUnknownSymbol(syms: []const Symbol) bool {
    var si: i32 = 0;
    while (si >= 0) {
        if (syms[@intCast(si)].id < 0) return true;
        si = syms[@intCast(si)].next;
    }
    return false;
}

fn countOutputIds(is_byte_level: bool, syms: []const Symbol) usize {
    var n_output: usize = 0;
    var ci: i32 = 0;
    while (ci >= 0) {
        const sym = &syms[@intCast(ci)];
        n_output += if (sym.id >= 0 or is_byte_level) 1 else sym.len;
        ci = sym.next;
    }
    return n_output;
}

fn appendMergedSymbolIds(model: *BpeModel, is_byte_level: bool, word: []const u8, syms: []const Symbol, out_ids: *IdList) !void {
    try out_ids.ensureUnusedCapacity(model.allocator, countOutputIds(is_byte_level, syms));

    var si: i32 = 0;
    while (si >= 0) {
        const sym = &syms[@intCast(si)];
        appendSymbolIdsAssumeCapacity(model, is_byte_level, word, sym, out_ids);
        si = sym.next;
    }
}

/// Core BPE encode: symbol init → merge loop → collect. No added-token check.
/// Used by the direct encode path where added tokens are pre-separated.
fn encodeWordCore(model: *BpeModel, tok: *ct.Tokenizer, word: []const u8, out_ids: *IdList) !void {
    const allocator = model.allocator;
    const is_byte_level = tok.pretokenizer.byte_level != 0;
    const use_raw_byte_init = is_byte_level and model.use_raw_byte_init and tok.pretokenizer.is_sequence == 0;

    // HuggingFace `ignore_merges=true`: prefer direct vocab hit for the full
    // pretokenized word before running the merge algorithm.
    if (model.ignore_merges) {
        if (lookupWholeWordId(model, word, use_raw_byte_init)) |id| {
            try out_ids.append(allocator, id);
            return;
        }
    }

    var stack_syms: [MAX_WORD_SYMBOLS]Symbol = undefined;
    const symbol_init = try initWordSymbols(model, word, use_raw_byte_init, &stack_syms);
    defer if (symbol_init.heap_syms) |heap_syms| allocator.free(heap_syms);

    const syms = symbol_init.syms;
    const n_syms = symbol_init.n_syms;
    if (n_syms == 0) return;

    if (n_syms == 1) {
        try appendSymbolIds(model, is_byte_level, word, &syms[0], out_ids);
        return;
    }

    if (n_syms == 2) {
        try encodeTwoSymbolWord(model, is_byte_level, word, syms, out_ids);
        return;
    }

    try mergeWordSymbols(model, syms, n_syms);

    if (containsUnknownSymbol(syms)) {
        if (lookupWholeWordId(model, word, symbol_init.used_raw_byte_init)) |id| {
            try out_ids.append(allocator, id);
            return;
        }
    }

    try appendMergedSymbolIds(model, is_byte_level, word, syms, out_ids);
}

// ============= C API callbacks =============

/// Encode a word via the C API path (vtable). Allocates IDs for the encoding struct.
fn bpe_encode_via_encoding(model: *BpeModel, tok: *ct.Tokenizer, text: []const u8, enc: *ct.TokenizerEncoding) c_int {
    var ids_list = IdList{};
    encodeWordAppend(model, tok, text, &ids_list) catch |err| {
        log.trace("tokenizer", "encodeWord failed", .{ .err = @errorName(err) }, @src());
        tok_fns.tokenizer_set_error(tok, "BPE encode failed");
        ids_list.deinit(model.allocator);
        return -1;
    };

    // Transfer ownership: toOwnedSlice gives exact-sized allocation
    const owned = ids_list.toOwnedSlice(model.allocator) catch {
        ids_list.deinit(model.allocator);
        return -1;
    };
    enc.ids_len = owned.len;
    enc.tokens_len = owned.len;
    enc.ids = @ptrCast(owned.ptr);
    return 0;
}

fn bpe_encode(tok: *ct.Tokenizer, input: [*c]const u8, enc: *ct.TokenizerEncoding) c_int {
    if (tok.model == null) return -1;
    const model = @as(*BpeModel, @ptrCast(@alignCast(tok.model.?)));
    const text = std.mem.sliceTo(input, 0);
    return bpe_encode_via_encoding(model, tok, text, enc);
}

/// Encode with explicit length (supports embedded null bytes) - implementation
fn bpe_encode_slice_impl(model: *BpeModel, tok: *ct.Tokenizer, text: []const u8, enc: *ct.TokenizerEncoding) c_int {
    return bpe_encode_via_encoding(model, tok, text, enc);
}

/// C API vtable adapter: extracts model from tokenizer and calls impl
fn bpe_encode_slice(tok: *ct.Tokenizer, text: []const u8, enc: *ct.TokenizerEncoding) c_int {
    if (tok.model == null) return -1;
    const model = @as(*BpeModel, @ptrCast(@alignCast(tok.model.?)));
    return bpe_encode_slice_impl(model, tok, text, enc);
}

/// Result of looking up an added token by ID.
const AddedTokenInfo = struct {
    content: [*:0]const u8,
    is_special: bool,
};

fn findAddedTokenInfoById(tok: *ct.Tokenizer, id: i32) ?AddedTokenInfo {
    var cur = tok.added;
    while (cur) |node| {
        if (node.id == id) {
            if (node.content) |content| {
                return .{
                    .content = @ptrCast(content),
                    .is_special = node.special != 0,
                };
            }
        }
        cur = node.next;
    }
    return null;
}

fn findAddedTokenById(tok: *ct.Tokenizer, id: i32) ?[*:0]const u8 {
    if (findAddedTokenInfoById(tok, id)) |info| {
        return info.content;
    }
    return null;
}

fn bpe_decode_with_options_impl(model: *BpeModel, tok: *ct.Tokenizer, ids: [*c]const i32, ids_len: usize, out: *[*c]u8, out_len: *usize, options: ct.DecodeOptions) c_int {
    const allocator = model.allocator;
    const unk_ptr: [*:0]const u8 = @ptrCast(&model.unk_token);

    const TokenInfo = struct { slice: []const u8, is_special: bool };
    const tokens = allocator.alloc(TokenInfo, ids_len) catch return -1;
    defer allocator.free(tokens);

    for (tokens, 0..) |*slot, id_idx| {
        const id = ids[id_idx];
        // Added tokens take precedence over plain vocab lookup so their
        // special-token semantics remain visible during decode.
        if (findAddedTokenInfoById(tok, id)) |added_info| {
            slot.* = .{ .slice = std.mem.sliceTo(added_info.content, 0), .is_special = added_info.is_special };
            continue;
        }
        // Check id_to_token array (regular vocab tokens are not special)
        if (id >= 0 and @as(usize, @intCast(id)) < model.id_to_token.len) {
            if (model.id_to_token[@as(usize, @intCast(id))]) |token| {
                slot.* = .{ .slice = token, .is_special = false };
                continue;
            }
        }
        slot.* = .{ .slice = std.mem.sliceTo(unk_ptr, 0), .is_special = false };
    }

    var result = std.ArrayListUnmanaged(u8){};
    defer result.deinit(allocator);

    for (tokens) |token| {
        // Skip special tokens if requested
        if (options.skip_special_tokens and token.is_special) {
            continue;
        }

        if (token.is_special) {
            // Special tokens are output as-is (no byte decoding)
            result.appendSlice(allocator, token.slice) catch return -1;
        } else {
            // ByteFallback: handle <0xXX> tokens (SentencePiece byte fallback)
            // These represent raw bytes that couldn't be encoded as regular tokens
            if (token.slice.len == 6 and
                token.slice[0] == '<' and
                token.slice[1] == '0' and
                token.slice[2] == 'x' and
                token.slice[5] == '>')
            {
                // Parse hex value from <0xXX>
                const hex_chars = token.slice[3..5];
                const byte_val = std.fmt.parseInt(u8, hex_chars, 16) catch {
                    // If parse fails, output as-is
                    result.appendSlice(allocator, token.slice) catch return -1;
                    continue;
                };
                result.append(allocator, byte_val) catch return -1;
                continue;
            }

            var idx: usize = 0;
            const use_byte_map = tok.pretokenizer.byte_level != 0;
            while (idx < token.slice.len) {
                const codepoint = utf8Decode(token.slice, &idx);
                // Only a configured Metaspace decoder should map ▁ back to space.
                if (codepoint == 0x2581 and tok.decoder.metaspace != 0) {
                    result.append(allocator, ' ') catch return -1;
                } else if (use_byte_map and codepoint >= 0 and codepoint < model.unicode_to_byte.len and model.unicode_to_byte[@as(usize, @intCast(codepoint))] >= 0) {
                    // GPT-2 byte-level: map Unicode codepoint back to original byte
                    result.append(allocator, @intCast(model.unicode_to_byte[@as(usize, @intCast(codepoint))])) catch return -1;
                } else if (codepoint >= 0) {
                    var utf8_buf: [4]u8 = undefined;
                    const byte_len = utf8Encode(codepoint, &utf8_buf);
                    result.appendSlice(allocator, utf8_buf[0..@as(usize, byte_len)]) catch return -1;
                }
            }
        }
    }

    // Strip leading spaces if configured (SentencePiece decoder behavior)
    // The "Strip" decoder with start=N removes N leading spaces from output
    var strip_start_count: usize = @intCast(@max(0, tok.decoder.strip_start));
    while (strip_start_count > 0 and result.items.len > 0 and result.items[0] == ' ') {
        _ = result.orderedRemove(0);
        strip_start_count -= 1;
    }

    // Strip trailing spaces if configured
    // The "Strip" decoder with stop=N removes N trailing spaces from output
    var strip_stop_count: usize = @intCast(@max(0, tok.decoder.strip_stop));
    while (strip_stop_count > 0 and result.items.len > 0 and result.items[result.items.len - 1] == ' ') {
        _ = result.pop();
        strip_stop_count -= 1;
    }

    // Strip leading space added by add_prefix_space (pretokenizer or Metaspace decoder).
    // Skip when strip_start already removed leading spaces — the Strip decoder in a
    // Sequence (e.g. Mistral) handles this; applying both would double-strip.
    if (tok.decoder.strip_start == 0 and (tok.pretokenizer.add_prefix_space != 0 or tok.decoder.add_prefix_space != 0) and result.items.len > 0 and result.items[0] == ' ') {
        _ = result.orderedRemove(0);
    }

    // Return actual length (before null terminator)
    out_len.* = result.items.len;

    // Add null terminator for C string convention
    result.append(allocator, 0) catch return -1;
    const owned_buf = result.toOwnedSlice(allocator) catch return -1;
    out.* = owned_buf.ptr;
    return 0;
}

fn bpe_destroy_impl(model: *BpeModel, tok: *ct.Tokenizer) void {
    tok.model = null;
    deinitBpeModelStorage(model, true);
    model.allocator.destroy(model);
}

/// Create a BPE tokenizer from a JSON buffer.
pub fn createBpeTokenizer(allocator: std.mem.Allocator, json_buffer: []const u8, json_owned: bool) !*ct.Tokenizer {
    var tok = try allocator.create(ct.Tokenizer);
    errdefer allocator.destroy(tok);

    tok.* = std.mem.zeroes(ct.Tokenizer);
    tok.type = ct.ModelType.bpe;
    tok.normalizer.lowercase = 0;
    tok.normalizer.nfd = 0;
    tok.postproc.cls_id = -1;
    tok.postproc.sep_id = -1;
    tok.postproc.add_special = 0;

    const model = create(allocator, json_buffer, json_owned) catch |err| {
        log.debug("tokenizer", "BPE model creation failed", .{
            .reason = @errorName(err),
            .json_bytes = json_buffer.len,
            .json_owned = @intFromBool(json_owned),
        }, @src());
        return err;
    };

    tok.model = model;
    model.owner = tok;
    errdefer bpe_destroy_impl(model, tok);

    if (tok_fns.tokenizer_pretokenizer_set(&tok.pretokenizer, DEFAULT_PATTERN.ptr) != 0) {
        tok_fns.tokenizer_set_error(tok, "Failed to compile BPE regex");
        log.warn("tokenizer", "BPE default pretokenizer init failed", .{
            .json_bytes = json_buffer.len,
        });
        return error.PretokenizerInitFailed;
    }

    return tok;
}

// =============================================================================
// Tests
// =============================================================================

fn destroyTestJsonBpeModel(model: *BpeModel) void {
    deinitBpeModelStorage(model, false);
    model.allocator.destroy(model);
}

fn expectCreateInvalidVocab(json: []const u8) !void {
    const model = create(std.testing.allocator, json, false) catch |err| {
        try std.testing.expectEqual(error.InvalidVocab, err);
        return;
    };
    defer destroyTestJsonBpeModel(model);
    return error.TestUnexpectedResult;
}

test "findVocabSize with simple vocab" {
    const json =
        \\{"vocab": {"a": 0, "b": 1, "c": 2}}
    ;
    const size = try findVocabSize(json);
    try std.testing.expectEqual(@as(usize, 3), size);
}

test "findVocabSize with gaps in IDs" {
    const json =
        \\{"vocab": {"a": 0, "b": 5, "c": 10}}
    ;
    const size = try findVocabSize(json);
    try std.testing.expectEqual(@as(usize, 11), size);
}

test "findVocabSize with large IDs" {
    const json =
        \\{"vocab": {"token1": 100, "token2": 50000, "token3": 25}}
    ;
    const size = try findVocabSize(json);
    try std.testing.expectEqual(@as(usize, 50001), size);
}

test "findVocabSize with whitespace" {
    const json =
        \\{
        \\  "vocab": {
        \\    "a": 0,
        \\    "b": 1,
        \\    "c": 2
        \\  }
        \\}
    ;
    const size = try findVocabSize(json);
    try std.testing.expectEqual(@as(usize, 3), size);
}

test "findVocabSize reports oversize sparse ids" {
    const json =
        \\{"vocab": {"a": 2147483647}}
    ;
    const size = try findVocabSize(json);
    try std.testing.expect(size > MAX_MODEL_VOCAB_SIZE);
}

test "findVocabSize ignores numeric fields outside vocab" {
    const json =
        \\{"model": {"dropout": 999999}, "vocab": {"a": 0, "b": 1}, "merges": [["a", "b"]]}
    ;
    const size = try findVocabSize(json);
    try std.testing.expectEqual(@as(usize, 2), size);
}

test "findSectionStart finds vocab section" {
    const json =
        \\{"model": "bpe", "vocab": {"a": 0}, "merges": []}
    ;
    const start = findSectionStart(json, "\"vocab\"");
    try std.testing.expect(start != null);
    try std.testing.expect(start.? > 0);
}

test "findSectionStart finds merges section" {
    const json =
        \\{"model": "bpe", "vocab": {"a": 0}, "merges": []}
    ;
    const start = findSectionStart(json, "\"merges\"");
    try std.testing.expect(start != null);
    try std.testing.expect(start.? > 0);
}

test "findSectionStart returns null for missing section" {
    const json =
        \\{"model": "bpe", "vocab": {"a": 0}}
    ;
    const start = findSectionStart(json, "\"missing\"");
    try std.testing.expect(start == null);
}

test "parseVocabAndMerges rejects malformed vocab ids" {
    const cases = [_][]const u8{
        \\{"vocab": {"a": -1}, "merges": []}
        ,
        \\{"vocab": {"a": +1}, "merges": []}
        ,
        \\{"vocab": {"a": "1"}, "merges": []}
        ,
        \\{"vocab": {"a": 1.0}, "merges": []}
        ,
        \\{"vocab": {"a": 1e3}, "merges": []}
        ,
        \\{"vocab": {"a": null}, "merges": []}
        ,
        \\{"vocab": {"a": {}}, "merges": []}
        ,
        \\{"vocab": {"a": []}, "merges": []}
        ,
        \\{"vocab": {"a": }, "merges": []}
        ,
        \\{"vocab": {"a": 1x}, "merges": []}
    };

    for (cases) |json| {
        try expectCreateInvalidVocab(json);
    }
}

test "parseVocabAndMerges rejects duplicate vocab ids and tokens" {
    try expectCreateInvalidVocab(
        \\{"vocab": {"a": 0, "b": 0}, "merges": []}
    );
    try expectCreateInvalidVocab(
        \\{"vocab": {"a": 0, "a": 1}, "merges": []}
    );
}

// =============================================================================
// Unit tests for critical internal functions
// =============================================================================

test "parseVocabAndMerges with minimal valid JSON" {
    const json =
        \\{"vocab": {"a": 0, "b": 1, "c": 2}, "merges": [["a", "b"]]}
    ;
    const allocator = std.testing.allocator;

    const model = try create(allocator, json, false);
    defer destroyTestJsonBpeModel(model);

    // Verify vocab was parsed (3 explicit + 1 auto-added from merge "a b" → "ab")
    try std.testing.expect(model.vocab_hash.count() == 4);
    try std.testing.expect(model.vocab_hash.get("a").? == 0);
    try std.testing.expect(model.vocab_hash.get("b").? == 1);
    try std.testing.expect(model.vocab_hash.get("c").? == 2);
    try std.testing.expect(model.vocab_hash.get("ab").? == 3);

    // Verify id_to_token was populated (including auto-added "ab")
    try std.testing.expect(model.id_to_token[0] != null);
    try std.testing.expect(model.id_to_token[1] != null);
    try std.testing.expect(model.id_to_token[2] != null);
    try std.testing.expect(model.id_to_token[3] != null);
    try std.testing.expectEqualStrings("ab", model.id_to_token[3].?);

    // Verify merges were parsed (array format)
    try std.testing.expect(model.merges.count() > 0);

    // Verify pair merge table was built
    try std.testing.expect(model.pair_merges.count() == 1);
}

test "parseVocabAndMerges with string format merges" {
    const json =
        \\{"vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]}
    ;
    const allocator = std.testing.allocator;

    const model = try create(allocator, json, false);
    defer destroyTestJsonBpeModel(model);

    // Verify vocab
    try std.testing.expect(model.vocab_hash.count() == 3);

    // Verify string format merges were parsed
    try std.testing.expect(model.merges.count() > 0);
    const rank = model.merges.get("a b");
    try std.testing.expect(rank != null);
    try std.testing.expect(rank.? == 0);
}

test "parseVocabAndMerges without merges (SentencePiece style)" {
    const json =
        \\{"vocab": {"<unk>": 0, "a": 1, "b": 2}}
    ;
    const allocator = std.testing.allocator;

    const model = try create(allocator, json, false);
    defer destroyTestJsonBpeModel(model);

    // Verify vocab was parsed
    try std.testing.expect(model.vocab_hash.count() == 3);

    // Verify no merges (SentencePiece models don't have merges)
    try std.testing.expect(model.merges.count() == 0);

    // Verify unk_id was set
    try std.testing.expect(model.unk_id == 0);
}

test "parseVocabAndMerges with escaped characters" {
    const json =
        \\{"vocab": {"a": 0, "\"": 1, "\\": 2}, "merges": []}
    ;
    const allocator = std.testing.allocator;

    const model = try create(allocator, json, false);
    defer destroyTestJsonBpeModel(model);

    // Verify escaped characters were parsed correctly
    try std.testing.expect(model.vocab_hash.count() == 3);
    try std.testing.expect(model.vocab_hash.get("a") != null);
    try std.testing.expect(model.vocab_hash.get("\"") != null);
    try std.testing.expect(model.vocab_hash.get("\\") != null);
}

test "parseVocabAndMerges finds special tokens" {
    const json =
        \\{"vocab": {"<s>": 0, "</s>": 1, "a": 2}, "merges": []}
    ;
    const allocator = std.testing.allocator;

    const model = try create(allocator, json, false);
    defer destroyTestJsonBpeModel(model);

    // Verify bos_id and eos_id were set
    try std.testing.expect(model.bos_id == 0);
    try std.testing.expect(model.eos_id == 1);
}

test "initByteMap creates valid mappings" {
    const allocator = std.testing.allocator;

    // Create a minimal model for testing
    var model = try allocator.create(BpeModel);
    defer allocator.destroy(model);

    model.* = .{
        .allocator = allocator,
        .json_buffer = "",
        .json_owned = false,
        .id_to_token = &[_]?[]const u8{},
        .vocab_hash = .{},
        .merges = .{},
        .pair_merges = .{},
        .merge_added_tokens = .{},
        .merge_strings = .{},
        .byte_to_unicode = undefined,
        .unicode_to_byte = [_]i32{-1} ** 65536,
        .byte_fallback_ids = [_]i32{-1} ** 256,
        .byte_vocab_ids = [_]i32{-1} ** 256,
        .orig_byte_vocab_ids = [_]i32{-1} ** 256,
        .use_raw_byte_init = false,
        .ignore_merges = false,
        .unk_token = undefined,
        .unk_id = 0,
        .bos_id = -1,
        .eos_id = -1,
        .vocab_size = 0,
        .owner = null,
    };

    // Initialize byte_to_unicode
    for (&model.byte_to_unicode) |*slot| {
        slot.* = "";
    }

    try initByteMap(model);
    defer {
        for (model.byte_to_unicode) |s| {
            if (s.len > 0) allocator.free(s);
        }
    }

    // Verify printable ASCII characters map to themselves
    for (33..127) |b| {
        const mapping = model.byte_to_unicode[b];
        try std.testing.expect(mapping.len > 0);

        // Decode the first UTF-8 character
        var idx: usize = 0;
        const codepoint = utf8Decode(mapping, &idx);
        try std.testing.expect(codepoint == @as(i32, @intCast(b)));
    }

    // Verify extended ASCII range maps to themselves
    for (161..173) |b| {
        const mapping = model.byte_to_unicode[b];
        try std.testing.expect(mapping.len > 0);
    }

    for (174..256) |b| {
        const mapping = model.byte_to_unicode[b];
        try std.testing.expect(mapping.len > 0);
    }

    // Verify all bytes have mappings
    for (model.byte_to_unicode) |mapping| {
        try std.testing.expect(mapping.len > 0);
    }

    // Verify unicode_to_byte reverse mapping is consistent
    for (33..127) |b| {
        const mapping = model.byte_to_unicode[b];
        var idx: usize = 0;
        const codepoint = utf8Decode(mapping, &idx);
        if (codepoint >= 0 and codepoint < model.unicode_to_byte.len) {
            const reverse = model.unicode_to_byte[@as(usize, @intCast(codepoint))];
            try std.testing.expect(reverse == @as(i32, @intCast(b)));
        }
    }
}

test "initByteMap handles control characters with offset" {
    const allocator = std.testing.allocator;

    var model = try allocator.create(BpeModel);
    defer allocator.destroy(model);

    model.* = .{
        .allocator = allocator,
        .json_buffer = "",
        .json_owned = false,
        .id_to_token = &[_]?[]const u8{},
        .vocab_hash = .{},
        .merges = .{},
        .pair_merges = .{},
        .merge_added_tokens = .{},
        .merge_strings = .{},
        .byte_to_unicode = undefined,
        .unicode_to_byte = [_]i32{-1} ** 65536,
        .byte_fallback_ids = [_]i32{-1} ** 256,
        .byte_vocab_ids = [_]i32{-1} ** 256,
        .orig_byte_vocab_ids = [_]i32{-1} ** 256,
        .use_raw_byte_init = false,
        .ignore_merges = false,
        .unk_token = undefined,
        .unk_id = 0,
        .bos_id = -1,
        .eos_id = -1,
        .vocab_size = 0,
        .owner = null,
    };

    for (&model.byte_to_unicode) |*slot| {
        slot.* = "";
    }

    try initByteMap(model);
    defer {
        for (model.byte_to_unicode) |s| {
            if (s.len > 0) allocator.free(s);
        }
    }

    // Control characters (0-32, 127-160, 173) should be mapped to offset codepoints (256+)
    const control_bytes = [_]usize{ 0, 10, 13, 32, 127, 128, 160, 173 };
    for (control_bytes) |b| {
        const mapping = model.byte_to_unicode[b];
        try std.testing.expect(mapping.len > 0);

        var idx: usize = 0;
        const codepoint = utf8Decode(mapping, &idx);
        // Control chars should be mapped to codepoints >= 256 (offset range)
        try std.testing.expect(codepoint >= 256 or codepoint == @as(i32, @intCast(b)));
    }
}

test "encodeWordDirect handles long words above MAX_WORD_SYMBOLS without cache overflow" {
    const allocator = std.testing.allocator;
    const json =
        \\{"vocab": {"a": 0}, "merges": [["a", "a"]]}
    ;

    const tokenizer = try createBpeTokenizer(allocator, json, false);
    defer allocator.destroy(tokenizer);
    defer tokenizer.destroy();

    const model: *BpeModel = @ptrCast(@alignCast(tokenizer.model.?));
    const merged_id = model.vocab_hash.get("aa").?;

    const word_len = MAX_WORD_SYMBOLS * 2 + 1;
    const long_word = try allocator.alloc(u8, word_len);
    defer allocator.free(long_word);
    @memset(long_word, 'a');

    var first_ids = IdList{};
    defer first_ids.deinit(allocator);
    try model.encodeWordDirect(tokenizer, long_word, &first_ids);

    var second_ids = IdList{};
    defer second_ids.deinit(allocator);
    try model.encodeWordDirect(tokenizer, long_word, &second_ids);

    const expected_token_count = word_len / 2 + 1;
    try std.testing.expectEqual(expected_token_count, first_ids.items.len);
    try std.testing.expectEqualSlices(i32, first_ids.items, second_ids.items);

    for (first_ids.items[0 .. first_ids.items.len - 1]) |id| {
        try std.testing.expectEqual(merged_id, id);
    }
    try std.testing.expectEqual(@as(i32, 0), first_ids.items[first_ids.items.len - 1]);
}

test "encodeWordDirect keeps merge behavior when ignore_merges is false" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "model": {
        \\    "type": "BPE",
        \\    "ignore_merges": false,
        \\    "vocab": {"<unk>": 0, "a": 1, "b": 2, "c": 3, "ab": 4, "abc": 5},
        \\    "merges": [["a", "b"]]
        \\  }
        \\}
    ;

    const tokenizer = try createBpeTokenizer(allocator, json, false);
    defer allocator.destroy(tokenizer);
    defer tokenizer.destroy();

    const model: *BpeModel = @ptrCast(@alignCast(tokenizer.model.?));
    var ids = IdList{};
    defer ids.deinit(allocator);
    try model.encodeWordDirect(tokenizer, "abc", &ids);
    try std.testing.expectEqualSlices(i32, &.{ 4, 3 }, ids.items);
}

test "encodeWordDirect prefers direct token when ignore_merges is true" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "model": {
        \\    "type": "BPE",
        \\    "ignore_merges": true,
        \\    "vocab": {"<unk>": 0, "a": 1, "b": 2, "c": 3, "ab": 4, "abc": 5},
        \\    "merges": [["a", "b"]]
        \\  }
        \\}
    ;

    const tokenizer = try createBpeTokenizer(allocator, json, false);
    defer allocator.destroy(tokenizer);
    defer tokenizer.destroy();

    const model: *BpeModel = @ptrCast(@alignCast(tokenizer.model.?));
    var ids = IdList{};
    defer ids.deinit(allocator);
    try model.encodeWordDirect(tokenizer, "abc", &ids);
    try std.testing.expectEqualSlices(i32, &.{5}, ids.items);
}
