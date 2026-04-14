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

/// BPE model. Vocab and merges are parsed synchronously during creation.
pub const BpeModel = struct {
    allocator: std.mem.Allocator,

    // Raw JSON buffer
    json_buffer: []const u8,
    json_owned: bool,

    // Vocab: array indexed by ID (zero-copy pointers into JSON)
    // id_to_token[i] points directly into json_buffer for token with ID i
    id_to_token: []?[]const u8,
    vocab_hash: std.StringHashMapUnmanaged(i32),

    // Merges: string-based hash map (used during init, kept for edge cases)
    merges: std.StringHashMapUnmanaged(i32),
    merge_strings: std.ArrayListUnmanaged([]const u8),

    // Fast merge table: (left_id, right_id) → { rank, new_id }
    pair_merges: PairMergeMap,

    // Tokens auto-added from merge rules not present in vocab.
    // Separately allocated; freed in destroy (JSON path only — files path
    // frees all id_to_token entries).
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
    pub fn encodeWordDirect(self: *BpeModel, tok: *ct.Tokenizer, word: []const u8, out_ids: *IdList) !void {
        return encodeWordCore(self, tok, word, out_ids);
    }

    /// Decode token IDs to text
    pub fn decode(self: *BpeModel, tok: *ct.Tokenizer, ids: [*c]const i32, ids_len: usize, out: *[*c]u8, out_len: *usize) c_int {
        return bpe_decode_impl(self, tok, ids, ids_len, out, out_len);
    }

    /// Decode token IDs to text with options (skip special tokens etc.)
    pub fn decodeWithOptions(self: *BpeModel, tok: *ct.Tokenizer, ids: [*c]const i32, ids_len: usize, out: *[*c]u8, out_len: *usize, options: DecodeOptions) c_int {
        return bpe_decode_with_options_impl(self, tok, ids, ids_len, out, out_len, options);
    }

    /// Free the model resources
    pub fn destroy(self: *BpeModel, tok: *ct.Tokenizer) void {
        if (self.json_buffer.len == 0) {
            bpe_destroy_files_impl(self, tok);
        } else {
            bpe_destroy_impl(self, tok);
        }
    }
};

/// Create a BPE model from a JSON buffer, parsing vocab and merges eagerly.
fn create(allocator: std.mem.Allocator, json_buffer: []const u8, json_owned: bool) !*BpeModel {
    var model = try allocator.create(BpeModel);
    errdefer allocator.destroy(model);

    // Determine vocab size by finding max ID in JSON (scan for largest number after ":")
    const vocab_size = try findVocabSize(json_buffer);
    if (vocab_size == 0 or vocab_size > MAX_MODEL_VOCAB_SIZE) return error.InvalidArgument;

    model.* = .{
        .allocator = allocator,
        .json_buffer = json_buffer,
        .json_owned = json_owned,
        .id_to_token = try allocator.alloc(?[]const u8, vocab_size),
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

/// Find vocab size by scanning for max ID in JSON
fn findVocabSize(json: []const u8) !usize {
    // Modern LLMs have varying vocab sizes (32k to 256k+)
    // Scan for largest ID to get accurate size
    var max_id: usize = 0;
    var scan_idx: usize = 0;
    while (scan_idx < json.len) {
        // Look for pattern ": <number>," or ": <number>}"
        if (json[scan_idx] == ':') {
            scan_idx += 1;
            // Skip whitespace
            while (scan_idx < json.len and (json[scan_idx] == ' ' or json[scan_idx] == '\n' or json[scan_idx] == '\t')) : (scan_idx += 1) {}
            // Parse number
            if (scan_idx < json.len and json[scan_idx] >= '0' and json[scan_idx] <= '9') {
                var num: usize = 0;
                while (scan_idx < json.len and json[scan_idx] >= '0' and json[scan_idx] <= '9') {
                    num = std.math.mul(usize, num, 10) catch return error.InvalidArgument;
                    num = std.math.add(usize, num, @as(usize, json[scan_idx] - '0')) catch return error.InvalidArgument;
                    scan_idx += 1;
                }
                if (num > max_id) max_id = num;
            }
        } else {
            scan_idx += 1;
        }
    }
    return std.math.add(usize, max_id, 1) catch error.InvalidArgument;
}

/// Parse vocab and merges from JSON - single pass, no pre-scanning
fn parseVocabAndMerges(model: *BpeModel) !void {
    var t_start: i128 = std.time.nanoTimestamp();

    const json = model.json_buffer;
    const allocator = model.allocator;

    // Find vocab start (quick string search)
    const vocab_start = findSectionStart(json, "\"vocab\"") orelse return error.NoVocab;

    // Find merges start (optional - SentencePiece models don't have merges)
    const merges_start: ?usize = findSectionStart(json, "\"merges\"");

    {
        const now = std.time.nanoTimestamp();
        const duration_ms = @as(f64, @floatFromInt(now - t_start)) / 1_000_000.0;
        log.trace("tokenizer", "Find sections", .{ .duration_ms = duration_ms }, @src());
        t_start = now;
    }

    // Parse vocab - from vocab_start to merges_start (or end of json)
    // First pass: just fill id_to_token array (zero-copy pointers)
    const vocab_end = merges_start orelse json.len;
    const vocab_json = json[vocab_start..vocab_end];
    var scan_idx: usize = 0;
    var count: usize = 0;
    var depth: usize = 0;

    while (scan_idx < vocab_json.len) {
        const ch = vocab_json[scan_idx];

        if (ch == '{') {
            depth += 1;
            scan_idx += 1;
        } else if (ch == '}') {
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
            if (scan_idx >= vocab_json.len) break;
            const key_end = scan_idx;
            scan_idx += 1;

            // Skip to number
            while (scan_idx < vocab_json.len and (vocab_json[scan_idx] < '0' or vocab_json[scan_idx] > '9')) : (scan_idx += 1) {}
            if (scan_idx >= vocab_json.len) break;
            const num_start = scan_idx;
            while (scan_idx < vocab_json.len and vocab_json[scan_idx] >= '0' and vocab_json[scan_idx] <= '9') : (scan_idx += 1) {}
            const id = std.fmt.parseInt(usize, vocab_json[num_start..scan_idx], 10) catch continue;

            // Store token - unescape JSON if needed
            const raw_token = vocab_json[key_start..key_end];
            const token = try json_utils.unescapeJsonString(allocator, raw_token);
            if (id < model.id_to_token.len) {
                model.id_to_token[id] = token;
            }
            count += 1;
        } else {
            scan_idx += 1;
        }
    }

    {
        const now = std.time.nanoTimestamp();
        const duration_ms = @as(f64, @floatFromInt(now - t_start)) / 1_000_000.0;
        log.trace("tokenizer", "Vocab scan", .{ .duration_ms = duration_ms, .entries = count }, @src());
        t_start = now;
    }

    // Build vocab_hash from id_to_token array (needed for encoding)
    try model.vocab_hash.ensureTotalCapacity(allocator, @intCast(count));
    for (model.id_to_token, 0..) |maybe_token, id| {
        if (maybe_token) |token| {
            model.vocab_hash.putAssumeCapacity(token, @intCast(id));
        }
    }

    {
        const now = std.time.nanoTimestamp();
        const duration_ms = @as(f64, @floatFromInt(now - t_start)) / 1_000_000.0;
        log.trace("tokenizer", "Vocab hash", .{ .duration_ms = duration_ms }, @src());
        t_start = now;
    }

    // Build byte fallback table for SentencePiece (looks for <0xNN> tokens)
    initByteFallback(model);

    // Build direct byte lookup table for single-byte vocab tokens
    initByteVocabIds(model);

    // Build direct raw-byte→vocab-ID for byte-level BPE
    initOrigByteVocabIds(model);

    // Parse merges if present (SentencePiece/Unigram models don't have merges)
    if (merges_start) |m_start| {
        // Pre-allocate a single buffer for all merge strings
        // Average merge is ~10 bytes, large models may have 400k+ merges = ~5MB
        const merge_buffer = try allocator.alloc(u8, 8 * 1024 * 1024);
        errdefer allocator.free(merge_buffer);
        var merge_buf_pos: usize = 0;

        try model.merges.ensureTotalCapacity(allocator, 500000);
        const merges_json = json[m_start..];
        var rank: i32 = 0;
        var merge_idx: usize = 0;
        depth = 0;

        while (merge_idx < merges_json.len) {
            const ch = merges_json[merge_idx];

            if (ch == '[') {
                depth += 1;
                if (depth == 2) {
                    // Start of merge pair ["a", "b"] (array format)
                    merge_idx += 1;

                    // Find first string
                    while (merge_idx < merges_json.len and merges_json[merge_idx] != '"') : (merge_idx += 1) {}
                    if (merge_idx >= merges_json.len) break;
                    merge_idx += 1;
                    const a_start = merge_idx;
                    while (merge_idx < merges_json.len and merges_json[merge_idx] != '"') {
                        if (merges_json[merge_idx] == '\\') merge_idx += 2 else merge_idx += 1;
                    }
                    if (merge_idx >= merges_json.len) break;
                    const a_end = merge_idx;
                    merge_idx += 1;

                    // Find second string
                    while (merge_idx < merges_json.len and merges_json[merge_idx] != '"') : (merge_idx += 1) {}
                    if (merge_idx >= merges_json.len) break;
                    merge_idx += 1;
                    const b_start = merge_idx;
                    while (merge_idx < merges_json.len and merges_json[merge_idx] != '"') {
                        if (merges_json[merge_idx] == '\\') merge_idx += 2 else merge_idx += 1;
                    }
                    if (merge_idx >= merges_json.len) break;
                    const b_end = merge_idx;
                    merge_idx += 1;

                    // Create merge key "a b" in pre-allocated buffer
                    // Unescape JSON strings if needed
                    const raw_a = merges_json[a_start..a_end];
                    const raw_b = merges_json[b_start..b_end];
                    const lhs = try json_utils.unescapeJsonString(allocator, raw_a);
                    const rhs = try json_utils.unescapeJsonString(allocator, raw_b);
                    const key_len = lhs.len + 1 + rhs.len;

                    if (merge_buf_pos + key_len <= merge_buffer.len) {
                        const merge_key = merge_buffer[merge_buf_pos .. merge_buf_pos + key_len];
                        @memcpy(merge_key[0..lhs.len], lhs);
                        merge_key[lhs.len] = ' ';
                        @memcpy(merge_key[lhs.len + 1 ..], rhs);

                        model.merges.putAssumeCapacity(merge_key, rank);
                        merge_buf_pos += key_len;
                        rank += 1;
                    }
                } else {
                    merge_idx += 1;
                }
            } else if (ch == ']') {
                depth -= 1;
                if (depth == 0) break; // End of merges array
                merge_idx += 1;
            } else if (ch == '"' and depth == 1) {
                // String format merge "a b"
                merge_idx += 1;
                const str_start = merge_idx;
                while (merge_idx < merges_json.len and merges_json[merge_idx] != '"') {
                    if (merges_json[merge_idx] == '\\') merge_idx += 2 else merge_idx += 1;
                }
                if (merge_idx >= merges_json.len) break;
                const str_end = merge_idx;
                merge_idx += 1;

                // The merge is already in "a b" format, just unescape and store
                const raw_merge = merges_json[str_start..str_end];
                const merge_str = try json_utils.unescapeJsonString(allocator, raw_merge);

                if (merge_buf_pos + merge_str.len <= merge_buffer.len) {
                    const merge_key = merge_buffer[merge_buf_pos .. merge_buf_pos + merge_str.len];
                    @memcpy(merge_key, merge_str);

                    model.merges.putAssumeCapacity(merge_key, rank);
                    merge_buf_pos += merge_str.len;
                    rank += 1;
                }
            } else {
                merge_idx += 1;
            }
        }

        // Store buffer pointer for cleanup
        try model.merge_strings.append(allocator, merge_buffer);

        {
            const now = std.time.nanoTimestamp();
            const duration_ms = @as(f64, @floatFromInt(now - t_start)) / 1_000_000.0;
            log.trace("tokenizer", "Merges parsed", .{ .duration_ms = duration_ms, .entries = model.merges.count() }, @src());
        }
    }

    // Build integer-pair merge table for fast encoding.
    // For each string merge "left right" → rank, resolve vocab IDs and store
    // (left_id, right_id) → { rank, new_id } where new_id = vocab("leftright").
    {
        t_start = std.time.nanoTimestamp();
        const merge_count = model.merges.count();
        try model.pair_merges.ensureTotalCapacity(allocator, @intCast(merge_count));

        // Scratch buffer for building the concatenated merge token
        var concat_buf = std.ArrayListUnmanaged(u8){};
        defer concat_buf.deinit(allocator);

        var merge_iter = model.merges.iterator();
        while (merge_iter.next()) |entry| {
            const key = entry.key_ptr.*; // "left right"
            const rank_val = entry.value_ptr.*;

            // Split on first space
            const space_pos = std.mem.indexOfScalar(u8, key, ' ') orelse continue;
            const left_str = key[0..space_pos];
            const right_str = key[space_pos + 1 ..];

            // Look up vocab IDs
            const left_id = model.vocab_hash.get(left_str) orelse continue;
            const right_id = model.vocab_hash.get(right_str) orelse continue;

            // Build concatenated token and look up its ID.
            // If the merged token isn't in vocab, auto-add it — merges
            // implicitly define vocab entries (matches HuggingFace behavior).
            concat_buf.clearRetainingCapacity();
            try concat_buf.appendSlice(allocator, left_str);
            try concat_buf.appendSlice(allocator, right_str);
            const new_id = model.vocab_hash.get(concat_buf.items) orelse blk: {
                const next_id: i32 = @intCast(model.vocab_size);
                const token_copy = try allocator.dupe(u8, concat_buf.items);
                errdefer allocator.free(token_copy);

                // Grow id_to_token array if needed
                if (model.vocab_size >= model.id_to_token.len) {
                    model.id_to_token = try allocator.realloc(model.id_to_token, model.vocab_size + 1);
                }
                model.id_to_token[model.vocab_size] = token_copy;
                try model.vocab_hash.put(allocator, token_copy, next_id);
                try model.merge_added_tokens.append(allocator, token_copy);
                model.vocab_size += 1;
                break :blk next_id;
            };

            model.pair_merges.putAssumeCapacity(
                .{ .left = left_id, .right = right_id },
                .{ .rank = rank_val, .new_id = new_id },
            );
        }

        const now = std.time.nanoTimestamp();
        const duration_ms = @as(f64, @floatFromInt(now - t_start)) / 1_000_000.0;
        log.trace("tokenizer", "Pair merges built", .{ .duration_ms = duration_ms, .entries = model.pair_merges.count() }, @src());
    }

    // Find unk_id
    const unk_slice = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&model.unk_token)), 0);
    if (model.vocab_hash.get(unk_slice)) |id| {
        model.unk_id = id;
    }

    // Find bos_id - try common BOS token names
    if (model.vocab_hash.get("<bos>")) |id| {
        model.bos_id = id;
    } else if (model.vocab_hash.get("<s>")) |id| {
        model.bos_id = id;
    }

    // Find eos_id - try common EOS token names
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

fn findJsonKeyPos(json_bytes: []const u8, field_bytes: []const u8) ?usize {
    var cursor: usize = 0;
    while (cursor < json_bytes.len) {
        if (json_bytes[cursor] == '"') {
            const key_start = cursor;
            cursor += 1;
            while (cursor < json_bytes.len) {
                if (json_bytes[cursor] == '\\' and cursor + 1 < json_bytes.len) {
                    cursor += 2;
                } else if (json_bytes[cursor] == '"') {
                    cursor += 1;
                    break;
                } else {
                    cursor += 1;
                }
            }
            const key_end = cursor;
            if (key_end - key_start == field_bytes.len and
                std.mem.eql(u8, json_bytes[key_start..key_end], field_bytes))
            {
                return key_start;
            }
        } else {
            cursor += 1;
        }
    }
    return null;
}

fn findJsonFieldValue(json_bytes: []const u8, field_bytes: []const u8) ?[]const u8 {
    const pos = findJsonKeyPos(json_bytes, field_bytes) orelse return null;
    var cursor = pos + field_bytes.len;
    while (cursor < json_bytes.len and
        (json_bytes[cursor] == ' ' or
            json_bytes[cursor] == ':' or
            json_bytes[cursor] == '\t' or
            json_bytes[cursor] == '\n' or
            json_bytes[cursor] == '\r')) : (cursor += 1)
    {}
    if (cursor >= json_bytes.len) return null;

    const value_start = cursor;
    while (cursor < json_bytes.len and
        json_bytes[cursor] != ',' and
        json_bytes[cursor] != '}' and
        json_bytes[cursor] != ']' and
        json_bytes[cursor] != ' ' and
        json_bytes[cursor] != '\n' and
        json_bytes[cursor] != '\r') : (cursor += 1)
    {}
    return json_bytes[value_start..cursor];
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

    if (findJsonFieldValue(search_scope, "\"ignore_merges\"")) |value| {
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

pub const IdList = std.ArrayListUnmanaged(i32);

fn findBestPair(
    tokens: []const []const u8,
    merges: *const std.StringHashMapUnmanaged(i32),
    scratch: *std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,
) !?struct { pos: usize, rank: i32 } {
    var best_rank: i32 = std.math.maxInt(i32);
    var best_pos: ?usize = null;
    if (tokens.len < 2) return null;
    for (tokens[0 .. tokens.len - 1], 0..) |tok, pair_idx| {
        scratch.clearRetainingCapacity();
        try scratch.appendSlice(allocator, tok);
        try scratch.append(allocator, ' ');
        try scratch.appendSlice(allocator, tokens[pair_idx + 1]);
        if (merges.get(scratch.items)) |rank| {
            if (rank < best_rank) {
                best_rank = rank;
                best_pos = pair_idx;
            }
        }
    }
    if (best_pos) |pos| return .{ .pos = pos, .rank = best_rank };
    return null;
}

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

    // --- Linked-list BPE merge with integer pair lookups ---

    // 1. Split word into UTF-8 characters → Symbol array.
    //    Stack-allocated for small words (word.len is an upper bound on char
    //    count since each UTF-8 char is >= 1 byte); heap fallback for words
    //    exceeding MAX_WORD_SYMBOLS bytes.
    var stack_syms: [MAX_WORD_SYMBOLS]Symbol = undefined;
    var heap_syms: ?[]Symbol = null;
    defer if (heap_syms) |hs| allocator.free(hs);

    var n_syms: usize = 0;
    var used_raw_byte_init = false;

    var syms: []Symbol = &stack_syms;

    if (use_raw_byte_init) {
        // Byte-level BPE with raw byte init: each byte maps directly to a
        // vocab token via orig_byte_vocab_ids. No UTF-8 parsing or hash
        // lookups needed. Requires applyByteLevel to be skipped (non-sequence
        // pretokenizer) so words contain raw bytes.
        if (word.len > MAX_WORD_SYMBOLS) {
            heap_syms = try allocator.alloc(Symbol, word.len);
            syms = heap_syms.?;
        }
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
        used_raw_byte_init = true;
    } else {
        // Non-byte-level: split by UTF-8 characters and look up each in vocab.
        if (word.len > MAX_WORD_SYMBOLS) {
            var n_chars: usize = 0;
            var bi: usize = 0;
            while (bi < word.len) {
                const cl = utf8CharLen(word[bi]);
                if (bi + cl > word.len) return error.InvalidUtf8;
                bi += cl;
                n_chars += 1;
            }
            if (n_chars > MAX_WORD_SYMBOLS) {
                heap_syms = try allocator.alloc(Symbol, n_chars);
                syms = heap_syms.?;
            }
        }

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
    }
    // Fast path: single-symbol words skip merge loop and collect entirely.
    if (n_syms == 1) {
        const sym = &syms[0];
        if (sym.id >= 0) {
            try out_ids.append(allocator, sym.id);
        } else if (!is_byte_level) {
            const token_bytes = word[sym.start .. sym.start + sym.len];
            for (token_bytes) |byte_val| {
                const fallback_id = model.byte_fallback_ids[byte_val];
                try out_ids.append(allocator, if (fallback_id >= 0) fallback_id else model.unk_id);
            }
        } else {
            try out_ids.append(allocator, model.unk_id);
        }
        return;
    }

    // Fast path: two-symbol words need at most one merge lookup.
    // Skips pair cache setup (pos_to_cache zeroing, Phase 1/2 loops).
    if (n_syms == 2) {
        const s0 = &syms[0];
        const s1 = &syms[1];
        if (s0.id >= 0 and s1.id >= 0) {
            if (model.pair_merges.get(.{ .left = s0.id, .right = s1.id })) |info| {
                try out_ids.append(allocator, info.new_id);
            } else {
                try out_ids.ensureUnusedCapacity(allocator, 2);
                out_ids.appendAssumeCapacity(s0.id);
                out_ids.appendAssumeCapacity(s1.id);
            }
        } else {
            for ([2]*const Symbol{ s0, s1 }) |sym| {
                if (sym.id >= 0) {
                    try out_ids.append(allocator, sym.id);
                } else if (!is_byte_level) {
                    const token_bytes = word[sym.start .. sym.start + sym.len];
                    for (token_bytes) |byte_val| {
                        const fallback_id = model.byte_fallback_ids[byte_val];
                        try out_ids.append(allocator, if (fallback_id >= 0) fallback_id else model.unk_id);
                    }
                } else {
                    try out_ids.append(allocator, model.unk_id);
                }
            }
        }
        return;
    }

    // 2. Merge loop.
    //    For short words (3-5 symbols), use simple re-scan: find best pair
    //    by scanning all pairs each iteration. Avoids pair cache setup
    //    (1KB pos_to_cache fill, Phase 1/2 bookkeeping) for the common case.
    if (n_syms >= 3 and n_syms <= 5 and model.pair_merges.count() > 0) {
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

            const left = &syms[@intCast(best_pos)];
            const right_idx = left.next;
            const right = &syms[@intCast(right_idx)];
            left.id = best_new_id;
            left.len += right.len;
            left.next = right.next;
            if (right.next >= 0) syms[@intCast(right.next)].prev = best_pos;
            right.id = -2;
        }
    } else if (n_syms >= 6 and model.pair_merges.count() > 0) {
        // Cached-pair approach for longer words.
        // Phase 1 builds a cache of all valid merge pairs (one full scan).
        // Phase 2 repeatedly picks the lowest-rank pair from the cache,
        // applies the merge, and updates only the 1-2 affected neighbors.
        // This reduces pair_merges hash lookups from O(n_pairs × n_passes)
        // to O(n_pairs + 2 × n_merges).
        const CachedPair = struct { pos: i32, rank: i32, new_id: i32 };
        var pair_cache_stack: [MAX_WORD_SYMBOLS]CachedPair = undefined;
        var pair_cache_heap: ?[]CachedPair = null;
        defer if (pair_cache_heap) |cache| allocator.free(cache);
        var pair_cache: []CachedPair = &pair_cache_stack;
        var n_cached: usize = 0;
        // Position-to-cache-index tracking for O(1) stale entry removal.
        // Use the actual symbol count here; long words already spill `syms`
        // to the heap and the cache must follow suit to avoid overflow.
        var pos_to_cache_stack: [MAX_WORD_SYMBOLS]i32 = undefined;
        var pos_to_cache_heap: ?[]i32 = null;
        defer if (pos_to_cache_heap) |cache| allocator.free(cache);
        var pos_to_cache: []i32 = &pos_to_cache_stack;
        if (n_syms > MAX_WORD_SYMBOLS) {
            pair_cache_heap = try allocator.alloc(CachedPair, n_syms);
            pair_cache = pair_cache_heap.?;
            pos_to_cache_heap = try allocator.alloc(i32, n_syms);
            pos_to_cache = pos_to_cache_heap.?;
        }
        @memset(pos_to_cache[0..n_syms], -1);

        // Phase 1: Build pair cache (single scan, one hash lookup per pair)
        {
            var si: i32 = 0;
            while (si >= 0) {
                const sym = &syms[@intCast(si)];
                if (sym.next >= 0 and sym.id >= 0) {
                    const right_sym = &syms[@intCast(sym.next)];
                    if (right_sym.id >= 0) {
                        if (model.pair_merges.get(.{ .left = sym.id, .right = right_sym.id })) |info| {
                            std.debug.assert(n_cached < pair_cache.len);
                            pos_to_cache[@intCast(si)] = @intCast(n_cached);
                            pair_cache[n_cached] = .{ .pos = si, .rank = info.rank, .new_id = info.new_id };
                            n_cached += 1;
                        }
                    }
                }
                si = sym.next;
            }
        }
        // Phase 2: Process merges from cache
        while (n_cached > 0) {
            // Find entry with minimum rank; leftmost position breaks ties
            // (matches original left-to-right scan behavior)
            var best_idx: usize = 0;
            for (1..n_cached) |i| {
                if (pair_cache[i].rank < pair_cache[best_idx].rank or
                    (pair_cache[i].rank == pair_cache[best_idx].rank and
                        pair_cache[i].pos < pair_cache[best_idx].pos))
                {
                    best_idx = i;
                }
            }
            const best = pair_cache[best_idx];

            // Swap-remove best entry from cache; update index for moved entry
            n_cached -= 1;
            if (best_idx < n_cached) {
                pair_cache[best_idx] = pair_cache[n_cached];
                pos_to_cache[@intCast(pair_cache[best_idx].pos)] = @intCast(best_idx);
            }
            pos_to_cache[@intCast(best.pos)] = -1;

            // Validate: symbols may have been invalidated by an earlier merge
            const left = &syms[@intCast(best.pos)];
            if (left.id < 0 or left.next < 0) continue;
            const right_idx = left.next;
            const right = &syms[@intCast(right_idx)];
            if (right.id < 0) continue;

            // Apply merge: extend left symbol, unlink right
            left.id = best.new_id;
            left.len += right.len;
            left.next = right.next;
            if (right.next >= 0) {
                syms[@intCast(right.next)].prev = best.pos;
            }
            right.id = -2;

            // Remove stale cache entries via position index (O(1) per entry).
            // After merging at pos P (right R removed):
            //   - Entry at pos==R is invalid (R is dead)
            //   - Entry at pos==P.prev is stale (P's id changed)
            for ([_]i32{ right_idx, left.prev }) |stale_pos| {
                if (stale_pos >= 0) {
                    const ci = pos_to_cache[@intCast(stale_pos)];
                    if (ci >= 0 and @as(usize, @intCast(ci)) < n_cached) {
                        const cui: usize = @intCast(ci);
                        n_cached -= 1;
                        if (cui < n_cached) {
                            pair_cache[cui] = pair_cache[n_cached];
                            pos_to_cache[@intCast(pair_cache[cui].pos)] = @intCast(cui);
                        }
                        pos_to_cache[@intCast(stale_pos)] = -1;
                    }
                }
            }

            // Add new pair: (left_neighbor, merged)
            if (left.prev >= 0) {
                const ln = &syms[@intCast(left.prev)];
                if (ln.id >= 0) {
                    if (model.pair_merges.get(.{ .left = ln.id, .right = left.id })) |info| {
                        std.debug.assert(n_cached < pair_cache.len);
                        pos_to_cache[@intCast(left.prev)] = @intCast(n_cached);
                        pair_cache[n_cached] = .{ .pos = left.prev, .rank = info.rank, .new_id = info.new_id };
                        n_cached += 1;
                    }
                }
            }
            // Add new pair: (merged, right_neighbor)
            if (left.next >= 0) {
                const rn = &syms[@intCast(left.next)];
                if (rn.id >= 0) {
                    if (model.pair_merges.get(.{ .left = left.id, .right = rn.id })) |info| {
                        std.debug.assert(n_cached < pair_cache.len);
                        pos_to_cache[@intCast(best.pos)] = @intCast(n_cached);
                        pair_cache[n_cached] = .{ .pos = best.pos, .rank = info.rank, .new_id = info.new_id };
                        n_cached += 1;
                    }
                }
            }
        }
    }

    // 3. Check if all symbols resolved to valid IDs. If any has id == -1
    //    (unknown) and the whole word is in vocab, fall back to direct lookup.
    {
        var all_known = true;
        var si: i32 = 0;
        while (si >= 0) {
            if (syms[@intCast(si)].id < 0) {
                all_known = false;
                break;
            }
            si = syms[@intCast(si)].next;
        }
        if (!all_known) {
            if (lookupWholeWordId(model, word, used_raw_byte_init)) |id| {
                try out_ids.append(allocator, id);
                return;
            }
        }
    }

    // 4. Collect IDs from linked list directly into output.
    //    Pre-count surviving symbols to reserve capacity, then append without
    //    per-token capacity checks.
    {
        var n_output: usize = 0;
        var ci: i32 = 0;
        while (ci >= 0) {
            const csym = &syms[@intCast(ci)];
            if (csym.id >= 0) {
                n_output += 1;
            } else if (!is_byte_level) {
                n_output += csym.len; // byte fallback: one ID per byte
            } else {
                n_output += 1;
            }
            ci = csym.next;
        }
        try out_ids.ensureUnusedCapacity(allocator, n_output);
    }
    var si: i32 = 0;
    while (si >= 0) {
        const sym = &syms[@intCast(si)];

        if (sym.id >= 0) {
            out_ids.appendAssumeCapacity(sym.id);
        } else if (!is_byte_level) {
            // SentencePiece: use byte fallback for unknown tokens
            const token_bytes = word[sym.start .. sym.start + sym.len];
            for (token_bytes) |byte_val| {
                const fallback_id = model.byte_fallback_ids[byte_val];
                out_ids.appendAssumeCapacity(if (fallback_id >= 0) fallback_id else model.unk_id);
            }
        } else {
            out_ids.appendAssumeCapacity(model.unk_id);
        }
        si = sym.next;
    }
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

    log.trace("tokenizer", "bpe_encode", .{ .text_len = text.len, .vocab_size = model.vocab_hash.count() }, @src());
    return bpe_encode_via_encoding(model, tok, text, enc);
}

/// Encode with explicit length (supports embedded null bytes) - implementation
fn bpe_encode_slice_impl(model: *BpeModel, tok: *ct.Tokenizer, text: []const u8, enc: *ct.TokenizerEncoding) c_int {
    log.trace("tokenizer", "bpe_encode_slice", .{ .text_len = text.len, .vocab_size = model.vocab_hash.count() }, @src());
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

/// Decode options for controlling output.
pub const DecodeOptions = struct {
    /// If true, skip special tokens (BOS/EOS/etc.) in the output.
    skip_special_tokens: bool = false,
};

fn bpe_decode_with_options_impl(model: *BpeModel, tok: *ct.Tokenizer, ids: [*c]const i32, ids_len: usize, out: *[*c]u8, out_len: *usize, options: DecodeOptions) c_int {
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

/// C API vtable adapter: extracts model from tokenizer and calls impl
fn bpe_decode_with_options(tok: *ct.Tokenizer, ids: [*c]const i32, ids_len: usize, out: *[*c]u8, out_len: *usize, options: DecodeOptions) c_int {
    if (tok.model == null) return -1;
    const model = @as(*BpeModel, @ptrCast(@alignCast(tok.model.?)));
    return bpe_decode_with_options_impl(model, tok, ids, ids_len, out, out_len, options);
}

fn bpe_decode_impl(model: *BpeModel, tok: *ct.Tokenizer, ids: [*c]const i32, ids_len: usize, out: *[*c]u8, out_len: *usize) c_int {
    return bpe_decode_with_options_impl(model, tok, ids, ids_len, out, out_len, .{});
}

fn bpe_decode(tok: *ct.Tokenizer, ids: [*c]const i32, ids_len: usize, out: *[*c]u8, out_len: *usize) c_int {
    return bpe_decode_with_options(tok, ids, ids_len, out, out_len, .{});
}

fn bpe_destroy_impl(model: *BpeModel, tok: *ct.Tokenizer) void {
    tok.model = null;

    // Free merge-implied tokens (separately allocated, not in json_buffer)
    for (model.merge_added_tokens.items) |s| model.allocator.free(s);
    model.merge_added_tokens.deinit(model.allocator);

    // Free merge strings (we allocated these)
    for (model.merge_strings.items) |s| model.allocator.free(s);
    model.merge_strings.deinit(model.allocator);
    model.merges.deinit(model.allocator);
    model.pair_merges.deinit(model.allocator);
    model.vocab_hash.deinit(model.allocator);
    model.allocator.free(model.id_to_token);

    // Free byte_to_unicode
    for (model.byte_to_unicode) |s| {
        if (s.len > 0) model.allocator.free(s);
    }

    if (model.json_owned and model.json_buffer.len > 0) {
        model.allocator.free(@constCast(model.json_buffer));
    }
    model.allocator.destroy(model);
}

/// C API vtable adapter: extracts model from tokenizer and calls impl
fn bpe_destroy(tok: *ct.Tokenizer) void {
    if (tok.model == null) return;
    const model = @as(*BpeModel, @ptrCast(@alignCast(tok.model.?)));
    bpe_destroy_impl(model, tok);
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

    var model = create(allocator, json_buffer, json_owned) catch |err| {
        log.warn("tokenizer", "BPE model creation failed", .{
            .reason = @errorName(err),
            .json_bytes = json_buffer.len,
            .json_owned = @intFromBool(json_owned),
        });
        return err;
    };
    errdefer {
        if (model.json_owned and model.json_buffer.len > 0) {
            allocator.free(@constCast(model.json_buffer));
        }
        allocator.destroy(model);
    }

    tok.model = model;
    model.owner = tok;

    // Attach pretokenizer

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

/// Destroy implementation for file-loaded models (same as lazy but also frees vocab strings)
fn bpe_destroy_files_impl(model: *BpeModel, tok: *ct.Tokenizer) void {
    tok.model = null;

    // Free vocab strings (we allocated these when parsing vocab.json)
    // This also frees any merge-implied tokens since they're in id_to_token.
    for (model.id_to_token) |maybe_token| {
        if (maybe_token) |token| {
            model.allocator.free(token);
        }
    }
    // Just deinit the list; entries already freed above via id_to_token.
    model.merge_added_tokens.deinit(model.allocator);

    // Rest is same as lazy destroy
    for (model.merge_strings.items) |s| model.allocator.free(s);
    model.merge_strings.deinit(model.allocator);
    model.merges.deinit(model.allocator);
    model.pair_merges.deinit(model.allocator);
    model.vocab_hash.deinit(model.allocator);
    model.allocator.free(model.id_to_token);

    for (model.byte_to_unicode) |s| {
        if (s.len > 0) model.allocator.free(s);
    }

    model.allocator.destroy(model);
}

/// C API vtable adapter: extracts model from tokenizer and calls impl
fn bpe_destroy_files(tok: *ct.Tokenizer) void {
    if (tok.model == null) return;
    const model = @as(*BpeModel, @ptrCast(@alignCast(tok.model.?)));
    bpe_destroy_files_impl(model, tok);
}

// =============================================================================
// C-API Spec-based loading
// =============================================================================

/// Create BPE tokenizer from a C-API specification struct
pub fn tokenizer_bpe_create_from_spec(spec: ?*const ct.BpeModelSpec) ?*ct.Tokenizer {
    if (spec == null) return null;
    const bpe_spec = spec.?;
    if (bpe_spec.vocab == null or bpe_spec.merges == null or bpe_spec.vocab_len == 0 or bpe_spec.merges_len == 0) return null;

    const allocator = std.heap.c_allocator;

    // Create tokenizer struct
    var tok = allocator.create(ct.Tokenizer) catch return null;
    tok.* = std.mem.zeroes(ct.Tokenizer);
    tok.type = ct.ModelType.bpe;
    tok.normalizer.lowercase = 0;
    tok.normalizer.nfd = 0;
    tok.postproc.cls_id = -1;
    tok.postproc.sep_id = -1;
    tok.postproc.add_special = 0;

    // Create model from spec
    var model = createFromSpec(allocator, bpe_spec) catch {
        allocator.destroy(tok);
        return null;
    };

    tok.model = model;
    model.owner = tok;

    // Attach pretokenizer

    if (tok_fns.tokenizer_pretokenizer_set(&tok.pretokenizer, DEFAULT_PATTERN.ptr) != 0) {
        tok_fns.tokenizer_set_error(tok, "Failed to compile BPE regex");
        bpe_destroy_files(tok);
        allocator.destroy(tok);
        return null;
    }

    return tok;
}

/// Create model from C-API spec
fn createFromSpec(allocator: std.mem.Allocator, spec: *const ct.BpeModelSpec) !*BpeModel {
    var model = try allocator.create(BpeModel);
    errdefer allocator.destroy(model);

    // Find max ID
    var max_id: usize = 0;
    const vocab_ptr: [*]const ct.TokenIdPair = @ptrCast(spec.vocab.?);
    const vocab = vocab_ptr[0..spec.vocab_len];
    for (vocab) |entry| {
        if (entry.id >= 0) {
            const next = @as(usize, @intCast(entry.id)) + 1;
            if (next > max_id) max_id = next;
        }
    }
    if (max_id == 0) return error.IncompleteSpec;

    model.* = .{
        .allocator = allocator,
        .json_buffer = "",
        .json_owned = false,
        .id_to_token = try allocator.alloc(?[]const u8, max_id),
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
        .vocab_size = max_id,
        .owner = null,
    };

    @memset(model.id_to_token, null);
    for (&model.byte_to_unicode) |*slot| slot.* = "";
    @memset(model.unk_token[0..], 0);
    @memcpy(model.unk_token[0..DEFAULT_UNK.len], DEFAULT_UNK);

    // Populate vocab
    try model.vocab_hash.ensureTotalCapacity(allocator, @intCast(spec.vocab_len));
    for (vocab) |entry| {
        if (entry.token == null or entry.id < 0) continue;
        const token_ptr: [*:0]const u8 = @ptrCast(entry.token.?);
        const token_slice = std.mem.sliceTo(token_ptr, 0);
        const token_dup = try allocator.dupe(u8, token_slice);
        const idx: usize = @intCast(entry.id);
        if (idx < model.id_to_token.len) {
            model.id_to_token[idx] = token_dup;
            model.vocab_hash.putAssumeCapacity(token_dup, entry.id);
        } else {
            allocator.free(token_dup);
        }
    }

    // Build byte fallback table for SentencePiece (looks for <0xNN> tokens)
    initByteFallback(model);

    // Build direct byte lookup table for single-byte vocab tokens
    initByteVocabIds(model);

    // Build direct raw-byte→vocab-ID for byte-level BPE
    initOrigByteVocabIds(model);

    // Build merges
    const merge_buffer = try allocator.alloc(u8, 8 * 1024 * 1024);
    var merge_buf_pos: usize = 0;
    try model.merge_strings.append(allocator, merge_buffer);
    try model.merges.ensureTotalCapacity(allocator, @intCast(spec.merges_len));

    const merges_ptr: [*]const ct.BpeMergePair = @ptrCast(spec.merges.?);
    const merges = merges_ptr[0..spec.merges_len];
    for (merges, 0..) |pair, rank| {
        if (pair.a == null or pair.b == null) continue;
        const left_ptr: [*:0]const u8 = @ptrCast(pair.a.?);
        const right_ptr: [*:0]const u8 = @ptrCast(pair.b.?);
        const left = std.mem.sliceTo(left_ptr, 0);
        const right = std.mem.sliceTo(right_ptr, 0);
        const key_len = left.len + 1 + right.len;

        if (merge_buf_pos + key_len <= merge_buffer.len) {
            const merge_key = merge_buffer[merge_buf_pos .. merge_buf_pos + key_len];
            @memcpy(merge_key[0..left.len], left);
            merge_key[left.len] = ' ';
            @memcpy(merge_key[left.len + 1 ..], right);
            model.merges.putAssumeCapacity(merge_key, @intCast(rank));
            merge_buf_pos += key_len;
        }
    }

    // Set unk token if provided
    if (spec.unk_token) |unk| {
        const unk_ptr: [*:0]const u8 = @ptrCast(unk);
        const unk_slice = std.mem.sliceTo(unk_ptr, 0);
        const copy_len = @min(unk_slice.len, model.unk_token.len - 1);
        @memcpy(model.unk_token[0..copy_len], unk_slice[0..copy_len]);
        model.unk_token[copy_len] = 0;
    }

    // Build byte map
    try initByteMap(model);

    // Finalize unk
    const unk_slice = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&model.unk_token)), 0);
    if (model.vocab_hash.get(unk_slice)) |id| {
        model.unk_id = id;
    }

    return model;
}

// =============================================================================
// Tests
// =============================================================================

// Integration tests: Most functions in this file require a fully initialized
// BPE tokenizer with a loaded vocabulary and merge rules.
// They are tested via integration tests in tests/tokenizer/ which test:
// - createBpeTokenizer: BPE tokenizer creation from JSON
// - tokenizer_bpe_create_from_spec: Tokenizer creation from C API spec
// - bpeEncode/bpeEncodeSlice: Text to token ID encoding
// - bpeDecode/bpeDecodeWithOptions: Token ID to text decoding
// - bpeDestroy: Resource cleanup and memory management

test "tokenizer_bpe_create_from_spec: findVocabSize with simple vocab" {
    const json =
        \\{"vocab": {"a": 0, "b": 1, "c": 2}}
    ;
    const size = try findVocabSize(json);
    try std.testing.expectEqual(@as(usize, 3), size);
}

test "tokenizer_bpe_create_from_spec: findVocabSize with gaps in IDs" {
    const json =
        \\{"vocab": {"a": 0, "b": 5, "c": 10}}
    ;
    const size = try findVocabSize(json);
    try std.testing.expectEqual(@as(usize, 11), size);
}

test "tokenizer_bpe_create_from_spec: findVocabSize with large IDs" {
    const json =
        \\{"vocab": {"token1": 100, "token2": 50000, "token3": 25}}
    ;
    const size = try findVocabSize(json);
    try std.testing.expectEqual(@as(usize, 50001), size);
}

test "tokenizer_bpe_create_from_spec: findVocabSize with whitespace" {
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

test "tokenizer_bpe_create_from_spec: findVocabSize reports oversize sparse ids" {
    const json =
        \\{"vocab": {"a": 2147483647}}
    ;
    const size = try findVocabSize(json);
    try std.testing.expect(size > MAX_MODEL_VOCAB_SIZE);
}

test "tokenizer_bpe_create_from_spec: findSectionStart finds vocab section" {
    const json =
        \\{"model": "bpe", "vocab": {"a": 0}, "merges": []}
    ;
    const start = findSectionStart(json, "\"vocab\"");
    try std.testing.expect(start != null);
    try std.testing.expect(start.? > 0);
}

test "tokenizer_bpe_create_from_spec: findSectionStart finds merges section" {
    const json =
        \\{"model": "bpe", "vocab": {"a": 0}, "merges": []}
    ;
    const start = findSectionStart(json, "\"merges\"");
    try std.testing.expect(start != null);
    try std.testing.expect(start.? > 0);
}

test "tokenizer_bpe_create_from_spec: findSectionStart returns null for missing section" {
    const json =
        \\{"model": "bpe", "vocab": {"a": 0}}
    ;
    const start = findSectionStart(json, "\"missing\"");
    try std.testing.expect(start == null);
}

test "bpeDecodeWithOptions: DecodeOptions default values" {
    const opts = DecodeOptions{};
    try std.testing.expect(opts.skip_special_tokens == false);
}

test "bpeDecodeWithOptions: DecodeOptions with skip_special_tokens enabled" {
    const opts = DecodeOptions{ .skip_special_tokens = true };
    try std.testing.expect(opts.skip_special_tokens == true);
}

// =============================================================================
// Unit tests for critical internal functions
// =============================================================================

test "tokenizer_bpe_create_from_spec: parseVocabAndMerges with minimal valid JSON" {
    const json =
        \\{"vocab": {"a": 0, "b": 1, "c": 2}, "merges": [["a", "b"]]}
    ;
    const allocator = std.testing.allocator;

    var model = try create(allocator, json, false);
    defer {
        // Note: vocab tokens are either borrowed from json_buffer (when no escapes)
        // or allocated by unescapeJsonString (when escapes present).
        // Since json is a string literal and simple tokens have no escapes,
        // the tokens are borrowed and should NOT be freed.
        // Only the byte_to_unicode mapping and merge_strings need cleanup.
        for (model.merge_added_tokens.items) |s| allocator.free(s);
        model.merge_added_tokens.deinit(allocator);
        model.vocab_hash.deinit(allocator);
        model.merges.deinit(allocator);
        model.pair_merges.deinit(allocator);
        for (model.merge_strings.items) |s| allocator.free(s);
        model.merge_strings.deinit(allocator);
        allocator.free(model.id_to_token);
        for (model.byte_to_unicode) |s| {
            if (s.len > 0) allocator.free(s);
        }
        allocator.destroy(model);
    }

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

test "tokenizer_bpe_create_from_spec: parseVocabAndMerges with string format merges" {
    const json =
        \\{"vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]}
    ;
    const allocator = std.testing.allocator;

    var model = try create(allocator, json, false);
    defer {
        for (model.merge_added_tokens.items) |s| allocator.free(s);
        model.merge_added_tokens.deinit(allocator);
        model.vocab_hash.deinit(allocator);
        model.merges.deinit(allocator);
        model.pair_merges.deinit(allocator);
        for (model.merge_strings.items) |s| allocator.free(s);
        model.merge_strings.deinit(allocator);
        allocator.free(model.id_to_token);
        for (model.byte_to_unicode) |s| {
            if (s.len > 0) allocator.free(s);
        }
        allocator.destroy(model);
    }

    // Verify vocab
    try std.testing.expect(model.vocab_hash.count() == 3);

    // Verify string format merges were parsed
    try std.testing.expect(model.merges.count() > 0);
    const rank = model.merges.get("a b");
    try std.testing.expect(rank != null);
    try std.testing.expect(rank.? == 0);
}

test "tokenizer_bpe_create_from_spec: parseVocabAndMerges without merges (SentencePiece style)" {
    const json =
        \\{"vocab": {"<unk>": 0, "a": 1, "b": 2}}
    ;
    const allocator = std.testing.allocator;

    var model = try create(allocator, json, false);
    defer {
        model.merge_added_tokens.deinit(allocator);
        model.vocab_hash.deinit(allocator);
        model.merges.deinit(allocator);
        model.pair_merges.deinit(allocator);
        for (model.merge_strings.items) |s| allocator.free(s);
        model.merge_strings.deinit(allocator);
        allocator.free(model.id_to_token);
        for (model.byte_to_unicode) |s| {
            if (s.len > 0) allocator.free(s);
        }
        allocator.destroy(model);
    }

    // Verify vocab was parsed
    try std.testing.expect(model.vocab_hash.count() == 3);

    // Verify no merges (SentencePiece models don't have merges)
    try std.testing.expect(model.merges.count() == 0);

    // Verify unk_id was set
    try std.testing.expect(model.unk_id == 0);
}

test "tokenizer_bpe_create_from_spec: parseVocabAndMerges with escaped characters" {
    const json =
        \\{"vocab": {"a": 0, "\"": 1, "\\": 2}, "merges": []}
    ;
    const allocator = std.testing.allocator;

    var model = try create(allocator, json, false);
    defer {
        // Escaped characters ("\"", "\\") are allocated by unescapeJsonString
        // and need to be freed. Simple "a" is borrowed.
        for (model.id_to_token) |maybe_token| {
            if (maybe_token) |token| {
                // Check if this is an allocated string (escaped chars)
                if (token.len > 0 and (token[0] == '"' or token[0] == '\\')) {
                    allocator.free(token);
                }
            }
        }
        model.merge_added_tokens.deinit(allocator);
        model.vocab_hash.deinit(allocator);
        model.merges.deinit(allocator);
        model.pair_merges.deinit(allocator);
        for (model.merge_strings.items) |s| allocator.free(s);
        model.merge_strings.deinit(allocator);
        allocator.free(model.id_to_token);
        for (model.byte_to_unicode) |s| {
            if (s.len > 0) allocator.free(s);
        }
        allocator.destroy(model);
    }

    // Verify escaped characters were parsed correctly
    try std.testing.expect(model.vocab_hash.count() == 3);
    try std.testing.expect(model.vocab_hash.get("a") != null);
    try std.testing.expect(model.vocab_hash.get("\"") != null);
    try std.testing.expect(model.vocab_hash.get("\\") != null);
}

test "tokenizer_bpe_create_from_spec: parseVocabAndMerges finds special tokens" {
    const json =
        \\{"vocab": {"<s>": 0, "</s>": 1, "a": 2}, "merges": []}
    ;
    const allocator = std.testing.allocator;

    var model = try create(allocator, json, false);
    defer {
        model.merge_added_tokens.deinit(allocator);
        model.vocab_hash.deinit(allocator);
        model.merges.deinit(allocator);
        model.pair_merges.deinit(allocator);
        for (model.merge_strings.items) |s| allocator.free(s);
        model.merge_strings.deinit(allocator);
        allocator.free(model.id_to_token);
        for (model.byte_to_unicode) |s| {
            if (s.len > 0) allocator.free(s);
        }
        allocator.destroy(model);
    }

    // Verify bos_id and eos_id were set
    try std.testing.expect(model.bos_id == 0);
    try std.testing.expect(model.eos_id == 1);
}

test "bpeEncode: findBestPair with single merge" {
    const allocator = std.testing.allocator;
    var merges = std.StringHashMapUnmanaged(i32){};
    defer merges.deinit(allocator);

    try merges.put(allocator, "a b", 0);

    var scratch = std.ArrayListUnmanaged(u8){};
    defer scratch.deinit(allocator);

    const tokens = [_][]const u8{ "a", "b", "c" };

    const result = try findBestPair(&tokens, &merges, &scratch, allocator);
    try std.testing.expect(result != null);
    try std.testing.expect(result.?.pos == 0);
    try std.testing.expect(result.?.rank == 0);
}

test "bpeEncode: findBestPair with multiple merges chooses lowest rank" {
    const allocator = std.testing.allocator;
    var merges = std.StringHashMapUnmanaged(i32){};
    defer merges.deinit(allocator);

    try merges.put(allocator, "a b", 5);
    try merges.put(allocator, "b c", 2);
    try merges.put(allocator, "c d", 10);

    var scratch = std.ArrayListUnmanaged(u8){};
    defer scratch.deinit(allocator);

    const tokens = [_][]const u8{ "a", "b", "c", "d" };

    const result = try findBestPair(&tokens, &merges, &scratch, allocator);
    try std.testing.expect(result != null);
    try std.testing.expect(result.?.pos == 1); // "b c" has lowest rank (2)
    try std.testing.expect(result.?.rank == 2);
}

test "bpeEncode: findBestPair returns null when no merges match" {
    const allocator = std.testing.allocator;
    var merges = std.StringHashMapUnmanaged(i32){};
    defer merges.deinit(allocator);

    try merges.put(allocator, "x y", 0);

    var scratch = std.ArrayListUnmanaged(u8){};
    defer scratch.deinit(allocator);

    const tokens = [_][]const u8{ "a", "b", "c" };

    const result = try findBestPair(&tokens, &merges, &scratch, allocator);
    try std.testing.expect(result == null);
}

test "bpeEncode: findBestPair returns null for single token" {
    const allocator = std.testing.allocator;
    var merges = std.StringHashMapUnmanaged(i32){};
    defer merges.deinit(allocator);

    try merges.put(allocator, "a b", 0);

    var scratch = std.ArrayListUnmanaged(u8){};
    defer scratch.deinit(allocator);

    const tokens = [_][]const u8{"a"};

    const result = try findBestPair(&tokens, &merges, &scratch, allocator);
    try std.testing.expect(result == null);
}

test "bpeEncode: findBestPair returns null for empty tokens" {
    const allocator = std.testing.allocator;
    var merges = std.StringHashMapUnmanaged(i32){};
    defer merges.deinit(allocator);

    var scratch = std.ArrayListUnmanaged(u8){};
    defer scratch.deinit(allocator);

    const tokens = [_][]const u8{};

    const result = try findBestPair(&tokens, &merges, &scratch, allocator);
    try std.testing.expect(result == null);
}

test "tokenizer_bpe_create_from_spec: initByteMap creates valid mappings" {
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

test "tokenizer_bpe_create_from_spec: initByteMap handles control characters with offset" {
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
